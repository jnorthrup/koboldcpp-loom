"""loom: a Hermes context engine that keeps a /goal nesting stack at the front of every request.

The model works like nested for loops. Entering a loop pushes a frame (its stable instructions);
leaving it pops the frame. The frames on the stack, outermost first, are rendered onto the end
of the system prompt after the /goal text. Each frame is an independent obstack frame: it holds
only its own text, nothing refers across frames, and a pop simply discards the top.

That ordering is the nested-loop shape: outer frames change least often and sit furthest
forward, the innermost frame changes most often and sits last, and the conversation (the loop
body) follows. The next sibling iteration is a pop plus a push, so only the tail after the
common frames moves. A prefix cache on the server (koboldcpp --loomcache, or any provider
prompt cache) keeps everything before the change; nothing about frames or boundaries is sent.

Compaction is inherited unchanged from the built-in compressor; it only rewrites conversation
turns, never the frames.

Enable with ``context.engine: loom`` in config.yaml.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from agent.context_compressor import ContextCompressor
from agent.context_engine import ContextEngine  # noqa: F401  (discovery scans the file head for this name)

logger = logging.getLogger(__name__)

TOOL_NAME = "loom"
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MAX_TEXT = 4000
_MAX_DEPTH = 64
FRAMES_OPEN = "[loom: nested work frames, outermost first; the last frame is the current loop]"
FRAMES_CLOSE = "[/loom]"

_ACTIVE: Optional["LoomEngine"] = None  # engine of the most recently started session, for /loom


def _clean_text(text: Any) -> str:
    return str(text or "").replace("\r\n", "\n").strip()


class LoomStack:
    """Obstack of prompt frames: push appends, pop discards the top. Nothing else."""

    def __init__(self) -> None:
        self.frames: List[Dict[str, str]] = []  # [{"label", "text"}], outermost first

    def to_dict(self) -> Dict[str, Any]:
        return {"frames": list(self.frames)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoomStack":
        stack = cls()
        for frame in (data.get("frames") or [])[:_MAX_DEPTH]:
            text = _clean_text(frame.get("text")) if isinstance(frame, dict) else ""
            label = str(frame.get("label") or "") if isinstance(frame, dict) else ""
            if text:
                stack.frames.append({"label": label if _LABEL_RE.match(label) else "", "text": text})
        return stack

    def push(self, text: Any, label: Any = "") -> None:
        text = _clean_text(text)
        label = str(label or "").strip()
        if not text:
            raise ValueError("push needs text")
        if len(text) > _MAX_TEXT:
            raise ValueError(f"frame is {len(text)} chars; limit {_MAX_TEXT}. Push a nested frame instead.")
        if label and not _LABEL_RE.match(label):
            raise ValueError(f"bad label {label!r}: use 1-64 chars of letters, digits, '_', '.', '-'")
        if len(self.frames) >= _MAX_DEPTH:
            raise ValueError(f"nesting is {_MAX_DEPTH} deep; pop before pushing")
        self.frames.append({"label": label, "text": text})

    def pop(self, count: int = 1) -> int:
        if count < 1:
            raise ValueError("pop count must be at least 1")
        if count > len(self.frames):
            raise ValueError(f"only {len(self.frames)} frame(s) to pop")
        del self.frames[len(self.frames) - count:]
        return count

    def _name(self, depth: int) -> str:
        label = self.frames[depth]["label"]
        return f"{depth + 1}" + (f" {label}" if label else "")

    def render(self, goal_block: str) -> str:
        parts = [FRAMES_OPEN]
        if goal_block:
            parts.append("== goal ==\n" + goal_block)
        for depth, frame in enumerate(self.frames):
            parts.append(f"== {self._name(depth)} ==\n{frame['text']}")
        parts.append(FRAMES_CLOSE)
        return "\n".join(parts)

    def outline(self) -> str:
        if not self.frames:
            return "(no frames)"
        lines = []
        for depth, frame in enumerate(self.frames):
            first = frame["text"].split("\n", 1)[0]
            if len(first) > 70:
                first = first[:67] + "..."
            lines.append(f"{'  ' * depth}{self._name(depth)}: {first}")
        return "\n".join(lines)


def _goal_block(session_id: str) -> str:
    """Goal text plus subgoals. Status is left out on purpose: it changes every turn and would
    move every frame behind it."""
    if not session_id:
        return ""
    try:
        from hermes_cli.goals import load_goal
        state = load_goal(session_id)
    except Exception as exc:
        logger.debug("loom: load_goal failed: %s", exc)
        return ""
    if state is None or state.status == "cleared" or not (state.goal or "").strip():
        return ""
    block = _clean_text(state.goal)
    subgoals = state.render_subgoals_block()
    return block + ("\n" + subgoals if subgoals else "")


def _append_to_content(content: Any, text: str) -> Any:
    if isinstance(content, str):
        return f"{content}\n\n{text}" if content else text
    if isinstance(content, list):
        return list(content) + [{"type": "text", "text": text}]
    return text if content is None else content


def _compression_kwargs() -> Dict[str, Any]:
    """Mirror the user's ``compression`` config for the inherited compressor (plugin engines are
    constructed without it)."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = (load_config_readonly() or {}).get("compression") or {}
    except Exception:
        return {}
    out: Dict[str, Any] = {}
    for key, name, cast in (("threshold", "threshold_percent", float), ("target_ratio", "summary_target_ratio", float),
                            ("protect_first_n", "protect_first_n", int), ("protect_last_n", "protect_last_n", int)):
        if cfg.get(key) is not None:
            try:
                out[name] = cast(cfg[key])
            except (TypeError, ValueError):
                pass
    if cfg.get("tail_mode"):
        out["tail_mode"] = str(cfg["tail_mode"]).strip().lower()
    return out


_TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Nest work for the active /goal like nested loops. 'push' a frame when you enter a level of work "
        "(its stable instructions, not findings); 'pop' when you leave it. Frames are shown to you at the "
        "front of the context, outermost first, and the last frame is your current loop. Move to a sibling "
        "with pop then push. Keep results and status in the conversation. "
        "Actions: push {text, label?}; pop {count?}; show."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["push", "pop", "show"]},
            "text": {"type": "string", "description": "push: the frame's stable instructions."},
            "label": {"type": "string", "description": "push: optional short name for the frame."},
            "count": {"type": "integer", "description": "pop: frames to pop (default 1)."},
        },
        "required": ["action"],
    },
}


class LoomEngine(ContextCompressor):
    """ContextCompressor plus a front-of-context /goal frame stack."""

    def __init__(self, model: str = "", **kwargs: Any) -> None:
        settings = _compression_kwargs()
        settings.update(kwargs)
        settings.setdefault("quiet_mode", True)
        super().__init__(model=model, **settings)
        self.stack = LoomStack()
        self._loom_session = ""
        self._loom_home = ""

    @property
    def name(self) -> str:
        return "loom"

    # -- state -------------------------------------------------------------------------------
    def _state_path(self) -> str:
        if not (self._loom_home and self._loom_session):
            return ""
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", self._loom_session)
        return os.path.join(self._loom_home, "loom", f"{safe}.json")

    def _save(self) -> None:
        path = self._state_path()
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.stack.to_dict(), f, ensure_ascii=True, indent=1)
        os.replace(tmp, path)

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        global _ACTIVE
        parent = getattr(super(), "on_session_start", None)
        if callable(parent):
            parent(session_id, **kwargs)
        self._loom_session = session_id or ""
        home = kwargs.get("hermes_home")
        if not home:
            try:
                from hermes_constants import get_hermes_home
                home = str(get_hermes_home())
            except Exception:
                home = ""
        self._loom_home = home or ""
        path = self._state_path()
        # A saved stack for this session wins; otherwise keep what we hold (a delegated child
        # starts inside its parent's frames, like an inner loop).
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.stack = LoomStack.from_dict(json.load(f))
            except Exception as exc:
                logger.warning("loom: could not read %s: %s", path, exc)
        _ACTIVE = self

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self.stack = LoomStack()

    # -- per-request selection ---------------------------------------------------------------
    def select_context(self, request_messages: List[Dict[str, Any]], **kwargs: Any) -> Optional[List[Dict[str, Any]]]:
        goal = _goal_block(self._loom_session)
        if not goal and not self.stack.frames:
            return None  # nothing nested: leave the request byte-identical
        if not request_messages or request_messages[0].get("role") != "system":
            return None
        first = dict(request_messages[0])
        first["content"] = _append_to_content(first.get("content"), self.stack.render(goal))
        return [first] + list(request_messages[1:])

    # -- tool --------------------------------------------------------------------------------
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [_TOOL_SCHEMA]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if name != TOOL_NAME:
            return super().handle_tool_call(name, args, **kwargs)
        args = args or {}
        action = str(args.get("action") or "").strip().lower()
        try:
            if action == "push":
                self.stack.push(args.get("text"), args.get("label"))
            elif action == "pop":
                self.stack.pop(int(args.get("count") or 1))
            elif action != "show":
                raise ValueError(f"unknown action {action!r}; use push, pop or show")
        except (ValueError, TypeError) as exc:
            return json.dumps({"error": str(exc), "depth": len(self.stack.frames), "frames": self.stack.outline()})
        if action != "show":
            self._save()
        return json.dumps({"ok": True, "depth": len(self.stack.frames), "frames": self.stack.outline()})


def _loom_command(raw_args: str = "") -> str:
    engine = _ACTIVE
    if engine is None:
        return "loom: no active session (set context.engine: loom in config.yaml)"
    goal = _goal_block(engine._loom_session)
    head = "goal: " + (goal.split("\n", 1)[0] if goal else "(none)")
    return head + "\n" + engine.stack.outline()


def register(ctx: Any) -> None:
    ctx.register_context_engine(LoomEngine())
    register_command = getattr(ctx, "register_command", None)
    if not callable(register_command):
        return
    try:  # the engine is loaded once per agent; register /loom only the first time
        from hermes_cli.plugins import get_plugin_manager
        if "loom" in get_plugin_manager()._plugin_commands:
            return
    except Exception:
        pass
    register_command("loom", _loom_command, description="Show the /goal frame stack kept at the front of context")
