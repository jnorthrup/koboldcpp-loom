"""loom: a Hermes context engine that keeps a /goal plan at the front of every request.

The plan is a tree of prompt segments. Each segment is an independent obstack frame: it is
pushed once, its text never changes, and it only refers to its parent. Nothing interlocks, so
switching work between sibling branches is a pop back to the shared ancestor plus a push of the
new branch.

On every model request the goal and every segment, in the order they were pushed, are rendered
onto the end of the system prompt. Pushing only appends to that text, so what is already there
never moves. The focus pointer, the fastest-moving fact, rides on the newest message, which the
server recomputes anyway. The shape is a whip: a long base that does not move and a short tip
that moves every turn. A prefix cache on the server (koboldcpp --loomcache, or any provider
prompt cache) therefore keeps the front and recomputes only the tail. Planning the tree up front
matters: a push made late re-sends the conversation that sits after the trunk.

There are no cache hints, tiers or segment counts anywhere: the server finds the reusable
prefix from the tokens alone. Compaction is inherited unchanged from the built-in compressor;
it only rewrites conversation turns, never the trunk.

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
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MAX_TEXT = 4000
_MAX_NODES = 256
TRUNK_OPEN = "[loom trunk: the planned work tree, in the order it was laid down; the current focus is noted on the newest message]"
TRUNK_CLOSE = "[/loom trunk]"

_ACTIVE: Optional["LoomEngine"] = None  # engine of the most recently started session, for /loom


def _clean_text(text: Any) -> str:
    return str(text or "").replace("\r\n", "\n").strip()


class LoomTree:
    """Append-only tree of prompt segments with a focus pointer (the obstack top)."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}  # id -> {"text", "parent"}; insertion ordered
        self.focus: Optional[str] = None

    # -- persistence -------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": [{"id": k, **v} for k, v in self.nodes.items()], "focus": self.focus}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoomTree":
        tree = cls()
        for node in data.get("nodes") or []:
            nid = str(node.get("id") or "")
            parent = node.get("parent")
            if _ID_RE.match(nid) and (parent is None or parent in tree.nodes):
                tree.nodes[nid] = {"text": _clean_text(node.get("text")), "parent": parent}
        focus = data.get("focus")
        tree.focus = focus if focus in tree.nodes else None
        return tree

    # -- structure ---------------------------------------------------------------------------
    def path(self, nid: Optional[str]) -> List[str]:
        out: List[str] = []
        while nid is not None:
            out.append(nid)
            nid = self.nodes[nid]["parent"]
        return list(reversed(out))

    def children(self, nid: Optional[str]) -> List[str]:
        return [k for k, v in self.nodes.items() if v["parent"] == nid]

    def add(self, nid: str, text: str, parent: Optional[str]) -> str:
        """Push a segment. Re-adding an identical segment is a no-op; changing one is refused."""
        text = _clean_text(text)
        if not _ID_RE.match(nid or ""):
            raise ValueError(f"bad id {nid!r}: use 1-64 chars of letters, digits, '_', '.', '-'")
        if parent is not None and parent not in self.nodes:
            raise ValueError(f"unknown parent {parent!r}")
        if not text:
            raise ValueError(f"segment {nid!r} has no text")
        if len(text) > _MAX_TEXT:
            raise ValueError(f"segment {nid!r} is {len(text)} chars; limit {_MAX_TEXT}. Split it into children.")
        existing = self.nodes.get(nid)
        if existing is not None:
            if existing["text"] == text and existing["parent"] == parent:
                return "unchanged"
            raise ValueError(
                f"segment {nid!r} already exists and segments are immutable; push a new id instead "
                "(editing a segment would move every cached token after it)")
        if len(self.nodes) >= _MAX_NODES:
            raise ValueError(f"tree is full ({_MAX_NODES} segments)")
        self.nodes[nid] = {"text": text, "parent": parent}
        return "added"

    def plan(self, specs: List[Dict[str, Any]], parent: Optional[str]) -> List[str]:
        """Add a nested list of ``{id, text, children}`` under ``parent``; all-or-nothing."""
        staged = LoomTree.from_dict(self.to_dict())
        added: List[str] = []

        def walk(items: Any, under: Optional[str]) -> None:
            if not isinstance(items, list):
                raise ValueError("'nodes' must be a list of {id, text, children}")
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("each node must be an object with id and text")
                nid = str(item.get("id") or "")
                if staged.add(nid, item.get("text"), under) == "added":
                    added.append(nid)
                walk(item.get("children") or [], nid)

        walk(specs, parent)
        self.nodes, self.focus = staged.nodes, staged.focus
        return added

    # -- rendering ---------------------------------------------------------------------------
    def render_trunk(self, goal_block: str) -> str:
        """Every segment in push order. Pushing only appends here and focus is not rendered, so
        earlier text never moves: the trunk grows at its end and nowhere else."""
        parts = [TRUNK_OPEN]
        if goal_block:
            parts.append("== goal ==\n" + goal_block)
        for nid, node in self.nodes.items():
            under = f" (under {node['parent']})" if node["parent"] else ""
            parts.append(f"== {nid}{under} ==\n{node['text']}")
        parts.append(TRUNK_CLOSE)
        return "\n".join(parts)

    def render_focus(self) -> str:
        if self.focus is None:
            return "[loom focus: root]"
        return "[loom focus: " + " > ".join(self.path(self.focus)) + "]"

    def render_outline(self) -> str:
        if not self.nodes:
            return "(empty)"
        on_path = set(self.path(self.focus))
        lines: List[str] = []

        def walk(under: Optional[str], depth: int) -> None:
            for nid in self.children(under):
                mark = "*" if nid == self.focus else ("|" if nid in on_path else " ")
                first = self.nodes[nid]["text"].split("\n", 1)[0]
                if len(first) > 70:
                    first = first[:67] + "..."
                lines.append(f"{mark} {'  ' * depth}{nid}: {first}")
                walk(nid, depth + 1)

        walk(None, 0)
        return "\n".join(lines)


def _goal_block(session_id: str) -> str:
    """Goal text plus subgoals, in their (append-only) stored order. Status is left out on purpose:
    it changes every turn and would move the whole trunk."""
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
        "Plan work for the active /goal as a tree of prompt segments kept at the front of the context. "
        "Segments are immutable once pushed and every segment is shown to you on every turn in push order; the "
        "current focus path is noted on the newest message. Lay the whole tree down up front with 'plan' (pushes "
        "made later re-send everything after the trunk), move between branches with 'focus' (free: it only "
        "changes the newest message), 'push' a child of the focus or 'pop' to its parent, and keep volatile "
        "findings in the conversation, not in segments. Actions: plan {nodes:[{id,text,children:[...]}], parent?}; "
        "push {id,text}; pop; focus {id}; show."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["plan", "push", "pop", "focus", "show"]},
            "id": {"type": "string", "description": "Segment id (push, focus)."},
            "text": {"type": "string", "description": "Segment text (push). Stable instructions only."},
            "parent": {"type": "string", "description": "plan: attach under this segment (default: root)."},
            "nodes": {
                "type": "array",
                "description": "plan: nested segments [{id, text, children: [...]}].",
                "items": {"type": "object"},
            },
        },
        "required": ["action"],
    },
}


class LoomEngine(ContextCompressor):
    """ContextCompressor plus a front-of-context /goal plan tree."""

    def __init__(self, model: str = "", **kwargs: Any) -> None:
        settings = _compression_kwargs()
        settings.update(kwargs)
        settings.setdefault("quiet_mode", True)
        super().__init__(model=model, **settings)
        self.tree = LoomTree()
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
            json.dump(self.tree.to_dict(), f, ensure_ascii=True, indent=1)
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
        # A saved tree for this session wins; otherwise keep what we hold (a delegated child
        # starts from its parent's tree, so it shares the parent's front of context).
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.tree = LoomTree.from_dict(json.load(f))
            except Exception as exc:
                logger.warning("loom: could not read %s: %s", path, exc)
        _ACTIVE = self

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self.tree = LoomTree()

    # -- per-request selection ---------------------------------------------------------------
    def select_context(self, request_messages: List[Dict[str, Any]], **kwargs: Any) -> Optional[List[Dict[str, Any]]]:
        goal = _goal_block(self._loom_session)
        if not goal and not self.tree.nodes:
            return None  # nothing planned: leave the request byte-identical
        if not request_messages or request_messages[0].get("role") != "system":
            return None
        first = dict(request_messages[0])
        first["content"] = _append_to_content(first.get("content"), self.tree.render_trunk(goal))
        out = [first] + list(request_messages[1:])
        # The focus pointer is the fastest-moving fact, so it rides on the newest row, which the
        # server recomputes anyway. Request-only: persisted history is untouched.
        if self.tree.nodes and len(out) > 1 and out[-1].get("role") in ("user", "tool"):
            last = dict(out[-1])
            last["content"] = _append_to_content(last.get("content"), self.tree.render_focus())
            out[-1] = last
        return out

    # -- tool --------------------------------------------------------------------------------
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [_TOOL_SCHEMA]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if name != TOOL_NAME:
            return super().handle_tool_call(name, args, **kwargs)
        try:
            result = self._dispatch(args or {})
        except ValueError as exc:
            return json.dumps({"error": str(exc), "outline": self.tree.render_outline()})
        self._save()
        result.setdefault("focus", self.tree.focus)
        result.setdefault("outline", self.tree.render_outline())
        return json.dumps(result)

    def _dispatch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        action = str(args.get("action") or "").strip().lower()
        tree = self.tree
        if action == "plan":
            parent = args.get("parent") or None
            added = tree.plan(args.get("nodes") or [], parent)
            if tree.focus is None and added:
                first_child = tree.children(parent)
                tree.focus = first_child[0] if first_child else added[0]
            return {"ok": True, "added": added}
        if action == "push":
            nid = str(args.get("id") or "")
            status = tree.add(nid, args.get("text"), tree.focus)
            tree.focus = nid
            return {"ok": True, "push": status}
        if action == "pop":
            if tree.focus is None:
                raise ValueError("already at the root")
            tree.focus = tree.nodes[tree.focus]["parent"]
            return {"ok": True}
        if action == "focus":
            nid = args.get("id") or None
            if nid is not None and nid not in tree.nodes:
                raise ValueError(f"unknown segment {nid!r}")
            tree.focus = nid
            return {"ok": True}
        if action == "show":
            return {"ok": True, "trunk": tree.render_trunk(_goal_block(self._loom_session))}
        raise ValueError(f"unknown action {action!r}; use plan, push, pop, focus or show")


def _loom_command(raw_args: str = "") -> str:
    engine = _ACTIVE
    if engine is None:
        return "loom: no active session (set context.engine: loom in config.yaml)"
    goal = _goal_block(engine._loom_session)
    head = "goal: " + (goal.split("\n", 1)[0] if goal else "(none)")
    return head + "\n" + engine.tree.render_outline() + "\n(* = focus, | = on the trunk path)"


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
    register_command("loom", _loom_command, description="Show the /goal plan tree kept at the front of context")
