"""/v1/looms: lifecycle and artifact storage for loom scripts (koboldcpp-loom).

The server-side mate of the loom Hermes plugin's loop precook (github.com/jnorthrup/
loom-hermes-plugin, precook/loom_loop.py). A loom script is an outline tree; running it walks
the outline depth-first as nested loops, one chat request per leaf, each carrying system prompt +
goal + the frames of every enclosing loop. Runs go through this server's own
/v1/chat/completions over loopback, so they take the normal queue, lock, and --loomcache path;
nothing here touches the model directly. Every run is stored as a replayable artifact.

Endpoints (OpenAI-style objects; enabled with --loomdir DIR):
  POST   /v1/looms                          store a loom script -> loom object (content-addressed id)
  GET    /v1/looms                          list
  GET    /v1/looms/{id}                     loom object + script
  DELETE /v1/looms/{id}                     delete script and its runs
  POST   /v1/looms/{id}/runs                start a run {prewarm?, request?, replay_of?} -> run object
  GET    /v1/looms/{id}/runs                list runs
  GET    /v1/looms/{id}/runs/{run_id}       run object (status, leaves, usage)
  GET    /v1/looms/{id}/runs/{run_id}/artifact   loom-run/1 JSONL (one request+response per leaf)
  POST   /v1/looms/{id}/runs/{run_id}/cancel

Storage is scoped per account: the owner is a hash of the bearer key the request presented, so
one key never sees another key's looms. Nothing is sent to the model beyond ordinary chat
requests; there are no cache hints.
"""

import hashlib
import json
import os
import re
import ssl
import threading
import time
import urllib.request

# --- frame rendering: byte-identical copy of loom-hermes-plugin/loomframes.py -----------------
FRAMES_FORMAT = "loom-frames/1"
FRAMES_OPEN = "[loom: nested work frames, outermost first; the last frame is the current loop]"
FRAMES_CLOSE = "[/loom]"


def render_frames(frames, goal=""):
    parts = [FRAMES_OPEN]
    if goal:
        parts.append("== goal ==\n" + goal)
    for depth, frame in enumerate(frames):
        label = frame.get("label") or ""
        parts.append(f"== {depth + 1}" + (f" {label}" if label else "") + f" ==\n{frame['text']}")
    parts.append(FRAMES_CLOSE)
    return "\n".join(parts)


def with_frames(system, frames, goal=""):
    if not frames and not goal:
        return system
    block = render_frames(frames, goal)
    return f"{system}\n\n{block}" if system else block
# -------------------------------------------------------------------------------------------------

SCRIPT_FORMAT = "loom-script/1"
DEFAULT_TASK = "Carry out the work of the innermost frame."
MAX_SCRIPT_BYTES = 8 * 1024 * 1024
MAX_LEAVES = 10000
_ID_RE = re.compile(r"^(loom|lrun)_[0-9a-f]{8,64}$")


class LoomError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def _validate(script):
    if not isinstance(script, dict):
        raise LoomError(400, "loom script must be a JSON object")
    if script.get("format", SCRIPT_FORMAT) != SCRIPT_FORMAT:
        raise LoomError(400, f"unsupported format {script.get('format')!r}; expected {SCRIPT_FORMAT}")
    outline = script.get("outline")
    if not isinstance(outline, list) or not outline:
        raise LoomError(400, "loom script needs a non-empty 'outline' list")
    if "request" in script and not isinstance(script["request"], dict):
        raise LoomError(400, "'request' must be an object")
    leaves, depth = 0, 0

    def walk(nodes, d):
        nonlocal leaves, depth
        for n in nodes:
            if not isinstance(n, dict) or not str(n.get("text") or "").strip():
                raise LoomError(400, "every outline node needs non-empty 'text'")
            kids = n.get("children") or []
            if not isinstance(kids, list):
                raise LoomError(400, "'children' must be a list")
            depth = max(depth, d)
            if kids:
                walk(kids, d + 1)
            else:
                leaves += 1
                if leaves > MAX_LEAVES:
                    raise LoomError(400, f"outline has more than {MAX_LEAVES} leaves")

    walk(outline, 1)
    return leaves, depth


def iter_leaves(script):
    """Depth-first nested-loop walk: yields (path_labels, request_body) per leaf.
    Same frames, same order, same bytes as loom_loop.Loom in the plugin."""
    frames, path = [], []
    system = str(script.get("system") or "")
    goal = str(script.get("goal") or "")
    base = dict(script.get("request") or {})

    def walk(nodes):
        for n in nodes:
            frames.append({"label": str(n.get("label") or ""), "text": str(n["text"]).replace("\r\n", "\n").strip()})
            path.append(str(n.get("label") or ""))
            try:
                kids = n.get("children") or []
                if kids:
                    yield from walk(kids)
                else:
                    body = dict(base)
                    body["messages"] = [
                        {"role": "system", "content": with_frames(system, frames, goal)},
                        {"role": "user", "content": str(n.get("task") or "").strip() or DEFAULT_TASK},
                    ]
                    yield list(path), body
            finally:
                frames.pop()
                path.pop()

    yield from walk(script["outline"])


class LoomStore:
    def __init__(self, root, loopback_url, loopback_key=None, log=print):
        self.root = os.path.abspath(root)
        self.loopback_url = loopback_url.rstrip("/")
        self.loopback_key = loopback_key
        self.log = log
        self.lock = threading.Lock()
        self.cancel = {}
        os.makedirs(self.root, exist_ok=True)
        self._recover()

    # -- layout: root/<owner>/<loom_id>/{loom.json,script.json,runs/<run_id>.{json,jsonl}} --------
    @staticmethod
    def owner_of(headers):
        auth = headers.get("Authorization") or headers.get("authorization") or ""
        token = auth[len("Bearer "):].strip() if auth.startswith("Bearer ") else ""
        return "acct_" + hashlib.sha256(token.encode()).hexdigest()[:16] if token else "local"

    def _dir(self, owner, loom_id=None):
        d = os.path.join(self.root, owner)
        if loom_id is not None:
            if not _ID_RE.match(loom_id):
                raise LoomError(404, "no such loom")
            d = os.path.join(d, loom_id)
        return d

    @staticmethod
    def _write_json(path, obj):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)

    @staticmethod
    def _read_json(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _recover(self):
        """A run that was in flight when the server stopped is marked interrupted, not left running."""
        for dirpath, _dirs, files in os.walk(self.root):
            for name in files:
                if name.startswith("lrun_") and name.endswith(".json"):
                    p = os.path.join(dirpath, name)
                    try:
                        run = self._read_json(p)
                        if run.get("status") in ("queued", "running"):
                            run["status"], run["error"] = "interrupted", "server stopped during the run"
                            self._write_json(p, run)
                    except Exception:
                        pass

    # -- looms ---------------------------------------------------------------------------------
    def create(self, owner, body):
        if len(body or b"") > MAX_SCRIPT_BYTES:
            raise LoomError(413, "loom script too large")
        try:
            script = json.loads(body)
        except Exception:
            raise LoomError(400, "body is not JSON")
        leaves, depth = _validate(script)
        canon = json.dumps(script, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        loom_id = "loom_" + hashlib.sha256(canon).hexdigest()[:24]
        d = self._dir(owner, loom_id)
        with self.lock:
            if os.path.exists(os.path.join(d, "loom.json")):
                return self._read_json(os.path.join(d, "loom.json"))
            os.makedirs(os.path.join(d, "runs"), exist_ok=True)
            self._write_json(os.path.join(d, "script.json"), script)
            obj = {"id": loom_id, "object": "loom", "created_at": int(time.time()), "format": SCRIPT_FORMAT,
                   "frames_format": FRAMES_FORMAT, "bytes": len(canon), "leaves": leaves, "depth": depth,
                   "name": str(script.get("name") or "")}
            self._write_json(os.path.join(d, "loom.json"), obj)
            return obj

    def list(self, owner):
        d = self._dir(owner)
        out = []
        if os.path.isdir(d):
            for loom_id in sorted(os.listdir(d)):
                p = os.path.join(d, loom_id, "loom.json")
                if os.path.exists(p):
                    out.append(self._read_json(p))
        return {"object": "list", "data": out}

    def get(self, owner, loom_id, with_script=True):
        d = self._dir(owner, loom_id)
        p = os.path.join(d, "loom.json")
        if not os.path.exists(p):
            raise LoomError(404, "no such loom")
        obj = self._read_json(p)
        if with_script:
            obj["script"] = self._read_json(os.path.join(d, "script.json"))
        return obj

    def delete(self, owner, loom_id):
        d = self._dir(owner, loom_id)
        if not os.path.exists(os.path.join(d, "loom.json")):
            raise LoomError(404, "no such loom")
        runs = os.path.join(d, "runs")
        for name in os.listdir(runs):
            if name.endswith(".json") and self._read_json(os.path.join(runs, name)).get("status") in ("queued", "running"):
                raise LoomError(409, "loom has a run in progress; cancel it first")
        with self.lock:
            for dirpath, dirs, files in os.walk(d, topdown=False):
                for f in files:
                    os.remove(os.path.join(dirpath, f))
                for sub in dirs:
                    os.rmdir(os.path.join(dirpath, sub))
            os.rmdir(d)
        return {"id": loom_id, "object": "loom.deleted", "deleted": True}

    # -- runs ----------------------------------------------------------------------------------
    def _run_path(self, owner, loom_id, run_id, ext):
        if not _ID_RE.match(run_id or ""):
            raise LoomError(404, "no such run")
        return os.path.join(self._dir(owner, loom_id), "runs", run_id + ext)

    def start_run(self, owner, loom_id, body, auth_header):
        loom = self.get(owner, loom_id)
        try:
            opts = json.loads(body) if body else {}
        except Exception:
            raise LoomError(400, "body is not JSON")
        if not isinstance(opts, dict):
            raise LoomError(400, "run options must be an object")
        replay_of = opts.get("replay_of")
        if replay_of:
            src = self._run_path(owner, loom_id, replay_of, ".jsonl")
            if not os.path.exists(src):
                raise LoomError(404, "no such run to replay")
        run_id = "lrun_" + hashlib.sha256(f"{loom_id}{time.time_ns()}{os.getpid()}".encode()).hexdigest()[:24]
        run = {"id": run_id, "object": "loom.run", "loom_id": loom_id, "created_at": int(time.time()),
               "status": "queued", "prewarm": bool(opts.get("prewarm")), "replay_of": replay_of or None,
               "leaves_total": loom["leaves"], "leaves_done": 0,
               "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "error": None}
        self._write_json(self._run_path(owner, loom_id, run_id, ".json"), run)
        self.cancel[run_id] = threading.Event()
        threading.Thread(target=self._run, daemon=True,
                         args=(owner, loom, run, opts, auth_header)).start()
        return run

    def _requests(self, owner, loom, run, opts):
        if run["replay_of"]:
            with open(self._run_path(owner, loom["id"], run["replay_of"], ".jsonl"), encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    yield rec.get("path") or [], dict(rec["request"])
            return
        extra = opts.get("request") if isinstance(opts.get("request"), dict) else {}
        for path, body in iter_leaves(loom["script"]):
            body.update(extra)
            yield path, body

    def _run(self, owner, loom, run, opts, auth_header):
        run_json = self._run_path(owner, loom["id"], run["id"], ".json")
        artifact = self._run_path(owner, loom["id"], run["id"], ".jsonl")
        stop = self.cancel[run["id"]]
        run["status"], run["started_at"] = "running", int(time.time())
        self._write_json(run_json, run)
        try:
            with open(artifact, "w", encoding="utf-8") as out:
                for path, body in self._requests(owner, loom, run, opts):
                    if stop.is_set():
                        run["status"] = "cancelled"
                        break
                    if run["prewarm"]:
                        body["max_tokens"] = 1
                    body["stream"] = False
                    started = time.time()
                    reply = self._post(body, auth_header)
                    usage = reply.get("usage") or {}
                    for k in run["usage"]:
                        run["usage"][k] += int(usage.get(k) or 0)
                    run["leaves_done"] += 1
                    out.write(json.dumps({"format": "loom-run/1", "frames_format": FRAMES_FORMAT, "path": path,
                                          "request": body, "response": reply,
                                          "seconds": round(time.time() - started, 3)}, ensure_ascii=False) + "\n")
                    out.flush()
                    self._write_json(run_json, run)
            if run["status"] == "running":
                run["status"] = "completed"
        except Exception as e:
            run["status"], run["error"] = "failed", str(e)[:500]
        finally:
            run["finished_at"] = int(time.time())
            self._write_json(run_json, run)
            self.cancel.pop(run["id"], None)
            self.log(f"LoomStore: run {run['id']} {run['status']} ({run['leaves_done']}/{run['leaves_total']} leaves)")

    def _post(self, body, auth_header):
        req = urllib.request.Request(self.loopback_url + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        if auth_header:
            req.add_header("Authorization", auth_header)
        ctx = None
        if self.loopback_url.startswith("https"):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE  # loopback to ourselves; the cert names the public host
        with urllib.request.urlopen(req, timeout=3600, context=ctx) as r:
            return json.load(r)

    def list_runs(self, owner, loom_id):
        self.get(owner, loom_id, with_script=False)
        runs = os.path.join(self._dir(owner, loom_id), "runs")
        data = [self._read_json(os.path.join(runs, n)) for n in sorted(os.listdir(runs)) if n.endswith(".json")]
        data.sort(key=lambda r: r.get("created_at", 0))
        return {"object": "list", "data": data}

    def get_run(self, owner, loom_id, run_id):
        p = self._run_path(owner, loom_id, run_id, ".json")
        if not os.path.exists(p):
            raise LoomError(404, "no such run")
        return self._read_json(p)

    def artifact(self, owner, loom_id, run_id):
        p = self._run_path(owner, loom_id, run_id, ".jsonl")
        if not os.path.exists(p):
            raise LoomError(404, "no artifact for this run")
        with open(p, "rb") as f:
            return f.read()

    def cancel_run(self, owner, loom_id, run_id):
        run = self.get_run(owner, loom_id, run_id)
        ev = self.cancel.get(run_id)
        if ev is None:
            raise LoomError(409, f"run is {run['status']}, not running")
        ev.set()
        return {**run, "cancel_requested": True}

    # -- HTTP routing --------------------------------------------------------------------------
    def handle(self, method, path, headers, body):
        """Returns (status, bytes, content_type), or None when the path is not a loom path."""
        path = path.split("?")[0].rstrip("/")
        m = re.match(r"^(?:/v1)?/looms(?:/([^/]+))?(?:/(runs))?(?:/([^/]+))?(?:/(artifact|cancel))?$", path)
        if not m:
            return None
        loom_id, runs, run_id, leaf = m.groups()
        owner = self.owner_of(headers)
        try:
            if loom_id is None:
                if method == "POST":
                    result = self.create(owner, body)
                elif method == "GET":
                    result = self.list(owner)
                else:
                    raise LoomError(405, "method not allowed")
            elif runs is None:
                if method == "GET":
                    result = self.get(owner, loom_id)
                elif method == "DELETE":
                    result = self.delete(owner, loom_id)
                else:
                    raise LoomError(405, "method not allowed")
            elif run_id is None:
                if method == "POST":
                    auth = headers.get("Authorization") or headers.get("authorization")
                    result = self.start_run(owner, loom_id, body, auth)
                elif method == "GET":
                    result = self.list_runs(owner, loom_id)
                else:
                    raise LoomError(405, "method not allowed")
            elif leaf == "artifact" and method == "GET":
                self.get(owner, loom_id, with_script=False)
                return 200, self.artifact(owner, loom_id, run_id), "application/x-ndjson"
            elif leaf == "cancel" and method == "POST":
                result = self.cancel_run(owner, loom_id, run_id)
            elif leaf is None and method == "GET":
                self.get(owner, loom_id, with_script=False)
                result = self.get_run(owner, loom_id, run_id)
            else:
                raise LoomError(405, "method not allowed")
            return 200, json.dumps(result).encode(), "application/json"
        except LoomError as e:
            return e.code, json.dumps({"error": {"message": str(e), "type": "loom_error", "code": e.code}}).encode(), "application/json"
