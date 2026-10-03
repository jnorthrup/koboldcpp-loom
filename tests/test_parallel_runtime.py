#!/usr/bin/env python3
"""Runtime tests for the LOOM parallel engine (--parallelrequests) on a real GGUF model.

Requires a built koboldcpp library in the repo root and a model with built-in MTP layers
(validated with Qwen3.5-0.8B, a hybrid attention/recurrent model):

    KCPP_TEST_MODEL=/path/to/model.gguf python3 tests/test_parallel_runtime.py
    KCPP_TEST_MODEL=... python3 -m pytest -q tests/test_parallel_runtime.py

Set KCPP_TEST_PROFILES=1 to also allocate the 128K/256K/512K/1M context profiles.
Evidence (JSON + server logs) is written to KCPP_TEST_OUT (default: a temp directory).
Uses only the standard library.
"""
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("KCPP_TEST_MODEL", "")
PYEXE = os.environ.get("KCPP_TEST_PYTHON", sys.executable)
OUT = os.environ.get("KCPP_TEST_OUT", "") or tempfile.mkdtemp(prefix="kcpp-parallel-")
CTX = 4096
SLOTS = 4
DRAFT = 3
LONG_N = 1200
EVIDENCE = {}
HTTP_TIMEOUT = int(os.environ.get("KCPP_TEST_HTTP_TIMEOUT", "600"))


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Server:
    def __init__(self, name, args, ctx=CTX):
        self.name = name
        self.args = list(args)
        self.ctx = ctx
        self.port = _free_port()
        self.proc = None
        self.logpath = os.path.join(OUT, name + ".log")

    def command(self):
        return [PYEXE, os.path.join(ROOT, "koboldcpp.py"), "--model", MODEL, "--host", "127.0.0.1",
                "--port", str(self.port), "--contextsize", str(self.ctx), "--gpulayers", "999",
                "--skiplauncher"] + self.args

    def start(self, timeout=600):
        os.makedirs(OUT, exist_ok=True)
        self.log = open(self.logpath, "w")
        self.proc = subprocess.Popen(self.command(), cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("server %s exited with %s, see %s" % (self.name, self.proc.returncode, self.logpath))
            try:
                status, _ = self.get("/api/extra/version", timeout=2)
                if status == 200:
                    self.load_seconds = round(time.time() - t0, 2)
                    return self
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError("server %s did not start within %ss" % (self.name, timeout))

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if getattr(self, "log", None):
            self.log.close()

    def _req(self, method, path, obj=None, timeout=None):
        timeout = HTTP_TIMEOUT if timeout is None else timeout
        data = None if obj is None else json.dumps(obj).encode()
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), data=data, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                return resp.status, (json.loads(body) if body else None)
        except urllib.error.HTTPError as e:
            body = e.read()
            try:
                return e.code, json.loads(body)
            except Exception:
                return e.code, {"raw": body.decode("utf-8", "ignore")}

    def get(self, path, timeout=60):
        return self._req("GET", path, None, timeout)

    def post(self, path, obj, timeout=None):
        return self._req("POST", path, obj, timeout)

    def runtime(self):
        status, body = self.get("/api/extra/runtime")
        assert status == 200, body
        return body

    def log_text(self):
        self.log.flush()
        with open(self.logpath, "r", errors="ignore") as f:
            return f.read()


def gen(server, prompt, max_length, **kw):
    payload = {"prompt": prompt, "max_length": max_length, "temperature": 0, "top_k": 1, "rep_pen": 1.0,
               "sampler_seed": 7}
    payload.update(kw)
    status, body = server.post("/api/v1/generate", payload)
    return status, body


def run_concurrently(fns):
    results = [None] * len(fns)

    def runner(i, f):
        results[i] = f()

    threads = [threading.Thread(target=runner, args=(i, f)) for i, f in enumerate(fns)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


WORDS = ["ZEBRA", "MANGO", "VIOLET", "COPPER"]


def secret_prompt(word):
    return ("The secret word is %s. " % word) * 4 + "The secret word is"


def check(name, cond, detail):
    EVIDENCE.setdefault("checks", []).append({"name": name, "ok": bool(cond), "detail": detail})
    assert cond, "%s: %s" % (name, detail)


def test_startup_rejections():
    if not MODEL:
        return
    cases = [
        ("no_noshift", ["--parallelrequests", "2"], "context shifting"),
        ("smartcontext", ["--parallelrequests", "2", "--noshift", "--smartcontext"], "--smartcontext"),
        ("rope_conflict", ["--parallelrequests", "2", "--noshift", "--ropescaling", "yarn", "--overridenativecontext", "8192"], "--ropescaling"),
    ]
    out = {}
    for name, extra, needle in cases:
        srv = Server("reject_" + name, extra)
        proc = subprocess.run(srv.command(), cwd=ROOT, capture_output=True, text=True, timeout=300)
        text = proc.stdout + proc.stderr
        out[name] = {"exit": proc.returncode, "message": next((l for l in text.splitlines() if needle in l), "")}
        check("startup rejects " + name, proc.returncode == 2 and needle in text, out[name])
    EVIDENCE["startup_rejections"] = out


def test_parallel_engine():
    if not MODEL:
        return
    srv = Server("parallel", ["--parallelrequests", str(SLOTS), "--noshift", "--usemtp", "--draftamount", str(DRAFT),
                              "--multiuser", "16"]).start()
    ev = {"load_seconds": srv.load_seconds}
    try:
        rt = srv.runtime()
        ev["runtime_at_start"] = rt
        check("parallel enabled", rt["parallel"]["enabled"] and rt["parallel"]["slots"] == SLOTS, rt["parallel"])
        check("mtp active", rt["mtp"]["active"] and rt["mtp"]["parallel_lane"] and rt["mtp"]["speculative_type"] == "draft-mtp", rt["mtp"])

        # solo baselines, then the same four requests concurrently
        solo = {}
        for w in WORDS:
            st, body = gen(srv, secret_prompt(w), 12)
            check("solo %s ok" % w, st == 200, body)
            solo[w] = body["results"][0]
        conc = run_concurrently([lambda w=w: gen(srv, secret_prompt(w), 12) for w in WORDS])
        rt2 = srv.runtime()
        iso = {}
        for w, (st, body) in zip(WORDS, conc):
            check("concurrent %s ok" % w, st == 200, body)
            r = body["results"][0]
            text = r["text"]
            others = [o for o in WORDS if o != w and o in text.upper()]
            iso[w] = {"text": text, "solo_text": solo[w]["text"], "identical_to_solo": text == solo[w]["text"],
                      "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                      "draft_tokens": r.get("draft_tokens"), "draft_accepted": r.get("draft_accepted")}
            check("isolation %s own word" % w, w in text.upper(), iso[w])
            check("isolation %s no foreign word" % w, not others, {"foreign": others, "text": text})
            check("prompt tokens stable %s" % w, r["prompt_tokens"] == solo[w]["prompt_tokens"], iso[w])
        ev["isolation"] = iso
        ev["identical_to_solo"] = sum(1 for w in WORDS if iso[w]["identical_to_solo"])
        check("requests overlapped in slots", rt2["parallel"]["totals"]["peak_live"] >= 2, rt2["parallel"]["totals"])

        # exact completion counting with EOS banned, and MTP draft counters
        res = run_concurrently([lambda w=w: gen(srv, secret_prompt(w), 40, ban_eos_token=True) for w in WORDS])
        counters = []
        for st, body in res:
            check("ban_eos ok", st == 200, body)
            r = body["results"][0]
            counters.append({k: r.get(k) for k in ("completion_tokens", "draft_tokens", "draft_accepted")})
            check("completion_tokens exact", r["completion_tokens"] == 40, r)
            check("draft_accepted <= draft_tokens", 0 <= r["draft_accepted"] <= r["draft_tokens"], r)
        drafted = sum(c["draft_tokens"] for c in counters)
        accepted = sum(c["draft_accepted"] for c in counters)
        ev["counters"] = {"per_request": counters, "drafted": drafted, "accepted": accepted}
        check("mtp drafted in parallel lane", drafted > 0 and accepted > 0, ev["counters"])
        rt3 = srv.runtime()
        check("runtime totals carry draft counters", rt3["parallel"]["totals"]["draft_tokens"] >= drafted, rt3["parallel"]["totals"])

        # exact rendered-chat admission with completion-space reservation
        msgs = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Name one primary color."}]
        st, cnt = srv.post("/v1/chat/completions/input_tokens", {"messages": msgs})
        check("input_tokens ok", st == 200 and cnt["input_tokens"] > 0, cnt)
        n = cnt["input_tokens"]
        st, adm = srv.post("/api/extra/admission", {"messages": msgs, "max_tokens": CTX - n})
        check("admission exact fit", st == 200 and adm["admissible"] and adm["reserve"] == CTX and adm["input_tokens"] == n, adm)
        st, ok = srv.post("/v1/chat/completions", {"messages": msgs, "max_tokens": CTX - n, "temperature": 0, "stop": ["\n"]})
        check("chat exact fit admitted", st == 200, ok)
        check("chat usage.prompt_tokens == input_tokens", ok["usage"]["prompt_tokens"] == n, {"usage": ok["usage"], "input_tokens": n})
        st, over = srv.post("/v1/chat/completions", {"messages": msgs, "max_tokens": CTX - n + 1, "temperature": 0})
        check("chat one-over rejected", st == 400 and over["error"]["type"] == "exceed_context_size_error"
              and over["error"]["n_prompt_tokens"] == n and over["error"]["n_ctx"] == CTX, over)
        st, sover = srv.post("/v1/chat/completions", {"messages": msgs, "max_tokens": CTX - n + 1, "stream": True})
        check("streaming overflow rejected before SSE", st == 400 and sover["error"]["type"] == "exceed_context_size_error", sover)
        ev["admission"] = {"input_tokens": n, "exact_fit_usage": ok["usage"], "one_over": over, "stream_one_over_status": st}

        # long prompt: rejected with the exact (untruncated) count
        long_prompt = "alpha " * 5000
        st, body = gen(srv, long_prompt, 8)
        check("long prompt rejected", st == 400 and body["error"]["type"] == "exceed_context_size_error"
              and body["error"]["n_prompt_tokens"] >= 5000, body)
        mid_prompt = "alpha " * 3000
        st, body2 = gen(srv, mid_prompt, 2000)
        check("prompt fits but completion does not -> rejected", st == 400 and body2["error"]["type"] == "exceed_context_size_error", body2)
        st, body3 = gen(srv, mid_prompt, 64)
        check("prompt + small completion admitted", st == 200, body3)
        ev["overflow"] = {"long": body, "completion_overflow": body2, "admitted_prompt_tokens": body3["results"][0]["prompt_tokens"]}

        # per-request stateful samplers run in the parallel lane, isolated
        grammar = 'root ::= "yes" | "no"'
        res = run_concurrently([
            lambda: gen(srv, "Is the sky blue? Answer:", 4, grammar=grammar),
            lambda: gen(srv, secret_prompt("COPPER"), 8),
            lambda: gen(srv, "List three fruits:", 24, temperature=0.8, top_k=40, top_p=0.9, dry_multiplier=0.8, xtc_probability=0.5, sampler_seed=11),
            lambda: gen(srv, "Count: one two three", 16, mirostat=2, temperature=0.7, sampler_seed=5),
        ])
        texts = [b["results"][0]["text"] if s == 200 else b for s, b in res]
        check("grammar request constrained", res[0][0] == 200 and texts[0].strip() in ("yes", "no"), texts[0])
        check("neighbour unaffected by grammar", res[1][0] == 200 and "COPPER" in texts[1].upper(), texts[1])
        check("dry/xtc and mirostat requests ok", res[2][0] == 200 and res[3][0] == 200, texts[2:])
        ev["samplers"] = texts

        # explicit rejection of serial-only features
        png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        st, body = gen(srv, "Describe:", 8, images=[png])
        check("multimodal rejected explicitly", st == 400 and body["error"]["type"] == "not_supported_error", body)
        st, body = gen(srv, "Hello", 8, banned_tokens=["the secret word"])
        check("phrase ban rejected explicitly", st == 400 and body["error"]["type"] == "not_supported_error", body)
        ev["unsupported"] = body

        # cancellation isolation
        res = {}

        def long_req(key):
            res[key] = gen(srv, "Write a long story about a lighthouse keeper.", LONG_N, ban_eos_token=True, genkey=key,
                           temperature=0.7, sampler_seed=3)

        ta = threading.Thread(target=long_req, args=("KEYA",))
        tb = threading.Thread(target=long_req, args=("KEYB",))
        before = srv.runtime()["parallel"]["totals"]["aborted"]
        ta.start(); tb.start()
        time.sleep(1.5)
        st, ab = srv.post("/api/extra/abort", {"genkey": "KEYA"})
        ta.join(); tb.join()
        after = srv.runtime()["parallel"]["totals"]["aborted"]
        ra = res["KEYA"][1]["results"][0]
        rb = res["KEYB"][1]["results"][0]
        ev["cancellation"] = {"abort_response": ab, "a_completion": ra["completion_tokens"], "b_completion": rb["completion_tokens"],
                              "aborted_before": before, "aborted_after": after}
        check("abort targeted A", st == 200 and ab.get("success") == "true", ab)
        check("A stopped early", ra["completion_tokens"] < LONG_N, ev["cancellation"])
        check("B unaffected", rb["completion_tokens"] == LONG_N, ev["cancellation"])
        check("exactly one abort counted", after == before + 1, ev["cancellation"])

        # slots endpoint and serial lane after parallel use
        st, slots = srv.get("/slots")
        check("slots endpoint", st == 200 and len(slots) == SLOTS and all("draft_n" in s or not s["is_processing"] for s in slots), slots)
        ev["slots"] = slots
        ev["runtime_at_end"] = srv.runtime()
    finally:
        srv.stop()
        EVIDENCE["parallel"] = ev


def test_parallelserial_routing():
    if not MODEL:
        return
    srv = Server("parallelserial", ["--parallelrequests", "2", "--noshift", "--parallelserial"]).start()
    ev = {}
    try:
        st, body = gen(srv, "The secret word is ZEBRA. The secret word is", 8, banned_tokens=["ZEBRA. The"])
        check("phrase ban routed to serial lane", st == 200, body)
        check("routing logged", "routed to the serial lane" in srv.log_text(), "log")
        res = run_concurrently([lambda w=w: gen(srv, secret_prompt(w), 8) for w in WORDS[:2]])
        check("parallel lane still serves", all(s == 200 for s, _ in res), res)
        ev = {"serial_text": body["results"][0]["text"], "parallel": [b["results"][0]["text"] for _, b in res]}
    finally:
        srv.stop()
        EVIDENCE["parallelserial"] = ev


PROFILE_LINE = re.compile(r"Context profile: .*")
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def passkey_prompt(srv, target_tokens, key):
    # place the key at 30% depth inside ~target_tokens of filler (exact count via the engine)
    st, cnt = srv.post("/api/extra/tokencount", {"prompt": FILLER})
    per = max(1, cnt["value"])
    reps = max(1, int(target_tokens / per))
    at = int(reps * 0.3)
    body = FILLER * at + ("The pass key is %d. Remember it. %d is the pass key. " % (key, key)) + FILLER * (reps - at)
    return body + "\nWhat is the pass key? The pass key is"


def test_context_profiles():
    if not MODEL or os.environ.get("KCPP_TEST_PROFILES", "") != "1":
        return
    profiles = {}
    plan = [(131072, []), (262144, []), (524288, ["--ropescaling", "yarn"]), (1048576, ["--ropescaling", "yarn"])]
    probe_depths = [int(x) for x in os.environ.get("KCPP_TEST_PROBE", "4000").split(",") if x]
    for ctx, rope in plan:
        extra = ["--quantkv", "q8_0", "--usemtp", "--draftamount", "2", "--noshift"] + rope
        srv = Server("profile_%d" % ctx, extra, ctx=ctx)
        try:
            srv.start(timeout=900)
            rt = srv.runtime()
            st, body = gen(srv, "The capital of France is", 8)
            m = PROFILE_LINE.search(srv.log_text())
            probes = []
            for depth in probe_depths:
                if depth + 64 > ctx:
                    continue
                key = 70000 + depth % 9973
                prompt = passkey_prompt(srv, depth, key)
                t0 = time.time()
                pst, pbody = gen(srv, prompt, 8)
                dt = time.time() - t0
                txt = pbody["results"][0]["text"] if pst == 200 else str(pbody)
                probes.append({"target_tokens": depth, "prompt_tokens": pbody["results"][0]["prompt_tokens"] if pst == 200 else None,
                               "seconds": round(dt, 2), "found": str(key) in txt, "text": txt})
            profiles[ctx] = {"load_seconds": srv.load_seconds, "context": rt["context"], "mtp_active": rt["mtp"]["active"],
                             "generate_ok": st == 200, "sample": body["results"][0]["text"] if st == 200 else body,
                             "log": m.group(0) if m else "", "passkey": probes}
            check("profile %d allocated" % ctx, rt["context"]["allocated_cells"] >= ctx and st == 200, profiles[ctx]["context"])
            check("profile %d quality not claimed" % ctx, rt["context"]["quality_verified"] is False, rt["context"]["quality_status"])
            check("profile %d mtp active" % ctx, rt["mtp"]["active"], rt["mtp"])
        except Exception as e:
            profiles[ctx] = {"error": str(e)}
            check("profile %d" % ctx, False, str(e))
        finally:
            srv.stop()
    EVIDENCE["profiles"] = profiles


def main():
    if not MODEL:
        print("KCPP_TEST_MODEL is not set; nothing to run")
        return 0
    failures = []
    for t in (test_startup_rejections, test_parallel_engine, test_parallelserial_routing, test_context_profiles):
        try:
            t()
            print("PASS", t.__name__)
        except Exception as e:
            failures.append((t.__name__, str(e)))
            print("FAIL", t.__name__, str(e)[:2000])
    n_checks = len(EVIDENCE.get("checks", []))
    if n_checks == 0:
        failures.append(("suite", "no checks executed"))
    EVIDENCE["failures"] = failures
    print("checks executed: %d, failed tests: %d" % (n_checks, len(failures)))
    path = os.path.join(OUT, "evidence.json")
    with open(path, "w") as f:
        json.dump(EVIDENCE, f, indent=1, default=str)
    print("evidence:", path)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
