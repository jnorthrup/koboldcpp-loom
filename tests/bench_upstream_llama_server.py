#!/usr/bin/env python3
"""Reference measurement for docs/evidence: upstream llama-server MTP on/off with the same prompts as bench_parallel_runtime.

    python3 tests/bench_upstream_llama_server.py /path/to/llama-server /path/to/model.gguf OUT_DIR

Build llama-server at 7fe450e19305b828c199d602c23a8337aaa1f03b (v0.5.0): cmake -B build -DGGML_METAL=ON -DLLAMA_CURL=OFF; cmake --build build --target llama-server.
"""
import json, os, socket, subprocess, sys, threading, time, urllib.request

BIN = sys.argv[1]
MODEL = sys.argv[2]
OUT = sys.argv[3]
GEN_N = 128
PROMPTS = ["Write a detailed essay about %s." % s for s in ("rivers", "mountains", "deserts", "forests")]


def port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def post(p, path, obj):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (p, path), data=json.dumps(obj).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def run(name, extra):
    p = port()
    log = open(os.path.join(OUT, "ref_%s.log" % name), "w")
    cmd = [BIN, "-m", MODEL, "--host", "127.0.0.1", "--port", str(p), "-c", "16384", "-np", "4", "-ngl", "999", "--no-webui"] + extra
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/health" % p, timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        body = {"prompt": PROMPTS[0], "n_predict": 4, "temperature": 0, "ignore_eos": True}
        post(p, "/completion", body)
        def one(pr):
            t0 = time.time()
            r = post(p, "/completion", {"prompt": pr, "n_predict": GEN_N, "temperature": 0, "ignore_eos": True, "cache_prompt": False})
            return time.time() - t0, r
        dt, r = one(PROMPTS[0])
        t = r.get("timings", {})
        res = {"gen1": {"seconds": round(dt, 3), "tok_s": round(GEN_N / dt, 2), "predicted_per_second": t.get("predicted_per_second"),
                        "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted")}}
        out = [None] * 4
        ths = [threading.Thread(target=lambda i=i: out.__setitem__(i, one(PROMPTS[i]))) for i in range(4)]
        t0 = time.time()
        [th.start() for th in ths]; [th.join() for th in ths]
        wall = time.time() - t0
        dn = sum((o[1].get("timings", {}).get("draft_n") or 0) for o in out)
        da = sum((o[1].get("timings", {}).get("draft_n_accepted") or 0) for o in out)
        res["gen4"] = {"wall_seconds": round(wall, 3), "aggregate_tok_s": round(4 * GEN_N / wall, 2), "draft_n": dn, "draft_n_accepted": da,
                       "acceptance": round(da / dn, 3) if dn else None}
        return res
    finally:
        proc.terminate(); proc.wait(timeout=30); log.close()


os.makedirs(OUT, exist_ok=True)
results = {"binary": BIN}
results["nomtp"] = run("nomtp", [])
print("nomtp", json.dumps(results["nomtp"]))
results["mtp3"] = run("mtp3", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"])
print("mtp3", json.dumps(results["mtp3"]))
json.dump(results, open(os.path.join(OUT, "ref-llama-server-v0.5.0.json"), "w"), indent=1)
