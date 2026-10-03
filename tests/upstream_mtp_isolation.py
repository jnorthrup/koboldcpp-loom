#!/usr/bin/env python3
"""Upstream llama-server: is MTP output batch-invariant? solo x2 vs concurrent pair.

    python3 tests/upstream_mtp_isolation.py /path/llama-server MAIN.gguf MTP_HEAD.gguf OUT_DIR

Compares greedy text of one prompt run alone (twice) and alongside a second prompt, with and
without the MTP head. Mirrors the kobold check so the two implementations can be compared.
"""
import json, os, socket, subprocess, sys, threading, time, urllib.request

BIN, MODEL, HEAD, OUT = sys.argv[1:5]
P1 = "Write a short poem about mountains."
P2 = "Write a short poem about rivers."


def port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def comp(p, prompt):
    body = {"prompt": prompt, "n_predict": 64, "temperature": 0, "ignore_eos": True, "cache_prompt": False}
    req = urllib.request.Request("http://127.0.0.1:%d/completion" % p, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())


def run(name, extra):
    p = port()
    log = open(os.path.join(OUT, "up_iso_%s.log" % name), "w")
    proc = subprocess.Popen([BIN, "-m", MODEL, "--host", "127.0.0.1", "--port", str(p), "-c", "8192", "-np", "2", "-ngl", "999",
                             "--no-webui", "-ctk", "q8_0", "-ctv", "q8_0"] + extra, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(1200):
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/health" % p, timeout=2); break
            except Exception:
                time.sleep(0.5)
        a = comp(p, P1)["content"]
        b = comp(p, P1)["content"]
        res = [None, None]
        ts = [threading.Thread(target=lambda i=i, q=q: res.__setitem__(i, comp(p, q))) for i, q in enumerate((P1, P2))]
        [t.start() for t in ts]; [t.join() for t in ts]
        c = res[0]["content"]
        i = next((k for k in range(min(len(a), len(c))) if a[k] != c[k]), None)
        t = res[0].get("timings", {})
        return {"solo_repeat_same": a == b, "solo_vs_pair_same": a == c, "first_diff_char": i, "len": len(a),
                "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted")}
    finally:
        proc.terminate(); proc.wait(timeout=60); log.close()


EXTRA = os.environ.get("UPSTREAM_EXTRA", "").split()  # e.g. "-kvu" to force a unified KV cache
ONLY = os.environ.get("UPSTREAM_ONLY", "")  # "mtp" or "nomtp"
os.makedirs(OUT, exist_ok=True)
out = {"extra": EXTRA}
if ONLY in ("", "mtp"):
    out["mtp"] = run("mtp" + "".join(EXTRA), ["-md", HEAD, "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"] + EXTRA)
if ONLY in ("", "nomtp"):
    out["nomtp"] = run("nomtp" + "".join(EXTRA), EXTRA)
print(json.dumps(out))
json.dump(out, open(os.path.join(OUT, "upstream-mtp-isolation%s.json" % "".join(EXTRA)), "w"), indent=1)
