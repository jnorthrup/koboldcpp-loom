#!/usr/bin/env python3
"""Upstream llama-server MTP parity probe: greedy text with MTP at depth D vs no MTP, solo, same prompt.

    python3 tests/upstream_mtp_parity.py /path/llama-server MAIN.gguf MTP_HEAD.gguf OUT_DIR [depths=1,2,3,4] [kvu=0|1]
"""
import json, os, socket, subprocess, sys, time, urllib.request

BIN, MODEL, HEAD, OUT = sys.argv[1:5]
DEPTHS = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "1,2,3,4").split(",")]
KVU = (sys.argv[6] if len(sys.argv) > 6 else "0") == "1"
PROMPT = "Write a short poem about mountains."
N = int(os.environ.get("UPSTREAM_N", "64"))


def port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def run(name, extra):
    p = port()
    log = open(os.path.join(OUT, "up_parity_%s.log" % name), "w")
    args = [BIN, "-m", MODEL, "--host", "127.0.0.1", "--port", str(p), "-c", "8192", "-np", "2", "-ngl", "999", "--no-webui"] + (["-kvu"] if KVU else []) + extra
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(1200):
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/health" % p, timeout=2); break
            except Exception:
                time.sleep(0.5)
        body = {"prompt": PROMPT, "n_predict": N, "temperature": 0, "ignore_eos": True, "cache_prompt": False}
        req = urllib.request.Request("http://127.0.0.1:%d/completion" % p, data=json.dumps(body).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        t = r.get("timings", {})
        return {"text": r["content"], "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
                "tok_s": t.get("predicted_per_second")}
    finally:
        proc.terminate(); proc.wait(timeout=60); log.close()


os.makedirs(OUT, exist_ok=True)
ref = run("off", [])
out = {"kvu": KVU, "off": {k: ref[k] for k in ("tok_s",)}}
for d in DEPTHS:
    r = run("d%d" % d, ["-md", HEAD, "--spec-type", "draft-mtp", "--spec-draft-n-max", str(d)])
    i = next((k for k in range(min(len(r["text"]), len(ref["text"]))) if r["text"][k] != ref["text"][k]), None)
    out["d%d" % d] = {"parity": r["text"] == ref["text"], "first_diff": i, "draft_n": r["draft_n"], "draft_n_accepted": r["draft_n_accepted"], "tok_s": r["tok_s"]}
    print("d%d" % d, json.dumps(out["d%d" % d]), flush=True)
json.dump(out, open(os.path.join(OUT, "upstream-mtp-parity%s.json" % ("-kvu" if KVU else "")), "w"), indent=1)
