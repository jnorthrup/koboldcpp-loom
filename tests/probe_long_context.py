#!/usr/bin/env python3
"""Deep passkey probes for long-context profiles (facts, not pass/fail).

    KCPP_TEST_MODEL=/path/model.gguf python3 tests/probe_long_context.py CTX DEPTH[,DEPTH...] [--ropescaling yarn] [extra args]

Loads one server at CTX, places a numeric pass key at 30% depth inside DEPTH tokens of filler, and asks for it.
Records prompt tokens, prefill time and tok/s, and whether the key was recalled. Writes probe-CTX.json to KCPP_TEST_OUT.
The HTTP timeout defaults to 3600 s (KCPP_TEST_HTTP_TIMEOUT) because full-depth prefill is long.
"""
import json
import os
import sys
import time

os.environ.setdefault("KCPP_TEST_HTTP_TIMEOUT", "3600")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_parallel_runtime as T  # noqa: E402


def main():
    if not T.MODEL or len(sys.argv) < 3:
        print(__doc__)
        return 1
    ctx = int(sys.argv[1])
    depths = [int(x) for x in sys.argv[2].split(",") if x]
    extra = ["--quantkv", "q8_0", "--noshift", "--batchsize", "2048"] + sys.argv[3:]
    srv = T.Server("probe_%d" % ctx, extra, ctx=ctx).start(timeout=900)
    out = {"ctx": ctx, "args": extra, "probes": []}
    try:
        out["context"] = srv.runtime()["context"]
        for depth in depths:
            key = 81000 + depth % 997
            prompt = T.passkey_prompt(srv, depth, key)
            t0 = time.time()
            st, body = T.gen(srv, prompt, 8)
            dt = time.time() - t0
            r = body["results"][0] if st == 200 else {}
            txt = r.get("text", str(body)[:300])
            probe = {"target_tokens": depth, "prompt_tokens": r.get("prompt_tokens"), "seconds": round(dt, 1),
                     "prefill_tok_s": round(r["prompt_tokens"] / dt, 1) if st == 200 and dt > 0 else None,
                     "found": str(key) in txt, "text": txt, "status": st}
            out["probes"].append(probe)
            print(json.dumps(probe), flush=True)
    finally:
        srv.stop()
    os.makedirs(T.OUT, exist_ok=True)
    path = os.path.join(T.OUT, "probe-%d.json" % ctx)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("probe:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
