#!/usr/bin/env python3
"""MTP draft-policy sweep: fixed depths vs adaptive, single stream, greedy, EOS banned.

    KCPP_TEST_MODEL=main.gguf [KCPP_TEST_DRAFT=mtp-head.gguf] python3 tests/bench_mtp_depth.py [n_tokens]

For each config loads the server once (parallel lane, 1 slot in use), runs the same prompts and records
decode tok/s, drafted/accepted counts and mean accepted length. 'off' is the serial control (no MTP).
The h used by --draftmode adaptive should be fitted from the fixed-depth curve on each backend.
Writes bench-mtp-depth.json to KCPP_TEST_OUT.
"""
import json
import os
import sys
import time

os.environ.setdefault("KCPP_TEST_HTTP_TIMEOUT", "1800")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_parallel_runtime as T  # noqa: E402

DRAFT = os.environ.get("KCPP_TEST_DRAFT", "")
PROMPTS = [
    "Write a short poem about mountains.",
    "Explain in detail how a TCP handshake works, step by step.",
    "Continue the sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,",
    "Translate to French: The quick brown fox jumps over the lazy dog. The weather is nice today.",
]


def configs():
    head = (["--draftmodel", DRAFT] if DRAFT else ["--usemtp"])
    out = [("off", [])]
    for d in (1, 2, 3, 4, 6):
        out.append(("fixed%d" % d, head + ["--draftamount", str(d)]))
    for h in os.environ.get("KCPP_BENCH_COSTS", "0.18,0.35").split(","):
        out.append(("adaptive_h%s" % h, head + ["--draftamount", "8", "--draftmode", "adaptive", "--draftcost", h]))
    only = os.environ.get("KCPP_BENCH_ONLY", "")
    return [c for c in out if not only or c[0] in only.split(",")]


def main():
    if not T.MODEL:
        print(__doc__)
        return 1
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    results = {}
    for name, extra in configs():
        srv = T.Server("depth_" + name, ["--parallelrequests", "2", "--noshift", "--multiuser", "8"] + extra, ctx=4096).start(timeout=1800)
        try:
            T.gen(srv, "Hello", 4)
            rows = []
            texts = []
            for p in PROMPTS:
                t0 = time.time()
                st, b = T.gen(srv, p, n, ban_eos_token=True)
                dt = time.time() - t0
                r = b["results"][0]
                rows.append({"tok_s": round(r["completion_tokens"] / dt, 2), "draft": r.get("draft_tokens", 0), "acc": r.get("draft_accepted", 0)})
                texts.append(r["text"])
            tps = [x["tok_s"] for x in rows]
            dr = sum(x["draft"] for x in rows)
            ac = sum(x["acc"] for x in rows)
            results[name] = {"per_prompt": rows, "mean_tok_s": round(sum(tps) / len(tps), 2), "drafted": dr, "accepted": ac,
                             "acceptance": round(ac / dr, 3) if dr else None, "texts": texts}
            print(name, json.dumps({k: results[name][k] for k in ("mean_tok_s", "drafted", "accepted", "acceptance")}), flush=True)
        finally:
            srv.stop()
    if "off" in results:
        for name, r in results.items():
            r["speedup_vs_off"] = round(r["mean_tok_s"] / results["off"]["mean_tok_s"], 3)
            r["parity_vs_off"] = r["texts"] == results["off"]["texts"]
    os.makedirs(T.OUT, exist_ok=True)
    with open(os.path.join(T.OUT, "bench-mtp-depth.json"), "w") as f:
        json.dump(results, f, indent=1)
    for name, r in results.items():
        print("%-18s %6.2f tok/s  x%s  parity=%s  acc=%s" % (name, r["mean_tok_s"], r.get("speedup_vs_off"), r.get("parity_vs_off"), r["acceptance"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
