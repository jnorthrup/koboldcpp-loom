#!/usr/bin/env python3
"""Throughput and MTP measurements for the LOOM parallel engine (facts, not pass/fail).

    KCPP_TEST_MODEL=/path/model.gguf python3 tests/bench_parallel_runtime.py [label]

Configurations: serial vs --parallelrequests 4, MTP off/on. For each: prefill time of a
~2000-token prompt, then 1 and 4 concurrent greedy generations of GEN_N tokens (EOS banned).
Writes bench-<label>.json into KCPP_TEST_OUT.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_parallel_runtime as T  # noqa: E402

GEN_N = int(os.environ.get("KCPP_BENCH_GEN", "128"))
PROMPTS = ["Write a detailed essay about %s." % s for s in ("rivers", "mountains", "deserts", "forests")]
LONG = "The quick brown fox jumps over the lazy dog. " * 200

CONFIGS = [
    ("serial", ["--parallelrequests", "1", "--noshift"]),
    ("serial_mtp", ["--parallelrequests", "1", "--noshift", "--usemtp", "--draftamount", "3"]),
    ("parallel4", ["--parallelrequests", "4", "--noshift"]),
    ("parallel4_mtp", ["--parallelrequests", "4", "--noshift", "--usemtp", "--draftamount", "3"]),
]


def one(srv, prompt, n):
    t0 = time.time()
    st, body = T.gen(srv, prompt, n, ban_eos_token=True)
    dt = time.time() - t0
    assert st == 200, body
    r = body["results"][0]
    return dt, r


def bench(name, args):
    srv = T.Server("bench_" + name, args + ["--multiuser", "16"]).start()
    out = {"args": args, "load_seconds": srv.load_seconds}
    try:
        one(srv, "Hello", 4)  # warm
        dt, r = one(srv, LONG, 1)
        out["prefill"] = {"prompt_tokens": r["prompt_tokens"], "seconds": round(dt, 3), "tok_s": round(r["prompt_tokens"] / dt, 1)}
        dt, r = one(srv, PROMPTS[0], GEN_N)
        out["gen1"] = {"seconds": round(dt, 3), "tok_s": round(r["completion_tokens"] / dt, 2),
                       "draft_tokens": r.get("draft_tokens"), "draft_accepted": r.get("draft_accepted")}
        t0 = time.time()
        res = T.run_concurrently([lambda p=p: one(srv, p, GEN_N) for p in PROMPTS])
        wall = time.time() - t0
        toks = sum(r["completion_tokens"] for _, r in res)
        dr = sum(r.get("draft_tokens", 0) for _, r in res)
        da = sum(r.get("draft_accepted", 0) for _, r in res)
        out["gen4"] = {"wall_seconds": round(wall, 3), "aggregate_tok_s": round(toks / wall, 2), "tokens": toks,
                       "draft_tokens": dr, "draft_accepted": da,
                       "acceptance": round(da / dr, 3) if dr else None,
                       "per_request_seconds": [round(dt, 2) for dt, _ in res]}
        out["runtime"] = srv.runtime()["parallel"]["totals"] if srv.runtime()["parallel"]["enabled"] else None
    finally:
        srv.stop()
    return out


def main():
    if not T.MODEL:
        print("KCPP_TEST_MODEL is not set")
        return 1
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    only = os.environ.get("KCPP_BENCH_ONLY", "")
    results = {}
    for name, args in CONFIGS:
        if only and name not in only.split(","):
            continue
        results[name] = bench(name, args)
        print(name, json.dumps({k: results[name][k] for k in ("prefill", "gen1", "gen4")}))
    path = os.path.join(T.OUT, "bench-%s.json" % label)
    os.makedirs(T.OUT, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print("bench:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
