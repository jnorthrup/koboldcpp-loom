#!/usr/bin/env python3
"""Large-model smoke + MTP check with a split MTP head (e.g. Qwen3.8-27B + mtp-*.gguf).

    KCPP_TEST_MODEL=main.gguf KCPP_TEST_DRAFT=mtp-head.gguf python3 tests/smoke_large_mtp.py [ctx] [slots]

Starts --parallelrequests SLOTS with --draftmodel <MTP head>, then records: runtime MTP state,
one greedy generation (text, tokens, draft counters, tok/s), SLOTS concurrent generations, and a
chat input_tokens/usage consistency check. Facts are written to smoke-large.json in KCPP_TEST_OUT.
"""
import json
import os
import sys
import time

os.environ.setdefault("KCPP_TEST_HTTP_TIMEOUT", "1800")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_parallel_runtime as T  # noqa: E402

DRAFT = os.environ.get("KCPP_TEST_DRAFT", "")


def main():
    if not T.MODEL:
        print(__doc__)
        return 1
    ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    slots = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    args = ["--parallelrequests", str(slots), "--noshift", "--multiuser", "16", "--quantkv", "q8_0", "--draftamount", "2"]
    if DRAFT:
        args += ["--draftmodel", DRAFT]
    else:
        args += ["--usemtp"]
    srv = T.Server("smoke_large", args, ctx=ctx).start(timeout=1800)
    out = {"model": T.MODEL, "draft": DRAFT, "ctx": ctx, "slots": slots, "load_seconds": srv.load_seconds}
    try:
        rt = srv.runtime()
        out["mtp"] = rt["mtp"]
        out["context"] = rt["context"]
        t0 = time.time()
        st, body = T.gen(srv, "Explain in two sentences why the sky is blue.", 96, ban_eos_token=True)
        dt = time.time() - t0
        r = body["results"][0] if st == 200 else body
        out["gen1"] = {"status": st, "seconds": round(dt, 2), "tok_s": round(r["completion_tokens"] / dt, 2) if st == 200 else None,
                       "completion_tokens": r.get("completion_tokens"), "draft_tokens": r.get("draft_tokens"),
                       "draft_accepted": r.get("draft_accepted"), "text": r.get("text")}
        print("gen1", json.dumps(out["gen1"]), flush=True)
        prompts = ["Write a short poem about %s." % s for s in ("rivers", "mountains", "deserts", "forests")][:slots]
        t0 = time.time()
        res = T.run_concurrently([lambda p=p: T.gen(srv, p, 64, ban_eos_token=True) for p in prompts])
        wall = time.time() - t0
        toks = sum(b["results"][0]["completion_tokens"] for s, b in res if s == 200)
        dr = sum(b["results"][0].get("draft_tokens", 0) for s, b in res if s == 200)
        da = sum(b["results"][0].get("draft_accepted", 0) for s, b in res if s == 200)
        out["genN"] = {"status": [s for s, _ in res], "wall_seconds": round(wall, 2), "aggregate_tok_s": round(toks / wall, 2),
                       "draft_tokens": dr, "draft_accepted": da, "acceptance": round(da / dr, 3) if dr else None}
        print("genN", json.dumps(out["genN"]), flush=True)
        msgs = [{"role": "user", "content": "Name one primary color."}]
        st, cnt = srv.post("/v1/chat/completions/input_tokens", {"messages": msgs})
        st2, ch = srv.post("/v1/chat/completions", {"messages": msgs, "max_tokens": 16, "temperature": 0})
        out["admission"] = {"input_tokens": cnt.get("input_tokens"), "usage_prompt_tokens": ch.get("usage", {}).get("prompt_tokens"),
                            "match": st == 200 and st2 == 200 and cnt.get("input_tokens") == ch.get("usage", {}).get("prompt_tokens")}
        print("admission", json.dumps(out["admission"]), flush=True)
        out["totals"] = srv.runtime()["parallel"]["totals"]
    finally:
        srv.stop()
        os.makedirs(T.OUT, exist_ok=True)
        with open(os.path.join(T.OUT, "smoke-large.json"), "w") as f:
            json.dump(out, f, indent=1, default=str)
    print("mtp", json.dumps(out.get("mtp")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
