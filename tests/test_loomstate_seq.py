#!/usr/bin/env python3
"""LOOMKV01 per-sequence round trip on the parallel lane (kind=seq).

    KCPP_TEST_MODEL=attention-only.gguf [KCPP_TEST_DRAFT=mtp-head.gguf] KCPP_TEST_PYTHON=python3.12 \
        KCPP_TEST_OUT=dir python3 tests/test_loomstate_seq.py

Asserts, with --parallelrequests 2:
  - a finished slot exports as a kind=seq envelope; a fresh process imports it into a slot and
    continues byte-identically while REUSING the imported prefix (reused_tokens grows by most of it)
  - with KCPP_TEST_DRAFT the draft/MTP sequence state travels too (envelope has a draft section) and the
    imported run reproduces text and draft counters
  - a kind=ctx envelope is refused by the seq path and a kind=seq envelope by the ctx path (bad_kind)
  - a different --quantkv is refused track_mismatch with the lane left working
  - a recurrent/hybrid model refuses explicitly (unsupported_model) when KCPP_TEST_EXPECT_UNSUPPORTED=1
Writes evidence-loomstate-seq.json to KCPP_TEST_OUT.
"""
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import test_parallel_runtime as T  # noqa: E402

PW = "loomstate-test"
DRAFT_HEAD = os.environ.get("KCPP_TEST_DRAFT", "")
EXPECT_UNSUPPORTED = os.environ.get("KCPP_TEST_EXPECT_UNSUPPORTED") == "1"
PROMPT = "You are a librarian. " * 40 + "Question: name a poet and describe one of their poems. Answer:"
N1, N2 = 16, 24
fails = []
EVIDENCE = {"draft_head": bool(DRAFT_HEAD)}


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  " + str(detail)))
    if not ok:
        fails.append(name)


def raw(srv, method, path, data=None, ctype="application/json"):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (srv.port, path), data=data, method=method)
    req.add_header("Content-Type", ctype)
    req.add_header("Authorization", "Bearer " + PW)
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def server(name, admindir, extra=()):
    mtp = (["--draftmodel", DRAFT_HEAD, "--draftamount", "2"] if DRAFT_HEAD else [])
    s = T.Server(name, ["--parallelrequests", "2", "--noshift", "--multiuser", "8", "--admin", "--admindir", admindir,
                        "--adminpassword", PW] + mtp + list(extra))
    return s.start(timeout=1800)


def generate(srv, prompt, n):
    payload = {"prompt": prompt, "max_length": n, "temperature": 0, "top_k": 1, "rep_pen": 1.0, "sampler_seed": 7, "ban_eos_token": True}
    st, body = raw(srv, "POST", "/api/v1/generate", json.dumps(payload).encode())
    assert st == 200, body
    return json.loads(body)["results"][0]


def reused(srv):
    st, body = raw(srv, "GET", "/api/extra/runtime")
    return json.loads(body)["parallel"]["totals"]["reused_tokens"]


def export_any(srv):
    last = None
    for slot in range(2):
        st, body = raw(srv, "POST", "/api/admin/export_state?lane=parallel", json.dumps({"slot": slot}).encode())
        if st == 200:
            return slot, body
        last = (st, body)
    return None, last


def main():
    if not T.MODEL:
        print("KCPP_TEST_MODEL is not set; nothing to run")
        return 0
    os.makedirs(T.OUT, exist_ok=True)
    admindir = tempfile.mkdtemp(prefix="loomstate-seq-admin-", dir=T.OUT)
    a = server("loomseq_a", admindir)
    try:
        first = generate(a, PROMPT, N1)
        if EXPECT_UNSUPPORTED:
            st, body = raw(a, "POST", "/api/admin/export_state?lane=parallel", json.dumps({"slot": 0}).encode())
            check("hybrid model refuses seq export explicitly", st == 409 and json.loads(body).get("code") == "unsupported_model", (st, body))
            return finish()
        slot, env = export_any(a)
        check("seq export returns an envelope", slot is not None and env[:8] == b"LOOMKV01", env if slot is None else env[:80])
        if slot is None:
            return finish()
        EVIDENCE["envelope_bytes"] = len(env)
        EVIDENCE["exported_slot"] = slot
        cont_prompt = PROMPT + first["text"]
        want = generate(a, cont_prompt, N2)
    finally:
        a.stop()

    # a ctx-kind envelope (SmartCache, serial lane) comes from a separate process: --parallelrequests excludes --smartcache
    s0 = T.Server("loomseq_s", ["--noshift", "--smartcache", "1", "--admin", "--admindir", admindir, "--adminpassword", PW]).start(timeout=1800)
    try:
        generate(s0, PROMPT, 4)
        raw(s0, "POST", "/api/admin/save_state", json.dumps({"slot": 0}).encode())
        st, ctxenv = raw(s0, "POST", "/api/admin/export_state?lane=serial", json.dumps({"slot": 0}).encode())
        check("serial-lane ctx export works in its own process", st == 200 and ctxenv[:8] == b"LOOMKV01", (st, ctxenv[:80]))
    finally:
        s0.stop()

    import loomstate
    import struct
    dlen = struct.unpack(">I", env[8:12])[0]
    desc = loomstate.cbor_decode(env[12:12 + dlen])
    check("envelope kind is seq", desc.get("kind") == "seq", desc.get("kind"))
    check("draft section present iff a draft head is loaded", ("draft" in [s["name"] for s in desc["sections"]]) == bool(DRAFT_HEAD), desc["sections"])

    b = server("loomseq_b", admindir)
    try:
        r0 = reused(b)
        st, body = raw(b, "POST", "/api/admin/import_state?lane=parallel&slot=0", env, "application/octet-stream")
        check("seq import into fresh process", st == 200 and json.loads(body).get("success"), (st, body))
        got = generate(b, cont_prompt, N2)
        gained = reused(b) - r0
        EVIDENCE["tokens_reused_after_import"] = gained
        EVIDENCE["cached_tokens_in_envelope"] = desc["n_tokens"]
        check("imported slot continues byte-identically", got["text"] == want["text"], {"want": want["text"][:80], "got": got["text"][:80]})
        check("imported prefix was reused, not re-prefilled", gained >= desc["n_tokens"] - 8, {"gained": gained, "cached": desc["n_tokens"]})
        if DRAFT_HEAD:
            check("draft counters reproduce", (got.get("draft_tokens"), got.get("draft_accepted")) == (want.get("draft_tokens"), want.get("draft_accepted")),
                  {"got": [got.get("draft_tokens"), got.get("draft_accepted")], "want": [want.get("draft_tokens"), want.get("draft_accepted")]})
            EVIDENCE["draft"] = [got.get("draft_tokens"), got.get("draft_accepted")]
        st, body = raw(b, "POST", "/api/admin/import_state?lane=parallel&slot=1", ctxenv, "application/octet-stream")
        check("seq path refuses a ctx envelope", st == 422 and json.loads(body).get("code") == "bad_kind", (st, body))
        st, body = raw(b, "POST", "/api/admin/import_state?lane=serial&slot=0&load=1", env, "application/octet-stream")
        check("ctx path refuses a seq envelope", st == 422 and json.loads(body).get("code") == "bad_kind", (st, body))
        bad = bytearray(env)
        bad[-1] ^= 1
        st, body = raw(b, "POST", "/api/admin/import_state?lane=parallel&slot=1", bytes(bad), "application/octet-stream")
        check("corrupted seq envelope rejected", st == 422 and json.loads(body).get("code") == "digest_mismatch", (st, body))
        st, body = raw(b, "POST", "/api/admin/import_state?lane=parallel&slot=9", env, "application/octet-stream")
        check("out-of-range slot rejected", st == 400 and json.loads(body).get("code") == "bad_slot", (st, body))
        check("lane healthy after refusals", len(generate(b, "Hello", 4)["text"]) > 0)
    finally:
        b.stop()

    if not DRAFT_HEAD:
        c = server("loomseq_c", admindir, ["--quantkv", "q8_0"])
        try:
            st, body = raw(c, "POST", "/api/admin/import_state?lane=parallel&slot=0", env, "application/octet-stream")
            doc = json.loads(body)
            check("refuse on quantkv mismatch", st == 422 and doc.get("code") == "track_mismatch", (st, doc))
            check("lane unchanged after refusal", len(generate(c, "Hello", 4)["text"]) > 0)
        finally:
            c.stop()
    return finish()


def finish():
    name = "evidence-loomstate-seq%s.json" % ("-mtp" if DRAFT_HEAD else "")
    with open(os.path.join(T.OUT, name), "w") as f:
        json.dump(EVIDENCE, f, indent=1)
    print("failed:", len(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
