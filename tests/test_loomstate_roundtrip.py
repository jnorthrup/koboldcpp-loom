#!/usr/bin/env python3
"""LOOMKV01 engine round trip: export a committed slot from one process, import it into a fresh one.

    KCPP_TEST_MODEL=model.gguf KCPP_TEST_PYTHON=python3.12 KCPP_TEST_OUT=dir python3 tests/test_loomstate_roundtrip.py

Asserts:
  - the imported process continues byte-identically to the exporting process (greedy, same prompt)
  - refuse-on-mismatch at the HTTP edge: different --contextsize and different --quantkv are 422
    track_mismatch and leave the importing engine untouched (it still generates normally)
  - a corrupted or non-envelope body is rejected before it reaches the engine
Writes evidence-loomstate.json to KCPP_TEST_OUT.
"""
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_parallel_runtime as T  # noqa: E402

PW = "loomstate-test"
PROMPT = "The quick brown fox jumps over the lazy dog. " * 12 + "Continue the story: "
N1, N2 = 24, 32
fails = []
EVIDENCE = {}


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  " + str(detail)))
    if not ok:
        fails.append(name)


def raw(srv, method, path, data=None, ctype="application/json"):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (srv.port, path), data=data, method=method)
    req.add_header("Content-Type", ctype)
    req.add_header("Authorization", "Bearer " + PW)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def server(name, admindir, extra=(), ctx=T.CTX):
    s = T.Server(name, ["--noshift", "--smartcache", "2", "--admin", "--admindir", admindir, "--adminpassword", PW] + list(extra), ctx=ctx)
    return s.start(timeout=900)


def generate(srv, prompt, n):
    payload = {"prompt": prompt, "max_length": n, "temperature": 0, "top_k": 1, "rep_pen": 1.0, "sampler_seed": 7, "ban_eos_token": True}
    st, body = raw(srv, "POST", "/api/v1/generate", json.dumps(payload).encode())
    assert st == 200, body
    return json.loads(body)["results"][0]["text"]


def main():
    if not T.MODEL:
        print("KCPP_TEST_MODEL is not set; nothing to run")
        return 0
    os.makedirs(T.OUT, exist_ok=True)
    admindir = tempfile.mkdtemp(prefix="loomstate-admin-", dir=T.OUT)
    a = server("loomstate_a", admindir)
    try:
        first = generate(a, PROMPT, N1)
        st, body = raw(a, "POST", "/api/admin/save_state", json.dumps({"slot": 0}).encode())
        check("exporter saved a slot", st == 200 and json.loads(body).get("success"), body)
        st, env = raw(a, "POST", "/api/admin/export_state", json.dumps({"slot": 0}).encode())
        check("export_state returns an envelope", st == 200 and env[:8] == b"LOOMKV01", (st, env[:200]))
        EVIDENCE["envelope_bytes"] = len(env)
        cont_prompt = PROMPT + first
        want = generate(a, cont_prompt, N2)
    finally:
        a.stop()

    check("export is non-trivial", len(env) > 1024, len(env))

    b = server("loomstate_b", admindir)
    try:
        st, body = raw(b, "POST", "/api/admin/import_state?slot=0&load=1", env, "application/octet-stream")
        check("import into fresh process", st == 200 and json.loads(body).get("loaded") is True, body)
        got = generate(b, cont_prompt, N2)
        check("imported process continues byte-identically", got == want, {"want": want[:80], "got": got[:80]})
        check("importer reused the state instead of re-prefilling", "Restored KV" in b.log_text(), "no 'Restored KV' line in the importer log")
        EVIDENCE["continuation_chars"] = len(got)

        bad = bytearray(env)
        bad[-1] ^= 1
        st, body = raw(b, "POST", "/api/admin/import_state?slot=1", bytes(bad), "application/octet-stream")
        check("corrupted envelope rejected", st == 422 and json.loads(body).get("code") == "digest_mismatch", (st, body))
        st, body = raw(b, "POST", "/api/admin/import_state?slot=1", b"garbage", "application/octet-stream")
        check("non-envelope rejected", st == 422 and json.loads(body).get("code") == "bad_magic", (st, body))
        check("engine still healthy after refusals", len(generate(b, "Hello", 4)) > 0)
    finally:
        b.stop()

    for label, extra, ctx in (("contextsize", [], T.CTX * 2), ("quantkv", ["--quantkv", "q8_0"], T.CTX)):
        c = server("loomstate_c_" + label, admindir, extra, ctx)
        try:
            st, body = raw(c, "POST", "/api/admin/import_state?slot=0&load=1", env, "application/octet-stream")
            doc = json.loads(body)
            check("refuse on %s mismatch" % label, st == 422 and doc.get("code") == "track_mismatch", (st, doc))
            EVIDENCE["refused_" + label] = doc.get("error")
            check("engine unchanged after %s refusal" % label, len(generate(c, "Hello", 4)) > 0)
        finally:
            c.stop()

    with open(os.path.join(T.OUT, "evidence-loomstate.json"), "w") as f:
        json.dump(EVIDENCE, f, indent=1)
    print("failed:", len(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
