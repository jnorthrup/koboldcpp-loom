#!/usr/bin/env python3
"""loomstate envelope tests (stdlib only; no model needed).

    python3 tests/test_loomstate.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import loomstate as L  # noqa: E402

# Produced by confix-rs `item::encode` (crates/confix-rs, cocaine-rats) for GOLDEN_ITEM below.
GOLDEN_HEX = ("a661616668c3a96c6c6f626161a2626b3101626b32026361727288001718181a000100001b0000000100000000"
              "f5f4f66362696e430001ff636e65673903e7647a65746119012c")
GOLDEN_ITEM = {
    "zeta": 300, "a": "h\u00e9llo", "neg": -1000, "bin": b"\x00\x01\xff",
    "arr": [0, 23, 24, 65536, 4294967296, True, False, None],
    "aa": {"k2": 2, "k1": 1},
}
TRACK = {"model_sha256": "ab" * 32, "kv_type_k": "f16", "kv_type_v": "f16", "n_ctx": 8192,
         "engine": "koboldcpp-loom@test", "backend": "metal", "arch": "qwen35"}
TOKENS = [1, 2, 3, 151643, 0, 7]
SECTIONS = [("main", b"M" * 100), ("draft", b"D" * 10), ("logits", b"L" * 8)]
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  " + str(detail)))
    if not ok:
        fails.append(name)


def refused(code, fn):
    try:
        fn()
    except L.LoomStateError as e:
        return e.code == code, e.code
    return False, "no error"


def main():
    check("canonical CBOR matches confix-rs bytes", L.cbor_encode(GOLDEN_ITEM).hex() == GOLDEN_HEX)
    check("CBOR round trip", L.cbor_decode(bytes.fromhex(GOLDEN_HEX)) == GOLDEN_ITEM)

    env = L.build_envelope(TRACK, TOKENS, SECTIONS)
    desc, secs, dbytes = L.parse_envelope(env, TRACK)
    check("round trip tokens", desc["tokens"] == TOKENS)
    check("round trip sections", secs == dict(SECTIONS))
    check("descriptor is canonical", L.cbor_encode(L.cbor_decode(dbytes)) == dbytes)
    check("build is deterministic", L.build_envelope(TRACK, TOKENS, SECTIONS) == env)

    # refuse-on-mismatch: every track field
    for f, other in (("model_sha256", "cd" * 32), ("kv_type_k", "q8_0"), ("kv_type_v", "q8_0"), ("n_ctx", 4096),
                     ("engine", "other"), ("backend", "cuda"), ("arch", "llama")):
        local = dict(TRACK, **{f: other})
        ok, got = refused("track_mismatch", lambda: L.parse_envelope(env, local))
        check("refuse on %s mismatch" % f, ok, got)

    # integrity
    ok, got = refused("bad_magic", lambda: L.parse_envelope(b"NOPE" + env[4:], TRACK))
    check("refuse bad magic", ok, got)
    ok, got = refused("digest_mismatch", lambda: L.parse_envelope(env[:-1] + bytes([env[-1] ^ 1]), TRACK))
    check("refuse flipped payload byte", ok, got)
    ok, got = refused("truncated", lambda: L.parse_envelope(env[:-3], TRACK))
    check("refuse truncated payload", ok, got)
    ok, got = refused("trailing", lambda: L.parse_envelope(env + b"x", TRACK))
    check("refuse trailing bytes", ok, got)
    # tampering with a token breaks the token digest (descriptor re-encoded canonically so only the digest catches it)
    d = L.cbor_decode(env[12:12 + int.from_bytes(env[8:12], "big")])
    d["tokens"][0] ^= 1
    nd = L.cbor_encode(d)
    forged = L.MAGIC + len(nd).to_bytes(4, "big") + nd + b"".join(b for _, b in SECTIONS)
    ok, got = refused("bad_tokens", lambda: L.parse_envelope(forged, TRACK))
    check("refuse forged token list", ok, got)
    # non-canonical descriptor (reordered map) is rejected so the signed bytes are unique
    dd = L.cbor_decode(env[12:12 + int.from_bytes(env[8:12], "big")])
    items = sorted(((L.cbor_encode(k), L.cbor_encode(v)) for k, v in dd.items()), reverse=True)
    nc = L._head(5, len(items)) + b"".join(k + v for k, v in items)
    ncenv = L.MAGIC + len(nc).to_bytes(4, "big") + nc + b"".join(b for _, b in SECTIONS)
    ok, got = refused("not_canonical", lambda: L.parse_envelope(ncenv, TRACK))
    check("refuse non-canonical descriptor", ok, got)

    # state classes and parent binding
    pre = L.build_envelope(TRACK, TOKENS, SECTIONS, state_class="prefix")
    ok, got = refused("bad_class", lambda: L.parse_envelope(pre, TRACK, accept_classes=("session",)))
    check("refuse state class not accepted", ok, got)
    sess = L.build_envelope(TRACK, TOKENS, SECTIONS, parent_sha256="11" * 32)
    ok, got = refused("parent_mismatch", lambda: L.parse_envelope(sess, TRACK, expected_parent_sha256="22" * 32))
    check("refuse wrong parent digest", ok, got)
    L.parse_envelope(sess, TRACK, expected_parent_sha256="11" * 32)
    check("accept matching parent digest", True)

    print("failed:", len(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
