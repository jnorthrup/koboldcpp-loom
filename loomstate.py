"""loomstate: portable, self-describing KV state envelope for koboldcpp-loom (LOOMKV01).

An envelope moves one committed sequence (main KV, optional draft/MTP KV, last logits) between
processes or nodes. It carries nothing the importer has to trust: every identity field is checked
against the importing process before a single byte is handed to llama.cpp, and a mismatch refuses
the import (the caller falls back to replaying the recipe with --prewarm).

Wire layout (all integers big-endian):
    "LOOMKV01"  8 bytes
    u32         descriptor length
    descriptor  canonical CBOR map (RFC 8949 canonical: minimal-width heads, map keys sorted by
                encoded key bytes). Byte-compatible with confix-rs `item::encode`; tests/golden
                vectors in tests/test_loomstate.py are produced by that encoder.
    payload     sections named in descriptor["sections"], concatenated in order
    The descriptor binds the payload through sha256 digests, so signing the descriptor bytes
    (ed25519, done by the mesh, not here) authenticates the whole envelope.

Identity ("track"): model file sha256, quant/KV types, context size, engine pin, backend. Two
processes may exchange state only inside the same track: llama.cpp state dumps are tied to the
exact model, KV type and build, and its kernels are not batch-invariant across builds.
"""
import hashlib
import struct

MAGIC = b"LOOMKV01"
VERSION = 1
# Identity fields that must match exactly for an import to be accepted.
TRACK_FIELDS = ("model_sha256", "kv_type_k", "kv_type_v", "n_ctx", "engine", "backend", "arch")
STATE_CLASSES = ("prefix", "session")  # immutable shared prefix vs private continuation


class LoomStateError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code
        self.msg = msg


# ---------------------------------------------------------------------------------------------
# canonical CBOR (subset: uint/negint, bytes, text, array, map with text keys, bool, null)
# ---------------------------------------------------------------------------------------------
def _head(major, value):
    mt = major << 5
    if value < 24:
        return bytes([mt | value])
    if value <= 0xFF:
        return bytes([mt | 24, value])
    if value <= 0xFFFF:
        return bytes([mt | 25]) + struct.pack(">H", value)
    if value <= 0xFFFFFFFF:
        return bytes([mt | 26]) + struct.pack(">I", value)
    return bytes([mt | 27]) + struct.pack(">Q", value)


def cbor_encode(item):
    if item is None:
        return b"\xf6"
    if item is True:
        return b"\xf5"
    if item is False:
        return b"\xf4"
    if isinstance(item, int):
        if item >= 0:
            return _head(0, item)
        return _head(1, -item - 1)
    if isinstance(item, (bytes, bytearray)):
        return _head(2, len(item)) + bytes(item)
    if isinstance(item, str):
        raw = item.encode("utf-8")
        return _head(3, len(raw)) + raw
    if isinstance(item, (list, tuple)):
        return _head(4, len(item)) + b"".join(cbor_encode(i) for i in item)
    if isinstance(item, dict):
        pairs = sorted(((cbor_encode(str(k)), v) for k, v in item.items()), key=lambda p: p[0])
        return _head(5, len(pairs)) + b"".join(k + cbor_encode(v) for k, v in pairs)
    raise TypeError("cannot CBOR-encode %r" % type(item))


def cbor_decode(data):
    value, end = _decode(memoryview(data), 0)
    if end != len(data):
        raise LoomStateError("bad_cbor", "trailing bytes after CBOR item")
    return value


def _decode(buf, pos):
    if pos >= len(buf):
        raise LoomStateError("bad_cbor", "truncated CBOR")
    ib = buf[pos]
    major, info = ib >> 5, ib & 0x1F
    pos += 1
    if major == 7:
        if info in (20, 21, 22):
            return {20: False, 21: True, 22: None}[info], pos
        raise LoomStateError("bad_cbor", "unsupported CBOR simple/float item")
    if info < 24:
        val = info
    elif info in (24, 25, 26, 27):
        n = 1 << (info - 24)
        if pos + n > len(buf):
            raise LoomStateError("bad_cbor", "truncated CBOR head")
        val = int.from_bytes(buf[pos:pos + n], "big")
        pos += n
    else:
        raise LoomStateError("bad_cbor", "unsupported CBOR item (indefinite length)")
    if major == 0:
        return val, pos
    if major == 1:
        return -1 - val, pos
    if major in (2, 3):
        if pos + val > len(buf):
            raise LoomStateError("bad_cbor", "truncated CBOR string")
        raw = bytes(buf[pos:pos + val])
        pos += val
        return (raw if major == 2 else raw.decode("utf-8")), pos
    if major == 4:
        out = []
        for _ in range(val):
            item, pos = _decode(buf, pos)
            out.append(item)
        return out, pos
    if major == 5:
        out = {}
        for _ in range(val):
            k, pos = _decode(buf, pos)
            v, pos = _decode(buf, pos)
            if not isinstance(k, str):
                raise LoomStateError("bad_cbor", "non-text map key")
            out[k] = v
        return out, pos
    raise LoomStateError("bad_cbor", "unsupported CBOR major type %d" % major)


# ---------------------------------------------------------------------------------------------
# envelope
# ---------------------------------------------------------------------------------------------
def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_envelope(track, tokens, sections, state_class="session", parent_sha256=None):
    """track: dict with TRACK_FIELDS. sections: ordered list of (name, bytes), e.g.
    [("main", ..), ("draft", ..), ("logits", ..)]. Returns the full envelope bytes."""
    missing = [f for f in TRACK_FIELDS if f not in track]
    if missing:
        raise LoomStateError("bad_track", "track is missing %s" % ", ".join(missing))
    if state_class not in STATE_CLASSES:
        raise LoomStateError("bad_class", "unknown state class %r" % state_class)
    desc = {
        "magic": MAGIC.decode(),
        "version": VERSION,
        "class": state_class,
        "track": {f: track[f] for f in TRACK_FIELDS},
        "n_tokens": len(tokens),
        "tokens_sha256": sha256_hex(struct.pack(">%dI" % len(tokens), *[t & 0xFFFFFFFF for t in tokens])),
        "tokens": list(tokens),
        "sections": [{"name": n, "bytes": len(b), "sha256": sha256_hex(b)} for n, b in sections],
        "parent_sha256": parent_sha256,
    }
    dbytes = cbor_encode(desc)
    return MAGIC + struct.pack(">I", len(dbytes)) + dbytes + b"".join(b for _, b in sections)


def parse_envelope(data, local_track, accept_classes=STATE_CLASSES, expected_parent_sha256=None):
    """Verify and split an envelope. Raises LoomStateError(code, msg) on any mismatch; never
    returns partial results. Returns (descriptor, {section_name: bytes}, descriptor_bytes)."""
    if len(data) < 12 or data[:8] != MAGIC:
        raise LoomStateError("bad_magic", "not a LOOMKV01 envelope")
    (dlen,) = struct.unpack(">I", data[8:12])
    if 12 + dlen > len(data):
        raise LoomStateError("truncated", "descriptor length exceeds envelope")
    dbytes = bytes(data[12:12 + dlen])
    desc = cbor_decode(dbytes)
    if not isinstance(desc, dict) or desc.get("magic") != MAGIC.decode():
        raise LoomStateError("bad_magic", "descriptor magic mismatch")
    if cbor_encode(desc) != dbytes:
        raise LoomStateError("not_canonical", "descriptor is not canonical CBOR")
    if desc.get("version") != VERSION:
        raise LoomStateError("bad_version", "unsupported envelope version %r" % desc.get("version"))
    if desc.get("class") not in accept_classes:
        raise LoomStateError("bad_class", "state class %r not accepted here" % desc.get("class"))
    track = desc.get("track") or {}
    for f in TRACK_FIELDS:
        if track.get(f) != local_track.get(f):
            raise LoomStateError("track_mismatch", "%s differs: envelope %r, local %r" % (f, track.get(f), local_track.get(f)))
    if expected_parent_sha256 is not None and desc.get("parent_sha256") != expected_parent_sha256:
        raise LoomStateError("parent_mismatch", "required parent state digest differs")
    tokens = desc.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != desc.get("n_tokens"):
        raise LoomStateError("bad_tokens", "token list does not match n_tokens")
    if sha256_hex(struct.pack(">%dI" % len(tokens), *[t & 0xFFFFFFFF for t in tokens])) != desc.get("tokens_sha256"):
        raise LoomStateError("bad_tokens", "token digest mismatch")
    off = 12 + dlen
    out = {}
    for sec in desc.get("sections") or []:
        n = sec["bytes"]
        if off + n > len(data):
            raise LoomStateError("truncated", "section %s runs past envelope end" % sec["name"])
        blob = bytes(data[off:off + n])
        if sha256_hex(blob) != sec["sha256"]:
            raise LoomStateError("digest_mismatch", "section %s digest mismatch" % sec["name"])
        out[sec["name"]] = blob
        off += n
    if off != len(data):
        raise LoomStateError("trailing", "bytes after the last section")
    return desc, out, dbytes
