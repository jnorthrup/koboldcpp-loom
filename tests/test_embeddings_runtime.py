#!/usr/bin/env python3
"""Embeddings endpoint check (the embeddings adapter's batch path moved to common_batch).

    KCPP_TEST_MODEL=/path/model.gguf python3 tests/test_embeddings_runtime.py

Loads the model as --embeddingsmodel, embeds four strings and checks: dimension > 0, unit norm
(the adapter L2-normalizes), determinism, and that near-paraphrases are closer than unrelated text.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_parallel_runtime as T  # noqa: E402


class EmbServer(T.Server):
    def command(self):
        return [T.PYEXE, os.path.join(T.ROOT, "koboldcpp.py"), "--embeddingsmodel", T.MODEL, "--host", "127.0.0.1",
                "--port", str(self.port), "--gpulayers", "999", "--skiplauncher"] + self.args


def embed(srv, text):
    st, body = srv.post("/v1/embeddings", {"input": text})
    assert st == 200, body
    return body["data"][0]["embedding"]


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main():
    if not T.MODEL:
        print("KCPP_TEST_MODEL is not set")
        return 1
    srv = EmbServer("embeddings", []).start()
    try:
        a = embed(srv, "The cat sat on the mat.")
        a2 = embed(srv, "The cat sat on the mat.")
        b = embed(srv, "A cat was sitting on a mat.")
        c = embed(srv, "Quarterly revenue rose sharply in the semiconductor sector.")
        norm = math.sqrt(sum(x * x for x in a))
        res = {"dim": len(a), "norm": round(norm, 4), "deterministic": a == a2,
               "cos_paraphrase": round(cos(a, b), 4), "cos_unrelated": round(cos(a, c), 4)}
        print(json.dumps(res))
        ok = len(a) > 0 and abs(norm - 1.0) < 1e-3 and a == a2 and res["cos_paraphrase"] > res["cos_unrelated"]
        print("PASS embeddings" if ok else "FAIL embeddings")
        return 0 if ok else 1
    finally:
        srv.stop()


if __name__ == "__main__":
    sys.exit(main())
