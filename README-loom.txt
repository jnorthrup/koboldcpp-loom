koboldcpp-loom: loom features
=============================

--loomcache
  SmartCache slots become branches of a prompt prefix tree found from the tokens.
  A request resumes from the deepest branch it shares; the branch it leaves is kept;
  on recurrent/hybrid models the point where siblings split is snapshotted. No hints.
  Implies --smartcache (slot count via --smartcache N). Note: SmartCache disables
  kobold's parallel batching, so --loomcache and --parallelrequests exclude each other.

--loomdir DIR   (/v1/looms, loomstore.py)
  Lifecycle and artifact storage for loom scripts, the server mate of the Hermes
  plugin's loop precook (github.com/jnorthrup/loom-hermes-plugin, precook/).
    POST   /v1/looms                          store a script (content-addressed id)
    GET    /v1/looms, /v1/looms/{id}          list / read
    DELETE /v1/looms/{id}                     delete script and runs
    POST   /v1/looms/{id}/runs                run {prewarm?, request?, replay_of?}
    GET    /v1/looms/{id}/runs[/{run}]        run status, leaves, usage
    GET    /v1/looms/{id}/runs/{run}/artifact loom-run/1 JSONL, one request+response per leaf
    POST   /v1/looms/{id}/runs/{run}/cancel
  A run walks the outline depth-first as nested loops and sends each leaf through this
  server's own /v1/chat/completions over loopback, so it takes the normal queue and
  --loomcache path. Storage is scoped per bearer key (hash of the key). Runs in flight
  at shutdown are marked interrupted on restart. Frame rendering is loom-frames/1,
  byte-identical to the plugin's loomframes.py; change both together.

Persistent state: the continuum from "a user's KV cache" to "a GPU's token dump"
  1 recipe            loom script                    done (/v1/looms)
  2 run artifact      requests + responses           done (loom-run/1, replay)
  3 warm process      KV slots in RAM                done (--loomcache; lost on idle-out)
  4 state file        IN PROGRESS (operator reversal, 2026-10-03): portable LOOMKV01 envelope,
  5 shared dump       loomstate.py. One committed sequence (main KV, draft KV, logits, tokens) with a
                      canonical-CBOR descriptor (byte-identical to confix-rs) and sha256 section
                      digests. Import refuses unless model hash, KV types, n_ctx, engine pin, backend
                      and arch all match; on refusal the caller replays the recipe with --prewarm.
                      Size is not GB-scale on hybrid models (Qwen3.8-27B: ~290 MiB per 4K-token q8_0
                      branch; only 16 of 64 layers keep KV) - dense models are ~4x larger.
                      Done: envelope + refuse-on-mismatch (tests/test_loomstate.py); engine
                      endpoints POST /api/admin/export_state {slot} and
                      POST /api/admin/import_state?slot=N&load=0|1[&parent=sha256] (admin auth, idle
                      engine, needs --smartcache); cross-process round trip continues byte-identically
                      (tests/test_loomstate_roundtrip.py).
                      Limits: SmartCache slot = whole-context snapshot, serial lane only; media-bearing
                      slots are not exportable; the track hashes the model file and the engine library.
                      Not done: per-sequence export for parallel slots, mesh signing, GCS placement.
  Hosting/billing proposal: runpod experiments/loom/docs/looms-hosting-proposal.md.
