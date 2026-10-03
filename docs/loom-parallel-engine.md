# LOOM parallel engine (`--parallelrequests`) on llama.cpp v0.5.0

Branch `loom-parallel-v0.5` of koboldcpp-loom. This document records what changed, why, how it was
verified, and what is still open. It is the reference for the runtime behaviour; application
(cocaine-rats) integration notes are at the end and are **not** implemented here.

## Revisions

| Item | Revision |
| --- | --- |
| Fork base | `loom-tier-cache` `86beac6bb` |
| Merge of upstream KoboldCpp | tag `v1.122.1` = `4959b8d3695cf740c1e2d508bedabdea70cbbd44` (2026-09-26), merge commit `c6ec02937b9118c21399173fe3e312b673c520bc` |
| llama.cpp | `v0.5.0` = `7fe450e19305b828c199d602c23a8337aaa1f03b` (annotated tag object `c13fcbf6…`) is an ancestor of the branch. The merged llama.cpp tree is `53ed051ce5e8193652e449f43216ca3859454f49` (v0.5.0 + 11 upstream commits, 2026-09-24), as carried by Kobold v1.122.1 |
| Upstream Kobold reference | `concedo_experimental` `fa641fcb` (2026-10-03): its batching block equals v1.122.1 except for the newer `common_batch`/`llama_process` API, so v1.122.1 is the behavioural reference |

The fork was 82 commits behind its own `origin/upstream` mirror and did not contain `7fe450e`.
Merging Kobold v1.122.1 moves the embedded llama.cpp forward to v0.5.0+11 in one step, with Kobold's
own adaptations already applied. Merging `7fe450e` directly into the Kobold tree conflicts on 177
modify/delete paths plus `CMakeLists.txt`, `ggml-cpu/repack.*` and `ggml-vulkan.cpp`.

## Upstream KoboldCpp `--parallelrequests` as found (v1.114.1 – v1.122.1)

Sources: release notes v1.114.1 ("only supports text gen requests with basic samplers, no multimodal,
no special stateful samplers (e.g. antislop, grammar); cannot be used together with context shifting
or fast forwarding"), v1.117.1 ("Prevent MTP drafting for parallel requests"), and the code.

- Ineligible requests silently fall back to the serial path (`_batch_fallback`).
  - "Ineligible" covers media, grammar, DRY, mirostat, XTC, top-a, TFS, dynatemp, nsigma, smoothing, adaptive-p, tools and non-default sampler order.
- Context shifting is turned off automatically with a warning.
- MTP or a draft model disables batching for every request.
- When `prompt + max_length > n_ctx`, the batch path truncates the prompt front instead of rejecting.
- The unified KV uses `n_seq_max = slots + 1`, with no per-request capacity reservation.
  - Concurrent long requests can exhaust the shared pool mid-generation, and `llama_decode` then fails for the whole batch.
- The EOS token is counted in `completion_tokens`.
- `/slots` returns 501. Abort and check by `genkey` only work for the serial request. Logprobs are global.

## What the native adaptation does

The experimental batch block in `gpttype_adapter.cpp` is replaced by a slot scheduler modelled on
llama.cpp v0.5.0 `tools/server/server-context.cpp`, adapted to Kobold's adapter and sampler stack.

### Lanes

- The **serial lane** is the original `gpttype_generate`, unchanged on sequence 0.
  - It keeps every stateful feature: context shift, fast forward, smartcache, multimodal, CFG, retained grammar, antislop and draft models.
- The **parallel lane** uses sequences `1..N` of the same unified KV pool.
- The two lanes never run concurrently (`BatchLegacyGuard`). Parallel admission is FIFO and resumes after a serial request finishes.

### Admission (exact, rendered, reservation-based)

1. HTTP handlers render the request exactly as for generation. This covers the chat template (Jinja or adapter), tools, memory and the GLM/Gemma prompt adjustments.
2. The engine tokenizes the result with the same `TokenizeString`/BOS/memory path the serial lane uses.
3. Admission requires `prompt_tokens + max_length <= min(n_ctx_seq, --contextsize, request max_context_length)`.
4. If that fails, the request is rejected **before any response bytes are sent**:
   ```
   HTTP 400 {"error":{"type":"exceed_context_size_error","n_prompt_tokens":…,"n_ctx":…,"message":…}}
   ```
   This matches the llama-server error type. Nothing is truncated, and `max_length` is not silently shrunk.
5. Admitted requests reserve `prompt + max_length` KV cells. A request is scheduled only when the pool can hold every live reservation.
   - When it can't, idle prefix caches are evicted LRU first.
   - The head of the queue is never overtaken.
   - Mid-generation KV exhaustion therefore cannot happen by construction.
6. `POST /v1/chat/completions/input_tokens`, `/v1/responses/input_tokens` and `/v1/messages/count_tokens` return the exact engine count, following the upstream `input_tokens` contract.
   - `POST /api/extra/admission` additionally returns `reserve`, `n_ctx` and `admissible`.

### Per-sequence state

Each request owns:

- Its KV sequence.
- Its sampler state: seed/RNG, rep-pen and presence history, DRY breakers and history, grammar, mirostat μ, adaptive-p EMA, reasoning budget and logprob history.
- Its output, stop and EOS handling (same rules as the serial lane).
- Its MTP draft state (`common_speculative` is created with one state per sequence).

The full Kobold sampler set is supported per request, including DRY, XTC, mirostat, grammar, logit bias, token bans, sampler order, dynatemp and smoothing. These are no longer reasons to fall back.

### Prefix reuse ("fast forward")

- Applies when the target memory supports partial `seq_rm` (pure-attention models).
- A finished slot keeps its tokens. The next request reuses the longest cached prefix and always re-evaluates at least the last prompt token.
- Recurrent/hybrid models (e.g. Qwen3.5) cannot truncate a sequence, so reuse is disabled for them. The slot is cleared instead.

### Rejected combinations

**At startup** (`exit_with_error`):

- Parallel with `--smartcontext`, `--smartcache` or `--loomcache` (whole-context snapshots).
- Parallel with `--draftmodel` (parallel drafting supports built-in MTP only).
- Parallel without `--noshift` (shifting rewrites one shared sequence; overflow is rejected instead).
- Parallel with `--usemtp` where MTP cannot be activated.
- Parallel with MTP on a model whose memory cannot roll back a single sequence. The required support is `part`, or `rs` with `n_rs_seq >= draft`.
- `--ropescaling` together with `--overridenativecontext`.

**Per request** (HTTP 400 `not_supported_error`), unless `--parallelserial` routes them to the serial lane:

- Image or audio input.
- CFG `negative_prompt`.
- `grammar_retain_state`.
- Phrase bans (antislop backtracking).

Interrogate (`/sdapi/v1/interrogate`) always uses the serial lane. Image generation, TTS, transcription, embeddings and music wait for running text batches, as before.

### MTP

- Follows the llama.cpp v0.5.0 server order:
  1. Draft for every generating sequence.
  2. Verify each pending token plus its drafts in one target decode.
  3. Call `common_speculative_process` after every target decode.
  4. Truncate the draft cache for re-decoded positions before the decode.
  5. Sample sequentially per request.
  6. Accept the matching prefix.
  7. `seq_rm` the rejected tail in both contexts.
  8. Call `common_speculative_accept(seq, k)`.
- Hybrid models use recurrent rollback snapshots (`n_rs_seq = draft`).
- Prompt positions request outputs so the MTP head sees hidden states for the whole prompt.
- Active state is reported at load time and in `GET /api/extra/runtime` under `mtp`: requested, active, speculative type, draft max, lane coverage, checkpoint vs RS rollback, and the reason when inactive.

### Counters

- `completion_tokens` counts emitted tokens, including an EOS that ends generation. This matches the serial lane.
- `draft_tokens` counts drafted tokens sent for verification; `draft_accepted` counts drafted tokens accepted and emitted.
- Both are reported:
  - per request in `generation_outputs`
  - as `usage.completion_tokens_details.{accepted,rejected}_prediction_tokens` (OpenAI)
  - as `timings.draft_n`/`draft_n_accepted` (llama-server)
  - per slot in `/slots`
  - as totals in `/api/extra/runtime`
- The serial lane reports the same pair through `/api/extra/perf` `last_draft_total`/`last_draft_success`.

### Cancellation

- Each parallel request is bound to its `genkey`.
- `/api/extra/abort` and `/api/extra/generate/check` resolve the key to that request only.
- Client disconnects abort only their own request:
  - A waiting request leaves the queue.
  - A live request is reaped by the worker before the next decode. Its slot and KV sequence are cleared and its reservation is returned.
- Other requests continue unchanged.

### Context / RoPE / YaRN profiles

- `--contextsize` accepts up to 1048576.
- `--ropescaling none|linear|yarn` selects the llama.cpp scaling type explicitly. The factor is `orig/contextsize`, where `orig` comes from `--yarnorigctx` or model metadata.
  - `--yarnextfactor`, `--yarnattnfactor`, `--yarnbetafast` and `--yarnbetaslow` expose the YaRN parameters.
  - Without `--ropescaling`, Kobold's automatic behaviour is unchanged.
- At load time a context profile is logged and exposed in `/api/extra/runtime` (`context`). It contains:
  - requested context and allocated cells (per sequence and total)
  - trained context
  - effective RoPE base, scale and YaRN parameters
  - KV types and bytes for target and draft
  - `quality_verified: false` with a status string
- **Allocated capacity is not a quality claim.** Long-context quality beyond the trained context is not verified by this runtime.

## Verification

Host: Apple M3 Pro, 36 GiB, macOS 15.8.1, Metal. Build: `make -j11 LLAMA_METAL=1 KCPP_PARROT=1 koboldcpp_default` (exit 0).

Model: `bartowski/Qwen_Qwen3.5-0.8B-GGUF@f36b1ea4` `Qwen_Qwen3.5-0.8B-Q4_K_M.gguf`, sha256 `fb044e93…9526`. It is a hybrid attention/recurrent model with 1 MTP layer, trained context 262144.

Suite: `tests/test_parallel_runtime.py` passes 62/62 checks (`docs/evidence/parallel-run3-qwen35-0.8b.json`), and again after the post-v0.5.0 backports.

- **Isolation:**
  - Four concurrent greedy requests produce output byte-identical to their solo runs. Each contains only its own secret word.
  - `peak_live=4`.
  - Prompt token counts are unchanged under concurrency.
- **MTP:**
  - `draft-mtp` is active in both lanes. On this hybrid model rollback uses recurrent snapshots (`n_rs_seq=3`).
  - With EOS banned, every request reports exactly 40 completion tokens and `draft_accepted <= draft_tokens`.
- **Admission:**
  - Chat `input_tokens` (22) equals `usage.prompt_tokens`.
  - `max_tokens = n_ctx - 22` is admitted. One more returns 400 `exceed_context_size_error` with `n_prompt_tokens=22, n_ctx=4096`, including for `stream: true` before any SSE bytes.
  - A 5001-token prompt is rejected with its exact untruncated count.
  - A 3001-token prompt asking for 2000 completion tokens is rejected; asking for 64 is admitted.
- **Samplers:** grammar, DRY+XTC and mirostat run concurrently. Grammar output is constrained, and the neighbouring request is unaffected.
- **Cancellation:** a keyed abort stopped request A at 67/1200 tokens while B completed 1200/1200. Exactly one abort was counted.
- **Rejections:**
  - At startup, parallel without `--noshift`, `--smartcontext`, and `--ropescaling` with `--overridenativecontext` each exit 2 with an explicit message.
  - Phrase bans and image input return 400 `not_supported_error`.
  - `--parallelserial` routes a phrase-ban request to the serial lane and still serves parallel requests.

Throughput (`tests/bench_parallel_runtime.py`, 128 greedy tokens, `docs/evidence/bench-qwen35-0.8b-vs-llama-server-v0.5.0.json`):

| config | 4 concurrent, aggregate tok/s | draft acceptance |
| --- | --- | --- |
| Kobold serial lane | 126 | – |
| Kobold serial lane + MTP | 78 | 0.361 |
| Kobold parallel lane (4 slots) | 193 | – |
| Kobold parallel lane + MTP | 157 | 0.361 |
| upstream llama-server v0.5.0 `-np 4` | 202 | – |
| upstream llama-server v0.5.0 `-np 4` + MTP | 157 | 0.358 |

The parallel lane is within about 5% of upstream without MTP and matches it with MTP. Draft and accept counts match upstream within one token.

On this 0.8B model MTP is slower than no-MTP in both implementations: acceptance of about 36% doesn't pay for the extra draft head and verification. That measures this model, not the integration.

Context profiles (`docs/evidence/profiles-qwen35-0.8b-128k-1m.json`; q8_0 KV, MTP draft 2, serial lane):

| requested | allocated cells | RoPE | target KV | MTP KV | passkey 4K / 32K |
| --- | --- | --- | --- | --- | --- |
| 131072 | 131328 | none | 875 MiB | 136 MiB | found / found |
| 262144 | 262400 | none | 1691 MiB | 272 MiB | found / found |
| 524288 | 524544 | YaRN ×2 (orig 262144) | 3323 MiB | 544 MiB | found / found |
| 1048576 | 1048832 | YaRN ×4 | 6587 MiB | 1088 MiB | found / found |

All four allocate and generate with MTP active. The shallow passkey probes show the scaled configurations still read early context. They are not evidence of quality at full depth; see deep probes below.

## Remaining limitations

- The parallel lane serves text only; multimodal is either rejected or routed to the serial lane with `--parallelserial`.
- The serial and parallel lanes alternate rather than overlap.
- No prefix reuse on recurrent/hybrid models, because their memory cannot truncate a sequence.
- Prompt-only (`input_tokens`) counting does not include media tokens; those requests return 400.
- Long-context quality (beyond trained context, with or without YaRN) is not measured here.
- CUDA/ROCm/Vulkan builds and multi-GPU behaviour were not exercised on this host (Apple M3 Pro, Metal).

## Application integration (cocaine-rats) — report only, not implemented

- `crates/loom-mesh/src/inference/supervisor.rs` launches with `--parallelrequests 1 --multiuser 1 --nofastforward --noshift`.
  - To use concurrent slots, pass `--parallelrequests N --noshift` and keep `--multiuser` at least N.
  - Use `/v1/chat/completions/input_tokens` (or `/api/extra/admission`) for exact admission, and treat HTTP 400 `exceed_context_size_error` as a non-retryable client error.
- MTP: pass `--usemtp --draftamount 2` explicitly; `deploy/inference-worker/gpu-fit.md` assumes 2 while Kobold defaults to 4.
  - Read `GET /api/extra/runtime` → `mtp.active` to prove MTP is on, rather than inferring it from flags.
- Memory planning (`gpu_fit.py`, `GGUF-VRAM.md`):
  - The parallel lane uses one unified KV pool of `--contextsize + 128` cells, padded to 256, shared by all slots. It is not one context per slot.
    - Measured: `--contextsize 4096` with 4 slots gives `pool_cells = n_ctx_seq = 4352`.
    - Any single request may use the whole context, but live reservations (`prompt + max_tokens`) must fit the pool together. Otherwise the request waits in FIFO order.
    - To serve N concurrent requests of up to C tokens each, set `--contextsize ≈ N × C`.
  - Hybrid recurrent state scales with `n_seq_max = slots + 1` and `(1 + draft)` snapshots.
- YaRN: the chart's statement that "`--ropeconfig` cannot select YaRN" is superseded by `--ropescaling yarn --yarnorigctx`.
  - The 1M context point is now accepted by the CLI.
  - The chart should keep capacity and quality separate.
- Artifact pins (`deploy/koboldcpp-artifacts.json`) must move to a binary built from this branch once it is published. Publishing is not part of this change.
