// C ABI for the optional Parrot (Kokoro ONNX via tts-rs) TTS backend.
// Only compiled in when koboldcpp is configured with KCPP_PARROT.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Load a Kokoro ONNX model directory (must contain one *.onnx and voices-v1.0.bin).
// threads <= 0 lets ONNX Runtime decide. espeak_bin / espeak_data may be NULL or ""
// to use espeak-ng from PATH. Returns 1 on success, 0 on failure (see parrot_last_error).
int parrot_load(const char * model_dir, int threads, const char * espeak_bin, const char * espeak_data);

// Synthesize mono float PCM. On success returns 1 and sets *out_samples (free with
// parrot_free_samples), *out_len, *out_rate. Returns 0 on failure.
int parrot_synthesize(const char * text, const char * voice, float speed,
                      float ** out_samples, size_t * out_len, uint32_t * out_rate);

void parrot_free_samples(float * samples, size_t len);

// 1 if the loaded model has this voice id (e.g. "af_heart"), else 0.
int parrot_has_voice(const char * voice);

// Newline-separated voice ids of the loaded model. Pointer valid until next call.
const char * parrot_list_voices(void);

void parrot_unload(void);
int parrot_is_loaded(void);

// Last error message for this thread's most recent failing call. Never NULL.
const char * parrot_last_error(void);

#ifdef __cplusplus
}
#endif
