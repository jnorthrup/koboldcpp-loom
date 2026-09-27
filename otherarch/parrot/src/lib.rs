//! C ABI shim over tts-rs (the Kokoro engine used by rishiskhare/parrot).
//! See kcpp_parrot.h for the contract.

use std::cell::RefCell;
use std::ffi::{c_char, c_float, c_int, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Mutex;

use tts_rs::engines::kokoro::{KokoroEngine, KokoroInferenceParams, KokoroModelParams};
use tts_rs::SynthesisEngine;

static ENGINE: Mutex<Option<KokoroEngine>> = Mutex::new(None);

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
    static VOICE_LIST: RefCell<CString> = RefCell::new(CString::default());
}

fn set_err(msg: impl Into<String>) {
    let s = msg.into().replace('\0', " ");
    LAST_ERROR.with(|e| *e.borrow_mut() = CString::new(s).unwrap_or_default());
}

unsafe fn opt_str(p: *const c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    let s = CStr::from_ptr(p).to_string_lossy().into_owned();
    if s.is_empty() { None } else { Some(s) }
}

fn guarded<F: FnOnce() -> Result<c_int, String>>(f: F) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_err(e);
            0
        }
        Err(_) => {
            set_err("parrot: panic inside TTS engine");
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn parrot_load(
    model_dir: *const c_char,
    threads: c_int,
    espeak_bin: *const c_char,
    espeak_data: *const c_char,
) -> c_int {
    guarded(|| {
        let dir = opt_str(model_dir).ok_or("parrot: empty model path")?;
        let mut dir = PathBuf::from(dir);
        if dir.is_file() {
            // Accept a path to the .onnx file; tts-rs wants the containing directory.
            dir = dir.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        }
        let bin = opt_str(espeak_bin).map(PathBuf::from);
        let data = opt_str(espeak_data).map(PathBuf::from);
        let mut eng = KokoroEngine::with_espeak(bin, data);
        let params = KokoroModelParams {
            num_threads: if threads > 0 { Some(threads as usize) } else { None },
            optimized_model_cache_path: None,
        };
        eng.load_model_with_params(&dir, params)
            .map_err(|e| format!("parrot: load failed from {}: {}", dir.display(), e))?;
        *ENGINE.lock().map_err(|_| "parrot: engine lock poisoned")? = Some(eng);
        Ok(1)
    })
}

#[no_mangle]
pub unsafe extern "C" fn parrot_synthesize(
    text: *const c_char,
    voice: *const c_char,
    speed: c_float,
    out_samples: *mut *mut c_float,
    out_len: *mut usize,
    out_rate: *mut u32,
) -> c_int {
    guarded(|| {
        if out_samples.is_null() || out_len.is_null() || out_rate.is_null() {
            return Err("parrot: null output pointer".into());
        }
        *out_samples = std::ptr::null_mut();
        *out_len = 0;
        *out_rate = 0;
        let text = opt_str(text).ok_or("parrot: empty text")?;
        let voice = opt_str(voice).unwrap_or_else(|| "af_heart".to_string());
        let speed = if speed.is_finite() && speed > 0.0 { speed.clamp(0.5, 2.0) } else { 1.0 };
        let mut guard = ENGINE.lock().map_err(|_| "parrot: engine lock poisoned")?;
        let eng = guard.as_mut().ok_or("parrot: model not loaded")?;
        let res = eng
            .synthesize(&text, Some(KokoroInferenceParams { voice, speed, style_index: None }))
            .map_err(|e| format!("parrot: synthesis failed: {}", e))?;
        let boxed = res.samples.into_boxed_slice();
        *out_len = boxed.len();
        *out_rate = res.sample_rate;
        *out_samples = Box::into_raw(boxed) as *mut c_float;
        Ok(1)
    })
}

#[no_mangle]
pub unsafe extern "C" fn parrot_free_samples(samples: *mut c_float, len: usize) {
    if !samples.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(samples, len)));
    }
}

#[no_mangle]
pub unsafe extern "C" fn parrot_has_voice(voice: *const c_char) -> c_int {
    let Some(v) = opt_str(voice) else { return 0 };
    let Ok(guard) = ENGINE.lock() else { return 0 };
    match guard.as_ref() {
        Some(e) => e.list_voices().iter().any(|x| *x == v) as c_int,
        None => 0,
    }
}

#[no_mangle]
pub extern "C" fn parrot_list_voices() -> *const c_char {
    let joined = ENGINE
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(|e| {
            let mut v = e.list_voices();
            v.sort_unstable();
            v.join("\n")
        }))
        .unwrap_or_default();
    VOICE_LIST.with(|c| {
        *c.borrow_mut() = CString::new(joined).unwrap_or_default();
        c.borrow().as_ptr()
    })
}

#[no_mangle]
pub extern "C" fn parrot_unload() {
    if let Ok(mut g) = ENGINE.lock() {
        *g = None;
    }
}

#[no_mangle]
pub extern "C" fn parrot_is_loaded() -> c_int {
    ENGINE.lock().map(|g| g.is_some() as c_int).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn parrot_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}
