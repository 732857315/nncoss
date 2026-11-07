use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::RefCell;
use std::ffi::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use crate::{ABI_VERSION, Error, Model, Result, SamplingConfig, Session};

thread_local! {
    static LAST_ERROR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

pub struct ModelHandle(Arc<Model>);

fn clear_error() {
    LAST_ERROR.with(|error| error.borrow_mut().clear());
}

fn set_error(error: Error) {
    LAST_ERROR.with(|last| {
        let mut last = last.borrow_mut();
        last.clear();
        last.extend_from_slice(error.to_string().as_bytes());
        last.push(0);
    });
}

fn ffi_status(function: impl FnOnce() -> Result<()>) -> i32 {
    match catch_unwind(AssertUnwindSafe(function)) {
        Ok(Ok(())) => {
            clear_error();
            0
        }
        Ok(Err(error)) => {
            let code = error.code();
            set_error(error);
            code
        }
        Err(_) => {
            set_error(Error::Panic);
            Error::Panic.code()
        }
    }
}

fn ffi_ptr<T>(function: impl FnOnce() -> Result<*mut T>) -> *mut T {
    match catch_unwind(AssertUnwindSafe(function)) {
        Ok(Ok(pointer)) => {
            clear_error();
            pointer
        }
        Ok(Err(error)) => {
            set_error(error);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_error(Error::Panic);
            std::ptr::null_mut()
        }
    }
}

fn ffi_value<T: Copy>(function: impl FnOnce() -> Result<T>, fallback: T) -> T {
    match catch_unwind(AssertUnwindSafe(function)) {
        Ok(Ok(value)) => {
            clear_error();
            value
        }
        Ok(Err(error)) => {
            set_error(error);
            fallback
        }
        Err(_) => {
            set_error(Error::Panic);
            fallback
        }
    }
}

unsafe fn raw_bytes<'a>(pointer: *const u8, length: usize) -> Result<&'a [u8]> {
    if pointer.is_null() {
        if length == 0 {
            return Ok(&[]);
        }
        return Err(Error::InvalidArgument("空指针"));
    }
    Ok(unsafe { std::slice::from_raw_parts(pointer, length) })
}

unsafe fn read_tokens(pointer: *const u32, length: usize) -> Result<Vec<u32>> {
    if pointer.is_null() {
        if length == 0 {
            return Ok(Vec::new());
        }
        return Err(Error::InvalidArgument("空指针"));
    }
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(length)
        .map_err(|_| Error::Allocation)?;
    for index in 0..length {
        tokens.push(unsafe { pointer.add(index).read_unaligned() });
    }
    Ok(tokens)
}

#[unsafe(no_mangle)]
pub extern "C" fn nncoss_llm_abi_version() -> u32 {
    ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn nncoss_llm_last_error() -> *const c_char {
    LAST_ERROR.with(|error| {
        let mut error = error.borrow_mut();
        if error.is_empty() {
            error.push(0);
        }
        error.as_ptr().cast()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn nncoss_llm_alloc(length: usize) -> *mut u8 {
    ffi_ptr(|| {
        let layout = Layout::array::<u8>(length).map_err(|_| Error::Allocation)?;
        if length == 0 {
            return Ok(std::ptr::NonNull::<u8>::dangling().as_ptr());
        }
        let pointer = unsafe { alloc_zeroed(layout) };
        if pointer.is_null() {
            return Err(Error::Allocation);
        }
        Ok(pointer)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nncoss_llm_dealloc(pointer: *mut u8, length: usize) -> i32 {
    ffi_status(|| {
        if length == 0 {
            return Ok(());
        }
        if pointer.is_null() {
            return Err(Error::InvalidArgument("空指针"));
        }
        let layout = Layout::array::<u8>(length).map_err(|_| Error::Allocation)?;
        unsafe { dealloc(pointer, layout) };
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nncoss_llm_model_load(
    pointer: *const u8,
    length: usize,
) -> *mut ModelHandle {
    ffi_ptr(|| {
        let model = Model::from_bytes(unsafe { raw_bytes(pointer, length) }?)?;
        let handle = Box::new(ModelHandle(Arc::new(model)));
        Ok(Box::into_raw(handle))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nncoss_llm_model_free(model: *mut ModelHandle) {
    if !model.is_null() {
        drop(unsafe { Box::from_raw(model) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nncoss_llm_model_vocab_size(model: *const ModelHandle) -> u32 {
    ffi_value(
        || {
            if model.is_null() {
                return Err(Error::InvalidArgument("空模型句柄"));
            }
            let model = unsafe { &*model };
            u32::try_from(model.0.config().vocab_size)
                .map_err(|_| Error::InvalidArgument("词表超过 u32 范围"))
        },
        0,
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nncoss_llm_model_max_context(model: *const ModelHandle) -> u32 {
    ffi_value(
        || {
            if model.is_null() {
                return Err(Error::InvalidArgument("空模型句柄"));
            }
            let model = unsafe { &*model };
            u32::try_from(model.0.config().max_context)
                .map_err(|_| Error::InvalidArgument("上下文超过 u32 范围"))
        },
        0,
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nncoss_llm_session_new(
    model: *const ModelHandle,
    max_context: usize,
) -> *mut Session {
    ffi_ptr(|| {
        if model.is_null() {
            return Err(Error::InvalidArgument("空模型句柄"));
        }
        let model = unsafe { &*model };
        let session = Session::new(model.0.clone(), max_context)?;
        Ok(Box::into_raw(Box::new(session)))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nncoss_llm_session_free(session: *mut Session) {
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nncoss_llm_session_reset(session: *mut Session) -> i32 {
    ffi_status(|| {
        if session.is_null() {
            return Err(Error::InvalidArgument("空会话句柄"));
        }
        unsafe { &mut *session }.reset();
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nncoss_llm_session_position(session: *const Session) -> usize {
    ffi_value(
        || {
            if session.is_null() {
                return Err(Error::InvalidArgument("空会话句柄"));
            }
            Ok(unsafe { &*session }.position())
        },
        0,
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nncoss_llm_prefill(
    session: *mut Session,
    tokens: *const u32,
    length: usize,
) -> i32 {
    ffi_status(|| {
        if session.is_null() {
            return Err(Error::InvalidArgument("空会话句柄"));
        }
        let tokens = unsafe { read_tokens(tokens, length) }?;
        unsafe { &mut *session }.prefill(&tokens)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nncoss_llm_decode(session: *mut Session, token: u32) -> i32 {
    ffi_status(|| {
        if session.is_null() {
            return Err(Error::InvalidArgument("空会话句柄"));
        }
        unsafe { &mut *session }.decode(token)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nncoss_llm_set_sampler(
    session: *mut Session,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: u64,
) -> i32 {
    ffi_status(|| {
        if session.is_null() {
            return Err(Error::InvalidArgument("空会话句柄"));
        }
        let config = SamplingConfig {
            temperature,
            top_k: top_k as usize,
            top_p,
        };
        unsafe { &mut *session }.set_sampler(config, seed)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nncoss_llm_sample(session: *mut Session, output: *mut u32) -> i32 {
    ffi_status(|| {
        if session.is_null() || output.is_null() {
            return Err(Error::InvalidArgument("空会话句柄或输出指针"));
        }
        let token = unsafe { &mut *session }.sample_next()?;
        unsafe { output.write_unaligned(token) };
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn nncoss_llm_copy_logits(
    session: *const Session,
    output: *mut f32,
    length: usize,
) -> i32 {
    ffi_status(|| {
        if session.is_null() {
            return Err(Error::InvalidArgument("空会话句柄"));
        }
        let logits = unsafe { &*session }.last_logits()?;
        if length < logits.len() {
            return Err(Error::OutputTooSmall {
                expected: logits.len(),
                actual: length,
            });
        }
        if output.is_null() {
            return Err(Error::InvalidArgument("空输出指针"));
        }
        for (index, value) in logits.iter().enumerate() {
            unsafe { output.add(index).write_unaligned(*value) };
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::tiny_model_bytes;

    #[test]
    fn wasm_c_abi_drives_a_complete_session() {
        let bytes = tiny_model_bytes();
        unsafe {
            let input = nncoss_llm_alloc(bytes.len());
            assert!(!input.is_null());
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), input, bytes.len());
            let model = nncoss_llm_model_load(input, bytes.len());
            assert!(!model.is_null());
            assert_eq!(nncoss_llm_dealloc(input, bytes.len()), 0);
            assert_eq!(nncoss_llm_abi_version(), ABI_VERSION);
            assert_eq!(nncoss_llm_model_vocab_size(model), 5);
            assert_eq!(nncoss_llm_model_max_context(model), 16);
            let session = nncoss_llm_session_new(model, 8);
            assert!(!session.is_null());
            let tokens = [1_u32, 2];
            assert_eq!(
                nncoss_llm_prefill(session, tokens.as_ptr(), tokens.len()),
                0
            );
            assert_eq!(nncoss_llm_session_position(session), 2);
            let mut token = u32::MAX;
            assert_eq!(nncoss_llm_sample(session, &mut token), 0);
            assert!(token < 5);
            let mut logits = vec![0.0; 5];
            assert_eq!(
                nncoss_llm_copy_logits(session, logits.as_mut_ptr(), logits.len()),
                0
            );
            assert!(logits.iter().any(|value| *value != 0.0));
            assert_eq!(nncoss_llm_decode(session, token), 0);
            assert_eq!(nncoss_llm_session_position(session), 3);
            assert_eq!(nncoss_llm_session_reset(session), 0);
            assert_eq!(nncoss_llm_session_position(session), 0);
            nncoss_llm_session_free(session);
            nncoss_llm_model_free(model);
        }
    }

    #[test]
    fn wasm_c_abi_reports_errors_without_panicking() {
        unsafe {
            assert!(nncoss_llm_model_load(std::ptr::null(), 10).is_null());
            let error = nncoss_llm_last_error();
            assert!(!error.is_null());
            assert_ne!(*error, 0);
            assert_eq!(
                nncoss_llm_prefill(std::ptr::null_mut(), std::ptr::null(), 0),
                -1
            );
            assert_eq!(nncoss_llm_dealloc(std::ptr::null_mut(), 1), -1);
            assert_eq!(nncoss_llm_dealloc(std::ptr::null_mut(), 0), 0);
            assert!(nncoss_llm_alloc(usize::MAX).is_null());
        }
    }

    #[test]
    fn ffi_helpers_convert_panics_to_errors_when_unwinding() {
        let status = ffi_status(|| -> Result<()> { panic!("ffi regression test") });
        assert_eq!(status, -99);
    }
}
