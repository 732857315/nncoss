import argparse
import ctypes
import math
import struct
import sys
from pathlib import Path


def tiny_model_bytes():
    output = bytearray(b"NNCSLLM\0")
    output.extend(struct.pack("<12I", 1, 1, 5, 4, 8, 1, 2, 1, 2, 16, 6, 0))
    output.extend(struct.pack("<2f", 1e-5, 10000.0))
    salt = 0

    def tensor(length):
        nonlocal salt
        salt += 1
        for index in range(length):
            value = (index * 17 + salt * 31) % 97
            output.extend(struct.pack("<f", (value - 48.0) / 192.0))

    tensor(5 * 4)
    tensor(4)
    tensor(2 * 2 * 4)
    tensor(1 * 2 * 4)
    tensor(1 * 2 * 4)
    tensor(2 * 2)
    tensor(1 * 2)
    tensor(1 * 2)
    tensor(2)
    tensor(2)
    tensor(4 * 2 * 2)
    tensor(4)
    tensor(8 * 4)
    tensor(8 * 4)
    tensor(4 * 8)
    tensor(4)
    tensor(5 * 4)
    return bytes(output)


def native_library_name():
    if sys.platform == "darwin":
        return "libnncoss_llm.dylib"
    if sys.platform == "win32":
        return "nncoss_llm.dll"
    return "libnncoss_llm.so"


def run_native(path, model_bytes):
    library = ctypes.CDLL(str(path))

    def signature(name, arguments, result):
        function = getattr(library, name)
        function.argtypes = arguments
        function.restype = result
        return function

    allocate = signature("nncoss_llm_alloc", [ctypes.c_size_t], ctypes.c_void_p)
    deallocate = signature("nncoss_llm_dealloc", [ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int)
    model_load = signature("nncoss_llm_model_load", [ctypes.c_void_p, ctypes.c_size_t], ctypes.c_void_p)
    model_free = signature("nncoss_llm_model_free", [ctypes.c_void_p], None)
    vocab_size = signature("nncoss_llm_model_vocab_size", [ctypes.c_void_p], ctypes.c_uint32)
    session_new = signature("nncoss_llm_session_new", [ctypes.c_void_p, ctypes.c_size_t], ctypes.c_void_p)
    session_free = signature("nncoss_llm_session_free", [ctypes.c_void_p], None)
    session_reset = signature("nncoss_llm_session_reset", [ctypes.c_void_p], ctypes.c_int)
    position = signature("nncoss_llm_session_position", [ctypes.c_void_p], ctypes.c_size_t)
    prefill = signature("nncoss_llm_prefill", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t], ctypes.c_int)
    decode = signature("nncoss_llm_decode", [ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int)
    sample = signature("nncoss_llm_sample", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)], ctypes.c_int)
    copy_logits = signature("nncoss_llm_copy_logits", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t], ctypes.c_int)
    buffer = allocate(len(model_bytes))
    if not buffer:
        raise RuntimeError("原生模型缓冲区分配失败")
    ctypes.memmove(buffer, model_bytes, len(model_bytes))
    model = model_load(buffer, len(model_bytes))
    if deallocate(buffer, len(model_bytes)) != 0 or not model:
        raise RuntimeError("原生模型加载失败")
    if vocab_size(model) != 5:
        raise RuntimeError("原生词表尺寸错误")
    session = session_new(model, 8)
    model_free(model)
    if not session:
        raise RuntimeError("原生会话创建失败")
    tokens = (ctypes.c_uint32 * 2)(1, 2)
    if prefill(session, tokens, 2) != 0 or position(session) != 2:
        raise RuntimeError("原生 prefill 失败")
    next_token = ctypes.c_uint32()
    if sample(session, ctypes.byref(next_token)) != 0:
        raise RuntimeError("原生采样失败")
    first_logits = (ctypes.c_float * 5)()
    if copy_logits(session, first_logits, 5) != 0:
        raise RuntimeError("原生 logits 复制失败")
    if decode(session, next_token.value) != 0 or position(session) != 3:
        raise RuntimeError("原生 decode 失败")
    second_logits = (ctypes.c_float * 5)()
    if copy_logits(session, second_logits, 5) != 0:
        raise RuntimeError("原生 decode logits 复制失败")
    if session_reset(session) != 0 or position(session) != 0:
        raise RuntimeError("原生 reset 失败")
    session_free(session)
    return list(first_logits), next_token.value, list(second_logits)


def run_wasm(path, model_bytes):
    try:
        from wasmtime import Config, Engine, Instance, Module, Store
    except ImportError as error:
        raise SystemExit("WASM 验收依赖 wasmtime，请安装 Python wasmtime 包") from error
    config = Config()
    config.memory_guard_size = 1024 * 1024
    config.memory_reservation = 64 * 1024 * 1024
    config.memory_reservation_for_growth = 64 * 1024 * 1024
    config.signals_based_traps = False
    config.parallel_compilation = False
    store = Store(Engine(config))
    module = Module.from_file(store.engine, str(path))
    instance = Instance(store, module, [])
    exports = instance.exports(store)
    memory = exports["memory"]

    def call(name, *arguments):
        return exports[name](store, *arguments)

    if call("nncoss_llm_abi_version") != 1:
        raise RuntimeError("WASM ABI 版本错误")
    buffer = call("nncoss_llm_alloc", len(model_bytes))
    if not buffer:
        raise RuntimeError("WASM 模型缓冲区分配失败")
    memory.write(store, model_bytes, buffer)
    model = call("nncoss_llm_model_load", buffer, len(model_bytes))
    if call("nncoss_llm_dealloc", buffer, len(model_bytes)) != 0 or not model:
        raise RuntimeError("WASM 模型加载失败")
    if call("nncoss_llm_model_vocab_size", model) != 5:
        raise RuntimeError("WASM 词表尺寸错误")
    session = call("nncoss_llm_session_new", model, 8)
    call("nncoss_llm_model_free", model)
    if not session:
        raise RuntimeError("WASM 会话创建失败")
    token_buffer = call("nncoss_llm_alloc", 8)
    memory.write(store, struct.pack("<2I", 1, 2), token_buffer)
    if call("nncoss_llm_prefill", session, token_buffer, 2) != 0 or call("nncoss_llm_session_position", session) != 2:
        raise RuntimeError("WASM prefill 失败")
    call("nncoss_llm_dealloc", token_buffer, 8)
    output_token = call("nncoss_llm_alloc", 4)
    if call("nncoss_llm_sample", session, output_token) != 0:
        raise RuntimeError("WASM 采样失败")
    next_token = struct.unpack("<I", memory.read(store, output_token, output_token + 4))[0]
    call("nncoss_llm_dealloc", output_token, 4)
    logits_buffer = call("nncoss_llm_alloc", 20)
    if call("nncoss_llm_copy_logits", session, logits_buffer, 5) != 0:
        raise RuntimeError("WASM logits 复制失败")
    first_logits = struct.unpack("<5f", memory.read(store, logits_buffer, logits_buffer + 20))
    if call("nncoss_llm_decode", session, next_token) != 0 or call("nncoss_llm_session_position", session) != 3:
        raise RuntimeError("WASM decode 失败")
    if call("nncoss_llm_copy_logits", session, logits_buffer, 5) != 0:
        raise RuntimeError("WASM decode logits 复制失败")
    second_logits = struct.unpack("<5f", memory.read(store, logits_buffer, logits_buffer + 20))
    call("nncoss_llm_dealloc", logits_buffer, 20)
    if call("nncoss_llm_session_reset", session) != 0 or call("nncoss_llm_session_position", session) != 0:
        raise RuntimeError("WASM reset 失败")
    call("nncoss_llm_session_free", session)
    return list(first_logits), next_token, list(second_logits)


def assert_close(actual, expected):
    if len(actual) != len(expected):
        raise AssertionError("logits 长度不一致")
    for left, right in zip(actual, expected):
        if not math.isclose(left, right, rel_tol=1e-5, abs_tol=2e-5):
            raise AssertionError(f"原生/WASM logits 不一致：{actual} != {expected}")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="执行原生与 WASM LLM ABI 跨目标数值验收")
    parser.add_argument("--native", type=Path, default=root / "target" / "release" / native_library_name())
    parser.add_argument("--wasm", type=Path, default=root / "target" / "wasm32-unknown-unknown" / "release" / "nncoss_llm.wasm")
    args = parser.parse_args()
    model = tiny_model_bytes()
    native = run_native(args.native, model)
    wasm = run_wasm(args.wasm, model)
    if native[1] != wasm[1]:
        raise AssertionError(f"原生/WASM 采样不一致：{native[1]} != {wasm[1]}")
    assert_close(native[0], wasm[0])
    assert_close(native[2], wasm[2])
    maximum = max(abs(left - right) for left, right in zip(native[0] + native[2], wasm[0] + wasm[2]))
    print(f"原生/WASM LLM 验收通过：token={native[1]}，最大 logits 误差={maximum:.3g}")


if __name__ == "__main__":
    main()
