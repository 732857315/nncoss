import argparse
import ctypes
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "crates" / "nncoss-llm" / "tests" / "fixtures" / "tiny_v1.nncossllm"
VOCAB = 5
MAX_CONTEXT = 16

OK = 0
INVALID_ARGUMENT = -1
TOKEN_OUT_OF_RANGE = -5
NO_LOGITS = -6
OUTPUT_TOO_SMALL = -7

C = ctypes
NATIVE_SIGNATURES = {
    "nncoss_llm_abi_version": ([], C.c_uint32),
    "nncoss_llm_last_error": ([], C.c_void_p),
    "nncoss_llm_alloc": ([C.c_size_t], C.c_void_p),
    "nncoss_llm_dealloc": ([C.c_void_p, C.c_size_t], C.c_int),
    "nncoss_llm_model_load": ([C.c_void_p, C.c_size_t], C.c_void_p),
    "nncoss_llm_model_free": ([C.c_void_p], None),
    "nncoss_llm_model_vocab_size": ([C.c_void_p], C.c_uint32),
    "nncoss_llm_model_max_context": ([C.c_void_p], C.c_uint32),
    "nncoss_llm_session_new": ([C.c_void_p, C.c_size_t], C.c_void_p),
    "nncoss_llm_session_free": ([C.c_void_p], None),
    "nncoss_llm_session_reset": ([C.c_void_p], C.c_int),
    "nncoss_llm_session_position": ([C.c_void_p], C.c_size_t),
    "nncoss_llm_prefill": ([C.c_void_p, C.c_void_p, C.c_size_t], C.c_int),
    "nncoss_llm_decode": ([C.c_void_p, C.c_uint32], C.c_int),
    "nncoss_llm_set_sampler": ([C.c_void_p, C.c_float, C.c_uint32, C.c_float, C.c_uint64], C.c_int),
    "nncoss_llm_sample": ([C.c_void_p, C.c_void_p], C.c_int),
    "nncoss_llm_copy_logits": ([C.c_void_p, C.c_void_p, C.c_size_t], C.c_int),
}
POINTER_RESULTS = {name for name, (_, result) in NATIVE_SIGNATURES.items() if result is C.c_void_p}


def native_library_name():
    if sys.platform == "darwin":
        return "libnncoss_llm.dylib"
    if sys.platform == "win32":
        return "nncoss_llm.dll"
    return "libnncoss_llm.so"


class NativeTarget:
    name = "原生"

    def __init__(self, path):
        library = ctypes.CDLL(str(path))
        self.functions = {}
        for name, (arguments, result) in NATIVE_SIGNATURES.items():
            function = getattr(library, name)
            function.argtypes = arguments
            function.restype = result
            self.functions[name] = function

    def call(self, name, *arguments):
        value = self.functions[name](*arguments)
        if name in POINTER_RESULTS:
            return value or 0
        return value

    def write(self, pointer, data):
        ctypes.memmove(pointer, data, len(data))

    def read(self, pointer, length):
        return ctypes.string_at(pointer, length)

    def read_c_string(self, pointer):
        return ctypes.string_at(pointer)


class WasmTarget:
    name = "WASM"

    def __init__(self, path):
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
        self.store = Store(Engine(config))
        module = Module.from_file(self.store.engine, str(path))
        self.exports = Instance(self.store, module, []).exports(self.store)
        self.memory = self.exports["memory"]

    def call(self, name, *arguments):
        value = self.exports[name](self.store, *arguments)
        if name in POINTER_RESULTS:
            return value & 0xFFFFFFFF
        return value

    def write(self, pointer, data):
        self.memory.write(self.store, data, pointer)

    def read(self, pointer, length):
        return bytes(self.memory.read(self.store, pointer, pointer + length))

    def read_c_string(self, pointer):
        output = bytearray()
        while True:
            byte = self.read(pointer + len(output), 1)
            if byte == b"\0":
                return bytes(output)
            output.extend(byte)


def expect(condition, target, message):
    if not condition:
        raise RuntimeError(f"{target.name} {message}")


def last_error(target):
    pointer = target.call("nncoss_llm_last_error")
    expect(pointer != 0, target, "last_error 返回空指针")
    return target.read_c_string(pointer).decode("utf-8")


def error_after(target, failed):
    return last_error(target) if failed else ""


def expect_status(target, status, expected, label, errors, message=None):
    expect(status == expected, target, f"{label} 状态码为 {status}，预期 {expected}")
    if expected != OK:
        message = last_error(target) if message is None else message
        expect(message, target, f"{label} 未设置错误消息")
        errors.append((label, status, message))


def with_buffer(target, length, action):
    pointer = target.call("nncoss_llm_alloc", length)
    expect(pointer != 0, target, f"分配 {length} 字节失败")
    try:
        return action(pointer)
    finally:
        expect(target.call("nncoss_llm_dealloc", pointer, length) == OK, target, "释放缓冲区失败")


def prefill(target, session, tokens):
    data = struct.pack(f"<{len(tokens)}I", *tokens)

    def run(pointer):
        target.write(pointer, data)
        status = target.call("nncoss_llm_prefill", session, pointer, len(tokens))
        return status, error_after(target, status != OK)

    return with_buffer(target, len(data), run)


def copy_logits(target, session, length):
    size = max(length, 1) * 4

    def run(pointer):
        status = target.call("nncoss_llm_copy_logits", session, pointer, length)
        if status != OK:
            return status, None, error_after(target, True)
        return status, list(struct.unpack(f"<{length}f", target.read(pointer, length * 4))), ""

    return with_buffer(target, size, run)


def sample(target, session):
    def run(pointer):
        status = target.call("nncoss_llm_sample", session, pointer)
        return status, struct.unpack("<I", target.read(pointer, 4))[0], error_after(target, status != OK)

    return with_buffer(target, 4, run)


def load_model(target, model_bytes):
    def run(pointer):
        target.write(pointer, model_bytes)
        model = target.call("nncoss_llm_model_load", pointer, len(model_bytes))
        return model, error_after(target, model == 0)

    return with_buffer(target, len(model_bytes), run)


def run_scenario(target, model_bytes):
    errors = []
    expect(target.call("nncoss_llm_abi_version") == 1, target, "ABI 版本错误")
    expect(target.call("nncoss_llm_dealloc", 0, 1) == INVALID_ARGUMENT, target, "空指针释放未报错")
    errors.append(("dealloc_null", INVALID_ARGUMENT, last_error(target)))

    truncated, message = load_model(target, model_bytes[:-1])
    expect(truncated == 0 and message, target, "截断模型被错误接受或未设置错误消息")
    errors.append(("model_load_truncated", None, message))

    model, message = load_model(target, model_bytes)
    expect(model != 0, target, "模型加载失败：" + message)
    expect(target.call("nncoss_llm_model_vocab_size", model) == VOCAB, target, "词表尺寸错误")
    expect(target.call("nncoss_llm_model_max_context", model) == MAX_CONTEXT, target, "上下文上限错误")
    session = target.call("nncoss_llm_session_new", model, 8)
    target.call("nncoss_llm_model_free", model)
    expect(session != 0, target, "会话创建失败")

    status, _, message = copy_logits(target, session, VOCAB)
    expect_status(target, status, NO_LOGITS, "copy_logits_before_prefill", errors, message)
    status, message = prefill(target, session, [1, 99])
    expect_status(target, status, TOKEN_OUT_OF_RANGE, "prefill_bad_token", errors, message)
    expect(target.call("nncoss_llm_session_position", session) == 0, target, "失败的 prefill 改变了位置")

    status, _ = prefill(target, session, [1, 2])
    expect_status(target, status, OK, "prefill", errors)
    expect(target.call("nncoss_llm_session_position", session) == 2, target, "prefill 位置错误")
    status, _, message = copy_logits(target, session, VOCAB - 1)
    expect_status(target, status, OUTPUT_TOO_SMALL, "copy_logits_too_small", errors, message)
    status = target.call("nncoss_llm_set_sampler", session, 1.0, 0, 0.0, 0)
    expect_status(target, status, INVALID_ARGUMENT, "set_sampler_bad_top_p", errors)

    status, next_token, _ = sample(target, session)
    expect_status(target, status, OK, "sample", errors)
    status, first_logits, _ = copy_logits(target, session, VOCAB)
    expect_status(target, status, OK, "copy_logits", errors)
    expect_status(target, target.call("nncoss_llm_decode", session, 99), TOKEN_OUT_OF_RANGE, "decode_bad_token", errors)
    expect_status(target, target.call("nncoss_llm_decode", session, next_token), OK, "decode", errors)
    expect(target.call("nncoss_llm_session_position", session) == 3, target, "decode 位置错误")
    status, second_logits, _ = copy_logits(target, session, VOCAB)
    expect_status(target, status, OK, "copy_logits_after_decode", errors)

    expect_status(target, target.call("nncoss_llm_session_reset", session), OK, "reset", errors)
    expect(target.call("nncoss_llm_session_position", session) == 0, target, "reset 位置错误")
    status, _, message = copy_logits(target, session, VOCAB)
    expect_status(target, status, NO_LOGITS, "copy_logits_after_reset", errors, message)
    target.call("nncoss_llm_session_free", session)
    return {
        "token": next_token,
        "first_logits": first_logits,
        "second_logits": second_logits,
        "errors": errors,
    }


def assert_close(actual, expected):
    if len(actual) != len(expected):
        raise AssertionError("logits 长度不一致")
    for left, right in zip(actual, expected):
        if not math.isclose(left, right, rel_tol=1e-5, abs_tol=2e-5):
            raise AssertionError(f"原生/WASM logits 不一致：{actual} != {expected}")


def main():
    parser = argparse.ArgumentParser(description="执行原生与 WASM LLM ABI 跨目标数值与错误路径验收")
    parser.add_argument("--native", type=Path, default=ROOT / "target" / "release" / native_library_name())
    parser.add_argument("--wasm", type=Path, default=ROOT / "target" / "wasm32-unknown-unknown" / "release" / "nncoss_llm.wasm")
    parser.add_argument("--model", type=Path, default=FIXTURE, help="与 Rust 单元测试共享的 tiny decoder 夹具")
    args = parser.parse_args()
    model = args.model.read_bytes()
    native = run_scenario(NativeTarget(args.native), model)
    wasm = run_scenario(WasmTarget(args.wasm), model)
    if native["token"] != wasm["token"]:
        raise AssertionError(f"原生/WASM 采样不一致：{native['token']} != {wasm['token']}")
    if native["errors"] != wasm["errors"]:
        raise AssertionError(f"原生/WASM 错误码或消息不一致：{native['errors']} != {wasm['errors']}")
    assert_close(native["first_logits"], wasm["first_logits"])
    assert_close(native["second_logits"], wasm["second_logits"])
    pairs = zip(native["first_logits"] + native["second_logits"], wasm["first_logits"] + wasm["second_logits"])
    maximum = max(abs(left - right) for left, right in pairs)
    print(
        f"原生/WASM LLM 验收通过：token={native['token']}，"
        f"错误路径={len(native['errors'])} 项一致，最大 logits 误差={maximum:.3g}"
    )


if __name__ == "__main__":
    main()
