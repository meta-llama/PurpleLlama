# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import enum
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable, List, Match, Optional, Pattern, Set, Tuple


# NOTE: by overriding this pattern, you can filter out stack frames that you
# don't want to be considered for patching, such as library that assists
# fuzzing. Overwriting
FUZZ_LIB_INVOCATION_PATTERN: Pattern[str] = re.compile(
    "OVERWRITE_THIS_PATTERN_FOR_FUZZER_API_INVOCATION_PATTERN"
)
Identifier = str
MAX_PRESENTABLE_STACKTRACE_FRAMES: int = 20
REPLACEMENT_CHARACTER: str = "\ufffd"


def sanitize_function_name_aggressively(function_name: str) -> str:
    # NOTE: This is no-op for now, but you can add more aggressive sanitization
    # of function names here, if desired.
    return function_name


class SecurityBugType(enum.Enum):
    DENIAL_OF_SERVICE = 1
    MEMORY_CORRUPTION = 2
    NOT_SECURITY_BUG = 3
    UNKNOWN = 4


class CrashTypeMeta(enum.EnumMeta):
    def from_value(self, value: str) -> Optional["CrashType"]:
        try:
            return super().__call__(value)
        except ValueError:
            return None


class CrashType(enum.Enum, metaclass=CrashTypeMeta):
    def __new__(cls, value: str, _) -> "CrashType":
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, _: str, security_bug_type: SecurityBugType) -> None:
        self._security_bug_type = security_bug_type

    @property
    def security_bug_type(self) -> SecurityBugType:
        return self._security_bug_type


class GenericCrashType(CrashType):
    UNKNOWN = ("unknown", SecurityBugType.UNKNOWN)
    MEMORY_LEAK = ("memory-leak", SecurityBugType.DENIAL_OF_SERVICE)
    LARGE_MALLOC = ("large-malloc", SecurityBugType.DENIAL_OF_SERVICE)
    TIMEOUT = ("timeout", SecurityBugType.DENIAL_OF_SERVICE)
    CHECK_FAILURE = ("check-failure", SecurityBugType.DENIAL_OF_SERVICE)
    ASSERTION_FAILURE = ("assertion-failure", SecurityBugType.DENIAL_OF_SERVICE)
    CUSTOM_INVARIANT_FAILURE = (
        "custom-invariant-failure",
        SecurityBugType.DENIAL_OF_SERVICE,
    )
    UNCAUGHT_EXCEPTION = ("uncaught-exception", SecurityBugType.DENIAL_OF_SERVICE)
    DEADLY_SIGNAL = ("deadly-signal", SecurityBugType.DENIAL_OF_SERVICE)
    FOLLY_UNINITIALIZED_SINGLETON = (
        "folly-uninitialized-singleton",
        SecurityBugType.NOT_SECURITY_BUG,
    )
    FUZZ_TARGET_EXITED = ("fuzz-target-exited", SecurityBugType.DENIAL_OF_SERVICE)


class LSanCrashType(CrashType):
    DIRECT_LEAK = ("direct-leak", SecurityBugType.DENIAL_OF_SERVICE)
    INDIRECT_LEAK = ("indirect-leak", SecurityBugType.DENIAL_OF_SERVICE)


class MSanCrashType(CrashType):
    USE_OF_UNINITIALIZED_VALUE = (
        "use-of-uninitialized-value",
        SecurityBugType.MEMORY_CORRUPTION,
    )


# Sourced from https://github.com/llvm/llvm-project/blob/master/compiler-rt/lib/asan/asan_errors.cpp
class ASanCrashType(CrashType):
    ALLOC_DEALLOC_MISMATCH = (
        "alloc-dealloc-mismatch",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    ALLOCATION_SIZE_TOO_BIG = (
        "allocation-size-too-big",
        SecurityBugType.DENIAL_OF_SERVICE,
    )
    BAD_FREE = ("bad-free", SecurityBugType.MEMORY_CORRUPTION)
    BAD_MALLOC_USABLE_SIZE = (
        "bad-malloc_usable_size",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    CALLOC_OVERFLOW = ("calloc-overflow", SecurityBugType.MEMORY_CORRUPTION)
    DOUBLE_FREE = ("double-free", SecurityBugType.MEMORY_CORRUPTION)
    INVALID_ALIGNED_ALLOC_ALIGNMENT = (
        "invalid-aligned-alloc-alignment",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    INVALID_ALLOCATION_ALIGNMENT = (
        "invalid-allocation-alignment",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    INVALID_POINTER_PAIR = ("invalid-pointer-pair", SecurityBugType.MEMORY_CORRUPTION)
    INVALID_POSIX_MEMALIGN_ALIGNMENT = (
        "invalid-posix-memalign-alignment",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    NEGATIVE_SIZE_PARAM = ("negative-size-param", SecurityBugType.MEMORY_CORRUPTION)
    NEW_DELETE_TYPE_MISMATCH = (
        "new-delete-type-mismatch",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    NULL_DEREF = ("null-deref", SecurityBugType.DENIAL_OF_SERVICE)
    ODR_VIOLATION = ("odr-violation", SecurityBugType.MEMORY_CORRUPTION)
    OUT_OF_MEMORY = ("out-of-memory", SecurityBugType.DENIAL_OF_SERVICE)
    PVALLOC_OVERFLOW = ("pvalloc-overflow", SecurityBugType.MEMORY_CORRUPTION)
    READ = ("read", SecurityBugType.MEMORY_CORRUPTION)
    READ_CONTAINER_OVERFLOW = (
        "read-container-overflow",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    READ_DYNAMIC_STACK_BUFFER_OVERFLOW = (
        "read-dynamic-stack-buffer-overflow",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    READ_DYNAMIC_STACK_BUFFER_OVERFLOW_FAR_FROM_BOUNDS = (
        "read-dynamic-stack-buffer-overflow-far-from-bounds",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    READ_GLOBAL_BUFFER_OVERFLOW = (
        "read-global-buffer-overflow",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    READ_GLOBAL_BUFFER_OVERFLOW_FAR_FROM_BOUNDS = (
        "read-global-buffer-overflow-far-from-bounds",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    READ_HEAP_BUFFER_OVERFLOW = (
        "read-heap-buffer-overflow",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    READ_HEAP_BUFFER_OVERFLOW_FAR_FROM_BOUNDS = (
        "read-heap-buffer-overflow-far-from-bounds",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    READ_HEAP_USE_AFTER_FREE = (
        "read-heap-use-after-free",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    READ_INITIALIZATION_ORDER_FIASCO = (
        "read-initialization-order-fiasco",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    READ_INTRA_OBJECT_OVERFLOW = (
        "read-intra-object-overflow",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    READ_STACK_BUFFER_OVERFLOW = (
        "read-stack-buffer-overflow",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    READ_STACK_BUFFER_OVERFLOW_FAR_FROM_BOUNDS = (
        "read-stack-buffer-overflow-far-from-bounds",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    READ_STACK_BUFFER_UNDERFLOW = (
        "read-stack-buffer-underflow",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    READ_STACK_BUFFER_UNDERFLOW_FAR_FROM_BOUNDS = (
        "read-stack-buffer-underflow-far-from-bounds",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    READ_STACK_USE_AFTER_RETURN = (
        "read-stack-use-after-return",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    READ_STACK_USE_AFTER_SCOPE = (
        "read-stack-use-after-scope",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    READ_USE_AFTER_POISON = ("read-use-after-poison", SecurityBugType.MEMORY_CORRUPTION)
    REALLOCARRAY_OVERFLOW = ("reallocarray-overflow", SecurityBugType.MEMORY_CORRUPTION)
    RSS_LIMIT_EXCEEDED = ("rss-limit-exceeded", SecurityBugType.DENIAL_OF_SERVICE)
    SIGNAL = ("signal", SecurityBugType.DENIAL_OF_SERVICE)
    STACK_OVERFLOW = ("stack-overflow", SecurityBugType.DENIAL_OF_SERVICE)
    THREAD_LIMIT_EXCEEDED = ("thread-limit-exceeded", SecurityBugType.DENIAL_OF_SERVICE)
    UNKNOWN_CRASH = ("unknown-crash", SecurityBugType.MEMORY_CORRUPTION)
    WILD_ADDR = ("wild-addr", SecurityBugType.MEMORY_CORRUPTION)
    WILD_ADDR_READ = ("wild-addr-read", SecurityBugType.MEMORY_CORRUPTION)
    WILD_ADDR_WRITE = ("wild-addr-write", SecurityBugType.MEMORY_CORRUPTION)
    WILD_JUMP = ("wild-jump", SecurityBugType.MEMORY_CORRUPTION)
    WRITE = ("write", SecurityBugType.MEMORY_CORRUPTION)
    WRITE_CONTAINER_OVERFLOW = (
        "write-container-overflow",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    WRITE_DYNAMIC_STACK_BUFFER_OVERFLOW = (
        "write-dynamic-stack-buffer-overflow",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    WRITE_DYNAMIC_STACK_BUFFER_OVERFLOW_FAR_FROM_BOUNDS = (
        "write-dynamic-stack-buffer-overflow-far-from-bounds",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    WRITE_GLOBAL_BUFFER_OVERFLOW = (
        "write-global-buffer-overflow",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    WRITE_GLOBAL_BUFFER_OVERFLOW_FAR_FROM_BOUNDS = (
        "write-global-buffer-overflow-far-from-bounds",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    WRITE_HEAP_BUFFER_OVERFLOW = (
        "write-heap-buffer-overflow",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    WRITE_HEAP_BUFFER_OVERFLOW_FAR_FROM_BOUNDS = (
        "write-heap-buffer-overflow-far-from-bounds",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    WRITE_HEAP_USE_AFTER_FREE = (
        "write-heap-use-after-free",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    WRITE_INITIALIZATION_ORDER_FIASCO = (
        "write-initialization-order-fiasco",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    WRITE_INTRA_OBJECT_OVERFLOW = (
        "write-intra-object-overflow",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    WRITE_STACK_BUFFER_OVERFLOW = (
        "write-stack-buffer-overflow",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    WRITE_STACK_BUFFER_OVERFLOW_FAR_FROM_BOUNDS = (
        "write-stack-buffer-overflow-far-from-bounds",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    WRITE_STACK_BUFFER_UNDERFLOW = (
        "write-stack-buffer-underflow",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    WRITE_STACK_BUFFER_UNDERFLOW_FAR_FROM_BOUNDS = (
        "write-stack-buffer-underflow-far-from-bounds",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    WRITE_STACK_USE_AFTER_RETURN = (
        "write-stack-use-after-return",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    WRITE_STACK_USE_AFTER_SCOPE = (
        "write-stack-use-after-scope",
        SecurityBugType.MEMORY_CORRUPTION,
    )
    WRITE_USE_AFTER_POISON = (
        "write-use-after-poison",
        SecurityBugType.MEMORY_CORRUPTION,
    )


# Sourced from https://github.com/llvm/llvm-project/blob/master/compiler-rt/lib/ubsan/ubsan_checks.inc
class UBSanCrashType(CrashType):
    GENERIC_UB = ("undefined-behavior", SecurityBugType.DENIAL_OF_SERVICE)
    NULL_POINTER_USE = ("null-pointer-use", SecurityBugType.DENIAL_OF_SERVICE)
    NULLPTR_WITH_OFFSET = ("nullptr-with-offset", SecurityBugType.DENIAL_OF_SERVICE)
    NULLPTR_WITH_NON_ZERO_OFFSET = (
        "nullptr-with-nonzero-offset",
        SecurityBugType.DENIAL_OF_SERVICE,
    )
    NULLPTR_AFTER_NON_ZERO_OFFSET = (
        "nullptr-after-nonzero-offset",
        SecurityBugType.DENIAL_OF_SERVICE,
    )
    POINTER_OVERFLOW = ("pointer-overflow", SecurityBugType.DENIAL_OF_SERVICE)
    MISALIGNED_POINTER_USE = (
        "misaligned-pointer-use",
        SecurityBugType.DENIAL_OF_SERVICE,
    )
    ALIGNMENT_ASSUMPTION = ("alignment-assumption", SecurityBugType.DENIAL_OF_SERVICE)
    INSUFFICIENT_OBJECT_SIZE = (
        "insufficient-object-size",
        SecurityBugType.DENIAL_OF_SERVICE,
    )
    SIGNED_INTEGER_OVERFLOW = (
        "signed-integer-overflow",
        SecurityBugType.DENIAL_OF_SERVICE,
    )
    UNSIGNED_INTEGER_OVERFLOW = (
        "unsigned-integer-overflow",
        SecurityBugType.DENIAL_OF_SERVICE,
    )
    INTEGER_DIVIDE_BY_ZERO = (
        "integer-divide-by-zero",
        SecurityBugType.DENIAL_OF_SERVICE,
    )
    FLOAT_DIVIDE_BY_ZERO = ("float-divide-by-zero", SecurityBugType.DENIAL_OF_SERVICE)
    INVALID_BUILTIN = ("invalid-builtin-use", SecurityBugType.DENIAL_OF_SERVICE)
    INVALID_OBJC_CAST = ("invalid-objc-cast", SecurityBugType.DENIAL_OF_SERVICE)
    IMPLICIT_UNSIGNED_INTEGER_TRUNCATION = (
        "implicit-unsigned-integer-truncation",
        SecurityBugType.DENIAL_OF_SERVICE,
    )
    IMPLICIT_SIGNED_INTEGER_TRUNCATION = (
        "implicit-signed-integer-truncation",
        SecurityBugType.DENIAL_OF_SERVICE,
    )
    IMPLICIT_INTEGER_SIGN_CHANGE = (
        "implicit-integer-sign-change",
        SecurityBugType.DENIAL_OF_SERVICE,
    )
    IMPLICIT_SIGNED_INTEGER_TRUNCATION_OR_SIGN_CHANGE = (
        "implicit-signed-integer-truncation-or-sign-change",
        SecurityBugType.DENIAL_OF_SERVICE,
    )
    INVALID_SHIFT_BASE = ("invalid-shift-base", SecurityBugType.DENIAL_OF_SERVICE)
    INVALID_SHIFT_EXPONENT = (
        "invalid-shift-exponent",
        SecurityBugType.DENIAL_OF_SERVICE,
    )
    OUT_OF_BOUNDS_INDEX = ("out-of-bounds-index", SecurityBugType.DENIAL_OF_SERVICE)
    UNREACHABLE_CALL = ("unreachable-call", SecurityBugType.DENIAL_OF_SERVICE)
    MISSING_RETURN = ("missing-return", SecurityBugType.DENIAL_OF_SERVICE)
    NON_POSITIVE_VLA_INDEX = (
        "non-positive-vla-index",
        SecurityBugType.DENIAL_OF_SERVICE,
    )
    FLOAT_CAST_OVERFLOW = ("float-cast-overflow", SecurityBugType.DENIAL_OF_SERVICE)
    INVALID_BOOL_LOAD = ("invalid-bool-load", SecurityBugType.DENIAL_OF_SERVICE)
    INVALID_ENUM_LOAD = ("invalid-enum-load", SecurityBugType.DENIAL_OF_SERVICE)
    FUNCTION_TYPE_MISMATCH = (
        "function-type-mismatch",
        SecurityBugType.DENIAL_OF_SERVICE,
    )
    INVALID_NULL_RETURN = ("invalid-null-return", SecurityBugType.DENIAL_OF_SERVICE)
    INVALID_NULL_ARGUMENT = ("invalid-null-argument", SecurityBugType.DENIAL_OF_SERVICE)
    DYNAMIC_TYPE_MISMATCH = ("dynamic-type-mismatch", SecurityBugType.DENIAL_OF_SERVICE)
    CFI_BAD_TYPE = ("cfi-bad-type", SecurityBugType.DENIAL_OF_SERVICE)


# Sourced from https://fburl.com/he08zdot
class SignalCrashType(CrashType):
    FPE = ("FPE", SecurityBugType.DENIAL_OF_SERVICE)
    ILL = ("ILL", SecurityBugType.DENIAL_OF_SERVICE)
    ABRT = ("ABRT", SecurityBugType.DENIAL_OF_SERVICE)
    SEGV = ("SEGV", SecurityBugType.DENIAL_OF_SERVICE)
    BUS = ("BUS", SecurityBugType.DENIAL_OF_SERVICE)
    TRAP = ("TRAP", SecurityBugType.DENIAL_OF_SERVICE)


class FrameKind(enum.Enum):
    def __new__(cls, _) -> "FrameKind":
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__) + 1
        return obj

    def __init__(self, pattern: str) -> None:
        self.pattern: Pattern[str] = re.compile(pattern)

    # Native
    # For afl-qemu harness, the entry function can be either afl_qemu_driver_stdin_input or LLVMFuzzerTestOneInput
    # See AFLplusplus/utils/aflpp_driver/aflpp_qemu_driver.c
    FUZZER_ENTRYPOINT = (
        r"LLVMFuzzerTestOneInput|rust_fuzzer_test_input|afl_qemu_driver_stdin_input"
    )
    LIBFUZZER = r" in fuzzer::"
    GLOG = r" in google::LogMessage"
    LIBC = r"/glibc/|libc.so.6|/libstdc\+\+"
    LIBC_ABORT = r"__GI_abort|__GI_raise"
    LIBC_INITIALIZATION = r"__libc_start_main| _start .*sysdeps\/x86_64\/start.S"
    FOLLY_THROW_EXCEPTION = r"folly::.*throw_exception"
    FOLLY_COROUTINE = r"folly::coro::.*| folly::resumeCoroutine.*"


@dataclass
class Frame(abc.ABC):
    raw_frame: str
    function: str
    file: str
    lineno: Optional[int]
    colno: Optional[int]

    @cached_property
    def frame_kinds(self) -> Set[FrameKind]:
        # A frame can be of multiple kinds, e.g. LIBC and LIBC_ABORT
        return {kind for kind in FrameKind if kind.pattern.search(self.raw_frame)}

    @classmethod
    @abc.abstractmethod
    def create(cls, raw_frame: str) -> Optional["Frame"]:
        # Factory method
        ...

    @property
    @abc.abstractmethod
    def identifier(self) -> Identifier: ...


class Crash(abc.ABC):
    def __init__(self, output: str) -> None:
        self._output = output

    @property
    def output(self) -> str:
        return self._output

    @cached_property
    def output_lines(self) -> List[str]:
        return self.output.split("\n")

    @property
    @abc.abstractmethod
    def has_stacktrace(self) -> bool: ...

    @property
    @abc.abstractmethod
    def is_symbolicated(self) -> bool: ...

    @property
    @abc.abstractmethod
    def crash_type(self) -> CrashType: ...

    @property
    @abc.abstractmethod
    def summary(self) -> str: ...


class CrashPattern:
    STACKTRACE_START: Pattern[str] = re.compile(
        r"^\w+: .*Assertion.*failed.*$|"
        r"\b[\w.]+:\d*\] Check failed:.*$|"
        r"(^==\d+== ?WARNING.*$)|"
        r"(^==\d+== ?ERROR.*$)|"
        r"^Program received signal SIG",
        flags=re.MULTILINE,
    )

    HASHED_BUCK_OUT_PATH: Pattern[str] = re.compile(
        r"(?P<stem>buck-out\/.*\/gen\/(fbsource\/)?)(?P<hash>[0-9a-f]+\/)"
    )

    ASAN_ALLOC_DEALLOC_MISMATCH: Pattern[str] = re.compile(
        r"AddressSanitizer: alloc-dealloc-mismatch \((?P<mismatch>.* vs .*)\)"
    )
    ASAN_NEGATIVE_SIZE_PARAM: Pattern[str] = re.compile(
        r"AddressSanitizer: negative-size-param: \((?P<size>size=-\d+)\)"
    )
    ASAN_OOM: Pattern[str] = re.compile(
        r"AddressSanitizer: allocator is out of memory trying to allocate (?P<size>0x[0-9a-f]+) bytes"
    )
    ASAN_READ_WRITE_SIZE: Pattern[str] = re.compile(
        r"(?P<type>(READ)|(WRITE)) of (?P<size>size \d+)"
    )
    ASAN_SCARINESS: Pattern[str] = re.compile(
        r"SCARINESS: (?P<scariness>\d+) \((?P<size>\S+-byte-)?(?P<crash_type>.*)\)",
    )
    ASSERT_FAIL: Pattern[str] = re.compile(r"^\w+: .*Assertion.*failed.*$")
    CHECK_FAIL_WITH_DIFFERENCE: Pattern[str] = re.compile(
        r"\b[\w.]+:\d*\] Check failed:(?P<check>.*)\((?P<difference>.* vs. .*)\).*$",
        flags=re.MULTILINE,
    )
    CHECK_FAIL_WITHOUT_DIFFERENCE: Pattern[str] = re.compile(
        r"\b[\w.]+:\d*\] Check failed:(?P<check>.*).*$",
        flags=re.MULTILINE,
    )
    FOLLY_UNINITIALIZED_SINGLETON: Pattern[str] = re.compile(
        r"folly::detail::singletonWarnCreateBeforeRegistrationCompleteAndAbort"
    )
    FUZZ_TARGET_EXITED: Pattern[str] = re.compile(r"libFuzzer: fuzz target exited")
    GDB_SIGNAL: Pattern[str] = re.compile(
        r"Program received signal SIG(?P<crash_type>[\w-]+)"
    )
    LIBFUZZER_DEADLY_SIGNAL: Pattern[str] = re.compile(
        r"^.*?ERROR:\s*?libFuzzer:\s*?deadly signal",
        flags=re.MULTILINE,
    )
    LIBFUZZER_OOM_LEAK: Pattern[str] = re.compile(
        r"==\d+== ERROR: libFuzzer: out-of-memory \((?P<usage>used: .*; limit: .*)\)"
    )
    LIBFUZZER_OOM_MALLOC: Pattern[str] = re.compile(
        r"==\d+== ERROR: libFuzzer: out-of-memory \(malloc\((?P<size>\d+)\)\)"
    )
    LSAN_DIRECT_LEAK: Pattern[str] = re.compile(
        r"Direct leak of \d+ byte\(s\) in \d+ object\(s\) allocated from:"
    )
    LSAN_INDIRECT_LEAK: Pattern[str] = re.compile(
        r"Indirect leak of \d+ byte\(s\) in \d+ object\(s\) allocated from:"
    )
    LSAN_LEAK_SIZE: Pattern[str] = re.compile(
        r"SUMMARY: AddressSanitizer: (?P<size>\d+ byte\(s\) leaked in \d+ allocation\(s\))"
    )
    MSAN_UNITIALIZED_VALUE_ALLOCATION: Pattern[str] = re.compile(
        r"Uninitialized value was created by an allocation of"
    )
    REPRO_STEPS: Pattern[str] = re.compile(
        r"====BEGIN REPRO STEPS====\n(?P<repro_steps>.*)====END REPRO STEPS====",
        flags=re.DOTALL,
    )
    REPRO_STEPS_COMMAND: Pattern[str] = re.compile(
        r"\* CLI repro command:\s*(?P<command>.*?$)", re.MULTILINE
    )
    REPRO_STEPS_FILE: Pattern[str] = re.compile(r"(?<=\* Wrote file ')([^']*)(?=')")
    SANITIZER_SUMMARY: Pattern[str] = re.compile(
        r"SUMMARY: \w+Sanitizer: (?P<crash_type>[\w-]+)"
    )
    TERMINATE: Pattern[str] = re.compile(
        r"terminate called[\S\s]*what\(\):\s*(?P<reason>.*)\n"
    )
    THREAD_LIMIT_EXCEEDED: Pattern[str] = re.compile(
        r"AddressSanitizer: Thread limit \(\d+ threads\) exceeded."
    )
    TIMEOUT: Pattern[str] = re.compile(
        r"^.*?ERROR:\s*?libFuzzer:\s*?timeout",
        flags=re.MULTILINE,
    )
    UNCAUGHT_EXCEPTION: Pattern[str] = re.compile(
        r"ERROR:\s*libFuzzer:\s*deadly signal[\S\s]*"
        r"#\d+ .* in std::terminate\(\) .*$\n\s*#\d+ .* in __cxa_(re)?throw",
        flags=re.MULTILINE,
    )


# Map of patterns to the crash types they identify. The order here matters,
# e.g. we look for CUSTOM_INVARIANT_FAILURE before CHECK_FAILURE
CRASH_PATTERN_CRASH_TYPES: List[Tuple[Pattern[str], CrashType]] = [
    (
        CrashPattern.FOLLY_UNINITIALIZED_SINGLETON,
        GenericCrashType.FOLLY_UNINITIALIZED_SINGLETON,
    ),
    (CrashPattern.FUZZ_TARGET_EXITED, GenericCrashType.FUZZ_TARGET_EXITED),
    (CrashPattern.LIBFUZZER_OOM_MALLOC, GenericCrashType.LARGE_MALLOC),
    (CrashPattern.LIBFUZZER_OOM_LEAK, GenericCrashType.MEMORY_LEAK),
    (CrashPattern.TIMEOUT, GenericCrashType.TIMEOUT),
    (CrashPattern.UNCAUGHT_EXCEPTION, GenericCrashType.UNCAUGHT_EXCEPTION),
    (CrashPattern.CHECK_FAIL_WITH_DIFFERENCE, GenericCrashType.CHECK_FAILURE),
    (CrashPattern.CHECK_FAIL_WITHOUT_DIFFERENCE, GenericCrashType.CHECK_FAILURE),
    (CrashPattern.ASSERT_FAIL, GenericCrashType.ASSERTION_FAILURE),
    (CrashPattern.THREAD_LIMIT_EXCEEDED, ASanCrashType.THREAD_LIMIT_EXCEEDED),
    (CrashPattern.LSAN_DIRECT_LEAK, LSanCrashType.DIRECT_LEAK),
    (CrashPattern.LSAN_INDIRECT_LEAK, LSanCrashType.INDIRECT_LEAK),
]

# Map of patterns to functions that extract formatted details about the crash
# from their matches.
# Order also matters here, similar to the above list.
CRASH_PATTERN_DETAILS: List[Tuple[Pattern[str], Callable[[Match[str]], str]]] = [
    (CrashPattern.ASAN_ALLOC_DEALLOC_MISMATCH, lambda m: m["mismatch"]),
    (CrashPattern.ASAN_NEGATIVE_SIZE_PARAM, lambda m: m["size"]),
    (CrashPattern.ASAN_READ_WRITE_SIZE, lambda m: m["size"]),
    (CrashPattern.ASAN_OOM, lambda m: f"allocating {int(m['size'], 0)} bytes"),
    (
        CrashPattern.CHECK_FAIL_WITH_DIFFERENCE,
        lambda m: f"{m['check'].strip()}, {m['difference'].strip()}",
    ),
    (CrashPattern.CHECK_FAIL_WITHOUT_DIFFERENCE, lambda m: f"{m['check'].strip()}"),
    (CrashPattern.LSAN_LEAK_SIZE, lambda m: m["size"]),
    (CrashPattern.LIBFUZZER_OOM_MALLOC, lambda m: f"{m['size']} bytes"),
    (CrashPattern.LIBFUZZER_OOM_LEAK, lambda m: m["usage"]),
    (CrashPattern.TERMINATE, lambda m: m["reason"]),
]


@dataclass
class CliReproSteps:
    command: Optional[str]
    files: List[Path]


@dataclass
class NativeFrame(Frame):
    index: int
    is_symbolicated: bool

    _RE_FRAME: Pattern[str] = re.compile(
        r"^\s*#(?P<index>\d+) \dx[0-9a-f]+( in )?"
        # Identify functions by consuming everything up until the final space,
        # excluding '/' directory separators
        r"(?P<function>[^/]+) "
        r"("
        # Identify an unsymbolicated file by looking for contents wrapped in
        # parantheses with an offset
        r"\((?P<unsymbolicated_file>.+)\+\dx[0-9a-f]+\)"
        # OR
        r"|"
        # Identify a symbolicated file by looking for contents with a file
        # extension, optionally with line and column numbers
        r"(?P<symbolicated_file>.+\.\w+?)(:(?P<lineno>\d+)(:(?P<colno>\d+))?)?"
        r")$"
    )

    @classmethod
    def create(cls, raw_frame: str) -> Optional["NativeFrame"]:
        if m := NativeFrame._RE_FRAME.match(raw_frame.strip()):
            return NativeFrame(
                raw_frame=raw_frame,
                index=int(m["index"]),
                function=m["function"],
                file=m["symbolicated_file"] or m["unsymbolicated_file"],
                is_symbolicated=bool(m["symbolicated_file"]),
                lineno=int(m["lineno"]) if m["lineno"] else None,
                colno=int(m["colno"]) if m["colno"] else None,
            )

    @cached_property
    def identifier(self) -> Identifier:
        identifier_file = self.file
        identifier_function = self.function

        # Don't include filenames for unsymbolicated frames from within the
        # harness, as their paths contain a hash of the binary that won't be
        # stable across crashes.
        if not identifier_file or (
            not self.is_symbolicated and identifier_file.startswith("/harnesses/")
        ):
            return f"[{identifier_function}]"

        # Strip configuration hashes from buck-out paths, which may not be
        # stable across builds (https://fburl.com/9kjkydd4)
        identifier_file = re.sub(
            CrashPattern.HASHED_BUCK_OUT_PATH, lambda m: m["stem"], identifier_file
        )

        return f"[{identifier_function}|{identifier_file}]"

    @cached_property
    def presentable_frame(self) -> str:
        lineno = f":{self.lineno}" if self.lineno else ""
        return (
            f"{self.get_presentable_function()}\n"
            f"    ({self.get_presentable_filename()}{lineno})"
        )

    def get_presentable_function(self, aggressive: bool = False) -> str:
        function = self.function

        # Return a list of inclusive ranges enclosing text with the given
        # delimiters at the specified nesting depth
        def _find_enclosed_range(
            s: str, opening_delim: str, closing_delim: str, depth: int
        ) -> List[Tuple[int, int]]:
            ranges: List[Tuple[int, int]] = []
            current_depth = 0
            index_opening = 0
            for index, char in enumerate(s):
                if char == opening_delim:
                    if current_depth == depth:
                        index_opening = index
                    current_depth += 1
                elif char == closing_delim:
                    current_depth -= 1
                    if current_depth == depth:
                        index_closing = index
                        ranges.append((index_opening, index_closing))
            return ranges

        # Replace <...> with <>, stripping all template types
        for start, end in reversed(_find_enclosed_range(function, "<", ">", 0)):
            function = "{}{}".format(function[: start + 1], function[end:])

        # Replace (...) with (), stripping nested parameters or all parameters if
        # `aggressive` specified,
        for start, end in reversed(
            _find_enclosed_range(function, "(", ")", 0 if aggressive else 1)
        ):
            function = "{}{}".format(function[: start + 1], function[end:])

        # Strip namespaces for top level parameters
        for start, end in reversed(_find_enclosed_range(function, "(", ")", 0)):
            parameters = function[start + 1 : end].split(",")
            stripped_parameters = [p.strip().rpartition("::")[2] for p in parameters]
            function = "{}{}{}".format(
                function[: start + 1], ", ".join(stripped_parameters), function[end:]
            )

        if aggressive:
            function = sanitize_function_name_aggressively(function)
            # Strip trailing qualifiers (e.g. const, volatile)
            end_index = function.rfind(")")
            if end_index != -1:
                function = function[: end_index + 1]

        return function

    def get_presentable_filename(self) -> str:
        # Remove repeating forward slashes from std::panicking frames for rust crashes
        return re.sub(r"/{2,}", r"/", self.file)

    def is_fuzz_lib_invocation(self) -> bool:
        return FUZZ_LIB_INVOCATION_PATTERN in self.frame_kinds


@dataclass
class NativeStacktrace:
    header: str
    frames: List[NativeFrame]

    @cached_property
    def is_symbolicated(self) -> bool:
        # We need to disregard libc initialization lines, which are always
        # symbolicated
        return any(
            frame.is_symbolicated
            for frame in self.frames
            if FrameKind.LIBC_INITIALIZATION not in frame.frame_kinds
        )

    @cached_property
    def minimized_stacktrace(self) -> str:
        return "\n".join(frame.raw_frame.strip() for frame in self.minimized_frames)

    @cached_property
    def minimized_frames(self) -> List[NativeFrame]:
        minimized_frames = list(self.frames)

        # Filter frames before entrypoint
        entrypoint_file = None
        for index, frame in enumerate(minimized_frames):
            if FrameKind.FUZZER_ENTRYPOINT in frame.frame_kinds:
                del minimized_frames[index + 1 :]
                entrypoint_file = frame.file
                break

        # Filter frames that correspond to fuzzing library or utility
        # (everything between the invocation frame and when we first re-enter
        # the harness file, inclusive)
        fuzzing_lib_invocation_start = None
        fuzzing_lib_invocation_end = None
        for index, frame in enumerate(minimized_frames):
            # Find the first harness file frame after the invocation
            if frame.file == entrypoint_file:
                fuzzing_lib_invocation_end = index
            if frame.is_fuzz_lib_invocation():
                fuzzing_lib_invocation_start = index
                break
        if fuzzing_lib_invocation_start and fuzzing_lib_invocation_end:
            del minimized_frames[
                fuzzing_lib_invocation_end : fuzzing_lib_invocation_start + 1
            ]

        # Filter other irrelevant frames
        return [
            frame
            for frame in minimized_frames
            if not {
                FrameKind.LIBFUZZER,
                FrameKind.GLOG,
                FrameKind.LIBC,
                FrameKind.LIBC_ABORT,
                FrameKind.FOLLY_THROW_EXCEPTION,
                FrameKind.FOLLY_COROUTINE,
            }
            & frame.frame_kinds
        ]

    @cached_property
    def symbolicated_minimized_frames(self) -> List[NativeFrame]:
        return [frame for frame in self.minimized_frames if frame.is_symbolicated]


class NativeCrash(Crash):
    @cached_property
    def stacktraces(self) -> List[NativeStacktrace]:
        stacktraces: List[NativeStacktrace] = []
        prev_line = ""
        for line in self.output_lines:
            # If the output has multiple crashes (with multiple errors), assume
            # the first few are recoverable and can be ignored. Keep the last.
            if CrashPattern.STACKTRACE_START.search(line) is not None:
                stacktraces.clear()
            if frame := NativeFrame.create(line):
                # Detect new stacktraces by looking for frames with an index of 0
                if frame.index == 0 or not stacktraces:
                    stacktraces.append(
                        NativeStacktrace(
                            header=prev_line.strip(),
                            frames=[frame],
                        )
                    )
                else:
                    stacktraces[-1].frames.append(frame)
            prev_line = line
        return stacktraces

    @property
    def has_stacktrace(self) -> bool:
        return len(self.stacktraces) > 0

    @cached_property
    def is_symbolicated(self) -> bool:
        return any(stacktrace.is_symbolicated for stacktrace in self.stacktraces)

    @cached_property
    def crash_type(self) -> CrashType:
        for pattern, crash_type in CRASH_PATTERN_CRASH_TYPES:
            if pattern.search(self.output):
                return crash_type

        if CrashPattern.LIBFUZZER_DEADLY_SIGNAL.search(self.output):
            return (
                GenericCrashType.ASSERTION_FAILURE
                if any(
                    FrameKind.LIBC_ABORT in frame.frame_kinds
                    for stacktrace in self.stacktraces
                    for frame in stacktrace.frames
                )
                else GenericCrashType.DEADLY_SIGNAL
            )

        if scariness_match := CrashPattern.ASAN_SCARINESS.search(self.output):
            if crash_type := ASanCrashType.from_value(scariness_match["crash_type"]):
                return crash_type

        if summary_match := CrashPattern.SANITIZER_SUMMARY.search(self.output):
            crash_type_val = summary_match["crash_type"]
            for crash_type_cls in (MSanCrashType, UBSanCrashType, SignalCrashType):
                if crash_type := crash_type_cls.from_value(crash_type_val):
                    return crash_type

            # If asan scariness was missing, fall back to parsing the asan crash
            # type from the summary line. Unlike the crash type specified in the
            # scariness line, the crash type here won't include a read/write
            # prefix so we need to add it manually
            if rw_match := CrashPattern.ASAN_READ_WRITE_SIZE.search(self.output):
                rw_type = rw_match["type"].lower()
                crash_type_val = f"{rw_type}-{crash_type_val}"
            if crash_type := ASanCrashType.from_value(crash_type_val):
                return crash_type

        return GenericCrashType.UNKNOWN

    @cached_property
    def summary(self) -> str:
        summary = (
            "unknown-crash"
            if self.crash_type is GenericCrashType.UNKNOWN
            else self.crash_type.value
        )

        for pattern, detail_fn in CRASH_PATTERN_DETAILS:
            if match := pattern.search(self.output):
                details = detail_fn(match)
                # Sanitize crash output
                details = "".join(
                    char if char.isprintable() else REPLACEMENT_CHARACTER
                    for char in details
                )
                summary = f"{summary} ({details})"
                break

        if self.has_stacktrace and self.stacktraces[0].symbolicated_minimized_frames:
            func = (
                self.stacktraces[0]
                .symbolicated_minimized_frames[0]
                .get_presentable_function(aggressive=True)
            )
            summary = f"{summary} in {func}"

        return summary


def minimize_frames(frames: List[NativeFrame]) -> List[NativeFrame]:
    pattern = re.compile(r"fbcode/third-party-buck/")
    return [
        frame
        for frame in frames
        # third-party-buck paths are pre-built binaries that can't be patched
        if not pattern.search(frame.raw_frame)
    ]
