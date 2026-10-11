# GoC design and implementation plan

GoC implements GPU arithmetic on a CPU for use by emulators. A call executes one
instruction for an entire wave, synchronously on the calling CPU thread. GoC
provides no scheduling, worker threads, synchronization, instruction decoder, or
GPU memory system. The caller manages the emulated machine and supplies decoded
operands and state.

## Scope and current support

The implementation targets RDNA3 and RDNA4 register arithmetic: scalar/vector
integer and floating operations, conversions, comparisons, packed operations,
DOT, WMMA/SWMMAC, and register routing. Selected guest-register operations are
also included. Memory traffic, barriers, scheduling, and general execution of GPU
binaries are outside this library's scope.

API presence does not promise every encoding, guest mode, wave size, bit-exact
result, or optional state update. The current per-instruction contracts live with
the declarations in
[`goc_instructions.h`](https://github.com/bjacob/goc/blob/main/include/goc/detail/goc_instructions.h).
Keep those comments and their tests authoritative instead of maintaining a second
instruction inventory here. In particular, loose support is broader than exact
support, and exact numerical results do not imply exact exception reporting.

## Public API

Include [`goc/goc.h`](https://github.com/bjacob/goc/blob/main/include/goc/goc.h).
The headers compile independently as C99 and C++17 and expose a C ABI. GoC is a
static library with position-independent code and hidden visibility, suitable for
embedding in a consumer's shared library without exporting GoC symbols. The
CMake target is `goc`.

Names follow `goc_<mnemonic>`. Shared semantics use a shared entry point. When a
newer architecture changes a mnemonic's semantics, append its architecture name,
for example `_rdna4`. Append `_wave64` last for dedicated Wave64 forms; the
default vector width is Wave32. The six WMMA mnemonics common to RDNA3 and RDNA4
use unsuffixed RDNA3 forms and suffixed RDNA4 forms.

### Representative instruction

```c
int goc_v_wmma_f32_16x16x16_f16_rdna4(
    uint64_t flags,
    uint64_t instruction_flags,
    uint32_t *const *d,
    const uint32_t *const *a,
    const uint32_t *const *b,
    const uint32_t *const *c);
```

This computes RDNA4 Wave32 FP16 matrix multiplication with FP32 accumulation.
`a` and `b` each point to four VGPR pointers; `c` and `d` each point to eight.
Each VGPR contains 32 contiguous `uint32_t` lane words. Factors are packed FP16
bits and accumulator/result words are FP32 bits, in the ISA's lane layout.
VGPRs need not be adjacent in CPU memory. WMMA ignores architectural EXEC,
including zero, so this API has no `exec_mask` parameter.

See [`wmma.cpp`](https://github.com/bjacob/goc/blob/main/src/wmma.cpp) and
[`wmma_test.cpp`](https://github.com/bjacob/goc/blob/main/tests/wmma_test.cpp)
for implementation and operand-layout examples.

### Operands, aliasing, and errors

- `flags` combines CPU selection, semantics, and guest FP state.
  `instruction_flags` holds decoded instruction modifiers such as NEG, ABS,
  half selection, CLAMP, and DPP controls, where supported.
- Explicit operands follow assembly order. Scalar inputs are passed by value;
  scalar outputs use pointers. Implicit scalar inputs follow explicit operands.
- VGPR inputs use `const uint32_t *const *`; outputs use `uint32_t *const *`.
  Words represent raw bits, avoiding nonstandard FP types in the ABI.
- Whole-VGPR aliases are supported, including shared VGPRs within multi-register
  operands. Reads observe original source values before destination writes.
  Distinct VGPR addresses must not partially overlap. Follow instruction-specific
  contracts for duplicate destinations and scalar-output overlap.
- Instructions affected by EXEC take `uint32_t exec_mask` for Wave32 or
  `uint64_t exec_mask` for Wave64. Inactive vector destinations are preserved.
  Scalar, pseudo-scalar, and matrix instructions that ignore EXEC omit it.
  Ordinary comparison/carry masks are required results and follow their own
  inactive-bit rules; they are not optional global-state outputs.
- Return `GOC_SUCCESS` on success. Validate before writing: every error leaves
  all outputs unchanged. Errors distinguish invalid flags, unsupported semantics,
  and unsupported requested global-state updates. Pointer/storage requirements
  are caller preconditions unless a particular contract explicitly allows null.

### CPU and semantics selection

[`goc_common.h`](https://github.com/bjacob/goc/blob/main/include/goc/detail/goc_common.h)
defines typed `static const` constants. CPU levels and semantics are enumerations,
not independently OR-able feature sets. C99 integer-constant-expression support
is not required for these names; consider C23 only if that need arises.

`goc_init_cpu_flags()` returns capabilities usable by both the CPU and OS, with
all other bits zero. OR its result with semantics and guest-state flags. Callers
may select a lower level for testing but must not claim unavailable features.
The CPU field reserves bits 0–15; implemented levels are baseline, x86-64-v3,
x86-64-v4, and Zen4. SIMD availability also depends on what was compiled.

`GOC_SEMANTICS_LOOSE` permits mathematically sound approximations, including
CPU floating arithmetic with different rounding stages. It does not excuse
wrong operands, routing, stage-dependent overflow, or omitted operations.
`GOC_SEMANTICS_EXACT_EMPIRICAL` requests empirically bit-exact behavior in the
supported domain. Semantics values 2 and 3 are reserved.

Without `GOC_SEMANTICS_STRICT`, unavailable exact result semantics may fall back
to loose. With it, return `GOC_ERROR_UNSUPPORTED_SEMANTICS`. Optional exact
state reporting has its own stricter rule below.

### Guest and host floating-point state

Guest flags currently include `GOC_FP16_OVFL`,
`GOC_FP_FLUSH_INPUT_DENORMALS`, `GOC_FP_FLUSH_OUTPUT_DENORMALS`,
`GOC_FP_IEEE_MODE`, and `GOC_FP_DX10_CLAMP_OFF`. Availability and effects are
instruction- and architecture-specific. There is no general guest directed-rounding
selector. Zero selects the default policy; it does not universally mean the
hardware preserves denormals.

Unless a contract says otherwise, assume host round-to-nearest, ties-to-even,
and preserved denormals for loose arithmetic. Production API implementations
must never explicitly mutate host rounding, exception masks, denormal controls,
or sticky exception flags, even temporarily. No save/modify/restore helpers.
Ordinary arithmetic may raise host sticky flags. Tests may change host controls.
This prohibition takes priority over borrowing upstream implementations.

### Implicit architectural register outputs

Append one nullable native-width pointer per implicit register the instruction
may write, named after that register. Do not duplicate an explicit output or add
parameters for registers the instruction does not write. RDNA3 exception output
uses `uint32_t *trapsts`; RDNA4 uses `uint32_t *excp_flag_user`. Their arithmetic
bits share the `GOC_EXCEPTION_*` masks, not their complete register layouts.
Exception bits accumulate with OR; only participating lanes contribute. Other
registers follow their own ISA update rules. `NULL` opts out.

Faithful optional state updates are required only for exact requests with a
non-null pointer, independently of `GOC_SEMANTICS_STRICT`. If unavailable, return
`GOC_ERROR_UNSUPPORTED_GLOBAL_STATE` without changing any output. Exact numerical
support alone is insufficient. Do not derive guest exceptions by sampling host
FP state.

Loose semantics makes no promise about optional state, even with a non-null
pointer. SIMD paths are expected to skip reporting and leave it unchanged;
inexpensive existing updates may remain, but callers cannot rely on them.
Loose mode never returns `GOC_ERROR_UNSUPPORTED_GLOBAL_STATE`. Deriving hardware
exception history from approximate arithmetic would not be faithful and must
not add reporting overhead or force a scalar fallback. Required instruction
results, including comparison masks and SCC, remain required in every mode.

## Implementation strategy

Use C++17 internally and sized integer types for raw bits and defined wrapping.
Use signed integers for ordinary indices. Keep algorithms explicit and readable;
share a helper when it centralizes a meaningful rule for existing callers, not
merely to reduce similar-looking syntax. Some local repetition is preferable to
policy scaffolding that obscures the arithmetic.

Borrow validated exact implementations from rocjitsu, adapting them without
host FP-control mutations. Simple ISA-derived exact integer/bit operations are
also appropriate. Developing nontrivial new numerical or exception models by
hardware reverse engineering is outside scope. If neither a compliant borrowed
model nor a simple specification-derived implementation is available, keep exact
support explicitly unavailable.

Portable implementations live in
[`src/`](https://github.com/bjacob/goc/tree/main/src); CPU-specialized translation
units live in [`src/x86_64/`](https://github.com/bjacob/goc/tree/main/src/x86_64).
GCC/Clang x86-64 builds enable paths after compiler-flag checks. Other targets and
compilers retain portable code. Keep CPU flags centralized in
[`src/x86_64/CMakeLists.txt`](https://github.com/bjacob/goc/blob/main/src/x86_64/CMakeLists.txt).
Prioritize x86-64-v3 and useful Zen4 paths; use x86-64-v4 where beneficial. Dispatch
must respect detected capabilities and compiled availability.

Supported modifiers and DPP should retain loose SIMD dispatch unless their
behavior is specifically unsuitable. Use masked stores where applicable.
Optimizing sparse EXEC masks is a non-goal: compute full results rather than
adding mask-density dispatch or active-lane algorithms. A zero-mask early return
is allowed. This is a performance policy, not a constant-time guarantee.

### Matrix arithmetic contracts

RDNA3 and RDNA4 have different factor/result layouts. Both dense FP16/BF16 and
integer WMMA support both wave sizes; RDNA4 also has native FP8/BF8 and sparse
forms. Follow the API comments for register counts, selector keys, and modifiers.

- RDNA3 floating WMMA chains eight borrowed DOT2 stages; packed outputs narrow
  each stage. Integer saturation is staged after pairs of IU8 products or quartets
  of IU4 products, not just at the final sum.
- RDNA4 native FP8/BF8 exact arithmetic uses DOT8 stages, not FP16 widening.
  Accepted dense C ABS/NEG aliases are ignored. Sparse forms use their compressed
  factor ordering; unsupported raw modifier aliases remain errors.
- Sparse FP16/BF16 uses four DOT4 stages. Packed outputs narrow after each stage
  in loose scalar/SIMD as well as exact execution: FP16 uses nearest-even and
  the finite-overflow policy; BF16 truncates.
- Floating matrix APIs accept but ignore guest denormal and IEEE controls.
  Packed FP16 still honors `GOC_FP16_OVFL`. Borrowed exact integer models avoid
  dependence on host rounding and FTZ/DAZ settings.

The models and compact matrix fixtures use rocjitsu at
[`ad2bb973ec4a3d254e705c5a8f8cc0d17bc1d034`](https://github.com/ROCm/rocm-systems/tree/ad2bb973ec4a3d254e705c5a8f8cc0d17bc1d034/emulation/rocjitsu).
Fixture comments retain source, hardware, MODE, seed, and encoding provenance.
Their domains bound empirical claims; they do not establish all-input correctness
or physical RDNA3.5 validation. Upstream raw encodings outside GoC's public
modifier/key contracts are not silently accepted to replay additional captures.

## Testing and maintenance

Every instruction needs meaningful API-level numerical or bitwise assertions,
not just linkage or successful return codes. Cover essential differences in
wave width, modifiers, EXEC, aliasing, packed halves, overflow, guest state,
error preservation, and available CPU paths. Balance coverage against test size,
latency, and maintenance cost.

Loose floating results use the NumPy-style rule
`abs(actual - expected) <= atol + rtol * abs(expected)` from
[`fuzzy_compare.h`](https://github.com/bjacob/goc/blob/main/tests/fuzzy_compare.h).
Use family/domain-specific empirically calibrated bounds, independent references,
and separate deterministic holdout inputs. Bounds live in
[`loose_tolerances.h`](https://github.com/bjacob/goc/blob/main/tests/loose_tolerances.h).
NaNs match NaNs and same-sign infinities match; check payloads, signed zeros,
untouched storage, masks, and flags separately when part of the contract.
Sampled error maxima are not universal accuracy guarantees.

Use fixed local PRNG seeds and fully specified engines or unsigned recurrences.
Avoid `std::default_random_engine` and standard distributions whose output is
implementation-defined. Test selection/order must not affect generated inputs.

Exact references use captured goldens, validated upstream models, or simple
ISA-derived proofs. Computed references are welcome when equivalence is established,
not assumed from mathematical plausibility or scalar/SIMD agreement. Keep
provenance beside fixtures and the maintained oracle records in
[`reference_reviews.json`](https://github.com/bjacob/goc/blob/main/tests/reference_reviews.json).
Compact corpus digests should have checkpoints or local witnesses for diagnosis.
Normal tests require neither GPU access nor a rocjitsu checkout.

Use GTest, found locally or fetched by CMake. Each `*_test.cpp` has its own
binary; CTest discovers individual cases. Compile public headers independently
as C and C++. Test static-library embedding/export behavior and run the benchmark
as a correctness smoke test with `--min-ms 0`.

Validate significant changes with Clang and GCC Release, portable-only builds,
and ASan/UBSan. Build outside the source tree. The maintained compiled-reference
checker is supplemental evidence, not proof of per-mode execution:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/testing_audit.py ../goc-build \
  --output /tmp/goc-reference-inventory.json
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools -p '*_test.py'
```

## Priorities and limits

Maintain accurate per-entry-point contracts while extending useful RDNA3/RDNA4
coverage. Prioritize reliable loose SIMD, supported modifiers and routing,
borrowed exact models, and faithful optional state where already justified.
Do not claim universal exactness or complete global-state reporting. Guest directed
rounding, unsupported raw encodings, and some privileged HWREG side effects remain
outside current contracts. New architectures follow after these foundations;
FP4 support and new hardware reverse engineering are not current requirements.

Measure correctness changes that add work, especially packed intermediate
narrowing. Benchmark with CPU affinity and a quiet SMT sibling when practical;
report the workload and timing method rather than treating noisy local results
as portable performance promises. Keep historical measurements and audits in git
history, not as parallel descriptions of the current implementation.
