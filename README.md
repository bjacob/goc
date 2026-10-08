# GoC: GPU arithmetic on a CPU

GoC implements whole-wave GPU arithmetic through a synchronous C API. The public
header is `include/goc/goc.h`, usable from C99 and C++17. It directly includes
`goc/detail/goc_common.h` (flags, status codes, CPU detection) and
`goc/detail/goc_rdna4.h` (RDNA4 instructions and modifiers). API users include
`goc/goc.h`; detail headers are still checked for self-containment. This is an
initial RDNA4 implementation; `PLAN.md` describes the broader intended coverage.
GPU-specific implementation, test, and fixture filenames carry the architecture
name, such as `rdna4_wmma.cpp` and `rdna4_wmma_test.cpp`.

## Build and test

```sh
cmake -S . -B ../goc-build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ../goc-build --parallel "$(nproc)"
ctest --test-dir ../goc-build --parallel "$(nproc)" --output-on-failure
```

Both `GOC_STATIC` and `GOC_SHARED` default to `ON`, producing `libgoc.a` and
`libgoc.so` on Linux. Disable either with `-DGOC_STATIC=OFF` or
`-DGOC_SHARED=OFF`; at least one must remain enabled. CMake consumers link
`goc_static` for private embedding or `goc_shared` for dynamic linking.
Static consumers inherit
`GOC_STATIC_DEFINE`, so GoC's API stays hidden when embedded in their shared
library; shared consumers see exported API declarations. Both variants are
position-independent and keep implementation symbols hidden.

The public API and C linkage are tested against every enabled variant. Each
`*_test.cpp` has its own executable, such as `tests/goc_rdna4_wmma_test_static` and
`tests/goc_rdna4_wmma_test_shared`; the private decoder uses `tests/goc_cpu_decode_test`.
Public headers are compiled independently once per language (C and C++), without
linking. Private CPU decoding is tested separately. On Linux, export
tests also check the shared API and static embedding. `tests/cpuinfo` links to
`goc_static` and is only built and tested when `GOC_STATIC` is enabled.

CMake uses a system GTest when available and otherwise fetches GTest 1.17.0.
CMake automatically enables x86-64 implementations for x86-64 targets using GCC
or Clang. Optional CPU paths are compiled in separate translation units after
compiler-flag checks. Other compilers and architectures use the portable paths.

## Quick demonstration

After the Release build above, run these commands from the source directory:

```sh
../goc-build/tests/cpuinfo
ctest --test-dir ../goc-build --output-on-failure \
  -R 'HardwareCapturedExactResults|HardwareIntermediateOverflowState|Fp16V3|Bf16V3|Bf16CpuLevels|SubbyteWmma|Arithmetic'
../goc-build/tests/goc_rdna4_wmma_benchmark_static
```

The selected tests demonstrate FMA/LOG, WMMA numeric formats, hardware-captured
FP16/BF16 exactness, intermediate FP16 overflow state, and SIMD/scalar agreement.
The full suite also covers DOT2, NEG/NEG_HI, wave64, masks, aliasing, C linkage,
header self-containment and library exports.

The benchmark compares scalar loose FP16 with x86-64-v3, and scalar loose BF16
with both x86-64-v3 and the Zen4 AVX-512 BF16 path. Integer rows cover
INT8 K=16 and INT4 K=16/K=32 on scalar, v3 and Zen4 VNNI, with unsigned
wrapping and signed CLAMP workloads. All integer rows request strict exact
semantics and compare against independent integer goldens.
Floating-point exact scalar timings are listed separately;
Floating-point SIMD rows do not imply empirical bit-exactness. Unsupported or uncompiled SIMD
paths are explicitly skipped. A corresponding `_shared` executable is built
when `GOC_SHARED` is enabled; use it instead for a shared-only build and omit
`cpuinfo`, which is static-only.

Every path checks all outputs against independent dense matrix goldens before
and after timing. The floating-point workload uses fixed small-integer matrices
and compares no modifiers, `NEG_LO_A` alone, and a mixed case
(`NEG_HI_A | NEG_LO_B | ABS_C | NEG_C`). Modified floating-point rows use loose
semantics and independent integer matrix references.
Integer workloads use dense full-range factors and accumulators near overflow,
with signedness and CLAMP as labeled. All workloads use full EXEC, separate C/D
storage and hot buffers. Timings include public API
dispatch, input conversions and output stores. Each reported time is the median
of seven samples after warmup. Each path starts at 128 calls (overridable by the
positional argument) and doubles the count until the timed batch takes at least
10 ms. Shorter batches are discarded. Subsequent samples retain that count and
double again if necessary, so every accepted sample meets the minimum duration.
Set `GOC_BENCH_MIN_MS` to a nonnegative integer to override the minimum milliseconds,
for example `GOC_BENCH_MIN_MS=50 ../goc-build/tests/goc_rdna4_wmma_benchmark_static`.
CTest uses a 0 ms minimum and starts at one call per sample for its correctness
smoke check, with no speedup assertion. Zero disables the minimum-duration
requirement; warmup and output checks still run.

An illustrative local run on a Ryzen 9 7950X3D, Clang 21.1.8, Release, static
linking and the earlier fixed 10,000 calls/sample measurement produced:

| Input / semantics | CPU path | ns per wave | Speedup over same-format scalar loose |
| --- | --- | ---: | ---: |
| FP16 loose | Scalar | 13,497 | 1.0x |
| FP16 loose | x86-64-v3 | 206 | 65.4x |
| BF16 loose | Scalar | 8,951 | 1.0x |
| BF16 loose | x86-64-v3 | 191 | 47.0x |
| BF16 loose | Zen4 AVX-512 BF16 | 99 | 90.2x |
| FP16 exact | Scalar integer model | 22,000 | — |
| BF16 exact | Scalar integer model | 23,192 | — |

The Zen4 eligibility scan is vectorized. Previously a scalar scan made the
public BF16 call take about 400 ns even though its AVX-512 kernel took only
42–44 ns. The vectorized scan brings the public call to about 90–100 ns, versus
about 190 ns for v3 on this host, while retaining the same conservative fallback
rules. These figures include dispatch and the scan; the kernel-only measurement
was a separate diagnostic.

Integer SIMD measurements on the same host (Clang Release, static, CPU 8),
using the median of three benchmark invocations with a 50 ms minimum per sample:

| Shape | Mode | Scalar ns/wave | v3 ns/wave | Zen4 VNNI ns/wave |
| --- | --- | ---: | ---: | ---: |
| INT8/K16 | u/u wrap | 4,883.7 | 66.0 | 57.2 |
| INT8/K16 | s/s clamp | 7,726.6 | 84.3 | 72.2 |
| INT4/K16 | u/u wrap | 1,640.8 | 73.1 | 55.5 |
| INT4/K16 | s/s clamp | 5,730.5 | 88.7 | 73.1 |
| INT4/K32 | u/u wrap | 9,719.1 | 153.0 | 130.8 |
| INT4/K32 | s/s clamp | 15,027.5 | 195.7 | 134.4 |

The SMT sibling was online for these measurements; timing variation remains.
All integer results were checked against the independent dense goldens.

These are CPU instruction-emulation microbenchmarks, not end-to-end emulator
throughput or GPU comparisons. Results vary with host, compiler, workload and
system load; the scalar reference is intentionally simple.

## Implemented instructions

The entry points below use RDNA4 wave32. Names have the `goc_rdna4_` prefix.
The FP16/BF16 WMMA forms additionally have scalar wave64 variants named
`goc_rdna4w64_...`, supporting loose and empirical exact modes.

| Mnemonic | Loose semantics | Empirical exact semantics |
| --- | --- | --- |
| `v_fma_f32` | Scalar, AVX2/FMA, AVX-512 | Not implemented |
| `v_log_f32` | Scalar `log2` | Not implemented |
| `v_dot2_f32_f16` | Scalar | Integer arithmetic model |
| `v_dot2_f32_bf16` | Scalar | Integer arithmetic model |
| `v_wmma_f32_16x16x16_f16` | Scalar, x86-64-v3 F16C/AVX2/FMA | Integer arithmetic model |
| `v_wmma_f32_16x16x16_bf16` | Scalar, x86-64-v3 AVX2/FMA, AVX-512 BF16 | Integer arithmetic model |
| `v_wmma_f16_16x16x16_f16` | Integer arithmetic model | Integer arithmetic model |
| `v_wmma_bf16_16x16x16_bf16` | Integer arithmetic model | Integer arithmetic model |
| `v_wmma_f32_16x16x16_{fp8,bf8}_{fp8,bf8}` (all four combinations) | Scalar | Not implemented |
| `v_wmma_i32_16x16x16_iu8` | Scalar, x86-64-v3, Zen4 VNNI | Same exact integer paths |
| `v_wmma_i32_16x16x16_iu4` | Scalar, x86-64-v3, Zen4 VNNI | Same exact integer paths |
| `v_wmma_i32_16x16x32_iu4` | Scalar, x86-64-v3, Zen4 VNNI | Same exact integer paths |

The FP16/BF16 WMMA forms support all six NEG/NEG_HI modifier bits, including
C absolute value. FP8/BF8 WMMA supports C negation and absolute value; A/B
negation bits are rejected. Integer WMMA supports independent signed/unsigned
A/B inputs and signed output saturation (`GOC_WMMA_CLAMP`); without CLAMP,
results wrap modulo 2^32. Other instructions accept only zero instruction flags.
The FP8/BF8 and integer entry points currently support wave32 only. FP16 and
BF16 SIMD WMMA paths handle loose wave32 calls with FP32 outputs and all 64
combinations of `NEG_LO`/`NEG_HI` on A/B and `NEG`/`ABS` on C. `ABS_C` is applied before
`NEG_C`. These sign-bit transformations preserve the magnitude-only Zen4 BF16
eligibility checks; rejected inputs still fall back to v3 when available.
The v3 paths handle subnormals, infinities and NaNs through widening and CPU FMA;
NaN payloads are unspecified in loose mode. Zen4 first tries the AVX-512 BF16
path, falling back to v3 for exceptional values and extreme product exponents,
or to scalar if v3 was not compiled. Packed outputs, wave64 and floating-point
empirical exact modes use scalar arithmetic.
Integer WMMA uses SIMD in both semantics, for every signedness and CLAMP
combination.
Its v3 path uses signed 16-bit pairwise multiply-adds; Zen4 uses AVX-512 VNNI
word dot products. Unsigned bytes widen to positive 16-bit values, avoiding
saturating byte-pair operations. Each kernel computes the complete dot before
adding C and optionally saturating, including cancellation near int32 limits.

## Calling convention

Call `goc_init_cpu_flags()` and OR the result with semantics flags. CPU levels
are enumerations, not independent bits. A caller may select a lower CPU level
but must never claim unavailable capabilities. Dispatch also respects which
implementations were compiled.

Each VGPR pointer addresses 32 contiguous `uint32_t` lane words (64 for
`rdna4w64`). A multi-VGPR
operand is an array of these pointers; the backing arrays need not be adjacent
or SIMD-aligned. FP8/BF8 and INT8 A/B use two VGPRs each. INT4 A/B
use one each for K=16, or two for K=32; all these forms use eight C/D VGPRs.
FP16/BF16 A/B use four VGPRs and C/D use eight for the implemented WMMA
FP32-output wave32 forms; wave64 uses two A/B VGPRs and four C/D VGPRs.
Packed-output forms halve the C/D register counts, packing adjacent rows into
the low and high 16 bits. Input and output operands may share whole VGPRs. Distinct backing
addresses must not overlap, and every pointer must refer to sufficient storage.

GoC applies `exec_mask` to destination writes, including WMMA, as specified by
its API contract. Inactive destination lanes remain unchanged; source lanes are
not masked. Errors preserve all destination registers. No pointer-validation
or allocation ownership service is provided.

Zero semantics bits select loose numerical behavior. Add `GOC_SEMANTICS_EXACT_EMPIRICAL`
for the empirical model; also add `GOC_SEMANTICS_STRICT` to require support.
Unsupported exact requests otherwise fall back to loose semantics. Unassigned
instruction/general flag bits are rejected, except reserved semantics values
which follow the same fallback policy. `GOC_FP16_OVFL` emulates GPU MODE.FP16_OVFL: finite FP16 overflow saturates
to signed 65504 instead of infinity. Input infinities remain infinite, and
BF16/FP32 instructions ignore this state. Packed results narrow after each
four-product step. The packed path and empirical exact paths use integer
arithmetic, preserving the caller's host rounding mode and exception flags.
Other loose paths require nearest-even rounding and denormals enabled.

## Validation and provenance

Tests exercise the C ABI, CPU/OS feature gating, forced usable CPU levels,
unaligned and noncontiguous backing storage, masks, operand overlap, strict
errors, sign modifiers, and deterministic dense matrix golden outputs.

The empirical RDNA4 DOT/WMMA model in `src/rdna4_dot.h` is adapted from
rocjitsu's `isa/arch/amdgpu/shared/gfx12_dot.h`. The hardware fixtures in
`tests/rdna4_dot_fixtures.h` and `tests/rdna4_wmma_fixtures.h` come from rocjitsu's
`tests/fixtures/float_dot/gfx1201_cases.h`: Radeon AI Pro R9700 (`gfx1201`),
TheRock `10.2.0a20260916`. They contain 121 DOT and 24 WMMA captures. No new
GPU measurements or reverse engineering were performed for this implementation.
The empirical qualification is limited to that existing model and evidence.
Packed-output support additionally borrows `shared/dot_packed16.h` and the
RDNA4 subset of `tests/fixtures/float_dot/packed_wmma_cases.h`: 28 captured
16x16 outputs across both formats and wave sizes, including subnormals,
cancellation, NaNs and rare accumulator-alignment boundaries. Tests run these
under all four host rounding modes with preexisting FP exception flags.
Another 128 modifier captures come from `float_dot/packed_operand_cases.h`.
The intermediate-overflow tests are adapted from rocjitsu's
`PackedWmma.HardwareOverflowModeAtIntermediateSteps`; they distinguish saturation
at each four-product step from saturation applied only to the final result.

FP8/BF8 conversions and integer WMMA borrow from rocjitsu's
`util/data_types.h` and `shared/mma_exec.h`. FP8 uses OCP E4M3FN (finite through
448); BF8 uses OCP E5M2 (with infinities), not the FNUZ encodings. Tests exercise
all 256 codes through the public API and use deterministic dense mathematical
goldens for every FP8/BF8 pairing and integer sign/clamp combination. These
are mathematical checks, not new hardware evidence for exact FP8 accumulation.
Floating-point SIMD modifier tests cover all 64 combinations across every usable
CPU level, masks, overlapping operands and noncontiguous/unaligned storage.
Special-value tests include signed zeros, subnormal factors and accumulators,
normal factors with subnormal products, overflow, infinities and NaNs.
Integer SIMD tests force every usable CPU level and cover unaligned, noncontiguous
VGPR storage, masks and aliasing. Independent int64 matrix references additionally
check extreme signed/unsigned factors, wrapping, final-only saturation and
intermediate cancellation; SIMD builds and scalar-only builds run the same tests.

CPU detection follows the CPUID/XCR0 gating approach in
`hrx-system/runtime/src/iree/base/internal/cpu_x86_64.c`, with GoC's coarse
feature bundles. Formatting and the MIT license are borrowed from rocjitsu.

Still pending: other GPU architectures, additional instructions/formats,
further GPU FP-mode flags, and wider performance tuning.
