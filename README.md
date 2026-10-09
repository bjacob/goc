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

The selected tests demonstrate FP32 unary arithmetic, FMA, WMMA numeric formats, hardware-captured
FP16/BF16 exactness, intermediate FP16 overflow state, and SIMD/scalar agreement.
The full suite also covers DOT2, NEG/NEG_HI, wave64, masks, aliasing, C linkage,
header self-containment and library exports.

The benchmark includes wave32 FP32 unary arithmetic (unmodified and
ABS/scaling/CLAMP), FP32 FMA (unmodified and NEG/ABS/scaling) on scalar, v3 and v4, and compares scalar loose FP16 with x86-64-v3, and scalar loose BF16
with both x86-64-v3 and the Zen4 AVX-512 BF16 path. Integer WMMA rows cover
INT8 K=16 and INT4 K=16/K=32 on scalar, v3 and Zen4 VNNI, with unsigned
wrapping and signed CLAMP workloads. Integer WMMA rows request strict exact
semantics. Integer DOT4/DOT8 rows compare scalar and v3 for signed/unsigned
accumulators and wrapping/CLAMP, using loose semantics. All integer rows check
independent integer goldens.
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
with signedness and CLAMP as labeled. All workloads run with full EXEC, separate
C/D storage and hot buffers. FMA uses independent integer goldens.
Speedups compare paths with the same input, semantics and instruction flags.
Timings include public API dispatch, input conversions and output stores. Each
reported time is the median of seven samples after warmup. Each path starts at 128 calls (overridable by the
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
| `v_min3_num_f16`, `v_max3_num_f16`, `v_minmax_num_f16`, `v_maxmin_num_f16` | Scalar, x86-64-v3 | Not implemented |
| `v_minimum3_f16`, `v_maximum3_f16`, `v_minimummaximum_f16`, `v_maximumminimum_f16`, `v_med3_num_f16` | Scalar, x86-64-v3 | Not implemented |
| `v_ldexp_f16`, `v_frexp_exp_i16_f16` | Scalar, x86-64-v3 | Not implemented |
| `v_trunc_f16`, `v_ceil_f16`, `v_rndne_f16`, `v_floor_f16`, `v_fract_f16` | Scalar, x86-64-v3 | Not implemented |
| `v_sqrt_f16`, `v_rcp_f16`, `v_rsq_f16`, `v_frexp_mant_f16` | Scalar, x86-64-v3 | Not implemented |
| `v_exp_f16`, `v_log_f16` | Scalar | Not implemented |
| `v_add_f16`, `v_sub_f16`, `v_subrev_f16`, `v_mul_f16` | Scalar, x86-64-v3 | Not implemented |
| `v_min_num_f16`, `v_max_num_f16`, `v_minimum_f16`, `v_maximum_f16` | Scalar, x86-64-v3 | Not implemented |
| `v_add_f32`, `v_sub_f32`, `v_subrev_f32`, `v_mul_f32`, `v_mul_dx9_zero_f32` | Scalar, x86-64-v3 | Not implemented |
| `v_min_num_f32`, `v_max_num_f32`, `v_minimum_f32`, `v_maximum_f32` | Scalar, x86-64-v3 | Not implemented |
| `v_min3_num_f32`, `v_max3_num_f32`, `v_minmax_num_f32`, `v_maxmin_num_f32` | Scalar, x86-64-v3 | Not implemented |
| `v_minimum3_f32`, `v_maximum3_f32`, `v_minimummaximum_f32`, `v_maximumminimum_f32` | Scalar, x86-64-v3 | Not implemented |
| `v_med3_num_f32` | Scalar, x86-64-v3 | Not implemented |
| `v_fma_f32`, `v_fma_dx9_zero_f32` | Scalar, AVX2/FMA, AVX-512 | Not implemented |
| `v_add_f64`, `v_mul_f64`, `v_fma_f64` | Scalar, x86-64-v3 | Not implemented |
| `v_min_num_f64`, `v_max_num_f64`, `v_minimum_f64`, `v_maximum_f64` | Scalar, x86-64-v3 | Not implemented |
| `v_trunc_f64`, `v_ceil_f64`, `v_rndne_f64`, `v_floor_f64`, `v_fract_f64` | Scalar, x86-64-v3 | Not implemented |
| `v_sqrt_f64`, `v_rcp_f64`, `v_rsq_f64` | Scalar, x86-64-v3 | Not implemented |
| `v_frexp_mant_f32`, `v_frexp_mant_f64` | Scalar, x86-64-v3 | Not implemented |
| `v_frexp_exp_i32_f32`, `v_frexp_exp_i32_f64` | Scalar, x86-64-v3 | Not implemented |
| `v_ldexp_f32`, `v_ldexp_f64` | Scalar, x86-64-v3, x86-64-v4 | Not implemented |
| `v_min_i32`, `v_max_i32`, `v_min_u32`, `v_max_u32` | Scalar, x86-64-v3, x86-64-v4 | Not implemented |
| `v_mul_lo_u32`, `v_mul_hi_u32`, `v_mul_hi_i32` | Scalar, x86-64-v3, x86-64-v4 | Not implemented |
| `v_add_nc_u32`, `v_sub_nc_u32`, `v_subrev_nc_u32`, `v_add_nc_i32`, `v_sub_nc_i32`, `v_add3_u32` | Scalar, x86-64-v3 (saturation), x86-64-v4 | Not implemented |
| `v_mul_i32_i24`, `v_mul_hi_i32_i24`, `v_mul_u32_u24`, `v_mul_hi_u32_u24` | Scalar, x86-64-v3, x86-64-v4 | Not implemented |
| `v_min3_{i32,u32}`, `v_max3_{i32,u32}`, `v_minmax_{i32,u32}`, `v_maxmin_{i32,u32}`, `v_med3_{i32,u32}` | Scalar, x86-64-v3, x86-64-v4 | Not implemented |
| `v_trunc_f32`, `v_ceil_f32`, `v_rndne_f32`, `v_floor_f32`, `v_fract_f32` | Scalar, x86-64-v3 | Not implemented |
| `v_sqrt_f32`, `v_rcp_f32`, `v_rsq_f32` | Scalar, x86-64-v3 | Not implemented |
| `v_sin_f16`, `v_cos_f16` | Scalar, x86-64-v3; half selectors and all modifiers | Rounded captured RDNA3/4 integer model |
| `v_sin_f32`, `v_cos_f32` | Scalar, x86-64-v3 | Captured RDNA3/4 integer model |
| `v_exp_f32`, `v_log_f32` | Scalar `exp2` / `log2` | Not implemented |
| `v_dot4_f32_{fp8,bf8}_{fp8,bf8}` (all four combinations) | Scalar, x86-64-v3 | Not implemented |
| `v_pk_add_i16`, `v_pk_sub_i16`, `v_pk_add_u16`, `v_pk_sub_u16` | Scalar, x86-64-v3; all half selectors and saturation | Not implemented |
| `v_mad_i32_i16`, `v_mad_u32_u16` | Scalar, x86-64-v3; factor half selectors and saturation | Not implemented |
| `v_mad_i32_i24`, `v_mad_u32_u24` | Scalar, x86-64-v3; saturation | Not implemented |
| `v_mad_i16`, `v_mad_u16` | Scalar, x86-64-v3; half selectors and saturation | Not implemented |
| `v_min3_i16`, `v_min3_u16`, `v_max3_i16`, `v_max3_u16`, `v_med3_i16`, `v_med3_u16` | Scalar, x86-64-v3; half selectors | Not implemented |
| `v_add_nc_i16`, `v_sub_nc_i16`, `v_add_nc_u16`, `v_sub_nc_u16` | Scalar, x86-64-v3; half selectors and saturation | Not implemented |
| `v_min_i16`, `v_max_i16`, `v_min_u16`, `v_max_u16`, `v_mul_lo_u16` | Scalar, x86-64-v3; half selectors | Not implemented |
| `v_lshlrev_b32`, `v_lshrrev_b32`, `v_ashrrev_i32` | Scalar, x86-64-v3 | Not implemented |
| `v_lshlrev_b64`, `v_lshrrev_b64`, `v_ashrrev_i64` | Scalar, x86-64-v3 | Not implemented |
| `v_lshlrev_b16`, `v_lshrrev_b16`, `v_ashrrev_i16` | Scalar, x86-64-v3; half selectors | Not implemented |
| `v_pk_mad_i16`, `v_pk_mad_u16` | Scalar, x86-64-v3; all half selectors and saturation | Not implemented |
| `v_pk_lshlrev_b16`, `v_pk_lshrrev_b16`, `v_pk_ashrrev_i16` | Scalar, x86-64-v3; all half selectors | Not implemented |
| `v_pk_min_i16`, `v_pk_max_i16`, `v_pk_min_u16`, `v_pk_max_u16`, `v_pk_mul_lo_u16` | Scalar, x86-64-v3; all half selectors | Not implemented |
| `v_dot4_i32_iu8`, `v_dot4_u32_u8` | Scalar, x86-64-v3 | Same integer result |
| `v_dot8_i32_iu4`, `v_dot8_u32_u4` | Scalar, x86-64-v3 | Same integer result |
| `v_dot2_f16_f16`, `v_dot2_bf16_bf16` | Scalar, x86-64-v3 | Not implemented |
| `v_dot2_f32_f16` | Scalar, x86-64-v3 | Integer arithmetic model |
| `v_dot2_f32_bf16` | Scalar, x86-64-v3 | Integer arithmetic model |
| `v_wmma_f32_16x16x16_f16` | Scalar, x86-64-v3 F16C/AVX2/FMA | Integer arithmetic model |
| `v_wmma_f32_16x16x16_bf16` | Scalar, x86-64-v3 AVX2/FMA, AVX-512 BF16 | Integer arithmetic model |
| `v_wmma_f16_16x16x16_f16` | Integer arithmetic model | Integer arithmetic model |
| `v_wmma_bf16_16x16x16_bf16` | Integer arithmetic model | Integer arithmetic model |
| `v_wmma_f32_16x16x16_{fp8,bf8}_{fp8,bf8}` (all four combinations) | Scalar, x86-64-v3 | Not implemented |
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

All four FP8/BF8 WMMA forms have v3 SIMD paths supporting NEG_C/ABS_C.
They decode each input element once and reuse it across output rows/columns;
all results are staged before masked writes to preserve operand aliasing.
Tests cross dense matrix goldens with all CPU levels, all C modifiers, 85 EXEC
masks and aliases, and check every input byte encoding through each operand.
Benchmark `wmma/f8f8`, `wmma/f8b8`, `wmma/b8f8`, and `wmma/b8b8` rows
compare scalar and SIMD with full EXEC, default flags and ABS_C/NEG_C.

FP8/BF8 DOT4 supports all four E4M3FN/E5M2 input combinations with FP32
accumulation and `GOC_DOT_ABS_C` / `GOC_DOT_NEG_C`, applied in that order.
Both scalar and v3 paths handle all encodings, including subnormals and special
values; no input-dependent fallback is needed. Tests exhaust all 65,536 input
byte pairs for every format combination, modifier combination and CPU level.
Strict exact requests are rejected. Benchmark labels `f8` and `b8` denote
FP8 and BF8; each combination has unmodified and ABS_C/NEG_C rows.

The true16 `v_dot2_f16_f16` and `v_dot2_bf16_bf16` instructions instead use
`GOC_ALU_` ABS/NEG modifiers on each whole operand, `GOC_ALU_HIGH_C` for
the accumulator half, and `GOC_ALU_HIGH_D` for the destination half. The
unselected destination half is preserved, including when D aliases an input.
They do not accept OMOD or CLAMP. Scalar and v3 paths retain FP32 product/sum
association followed by nearest-even narrowing, matching rocjitsu's loose
evaluation. BF16 input/output denormals flush independently of other FP flags;
FP16 supports `GOC_FP16_OVFL`. Host nearest-even rounding and enabled denormals
remain required by the loose FP contract. Tests cover every accumulator encoding,
all 256 modifier/half-selector combinations with 85 masks and aliases, literal
rounding/overflow cases, and random scalar/SIMD comparisons. Benchmark
`dot2/f16` / `dot2/b16` rows denote these 16-bit-output instructions.

FP16/BF16 DOT2 supports independent negation of each selected A/B half and C.
The four half-selection flags can swap or replicate halves; zero flags select
low then high as before. Loose v3 paths retain SIMD for every modifier combination;
exact requests keep the existing scalar model. Following rocjitsu's GFX12 DOT2
implementation, CLAMP is accepted but has no effect. Tests cover all 1,024
sign/selection/CLAMP combinations in both semantics, masks and aliases, with
hardware-captured special-value fixtures recovered through inverse modifiers.
The benchmark compares default, NEG_LO_A, and combined selection/NEG_HI_B cases.

Integer DOT4/DOT8 use one VGPR for each operand. I32_IU forms support independent
`GOC_DOT_SIGNED_A` / `GOC_DOT_SIGNED_B` flags and a signed accumulator;
U32_U forms use unsigned factors and accumulators. `GOC_DOT_CLAMP` saturates
only after the complete dot plus accumulator; otherwise results wrap modulo
2^32. Every flag combination remains on the v3 path. Widening to signed 16-bit
factors avoids the unwanted intermediate saturation of x86 byte-pair dot
instructions. Tests cover all signedness/CLAMP modes, overflow boundaries,
85 masks, source/destination aliases and unchanged host FP state.

FP64 ADD, MUL and FMA use two VGPRs per operand: element zero of each pointer
array names the low-word buffer and element one names the high-word buffer.
Each buffer still contains 32 lane words; their addresses need not be adjacent.
Scalar and x86-64-v3 paths support all applicable ALU source/output modifiers
(128 combinations for binary operations, 512 for FMA). The SIMD path processes
four FP64 lanes at a time and stages both result halves before masked stores.
Tests cross all modifiers, CPU levels and 85 masks with ten destination layouts,
including reversed halves, aliases spanning different operands, and identical
output buffers; the latter receive the high-word write last. Literal cases
cover fused rounding, subnormals, overflow, signed zero, NaNs and CLAMP.
These loose paths require host nearest-even rounding with denormals enabled.

FP64 `MIN_NUM`, `MAX_NUM`, `MINIMUM` and `MAXIMUM` use the same staged
scalar/four-lane SIMD paths and support all 128 A/B modifier combinations.
Number variants ignore a lone NaN; propagating variants prefer signaling NaNs
and quiet them. Both families order negative zero below positive zero. Tests
cover all 576 pairs from a 24-value special-value set with every modifier and
CPU level, plus literal NaN-priority/zero cases and the full mask/alias matrix.

FP64 TRUNC, CEIL, RNDNE, FLOOR, FRACT, SQRT, RCP and RSQ share that layout and
four-lane SIMD implementation, with all 32 A ABS/NEG/OMOD/CLAMP combinations.
RNDNE uses ties-to-even rounding and preserves signed zero; FRACT caps its
fractional result at `0x3fefffffffffffff` before output modifiers. Tests cross
all modifiers, CPU levels, 85 masks and six destination layouts with signed
zeros, subnormals, infinities and NaNs, using higher-precision references and
literal rounding/boundary cases. Unary paths also stage both output halves
before writes, including reversed or identical destination buffers.

FP32 ADD, SUB, SUBREV and MUL support A/B ABS/NEG, output scaling and CLAMP
on scalar and v3 paths. Tests cross all 128 modifier combinations with all CPU
levels, 85 masks and aliases, including signed zeros, subnormals, infinities,
NaNs and overflow. Their benchmark rows compare default and modified cases.
The DX9 zero-multiplication variant supports the same paths and modifiers.
Either signed-zero input forces positive zero, even with NaN or infinity in
the other operand. Nonzero products that underflow retain their usual sign.

FP32 `MIN_NUM`, `MAX_NUM`, `MINIMUM` and `MAXIMUM` have the same modifier,
mask, alias and CPU-path coverage. Number variants select a numeric operand
over either kind of NaN; propagating variants prefer signaling NaNs and quiet
them. Both families order negative zero below positive zero. Literal tests
check NaN selection and quieting, signed zeros, infinities and subnormals.
The SIMD path handles these rules explicitly and supports all 128 modifiers.

All eight three-input FP32 min/max variants use the same scalar/v3 selection
rules. They select between A/B first, then between that result and C, applying
output scaling and CLAMP only at the end. All 512 A/B/C modifier combinations
remain on the SIMD path. Tests cross those combinations with 85 masks, every
CPU level and each destination/source alias, and include literal evaluation-order,
NaN-priority and signed-zero cases plus 4,096 random input triples. Benchmark
labels use `min3`/`max3` for repeated selections, `mnmx`/`mxmn` for mixed
selections, and an `n` suffix for number-preferring variants.

FP32 median selection also supports all 512 modifiers on scalar/v3 paths.
With any NaN input it returns the three-input minimumNumber result; otherwise
it follows the ISA rule of removing the first input numerically equal to the
maximum and selecting the maximum of the other two. Tests include signed-zero
ties, where this rule differs from sorting by a total order that distinguishes
the signs of zero. The benchmark labels this instruction `f32/med3n`.

FMA supports all three source ABS/NEG pairs, OMOD scaling and CLAMP on scalar,
x86-64-v3 and x86-64-v4 paths. Tests cross all 512 modifier combinations with
85 masks, all CPU levels and output aliasing each source; literal bit patterns
add fused-rounding, signed-zero, subnormal, overflow and NaN-clamping cases.
The DX9 FMA variant has the same scalar/v3/v4 paths and full modifier/mask/alias
coverage. If either factor is signed zero, it selects modified C before output
scaling and CLAMP. Without output modifiers, the selected C retains its signed
zero and NaN bits. Dedicated tests cross every modifier with exceptional factors and accumulators, and retain
a literal fused-rounding witness for nonzero products.

FP32 `FRACT` computes `x - floor(x)` and caps it at `0x3f7fffff` before
output modifiers, so tiny negative inputs stay strictly below one. Scalar/v3
paths support all 32 modifiers, with literal boundary and signed-zero tests.

FP32 `SIN` and `COS` take inputs in turns (`sin(2*pi*x)` and `cos(2*pi*x)`).
Integer range reduction handles all finite FP32 encodings, including values too
large for a float-to-integer conversion. Empirical exact semantics borrow
rocjitsu's captured RDNA3/4 staged integer model and preserve the entire host
floating-point environment. The eight-lane v3 loose path shares its coefficients
but evaluates the cubic polynomials with vector FMA. All 32 ABS/NEG/OMOD/CLAMP
combinations stay on SIMD. Active OMOD flushes subnormal outputs and signed
zeros to positive zero. Tests include 35 hardware-captured cases, interval
boundaries, random raw encodings, all modifiers, masks and aliases, and exact
results under every host rounding mode and x86 denormal-control setting.
Benchmarks compare full-wave default and modified calls against independent
mathematical references; only these approximate FP32 rows use a numerical
tolerance instead of bitwise output comparisons.

FP16 `SIN` and `COS` share that model and the eight-lane SIMD evaluation,
with `HIGH_A/D` selectors and all 128 modifier combinations. They round the
trig result to FP16 before OMOD; active OMOD flushes tiny values before scaling
and tiny rounded outputs afterward. The other destination half is preserved.
Exact semantics preserve all host FP state. Exhaustive tests cover every half
encoding against digests generated directly from rocjitsu, and verify loose
SIMD outputs within one half-precision ULP of the model. Further tests cover
all modifiers, half selectors, aliases, masks, host FP settings, and literal
OMOD underflow boundaries. `GOC_FP16_OVFL` is accepted but has no effect because
finite trig outputs and their permitted scaling cannot overflow FP16.

The 32- and 64-bit reverse shifts (`LSHLREV`, `LSHRREV`, `ASHRREV`) take
the shift count in `A` and the value in `B`. Counts wrap modulo the value width.
The 64-bit forms use two VGPRs for `B` and `D`, low word first, and one for `A`.
Eight-lane v3 paths operate directly on the separate low/high words, including
arithmetic sign extension across the 32-bit boundary. No instruction modifiers
apply. Both scalar and SIMD preserve all host FP state. Tests cover every
count and bit position, random full-width counts, single-active/inactive-lane
masks, and every destination alias with the count or either value half.

FP32/FP64 `FREXP_MANT` extracts a signed binary significand with magnitude in
[0.5, 1) for finite nonzero inputs. Subnormals are normalized on the SIMD path;
zero, infinity and NaN bits pass through before output modifiers. Both widths
support all 32 unary modifier combinations and the same mask/alias guarantees
as their other arithmetic operations. Literal tests check subnormal boundaries
and exceptional-value bit preservation, including signaling NaNs.

`FREXP_EXP_I32_F32` and `FREXP_EXP_I32_F64` return a signed binary exponent
in one destination VGPR, using one or two source VGPRs respectively. Zero,
infinity and NaN inputs return zero. Both scalar and v3 paths handle subnormals;
ABS/NEG, OMOD and CLAMP are accepted without changing the integer result.
Tests cover all 32 modifier combinations, every exponent field and subnormal
leading-bit position, special values, masks, and aliases with either FP64 half.

FP32/FP64 `LDEXP` scales A by an integer power of two held in one B VGPR.
A and D use one VGPR for FP32 or low/high pairs for FP64. Both widths have
scalar, v3 and v4 implementations, supporting all 32 A ABS/NEG, OMOD and CLAMP
combinations without falling back to scalar. B is an integer and has no
floating-point modifiers. The v3 path adjusts exponents and rounds underflowing
results once; v4 uses native vector scaling. Tests cover every exponent field,
subnormal rounding ties, extreme signed exponents, special values, masks and
aliases, including FP64 destination halves that overwrite A or B.

Signed and unsigned 32-bit integer min/max instructions cover two-input
selection, three-input min/max, mixed min/max and median. Mixed operations
combine A/B first: `MINMAX = max(min(A, B), C)` and
`MAXMIN = min(max(A, B), C)`. Every form has scalar, eight-lane v3 and sixteen-lane v4 paths,
masked stores and whole-register aliases. These instructions have no arithmetic
modifiers; `instruction_flags` must be zero. Tests cover boundary Cartesian
products, random inputs, all mask patterns, source/destination aliases, signed
versus unsigned ordering, operand grouping, and host FP-environment preservation.

Integer multiply covers low/high 32-bit products and all four signed/unsigned
24-bit forms. The 24-bit forms discard each source's upper byte and sign-extend
signed inputs. `v_mul_i32_i24` and `v_mul_u32_u24` accept `GOC_ALU_CLAMP` for signed
or unsigned saturation; other forms require zero instruction flags. Scalar,
eight-lane v3 and sixteen-lane v4 paths support masks and whole-register aliases.
Saturation stays on SIMD by checking the high product word against the low
word's sign extension, or against zero for unsigned products. Tests include
literal high-word results, saturation thresholds, upper-byte noise, random
products, masks, aliases and host FP-environment preservation.

Integer MAD with 32-bit C/D supports signed/unsigned 16-bit and 24-bit factors.
The 16-bit forms select A/B halves through `GOC_ALU_HIGH_A/B`; 24-bit forms
discard the factors' upper bytes. CLAMP saturates the full product plus the
32-bit accumulator, with no intermediate truncation or saturation. Scalar and
AVX2 paths support all modifiers, masks and aliases. AVX2 uses wide 64-bit sums
for saturation and native 32-bit arithmetic for wrapping results. Tests cover
boundary triples, all 16-bit factor encodings, discarded upper-byte combinations,
full-width accumulators, overflow/cancellation goldens, masks and all aliases.

Ordinary 16-bit signed/unsigned MAD, three-input min/max and median use
`GOC_ALU_HIGH_A/B/C/D` to select each source and destination half, preserving
the other destination half. MAD also accepts CLAMP after full-precision
multiply-add. Scalar and AVX2 paths support every modifier, mask and alias.
Tests cover every source encoding, boundary triples, all 15 whole-register
alias layouts, every modifier, masks and literal saturation/selection witnesses.

Ordinary 16-bit integer ADD/SUB, min/max, low-word multiply and shifts select
one A/B half and write one D half through `GOC_ALU_HIGH_A/B/D`. The other D half
is preserved, including when D aliases either input. ADD/SUB additionally
support `GOC_ALU_CLAMP` for saturation; other forms reject it. Scalar and AVX2
paths share the packed arithmetic templates and keep every valid modifier on SIMD.
Tests cover every input encoding, boundary pairs, all modifiers, masks, aliases,
literal half-preservation witnesses and host FP-state preservation.

Packed 16-bit integer ADD/SUB, min/max, and low-word multiply compute two
results per VGPR lane. All 32 combinations of A/B half selection and
`GOC_PK_CLAMP` stay on the eight-VGPR-lane AVX2 path, using sixteen native
16-bit operations per vector. ADD/SUB wrap unless CLAMP requests signed or
unsigned saturation. Min/max and multiply ignore CLAMP, matching rocjitsu;
negation and C flags are invalid. Both halves read the original inputs before
masked stores. Tests cover every 16-bit encoding, boundary pairs, every modifier
combination, masks, aliases, literal saturation witnesses and FP-state preservation.

Packed signed/unsigned 16-bit multiply-add supports all 128 combinations of
A/B/C half selectors and CLAMP. Saturation applies to the full `A * B + C`
result, without truncating or saturating the intermediate product. Scalar and
AVX2 paths support all modifiers, masks and aliases; AVX2 uses packed 16-bit
arithmetic for wrapping results and widens to 32 bits for saturation.
Tests cover boundary triples, every source encoding, every modifier, all 15
whole-register alias layouts, masks, saturation/cancellation witnesses and FP state.

Packed 16-bit shifts take the counts from A and the values from B. Counts
use only their low four bits; arithmetic right shift propagates the selected
half's sign. Scalar and AVX2 paths support every A/B half selector, ignored
CLAMP, EXEC masks and whole-register aliases. AVX2 shifts the low and high
halves separately in 32-bit lanes, then packs them before masked stores.
Tests exercise every count encoding with every selector combination, every
value encoding, boundary values and literal sign-extension witnesses.

Packed FP16 `PK_ADD`, `PK_MUL`, `PK_MIN_NUM`, `PK_MAX_NUM`, `PK_MINIMUM`
and `PK_MAXIMUM` compute two results per VGPR lane. They support all 512
combinations of A/B `GOC_PK_*` negation, half selection and CLAMP, plus
`GOC_FP16_OVFL`. C flags are invalid. Number min/max ignore a lone NaN;
propagating min/max return NaN, and both families order negative zero below
positive zero. Scalar and eight-packed-lane x86-64-v3 paths share the existing
ordinary FP16 arithmetic/selection helpers. These are loose semantics.
Tests cover every half encoding, special-value Cartesian pairs, all modifiers,
overflow policy, masks, all aliases and literal cross-half/signed-zero/NaN cases.

Packed FP16 `PK_FMA` computes both halves of `A * B + C`; `PK_FMAC` uses
D as its accumulator. Each uses one VGPR per operand. `PK_FMA` supports all
8,192 combinations of independent low/high-result source negation, half selection
and CLAMP through `GOC_PK_*` flags. Zero flags select corresponding halves;
selectors can swap or replicate source halves. RDNA4's `PK_FMAC` has no
instruction flags. Both support `GOC_FP16_OVFL` and the empirical exact scalar
FMA model. The loose x86-64-v3 path computes eight packed lanes at a time
(two vectors of eight FP16 results) and supports every modifier combination.

Both result halves are computed from the original inputs before destination
stores, including cross-half source selections when D aliases an input. Tests
cover every flag combination and half encoding, random triples, overflow policy,
masks, all whole-register aliases, explicit cross-half alias witnesses, packed
rocjitsu hardware cases and exact host-environment preservation. Packed FMAC
benchmarks include the same fixed-accumulator reset as ordinary FMAC.

FP16/FP32 `FMAMK` and `FMAAK` take a scalar literal by value, in assembly
operand order: `D, A, literal, B` for multiply-literal and `D, A, B, literal`
for add-literal. Literals are raw encodings (`uint16_t` / `uint32_t`). FP16
accepts `HIGH_A/B/D` and preserves the other destination half; FP32 accepts
no instruction flags. Both provide scalar and SIMD paths with direct literal
broadcasts: eight lanes for FP16, eight or sixteen for FP32. FP16 also supports
`GOC_FP16_OVFL` and the empirical exact FMA model. Tests cover every FP16 literal
encoding with random operands, FP32 random triples, exceptional-value priority,
fused rounding, all half selectors, masks, aliases, validation and the C ABI.

FP16/FP32 `FMAC` multiplies A and B and adds the previous value of D.
It supports A/B ABS/NEG, OMOD and CLAMP; FP16 also supports A/B/D half
selection and `GOC_FP16_OVFL`. The destination half selects the accumulator
half as well. Independent C modifiers, including `HIGH_C`, are invalid.
All 1,024 FP16 and 128 FP32 modifier combinations use the existing FMA
scalar/SIMD paths: eight lanes for FP16, eight or sixteen for FP32. FP16
also inherits the empirical exact FMA model and host-environment preservation.
Tests cover modifiers, masks, every whole-register accumulator alias, half
selection, overflow policy, fused rounding and rocjitsu's FMA/FMAC witnesses.
FMAC benchmark timings include resetting D to a fixed accumulator before each
call, equally for all CPU paths, to avoid drift during repeated accumulation.

FP16 `FMA` supports all 8,192 combinations of source ABS/NEG, OMOD, CLAMP
and A/B/C/D half selectors, plus `GOC_FP16_OVFL`. Loose semantics have scalar
and eight-lane x86-64-v3 paths. The SIMD path retains a product/sum residual
to avoid double rounding when narrowing to FP16; modifiers stay on SIMD.
Arithmetic rounds to FP16 before OMOD. Active OMOD flushes tiny arithmetic
results to positive zero and newly tiny scaled results to signed zero, matching
rocjitsu's captured RDNA4 behavior. Consequently, output scaling cannot recover
an arithmetic overflow or a flushed tiny value. CLAMP applies last.

An empirical exact scalar path borrows rocjitsu's FMA exceptional-value and
rounding policies, including NaN payload priority and invalid-product ordering.
It uses nearest-even arithmetic with preserved denormals and restores the host
floating-point environment. Tests use an independent integer-significand oracle,
all half encodings, random triples, all modifier combinations, mask/alias cases,
rocjitsu hardware witnesses, double-rounding and tininess boundaries, and all
four host rounding modes. Benchmarks compare full-EXEC scalar/SIMD loose paths
and report exact scalar separately, with default and modified inputs.

Three-input FP16 min/max and median share the FP32 selection rules. The mixed
forms select A/B first and then C; median uses the minimumNumber result if any
input is NaN and follows the ISA's first-maximum removal rule for signed-zero
ties. Scalar and eight-lane SIMD paths support all 8,192 combinations of A/B/C
ABS/NEG, output scaling/clamp, and independent A/B/C/D half selectors, plus
`GOC_FP16_OVFL`. Output modifiers apply after the final selection. Tests cover
every half encoding, special-value Cartesian products, all modifier combinations,
all mask patterns, all whole-register alias layouts, and literal operand-order,
NaN and signed-zero cases. Nonselected destination halves remain unchanged.

FP16 `LDEXP` reads a floating half from A and a signed integer half from B,
scales A by that power of two, and writes the selected D half. It accepts
`HIGH_A/B/D`, A ABS/NEG, output scaling/clamp and `GOC_FP16_OVFL`. Scalar and
eight-lane SIMD paths bound extreme integer exponents while preserving every
FP16 rounding outcome, including output scaling at the underflow/overflow
boundary. `FREXP_EXP_I16_F16` writes a signed 16-bit exponent into the selected
D half, returning zero for zeros, infinities and NaNs. Source sign modifiers,
OMOD and CLAMP do not change that integer result. Scalar and SIMD FREXP paths
preserve the host FP environment, including with signaling NaNs and nondefault
rounding modes.
Both instructions preserve the unselected D half and support whole-register
aliases. Tests cover all half encodings, every signed 16-bit exponent, all
modifier combinations, all mask patterns, aliases and literal boundaries.

Unary FP16 operations use the same selected-half storage and output-modifier
rules as binary FP16. Rounding, reciprocal, square root, reciprocal square root,
fraction and mantissa extraction have eight-lane x86-64-v3 paths for all 128
combinations of source ABS/NEG, output scaling/clamp and A/D half selection.
Base-two EXP/LOG use scalar libm paths. FRACT caps its result at the largest
half below one before output modifiers; finite EXP overflow honors
`GOC_FP16_OVFL`, even when the mathematical result exceeds FP32's range.
Tests cover all 65,536 half encodings, both overflow policies, every modifier
combination, masks, aliases, untouched destination halves and literal boundaries.
Transcendental accuracy is tested to within one adjacent half encoding of an
independent double-precision reference; exact semantics are not claimed.

Binary FP16 arithmetic and min/max use one selected half of each VGPR. The
`GOC_ALU_HIGH_A`, `GOC_ALU_HIGH_B` and `GOC_ALU_HIGH_D` flags select high halves;
low halves are the default. The other destination half remains unchanged,
including when the destination aliases a source. All 1,024 combinations of
source ABS/NEG, output scaling/clamp, and half selectors stay on the eight-lane
x86-64-v3 path. Scalar and SIMD paths widen to FP32, perform the operation and
output modifiers, then narrow to FP16 with nearest-even rounding. `GOC_FP16_OVFL`
saturates finite overflow to the largest finite half. These are loose semantics;
host rounding must be nearest-even with denormals enabled. Tests cover every
half encoding, random pairs, all modifier combinations, all mask patterns,
source/destination aliases, signed zeros, NaN rules, rounding ties and overflow.

Non-carry integer add/subtract covers unsigned addition, subtraction and reverse
subtraction, signed addition/subtraction, and unsigned three-input addition.
Two-input forms support `GOC_ALU_CLAMP`; `v_add3_u32` wraps modulo 2^32 and has no
arithmetic modifiers. Every form has scalar and sixteen-lane v4 paths; eight-lane
v3 is selected for saturation, where it beats the baseline implementation.
Wrapping forms use the baseline path on v3 CPUs because AVX2 masked-store
overhead outweighs their arithmetic savings. Saturation remains SIMD, using
integer overflow/borrow detection. Tests cover signed limits, unsigned
carry/borrow, operand order, three-input wrap, masks, all whole-register alias
layouts and host FP-environment preservation.

Unary FP32 instructions support `GOC_ALU_ABS_A`, `GOC_ALU_NEG_A`, output
scaling (`GOC_ALU_OMOD_2`, `GOC_ALU_OMOD_4`, `GOC_ALU_OMOD_HALF`), and
`GOC_ALU_CLAMP` on both scalar and SIMD paths. ABS precedes NEG; scaling
precedes CLAMP. CLAMP maps NaNs to positive zero and clamps to [0, 1].
These loose semantics explicitly apply the requested scaling; GPU FP-state
rules that conditionally suppress OMOD are not yet modeled.
The scalar ties-to-even helper is adapted from rocjitsu's
[`rndne_scalar`](https://github.com/ROCm/rocm-systems/blob/develop/emulation/rocjitsu/lib/util/include/util/simd.h).
Unary tests cross all 32 modifier combinations with 85 masks, both separate and
aliased output, and all available CPU levels, including signed zeros, subnormals,
infinities, NaNs, half-integer ties and large integral values.

GoC applies `exec_mask` to destination writes, including WMMA, as specified by
its API contract. Inactive destination lanes remain unchanged; source lanes are
not masked. Empty effective EXEC masks return immediately after flag validation,
including high-bits-only masks in wave32. Invalid flags and unsupported strict
semantics still return errors. Nonempty masks stay on the same SIMD paths;
SIMD FMA/WMMA use masked stores, and WMMA stages results before writes to support
aliasing. Improving sparse-mask performance is an explicit non-goal: the intended
performance is independent of the mask. Compute full results and mask destination
stores, without mask-density checks or sparse-mask specializations that add code
size and runtime overhead. The existing empty-mask early return is an exception;
this design goal is not a constant-time guarantee.
Errors preserve all destination registers. No pointer-validation
or allocation ownership service is provided.

Zero semantics bits select loose numerical behavior. Add `GOC_SEMANTICS_EXACT_EMPIRICAL`
for the empirical model; also add `GOC_SEMANTICS_STRICT` to require support.
Unsupported exact requests otherwise fall back to loose semantics. Unassigned
instruction/general flag bits are rejected, except reserved semantics values
which follow the same fallback policy. `GOC_FP16_OVFL` emulates GPU MODE.FP16_OVFL: finite FP16 overflow saturates
to signed 65504 instead of infinity. Input infinities remain infinite, and
BF16/FP32 instructions ignore this state. Packed results narrow after each
four-product step. The packed WMMA and empirical exact WMMA paths use integer
arithmetic, preserving the caller's host rounding mode and exception flags.
Exact FP16 FMA instead saves and restores the host floating-point environment.
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

FP16 FMA borrows `fma_f16` / `finish_fma_f16` from rocjitsu's
`shared/fp_mode.h`, restricted to nearest-even with denormals preserved. Its
NaN/OMOD literal witnesses come from `tests/valu_fp_mode_test.cpp`
(`f16_fma_nan_cases` and `f16_fma_omod_cases`), which records gfx1201 captures.
The integer test oracle and SIMD residual implementation are independent.

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
A shared 85-mask corpus covers every single-active and single-inactive wave32
lane, both alternating patterns, empty/full/high-bits-only masks and 16 seeded
random masks. FMA and integer WMMA cross it with all usable CPU levels and
operand overlap; floating WMMA does so with representative modifiers, retaining
the existing all-64-modifier tests. Empty-mask tests cover every entry point,
flag-validation errors, unchanged registers and host FP exception state. Wave64
tests explicitly exercise lone active lanes 32 and 63.
Integer SIMD tests force every usable CPU level and cover unaligned, noncontiguous
VGPR storage, masks and aliasing. Independent int64 matrix references additionally
check extreme signed/unsigned factors, wrapping, final-only saturation and
intermediate cancellation; SIMD builds and scalar-only builds run the same tests.

CPU detection follows the CPUID/XCR0 gating approach in
`hrx-system/runtime/src/iree/base/internal/cpu_x86_64.c`, with GoC's coarse
feature bundles. Formatting and the MIT license are borrowed from rocjitsu.

Still pending: other GPU architectures, additional instructions/formats,
further GPU FP-mode flags, and wider performance tuning.
