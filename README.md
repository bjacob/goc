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
../goc-build/tests/goc_rdna4_benchmark_static
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

Every workload checks its outputs against independent goldens before and after
timing. Floating-point WMMA workloads use fixed small-integer matrices
and compares no modifiers, `NEG_LO_A` alone, and a mixed case
(`NEG_HI_A | NEG_LO_B | ABS_C | NEG_C`). Modified floating-point rows use loose
semantics and independent integer matrix references.
Integer workloads use dense full-range factors and accumulators near overflow,
with signedness and CLAMP as labeled. All workloads run with full EXEC, separate
C/D storage and hot buffers. FMA uses independent integer goldens.
The `Instruction` column uses standard instruction mnemonics, including operand types.
The `Wave` column distinguishes wave32 and wave64 workloads.
The `FP state` column records guest FP settings separately from instruction
modifiers; `flush-input` selects `GOC_FP_FLUSH_INPUT_DENORMALS`, and
`fp16-ovfl` selects `GOC_FP16_OVFL`. `flush-io-ovfl` combines input/output
flushing and FP16 overflow saturation.
Pass `--csv` for comma-separated output on stdout, with one header row and
numeric speedup ratios (empty when unavailable). Explanatory text goes to stderr.
For example:

```sh
../goc-build/tests/goc_rdna4_benchmark_static --csv > results.csv
```

`--csv` and `--min-ms` can appear before or after the optional initial iteration count.
Speedups compare paths with the same instruction, wave size, semantics and instruction flags.
Timings include public API dispatch, input conversions and output stores. Each
reported time is the median of seven samples after warmup. Each path starts at 128 calls (overridable by the
positional argument) and doubles the count until the timed batch takes at least
10 ms. Shorter batches are discarded. Subsequent samples retain that count and
double again if necessary, so every accepted sample meets the minimum duration.
Pass `--min-ms` with a nonnegative integer to override the minimum milliseconds,
for example `../goc-build/tests/goc_rdna4_benchmark_static --min-ms 50`.
Benchmark controls are command-line arguments; no environment variables are read.
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
| `v_fma_mix_f32` | Scalar, x86-64-v3 | Not implemented |
| `v_fma_mixlo_f16`, `v_fma_mixhi_f16` | Scalar, x86-64-v3 | Scalar, rocjitsu-derived direct FP16 rounding |
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
| `v_clz_i32_u32`, `v_ctz_i32_b32`, `v_cls_i32`, `v_bcnt_u32_b32` | Scalar, x86-64-v3, x86-64-v4 | Not implemented |
| `v_mbcnt_lo_u32_b32` (wave32 and wave64), `v_mbcnt_hi_u32_b32` (wave64) | Scalar, x86-64-v3, x86-64-v4 | Not implemented |
| `v_mbcnt_hi_u32_b32` (wave32) | Scalar, x86-64-v4 | Not implemented |
| `v_and_b16`, `v_or_b16`, `v_xor_b16`, `v_not_b16` | Scalar, x86-64-v4 | Not implemented |
| `v_sat_pk_u8_i16`, `v_pack_b32_f16` | Scalar, x86-64-v3, x86-64-v4; all supported modifiers | Not implemented |
| `v_and_b32`, `v_or_b32`, `v_xor_b32`, `v_not_b32`, `v_xnor_b32` | Scalar, x86-64-v4 | Not implemented |
| `v_bfe_u32`, `v_bfe_i32`, `v_bfm_b32`, `v_bfrev_b32`, `v_alignbit_b32`, `v_alignbyte_b32`, `v_perm_b32` | Scalar, x86-64-v3, x86-64-v4 | Not implemented |
| `v_bfi_b32` | Scalar, x86-64-v4 | Not implemented |
| `v_lshl_add_u32`, `v_add_lshl_u32`, `v_lshl_or_b32`, `v_lerp_u8` | Scalar, x86-64-v3, x86-64-v4 | Not implemented |
| `v_and_or_b32`, `v_or3_b32`, `v_xor3_b32`, `v_xad_u32` | Scalar, x86-64-v4 | Not implemented |
| `v_sad_u8`, `v_sad_hi_u8`, `v_sad_u16`, `v_sad_u32`, `v_msad_u8` | Scalar, x86-64-v3; saturation | Not implemented |
| `v_qsad_pk_u16_u8`, `v_mqsad_pk_u16_u8`, `v_mqsad_u32_u8` | Scalar, x86-64-v3; independent packed/full-width saturation | Not implemented |
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
saturating byte-pair operations. With CLAMP, each kernel saturates after the
products with even `(k / 8)`, then after those with odd `(k / 8)`. For K=32,
the first stage therefore combines positions 0–7 and 16–23. GPU captures of
98,304 results verify these boundaries, including cancellation near int32
limits where final-only saturation gives a different answer. Scalar, v3 and
VNNI paths use the same staged semantics. Compile-time stage specialization
keeps CLAMP on fast SIMD paths: pinned-core measurements on the Ryzen 9 7950X3D
put signed CLAMP at 86–157 ns for v3 and 72–135 ns for VNNI across the three
shapes (seven samples, each at least 10 ms).

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
Benchmark rows for all four `v_wmma_f32_16x16x16_*` mnemonics
compare scalar and SIMD with full EXEC, default flags and ABS_C/NEG_C.

FP8/BF8 DOT4 supports all four E4M3FN/E5M2 input combinations with FP32
accumulation and `GOC_DOT_ABS_C` / `GOC_DOT_NEG_C`, applied in that order.
Both scalar and v3 paths handle all encodings, including subnormals and special
values; no input-dependent fallback is needed. Tests exhaust all 65,536 input
byte pairs for every format combination, modifier combination and CPU level.
Strict exact requests are rejected. Each combination has unmodified and
ABS_C/NEG_C benchmark rows, labeled with its full `v_dot4_f32_*` mnemonic.

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
rounding/overflow cases, and random scalar/SIMD comparisons. Benchmark rows
use the full mnemonics to distinguish these 16-bit-output instructions.

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
rows use the full mnemonics, including `_num` for number-preferring variants.

FP32 median selection also supports all 512 modifiers on scalar/v3 paths.
With any NaN input it returns the three-input minimumNumber result; otherwise
it follows the ISA rule of removing the first input numerically equal to the
maximum and selecting the maximum of the other two. Tests include signed-zero
ties, where this rule differs from sorting by a total order that distinguishes
the signs of zero. The benchmark labels this instruction `v_med3_num_f32`.

FMA supports all three source ABS/NEG pairs, OMOD scaling and CLAMP on scalar,
x86-64-v3 and x86-64-v4 paths. Tests cross all 512 modifier combinations with
85 masks, all CPU levels and output aliasing each source; literal bit patterns
add fused-rounding, signed-zero, subnormal, overflow and NaN-clamping cases.
The DX9 FMA variant has the same scalar/v3/v4 paths and full modifier/mask/alias
coverage. It flushes all three inputs and the result to signed zero regardless
of guest denormal mode. If either flushed factor is zero, the product becomes
positive zero before addition to C. Thus a negative-zero C produces positive
zero in RNE, and an addend signaling NaN is quieted. OMOD and CLAMP apply after
this arithmetic. Input/output flush flags are accepted and redundant here.
A GFX1201 capture checks 262,144 results across all 512 modifiers and all four
hardware denormal modes; tests cover every destination alias, exceptional
factors and accumulators, output-underflow boundaries, and fused rounding.
With NEG/ABS/OMOD, DX9 FMA measures 18.4 ns on AVX2 and 8.9 ns on
AVX-512 (6.2× and 12.7× scalar speed on the Ryzen 9 7950X3D).

Sparse WMMA covers the four FP16/BF16 `v_swmmac_*_16x16x32_*` forms,
with FP32 or matching packed output. D is the in/out accumulator, A contains
2:4 compressed rows, B is dense, and a separate index VGPR selects positions.
Both index keys and all A/B low/high negation combinations are supported on
scalar, x86-64-v3 (eight columns) and x86-64-v4 (sixteen columns) paths.
The B negation modifiers apply to selected pair positions. Metadata pairs must
contain strictly increasing positions; malformed sparse metadata is outside the
API contract. These forms expose loose FP32 FMA accumulation, with nearest-even
packed narrowing and `GOC_FP16_OVFL` support for finite FP16 overflow.

Tests retain GPU digests for 32,768 logical results across all four forms and
all 32 modifier combinations. They cover every EXEC lane, overlapping inputs
and destinations (including the index register), duplicate destinations,
packed-rounding ties and finite overflow versus infinity. All operands are
snapshotted before masked stores. Benchmarks include resetting the accumulator
on every call, equally for all CPU paths. Pinned-core Ryzen 9 7950X3D timings
measured 4.77–11.19x for v3 and 4.81–11.84x for v4 versus scalar, including
negation and index-key selection (seven samples, each at least 10 ms).

Sparse FP8/BF8 WMMA covers all four `v_swmmac_f32_16x16x32_*_*`
combinations. A uses two VGPRs, B four, the index one, and in/out D eight.
Both index keys are supported; these instructions have no negation or CLAMP
modifiers. The scalar and shared floating-point v3/v4 backends use loose FP32
FMA semantics, with full input snapshots and masked destination stores.
Tests check 32,768 GPU-captured logical results, all 256 encodings of each
input type, both index keys, every EXEC lane and destination/source/index
aliases. The exhaustive encoding checks include subnormals, signed zeros,
BF8 infinities and both formats' NaNs. Pinned-core timings on the same CPU
measured 7.30–8.17x for v3 and 7.77–8.75x for v4, including index-key selection
and the per-call accumulator reset (seven samples, each at least 10 ms).

Sparse integer WMMA covers `v_swmmac_i32_16x16x32_iu8`,
`v_swmmac_i32_16x16x32_iu4`, and `v_swmmac_i32_16x16x64_iu4`, with signed or
unsigned factors, CLAMP, loose and empirical exact semantics, and scalar/v3/v4
paths. K=32 supports both index keys; K=64 consumes all metadata bits and rejects
index-key selection. The in/out accumulator uses eight VGPRs. Inputs are read
before masked stores, and host FP state is preserved.

GPU captures establish two-stage CLAMP behavior: K=32 saturates after compressed
positions 0–7 and again after 8–15. K=64 saturates after positions 0–7 plus 16–23,
then after 8–15 plus 24–31. This differs from rocjitsu's current final-sum model.
Tests check 163,840 captured results across every modifier combination, plus
EXEC masks, aliases, and a literal witness where the two stages cancel
mathematically but intermediate saturation changes the result. V3 processes
eight columns and v4 sixteen, with modifiers retained on both paths. Pinned-core
Ryzen 9 7950X3D measurements show 1.39–1.86x for v3 and 1.77–2.34x for v4,
including signed CLAMP and index-key selection; the accumulator reset is included
in every path's timing (seven samples, each at least 10 ms).

FP32 interpolation (`v_interp_p10_f32`, `v_interp_p2_f32`) broadcasts parameter
values within each four-lane quad, then performs an FMA. P10 takes A from quad
lane 1 and C from quad lane 0; P2 takes A from quad lane 2 and C from the current
lane. B always comes from the current lane. The broadcasts read inactive source
lanes too, while EXEC controls destination writes. NEG_A/B/C and CLAMP stay on
both eight-lane v3 and sixteen-lane v4 paths. All eight WAIT_EXP values are
accepted; this scheduling field has no effect on synchronous CPU execution.

The implementation follows sections 12.3 and 16.13 of the
[AMD RDNA4 ISA guide](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf).
Tests compare 8,192 GPU-captured results across modifiers and wait counts, and
36,864 results across empty/full/alternating/single-lane EXEC masks. Further tests
cover aliases, unaligned storage, special values and random FP32 inputs against
a higher-precision reference. These forms support loose semantics and require
host nearest-even rounding with denormals enabled; FP exception flags may change.
Pinned-core Ryzen 9 7950X3D timings show 4.39–5.13x for v3 and 17.19–19.38x
for v4 versus scalar, including NEG and CLAMP (seven samples, each at least
10 ms).

Mixed FP16 interpolation adds `v_interp_p10_f16_f32`, `v_interp_p2_f16_f32`
and their `p10_rtz`/`p2_rtz` variants with the same quad broadcasts. P10 reads
FP16 A/C, FP32 B and produces FP32; P2 reads FP16 A, FP32 B/C and rounds directly
to FP16, preserving the other destination half. NEG, CLAMP, half selectors,
WAIT_EXP and finite-overflow saturation remain on the SIMD paths. RTZ forms
round toward zero; finite FP16 overflow already saturates under RTZ.

The scalar and v3 paths share the existing mixed-FMA rounding machinery derived
from rocjitsu. P10 uses eight FP32 SIMD lanes; P2 uses four FP64 lanes for its
intermediate arithmetic, retaining discarded-bit information before narrowing.
RTZ P10 temporarily changes and restores the x86 thread's rounding control;
all forms require host nearest-even rounding with denormals enabled. Tests
compare 524,288 GPU-captured results (canonicalizing NaN payloads), an independent
integer FMA oracle, rounding boundaries, modifiers, masks, aliases, half
preservation, overflow settings and restoration of host rounding. Pinned-core
Ryzen 9 7950X3D measurements show 5.76–5.96x for P10, 11.99–12.38x for RTZ
P10, and 3.78–4.10x for the P2 forms versus scalar, including modifiers
(seven samples, each at least 10 ms).

`v_rcp_iflag_f32` computes a reciprocal and returns sticky guest exception
status through a separate scalar output. Active signed-zero and subnormal inputs
raise `GOC_RDNA4_EXCEPTION_INT_DIV0`; CLAMP suppresses a new cause but preserves
any pre-existing status bit. Other status bits survive unchanged. The scalar
status is written after VGPR stores, taking precedence if its storage aliases
an input or output word. Empty EXEC leaves VGPRs untouched and copies the incoming
status to the scalar output.

The scalar, eight-lane v3 and sixteen-lane v4 paths support every ABS/NEG/OMOD/
CLAMP combination. Input/output subnormals always flush, independently of guest
FP-mode settings. Numeric results use loose semantics; host nearest-even rounding
and enabled denormals are required, and host exception flags may change.
The status rules correct two details in rocjitsu's classifier: flushed subnormal
inputs also raise INT_DIV0, and CLAMP suppresses the new cause.

Tests use GPU captures of 4,718,592 lane results and 147,456 exact status values,
covering modifiers, full/empty/partial EXEC, denormal modes and initial sticky
flags. Further tests cover every single active/inactive lane, numeric special
values, unaligned storage, source/destination/status aliases, invalid flags and
strict-semantics rejection. The full capture also matches the compiled API
across every available CPU level within two numeric ULPs, with exact zeros,
infinities and status bits. Pinned-core Ryzen 9 7950X3D timings show 1.47–2.30x
for v3 and 3.78–5.30x for v4 versus scalar, including modifiers and CLAMP
(seven samples, each at least 10 ms).

Pseudo-scalar math supports `v_s_exp_f16/f32`, `v_s_log_f16/f32`,
`v_s_rcp_f16/f32`, `v_s_rsq_f16/f32` and `v_s_sqrt_f16/f32`. These instructions
read one SGPR value and write one SGPR result. They execute regardless of EXEC,
including empty EXEC. FP16 reads the low input half and clears the destination's
upper half. ABS/NEG, OMOD, CLAMP and FP16 overflow saturation are supported;
source/destination half selectors are not part of these instructions.

These loose implementations borrow stage ordering and narrowing machinery from
rocjitsu. FP32 always flushes input/output subnormals. FP16 obeys
`GOC_FP_FLUSH_INPUT_DENORMALS` and `GOC_FP_FLUSH_OUTPUT_DENORMALS`, with both
clear by default. Rounding to the result format happens before OMOD. Nonzero
OMOD flushes tiny values before and after scaling; a pre-existing zero becomes
positive, while newly created negative underflow retains its sign. The latter
behavior corrects rocjitsu's half finalization based on GFX1201 captures.
Host nearest-even rounding and enabled denormals are required; rounding is
preserved, but arithmetic exception flags may change. Strict empirical-exact
requests are rejected.

The compiled API matches 10,485,760 distinct GPU outputs within one FP16 or two
FP32 ULPs, requiring exact zeros and infinities and allowing NaN payload variation.
The 31,457,280-result capture also verifies identical full, empty and partial
EXEC behavior. Committed fixtures retain boundary and random samples across every
modifier and FP-state combination; further tests cover scalar aliasing, ignored
EXEC, invalid flags, strict semantics, denormal stages and overflow. Each call
has only one scalar result, so all CPU levels use the same implementation. The
benchmark measures these scalar calls with default and modified FP settings.
Pinned-core Ryzen 9 7950X3D timings for those workloads range from 2.2 to
14.1 ns per scalar instruction (seven samples, each at least 10 ms).

Floating-point comparison supports all 14 RDNA4 predicates for FP16/FP32/FP64,
including CMP and CMPX (84 entry points). CMP returns a scalar condition mask;
CMPX returns replacement EXEC. Both clear inactive bits, and their scalar output
may alias any input word. ABS/NEG and independent FP16 source-half selectors
remain on every SIMD path. Both semantics use the raw integer ordering model
borrowed from rocjitsu, with signed zeros equal and the prescribed ordered or
unordered behavior for every NaN, including signaling NaNs.

`GOC_FP_FLUSH_INPUT_DENORMALS` flushes guest input subnormals to signed zero
after source modifiers. Its default value preserves them. This setting is
independent of host DAZ/FTZ and rounding, and these comparisons preserve the
complete host FP environment. CLASS also accepts the flag but still classifies
raw encodings. Pseudo-scalar math also accepts input/output flushing settings;
other instructions reject settings whose behavior is not yet implemented.
`GOC_FP_FLUSH_OUTPUT_DENORMALS` has no effect on comparison masks.

The v3 path processes eight FP16/FP32 or four FP64 lanes; v4 processes sixteen
or eight respectively. Tests check 6,881,280 GPU-captured masks covering every
predicate, modifier, half selector, five EXEC masks and all four GPU denormal
modes. Independent tests cover boundary pairs, every FP16 encoding, source and
output aliases, unaligned storage, further EXEC masks, invalid flags, and host
rounding/denormal/exception-state preservation. The benchmark includes default,
modified, and modified-with-input-flushing cases for every instruction.
Pinned-core Ryzen 9 7950X3D timings show 3.70–8.13x for v3 and 4.82–12.23x
for v4 versus scalar, including input flushing (seven samples, each at least
10 ms). FP64 gains alone are 3.70–4.70x and 4.82–6.62x respectively.

Integer comparison supports LT/EQ/LE/GT/NE/GE for signed and unsigned
16-, 32- and 64-bit operands, including every corresponding CMPX form (72 entry
points). CMP returns a scalar condition mask and CMPX a replacement EXEC mask;
inactive bits are zero. The 16-bit forms support independent source-half
selection. The scalar destination may alias any input word, with all source
reads completed before its write. These integer models follow rocjitsu and
support both semantics while preserving all host FP state.

Both SIMD paths retain all modifiers. Comparisons process eight lanes on v3
and sixteen on v4, including 64-bit inputs: high-word ordering and unsigned
low-word tie-breaking avoid assembling narrower vectors of 64-bit elements.
Tests compare 1,474,560 GPU-captured masks across every instruction and half
selector, plus independent boundary/reference tests, source/output aliases,
unaligned storage, EXEC masks, invalid flags and host FP-state preservation.
Pinned-core Ryzen 9 7950X3D timings show 1.24–7.12x for v3 and 1.26–9.62x
for v4 versus baseline, including half selectors (seven samples, each at least
10 ms). Simple 32-bit predicates have the smallest gains because the baseline
compiler already lowers them efficiently.

Floating-point classification (`v_cmp_class_f16/f32/f64` and their CMPX forms)
tests raw source encodings against the ten-bit class mask in B. CMP returns a
scalar condition mask; CMPX returns a replacement EXEC mask. Inactive bits are
zero, including for empty EXEC. The scalar output may alias any input word.
ABS/NEG and FP16 source-half selectors stay on the eight-lane v3 and sixteen-lane
v4 paths. Both semantics use the integer class model borrowed from rocjitsu,
preserving host rounding, denormal controls and exception flags. Subnormals
remain a distinct class even when GPU denormal flushing is enabled; signaling
NaNs are classified without quieting them.

Tests compare 983,040 GPU-captured masks across all three formats, CMP/CMPX,
modifiers, five EXEC masks and both GPU denormal modes. Further tests cover all
1,024 class masks, every class, source/output aliases, unaligned storage, a
larger set of EXEC masks, invalid flags and complete host FP-state preservation.
Pinned-core Ryzen 9 7950X3D measurements show 5.25–6.67x for v3 and
6.73–9.45x for v4 versus scalar, including ABS/NEG and half selection (seven
samples, each at least 10 ms).

Conditional selection (`v_cndmask_b32` and `v_cndmask_b16`) selects B for set
bits in a separate wave32 condition mask and A for clear bits. EXEC independently
controls destination writes. ABS/NEG modify only source sign bits; all payload
bits, including signaling NaNs, are preserved. The 16-bit form supports every
source/destination half selector and preserves the unwritten destination half.
Both forms preserve host FP state and have scalar and sixteen-lane v4 paths
with all their modifiers. AVX2 candidates did not provide a substantial gain,
so v3 CPUs use the portable path. OMOD and CLAMP are not supported by these
instructions.

Tests compare all source-modifier and half-selector combinations against
18,874,368 GPU-captured outputs, covering every FP16 input encoding and both
choices of source. Additional tests cross modifiers with EXEC masks, independent
condition masks, source/destination aliases and unaligned storage. Pinned-core
Ryzen 9 7950X3D measurements show 4.39–4.50x for B32 and 2.09–2.13x for B16
on v4 versus scalar, including ABS/NEG and half selection (seven samples, each
at least 10 ms).

Trigonometric range reduction (`v_trig_preop_f64`) supports scalar, four-lane
x86-64-v3 and eight-lane x86-64-v4 table lookups, including ABS/NEG, OMOD and
CLAMP. Both loose and empirical-exact semantics use an integer implementation
that preserves host FP state. The table model comes from rocjitsu, with two
GFX1201 corrections: the binary expansion ends at bit 1184, and OMOD flushes a
subnormal lookup result before scaling as well as flushing a tiny scaled result.
Only A's encoded exponent and B's low five bits affect the result; input NaNs
and infinities use their encoded exponent too.

Tests check all 2,097,152 combinations of exponent, selector and modifiers
against GPU-captured digests. A second GPU capture with varied fraction bits,
including NaNs, produced identical results. Mask/alias tests cover all 25
assignments of the two destination registers among five backing registers,
including duplicate destinations, plus unaligned storage and host rounding and
exception-state preservation. Pinned-core Ryzen 9 7950X3D measurements show
1.48–1.59x for v3 and 3.65–3.80x for v4 versus scalar, including ABS/NEG,
scaling and CLAMP (seven samples, each at least 10 ms).

Lighting multiply (`v_mullit_f32`) supports all 512 combinations of source
ABS/NEG, output scaling and CLAMP on scalar, eight-lane v3 and sixteen-lane v4.
After source modifiers, invalid lighting inputs (nonpositive/NaN C, or B equal
to negative FLT_MAX, negative infinity or NaN) return negative FLT_MAX before
output modifiers. Otherwise a zero factor produces positive zero, and nonzero
factors multiply normally. OMOD flushes tiny unscaled results to positive zero,
while underflow from scaling a normal result retains its sign. CLAMP maps NaN
and negative zero to positive zero. Loose semantics require host nearest-even
rounding with denormals enabled; exception flags may change.

Tests check 65,536 GPU-captured special-value results (ignoring NaN payloads),
plus all 512 modifiers crossed with EXEC masks, unaligned storage and every
whole-register destination/source alias. Pinned-core Ryzen 9 7950X3D timings
measured 3.17–3.70x for v3 and 8.02–9.67x for v4 versus scalar, including
ABS/NEG, scaling and CLAMP (seven samples, each at least 10 ms).

Wide integer MAD covers `v_mad_co_u64_u32` and `v_mad_co_i64_i32`, the RDNA4
names for unsigned/signed 32x32 multiplication plus a 64-bit accumulator. A/B use
one VGPR each; C/D use low/high pairs. CLAMP saturates to the corresponding
64-bit integer range. The scalar output contains bit 64 of the full sum. For
signed MAD this is its extended sign, not a signed-overflow indication; GPU
captures establish this distinction from rocjitsu's current handler.

Scalar and v3/v4 paths support loose and empirical exact semantics, CLAMP,
every EXEC mask and all whole-register aliases. V3 processes four lanes and v4
eight; stores commit D0, D1, then the scalar output after all input reads.
Inactive scalar bits are cleared, including zero EXEC, while inactive VGPR lanes
are preserved. Host FP state is untouched. Tests compare 49,152 GPU-captured
result/mask pairs, use an independent 128-bit reference for random inputs, and
cover signed-overflow witnesses, all destination-pair aliases, shared sources,
unaligned storage, scalar-output overlap and host FP-state preservation. Pinned-core
Ryzen 9 7950X3D timings measured 1.44–2.03x for v3 and 5.76–7.52x for v4 versus
scalar, including CLAMP (seven samples, each at least 10 ms).

Carry/borrow arithmetic covers `v_add_co_u32`, `v_sub_co_u32`,
`v_subrev_co_u32`, and their `co_ci` forms. The scalar carry/borrow output follows
D in assembly operand order; CI forms take the input mask by value after A/B.
CLAMP saturates the VGPR result while preserving the unsaturated carry/borrow
indication. Inactive scalar output bits are cleared, including for zero EXEC;
inactive VGPR lanes remain unchanged. The scalar output is written last and may
share storage with a VGPR. All paths preserve host FP state.

Scalar, v3 (eight lanes), and v4 (sixteen lanes) paths support both semantics,
CLAMP and all aliases. GPU captures verify full, partial and zero EXEC behavior,
including saturation. Independent widened-integer references cover boundary
Cartesian products and random inputs. Tests cross all 85 EXEC masks with all 85
input-carry mask patterns, check shared sources, unaligned storage, destination
aliases and scalar-output overlap, and verify host FP-state preservation. Pinned-core
Ryzen 9 7950X3D measurements show 1.43–2.09x for v3 and 5.36–7.84x for v4 versus
scalar, including CLAMP (seven samples, each at least 10 ms).

Division fused post-scaling covers `v_div_fmas_f32` and `v_div_fmas_f64`, with
loose and empirical exact semantics borrowed from rocjitsu. The API takes the
implicit wave32 VCC condition mask by value after A/B/C. Set lane bits select
post-scaling by 2^64 or 2^128 when C's modified encoded exponent exceeds its bias,
or by the reciprocal power otherwise. Scaling happens before the final rounding,
avoiding intermediate overflow and double rounding at subnormal boundaries.

All ABS/NEG, OMOD and CLAMP combinations have scalar, v3 and v4 implementations.
These use integer arithmetic, preserving host FP rounding and exception state.
V3 evaluates four lanes at a time and v4 eight, in both formats; FP64 products
retain all 106 bits before alignment and rounding. Every EXEC mask and whole-VGPR
alias is supported, including FP64 cross-half aliases. Active OMOD rounds at
normal precision before flushing tiny results, a hardware detail beyond simply
applying OMOD to rocjitsu's already-rounded result.

Tests retain 262,144 GPU-captured outputs from Cartesian and deterministic random
corpora, plus literal fused-rounding, overflow and cancellation witnesses. They
also cover every modifier, mixed condition masks, shared sources, unaligned
storage, destination aliases and preservation of all host rounding modes. Pinned-core
Ryzen 9 7950X3D timings measured 2.47–2.81x for v3 and 3.78–4.36x for v4 versus
scalar, including modified forms (seven samples, each at least 10 ms).

Division pre-scaling covers `v_div_scale_f32` and `v_div_scale_f64`, with loose
and empirical exact semantics adapted from rocjitsu. B is the denominator, C the
numerator, and A must equal B or C after source NEG modifiers. The result includes
a `uint32_t` scalar condition mask in addition to the scaled VGPR value. Inactive
condition bits are cleared, even for zero EXEC, while inactive VGPR lanes remain
unchanged. NEG, OMOD and CLAMP stay on SIMD paths; ABS is not supported by this
instruction encoding. FP64 supports cross-half aliases. Host FP rounding and
exception state are preserved. Tests compare 147,456 GPU-captured value/condition
pairs across all modifiers and both source roles, and exercise masks, aliases,
random inputs and all host rounding modes. Pinned-core Ryzen 9 7950X3D measurements
show 2.21–3.97x speedups for v3 and 3.66–6.40x for v4, including modified forms
(seven samples per case, each at least 10 ms).

Division fixup covers `v_div_fixup_f16`, `v_div_fixup_f32`, and
`v_div_fixup_f64`. A supplies a provisional quotient, B the original denominator,
and C the original numerator. These instructions repair the quotient's sign and
exceptional cases; they do not compute a general division. C's NaN takes priority
over B's NaN. FP32/FP64 also implement the extreme encoded-exponent underflow
shortcut. FP16 supports all source/destination half selectors and preserves the
unselected destination half; FP64 uses low/high VGPR pairs with full cross-half
alias support and D1 winning if the two destination pointers coincide.

The rocjitsu-derived bit model supports loose and empirical exact semantics,
all source ABS/NEG, OMOD and CLAMP, and `GOC_FP16_OVFL` for FP16. Hardware captures
establish two FP16 details beyond rocjitsu's promoted model: OMOD flushes a
subnormal provisional result before scaling, and saturation of a nonfinite
provisional quotient for finite nonzero operands happens before OMOD. Integer-only
scalar and SIMD paths preserve host FP state and retain vectorization for every
modifier. V3 processes eight FP16/FP32 or four FP64 lanes; v4 processes sixteen
FP16/FP32 or eight FP64 lanes.

Tests verify 122,880 captured RX 9070 outputs using compact digests of three
4,096-input Cartesian corpora across ten modifier/overflow configurations.
Independent numeric references cover all 8,192 FP16 selector/modifier combinations,
all 512 FP32/FP64 modifier combinations, both saturation settings, special values
and random bits. Additional coverage includes 85 EXEC masks, unaligned storage,
all FP64 destination-pair aliases, shared sources and host rounding/exception
preservation. Pinned-core benchmarks on the development Ryzen 9 7950X3D measured
2.01–4.01x for v3 and 3.66–8.44x for v4 versus scalar (seven samples, each at
least 10 ms), including nondefault modifiers.

Cube-map arithmetic covers `v_cubeid_f32`, `v_cubesc_f32`, `v_cubetc_f32`,
and `v_cubema_f32`, with scalar, eight-lane v3 and sixteen-lane v4 paths.
The borrowed rocjitsu bit-level model supports both loose and empirical exact
semantics on every path. A/B/C hold X/Y/Z; face IDs are 0/1 for positive/negative
X, 2/3 for Y and 4/5 for Z. Z wins magnitude ties, then Y, then X. The major-axis
instruction returns twice the **signed** major component.

Comparisons flush subnormal magnitudes, while coordinate selection preserves
selected source bits and quiets NaNs. Nonzero OMOD flushes subnormal inputs and
outputs independently of host FP settings; zero and NaN handling follows the
captured hardware rules. All source ABS/NEG, OMOD and CLAMP combinations stay
vectorized. The implementation uses integer operations throughout, preserving
host FP state and providing identical results under every host rounding mode.

RX 9070 captures matched the model for 87,040 outputs. Regression tests retain
32 literal input/output cases and compact output digests covering the complete
4,096-input Cartesian hardware corpus across 20 instruction/modifier cases.
Independent numeric references test all 512 modifier combinations, exceptional
values and random bit patterns; mask/alias tests cross every modifier with all
85 EXEC masks and every destination alias. Benchmarks use full masks and default
or mixed modifiers. Pinned-core measurements on the development Ryzen 9 7950X3D
showed 2.42–6.48x for v3 and 4.29–11.99x for v4 versus scalar (seven samples,
each at least 10 ms).

FP8/BF8 narrowing covers `v_cvt_pk_fp8_f32`, `v_cvt_pk_bf8_f32`,
`v_cvt_sr_fp8_f32`, and `v_cvt_sr_bf8_f32`. Packed forms round two FP32
sources to nearest-even and replace the selected destination half. Stochastic
forms take the seed directly from B and replace one selected destination byte;
there is no internal random-number generator. Both preserve the unselected
parts of D and support source ABS/NEG, full EXEC masking and every whole-register
alias. `GOC_FP16_OVFL` saturates finite overflow while preserving the input
infinity behavior. Input NaNs produce canonical `0xff` (FP8) or `0xfe` (BF8).

The conversion model adapts rocjitsu's integer rounding logic, with RX 9070
captures establishing RDNA4-specific NaN, infinity and stochastic-underflow
behavior. Subnormal stochastic conversion first aligns the significand,
discarding shifted-out bits, then adds the seed's high 20/21 bits. All four
instructions have scalar, eight-lane v3 and sixteen-lane v4 paths using only
integer operations; every supported modifier stays vectorized and host FP state
is preserved. These APIs currently expose loose semantics.

Tests retain 116 hardware input/seed cases across 20 configurations, and cover
rounding boundaries, exceptional values, stochastic underflow, both overflow
settings, every modifier and destination selector, 85 masks, unaligned storage,
aliases and all host rounding modes. Pinned-core benchmarks on the development
Ryzen 9 7950X3D measured 1.82–3.57x for v3 and 5.09–6.37x for v4 versus scalar
(seven samples, each at least 10 ms), including modified and saturating cases.

`v_cvt_off_f32_i4` interprets the low nibble as signed i4 and converts it to
FP32 divided by 16, then applies OMOD/CLAMP. Higher source bits are ignored.
`v_cvt_pk_u8_f32` rounds FP32 A to nearest-even, saturates to [0,255], maps
NaNs to zero, and replaces byte `(B & 3)` of C. Its ABS/NEG modifiers apply only
to A; CLAMP is accepted without numeric effect. Per-lane byte selection uses
SIMD variable shifts and stays vectorized with every supported modifier.
Both expose loose semantics with scalar, eight-lane v3 and sixteen-lane v4
paths, full EXEC masking and all source/destination aliases.

RX 9070 (`gfx1201`) captures of 36 inputs across eight instruction/modifier
combinations establish these rules. In particular, byte packing rounds 1.5 to
2, whereas rocjitsu's current handler truncates. Tests preserve those hardware
results and cover every nibble, FP32 neighbors of every byte-rounding midpoint,
all byte positions and modifiers, ignored selector/source bits, 85 masks,
unaligned storage and every whole-register alias layout. Results are independent
of host rounding; nibble-offset conversion also preserves FP exception flags.
Benchmarks use full EXEC, mixed per-lane byte positions, and default/modified
instructions. On the development Ryzen 9 7950X3D, pinned-core timings showed
1.11–5.13x for v3 and 5.02–18.63x for v4 versus scalar (seven samples, each
at least 10 ms); v3's larger offset-conversion gain is on the modified workload.

Integer conversions cover `v_cvt_i32_i16`, `v_cvt_u32_u16`,
`v_cvt_pk_i16_i32`, and `v_cvt_pk_u16_u32`. Widening selects either source
half with `GOC_ALU_HIGH_A` and sign- or zero-extends it. Packing follows
rocjitsu's saturating conversion handlers, placing A/B into the low/high
result halves; it accepts no numeric modifiers. All four support loose
semantics, EXEC masking, aliases, and preservation of the host FP environment.

Sixteen-lane v4 paths cover all four operations. Eight-lane v3 paths cover
packing; widening candidates were slightly slower than baseline and were
removed. The signed scalar packing path uses an unsigned interval test to
avoid expensive 64-bit comparisons. Tests cover every half encoding, saturation
boundaries, full-word inputs, all source/destination alias layouts, 85 masks,
unaligned storage, validation, and host FP-state preservation. Benchmarks use
full EXEC and both half selectors where applicable. On the development Ryzen 9
7950X3D, pinned-core measurements showed 1.21–1.43x for v3 packing and
3.82–6.14x for v4 versus scalar (seven samples, each at least 10 ms).

Normalized conversions cover `v_cvt_pk_norm_i16_f32`, `v_cvt_pk_norm_u16_f32`,
`v_cvt_pk_norm_i16_f16`, `v_cvt_pk_norm_u16_f16`, `v_cvt_norm_i16_f16`, and
`v_cvt_norm_u16_f16`. Signed results scale by 32767 and saturate to
[-32767,32767]; unsigned results scale by 65535 and saturate to [0,65535].
Both round once to nearest-even and map NaNs to zero. Packed forms write A/B
into the two destination halves; unary forms preserve the unselected half.
Source ABS/NEG and applicable half selectors remain on the scalar, eight-lane
v3 and sixteen-lane v4 paths. CLAMP has no numeric effect; unary OMOD is also
accepted without numeric effect. All six expose loose semantics.

The scalar model borrows rocjitsu's exact double product. SIMD borrows its
`round_normalized_simd` FMA-residual correction: a rounded FP32 product can
land on a false integer midpoint, so the residual determines the proper side.
Tests use an independent integer-significand oracle and include every FP16
encoding, FP32 neighbors of every rounding boundary, all modifiers, 85 EXEC
masks, every whole-register alias layout, and host-rounding independence.
Another 54 input cases captured on the RX 9070 (`gfx1201`) cover 20 opcode/modifier
combinations, including false ties, saturation and NaNs. Full-EXEC benchmark
rows compare default and modified instructions. On the development Ryzen 9
7950X3D, pinned-core measurements showed 5.27–8.22x for v3 and 12.25–17.23x
for v4 versus scalar (seven samples, each at least 10 ms).

Packed FP32 conversions cover `v_cvt_pk_rtz_f16_f32`, `v_cvt_pk_i16_f32`,
and `v_cvt_pk_u16_f32`: A converts into the low destination half, B into the
high half. All source ABS/NEG combinations stay on the scalar, eight-lane v3
and sixteen-lane v4 paths. CLAMP is accepted without numeric effect, as is
OMOD for the FP16 RTZ form. Integer forms truncate, saturate overflow and map
NaNs to zero. RTZ borrows rocjitsu's `f32_to_f16_rtz` integer conversion,
quieting source NaNs to match RDNA4; finite overflow saturates regardless of
`GOC_FP16_OVFL`. All three expose loose semantics independent of host rounding.

Sixteen literal input pairs captured on the RX 9070 (`gfx1201`) verify nine
opcode/modifier combinations with `FP16_OVFL` both clear and set. Tests also
cover every FP32 exponent, every FP16 boundary and adjacent FP32 values,
random words, all modifiers, and 85 EXEC masks crossed with whole-register
aliases and unaligned storage. The RTZ reference independently searches the
FP16 representable values; integer conversion uses an integer-significand
reference. Benchmarks compare default and modified full-EXEC workloads. On the
development Ryzen 9 7950X3D, pinned-core measurements showed 1.16–3.95x for v3
and 3.99–14.28x for v4 versus scalar (seven samples, each at least 10 ms).

FP8/BF8 expansion covers `v_cvt_f32_fp8`, `v_cvt_f32_bf8`,
`v_cvt_pk_f32_fp8`, and `v_cvt_pk_f32_bf8`, borrowing rocjitsu's OCP E4M3FN
and E5M2 decoders. Single-result forms accept `GOC_CVT_BYTE_0` through
`GOC_CVT_BYTE_3`; packed forms accept `GOC_ALU_HIGH_A` and write the selected
half's two bytes into two FP32 VGPRs. They reject ABS/NEG, OMOD and CLAMP.
All selectors remain on the eight-lane v3 and sixteen-lane v4 paths. Finite
values expand exactly, including subnormals and signed zeros; NaNs become
sign-preserving canonical quiet NaNs. These APIs expose loose semantics and
preserve the host FP environment under every rounding mode.

Tests cover every byte encoding, every packed byte pair, all selectors,
unselected bits, and all source/destination alias layouts crossed with 85 EXEC
masks and unaligned storage. If the two destinations alias, the second wins.
Literal format-boundary cases complement an independent integer-significand
reference. Benchmarks use full EXEC with default and upper-byte/half selectors.
On the development Ryzen 9 7950X3D, pinned-core timings showed 2.46–3.26x for
v3 and 9.22–13.88x for v4 versus scalar (seven samples, each at least 10 ms).

The four `v_cvt_f32_ubyte0` through `v_cvt_f32_ubyte3` instructions extract
one unsigned byte from a VGPR and convert it to FP32, following rocjitsu's
byte-conversion handlers. All eight OMOD/CLAMP combinations stay on the
available SIMD paths; source ABS/NEG and half selectors are invalid. Results
are exact for all byte values and output modifiers, so all host rounding modes
produce the same bits and preserve existing FP exception flags. The API exposes
loose semantics. Tests exhaust byte values, byte positions, unselected source
bits and modifiers, and cross 85 EXEC masks with aliases and unaligned storage.
Eight-lane v3 and sixteen-lane v4 paths support all modifiers. Benchmarks cover
full-EXEC default and combined OMOD/CLAMP workloads. On the development Ryzen 9
7950X3D, pinned-core measurements showed v3 roughly tied with scalar for default
instructions and 2.4–2.6x faster with modifiers; v4 was 3.7–5.0x faster for
defaults and 9.3–10.7x with modifiers (seven samples, each at least 10 ms).

Six FP16 conversions cover `v_cvt_f16_i16`, `v_cvt_f16_u16`,
`v_cvt_i16_f16`, `v_cvt_u16_f16`, `v_cvt_f16_f32`, and `v_cvt_f32_f16`.
Each operand occupies one VGPR. Half selectors preserve the unused destination
half; floating sources support ABS/NEG, and floating destinations support
OMOD/CLAMP. Integer outputs truncate with saturation and NaN-to-zero, accepting
CLAMP/OMOD without numeric effect. FP16 narrowing uses nearest-even with
`GOC_FP16_OVFL` saturation before OMOD and again after scaling. Active OMOD
flushes initially tiny values to positive zero, while division underflow from
normal values preserves the sign of zero. All six expose loose semantics only.

Scalar, eight-lane x86-64-v3 and sixteen-lane x86-64-v4 paths support every valid
modifier, full EXEC masking and in-place operation. Tests exhaust all 65,536
half/integer input encodings, every FP16 narrowing midpoint and its adjacent
FP32 values, all modifiers, both overflow modes, and 85 masks crossed with
aliasing and unaligned storage. An independent integer-bit reference checks
random full words and rounding boundaries. Another 42 literal input cases
were captured on an RX 9070 (`gfx1201`) with nearest-even rounding and denormals
enabled, each under eight instruction/modifier combinations and both overflow
modes. These establish rounding-before-OMOD, the extra precision used for
tininess at the normal boundary, NaN quieting, signed-zero and overflow rules.
Benchmark rows use full EXEC, overflow saturation enabled, and both default
and modified instructions. On the development Ryzen 9 7950X3D, pinned-core
measurements gave 4.12–10.28x for v3 and 11.07–19.01x for v4 versus scalar
across those workloads (seven samples, each at least 10 ms).

Six FP64 conversions cover `v_cvt_f64_i32`, `v_cvt_f64_u32`,
`v_cvt_i32_f64`, `v_cvt_u32_f64`, `v_cvt_f64_f32`, and `v_cvt_f32_f64`.
FP64 operands occupy two VGPRs, low word first. Source/destination halves may
alias in any combination; if destination halves share storage, the high word
wins. Integer inputs convert exactly to FP64 before output scaling and CLAMP.
FP64-to-integer truncates with saturation and NaN-to-zero, accepting source
ABS/NEG and numerically ignoring CLAMP/OMOD. FP32/FP64 conversions support all
32 ABS/NEG/OMOD/CLAMP combinations. Narrowing rounds to FP32 before applying
OMOD, including cases where scaling first would avoid overflow or underflow.
All six currently expose loose semantics only.

Eight-lane x86-64-v4 paths cover all six; four-lane v3 paths cover all except
signed FP64-to-integer, whose candidate was roughly tied with baseline.
No supported modifier forces an otherwise available SIMD path to scalar.
Tests use an independent integer-bit reference, literal rounding/saturation
witnesses, narrowing midpoints across the FP32 range, every source exponent,
random full words, all modifiers, and all 32 source/destination layouts crossed
with 85 EXEC masks. They also check unaligned storage, validation without writes,
and host-rounding independence for integer outputs. Benchmark rows compare
unmodified and modified full-EXEC workloads.

Six 32-bit numeric conversions cover `v_cvt_f32_i32`, `v_cvt_f32_u32`,
`v_cvt_i32_f32`, `v_cvt_u32_f32`, `v_cvt_nearest_i32_f32`, and
`v_cvt_floor_i32_f32`. Float-to-integer conversions saturate overflow and map
NaNs to zero. The ordinary forms truncate; `NEAREST` breaks ties toward positive
infinity, and `FLOOR` rounds downward. Integer-to-FP32 uses host nearest-even
rounding followed by OMOD/CLAMP. Float-to-integer accepts ABS/NEG; CLAMP is a
numeric no-op, as is OMOD on the two truncating forms. The explicit rounding
forms reject OMOD. GPU exception reporting is not modeled. These entry points
currently expose loose semantics only.

All six have scalar and 16-lane x86-64-v4 implementations. Eight-lane v3
implementations cover integer-to-float, NEAREST, and FLOOR, including every
accepted modifier combination. The truncating v3 candidates gained only 7–10%
over baseline and were removed. Tests use an independent integer-bit reference,
literal tie/overflow/NaN witnesses, precision boundaries, random words, all
modifiers, all mask patterns, unaligned storage, and in-place aliases. They also
check that float-to-integer rounding is independent of the host rounding mode.
Benchmark rows exercise full EXEC with default and modified operands.

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

Bit-count instructions implement leading zeros (`CLZ`), trailing zeros (`CTZ`),
leading sign bits (`CLS`), and population count plus a wrapping accumulator
(`BCNT`). CLZ/CTZ return `0xffffffff` for zero; CLS counts the sign bit itself
and returns `0xffffffff` for all-zero or all-one values. These rules and the
integer bit-propagation approach follow rocjitsu's bit-scan helpers. V3 uses
byte lookup/shuffle population count; v4 additionally uses native leading-zero
count. All host FP state is preserved, including exception flags.

`MBCNT_LO` counts source bits below the physical lane index, capped at 32;
`MBCNT_HI` counts source bits below the lane index minus 32, or zero for lanes
0–31. Both add `B` with wrapping. The lane index does not depend on EXEC.
Dedicated `goc_rdna4w64_` forms take 64 words per VGPR because two wave32 calls
cannot reproduce the upper-half lane indices. Wave32 MBCNT_HI is a masked copy
of `B`; its v3 candidate showed little benefit and was removed. Tests cover every pair of bit
positions, complements, all population counts, accumulator wrapping, sentinel
results, EXEC masks and aliases, including an in-place LO/HI sequence that
constructs physical lane numbers under sparse EXEC masks. No modifiers apply.

Packing supports `v_sat_pk_u8_i16` and `v_pack_b32_f16`. The former saturates
each signed I16 half to U8 and packs the bytes into the selected destination
half, preserving the other half (`HIGH_D`). The latter selects source halves
with `HIGH_A/B`, applies `ABS_A/B` before `NEG_A/B`, quiets signaling NaNs, and
packs A low/B high. Other FP16 bits, including subnormals and NaN payloads,
are preserved. All modifiers stay on eight-lane v3 and sixteen-lane v4 paths;
host FP state is unchanged. Only loose semantics are exposed.

Tests retain digests for 4,325,376 GPU results spanning every I16/FP16 encoding
and modifier combination, plus masks, aliases, unaligned storage and host
FP-state checks. The GPU corpus establishes both destination-half preservation
and signaling-NaN quieting. Pinned-core Ryzen 9 7950X3D timings measured
1.22–1.27x for v3 and 2.77–4.25x for v4 versus scalar, including modifiers
(seven samples, each at least 10 ms).

AND, OR, XOR, and NOT support 16-bit and 32-bit values; XNOR supports 32-bit values. The 16-bit forms
select source and destination halves with `HIGH_A`, `HIGH_B` (binary forms),
and `HIGH_D`, preserving the other destination half. The 32-bit forms have
no instruction modifiers. SIMD processes sixteen lanes on v4, including every
half selector. AVX2 candidates were slower than baseline, so v3 CPUs use the
portable path. Tests cover every half encoding,
per-bit truth tables, every selector combination, unaligned storage, EXEC
masks, and source/destination aliases. Invalid flags leave all registers
unchanged, and every path preserves host FP state.

Bit-field instructions include unsigned/signed extraction, insertion, mask
creation, and reversal. `BFE` takes its offset from `B` and width from `C`;
`BFM` takes width from `A` and offset from `B`. Offsets and widths use only
five bits, so width 32 produces zero. Signed extraction first sign-extends the
source beyond bit 31, then sign-extends the selected field. `BFI` selects bits
from `B` wherever `A` is set and from `C` elsewhere. `BFREV` reverses all 32 bits.
`ALIGNBIT` extracts the low word of the concatenation `A:B` shifted right by
`C & 31` bits; `ALIGNBYTE` shifts it by `(C & 3) * 8` bits. A zero shift
returns `B`, and high count bits are ignored.
`PERM` takes four selector bytes from `C`: 0..7 select bytes of `A:B`, 8..11
replicate the sign bits of its four 16-bit halves, 12 selects zero, and 13..255
select `0xff`. Its eight-lane v3 and sixteen-lane v4 implementations use byte
shuffles and retain full EXEC masking. Tests check all selector bytes against
65,536 GPU-captured outputs, every source bit and its complement, plus masks,
whole-register aliases, unaligned storage and preservation of host FP state.
Pinned-core Ryzen 9 7950X3D timings for permutation are 68.8 ns scalar, 14.0 ns
v3 (4.91x), and 4.5 ns v4 (15.40x), using seven samples of at least 10 ms.
These rules follow rocjitsu's integer helpers. No instruction modifiers apply;
all host FP state is preserved. Eight-lane v3 and sixteen-lane v4 paths use
variable shifts, Boolean operations, and byte lookup/shuffle reversal. `BFI`
uses scalar on v3 because its AVX2 candidate showed little benefit; v4 uses
native ternary logic. Independent per-bit tests cover every offset, width and
source bit, signed boundary crossings, ignored high count bits, truth tables,
EXEC masks, and whole-register aliases. Alignment tests exercise every bit of
the 64-bit concatenation at every shift. Pinned-core Ryzen 9 7950X3D measurements
show 1.31x for alignment on v3, 4.75–4.89x on v4, and 3.61x for XNOR on v4
(seven samples, each at least 10 ms).

Combined integer instructions include shift/add, add/shift, shift/OR, AND/OR,
three-input OR/XOR, and XOR/add. Shifts mask their count to five bits, and
arithmetic wraps to 32 bits. `LERP_U8` independently averages four byte pairs;
the low bit of each `C` byte selects whether an odd sum rounds up or down.
Other `C` bits are ignored. These forms have no instruction modifiers and
preserve host FP state. All have scalar and sixteen-lane v4 paths; v4 uses
native three-input Boolean operations. Eight-lane v3 paths accelerate the three
shift combinations and byte interpolation. The simple Boolean/XOR-add forms
remain on baseline for v3 CPUs because measurements showed no AVX2 gain.
Tests cover truth tables
at every bit position, wrapped counts and arithmetic, every byte pair and all
sixteen rounding-control combinations, EXEC masks, and whole-register aliases.

All eight SAD-family instructions have scalar and eight-lane v3 paths,
including `CLAMP` saturation. Ordinary SAD sums unsigned byte, halfword or
full-word differences and adds `C`; `SAD_HI_U8` shifts the byte sum left by 16
before that addition. `MSAD` ignores positions whose `B` byte is zero.
The quad forms compare four overlapping four-byte windows from the two-VGPR
`A` against `B`, with four independent accumulators. Packed `U16` results use
two `C`/`D` VGPRs; `U32` uses four. Saturation and wrapping apply independently
to each accumulator. Tests cover every byte pair and halfword encoding,
overflow boundaries, all masked-byte patterns, sliding windows, the unused
high source byte, and destination/source aliases under varied EXEC masks.
The arithmetic and saturation rules follow rocjitsu's existing SAD helpers.

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

FP64 ADD/MUL/FMA, rounding, FRACT, min/max, FREXP mantissa, and SQRT/RCP/RSQ
now apply hardware-checked OMOD zero/denormal rules on scalar and AVX2 paths.
An unscaled subnormal or either zero sign becomes positive zero; halving a
normal below twice minimum normal produces signed zero. The GFX1201 tests
check all output modifiers with NEG and CLAMP and cross-half aliases. The
thirteen non-transcendental operations use raw-result comparisons (NaN payloads
are ignored); SQRT/RCP/RSQ use numerical comparisons because the hardware
instructions themselves provide approximations substantially below FP64
precision. FP64 LDEXP and FP64-output conversions also apply these output
rules on scalar, AVX2, and AVX-512 paths, with additional hardware captures
covering all applicable modifiers. With ABS/NEG/OMOD/CLAMP, these sixteen FP64
AVX2 arithmetic paths measure 1.9–3.4× scalar speed on the Ryzen 9 7950X3D.

FP32/FP64 `LDEXP` scales A by an integer power of two held in one B VGPR.
A and D use one VGPR for FP32 or low/high pairs for FP64. Both widths have
scalar, v3 and v4 implementations, supporting all 32 A ABS/NEG, OMOD and CLAMP
combinations without falling back to scalar. B is an integer and has no
floating-point modifiers. The v3 path adjusts exponents and rounds underflowing
results once; v4 uses native vector scaling. Tests cover every exponent field,
subnormal rounding ties, extreme signed exponents, special values, masks and
aliases, including FP64 destination halves that overwrite A or B. Both widths
check tininess before rounding when OMOD is active, including results that
would round up to minimum normal. The FP64 scalar check uses the source
exponent and does not require an extended-precision host type. GPU captures
also verify that active OMOD converts negative zero to positive zero when
widening FP32 to FP64; integer widening remains unchanged numerically.
With modifiers enabled, FP64 LDEXP measures 63.5 ns on AVX2 and 12.3 ns on
AVX-512 (2.2× and 11.3× scalar speed on the Ryzen 9 7950X3D). FP32-to-FP64
conversion measures 26.9 ns and 8.2 ns (2.0× and 6.7×); its unmodified AVX2
path is roughly tied with scalar.

FP32 `LDEXP` and FP64-to-FP32 conversion apply the FP32 OMOD zero/denormal
rules on all scalar and SIMD paths. Active OMOD also flushes an exact tiny
result before FP32 rounding, even when rounding would otherwise produce minimum
normal. A 12,288-result GFX1201 capture checks all modifiers, underflow and
normal-boundary cases, extreme LDEXP exponents, and FP16-to-FP32 conversion as
a cross-check. Tests repeat the capture checks at every CPU level and with
source/destination aliases; the existing mask and rounding tests use the same
architectural rule. On the Ryzen 9 7950X3D, modified FP32 LDEXP measures
25.9 ns on AVX2 and 5.5 ns on AVX-512 (4.8× and 22.6× scalar speed).
Modified FP64-to-FP32 conversion measures 23.6 ns and 8.7 ns (2.3× and 6.1×);
its unmodified AVX2 path is roughly tied with scalar.

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

Mixed FMA supports FP32 output (`FMA_MIX_F32`) or one selected FP16 output
half (`FMA_MIXLO_F16` / `FMA_MIXHI_F16`), preserving the other half.
Each source independently selects FP32 or FP16 using `GOC_MIX_F16_A/B/C`;
FP16 sources use `GOC_ALU_HIGH_A/B/C` to select their half, while FP32
sources ignore those selectors. ABS/NEG and CLAMP complete the 8,192 modifier
combinations, all supported by scalar and SIMD paths. OMOD and `HIGH_D` do
not apply. `GOC_FP16_OVFL` controls finite FP16 overflow without changing
infinities or FP32 outputs.

The v3 FP32-output path processes eight lanes with native FMA. FP16 outputs
use four FP64 lanes: the FP32 product is exact in FP64, and an error-free sum
retains the addend residual before direct FP16 rounding. This borrows
rocjitsu's `fma_f32_to_f16_nearest_environment` and `mixed_fma_simd.h` model;
it avoids double rounding even when an FP32 subnormal perturbs an exact FP16
midpoint. Exceptional FP16 lanes use the scalar NaN/invalid-product policy.
An empirical-exact scalar mode for the FP16 outputs preserves host rounding,
flush controls and exception state; it widens FP32 encodings through integer
bits so host DAZ cannot discard an input. FP32 output currently offers loose
semantics only. Tests use an independent 640-bit integer oracle and cover all
modifiers, every FP16 source encoding, all finite half midpoints with tiny
positive/negative perturbations, captured NaN and fused-cancellation witnesses,
EXEC masks, whole-register aliases, and hostile host FP settings.

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
OMOD flushes an unscaled subnormal or either zero sign to positive zero;
halving a normal value below twice minimum normal produces signed zero.
SQRT, RCP, RSQ, EXP, and LOG additionally flush subnormal inputs and results
with the original sign, independently of guest denormal mode. The FP16 unary
paths retain their separate rules. GPU captures cover all eleven FP32 unary
operations with all OMOD values, NEG, and CLAMP; tests require exact special
values and signed zeros, allowing a few ULPs for finite transcendental results.
The scalar ties-to-even helper is adapted from rocjitsu's
[`rndne_scalar`](https://github.com/ROCm/rocm-systems/blob/develop/emulation/rocjitsu/lib/util/include/util/simd.h).
Unary tests cross all 32 modifier combinations with 85 masks, both separate and
aliased output, and all available CPU levels, including signed zeros, subnormals,
infinities, NaNs, half-integer ties and large integral values. With ABS/OMOD/CLAMP
enabled, the nine AVX2 unary paths measure 3.2–6.5× scalar speed on the
Ryzen 9 7950X3D; EXP and LOG remain scalar.

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

Scalar integer arithmetic includes 32-bit add/subtract with carry, borrow or
signed-overflow SCC; ABS/ABSDIFF; signed/unsigned MIN/MAX; low/high multiply;
64-bit modular add/subtract/multiply; and signed 16-bit immediate add/multiply.
These scalar-register instructions execute once per wave regardless of EXEC,
including empty EXEC. They use a scalar path on every CPU level because they
produce only one result per call. Exact models follow rocjitsu with a
hardware-verified correction: `s_absdiff_i32` wraps subtraction before ABS.
Tests cover 983,040 GFX1201 result/SCC triples, every signed immediate, aliasing,
and host FP-state preservation. Benchmarks include all 20 instructions.

Scalar bitwise arithmetic also includes AND/OR/XOR and complemented forms,
NOT, bit reversal, logical/arithmetic shifts in 32 and 64 bits, and four fused
shift-add forms. Models borrowed from rocjitsu match 737,280 GFX1201 captured
result/SCC triples. Shift counts wrap to the operand width; bit reversal leaves
SCC unchanged. Tests cover overlapping scalar outputs, empty EXEC, every shift
count, and host FP-state preservation. These single-result operations use scalar
paths on all CPU levels and are included in the benchmark.

Scalar bitfield support covers signed/unsigned extraction, mask construction,
bit clear/set, population counts, and leading/trailing/sign-bit counts in 32 and
64 bits. Models borrowed from rocjitsu match 1,966,080 GFX1201 result/SCC triples.
Extraction clips widths at the operand boundary; count sentinels, SCC effects,
empty EXEC, and overlapping outputs are tested. All 20 instructions have scalar
benchmarks; each produces one scalar result per wave.

Scalar FP16/FP32 ADD, SUB, MUL, MIN_NUM, MAX_NUM, MINIMUM and MAXIMUM support
independent guest input/output denormal flushing and FP16 finite-overflow
saturation. They use loose semantics with host nearest-even rounding and enabled
denormals. GFX1201 tests cover 2,752,512 result/SCC pairs across all eight FP
states, including a dedicated multiplication-underflow grid. Scalar min/max
ignores output flushing. Multiplication tininess detection precedes destination
subnormal rounding; tests cover values that otherwise round up to normal.
All 14 instructions have scalar benchmarks, including nondefault FP states.

Scalar fused multiply-add includes FP16/FP32 FMAC and FP32 FMAAK/FMAMK literal
forms. These preserve fusion and SCC, support guest input/output flushing and
FP16 overflow saturation, and snapshot accumulator inputs before writing.
Tests cover 2,555,904 GFX1201 result/SCC pairs, twelve literal bit patterns,
all eight FP states, cancellation, tininess boundaries, EXEC and aliases.
Loose semantics require host nearest-even rounding and enabled denormals.
All four instructions have benchmarks with default and nondefault FP settings.

Scalar FP16/FP32 CEIL, FLOOR, TRUNC and RNDNE use integer rounding derived from
rocjitsu's nearest-even model. Both semantics are supported, with all host FP
state preserved independently of host rounding. Tests match 12,582,912 raw
GFX1201 result/SCC pairs, including every FP16 pattern under all eight FP states.
Signed zeros and NaN payloads/signs are preserved while signaling NaNs quiet.
Input flushing is supported; output flushing and FP16 overflow saturation have
no effect. Benchmarks cover all eight instructions and input-flush variants.

All eight scalar conversions are implemented: signed/unsigned integer–FP32,
FP32–FP16, high-half FP16 widening, and packed FP32-to-FP16 RTZ. Models borrowed
from rocjitsu match 12,582,912 raw GFX1201 result/SCC pairs, including every FP16
pattern in both source halves and all eight guest FP states. Tests cover
saturation, NaNs, ties, underflow, packed ordering, aliases, and EXEC. The packed
RTZ helper is shared with vector conversions. Benchmarks include all eight
instructions with default and nondefault FP settings; semantics are loose.

All 46 scalar comparison instructions are implemented: signed/unsigned integer
predicates, 32/64-bit bit tests, and all FP16/FP32 predicates. They return one SCC
bit, ignore EXEC, and preserve all host FP state. Both semantics use raw-integer
models borrowed from rocjitsu and share predicate helpers with vector comparisons.
Tests match 4,521,984 GFX1201 SCC results across guest FP states and cover NaNs,
signed zeros, bit-index masking, source/output overlap, and host rounding modes.
Benchmarks cover each instruction and floating input-flush variants.

Scalar packing and selection covers all four halfword PACK forms, BITREPLICATE,
32/64-bit CSELECT, QUADMASK and WQM. Both semantics preserve all host FP state.
Tests match 270,336 GFX1201 result/SCC triples and cover every input bit, all
65,536 quad-presence patterns, source/output overlap, SCC and EXEC behavior.
All eleven operations have scalar benchmarks.

Scalar `s_sext_i32_i8` and `s_sext_i32_i16` sign-extend low register bits while
preserving SCC and all host FP state. Tests exhaust every low 16-bit pattern,
match 786,432 GFX1201 result/SCC triples, and cover aliases, EXEC, and errors.
Both semantics and scalar benchmarks are available.

The four wave32 lane permutations (`v_permlane16_b32`, `v_permlanex16_b32`,
and their `_var` forms) support scalar, AVX2, and AVX-512 paths, including FI
and BOUND_CTRL. Selectors may come from two scalar words or one VGPR. Inactive
source lanes either supply their value, produce zero, or preserve the destination,
as selected by the flags; inactive destinations remain unchanged. Every path
supports whole-register aliasing and both semantics without modifying host FP
state. Tests compare a 65,536-result GFX1201 hardware corpus and exercise all
single-lane EXEC patterns, random masks and selectors, and in-place permutations.
On the Ryzen 9 7950X3D, pinned CPU-8 measurements (seven samples, at least
10 ms each) show AVX2 at 11.8–13.2 ns/wave (2.3–3.3× scalar) and AVX-512
at 2.4–5.2 ns/wave (5.9–14.3×), including FI/BOUND_CTRL configurations.

FP32 `v_fma_f32` and `v_fmac_f32` now accept DPP8 lane permutations. Set
`GOC_DPP8` and pack eight 3-bit selectors at `GOC_DPP8_SELECT_SHIFT`; add
`GOC_DPP_FI` to fetch inactive source lanes instead of substituting positive
zero. Permutation precedes ABS/NEG and preserves the existing arithmetic,
clamp, EXEC, and whole-register aliasing behavior. Both AVX2 and AVX-512 paths
remain active with every supported modifier combination. The DX9-zero FMA
variant has no DPP encoding and rejects these flags.

DPP8 tests include a 16,384-result GFX1201 FMA/FMAC capture, all arithmetic
modifier combinations, random selectors, EXEC patterns, and aliases. Pinned
CPU-8 measurements on the Ryzen 9 7950X3D (seven samples of at least 10 ms)
measure 16.5–17.4 ns/wave for AVX2 (5.3–6.9× scalar) and 7.3 ns/wave for
AVX-512 (12.7–15.8×), including FI/NEG/output scaling.

The same FMA/FMAC entry points support DPP16 via `GOC_DPP16`, with all 335
valid RDNA4 controls: quad permutations, row shifts/rotations, mirrors,
broadcasts, and XOR permutations. Row/bank fields filter destination writes;
`GOC_DPP_BOUND_CTRL` selects zero-input arithmetic instead of preserving a
write with an unreadable source. FI enables inactive in-range sources and
never overrides an out-of-range boundary. Set `GOC_DPP_ROW_MASK` and
`GOC_DPP_BANK_MASK` for full row/bank coverage; zero fields disable writes.
DPP8 and DPP16 are mutually exclusive.

DPP16 tests cover every control, all row/bank fields, arithmetic modifiers,
EXEC masks and aliases, plus a 65,536-result GFX1201 hardware corpus. The
`row_shl:15` benchmark with boundary zeroing measures 22.4–23.6 ns/wave on
AVX2 (3.9–5.4× scalar) and 11.8–12.0 ns/wave on AVX-512 (7.6–10.2×),
including FI/NEG/output scaling.

DPP8/DPP16 also cover all nine binary FP32 arithmetic operations and all nine
three-input FP32 min/max/median operations. They retain their AVX2 arithmetic
paths, using AVX-512 source permutation on x86-64-v4 hosts. A shared adapter
validates each instruction's own flags before reading sources and snapshots
the permuted input before any destination write. Tests cover every arithmetic
modifier combination, special values, EXEC patterns, and all operand aliases.
Two GFX1201 corpora validate 193,536 raw results, including output-scaling
boundaries. Representative ADD/MUL/min3/median benchmarks measure 1.8–4.8×
scalar speed on x86-64-v3 and 1.9–4.9× on x86-64-v4 with DPP enabled.

DPP8/DPP16 also cover all eleven FP32 unary operations: TRUNC, CEIL, RNDNE,
FLOOR, SQRT, RCP, RSQ, EXP, LOG, FRACT, and FREXP mantissa. Permutation occurs
before ABS/NEG and arithmetic, with FI/BOUND/row/bank controls and inactive
write preservation. Nine operations retain AVX2 arithmetic; x86-64-v4 adds
AVX-512 permutation. EXP and LOG retain scalar arithmetic after SIMD permutation.
A GFX1201 capture checks 39,424 results across seven descriptors, two modifier
settings, and eight EXEC masks. Independent tests cross all 32 unary modifier
combinations with all mask patterns and separate/aliased output, including
subnormals, signed zeros, infinities, and NaNs. Invalid flags are checked before
operand access. Benchmarks cover representative rounding, SQRT, RCP, and FRACT
with full EXEC and both DPP kinds. On the Ryzen 9 7950X3D, these paths measure
18.6–25.2 ns on AVX2 (1.9–4.9× scalar speed) and 17.9–24.1 ns with AVX-512
permutation plus AVX2 arithmetic (2.0–5.1×).

DPP8/DPP16 also support wave32 CLZ, CTZ, CLS, BCNT, MBCNT_LO, MBCNT_HI,
and 32-bit AND, OR, XOR, XNOR, NOT. MBCNT's bit range remains determined by
the destination lane after source permutation. Integer host FP state is
preserved, including rounding mode and existing exception flags. A GFX1201
capture checks 19,712 results; independent tests combine all mask patterns,
source/destination aliases, exceptional bit patterns, and random words. SIMD
permutation composes with each instruction's existing arithmetic path, including
AVX-512 Boolean operations. Wave64 count DPP forms remain pending. Benchmarks cover representative leading/population/masked counts and
Boolean operations with full EXEC. On the Ryzen 9 7950X3D, these cases measure
16.8–23.5 ns on x86-64-v3 (1.9–2.4× scalar speed) and 9.1–13.4 ns on
x86-64-v4 (3.3–4.9×), using seven pinned samples of at least 10 ms each.

The three 32-bit shifts (`v_lshlrev_b32`, `v_lshrrev_b32`, and `v_ashrrev_i32`)
also support DPP8/DPP16. Source permutation applies to the shift count; the
shifted value remains local to the destination lane. A GFX1201 capture checks
5,376 results, and the integer DPP mask/alias/random-word and host FP-state tests
cover all three operations. AVX2 arithmetic follows scalar, AVX2, or AVX-512
source permutation according to the selected CPU level. The benchmark includes
arithmetic right shift with both DPP kinds and full EXEC. On the Ryzen 9 7950X3D,
these cases measure 17.7–21.2 ns on x86-64-v3 (2.0–2.3× scalar speed) and
17.0–19.9 ns with AVX-512 permutation plus AVX2 arithmetic (2.2–2.4×).

DPP8/DPP16 also cover all fourteen 32-bit signed and unsigned integer min/max
operations: MIN, MAX, MIN3, MAX3, MINMAX, MAXMIN, and MED3. A GFX1201 capture
checks 25,088 results across seven descriptors and eight EXEC masks. Independent
tests cover all source-equality patterns, destination aliases, signed boundaries,
random words, all EXEC patterns, and host FP-state preservation. Both AVX2 and
AVX-512 arithmetic paths remain available with DPP; benchmarks include signed
and unsigned binary MIN and ternary MED3 with full EXEC. On the Ryzen 9 7950X3D,
these cases measure 17.7–22.2 ns on AVX2 (1.9–2.4× scalar speed) and 8.0–11.2 ns
on AVX-512 (3.7–5.2×), using seven pinned samples of at least 10 ms each.

DPP8/DPP16 also cover the eight combined integer operations: LSHL_ADD,
ADD_LSHL, LSHL_OR, AND_OR, OR3, XOR3, XAD (XOR/add), and byte LERP. A GFX1201
capture checks 14,336 results; independent tests combine all source-equality
patterns and destination aliases with every EXEC pattern, integer boundaries,
and random words. Host FP state is preserved. Existing AVX2 paths for combined
shifts and LERP, and AVX-512 paths for all eight operations, remain available
with DPP. The benchmark covers LSHL_ADD, AND_OR, and LERP with full EXEC.
The baseline combined-shift loops explicitly prevent Clang from replacing
integer shifts with FP conversions that raise `FE_INVALID` at shift count 31;
regression tests cover this boundary with and without DPP, and every count
without DPP. On the Ryzen 9 7950X3D, the DPP benchmark cases measure
17.8–21.9 ns on x86-64-v3 (1.9–2.8× scalar speed) and 8.2–11.5 ns on
x86-64-v4 (3.7–6.4×), using seven pinned samples of at least 10 ms each.

DPP8/DPP16 also cover signed/unsigned BFE, BFI, BFM, BFREV, ALIGNBIT, ALIGNBYTE,
and PERM. Source A is permuted before bitfield extraction, selection, or byte
permutation; the other operands remain local to the destination lane. A GFX1201
capture checks 14,336 results, supplemented by independent mask/alias/boundary
and random-word tests. AVX2 and AVX-512 paths compose with DPP (BFI retains its
baseline arithmetic at the v3 CPU level). Baseline BFE explicitly prevents
Clang's FP-based vector shift lowering from raising host `FE_INVALID` for width
31; regression tests check every width/offset without DPP and the boundary with
DPP. Benchmarks cover signed BFE, BFM, and PERM with full EXEC. On the Ryzen 9
7950X3D, these cases measure 17.9–23.6 ns on x86-64-v3 (2.3–4.3× scalar speed)
and 9.1–12.6 ns on x86-64-v4 (4.3–8.8×), using seven pinned samples of at least
10 ms each.

DPP8/DPP16 also support 32-bit unsigned ADD_NC/SUB_NC/SUBREV_NC, signed
ADD_NC/SUB_NC, and ADD3. The five binary operations support CLAMP together with
DPP; ADD3 retains wrapping semantics. SUBREV_NC_U32 permutes B and subtracts
lane-local A, as observed on GFX1201 in both VOP2 and VOP3 encodings, while the
other operations permute A.
A GFX1201 capture checks 19,712 results,
including saturation boundaries. Independent tests combine both modifier modes
with all EXEC patterns and source/destination aliases, and check host FP-state
preservation. AVX2 saturation and all existing AVX-512 arithmetic paths remain
available; wrapping arithmetic at the v3 CPU level retains the baseline arithmetic
with AVX2 permutation. Benchmarks cover signed/unsigned addition and unsigned
reverse subtraction with and
without CLAMP and both DPP kinds, using full EXEC. On the Ryzen 9 7950X3D,
these cases measure 17.5–22.3 ns on x86-64-v3 (1.8–3.3× scalar speed) and
8.9–12.3 ns on x86-64-v4 (3.5–5.5×), using seven pinned samples of at least 10 ms
each.

DPP8/DPP16 also support the four signed/unsigned 24-bit multiply operations,
including high-half results and CLAMP for the low-result forms. A GFX1201
capture checks 10,752 results. Independent tests combine wrapping/saturating
modes with all EXEC patterns, source/destination aliases, boundary and random
words, and host FP-state preservation. AVX2 and AVX-512 arithmetic remain
available for every supported modifier combination. Benchmarks cover all four
operations and both DPP kinds with full EXEC. The full-width MUL_LO_U32,
MUL_HI_U32, and MUL_HI_I32 instructions have no DPP encoding on gfx120x and
continue to reject these flags. On the Ryzen 9 7950X3D, the supported DPP cases
measure 18.5–24.0 ns on AVX2 (1.8–3.6× scalar speed) and 9.6–14.3 ns on AVX-512
(3.5–5.2×), using seven pinned samples of at least 10 ms each.

DPP8/DPP16 also cover all four integer MAD operations: signed/unsigned 16-bit
and 24-bit multiplication with 32-bit accumulation. DPP composes with CLAMP and
all HIGH_A/HIGH_B combinations on the 16-bit forms. A GFX1201 capture checks
35,840 results across all 20 operation/modifier combinations; independent tests
combine these with all EXEC patterns, source-equality patterns, destination
aliases, boundary/random words, and host FP-state preservation. Source permutation
precedes half selection. The AVX2 arithmetic paths remain available for every
combination, with AVX-512 permutation when selected. Benchmarks cover all four
operations with default and combined modifiers, both DPP kinds, and full EXEC.
On the Ryzen 9 7950X3D, these cases measure 18.2–26.2 ns on AVX2 (1.9–3.6×
scalar speed) and 17.7–24.6 ns with AVX-512 permutation plus AVX2 arithmetic
(2.0–3.7×), using seven pinned samples of at least 10 ms each.

DPP8/DPP16 also support the 16-bit AND, OR, XOR, and NOT operations with all
source and destination half selectors. A GFX1201 capture checks 50,176 results
across all 28 operation/selector combinations. Tests verify the untouched
destination half, all EXEC patterns, source/destination aliases, random words,
and host FP-state preservation. The v3 path combines AVX2 permutation with
baseline Boolean arithmetic; v4 uses AVX-512 arithmetic and permutation.
Benchmarks cover AND and NOT with low/high selections and full EXEC. On the
Ryzen 9 7950X3D, these cases measure 12.8–24.2 ns on x86-64-v3 (1.7–2.6× scalar
speed) and 8.9–12.2 ns on x86-64-v4 (3.0–4.6×), using seven pinned samples of at
least 10 ms each.

DPP8/DPP16 also support all twelve non-packed 16-bit integer arithmetic forms:
signed/unsigned ADD_NC, SUB_NC, MIN, MAX, low multiply, and the three reverse
shifts. All source/destination half selectors and applicable CLAMP settings
compose with DPP. A GFX1201 capture checks 229,376 results across all 128
operation/modifier combinations. Independent tests cover all EXEC patterns,
source/destination aliases, special and random words, untouched destination
halves, and host FP-state preservation. The AVX2 arithmetic paths remain active
with every supported modifier, with AVX-512 permutation at the v4 CPU level.
Benchmarks cover addition, minimum, multiplication, and left shift with default
and combined selectors/CLAMP, both DPP kinds, and full EXEC. On the Ryzen 9
7950X3D, these cases measure 18.9–23.8 ns on AVX2 (2.0–3.7× scalar speed) and
18.4–22.1 ns with AVX-512 permutation plus AVX2 arithmetic (2.2–3.8×), using
seven pinned samples of at least 10 ms each.

The hardware probes also corrected OMOD behavior in the 18 binary/ternary operations,
with and without DPP: an unscaled FP32 subnormal or either zero sign becomes
positive zero; halving a normal magnitude below twice the minimum normal
produces signed zero. These rules apply independently of guest denormal mode.
The same correction now covers `v_fma_f32` and `v_fmac_f32`, including their
DPP forms, on scalar, AVX2, and AVX-512 paths. A separate GFX1201 capture checks
all OMOD settings with NEG/CLAMP, signed zeros, subnormal boundaries, overflow,
and NaNs; tests combine those results with EXEC masks and FMA destination aliases.
On the Ryzen 9 7950X3D, FMA with NEG/ABS/OMOD measures 16.5 ns on AVX2
and 7.0 ns on AVX-512 (5.3× and 12.6× scalar speed).

Other FP families still need this OMOD boundary audit, and other applicable
instructions still need DPP support. FP32 unary operations now also have the
OMOD correction and mandatory transcendental denormal flushing described above.
DX9 FMA now also has mandatory denormal flushing and positive-zero-product
addition, validated independently across all hardware denormal modes.

All public instruction entry points take a 64-bit `instruction_flags` value.
Existing modifier bits keep their meanings. The extra width accommodates the
24-bit DPP8 lane selector together with arithmetic modifiers; unsupported upper
bits return `GOC_ERROR_INVALID_FLAGS` before operand access. An API-wide
test covers this rejection contract for all 669 wave32/wave64 entry points.
Function-pointer adapters must use `uint64_t` for this parameter too.

RDNA4 coverage still needs remaining scalar-register arithmetic, dual-operation
forms, data-permutation modifiers, and a complete wave64/FP-mode audit. Instruction
name coverage alone does not establish complete architectural support. Other GPU
architectures and further performance tuning also remain future work.
