# GoC: GPU arithmetic on a CPU

GoC implements whole-wave GPU arithmetic through a synchronous C API. The public
header is `include/goc.h`, usable from C99 and C++17. It directly includes
`goc_common.h` (flags, status codes, CPU detection) and `goc_rdna4.h` (RDNA4
instructions and modifiers). Each component header is independently usable. This is an initial RDNA4
implementation; `PLAN.md` describes the broader intended coverage.

## Build and test

```sh
cmake -S . -B ../goc-build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ../goc-build --parallel "$(nproc)"
ctest --test-dir ../goc-build --parallel "$(nproc)" --output-on-failure
```

CMake uses a system GTest when available and otherwise fetches GTest 1.17.0.
Use `-DGOC_ENABLE_X86=OFF` for the portable implementation. Optional CPU paths
are compiled in separate translation units after compiler-flag checks. GCC and
Clang are supported; other compilers use the portable paths.

## Implemented instructions

The entry points below use RDNA4 wave32. Names have the `goc_rdna4_` prefix.
The two WMMA forms additionally have scalar wave64 variants named
`goc_rdna4w64_...`, supporting loose and empirical exact modes.

| Mnemonic | Loose semantics | Empirical exact semantics |
| --- | --- | --- |
| `v_fma_f32` | Scalar, AVX2/FMA, AVX-512 | Not implemented |
| `v_log_f32` | Scalar `log2` | Not implemented |
| `v_dot2_f32_f16` | Scalar | Integer arithmetic model |
| `v_dot2_f32_bf16` | Scalar | Integer arithmetic model |
| `v_wmma_f32_16x16x16_f16` | Scalar | Integer arithmetic model |
| `v_wmma_f32_16x16x16_bf16` | Scalar, AVX-512 BF16 | Integer arithmetic model |

WMMA supports all six NEG/NEG_HI modifier bits, including C absolute value.
Other instructions currently accept only zero instruction flags. The BF16
fast path handles unmodified loose calls; exceptional values and extreme
product exponents conservatively use scalar arithmetic. Exact modes never use
the approximate SIMD path.

## Calling convention

Call `goc_init_cpu_flags()` and OR the result with semantics flags. CPU levels
are enumerations, not independent bits. A caller may select a lower CPU level
but must never claim unavailable capabilities. Dispatch also respects which
implementations were compiled.

Each VGPR pointer addresses 32 contiguous `uint32_t` lane words (64 for
`rdna4w64`). A multi-VGPR
operand is an array of these pointers; the backing arrays need not be adjacent
or SIMD-aligned. A/B use four VGPRs and C/D use eight for the implemented WMMA
wave32 forms; wave64 uses two A/B VGPRs and four C/D VGPRs. Input and output operands may share whole VGPRs. Distinct backing
addresses must not overlap, and every pointer must refer to sufficient storage.

GoC applies `exec_mask` to destination writes, including WMMA, as specified by
its API contract. Inactive destination lanes remain unchanged; source lanes are
not masked. Errors preserve all destination registers. No pointer-validation
or allocation ownership service is provided.

Zero semantics bits select loose numerical behavior. Add `GOC_SEMANTICS_EXACT`
for the empirical model; also add `GOC_SEMANTICS_STRICT` to require support.
Unsupported exact requests otherwise fall back to loose semantics. Unassigned
instruction/general flag bits are rejected, except reserved semantics values
which follow the same fallback policy. No GPU FP-environment flag bits have
been assigned yet. The caller's host FP environment must use nearest-even
rounding and support denormals.

## Validation and provenance

Tests exercise the C ABI, CPU/OS feature gating, forced usable CPU levels,
unaligned and noncontiguous backing storage, masks, operand overlap, strict
errors, sign modifiers, and deterministic dense matrix golden outputs.

The empirical RDNA4 DOT/WMMA model in `src/rdna4_dot.h` is adapted from
rocjitsu's `isa/arch/amdgpu/shared/gfx12_dot.h`. The hardware fixtures in
`tests/dot_fixtures.h` and `tests/wmma_fixtures.h` come from rocjitsu's
`tests/fixtures/float_dot/gfx1201_cases.h`: Radeon AI Pro R9700 (`gfx1201`),
TheRock `10.2.0a20260916`. They contain 121 DOT and 24 WMMA captures. No new
GPU measurements or reverse engineering were performed for this implementation.
The empirical qualification is limited to that existing model and evidence.

CPU detection follows the CPUID/XCR0 gating approach in
`hrx-system/runtime/src/iree/base/internal/cpu_x86_64.c`, with GoC's coarse
feature bundles. Formatting and the MIT license are borrowed from rocjitsu.

Still pending: other GPU architectures, additional instructions/formats,
GPU FP-mode flags, and wider performance tuning.
