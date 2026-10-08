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

Both `GOC_STATIC` and `GOC_SHARED` default to `ON`, producing `libgoc.a` and
`libgoc.so` on Linux. Disable either with `-DGOC_STATIC=OFF` or
`-DGOC_SHARED=OFF`; at least one must remain enabled. CMake consumers link
`goc_static` for private embedding or `goc_shared` for dynamic linking.
Static consumers inherit
`GOC_STATIC_DEFINE`, so GoC's API stays hidden when embedded in their shared
library; shared consumers see exported API declarations. Both variants are
position-independent and keep implementation symbols hidden.

The public API and C linkage are tested against every enabled variant. Each
`*_test.cpp` has its own executable, such as `tests/goc_wmma_test_static` and
`tests/goc_wmma_test_shared`; the private decoder uses `tests/goc_cpu_decode_test`.
Public headers are compiled independently once per language (C and C++), without
linking. Private CPU decoding is tested separately. On Linux, export
tests also check the shared API and static embedding. `tests/cpuinfo` links to
`goc_static` and is only built and tested when `GOC_STATIC` is enabled.

CMake uses a system GTest when available and otherwise fetches GTest 1.17.0.
CMake automatically enables x86-64 implementations for x86-64 targets using GCC
or Clang. Optional CPU paths are compiled in separate translation units after
compiler-flag checks. Other compilers and architectures use the portable paths.

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
| `v_wmma_f32_16x16x16_f16` | Scalar | Integer arithmetic model |
| `v_wmma_f32_16x16x16_bf16` | Scalar, AVX-512 BF16 | Integer arithmetic model |
| `v_wmma_f16_16x16x16_f16` | Integer arithmetic model | Integer arithmetic model |
| `v_wmma_bf16_16x16x16_bf16` | Integer arithmetic model | Integer arithmetic model |

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
`tests/dot_fixtures.h` and `tests/wmma_fixtures.h` come from rocjitsu's
`tests/fixtures/float_dot/gfx1201_cases.h`: Radeon AI Pro R9700 (`gfx1201`),
TheRock `10.2.0a20260916`. They contain 121 DOT and 24 WMMA captures. No new
GPU measurements or reverse engineering were performed for this implementation.
The empirical qualification is limited to that existing model and evidence.
Packed-output support additionally borrows `shared/dot_packed16.h` and the
RDNA4 subset of `tests/fixtures/float_dot/packed_wmma_cases.h`: 28 captured
16x16 outputs across both formats and wave sizes, including subnormals,
cancellation, NaNs and rare accumulator-alignment boundaries. Tests run these
under all four host rounding modes with preexisting FP exception flags.

CPU detection follows the CPUID/XCR0 gating approach in
`hrx-system/runtime/src/iree/base/internal/cpu_x86_64.c`, with GoC's coarse
feature bundles. Formatting and the MIT license are borrowed from rocjitsu.

Still pending: other GPU architectures, additional instructions/formats,
further GPU FP-mode flags, and wider performance tuning.
