# SIMD, EXEC, and architectural-state audit

This pass starts from `303246c`. It reviews the current production dispatch
families, public EXEC contracts, host FP-control writes, and optional-state
reporting. It fixes two substantial loose-SIMD gaps without introducing a new
bit-exact model. This is a source/dispatch audit backed by the existing test suite
and added boundary/layout tests, not exhaustive hardware validation of every
operand, encoding, or guest mode.

## Follow-up completion checklist

The active follow-up covers the actionable gaps below. Historical findings and
timings later in this document describe the initial audit pass.

- [x] SIMD RDNA3 IU4/IU8 WMMA in both widths, all signs and final CLAMP.
- [ ] SIMD loose packed-result FP16/BF16 WMMA with intermediate narrowing.
- [ ] AVX2 coverage for Boolean/CNDMASK, bitfield insert, non-saturating
  integer add, and the remaining ternary/count/conversion gaps listed below.
- [ ] Review incomplete exact-state families against rocjitsu and the ISA;
  implement available borrowed/simple rules, retaining explicit unsupported
  results where completion would require forbidden reverse-engineering.
- [ ] Assess scalar routing/copy opportunities and justify any retained scalar
  paths by their operation, rather than merely their current implementation.
- [ ] Final modifier/mask/global-state regression validation and updated results.

## Findings and changes

| Requirement | Finding and action |
| --- | --- |
| Supported modifiers retain SIMD | No generic nondefault-modifier fallback found. NEG/ABS, OMOD/CLAMP, half selection, integer signedness, guest denormal policy, and DPP reach the corresponding vector kernels where implemented. Instruction-specific unsupported combinations still return errors. |
| Loose arithmetic normally has SIMD | Added AVX2 FP32 EXP/LOG and FP32-result FP16/BF16 WMMA for RDNA3 Wave32/Wave64 and RDNA4 Wave64. Existing RDNA4 Wave32 uses the same generalized kernel. Remaining exceptions are listed below. |
| No host FP-environment writes | No explicit setters, save/restore guards, inline assembly, or control-register writes found in production sources. Clang archive disassembly and undefined symbols also contain none of the searched FP-control setters/instructions. Ordinary arithmetic may still raise sticky exceptions. |
| Correct EXEC behavior | Vector arithmetic retains masked writes; comparisons and carry outputs mask their scalar result bits. DPP computes the participating mask before execution. Scalar/pseudo-scalar/matrix APIs omit execution masks. No signature correction was needed. |
| Exact optional state | Existing supported reporting remains available; incomplete reporting remains an explicit error only for non-null exact requests. Loose reporting does not gate SIMD. No new exception model was inferred. |

The EXP/LOG implementation shares the existing FP16 reduced-polynomial helpers.
FP32 EXP additionally handles the upper exponent boundary without prematurely
turning the power-of-two scale into infinity, and implements the instruction's
mandatory result flushing. LOG receives already-flushed FP32 inputs. Both use
8-lane arithmetic and masked stores, including after DPP staging. Architecture-
specific OMOD/CLAMP policies remain in the existing wrappers.

The WMMA implementation shares widening, source modifiers, FMA accumulation,
and result staging. Its added compile-time parameters describe the actual
Wave32/Wave64 and RDNA3/RDNA4 lane layouts. RDNA3 replicated groups are read
independently; the implementation does not assume their contents are identical.
All results are staged before destination writes, preserving whole-VGPR aliases.
There is no EXEC parameter or masked output store for these instructions.

## Dispatch review and remaining scalar cases

The top-level scan covers 91 production `.cpp` files; CPU kernels live in 101
`src/x86_64/*.cpp` files. Checking dispatch does not mean every kernel was newly
proved correct. The following groups describe the paths and their exceptions.

| Families | SIMD behavior and limitations |
| --- | --- |
| FP16/FP32 arithmetic, FMA, packed/mixed FMA, unary, trig, min/max, exponent operations | Loose v3 kernels retain supported source/output modifiers and guest policies. Selected operations also have v4 kernels. Empirical exact numerical paths may use scalar integer models. EXP/LOG gaps fixed here. |
| FP64 arithmetic/unary, conversions, interpolation, cube, MULLIT, TRIG_PREOP | Existing v3/v4 paths accept the supported modifiers. Certain conversions require v4; this is an ISA-tier coverage gap, not a modifier-triggered fallback. |
| Integer arithmetic, dot, SAD, carry/MAD64, compares, bit operations | Existing kernels accept their supported flags. Some operations have only v4 kernels: Boolean/CNDMASK and bitfield insert, for example. v3 integer add has a dedicated saturating path; non-saturating cases remain scalar below v4. Some ternary, count, and conversion variants also remain scalar below v4. |
| Floating comparisons, DIV_FIXUP, DIV_FMAS | Exact non-null reporting can select scalar classification. Their reporting predicate excludes loose semantics, so passing a pointer in loose mode does not disable SIMD. Comparison results remain required even when optional reporting is skipped. |
| WMMA/SWMMAC | FP32-result FP16/BF16 WMMA now has v3 for both architectures and widths. Existing FP8, RDNA4 integer WMMA, and sparse kernels retain their vector paths and supported modifiers. The RDNA4 Wave32 Zen4 BF16 eligibility fallback remains data-dependent, but sign modifiers do not disable it; rejection falls back to v3 when available. |
| DPP/PERMLANE | Vector routing kernels feed the ordinary arithmetic kernels. DPP can add staging and mask preparation without forcing scalar arithmetic. Wave64 adjusts the two halves' DPP row enables. |
| Scalar/pseudo-scalar instructions, hardware-register accesses | One scalar operation per wave; multi-lane SIMD is not appropriate. SAVEEXEC/WREXEC take architectural EXEC as data, not as an execution predicate. |
| Moves, relative addressing, DS routing, lane transfers, swaps | Small copies, gathers, or collision/order-sensitive routing use scalar/staging code. These are separate routing/copy optimization opportunities, not arithmetic falling off SIMD because of a flag. Dual issue forwards to the ordinary operation kernels. Constant comparisons construct the result mask directly. |

Two matrix limitations deserve explicit visibility:

- **Packed-result FP16/BF16 WMMA remains scalar**, including loose mode. Its
  existing implementation repeatedly narrows the accumulator through the
  borrowed dot model. Replacing it with one FP32 accumulation and final narrowing
  can materially change intermediate overflow/saturation. A vector implementation
  must retain those stages; this is a reasonable conservative exception, not a
  claim that SIMD is impossible.
- **RDNA3 IU4/IU8 WMMA remains scalar**, for both semantics and widths. Its
  signed-C/final-saturation rules differ from RDNA4's staged clamp. Reusing the
  RDNA4 kernel unmodified would be wrong. A dedicated vector accumulation is
  still an optimization opportunity; it is not inherently SIMD-unfriendly.

Similarly, the v4-only integer/selection operations above are not inherently
unsuitable for AVX2. This audit does not claim every operation has a dedicated
kernel at every CPU level. The default scalar CPU level and builds without
specialized kernels naturally retain scalar execution.

## EXEC and host FP controls

AVX2 vector destinations generally use `_mm256_maskstore_epi32`/64-bit equivalents;
AVX-512 uses mask stores. Selected-half writes merge the other destination half
before storing. Vector arithmetic computes full groups and masks the stores;
there is no new sparse-mask specialization. Temporary buffers and EXEC-ignoring
matrix destinations correctly use ordinary stores. Scalar reductions, comparison
masks, and carry masks are not VGPR stores and instead apply EXEC to their bits.

Existing mask/alias tests cover zero/full, single-bit, alternating, and partial
masks, Wave64 upper lanes, DPP filtering, destination guards, and selected-half
preservation. This pass adds EXP/LOG full-range tests with modifiers, masks,
aliases, both architectures and widths, and an independent numerical reference.
WMMA tests now exercise all 64 floating sign-modifier combinations for RDNA3,
distinct replicated-group data, and all CPU levels for RDNA4 Wave64 cases.

The FP-control audit searched `src/` and `include/` for fenv setters, MXCSR
setters, assembly, x87 control writes, and non-x86 control-register names. The
Clang static archive was also checked for imported setters and instructions such
as `ldmxcsr`, `fldcw`, FP-environment restores, clears, and initialization. No
matches were found. This is evidence about GoC's code, not a guarantee about
arbitrary consumer libraries or every external math-library implementation.
Loose paths retain their documented host rounding/denormal prerequisites; GoC
does not establish those prerequisites by modifying the environment.

## Optional architectural state

The signature audit reports **962 exception-output signatures, zero register-name
mismatches, zero unresolved register names**. That audit checks naming, not
numerical reporting completeness.

Existing RDNA4 reporting includes floating comparisons, scalar rounding,
RCP_IFLAG, DIV_FIXUP, DIV_FMAS, and selected FP16 FMA/packed FMA and trig modes.
FP16 FMA and trig reporting require the supported denormal-preserving policies;
RDNA3 variants and unsupported guest policies continue to reject exact reporting.
FP32 FMA, unary, conversion, and many RDNA3 exact-state paths remain incomplete.
See [the reporting contract and implemented subsets](README.md#architectural-exception-outputs).

Only non-null exact requests require faithful optional updates. Unsupported
requests preserve outputs and return `GOC_ERROR_UNSUPPORTED_GLOBAL_STATE`, even
without STRICT. Null opts out. Loose callers retain SIMD and normally leave the
optional register unchanged. Required comparison, carry, SCC, EXEC, and hardware-
register outputs are not optional exception history and remain implemented.
Existing tests exercise accumulation, inactive-lane exclusion, DPP, null pointers,
unsupported-request preservation, and loose calls with non-null state pointers.

The unresolved DX9 FMAC exception-contract review in [COVERAGE.md](COVERAGE.md)
remains documented. It does not justify new reverse-engineering or prevent use of
its loose implementation. This pass adds no bit-exact implementation or classifier.

## Validation and performance

- Clang, GCC, and a build with specialized SIMD kernels disabled: **1,185 tests
  passed in each configuration**.
- ASan/UBSan: **66 targeted unary, half-unary, DPP, WMMA, and exception tests passed**.
- Audit tooling: **10 tests passed**.

Targeted timings use the Ryzen 9 7950X3D, CPU 2 affinity, thread CPU time, at least
50 ms per sample, seven alternating before/after pairs, and median times. Both
executables use the same benchmark source. The saved baseline archive is from
`3e02b5d`; the affected production files are unchanged between that commit and
this pass's starting commit `303246c`. Inputs are finite small values; these are
microbenchmarks, not a workload-weighted performance estimate.

| Cases, with default and nondefault modifiers | Before | After | Result |
| --- | --- | --- | --- |
| FP32 EXP/LOG, both architectures/widths | 85–230 ns/wave | 25–63 ns/wave | 3.2–3.9× faster |
| RDNA3 FP32-result FP16/BF16 WMMA, both widths | 7.76–9.32 µs/wave | 121–179 ns/wave | 52–66× faster |
| RDNA4 Wave64 FP32-result FP16/BF16 WMMA | 8.31–11.73 µs/wave | 135–161 ns/wave | 53–87× faster |
| Existing RDNA4 Wave32 v3 WMMA | 140–165 ns/wave | 139–164 ns/wave | −9.2% to +2.3% time change |

Raw timing source, samples, summaries, and validation logs are under the local
`../goc-build/coverage-work/simd-audit-*` paths; GCC/no-SIMD/sanitizer logs are in
the corresponding build directories. Those generated artifacts are not checked in.

## Follow-up: RDNA3 integer WMMA

RDNA3 IU4/IU8 now dispatch to AVX2 in both semantics and wave sizes. The shared
integer WMMA kernel centralizes signed/unsigned widening, pairwise products,
overflow detection, and delayed stores. Compile-time layout parameters select
RDNA3 replicated groups and its single final saturation step; RDNA4 keeps its
two stages. No new executor or policy abstraction was introduced. All supported
sign and CLAMP combinations retain SIMD.

Expanded tests combine every CPU level, signedness/CLAMP, both semantics,
Wave32/Wave64, destination aliases, guards, and independently populated
replicated groups. Existing saturation-cancellation cases distinguish the two
architectures. Clang and GCC each pass all 1,185 tests; ASan/UBSan passes 82
matrix tests.

Seven paired pinned measurements (same timing protocol as above) put RDNA3 at
106–151 ns/wave versus 964–4,311 ns before: 6.7–40.0× faster. RDNA4 control
cases changed by −4.2% to +4.7%, within the agreed 5% trade-off threshold.
Artifacts: `../goc-build/coverage-work/audit-integer-performance/` and the
`audit-integer-*` validation logs in each build directory.
