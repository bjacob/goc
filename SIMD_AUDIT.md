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
- [x] SIMD loose packed-result FP16/BF16 WMMA with intermediate narrowing.
- [x] AVX2 coverage for Boolean/CNDMASK, bitfield insert, non-saturating
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

## Initial dispatch review and scalar cases

This section records the initial pass; completed follow-ups below supersede its
WMMA, arithmetic, and DS routing gaps. The top-level scan covers 91 production `.cpp` files; CPU kernels live in 101
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

## Follow-up: packed-result WMMA

Both architectures and wave sizes now have loose scalar/AVX2 paths for packed
FP16/BF16 results. They share the existing WMMA vector widening, source modifiers,
FMA, and staged destination writes. Packed output adds explicit accumulator
narrowing and output packing, not a separate matrix executor. RDNA3 narrows after
each two products and preserves the unselected destination half; RDNA4 narrows
after four products, follows its physical K-stage order, and packs adjacent rows.
BF16 narrowing truncates; FP16 uses nearest-even and the finite-overflow policy.
All sign modifiers, RDNA3 high-half selection, and FP16 saturation retain SIMD.

Scalar loose calls previously happened to use the exact model. They now use the
same staged host-FMA approximation as SIMD. Consequently loose results can differ
from those old empirical results, especially around cancellation. Exact calls
retain the borrowed integer model unchanged; no bit-exact promise was weakened.

New tests cover all modifiers, layouts, aliases, untouched halves, and guards;
an independent small-integer reference; architecture-distinguishing intermediate
rounding; finite overflow versus input infinity; and scalar/SIMD comparisons on
random encodings, subnormals, NaNs, and infinities. Clang and GCC each pass all
1,188 tests. ASan/UBSan and no-SIMD builds each pass 85 matrix tests.

Packed loose timings now range from 168–615 ns/wave versus 20.2–34.5 µs before,
a 49.8–188.3× improvement in the targeted small-value cases. The old baseline
used the empirical integer model even in loose mode, so these numbers include
both selecting loose arithmetic and vectorization. Seven paired samples use the
same pinned/thread-time protocol above. Artifacts live in
`../goc-build/coverage-work/audit-packed-performance/`.

The sixteen existing FP32-result WMMA control cases changed by −4.0% to +0.9%
(median −1.1% time), with no material regression. Their paired samples are in
`../goc-build/coverage-work/audit-packed-f32-control/`.

## Follow-up: AVX2 arithmetic gaps

New eight-lane paths cover 32-bit Boolean/CNDMASK, 16-bit AND, BFI, the four
simple ternary operations, wrapping integer add/subtract/ADD3, signed/unsigned
16-to-32 widening, FP32-to-I32/U32 truncation, and FP64-to-I32 truncation.
DPP and Wave64 continue to use the same arithmetic kernels; accepted modifiers
and optional loose-state pointers do not disable SIMD. Existing packing,
saturation, and floating conversion kernels share their arithmetic rules where
applicable, without adding a generic vector executor.

Candidates were also implemented and tested for half-word OR/XOR/NOT and
CNDMASK. They were removed after paired measurements showed slower performance
for both full and alternating EXEC, with default and nondefault selectors.
Those opcodes retain the portable half-word path below v4. This is an
opcode-level choice, not a modifier- or mask-dependent fallback. Wave32
MBCNT_HI is also deliberately retained: its mathematical result is simply B,
so there is no population-count arithmetic to vectorize. Its Wave64 form
already has AVX2 arithmetic.

The existing tests exercise all half encodings, truth tables, modifier/mask
combinations, DPP and Wave64, random raw words, host-rounding-independent
integer conversions, and aliases. A new truncation test combines random masks,
in-place output, every source/output modifier, and a non-null loose-state
pointer against the independent integer-bit reference.

Validation: Clang and GCC each pass all 1,189 tests; ASan/UBSan and no-SIMD
builds each pass the 89 affected-family tests. Initial candidate measurements
are preserved in `../goc-build/coverage-work/audit-avx2-performance/`, including
the rejected half-word kernels. Final selected-kernel measurements and existing
packing/saturation controls are in `audit-avx2-final-performance/`.

The Clang archive's aggregate `size` text total increases from 2,135,862 to
2,142,141 bytes (+6,279 bytes, 0.29%) for these AVX2 additions. The changes add
ordinary intrinsic kernels and remove the obsolete ternary-support predicate;
there is no new policy/adapter layer. The two retained half-word exceptions
avoid adding kernels that regress their measured workloads.

Final seven-pair Clang timings cover 80 cases (full and alternating EXEC,
default and nondefault modifiers, including unchanged controls). The 28 newly
covered full-wave cases improve by 2.6–27.1%, median 14.5%. Their alternating-
mask cases range from 10.1% faster to 9.4% slower, median 2.4% faster: the old
scalar stores sometimes did less work. No mask-density dispatch was added.

Control cases have a median 0.5% time increase. One exception is default-modifier
half-word CNDMASK, which measured 16–20% slower despite retaining byte-identical
machine code for its 2,198-byte arithmetic function. Its address changed as the
32-bit specialization grew. Separate alignment and direct-store experiments did
not produce a clean improvement and were discarded. This code-placement-sensitive
microbenchmark regression is explicitly retained in the results; it must not be
misrepresented as every case getting faster. Modified CNDMASK16 and the existing
packing/saturation controls remain near their previous times. Static consumers
will have their own code layout.

## Follow-up: DS routing

`ds_swizzle_b32` and ordinary `ds_bpermute_b32` now use AVX2 for both wave
widths and both semantics. Swizzle retains SIMD for all four offset classes.
Backward permute retains the RDNA3 half-wave/RDNA4 full-wave distinction.
Both compute bounded source indices in vectors, filter source participation,
snapshot results before any destination writes, and use masked stores. There
is no mask-density dispatch. A single non-template kernel per operation takes
wave/group sizes; two local helpers centralize source-EXEC filtering and staged
masked writes without introducing a generic routing framework.

Forward permute remains scalar because its ordered scatter resolves collisions
in favor of the highest active source. Force-inactive backward permute also
remains scalar: a tested AVX2 gather candidate made its full-mask Wave32 case
about 30% slower and Wave64 about 12% slower. Unlike ordinary backward permute,
this operation has no source-EXEC filtering work to amortize gather overhead.
That rejected dispatch is not retained. Copy/swap assessment remains open.

All 11 existing DS tests now execute at every available CPU level. They cover
all 65,536 swizzle offsets in both wave widths, captured swizzle corpora,
independent routing references, inactive sources, every gather source/destination
pair, address overflow, aliases, guards, zero EXEC, and error preservation.
Clang and GCC each pass all 1,189 tests; ASan/UBSan and no-SIMD builds each pass
the 11 DS tests. The added vector object contains 1,821 text bytes in the Clang
build. Source, timing samples, and validation logs are saved under local
`../goc-build/coverage-work/audit-routing-*` paths and corresponding build trees.

Final before/after timings compare `b3c65a7` against this change on CPU 2 of the
Ryzen 9 7950X3D, using thread CPU time, at least 50 ms per sample, and five
alternating pairs. Representative controls cover all four swizzle classes,
both wave widths, both backward-permute groupings, and full/alternating EXEC.
Full-mask swizzle improves 1.18–1.93×; ordinary backward permute improves
1.13–1.26×. Alternating-mask cases improve 1.10–1.83×. The retained scalar
force-inactive full-mask controls are unchanged within 0.3%; its alternating
controls vary from 8.6% faster to 10.5% slower. These are targeted microbenchmarks,
not a whole-library speedup or a promise of mask-independent cycle counts.
