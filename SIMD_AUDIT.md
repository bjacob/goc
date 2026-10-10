# SIMD, EXEC, and architectural-state audit

This pass starts from `303246c`. It reviews the current production dispatch
families, public EXEC contracts, host FP-control writes, and optional-state
reporting. It fixes two substantial loose-SIMD gaps without introducing a new
bit-exact model. This is a source/dispatch audit backed by the existing test suite
and added boundary/layout tests, not exhaustive hardware validation of every
operand, encoding, or guest mode.

## Follow-up completion checklist

The completed follow-up covers the actionable gaps below. Historical findings and
timings later in this document describe the initial audit pass.

- [x] SIMD RDNA3 IU4/IU8 WMMA in both widths, all signs and final CLAMP.
- [x] SIMD loose packed-result FP16/BF16 WMMA with intermediate narrowing.
- [x] AVX2 coverage for Boolean/CNDMASK, bitfield insert, non-saturating
  integer add, and the remaining ternary/count/conversion gaps listed below.
- [x] Review incomplete exact-state families against rocjitsu and the ISA;
  implement available borrowed/simple rules, retaining explicit unsupported
  results where completion would require forbidden reverse-engineering.
- [x] Assess scalar routing/copy opportunities and justify any retained scalar
  paths by their operation, rather than merely their current implementation.
- [x] Final modifier/mask/global-state regression validation and updated results.

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
That rejected dispatch is not retained. Copy/swap assessment follows below.

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


## Follow-up: move and swap assessment

The remaining routing/copy families were inspected individually:

- MOV16's half extraction, sign change and destination merge are parallel lane
  operations. The AVX2 implementation retains every supported modifier and consumes
  the existing DPP staging output; OMOD/CLAMP stay ignored as specified.
- SWAP16 must preserve the first write when A and D name different halves of
  one VGPR. Its vector implementation retains this ordering, with two masked stores.
- Wave64 PERMLANE64 reads paired lower/upper chunks before either write, allowing
  in-place operation without a separate whole-wave staging array.
- MOV32 is a short masked copy, with optional sign-bit manipulation. A measured
  explicit AVX2 candidate was approximately tied for the default full mask and
  slower for a partial mask; it does not warrant another dispatch/kernel merely
  to label the operation SIMD. SWAP32's two-sided vector-store candidate was
  also slower in representative default and partial-mask cases. Retain their
  simple loops. These measurements do not prove that no faster implementation
  could exist.
- Relative MOV/SWAP select a register uniformly for the wave and delegate to
  ordinary MOV32/SWAP32. Duplicating those kernels would add no benefit.
- READLANE/WRITELANE perform a single scalar access, and READFIRSTLANE performs
  a bit scan followed by a single access; no multi-lane SIMD work is present.
- Scalar-register operations similarly execute once per wave. Wave32
  PERMLANE64 is a specified no-op. Constant comparisons construct an output
  mask directly. Dual issue already delegates to the ordinary operation kernels.

The scratch candidate and API benchmark sources are saved under
`../goc-build/coverage-work/audit-copy-performance/`. These tests include full
and alternating masks to assess dispatch choices, without adding masked cases
to the public benchmark or adding mask-density dispatch.


The kept implementations are three direct intrinsic loops in one small source
file, with no policy/template framework. MOV and SWAP each retain their own
half-selection logic because their read/write ordering differs. The first
SWAP16 candidate reloaded A after the masked D store; same-register partial
masks made that substantially slower. The final version selects the preserved
value from vector registers using a uniform whole-VGPR alias comparison. This
avoids store-forwarding costs without adding EXEC-dependent work or dispatch.

All MOV/SWAP tests now run at every available CPU level, including the existing
all-half-encoding test, captured modifier corpus, DPP references, Wave64 row
controls, guard words, masks, aliases, relative indexing, and validation. Clang
and GCC each pass all 1,189 tests. ASan/UBSan and no-SIMD builds pass all 29
selected move/swap/relative-move tests. The new Clang vector object contains
1,016 text bytes. Test logs use the `audit-copy-*` prefix in each build tree.

Final timing pairs compare `dcea424` with this change using identical API-call
benchmarks, CPU 2 affinity on the Ryzen 9 7950X3D, thread CPU time, a 50 ms
minimum sample, and five alternating before/after pairs. The 36 cases include
both wave widths where applicable, default/selected-half modifiers, full and
alternating EXEC, and separate/same-register operands:

| Operation | Full EXEC | Alternating EXEC |
| --- | --- | --- |
| MOV16 | 1.74–1.81× faster | 1.30–1.40× faster |
| SWAP16 | 1.48–1.62× faster | Wave32 6.7–7.7% slower; Wave64 −2.9% to +2.2% time change |
| Wave64 PERMLANE64 | 1.50× faster | 1.69–1.70× faster |

The SWAP16 partial-mask trade-off is explicit: retain the simple SIMD path for
its substantial full-wave improvement and roughly constant work across masks,
rather than adding the expressly unwanted mask-density specialization. The
same-register reload regression is eliminated. These measurements describe
these cases, not a workload-weighted performance estimate.


## Follow-up: exact architectural-state review

Rechecked rocjitsu at `ffc144c564c`, including `shared/alu_exceptions.h`,
`generated/shared/execute_shared.h`, and the numerical-model map and classifier
review in [COVERAGE.md](COVERAGE.md#borrowed-numerical-model-map). The upstream
explicit arithmetic classifiers cover MUL, SQRT, RCP_IFLAG, and DIV_FIXUP;
sharing an exact numerical model does not supply an exception model.

| Family | Disposition |
| --- | --- |
| RDNA4 DIV_FIXUP FP16/32/64 | Extended reporting to input-flushing modes. Rocjitsu flushes denominator/numerator before classifying arithmetic and suppresses their input-denormal causes. GoC's existing staging already performs this transformation; removing the premature rejection lets the retained classifier consume those transformed bits. The provisional quotient remains untouched. |
| RDNA3 DIV_FIXUP | Retain unsupported exact reporting. Upstream's recovery-flag evidence is gfx1201; GoC's retained corrections, including CLAMP suppression, cannot establish RDNA3 behavior merely by reusing the numerical model. |
| FP32 MUL and arithmetic/FMA/literal forms | Upstream MUL classifier lacks invalid-arithmetic/signaling-NaN branches and computes rounded intermediates with host arithmetic. It is not a complete model for GoC's exact optional-state contract. No complete upstream FP32 FMA exception model was found. Retain explicit unsupported reporting. |
| SQRT and other unary/pseudo-scalar numerical ports | Upstream SQRT only classifies negative input. It does not establish the other exception causes or guest-mode interactions. RCP/RSQ/EXP/LOG numerical helpers likewise do not supply complete reporting. No new classifiers are inferred. |
| RCP_IFLAG | Keep existing RDNA4 reporting. Upstream's original-input zero comparison omits CLAMP/guest-flush treatment and is called before DPP routing; it does not establish complete RDNA3 reporting. |
| FP16 FMA/packed FMA, trig, DIV_FMAS, mixed FMA and DOT2 | Keep existing documented reporting subsets and existing corpus tests. Unsupported architecture/guest-policy combinations remain explicit; extrapolating captured policies is not an available borrowed rule. |
| Floating/scalar comparisons and scalar rounding | Keep existing RDNA4 classifiers. RDNA3 and constant-predicate reporting remain explicitly unsupported; numeric mask/rounding behavior alone does not establish signaling/denormal flag behavior. |
| Conversions/packing, FREXP/LDEXP, min/max, FP64 arithmetic and remaining floating helpers | No complete upstream optional-state model for their currently unsupported combinations. Some have exact bit-based numeric transformations, others are loose-only; neither condition is permission to invent hardware flags. Preserve the non-null exact rejection contract. |
| Integer/carry/SCC/EXEC, MODE and ordinary hardware-register updates | Existing required outputs and simple register-field updates remain implemented; no FP classification is involved. Exact FLUSH_IB reporting remains unsupported because its side effects are not modeled. |

The new DIV_FIXUP test covers all three formats and both widths, all CPU levels,
selected halves, NEG/OMOD/CLAMP, destination aliases, inactive exceptional lanes,
prior-state accumulation, and null opt-out. Literal expected causes cover flushed
zero/zero, divide-by-zero, NaNs, infinity, signed subnormals, and an unflushed
subnormal provisional quotient. Existing architecture rejection tests now reserve
the old unsupported expectation for RDNA3. Numerical results with reporting match
the opt-out path. The new behavior introduces neither host-control changes nor
classification in loose SIMD paths.

This closes the available-model review; it does **not** claim all exact reporting
is implemented. Completing the listed unsupported numerical policies would exceed
the user's borrowed/simple-model scope. The historical DX9 FMAC issue is retained
as a documented limitation, not a blocker: the user explicitly accepted loose-only
support rather than requiring new hardware reverse-engineering.

## Final requirement-to-evidence review

The completion claim is limited to the actionable audit items and the agreed
borrowed/simple-model scope. It is not a claim of complete hardware equivalence
for every instruction or unsupported encoding.

| Requirement | Current evidence |
| --- | --- |
| Supported modifiers retain loose SIMD unless justified | Dispatch review above and new WMMA, integer, DS, MOV16/SWAP16 kernels; source/output modifiers and DPP feed the vector paths. Existing and expanded per-family tests run all supported CPU levels. Retained scalar operations have operation-specific or measured reasons. |
| Loose arithmetic normally uses SIMD | Matrix and AVX2 arithmetic gaps addressed in the committed follow-ups. Explicit exceptions include scalar-register operations, small measured copy/half-Boolean cases, ordered scatter, and already-documented model/tier limitations. No density-dependent dispatch was introduced. |
| EXEC and aliases remain faithful | Full test suites include zero/full/partial masks, upper Wave64 lanes, DPP row enables, source participation, guard words and aliases. New routing kernels stage required cross-lane inputs or paired exchange chunks before writes. Destinations use masked vector stores. |
| EXEC absent where ignored | Public scalar/pseudo-scalar and WMMA/SWMMAC interfaces retain their EXEC-free signatures; no API migration in these follow-ups reintroduced EXEC. SAVEEXEC/WREXEC take it as required register data. |
| Optional state contract | Exact non-null rejection/atomicity tests, existing reporting corpora, new DIV_FIXUP flush rules, and loose non-null tests. Numeric outputs stay mandatory; loose state computation is not added to SIMD kernels. |
| No explicit host FP-control mutation | Earlier one-time source/archive audit remains recorded above. Follow-ups add integer/vector operations or ordinary host arithmetic, with no environment-setting helpers. Existing host-control regression tests remain in the final suites; no repeated host-write audit was required. |
| Stay within numerical-model scope | Provenance review above; no hardware probes or new reverse-engineered model were added. Incomplete reporting remains explicit rather than inferred from host exceptions or numerical results. |
| Maintain readable organization | Direct per-operation kernels and limited existing-kernel extensions, with shared rules only where already used. Rejected slow candidates removed. Per-family performance and costs documented rather than reduced to a misleading overall percentage. |
| Public/API/tooling validation | Standalone C99/C++17 header compilation, linkage/embedding checks, API-wide high-bit validation tests, regenerated architecture and global-state naming reports, and ten tooling tests. Naming reports make no reporting-accuracy claim. |

Validation of the final DIV_FIXUP change: **1,190 tests pass** in each of Clang,
GCC, and the no-SIMD build. **32 DIV_FIXUP/global-state tests pass under
ASan/UBSan**, and all ten audit-tooling tests pass. Regenerated architecture and
state-naming reports are unchanged (587 shared-mnemonic reviews, 962 resolved
state-output signatures). The full ASan/UBSan run also passed **all 1,190 tests**
in 279 seconds, including the exhaustive DPP cases and benchmark smoke tests.


**Disposition: all actionable audit items are complete within the agreed scope.**
The final production implementation is `127ecf1`; the closing documentation
commit changes no production code. Final full-suite results are 1,190/1,190 in
Clang, GCC, Clang without specialized SIMD, and ASan/UBSan. The corresponding
logs are `coverage-work/audit-state-tests.log` in each out-of-tree build directory;
tooling results are in `goc-build/coverage-work/audit-final-tooling-tests.log`.
Local implementation commits are `3d50b6d`, `85c45ad`, `b3c65a7`, `dcea424`,
`df562cb`, and `127ecf1`, following the initial audit/fixes at `1d0b6b6`.

Completion includes the documented measured scalar exceptions and explicit
unsupported exact-state policies; it does not turn those limitations into a claim
of universal SIMD or complete exception emulation. No new empirical numerical
model was reverse-engineered, no host FP-environment setter was introduced, and
no sparse-mask specialization was added.
