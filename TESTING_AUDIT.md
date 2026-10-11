# GoC testing audit

Audited source: `536bb22`, with a clean working tree. This is an audit, not a test-suite rewrite.

## Verdict

The suite has broad instruction and behavioral coverage, but does **not yet consistently meet the requested testing policy**. The clearest actionable gaps are toolchain-independent RNG sequencing and a shared, empirically justified loose-comparison policy. Reference provenance needs a more explicit inventory; computed references are not intrinsically a deficiency.

| Requirement | Finding |
| --- | --- |
| Every instruction has a unit test | All **1,791 public instruction APIs** have references from currently compiled family test translation units, excluding generic API/exception/empty-EXEC checks. No completely unreferenced instruction API found. This is a coverage floor, not proof that each API has every necessary numerical assertion. |
| Loose floating arithmetic uses empirically chosen `(atol, rtol)` | **Inconsistent.** Some tests already implement the desired formula; others use ULP checks, absolute-only bounds, or exact scalar/SIMD equality. No consolidated calibration artifact or per-family tolerance provenance was found in the reviewed tests and documentation. |
| Deterministic pseudorandom inputs, independent of toolchain | **Mostly good, with a concrete defect.** Explicit standard engines and simple integer recurrences predominate; seven expressions make RNG call ordering unspecified. |
| Justified bit-exact references | **Mixed evidence, not a blanket failure.** There are extensive hardware captures, fixed digests, offline golden matrices, and computed integer oracles. Computed references are welcome where their equivalence is established. Their provenance/equivalence is not systematically recorded per instruction and mode. |
| Computed references for loose-only implementations | Already common and appropriate, e.g. independent FP8 decoding and mathematical accumulation. Comparator policy still needs harmonization. |
| Size, maintainability, latency, and essential behavior | Strong attention to masks, aliases, modifiers, architecture differences, rounding boundaries and optional state. Some Cartesian test products are disproportionately expensive; reducing repeated combinations should precede reducing semantic cases. |

## Scope and strength of evidence

The entry-point census is exhaustive over declarations in `include/goc/detail/goc_instructions.h`. It uses undefined `goc_*` references in the current Clang test objects, including references produced by adapters/macros. Objects are filtered against the current compilation database; stale `rcp_exact_test.cpp` and `scalar_rdna3_test.cpp` objects were excluded. This avoids claiming coverage from obsolete binaries.

The detailed map is [API_TEST_MAP.md](/tmp/goc-testing-audit/API_TEST_MAP.md), with machine-readable [api-test-map.json](/tmp/goc-testing-audit/api-test-map.json) and [scan.py](/tmp/goc-testing-audit/scan.py). All 1,791 APIs still have a nongeneric test reference after excluding stale objects. A symbol reference alone cannot establish that a call exercises exact arithmetic, fuzzy loose arithmetic, or a validated oracle. A complete per-entry-point provenance/semantic-assertion manifest remains a recommendation, not an accomplishment claimed here.

The comparator, reproducibility, reference and interaction review combines repository-wide searches with source inspection of representative arithmetic, unary, FP8 dot, WMMA, packed WMMA, integer, routing, DPP and exception tests. It does not claim a line-by-line proof of every reference algorithm. Existing full validation at this commit reports **1,190 passing CTest cases** in Clang, GCC, no-SIMD and ASan/UBSan, plus ten tooling tests. These results establish regression health, not test adequacy. No additional full run was necessary for this read-only audit.

## 1. Loose comparisons: consolidate policy and calibrate it

The requested implementation is already available locally in [hrx-system math.h](/home/benoit/workspace/hrx-system/runtime/src/iree/base/internal/math.h:43), and documented in [IREE's matchers](/home/benoit/workspace/iree/runtime/src/iree/tooling/buffer_view_matchers.h:42):

`abs(actual - expected) <= atol + rtol * abs(expected)`

The reference operand controls the relative term. Equal infinities compare equal; NaN handling is explicit (the existing helper accepts two NaNs). Signed-zero/payload requirements, when relevant to a contract, need separate bit checks rather than being accidentally imposed or erased by fuzzy comparison. Integer results, comparison masks, carry/SCC, preserved lanes and optional-state bits remain exact even when instruction semantics are loose.

Examples of current divergence:

- [wmma_test.cpp:50](/home/benoit/workspace/goc/tests/wmma_test.cpp:50) has a private NumPy-style helper with `atol = rtol = 1e-5`. This is the right formula, but it is local and its chosen constants have no attached calibration record.
- [interp32_test.cpp:28](/home/benoit/workspace/goc/tests/interp32_test.cpp:28) has another equivalent finite comparison, using `1e-44 + 2e-7 * abs(expected)`.
- [unary_test.cpp:61](/home/benoit/workspace/goc/tests/unary_test.cpp:61) and [fp8_dot_test.cpp:38](/home/benoit/workspace/goc/tests/fp8_dot_test.cpp:38) use GoogleTest floating-equality checks instead of an explicit absolute/relative contract.
- [half_unary_test.cpp:33](/home/benoit/workspace/goc/tests/half_unary_test.cpp:33) and related half tests use a one-encoding-step bound; trig tests use absolute bounds such as `3e-7`, and some use domain-specific bounds.
- [wmma_test.cpp:410](/home/benoit/workspace/goc/tests/wmma_test.cpp:410) compares loose scalar and SIMD output words exactly except for NaNs, assuming the same accumulation order. This is useful differential testing, but it overconstrains future mathematically sound implementations if it serves as the only acceptance criterion for that domain.
- Exact checks for deliberately exactly representable inputs, bit routing, untouched lanes, and explicit architectural discontinuities are valuable. Do not mechanically convert every `EXPECT_EQ` to fuzzy comparison.

The scan found 13 `FLOAT_EQ`/`DOUBLE_EQ` assertion sites in four test files, six `NEAR` sites in four files, and no named `atol`/`rtol` variables in `*_test.cpp`. These counts describe syntax only; they are not counts of instructions tested numerically.

Recommended change:

1. Borrow the small hrx-system comparator into one test helper, preserving attribution and explicit special-value behavior. Avoid a general-purpose test framework.
2. Define named tolerances by operation/format and, only when justified, numerical domain. Share a tolerance between CPU implementations that implement the same loose contract.
3. Calibrate against deterministic, stratified inputs and an appropriate independent reference: normal values over a broad exponent range, cancellation, near-zero outputs, subnormal boundaries and overflow. Record the tested implementation/compiler/CPU, corpus identity, worst errors and selected margin. Use a separate deterministic holdout corpus.
4. Keep tolerances fixed in committed tests. Never adapt them to errors observed during a normal test run. Do not widen a bound to hide a lane-map, sign, omitted-operation, CLAMP, or reference bug.
5. Explicitly document domains where one global pair is too permissive or too restrictive, especially cancellation in matrix reductions. Keep categorical infinity/NaN/overflow and stage-boundary tests alongside the finite fuzzy tests.

This audit does **not** set new tolerance values: the measurements needed to justify them have not been performed.

## 2. Deterministic generation: fix RNG sequencing

The suite already follows the important part of the plan: specific engines such as `std::mt19937` and `std::minstd_rand`, or explicitly coded unsigned recurrences, with local seeds. The scan found `mt19937` in 59 test translation units and no `default_random_engine`, distributions, `random_device`, or `std::shuffle` in test C++ sources.

However, these seven sites call the same engine twice in operands whose evaluation order is unspecified:

- [conversion64_test.cpp:153](/home/benoit/workspace/goc/tests/conversion64_test.cpp:153)
- [div_fixup_architecture_test.cpp:62](/home/benoit/workspace/goc/tests/div_fixup_architecture_test.cpp:62)
- [div_fmas_test.cpp:397](/home/benoit/workspace/goc/tests/div_fmas_test.cpp:397)
- [div_fmas_architecture_test.cpp:76](/home/benoit/workspace/goc/tests/div_fmas_architecture_test.cpp:76)
- [ds_permute_test.cpp:39](/home/benoit/workspace/goc/tests/ds_permute_test.cpp:39)
- [wmma_test.cpp:421](/home/benoit/workspace/goc/tests/wmma_test.cpp:421)
- [wmma_test.cpp:466](/home/benoit/workspace/goc/tests/wmma_test.cpp:466)

For example, `(uint64_t(random()) << 32) | random()` does not specify which draw supplies which half. This is a portability failure even if today's two compilers happen to choose the same order. Separate declarations and draws fix it. Do not flag comma-separated declarations or mutually exclusive conditional branches as the same problem.

A tiny sequenced word-pair helper and fixed known-output checks for the generators used by captured digests would guard this requirement cheaply. Continue explicit unsigned arithmetic and fixed bit-to-float transformations; do not add standard distributions or libm-based random-input transforms. Follow up on implementation-defined signed narrowing/shifts where an input or reference relies on them: this audit has not exhaustively proved their portability.

Pseudorandom inputs should supplement curated values, not replace them. Uniform raw words rarely hit the exact tie, overflow, saturation, NaN-precedence or cross-lane boundary that defines an instruction's behavior.

## 3. Exact references: equivalence and provenance, not mandatory golden storage

The clarified requirement is applied here: a computed reference is preferable to bulky golden data when its equivalence to an already-validated bit-exact implementation is established, including evidence inherited from rocjitsu. Its mathematical plausibility or its current agreement with GoC alone does not establish that equivalence for arbitrary hardware behavior.

Useful existing patterns:

- [dot_fixtures.h](/home/benoit/workspace/goc/tests/dot_fixtures.h:5) records hardware, architecture and raw expected words, including exceptional cases.
- [dense_golden.h](/home/benoit/workspace/goc/tests/dense_golden.h:5) records a deterministic input stream and independent offline integer matrix product. This provides a compact, understandable lane/arithmetic oracle.
- [packed_wmma_test.cpp](/home/benoit/workspace/goc/tests/packed_wmma_test.cpp:65) and [wmma_replicated_exact_test.cpp](/home/benoit/workspace/goc/tests/wmma_replicated_exact_test.cpp:61) use captured matrices and modifier/architecture boundary cases.
- Hardware hashes, e.g. [scalar_integer_test.cpp](/home/benoit/workspace/goc/tests/scalar_integer_test.cpp:13), compact large corpora. The byte-wise hash helper explicitly defines byte order. Fixed expected digests are golden references, even though output words are not individually stored.
- Independent computed models such as [half_fma_reference.h](/home/benoit/workspace/goc/tests/half_fma_reference.h:14) and [conversion32_reference.h](/home/benoit/workspace/goc/tests/conversion32_reference.h:25) can provide cheap broad coverage. Preserve them when their relevant domains are validated; do not replace them simply because they compute expected values.
- Simple Boolean truth tables can be justified directly from the ISA. They do not require a hardware capture for every word. Nontrivial floating rounding, NaN priority and exception policies need stronger domain-specific evidence.

The gap is traceability. There is no single per-family record connecting **computed oracle → established equivalence evidence → supported domain/modes**. An oracle validated for finite RNE arithmetic is not automatically validated for NaN payloads, flushing, OMOD, clamp or a different architecture. Computed checks and hardware fixtures often coexist, but their exact relationship is not consistently explicit.

Recommended rule for the inventory:

- **Established computed reference:** cite upstream provenance, ISA proof for a simple operation, or validation evidence and its domain. Keep it; this is a positive result.
- **Captured/offline golden:** retain where needed for empirical behavior; include generator, input ordering, architecture and relevant FP/mode settings.
- **Differential reference:** scalar/SIMD agreement or reuse of a production numerical helper. Useful supplemental coverage, but not independent evidence that both match the hardware contract.
- **Unestablished computed reference:** preserve provisionally as a regression check, identify the unproven domain, and add only the compact independent evidence needed. Do not demand giant golden corpora by default.

Digest failures are difficult to localize. Prefer per-instruction/modifier/domain checkpoints plus a small readable edge corpus, rather than one enormous digest or storing every output. Numeric fixtures should not be regenerated automatically from GoC itself during tests.

## 4. Coverage trade-offs and latency

Source census: **161 `*_test.cpp` files, 46,708 lines**; **63 `*_reference.h` files, 4,613 lines**; **64 `*_hardware.h` files, 13,274 lines / about 0.91 MB**. Naming-based counts exclude other fixtures and inline golden constants. Size alone is not evidence of waste.

Recent full-run wall times were Clang 36.75 s, GCC 24.47 s, no-SIMD 30.98 s, and ASan/UBSan 279.06 s at `-j24`. Some builds/tests ran concurrently, so these are observed validation costs, not isolated performance benchmarks. Summed per-case durations are elapsed-time totals, not CPU time.

| Test | Clang seconds | ASan/UBSan seconds |
| --- | ---: | ---: |
| DppHalfMinmax.MasksAliasesAndRandomWords | 36.70 | 279.01 |
| DppInteger16Ternary.MasksAliasesAndRandomWords | 29.20 | 234.28 |
| DppHalfBinary.MasksAliasesAndRandomWords | 27.95 | 202.37 |
| DppInteger16.MasksAliasesAndRandomWords | 21.90 | 197.43 |
| DppHalfFma.MasksAliasesAndRandomWords | 21.44 | 168.67 |

For example, `DppHalfMinmax.MasksAliasesAndRandomWords` crosses CPU level × nine operations × DPP/modifier combinations × EXEC masks × five source layouts × four destination aliases, while restarting the same input recurrence in the innermost case. This spends substantial time reusing the same numerical words across control combinations. The coverage is valuable, but the full product is not automatically necessary.

Keep these essential behavior classes explicitly covered:

- Numerics: ordinary values, cancellation, rounding ties, exponent extremes, signed zeros, NaNs/infinities, denormal controls and saturation/overflow boundaries where relevant.
- Operand mapping: distinct lane/register values, packed high/low selectors, Wave64 upper half, WMMA replication and intermediate narrowing, architecture-specific differences.
- Modifiers: each modifier independently and interactions that change stage ordering, especially ABS+NEG, OMOD+CLAMP, half selection+write preservation, DPP+source modifiers.
- Execution: zero/full/alternating/single-boundary masks, inactive source participation, DPP row/bank filtering, and no EXEC parameter for instructions that ignore it.
- Memory/API behavior: legal aliases, untouched registers/lanes/halves, unaligned storage and guards, invalid requests preserving outputs.
- Global state: required masks/SCC/EXEC results, optional accumulation, opt-out, unsupported exact requests, and loose calls with non-null state.
- Implementations: each available scalar/SIMD path and architecture-specific variant. A test passing on one host does not prove an unavailable ISA tier was exercised.

Recommended structure: compact golden or established-reference anchors for every semantic branch; deterministic stratified numerical samples; exhaustive small encodings where genuinely cheap; focused interaction cases instead of the largest Cartesian products. Test DPP routing exhaustively in its dedicated tests and retain a smaller but deliberate arithmetic-integration matrix. Consider extended/nightly labels for expensive products only after the default suite retains all essential behavior classes. Preserve readable direct loops and instruction-specific cases rather than introducing a policy-heavy test framework.

## Prioritized follow-up

1. **Fix the seven RNG sequencing sites.** Small, concrete portability correction; add generator stream checks where fixture identity depends on the stream.
2. **Centralize fuzzy comparison and establish tolerance calibration.** Start with WMMA, unary, trig, FP8 dot and interpolation; retain exact checks where the tested behavior is exact even in loose mode.
3. **Record reference equivalence by family/domain.** Use the compiled API map as a starting point, then map each entry to a semantic assertion and its oracle evidence. Accept proven computed references; distinguish differential checks. Do not claim exact/loose provenance completeness from symbol references.
4. **Reduce the five expensive DPP products deliberately.** Preserve the behavior checklist above and measure test-duration changes under the same conditions. Do not trade away architecture or modifier interactions just to improve the headline count/time.
5. **Improve compact-corpus failure diagnostics.** Add checkpoints/readable anchor cases where a digest currently gives little localization, rather than expanding all captures into raw output arrays.

No production or test files were changed by the original audit. The remaining work is concrete, but the suite's breadth and existing captures provide a strong base; it does not need wholesale replacement.


## Implementation checklist

The active follow-up is to complete all five action items above, preserving the
clarified policy that established computed exact references are welcome.

- [x] Fix the identified RNG sequencing sites and pin fixture-dependent input streams.
- [ ] Centralize loose fuzzy comparisons and record empirical tolerance calibration.
- [ ] Map instruction tests to reference evidence and establish missing oracle provenance.
- [x] Reduce the expensive DPP products while preserving essential interactions; measure latency.
- [ ] Improve compact-corpus diagnostics with justified checkpoints/readable anchors.
- [ ] Run final compiler/SIMD/sanitizer validation and record remaining limitations.


### RNG sequencing completed

All seven identified two-draw expressions now assign each draw to a separate
local before combining words. No engine, seed or distribution policy changed;
the previously unspecified draw-to-word mapping is now explicit. A repeat scan
finds no remaining same-line two-draw bitwise expressions in unit-test sources.
`DeterministicInputs.ExecMaskStream` pins the 84-entry mask sequence, and
`DeterministicInputs.DenseMatrixFixtureStream` pins the 768 mapped words used by
the offline dense matrix fixture. Their digests were calculated independently
from the specified integer recurrences. All 80 affected tests pass in both Clang
and GCC; logs are `coverage-work/testing-rng-tests.log` in those build trees.

### Shared fuzzy comparator: first migration and measurements

Added `tests/fuzzy.h`, with formula/special-value/asymmetry/boundary tests and
failure diagnostics, and `tests/loose_tolerances.h` with named existing bounds.
WMMA's captured loose results, interpolation's computed comparisons, and
FP32/FP16 trig's numerical comparisons now use the shared helper. Exact
architectural/untouched-bit checks remain unchanged. All 70 affected tests pass
with Clang and GCC.

Initial measured error envelopes and their limitations are recorded in
[tests/LOOSE_TOLERANCES.md](tests/LOOSE_TOLERANCES.md). No tolerance was widened.
This is partial completion: remaining loose comparisons, deterministic holdout
calibration, and the reference-provenance inventory still need work.

### DPP integration products reduced

The five `MasksAliasesAndRandomWords` tests below still exercise every previously
covered operation, CPU selector, DPP descriptor, arithmetic modifier, source
layout, destination alias and numerical batch. For every DPP descriptor,
unmodified arithmetic retains all 84 masks, including every single active and
inactive lane and the deterministic random sequence. Modified arithmetic uses
14 masks: zero/full, both alternating patterns, both half-waves, and single lanes
at both ends of every eight-lane SIMD chunk. This retains inactive-source,
FI/BOUND, row/bank filtering and mask/alias interactions without repeating every
random/single-inactive mask under every modifier. Dedicated exhaustive routing,
modifier-encoding and captured hardware tests are unchanged.

This is a deliberate reduction in cross-product coverage, not an assertion that
all interactions are equivalent. A future mask/modifier regression should add a
focused case. One short shared mask-selection helper expresses the rule for the
five callers; their instruction-specific reference loops remain directly readable.

Controlled Clang Release measurements (median of three runs, these five tests
only, `ctest -j24`, no concurrent builds/tests):

| Test family | Before (s) | After (s) |
| --- | ---: | ---: |
| DppHalfMinmax | 7.30 | 1.43 |
| DppInteger16Ternary | 5.38 | 1.07 |
| DppHalfBinary | 5.54 | 1.10 |
| DppInteger16 | 4.27 | 1.05 |
| DppHalfFma | 3.89 | 0.75 |
| Total wall time | 7.32 | 1.45 |

Logs: `goc-build/coverage-work/testing-dpp-cost/{before,after}-{0,1,2}.log`.
These are test-duration improvements, not production performance changes.

Validation: all 1,195 tests pass in Clang and GCC Release builds. All 22 tests
in the five affected DPP families pass under ASan/UBSan, including their unchanged
hardware-corpus and exhaustive modifier tests. Logs are
`coverage-work/testing-dpp-full.log` in both Release build trees and
`coverage-work/testing-dpp-targeted.log` in the sanitizer build tree.

### Unary numerical comparisons and independent holdout

Migrated the six FP32 unary numerical comparison sites from implicit ULP bounds
to the shared `(atol, rtol)` helper. Two deterministic stratified numerical tests
exercise all eleven operations, 32 modifiers and available CPU selectors.
The chosen relative bound of 2.5e-7 has measured margin on calibration and a fresh
holdout stream under both Clang and GCC; see
[tests/LOOSE_TOLERANCES.md](tests/LOOSE_TOLERANCES.md). Signed-zero, classification,
hardware exact cases and untouched-bit assertions remain unchanged. Trig
architecture comparisons also use the shared helper, preserving their bounds.
All 17 affected tests pass with both compilers. FP8/FP64 and the remaining
family/domain coverage and calibration work are still open.

### FP8 dot, FP64 and DPP unary numerical comparisons

All remaining GoogleTest floating/near assertions now use explicit shared
numerical tolerances, except small-integer FP8 checks which now require exact
numeric equality. Added deterministic calibration/holdout corpora for mixed
FP8/BF8 reductions (including cancellation) and FP64 sqrt/rcp/rsq. Measurements
and domains are recorded in [tests/LOOSE_TOLERANCES.md](tests/LOOSE_TOLERANCES.md);
all 27 affected tests pass on Clang and GCC. Hardware FP64 SFU comparisons retain
their separate, intentionally looser hardware-relative bound. Half-format ULP
comparisons, remaining calibration domains and the oracle inventory are still
open; removal of comparison macros is not used as a completion criterion.

### Half-format, pseudo-scalar and reciprocal tolerance migration

Replaced the remaining encoding-distance numerical checks with the shared fuzzy
rule, while keeping exact structural/classification and hardware-reference checks.
Measured the exhaustive half sweeps and existing modifier/architecture corpora;
results and limitations are in [tests/LOOSE_TOLERANCES.md](tests/LOOSE_TOLERANCES.md).
The plain numerical comparison now lives in `tests/fuzzy_compare.h`; `fuzzy.h`
adds assertion diagnostics. This extra header serves the existing benchmark
caller of the reciprocal helper, so it can share the rule without GoogleTest.
All 1,201 tests pass on Clang and GCC; the last two migrated files additionally
pass 11 targeted Clang tests. No production code changed.

The tolerance audit is not yet closed: broader WMMA/interpolation calibration
and the classification of existing bitwise/differential numerical checks are
still needed alongside the reference-evidence inventory.

### Dense unary digest localization

Added 88 cumulative checkpoints to the large RCP/SQRT/RSQ/EXP/LOG domains,
shared by vector, pseudo-scalar and RCP_IFLAG tests. They come from unchanged
rocjitsu headers, with all final hashes verified against the retained hardware
captures before fixture emission. No GoC output was used as the oracle.
See [tests/CORPUS_DIAGNOSTICS.md](tests/CORPUS_DIAGNOSTICS.md) for exact provenance,
the standalone generator and reproduction commands. Seven affected tests pass
under Clang and GCC; an out-of-tree injected checkpoint error correctly reports
its million-input interval. Other digest families remain part of the pending
reference/diagnostic review, so the overall diagnostics item remains open.

### Reproducible reference inventory and first explicit reviews

Added `tools/testing_audit.py`, with regression tests for stale/missing objects,
generic-only references, explicit review attachment and broken evidence links.
It replaces the one-off `/tmp` symbol scan with a reproducible inventory covering
all 1,791 declared instruction APIs; none lacks a compiled family reference.
Fourteen tooling tests pass. Five manually inspected family-file reviews now
record assertions, oracle provenance/equivalence scope and diagnostics in
`tests/reference_reviews.json`; 149 family files remain to be reviewed.
See [tests/REFERENCE_COVERAGE.md](tests/REFERENCE_COVERAGE.md). Compiled references
and reviewed-file membership are expressly not semantic-completeness proofs.

### Integer/Boolean/routing oracle review

Reviewed and recorded twenty more family files, bringing the explicit inventory
to 25 reviewed and 129 pending. These use bounded integer arithmetic, truth
tables or direct routing models, with literal/hardware anchors for semantic
exceptions. Exact assertions in loose mode are intentional for these bit/integer
results; replacing them with fuzzy comparisons would weaken the contract.
`Permlane.HardwareCorpus` already checks every result alongside its hardware
digest; it now includes the missing mask/selector/opcode/lane failure context.
All six permlane tests pass with Clang and GCC, and all fourteen audit-tool tests
pass. The reference and diagnostic audit remains incomplete for other families.

### Scalar-register and DPP integration oracle review

Added 22 explicit reviews: scalar result/SCC/EXEC/conversion/rounding/move
families, ten DPP integer families and DPP8/DPP16. The inventory now has 47
reviewed family files and 107 pending. Captured state/result references remain
distinct from alias-only differential checks and unsupported strict-exact
conversion requests. Documented the bounded dyadic proof for exact arithmetic
assertions in DPP routing tests, and added missing control/EXEC/lane diagnostics
to DPP8/DPP16 hardware corpora. All seven affected DPP tests pass on Clang/GCC;
the fourteen tooling tests pass. No production code changed.

### Integer/byte conversion and packing reference review

Recorded four more family reviews (51 reviewed, 103 pending). Computed references
for sign extension, saturation, small-integer power-of-two scaling and FP16 field
packing have explicit scope and justification. Retained hardware digests and
added per-output reference assertions to integer/byte DPP captures and both
packing encoding corpora, improving failure localization without new fixture
arrays. All 29 affected tests pass on both Clang and GCC. The inventory still
finds family references for all 1,791 APIs; this does not establish complete
semantic coverage. The broader reference, tolerance and final validation work
remains open.

### FP8 widening and narrowing oracle evidence

Reviewed both FP8 conversion families (53 reviewed, 101 pending). Lossless finite
widening has a direct encoding proof, with raw hardware captures covering all
256 codes and exceptional NaN encodings. Added per-code failure localization to
that retained capture. Narrowing now explicitly checks its computed reference
against the stored captured outputs, independently of the implementation under
test; documented that this establishes captured-domain agreement, not universal
hardware equivalence. Both widths retain their golden comparisons and now report
source/seed/column context. All 22 affected tests pass on Clang and GCC, and all
14 tooling tests pass. No production code or golden values changed.

### Packed and normalized conversion references

Three more reviews bring the queue to 56 reviewed and 98 pending. Recorded the
loose/fallback-only scope explicitly, including strict-semantics rejection and
optional-state behavior. The integer-significand and RTZ-search references are
now checked directly against existing hardware capture columns in both widths;
new traces identify source, column, CPU and applicable saturation mode. This
retains existing golden evidence rather than assuming agreement with GoC proves
equivalence. All 27 affected/integration tests pass on Clang and GCC; all 14
tooling tests pass. Production implementations and API support are unchanged.

### Byte quantization and packing architecture review

Reviewed two more families (58 reviewed, 96 pending). Recorded byte rounding
boundary/adjacent-value coverage, preserved-byte and alias checks, and the
architecture-specific strict-exact acceptance/rejection contract. The byte
reference is directly checked against the existing capture columns; its DPP
hardware digest now has per-lane computed checks and configuration diagnostics.
Packing architecture DPP failures now identify their configuration as well.
All 13 affected tests pass on Clang and GCC; all 14 tooling tests pass. Existing
hardware digests and production implementations are unchanged.

### Integer/FP32 conversion reference review

Two more family reviews bring the inventory to 60 reviewed and 94 pending.
Recorded integer-only rounding and saturation references, literal tie/NaN/limit
witnesses, retained hardware digests and the architecture OMOD policy witness.
Strict-exact rejection is explicit; encoded equality in these tests is not a
claim of strict-exact API support. Added configuration/lane context to existing
per-output hardware assertions. All 15 family/integration tests pass with Clang
and GCC, and all 14 tooling tests pass. No production code changed.

### FP64 conversion scope and NaN review

Reviewed two more families (62 reviewed, 92 pending). Recorded integer-significand
rounding, representative midpoint coverage across FP32 exponents, split-word
alias layouts and the separate loose NaN classification policy. Literal numerical
witnesses are not mislabeled as hardware captures. Existing per-input diagnostics
and independent literal cases were adequate; no redundant test loops or fixtures
were added. The regenerated inventory validates all review links and the 14
tooling tests pass. This batch changes documentation only.

### FP16 conversion review

Two reviews bring the explicit queue to 64 reviewed and 90 pending. Recorded
half-encoding/midpoint coverage, half preservation, hardware-anchored overflow
and scaling rules, and the separate upstream-derived RDNA3 raw NaN widening
policy. Strict-exact API rejection is distinguished from these reference checks.
Improved DPP capture failure context while retaining its hardware digest and
per-output checks. All 17 affected/integration tests pass with Clang and GCC;
all 14 tooling tests pass. No production changes or new golden arrays.

### Rounding and fractional-part review

Two reviews bring the queue to 66 reviewed and 88 pending. Distinguished retained
hardware result hashes from literal witnesses and differential DPP references.
Removed a vacuous exact-rounding assertion that checked a local state variable
although the API received a null state pointer; this test does not establish
exception-output correctness. Added exact DPP failure context. All four affected
rounding tests pass on Clang/GCC and all 14 tooling tests pass. FRAC remains
explicitly loose/fallback-only; no new numerical model or production change.

### Exponent and mantissa extraction review

Two more reviews bring the queue to 68 reviewed and 86 pending. Documented the
independent leading-bit exponent model and the different hardware/literal versus
differential mantissa evidence. FP32 exponent hardware corpora now check each
input independently, retaining the final captured digest. Optional-state checks
in DPP integration apply only to loose calls passing the pointer. After fixing
GCC's dangling-else warning around GoogleTest macros with explicit braces, all
19 affected tests pass on rebuilt Clang/GCC binaries. All 14 tooling tests pass.

### Exponent scaling and additional deterministic-input fix

Four more reviews bring the queue to 72 reviewed and 82 pending. Recorded
FP16/32/64 scaling references, literal boundaries, architecture policy witnesses
and the wider-range long-double assumption of the validated x86 builds.
Found an eighth unsequenced RNG site missed by the initial scan: LDEXP passed
two draws as emplace_back arguments. Input and exponent draws are now separate
statements. Follow-up multi-random-call statement inspection found only ordered
declarations and mutually exclusive conditional branches. All ten LDEXP tests
pass on Clang/GCC, and all 14 tooling tests pass. The RNG finding corrects the
initial seven-site inventory; it does not alter the deterministic-input rule.

### Move, relative-move and swap review

Three reviews bring the queue to 75 reviewed and 79 pending. Recorded exact
field/lane references, before-image alias rules, inactive sources and independent
relative-offset/allocation bounds. Half-move hardware capture now has per-output
checks and lane/modifier diagnostics while retaining its digest. Relative-move
baseline-only CPU coverage is stated explicitly rather than generalized to all
selectors. All seven affected Move tests pass on Clang/GCC; all 14 tooling tests
pass. Production code and hardware fixtures are unchanged.

### Hardware registers and constant comparisons

Two reviews bring the queue to 77 reviewed and 77 pending. Recorded independent
per-bit register references, architecture/privilege masks and Boolean predicate
proofs, including optional-state rejection rather than claiming state generation.
Hardware-register captures now check individual outputs with field diagnostics
while retaining their independent hardware digests. All 21 affected tests pass
on Clang/GCC; all 14 tooling tests pass. No production changes or new fixtures.

### Carry, wide MAD and Wave64 integer review

Five reviews bring the queue to 82 reviewed and 72 pending. Carry's computed
reference now checks captured values/masks directly. Carry DPP and wide-MAD
captures retain hardware digests with per-output diagnostics. Documented the
wide-MAD model's shared production Uint128 dependency rather than overstating
reference independence; hardware and literal extended-sign witnesses remain.
Wave64 records cover supported modes, upper-lane masks, aliases and guards with
explicit limits on native capture evidence. All 28 affected/family tests pass
on Clang/GCC, and all 14 tooling tests pass. Production code is unchanged.

### SAD and packed integer MAD review

Three reviews bring the queue to 85 reviewed and 69 pending. Recorded independent
widened/window arithmetic, literal threshold and half-selection witnesses, and
limits of Wave32 loose-only exhaustive sweeps. Compact Wave32 SAD/packed-MAD
literal tests now also select strict exact semantics. SAD's DPP hardware digest
now has per-output diagnostics. All 18 family tests pass with Clang/GCC; all 14
tooling tests pass. Production and captured data are unchanged.

### Floating-comparison integration and exception review

Three reviews bring the queue to 88 reviewed and 66 pending. Hardware exception
pairs now check predicate results independently, in addition to captured flags.
Removed vacuous state assertions on null-pointer calls and improved pair/mode
failure context. Recorded RDNA3 unsupported-reporting behavior, RDNA4 captured
state evidence, and Wave64/CPU coverage limits explicitly. All 12 affected tests
pass with Clang/GCC; all 14 tooling tests pass. Production is unchanged.

### Floating min/max selection review

Four reviews bring the queue to 92 reviewed and 62 pending. Recorded why raw
selection/scaling/quantization checks are intentional despite loose-only API
support, with separate NaN classification versus payload witnesses. Half-median
hardware captures now include per-input checks while retaining their digests and
NaN canonicalization. DPP failure context includes EXEC. All 32 family tests pass
with Clang/GCC; all 14 tooling tests pass. No production or fixture changes.

### Cube and division architecture review

Three reviews bring the queue to 95 reviewed and 59 pending. Cube numeric
reference is checked directly against retained literal captures; both-width
Cartesian digests now have per-input companions. Recorded captured-domain limits,
borrowed division witnesses and the differential nature of DIV_FMAS policy tests.
DIV_FIXUP state assertions now apply only when its pointer is passed. All 28
family tests pass with Clang/GCC; all 14 tooling tests pass. Production unchanged.

### DIV_SCALE and DIV_FMAS review

Two reviews bring the queue to 97 reviewed and 57 pending. Distinguished
hardware numerical/exception digests, literal fused-boundary outputs and borrowed
architecture witnesses from differential integration tests. DIV_FMAS captures now
preserve/check all bits outside the seven captured flags, and result failures
identify the corpus. An attempted ordinary-FMA invalid witness failed because
DIV_FMAS suppresses those flags; the unproven assumption was removed, not used
to change production. All 29 family tests pass on rebuilt Clang/GCC binaries;
all 14 tooling tests pass. DIV_FIXUP numerical review remains pending.

### DIV_FIXUP review

Two reviews bring the queue to 99 reviewed and 55 pending. Both-width Cartesian
captures now check each computed result before the retained hardware digest,
including FP16 half preservation. Exception corpora check all non-flag bits.
Recorded captured-domain limits, long-double assumptions, independent flushed-input
causes and upstream half-ordering witnesses, distinct from differential DPP state
reduction. All 23 family tests pass on Clang/GCC; all 14 tooling tests pass.
Production and hardware fixtures are unchanged.

### TRIG_PREOP review

Two reviews bring the queue to 101 reviewed and 53 pending. Recorded the hardware
sample/digest evidence and inherited table-extraction reference, including width
sampling limits. Removed a vacuous null-output state assertion. All 11 family
tests pass on Clang/GCC. Production and fixtures are unchanged.

Follow-up identified while reviewing MULLIT: general loose multiplication in its
computed-reference tests still uses raw equality. Preserve sentinel, signed-zero,
hardware-digest and untouched-storage checks, but calibrate fuzzy comparisons for
ordinary written numerical results before marking those two files reviewed.

### MULLIT review and fuzzy comparisons

The MULLIT follow-up above is resolved. Two reviews bring the queue to 103
reviewed and 51 pending. General computed-result checks now use a measured
absolute/relative bound; sentinel, signed-zero, hardware digests and storage
preservation remain exact. Calibration and a second deterministic seed found
zero reference differences on Clang/GCC; documentation distinguishes that
observation from the deliberately nonzero allowance for future loose arithmetic.
Added comparator boundary checks and per-lane architecture diagnostics. All 15
family tests and all 14 tooling tests pass; all 1,791 APIs retain active family
references. Production code and hardware fixtures are unchanged. Broader family
review and final full-build validation remain pending.

### FMAC and literal-FMA review

Two reviews bring the queue to 105 reviewed and 49 pending. General loose
numerical comparisons use calibrated FP16/FP32 bounds, while exact references,
preserved storage, signed zeros and finite fused-rounding literals remain raw.
Clang/GCC calibration and alternate seeds measured zero numerical error; the
allowance above zero is documented explicitly. Loose NaN payloads stay
unspecified. FP32 literal-FMA hardware digests now have per-input companions,
and a null-pointer state assertion is no longer treated as reporting evidence.
All 14 affected tests pass on Clang/GCC. Production and fixtures are unchanged.

### Half-FMA and DPP review

Two reviews bring the queue to 107 reviewed and 47 pending. Calibrated fuzzy
checks now cover ordinary loose writes while preserving raw hardware, exact-mode,
boundary, zero-sign and storage assertions. Alternate deterministic seeds also
match the computed reference on both compilers. Exception corpora check every
nonflag bit, and direct host-intrinsic includes replace a transitive dependency.
All 22 affected tests pass on Clang/GCC. Production and fixtures are unchanged.

### Mixed-FMA review

One review brings the queue to 108 reviewed and 46 pending. General loose outputs
now use calibrated numerical bounds, with independent exact checks for inactive
storage and preserved halves. Fixed an alias test that selected exact semantics
but passed a loose comparator selector. Removed vacuous null-output state checks
and added comparator regression cases. All 23 family tests pass on Clang/GCC;
14 tooling tests pass. Existing and alternate random inputs measure zero
reference error on both compilers. Production and hardware fixtures are unchanged.

### Exact FMA and rounding-exception review

Two reviews bring the queue to 110 reviewed and 44 pending. Added sampled
per-input diagnostics alongside full exact-FMA hardware digests, improved
modifier failure context, and removed a vacuous null-output state check.
Rounding-exception captures now verify every nonflag bit remains unchanged.
All six affected tests pass on Clang/GCC. Production and fixtures are unchanged.

### Output-scaling and scalar guest-state review

Two reviews bring the queue to 112 reviewed and 42 pending. Removed a production
helper dependency from captured input construction, clarified the original hash
serialization, and extended GETPC boundary/alias coverage to loose semantics.
Improved immediate failure context. All six affected tests pass on Clang/GCC;
production and hardware fixtures are unchanged.

### FP64 output review

One review brings the queue to 113 reviewed and 41 pending. Cross-half alias
tests now check every nondestination register; hardware transcendental failures
include modifier/lane context. Removed a production decoding dependency and
clarified hash serialization without changing capture data. Both affected tests
pass on Clang/GCC. Production is unchanged.

### FP64 arithmetic review

One review brings the queue to 114 reviewed and 40 pending. Added deterministic
random ADD/MUL/FMA proximity tests with cancellation, both widths and all CPU
selectors. Calibration and alternate seeds pass on Clang/GCC, including a measured
nonzero difference from the wider reference. Existing raw dyadic, selection and
literal boundary checks remain. All 14 family tests pass on both compilers;
production is unchanged. The new reference explicitly requires wider long double.

### FP64 unary review

One review brings the queue to 115 reviewed and 39 pending. Tightened coincident
output alias checks for NaNs and raw mantissa-extraction checks; retained calibrated
fuzzy SFU tests. Moved numerical diagnostic formatting to failures only. All 11
family tests pass on Clang/GCC. Production and capture data are unchanged.

### Trigonometric exception review

One review brings the queue to 116 reviewed and 38 pending. Both-width capture
tests now check every nonflag bit, and DPP failures have route/mask/lane context.
Recorded independent hardware evidence separately from differential integration.
All seven affected tests pass on Clang/GCC. Production and fixtures are unchanged.

### Trigonometric architecture review

Two reviews bring the queue to 118 reviewed and 36 pending. Centralized and
measured FP32 architecture tolerances on Clang/GCC; both compilers report the
same worst error, below the retained bound. Documented half-reference dependence
on the separately validated FP32 API, removed production bitcast dependencies
and vacuous null-output state assertions, and improved numerical diagnostics.
All seven affected tests and 14 tooling tests pass. Production and captures are
unchanged; core trig numerical suites still have separate pending reviews.

### Core trigonometric review

Two reviews bring the queue to 120 reviewed and 34 pending. Recorded the exact
scope of captured and rocjitsu-digest evidence, improved half per-input and DPP
failure diagnostics, and removed production bitcast dependencies. Alternate
random seeds retain tolerance margin on Clang/GCC. All 21 affected tests pass
on both compilers; production and original capture constants are unchanged.

### Half unary and log/exp architecture review

Two reviews bring the queue to 122 reviewed and 32 pending. Tightened inactive
hardware-corpus lane checks to raw whole-word equality and checked all source
words in disjoint Wave64 calls. Numerical helpers now expose failures with lane
and case context; removed a vacuous exact null-state assertion. Recorded the
scope of mathematical, hardware-digest and differential policy references.
All 12 affected tests pass on Clang/GCC, as do 14 tooling tests. No production,
capture or tolerance changes were needed.

### Exact half-unary review

One review brings the queue to 123 reviewed and 31 pending. Added independent
rocjitsu RCP/RSQ/SQRT checkpoints covering all half encodings in both architectures
and widths on all CPU selectors. Clang/GCC independently generate identical
384-byte fixtures. New numerical coverage takes about 40 ms and localizes failures
to 4096-input intervals. Improved host/policy diagnostics and removed vacuous
state checks; loose SFU comparisons use the existing calibrated bound, while
inactive storage stays raw. All eight family tests pass on both compilers, as do
14 tooling tests. Production remains unchanged.

### DPP half-unary review

One review brings the queue to 124 reviewed and 30 pending. Corrected reference
construction to use saved sources, checked all nondestination registers, and
made inactive-lane preservation bitwise before fuzzy/NaN handling. Added lane
context to numerical failures. Existing and alternate LCG inputs pass the fixed
bound on Clang/GCC; all five family tests and 14 tooling tests pass. Production
and hardware witnesses are unchanged.

### DPP FP32 unary review

One review brings the queue to 125 reviewed and 29 pending. Added per-lane
hardware-corpus diagnostics and disjoint source-preservation checks. Simple
rounding/extraction results now use bitwise comparisons; SFUs retain the calibrated
bound, with matching Clang/GCC boundary measurements. All three family tests and
14 tooling tests pass. Production and the original capture digest are unchanged.

### Core FP32 unary review

One review brings the queue to 126 reviewed and 28 pending. Centralized the
suite's distinction between exact simple/categorical results and fuzzy SFUs,
including random calibration/holdout calls. Added source/out-of-wave preservation
checks and removed a vacuous state assertion. Failure-only formatting reduced
the existing EXP/LOG test from about 1.15 seconds to 0.06–0.07 seconds in these
runs, with unchanged inputs. All 14 family tests pass on Clang/GCC, as do 14
tooling tests. Production, tolerances and captured fixtures are unchanged.

### RCP_IFLAG review

One review brings the queue to 127 reviewed and 27 pending. Exact captured
numeric outputs now require bitwise equality; loose comparisons retain their
calibrated tolerance. Both-width null-output tests now cover exact semantics.
Improved DPP failure context and removed null-state no-op assertions. All 17
family tests pass on Clang/GCC, as do 14 tooling tests. Production, capture
fixtures and tolerance values are unchanged.

### Pseudo-scalar review

One review brings the queue to 128 reviewed and 26 pending. Reused independent
half-SFU checkpoints to validate every unmodified preserved-denormal encoding,
while explicitly distinguishing mathematical proximity from exact evidence.
Loose hardware-corpus tests now exercise non-null state preservation. All eight
family tests pass on Clang/GCC, as do 14 tooling tests. Production, tolerances
and fixture contents are unchanged.

### Scalar FMA review

One review brings the queue to 129 reviewed and 25 pending. Loose scalar FMA
outputs now use calibrated fuzzy comparisons while their references retain
independent hardware-digest validation. The computed FP16 oracle matches the
capture, avoiding new fixtures. Tightened error destination-preservation checks.
All six family tests and numerical measurements pass on Clang/GCC, as do 14
tooling tests. Production and captured values remain unchanged.

### Scalar floating arithmetic review

One review brings the queue to 130 reviewed and 24 pending. Computed arithmetic
references match both hardware datasets, allowing loose ADD/SUB/MUL proximity
checks with per-input diagnostics. Selection, categorical and storage results
remain exact. All six family tests and measurements pass on Clang/GCC, as do 14
tooling tests. Production and existing hardware captures are unchanged.

### Arithmetic exception integration review

One review brings the queue to 131 reviewed and 23 pending. Added independent
nonflag-bit preservation checks, all-register storage checks and active exact
null-reporting equivalence. Documented the single-lane differential reference
and separate hardware evidence. All three family tests pass on Clang/GCC, as do
14 tooling tests. Production remains unchanged.

### Dual-slot review

One review brings the queue to 132 reviewed and 22 pending. Extended RDNA3 dot
captures to all CPU selectors, added source-preservation checks, allowed calibrated
loose adapter comparisons, and improved integer failure localization. Recorded
independent versus differential evidence and caller-staged pair behavior. All
nine family tests pass on Clang/GCC, as do 14 tooling tests. Production is unchanged.

### Packed binary review

One review brings the queue to 133 reviewed and 21 pending. Loose packed ADD/MUL
now compare each half numerically with the existing half arithmetic bound;
selection, literals and storage stay exact. Added missing Wave64 source checks
and per-half diagnostics. Existing/alternate seeds and all 11 family tests pass
on Clang/GCC, along with 14 tooling tests. Production remains unchanged.


### Packed FMA/FMAC review

Reviewed all 14 packed FMA tests across RDNA3/RDNA4 and both widths. Loose finite
arithmetic now uses the shared calibrated half-FMA comparison while exact
integer-grid references, inherited literal witnesses and storage stay bitwise.
Added missing source-preservation checks, checked all unrelated exception bits,
expanded architecture guest-policy coverage to every available CPU selector, and
removed vacuous state assertions when the reporting pointer is null.
Both compilers pass the 14 tests and original/alternate-seed calibration without
loosening bounds; each seed checks about 6.96 million finite half results with
zero observed numerical error. The explicit inventory reaches 134/154 reviewed
family files; 20 remain, followed by final broad verification. Production is
unchanged.


### DOT2 architecture integration

Reviewed the RDNA3 F32-output and both-architecture half-output DOT2 families.
Added deterministic fractional loose tests with an independent double reference,
random modifiers, aliases and raw storage checks on every CPU selector. Original
and holdout seeds each cover 114,432 finite outputs and pass the measured
(2e-6, 2e-7) bound on Clang and GCC. Explicitly separated borrowed-model differential
modifier coverage from independent gfx1100 captures and verified the model against
those captures directly. Half-output literal policy tests now check sources and
out-of-wave storage. The 16 tests pass on both compilers. The review inventory
reaches 136/154 family files; 18 and final broad verification remain.


### Half-output DOT2 numerical review

Added independent deterministic fractional references for both architectures and
wave widths, using measured FP16/BF16 tolerances. Preserved exact integer routing,
literal policy witnesses and captured DPP digests; classified existing random
scalar/SIMD comparisons as differential and gave their finite results fuzzy
comparisons. Every unselected destination half and inactive/source word remains
raw, with missing source checks added to the ordinary alias sweeps.
All 15 tests and both calibration seeds pass on Clang and GCC. Inventory:
137/154 family files reviewed, 17 remain before final broad verification.


### FP8/BF8 DOT4 review

Retained existing independently calibrated exhaustive-pair and mixed-byte
numerical checks, including the fixed holdout. Added missing alias-source and
out-of-wave storage checks, made source snapshots independent of post-call
operands, and removed successful per-lane diagnostic formatting overhead.
All 11 tests pass on Clang and GCC with unchanged tolerance bounds and input
corpora. The explicit inventory now has 138/154 family-file reviews, with 16
remaining before final broad verification.


### Interpolation reference and tolerance review

Completed FP32 and FP16-input interpolation reviews. Added direct independent
integer-reference validation against the retained FP16-input hardware digests.
General loose arithmetic now uses output-format atol/rtol while selected-field
NaN handling, unselected halves and inactive storage remain separate. Random
references now have source-preservation checks in both families. Original and
alternate seeds agree exactly with the computed references on Clang and GCC;
all 21 tests pass. Explicit inventory: 140/154 reviewed family files, with 14
remaining before final broad verification.


### Half binary arithmetic and DPP review

Separated general loose finite arithmetic from raw min/max and categorical
comparisons, with one small shared test comparison rule and a focused regression
protecting storage/categories. Fixed the DPP exhaustive-modifier test's
canonicalization of unwritten NaNs and protected references with input snapshots.
Added missing Wave64/source/out-of-wave storage checks. All 20 affected tests
pass on Clang and GCC; original and alternate-seed calibration has zero observed
arithmetic error without widening bounds. The inventory reaches 142/154 reviewed
family files; 12 and final broad verification remain.


### RDNA4 DOT2 review

Preserved exact hardware-fixture transformations and independent integer routing.
Extended the existing loose fractional corpus to RDNA4, added ordinary alias
source checks, and made the captured zero-exception tests expose every possible
erroneous arithmetic flag bit. General loose scalar/SIMD comparisons are fuzzy;
inactive storage is raw. All 24 DOT2/architecture tests pass on Clang and GCC,
including original/alternate-seed calibration with unchanged bounds.
The explicit inventory reaches 143/154 family files; 11 and final broad
verification remain.


### FP32 FMA and DPP arithmetic review

Added independent long-double arbitrary-word FMA testing for both architectures
and widths, including aliases, modifiers and full/mixed masks. General loose
Wave64/DPP arithmetic now uses calibrated comparisons; literal hardware,
selection and categorical behavior remains separate. Added source checks and
removed state assertions on variables never passed to DX9 APIs. All 27 affected
tests (including C linkage) pass on Clang and GCC; original/alternate FMA seeds
measure zero error without relaxing bounds. Inventory: 145/154 family files
reviewed, nine and final broad verification remain.


### FP32 binary review

Separated finite loose arithmetic from exact ADD/MUL and raw min/max policy
checks. Added deterministic arbitrary-word binary64 references across both
architectures and widths, with masks and aliases, and strengthened source and
out-of-wave storage checks throughout the integration tests. Borrowed exact
policy vectors remain bitwise, including host-rounding/denormal-control tests.
All 14 tests and both calibration seeds pass on Clang and GCC with zero
observed arithmetic error. Inventory: 146/154 family files reviewed; the eight
remaining matrix families and final broad verification are still pending.


### Sparse integer WMMA review

Verified independent both-width hardware digests, native Wave64 selector coverage,
deterministic input draw order, staged saturation witnesses and differential
alias/duplicate-destination checks. Added loose semantics to native Wave64
capture checks, all non-output storage preservation, more precise digest
diagnostics and complete Wave32 invalid-bit/error-preservation checks.
All ten tests pass on Clang and GCC. Inventory: 147/154 families reviewed,
seven matrix families and final broad verification remain.


### Packed WMMA reference review

Retained borrowed hardware captures and simple exactly representable computed
modifier references, consistent with the clarified oracle policy. Expanded
matrix captures to all CPU selectors, added non-output storage preservation,
and exercised computed modifier cases in both semantics on all CPU selectors.
Separated validation into a full-storage preservation test covering all unknown
modifier bits, invalid global flags and reserved strict semantics. Six tests
pass on Clang and GCC. Inventory: 148/154 families reviewed; six matrix families,
including the companion packed-loose tolerance review, and final broad
verification remain pending.


### Sparse FP8/BF8 WMMA review

Added independent bounded fractional matrix references for both widths, all
four type pairs, all index keys and every available CPU selector. Primary and
holdout seeds each provide 1,572,864 finite fuzzy comparisons per compiler;
maximum absolute errors are 2.70e-6 and 3.28e-6 within the unchanged WMMA bound.
Retained exact small-integer hardware digests and exhaustive encoding/scaling
checks with their restricted oracle domains documented. Expanded capture source
preservation and both-width validation/error checks. All ten tests pass on
Clang and GCC. Inventory: 149/154 families reviewed; five matrix families and
final broad verification remain pending.


### Sparse FP16/BF16 WMMA review

Added independent fractional references across output formats, both widths,
all keys/negation combinations and all available CPU selectors. Measured FP32,
FP16 and BF16 bounds with primary and holdout seeds (1,572,864 comparisons per
seed/compiler), retaining exact small-integer captures and literal midpoint,
overflow and pair-position witnesses. Added capture source preservation and
full-storage invalid-request checks. All fourteen tests pass on Clang and GCC.
Inventory: 150/154 families reviewed; four matrix families and final broad
verification remain pending.


### Subbyte WMMA review

Reviewed both-width integer staged-clamp evidence, offline mathematical goldens,
independent encoding checks and native Wave64 floating captures. Added fractional
FP8/BF8 matrix references across widths, formats, C modifiers and aliases, with
524,288 comparisons per primary/holdout seed on each compiler. Maximum absolute
error is 2.38e-6 / 1.91e-6 within the unchanged WMMA tolerance. Expanded source,
guard and floating validation preservation checks; removed production bit-cast
dependencies. All fifteen tests pass on Clang and GCC. Inventory: 151/154
families reviewed; three dense/replicated WMMA families and final broad
verification remain pending.


### Dense and replicated FP32-output WMMA reviews

Added independent bounded fractional references across RDNA3/RDNA4, both widths,
formats, all modifiers, aliases and CPU selectors. RDNA3 replicas deliberately
differ. Original/holdout seeds each provide 524,288 comparisons per compiler;
maximum absolute errors are 5.59e-6 and 6.43e-6 within the existing WMMA bound.
Preserved borrowed hardware exact fixtures, integer goldens and category/fallback
witnesses; changed general finite differential checks to fuzzy comparisons and
strengthened matrix source/error preservation. All 27 tests and calibration
runs pass on Clang and GCC. Inventory: 153/154 families reviewed; the packed/
replicated exact family and final broad verification remain pending.
