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
