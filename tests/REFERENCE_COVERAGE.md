# Instruction-test reference evidence

The machine-readable review record is [reference_reviews.json](reference_reviews.json).
Each entry identifies actual semantic assertions, oracle evidence, its applicable
domain, and diagnostic limits. A computed exact oracle is acceptable when its
integer/Boolean proof or established upstream equivalence is recorded. Loose
mathematical references and scalar/SIMD differential checks are different kinds
of evidence and do not establish hardware bit-exactness on their own.

Regenerate the compiled API-to-test inventory after building:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/testing_audit.py ../goc-build \
  --output /tmp/goc-reference-inventory.json
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools -p '*_test.py'
```

The inventory covers every declaration in `goc_instructions.h`. It reads exact
object paths from `compile_commands.json`, rejects missing active test objects,
and ignores stale objects. Generic API/flag/exception/empty-mask tests cannot
satisfy the family-reference requirement. It verifies that each review points
to existing test cases and evidence files, then reports unreviewed family files.
Those checks detect inventory mistakes; they cannot prove oracle equivalence or
that a test exercises every referenced API in every semantics/mode.

Current inventory: 1,791 instructions, 163 active test objects, no missing
non-generic family references. Fifty-three explicit family-file reviews exist and 101
remain. Reviewed-file membership must not be interpreted as complete per-API coverage. Consult the record's
scope, especially wave width, semantics and optional state.

Initial reviews:

| Test file | Oracle and applicable scope |
| --- | --- |
| `boolean_test.cpp` | Per-bit truth-table proof; nine Wave32 operations, half selectors and preservation. Exact integer comparison is appropriate in loose semantics too. |
| `integer_add_test.cpp` | Bounded widened integer arithmetic, saturation and modulo narrowing; six Wave32 operations, supplemented by literal ordering/overflow cases. |
| `scalar_integer_test.cpp` | Hardware result/SCC digests plus simple computed immediate references and literal wrap-order anchors. Per-row digests already localize failures to 4,096 inputs. |
| `float_compare_test.cpp` | Hardware predicate masks plus independent integer ordering/NaN/zero/flush model; 84 RDNA4 Wave32 comparison entry points. Predicate masks require exact comparisons, not fuzzy FP comparisons. |
| `unary_exact_test.cpp` | Hardware dense/modifier digests for five borrowed models, validated upstream checkpoints and literal architecture anchors. Differential DPP tests establish routing, not independent numerical equivalence. |

This inventory is an explicit work queue, not a completeness claim. Remaining
work includes reviewing the other family files, identifying which bitwise checks
are intentional architectural checks versus overly restrictive loose numerical
checks, and assessing whether each digest has useful companion diagnostics.

## Integer, Boolean and routing review

Twenty additional records cover bit counts/fields, multiplication, min/max,
integer ternaries/MAD/dot, selected and packed half-integers, shifts, Wave64
Boolean/bitwise operations, integer comparisons, conditional masks, raw FP
classification, lane transfers, DS routing and permlane routing.

The computed models use bounded widened arithmetic, per-bit Boolean definitions,
explicit sign extension, or direct lane-index rules. Literal anchors cover
stage ordering, ignored flags, width reduction, saturation and preserved halves.
Hardware fixtures remain for empirical details such as DOT4 clamp behavior,
PERM selectors, physical-lane MBCNT behavior and extended DS routing. These
references justify exact result checks even when the semantics selector is loose;
there is no approximate floating arithmetic to compare fuzzily in these domains.

The diagnostic review found useful per-output companions to the broad hardware
digests. `Permlane.HardwareCorpus` now adds CPU/EXEC/opcode/mode/selector-pattern
context and the failing lane, preserving its independent hardware digest.
No new golden data or production implementation changes were needed in this
batch. The remaining file queue is still explicit in the generated inventory.

## Scalar state and DPP integration review

Twenty-two additional records cover scalar bit/compare/convert/field/pack/round,
EXEC manipulation, moves/relative moves/spellings, ten integer DPP families, and
the DPP8/DPP16 routing suites. Captured scalar result/SCC/exception digests are
kept distinct from differential alias checks. The scalar conversion family is
explicitly recorded as loose/fallback-only rather than inferring strict-exact
support from agreement with its hardware fixtures.

The DPP integer captures already compare every result with the independent
integer model composed with the routing reference, so their final digests are
not the only diagnostic. DPP8/DPP16 use small integer or 1/8-step arithmetic:
products, sums and power-of-two output scaling are exactly representable in the
covered range. This is a bounded proof for these routing tests, not permission
to assume general floating arithmetic is bit-exact. Source comments record that
reason for exact assertions. Hardware-test failure traces now include routing
controls and EXEC as well as the lane.

## Integer and byte conversion references

Four more reviews cover integer half conversion, byte/nibble conversion, its
architecture policies, and FP16 field packing. Exact integer and bounded dyadic
references are appropriate here: there is no rounding ambiguity in extending an
integer half or scaling a byte by these powers of two. The architecture review
records the separate RDNA3 OMOD policy evidence.

Integer/byte DPP hardware captures now retain their independent hardware totals
while checking individual outputs against these simple references. FP16 packing
likewise checks every captured input against the field model in both wave widths;
its hardware totals still anchor NaN quieting and modifier ordering. Failures now
identify the input or lane and configuration without adding golden arrays.

## FP8 conversion evidence

The widening review separates the lossless finite decode proof from hardware
validation of special encodings. All 256 raw codes are hardware-anchored,
including NaN sign; per-code checks now localize capture failures in both widths.
The narrowing review retains raw captures for stochastic rounding and special
values. Its computed reference is now checked directly against those captured
outputs, independently of GoC. This demonstrates equivalence on the captured
domain; the broader random/model tests do not prove arbitrary-input hardware
equivalence. Existing goldens remain in place.
