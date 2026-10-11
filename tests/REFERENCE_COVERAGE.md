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
non-generic family references. Five explicit family-file reviews exist and 149
remain. 144 API symbols are linked to at least one reviewed file; that number
must not be interpreted as 144 completely audited APIs. Consult the record's
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
