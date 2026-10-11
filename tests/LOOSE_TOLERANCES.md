# Loose floating comparison policy and measurements

`fuzzy.h` implements the NumPy-style contract already used by GoC WMMA and
hrx-system/IREE: `abs(actual - expected) <= atol + rtol * abs(expected)`.
The relative term uses the reference operand. NaNs match NaNs; same-sign
infinities and either signed zero match. Other nonfinite pairs fail. Tests that
require payload, zero-sign, masks, flags, or preserved bits check those separately.
Failure diagnostics include both operands, the absolute error and allowed bound.

`loose_tolerances.h` centralizes the existing bounds for the migrated families.
This first migration does not widen bounds, claim all loose comparisons have
been migrated, or claim the current corpora establish universal error limits.

## Initial measurement, following source commit `16a87aa`

Release Clang 21.1.8 on the Ryzen 9 7950X3D, all available CPU selectors
(baseline, v3, v4, Zen4). The production library is unchanged from `127ecf1`.
The 70 affected tests also pass with GCC 15.2.0; the measured maxima below are
from Clang, not an assertion that GCC produces identical intermediate values.

An out-of-tree instrumented copy of the test comparator records finite pairs
before comparison. It runs the complete WMMA, interpolation and FP32/FP16 trig
test binaries. Counts include repeated inputs across CPU levels, modifiers and
aliases; they are not counts of unique random inputs. No instrumentation or
adaptive tolerance selection is present in ordinary tests.

| Comparison domain | `(atol, rtol)` | Finite comparisons | Maximum absolute error | Maximum fraction of allowed bound |
| --- | --- | ---: | ---: | ---: |
| Captured FP32-result WMMA, Wave32/64 | `(1e-5, 1e-5)` | 92,160 | 0.03125 | 0.008762 |
| FP32 interpolation vs computed reference | `(1e-44, 2e-7)` | 1,004,608 | 0 | 0 |
| FP32 trig, unscaled bound | `(3e-7, 0)` | 889,048 | 1.251139e-7 | 0.417047 |
| FP32 trig, half-scaled bound | `(1.5e-7, 0)` | 366,488 | 5.960465e-8 | 0.397365 |
| FP32 trig, double-scaled bound | `(6e-7, 0)` | 366,704 | 2.384186e-7 | 0.397365 |
| FP32 trig, quadruple-scaled bound | `(1.2e-6, 0)` | 366,728 | 4.768372e-7 | 0.397365 |
| FP16 trig against mathematical reference | `(0.0005, 0)` | 2,031,616 | 0.000244111 | 0.488221 |

FP32 trig includes the deterministic full-range/interval-boundary corpus and
comparisons to the existing exact model across modifiers. FP16 trig uses the
exhaustive half-input mathematical checks in both widths. Interpolation uses its
existing special-value and deterministic random corpora. WMMA's measured fuzzy
calls use the existing captured fixtures; its broader scalar/SIMD differential
checks are not silently counted as independent hardware comparisons.

The trig bounds have observed margin in these domains. WMMA's generous observed
margin is **not** enough evidence to tighten it across arbitrary reductions and
cancellation. Interpolation's zero error does **not** establish that every sound
loose implementation must be bit-identical. Keep these bounds pending broader
calibration and deterministic holdout evaluation. Unary, FP8 dot, remaining loose
comparators, and reference provenance remain tracked in `TESTING_AUDIT.md`.

Local measurement source, instrumented snapshots, full test logs and CSV results:
`goc-build/coverage-work/testing-fuzzy-calibration/`. The two normal build trees
record affected-suite results in `coverage-work/testing-fuzzy-tests.log`.
