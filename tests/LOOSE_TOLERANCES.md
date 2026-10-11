# Loose floating comparison policy and measurements

`fuzzy.h` implements the NumPy-style contract already used by GoC WMMA and
hrx-system/IREE: `abs(actual - expected) <= atol + rtol * abs(expected)`.
The relative term uses the reference operand. NaNs match NaNs; same-sign
infinities and either signed zero match. Other nonfinite pairs fail. Tests that
require payload, zero-sign, masks, flags, or preserved bits check those separately.
Failure diagnostics include both operands, the absolute error and allowed bound.

`loose_tolerances.h` centralizes bounds for the migrated families. The initial
migration preserved existing bounds; subsequent calibration is documented below.
Migration is still incomplete, and sampled corpora do not establish universal
error limits.

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

## Unary calibration and holdout

The eleven FP32 unary operations now use `(atol=0, rtol=2.5e-7)` for numerical
comparisons, replacing GoogleTest's implicit four-ULP comparison. Existing
bit-pattern checks for exact cases, signed zeros, classifications and untouched
lanes remain intact. The reference evaluates the mathematical operation in
binary64, then applies the specified binary32 rounding, flushing and modifiers;
this is a loose reference, not a claim of exact hardware equivalence for SFUs.

`Unary.DeterministicNumericalCalibration` uses MT19937 seed 18731;
`Unary.DeterministicNumericalHoldout` uses seed 61357. Each generates 64 batches
of 32 words, stratified between raw binary32 bit patterns, multiples of 1/256
in [-128,128], and encodings within 4096 words of 1. Every operation is checked
with all 32 ABS/NEG/OMOD/CLAMP combinations and all four available CPU selectors.
Inputs are generated outside the operation/modifier/CPU loops, so comparisons
share the same corpus. Neither test tunes its tolerance at runtime.

| Corpus | Finite comparisons per compiler | Max absolute error | Max relative error | Fraction of bound |
| --- | ---: | ---: | ---: | ---: |
| Calibration | 2,674,680 | 2.028241e31 | 1.840310e-7 | 0.736124 |
| Fresh holdout | 2,674,128 | 1.267651e30 | 1.769156e-7 | 0.707663 |

Clang 21.1.8 and GCC 15.2.0 produced the same statistics on the available host.
Large absolute errors occur at large exponents; the relative bound is decisive.
An exploratory second stream (seed 94217) reached relative error 1.939394e-7.
The 2.5e-7 bound was fixed after those exploratory measurements, then checked
against the fresh seed 61357. Existing unary cases reached 1.192093e-7. These
observations support a modest margin, not a proof of a universal approximation
bound or coverage of other host libm implementations/FP environments.

The RDNA3/RDNA4 trig architecture test now uses the same shared comparator with
its existing absolute bounds (8e-7 for normals, three minimum subnormals for
subnormal expectations). Its 175,980 measured finite fuzzy calls reached
4.768372e-7 absolute error; this corpus did not exercise the subnormal branch.
That branch's bound is preserved, not claimed calibrated by these measurements.

All 17 affected tests pass on Clang and GCC. Measurement snapshots, scripts and
logs are in `coverage-work/testing-unary-calibration/` in each build directory;
the ordinary test logs are `coverage-work/testing-unary.log`.
