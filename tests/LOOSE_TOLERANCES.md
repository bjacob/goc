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

## FP8 dot and FP64 calibration

FP8/BF8 dot has two deliberately different domains. Exhaustive repeated-byte
pairs span all encodings with C = +/-1.25; they use `(0, 2.5e-7)`. New mixed-byte
corpora bound finite factors by 4 and use `(4e-6, 2e-7)`. C alternates between
multiples of 1/256 in [-4,4] and the negated rounded dot product, creating strong
cancellation. All four format combinations, C modifiers, CPU selectors and both
wave widths are covered. References decode independently and accumulate in
binary64 before rounding once. Integer-only alias/routing tests retain exact
checks because their small products and sums are exactly representable.

The mixed tests use MT19937 seed 39217 for calibration and 71993 for holdout,
64 batches of 64 lanes per format combination. These bounds are specific to
those domains; neither is advertised as a universal bound for arbitrary
mixed-exponent reductions with cancellation.

FP64 sqrt/rcp/rsq numerical tests use `(4 * minimum_subnormal, 5e-16)` against the
existing long-double mathematical reference. The new calibration/holdout tests
use MT19937 seeds 33457/81541, 32 batches of 64 lanes, all 32 modifiers, CPU levels
and both wave widths. Separately sequenced high/low draws form binary64 words;
strata cover raw words, encodings close to 1, and subnormals. Existing exact
rounding, alias and special-value assertions remain. The reference precision
is host-dependent; these measurements use x86 extended-precision long double,
not a claim of hardware-exact SFU behavior or identical libm on every host.

FP64 comparisons to the captured hardware SFU outputs keep their previous
relative tolerance `2^-23`, now expressed directly as `(0, 2^-23)` instead of
subtracting 1 from the actual/reference quotient. Sign and classification
checks remain separate. Hardware approximations need not have binary64 accuracy.

| Domain | Finite pairs | Max absolute error | Max relative error | Fraction of bound |
| --- | ---: | ---: | ---: | ---: |
| FP8 repeated pairs | 8,064,128 | 2 | 1.907387e-7 | 0.762955 |
| FP8 mixed calibration | 393,216 | 1.907349e-6 | 15 | 0.254315 |
| FP8 mixed holdout | 393,216 | 1.907349e-6 | 15 | 0.256883 |
| FP64 mathematical existing + calibration | 3,324,752 | 2.381705e139 | 2.220447e-16 | 0.444090 |
| FP64 mathematical holdout | 952,384 | 1.995841e292 | 2.220447e-16 | 0.444090 |
| FP64 hardware SFU | 15,792 | 3.672847e300 | 4.086178e-8 | 0.342774 |

Clang and GCC produced identical statistics on this host. The large relative
error for FP8 cancellation demonstrates why the absolute term matters. The
large FP64 absolute errors are at large exponents, where the relative term
controls the comparison. All 24 FP8/FP64 tests pass with both compilers.
Measurement scripts/snapshots/logs are in each build tree's
`coverage-work/testing-fp8-fp64-calibration/`.

The three DPP unary tests now also use the calibrated FP32 unary comparator and
pass under both compilers. No `EXPECT/ASSERT_FLOAT_EQ`, `DOUBLE_EQ` or `NEAR`
calls remain in test sources. That textual fact does not establish that every
loose numerical comparison has been migrated: half-format ULP checks and the
meaning of existing bitwise/differential comparisons remain under review.
