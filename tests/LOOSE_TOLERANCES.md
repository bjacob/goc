# Loose floating comparison policy and measurements

`fuzzy_compare.h` implements the NumPy-style contract already used by GoC WMMA and
hrx-system/IREE: `abs(actual - expected) <= atol + rtol * abs(expected)`.
The relative term uses the reference operand. NaNs match NaNs; same-sign
infinities and either signed zero match. Other nonfinite pairs fail. Tests that
require payload, zero-sign, masks, flags, or preserved bits check those separately.
`fuzzy.h` adds GoogleTest failure diagnostics with both operands, the absolute
error and allowed bound. The numerical helper has no GoogleTest dependency so
benchmark checks and unit tests can share the same rule.

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

## Half-format and pseudo-scalar comparisons

Half SFU comparisons to rounded half references use `(2^-24, 1.1e-3)`.
These replace adjacent-encoding checks with an explicit numerical contract:
the absolute term covers a minimum-subnormal step, and the relative term has
margin above the observed adjacent-normal differences. This is not algebraically
identical to a one-ULP test. Sign checks, exact zero/infinity cases, preserved
halves, NaN handling and hardware bitwise/digest comparisons remain separate.
The half trig comparison to an unrounded mathematical reference keeps its
previous `(0.0005, 0)` bound.

Clang measurements across the existing corpora:

| Family/domain | Finite pairs | Max absolute error | Max relative error | Fraction of bound |
| --- | ---: | ---: | ---: | ---: |
| HalfUnary | 16,529,844 | 0.00390625 | 0.000975610 | 0.886906 |
| DppHalfUnary | 6,864,652 | 0 | 0 | 0 |
| HalfLogExpArchitecture | 527,064 | 0 | 0 | 0 |
| HalfTrig vs rounded reference | 4,634,704 | 0.000244141 | 0.000752446 | 0.683928 |
| HalfTrigArchitecture loose comparisons | 2,939,840 | 0 | 0 | 0 |
| HalfUnaryExact loose-policy comparisons | 27,427,072 | 0 | 0 | 0 |
| PseudoScalar half comparisons | 1,125,024 | 0 | 0 | 0 |
| PseudoScalar FP32 comparisons | 129,536 | 1.980705e28 | 1.188431e-7 | 0.475373 |
| RcpIflag comparisons | 14,958,448 | 1.014121e31 | 1.188431e-7 | 0.475373 |

The half unary/trig suites enumerate every input encoding, with representative
modifier/FP policies; other tests cross selected inputs with the broader modifier,
DPP, alias and policy matrices. There is no unseen input encoding to reserve as
a holdout for exhaustive sweeps. This is not exhaustive over every encoding ×
policy × alias combination. Pseudo-scalar and reciprocal checks use their existing
hardware/computed corpora and the calibrated FP32 unary bound `(0, 2.5e-7)`.
`PseudoScalar`'s exact-half mathematical proximity check is supplemental; its
hardware anchors and borrowed-model digests establish the exact result contract.
A zero error in a computed proximity corpus alone is not an equivalence proof.

These bounds pass the full 1,201-test Clang/GCC suites, including benchmark smoke
checks; the final two migrated test files also pass all 11 targeted Clang tests.
Measurements are in `goc-build/coverage-work/testing-half-calibration/`.
The numeric comparator is shared with the benchmark's reciprocal correctness
checks without introducing a GoogleTest dependency there. Small wrapper helpers
retain their instruction-specific classification/preservation rules.

## MULLIT lighting arithmetic

Ordinary computed-result comparisons use `(2^-149, 1.2e-7)`. Measurements against
our independently expressed, rounded FP32 lighting reference found zero error in
2,969,024 finite nonsentinel pairs from `Mullit`, and 1,687,776 from
`MullitArchitecture` (seed 58210), on both Clang and GCC. Replacing the architecture
random seed with 91367 gave 1,687,244 finite pairs, again zero error on both.
The latter includes unchanged deterministic Cartesian cases; only the random
policy inputs are a holdout. Both architectures, wave widths, CPU selectors,
all 512 modifiers, guest policies and sampled DPP/alias/mask combinations are
represented. This is not exhaustive over their full Cartesian product.

Zero observed error does not justify requiring bitwise identity for future loose
implementations. The chosen margin allows roughly one FP32 encoding step for
normal results, plus one minimum-subnormal step; it is an explicit allowance
above the measured zero, not an estimate of an observed nonzero error. Tests pin
acceptance/rejection at representative boundaries. Signed zero, the lighting
sentinel, untouched storage and literal policy witnesses remain raw checks.
Hardware digests remain separate exact regression anchors. These measurements
establish reference proximity on the tested domains, not hardware equivalence
for arbitrary input bits. Reproduction script and instrumented snapshots/logs:
`goc-build/coverage-work/testing-mullit-calibration/run.py`.

## FMAC and literal FMA

Loose numerical checks in `fmac_test.cpp` and `literal_fma_test.cpp` use
`(2^-24, 1e-3)` for FP16 and `(2^-149, 1.2e-7)` for FP32. Both allow roughly one
normal encoding step plus one minimum-subnormal step. The measured error against
the rounded computed references was zero in every row below on Clang and GCC;
the nonzero margin is deliberate flexibility for loose arithmetic, not a measured
hardware error. Zero results retain sign checks; preserved halves/storage and
finite fused-rounding witnesses remain exact. Loose NaNs require classification,
while exact semantics retain payload/priority checks.

| Family | Calibration FP32 pairs | Calibration FP16 pairs | Alternate-seed FP32 pairs | Alternate-seed FP16 pairs |
| --- | ---: | ---: | ---: | ---: |
| FMAC | 250,328 | 1,126,332 | 240,536 | 1,140,604 |
| Literal FMA | 1,154,706 | 14,803,752 | 1,154,960 | 14,807,208 |

Counts include finite nonzero loose-reference comparisons only. FMAC covers all
modifiers, masks/aliases and every half A encoding with sampled B/C. Literal FMA
covers every half literal with random operands, all selectors, overflow policy,
FP32 random triples and architecture guest-mode cases. Alternate inputs replace
FMAC seeds 198/429/45239 with 8791/9817/89317 and offset the encoding-sweep filler
seed by 9371; literal seeds 431/9341 become 71059/61169. Fixed edge cases remain
unchanged and are not independent holdout data. Reproduction script, instrumented
snapshots and logs: `goc-build/coverage-work/testing-fma-calibration/run.py`.
These observations do not prove hardware equivalence outside the reference's
inherited rules and captured domains.

The same half-FMA bound was separately measured in the ordinary and DPP
half-FMA suites. Both Clang and GCC found zero error for all finite, nonzero
loose comparisons:

| Family | Existing corpus pairs | Alternate-seed pairs |
| --- | ---: | ---: |
| HalfFma | 2,376,872 | 2,378,480 |
| DppHalfFma | 10,942,172 | 11,018,060 |

The alternate HalfFma seed is 61387 instead of 84712. DPP uses LCG seeds 7919
instead of 231, and `65537 + variant` instead of `997 + variant`. Literal
boundary/hardware witnesses and exact semantics remain unchanged; repeated fixed
cases do not count as independent holdout inputs. Every half A encoding is
covered, with sampled B/C, policies and routing combinations. Scripts, snapshots
and logs: `goc-build/coverage-work/testing-half-fma-calibration/run.py`.

## Mixed FMA

Mixed FMA uses the same explicit FMA bounds, separately measured against its
integer-grid reference. Clang and GCC both observed zero error:

| Input seed | FP32 finite nonzero loose pairs | FP16 finite nonzero loose pairs |
| --- | ---: | ---: |
| 4671 (existing) | 4,878,668 | 8,771,008 |
| 73939 (alternate random inputs) | 4,878,628 | 8,770,464 |

Fixed modifier, routing, encoding and midpoint sweeps are repeated in both runs;
only the pseudorandom triples change. These are domain measurements, not a proof
that arbitrary hardware results equal the computed reference. Hardware digests
and literal rounding witnesses remain exact checks. Unwritten storage and the
unselected half are always bitwise, including in loose semantics. Dedicated
comparator tests inject differences deliberately and are excluded from numerical
calibration. Script and logs:
`goc-build/coverage-work/testing-mixed-fma-calibration/run.py`.

## FP64 ADD/MUL/FMA

`Fp64.DeterministicLooseArithmetic` uses `(2^-1074, 2.5e-16)` against long-double
ADD/MUL/fused-FMA, rounded to binary64 before the independently expressed OMOD.
It tests 32 batches of 64 deterministic inputs, 16 scale/source-modifier settings,
three operations, both wave widths and all CPU selectors. Random finite inputs
span biased exponents 600 through 1399; products stay in binary64 range. Every
fourth lane cancels a rounded product and the next lane uses neighboring opposite
operands. Literal tests separately cover overflow, subnormals and zero signs.
The reference requires wider long double; this test explicitly skips otherwise.

Clang and GCC each measured 589,824 finite pairs per seed. Seed 63017 matched
exactly; alternate seed 88721 had maximum absolute error `5.9347298410998742e-67`
and maximum relative error `1.5295430857584179e-16`. The chosen relative bound
allows about one binary64 encoding step with margin; the absolute term supplies
a minimum-subnormal floor. Wider host arithmetic is a proximity reference, not
a hardware bit-exact oracle. Script/snapshots/logs:
`goc-build/coverage-work/testing-fp64-arithmetic-calibration/run.py`.

## Trigonometric architecture integration

The captured FP32 architecture corpus retains `(8e-7, 0)` for normal results
and `(3 * 2^-149, 0)` for subnormal results, now named centrally. Clang and GCC
measure 1,023,392 finite normal pairs, maximum absolute error
`4.76837158203125e-7` (about 60% of the bound), and 2,956 subnormal pairs with
zero error. Both architectures, wave widths, CPU selectors, eight guest policies,
32 modifiers and eight routing choices participate. Exact zeros and NaN payloads
remain raw checks. This calibration describes the fixed captured boundary corpus;
it has no independent random holdout and is not an arbitrary-input error bound.
Broader mathematical comparisons remain in the core trig tests.

The half architecture integration compares 2,939,840 finite loose pairs with zero
error on both compilers, using the existing half SFU tolerance. Its computed
reference depends on GoC's separately validated FP32 API and independent half
narrowing, so this measurement demonstrates integration consistency rather than
independent hardware equivalence. Script, source snapshots and logs:
`goc-build/coverage-work/testing-trig-architecture-calibration/run.py`.

## Core trigonometric holdout

Clang and GCC give the same measurements for the core trig tests:

| Domain | Corpus | Finite pairs | Maximum absolute error | Maximum fraction of bound |
| --- | --- | ---: | ---: | ---: |
| FP32 mathematical/scalar comparisons, scaled bounds | Existing | 1,988,968 | 4.76837158203125e-7 | 0.417047 |
| FP32 mathematical/scalar comparisons, scaled bounds | Alternate seeds | 1,988,488 | 4.76837158203125e-7 | 0.416732 |
| FP16 mathematical, both widths | Exhaustive encodings | 2,031,616 | 0.00024411025959947263 | 0.488221 |
| FP16 rounded reference | Existing | 4,634,704 | 0.000244140625 | 0.683928 |
| FP16 rounded reference | Alternate seeds | 4,639,328 | 0.0009765625 | 0.684013 |

FP32 seeds 781/8287 change to 65371/93179. FP16 modifier seed 815 changes to
72341; alias/DPP input seeds also change. Fixed captured and exhaustive inputs
repeat, so only the random modifier/input portions are holdout data. There is no
unseen half encoding beyond the exhaustive mathematical sweep. All existing
bounds retain margin; output scaling explains the larger absolute errors.
The unchanged raw comparisons cover categorical results and storage contracts.
The half computed table's unmodified domain is checked against the independent
rocjitsu digest, rather than assuming FP32 narrowing matches arbitrary hardware.
Script and complete logs: `goc-build/coverage-work/testing-core-trig-calibration/run.py`.

The exact-half-unary review also applies the existing half SFU bound to loose
architecture-policy comparisons that previously required raw equality. Clang and
GCC each measure 28,095,424 finite loose pairs across that suite with zero error,
including the exhaustive guest-mode sweep already recorded above. This confirms
current margin, not an obligation for future loose paths to reproduce exact bits.
No tolerance was changed. Measurement script and snapshots:
`goc-build/coverage-work/testing-half-sfu/measure.py`.

After making DPP half-unary inactive-lane checks bitwise, Clang and GCC each
measure 6,725,492 active finite loose pairs with zero error. Alternate LCG seeds
(`997 + variant` to `65537 + variant`, `231 + batch` to `7919 + batch`) measure
6,710,168 pairs, also with zero error. Fixed hardware and walking-bit batches
repeat; only random portions are holdout data. The earlier count included some
unwritten destinations and is retained as historical measurement, not numerical
coverage. The half SFU bound remains unchanged. Script and logs:
`goc-build/coverage-work/testing-dpp-half-unary-calibration/run.py`.

The FP32 DPP-unary boundary suite, restricted to numerical SFU comparisons,
measures 3,751,808 finite pairs on both Clang and GCC. Maximum relative error is
`1.1920928955078125e-7`, below the shared `2.5e-7` bound; maximum absolute error
is `2199023255552` for large-magnitude results. This fixed boundary corpus has no
random holdout; core unary random calibration/holdout is recorded above. Rounding,
fraction/mantissa extraction and storage are checked bitwise. Local script/logs:
`goc-build/coverage-work/testing-dpp-unary-calibration/run.py`.

## Scalar FMA captured corpus

The scalar FMA suite now uses the existing FMA FP32/FP16 bounds for loose
outputs. Clang and GCC each measure 1,818,448 finite nonzero FP32 pairs and
97,928 FP16 pairs, all with zero error. The per-input references themselves must
match the original independent hardware digest: FP32 uses the exact path; FP16
uses the existing independent integer-grid oracle. This validates that computed
reference over the captured domain (with normalized NaNs), not arbitrary inputs.
No new tolerance is inferred from zero error; the existing FMA bounds and their
broader calibration/holdout remain above. This fixed capture has no fresh random
holdout. Script and logs: `goc-build/coverage-work/testing-scalar-fma-calibration/run.py`.
