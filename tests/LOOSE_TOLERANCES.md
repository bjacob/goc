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

## Scalar ADD/SUB/MUL

The two existing hardware corpora validate the new computed reference directly:
double arithmetic, explicit input-denormal policy, pre-rounding tininess threshold
and numeric half conversion. All reference digests match the retained captures.
Clang and GCC measure 665,512 finite nonzero FP32 comparisons and 656,632 FP16
comparisons with zero error. Bounds are `(2^-149, 1.2e-7)` and `(2^-24, 1e-3)`:
a one-minimum-subnormal floor and approximately one encoding step with margin,
consistent with the FMA format budgets. Nonzero allowances permit future loose
implementations; they are not presented as observed nonzero error. Zero/infinity
signs, upper-half zeroing and min/max selection are separate exact checks.
The second dataset concentrates on min-normal boundaries; neither fixed capture
is claimed as fresh random holdout. Script and logs:
`goc-build/coverage-work/testing-scalar-fp-calibration/run.py`.

Dual floating adapters use the existing FP32 FMA bound against the corresponding
ordinary-instruction paths. Both compilers measure 8,992 finite nonzero comparisons
with zero error. This is differential integration over fixed boundary operands,
not independent hardware calibration or fresh random holdout; ordinary arithmetic
families provide the numerical evidence. Storage and categorical results remain
raw. Script/logs: `goc-build/coverage-work/testing-dual-calibration/run.py`.

Packed half ADD/MUL reuse `(2^-24, 1e-3)` per half. Clang and GCC each measure
5,354,256 finite nonzero comparisons with zero error using seed 7231, and
5,352,732 with zero error using alternate seed 51787. Every A encoding is paired
with pseudorandom B; only B changes in the holdout, while fixed modifier/policy
and special-pair sweeps repeat. Selection, literal witnesses, zero signs and
unwritten storage remain exact. Script/snapshots/logs:
`goc-build/coverage-work/testing-packed-binary-calibration/run.py`.


Packed half FMA/FMAC reuse the established half-FMA bound (2^-24, 1e-3).
Clang and GCC each measure 6,958,368 finite nonzero comparisons with zero error
using seed 2371, and 6,955,040 with zero error using alternate seed 61981.
The holdout changes random triples and random packed modifiers; the exhaustive
A-encoding prefix and fixed boundary/policy/alias sweeps remain. Exact semantics,
literal witnesses, categorical results and storage remain bitwise checks.
Script/snapshots/logs:
goc-build/coverage-work/testing-packed-fma-calibration/run.py.


RDNA3 F32-output DOT2/DOT2ACC with FP16/BF16 normal factors bounded by four
and FP32 C in approximately [-4, 4] uses (atol=2e-6, rtol=2e-7).
The absolute term accommodates cancellation and alternative accumulation orders.
Seeds 42683 and 79393 each exercise 114,432 finite results over both widths,
available CPU selectors and random signs/selections. Both Clang and GCC measure
maximum absolute error 1.1920928955078125e-6 on each seed; maximum relative errors
are 8.6570342731986881e-6 and 0.00035398230088495576 respectively near cancellation.
The held-out seed passes without changing the bound. Exact RDNA3 captures are
not compared with this mathematical reference or weakened by this tolerance.
Script/snapshots/logs: goc-build/coverage-work/testing-dot2-architecture-calibration/run.py.


Half-output DOT2 uses the existing FP16 (2^-24, 1e-3) bound and a BF16 bound
(2^-126, 4e-3). Independent double sums use normal inputs bounded by four on both
architectures and widths. The BF16 relative term is just above the observed
half-ULP relative rounding scale; the FP16 absolute term covers near-zero
subnormal rounding. Existing all-encoding scalar/SIMD differential comparisons
also use these bounds, while NaN classification, infinity/zero policy witnesses
and storage are checked separately.
Clang and GCC report identical measurements. Original seeds 37493 (fractional)
and 11719 (differential) give 89,564 FP16 finite comparisons, maximum absolute
error 0.0038810186088085175 and relative error 0.0074560095436922158; BF16 has
88,513 comparisons, absolute error 0.030745506286621094 and relative error
0.0038033074399128927. Alternate seeds 81559 and 63727 give 89,675 FP16 comparisons,
absolute error 0.0038728713989257812 and relative error 0.00047246083704427944;
BF16 has 88,368 comparisons, absolute error 0.031131744384765625 and relative
error 0.0038610038610038611. Both seeds pass the final bound. Counts include
differential checks and repeated architecture/CPU cases, not unique hardware samples.
Script/snapshots/logs: goc-build/coverage-work/testing-half-dot-calibration/run.py.


Interpolation holdout and FP16-input calibration: Clang and GCC agree on all
measurements. FP32 interpolation seed 971 produces 1,004,608 finite comparisons,
and alternate seed 51263 produces 1,004,288; all have zero observed error.
FP16-input interpolation uses existing FP32/FP16 FMA bounds against its integer
reference: seed 314159 gives 23,290,304 FP32-result and 15,471,200 half-result
finite comparisons, while seed 74117 gives 23,311,808 and 15,473,568 respectively.
All observed errors are zero. The large counts include repeated modifier, mask,
alias and CPU configurations; only random operands change in the holdout.
Fixed captured/RTZ-host-rounding witnesses and storage remain raw. Direct
reference-to-hardware-digest validation establishes the integer reference on the
retained capture corpus without implying arbitrary hardware-exact API support.
Script/snapshots/logs: goc-build/coverage-work/testing-interpolation-calibration/run.py.


Half ADD/SUB/SUBREV/MUL and their DPP forms reuse (2^-24, 1e-3); min/max selection,
fixed hardware witnesses, zeros/infinities and storage remain raw.
Ordinary half-binary tests measure 10,881,860 finite nonzero arithmetic comparisons
with seeds 427/12 and 10,829,636 with alternate seeds 73951/547.
DPP tests measure 6,828,136 with LCG starts 997+variant and 231+batch,
and 6,996,660 with starts 65537+variant and 7919+batch.
All observed errors are zero on both Clang and GCC. The counts include repeated
masks/modifiers/CPU paths; the holdout changes random operands, not fixed witnesses.
Calibration excludes the dedicated comparator regression's deliberately rejected
examples. Script/snapshots/logs: goc-build/coverage-work/testing-half-binary-calibration/run.py.


The bounded DOT2 architecture corpus now includes RDNA4 F16/BF16 forms.
Each seed checks 190,720 finite outputs on Clang and GCC, with unchanged maximum
absolute/relative errors from the original RDNA3 measurements above:
absolute 1.1920928955078125e-6, relative 8.6570342731986881e-6 for seed 42683
and 0.00035398230088495576 for seed 79393. The existing bound (2e-6, 2e-7)
passes both. RDNA4's separate fixed special-value scalar/SIMD differential
test contributes 17,960 finite comparisons with zero error under the FP32 FMA
bound; it is repeated unchanged rather than treated as a random holdout.
Script/snapshots/logs: goc-build/coverage-work/testing-dot2-calibration/run.py.


FP32 FMA integration uses the existing (2^-149, 1.2e-7) bound. Its fixed Wave64
special-value integrations plus deterministic arbitrary-word FMA corpus produce
1,333,176 finite comparisons with seed 41761 and 1,332,900 with alternate 97613,
all with zero observed error on Clang and GCC. The separate fixed DPP arithmetic
sweep contributes 372,524 finite nonzero comparisons with zero error under the
same numeric bound. DPP inputs are unchanged in the second run and are not a
random holdout. Storage/selection/captured boundaries stay raw. Script and logs:
goc-build/coverage-work/testing-arithmetic-calibration/run.py.


FP32 binary loose arithmetic uses the existing (2^-149, 1.2e-7) bound against
independent binary64 operations and the documented OMOD policy. Combined fixed
modifier/alias/DPP cases and arbitrary-word seed 68147 produce 4,621,604 finite
nonzero comparisons; alternate seed 95383 produces 4,620,508. Both have zero
observed error on Clang and GCC. Only random operands/modifiers change in the
holdout; fixed policy witnesses remain. Exact ADD/MUL, selection and storage
assertions are excluded from these fuzzy counts.
Script/snapshots/logs: goc-build/coverage-work/testing-binary-calibration/run.py.


### Sparse FP8/BF8 WMMA

`Swmmac8.FractionalMatricesAgainstDoubleReference` constructs logical compressed
A, dense B and legal 2:4 selectors, packs the two ISA layouts, and compares
against independent double accumulation. Four type pairs, both widths, every
index key and all four available CPU selectors cover normal signed factors
bounded by 4 and fractional accumulators bounded by 4. Each seed performs
1,572,864 finite comparisons on each compiler. Seed 71849 measured maximum
absolute error `2.6971101760864258e-6` and relative error
`0.00060938452163315055`; holdout 92717 measured `3.2782554626464844e-6` and
`0.0003808543126461264`. Clang and GCC measurements match. Cancellation accounts
for the relative peaks; the absolute term covers them. The existing WMMA bound
`(atol=1e-5, rtol=1e-5)` passes unchanged, including the holdout. These measurements
do not establish a tolerance for arbitrary unbounded FP8/BF8 matrices.

The separate hardware digests deliberately retain raw comparisons: their
small-integer inputs and sums are exactly representable. Exhaustive encoding
checks use exact multiplication by sixteen or explicit NaN classification.


### Sparse FP16/BF16 WMMA

`Swmmac16.FractionalMatricesAgainstDoubleReference` packs independent logical
sparse matrices, applying A/B negation by compressed pair position. It covers
both widths, all keys, all negation combinations, and all four CPU selectors.
Operands and accumulators are normal signed values below four, with eight
operand exponents. Double accumulation provides the mathematical loose oracle;
packed outputs are decoded and compared as real numbers. The existing FP32
WMMA bound remains `(1e-5, 1e-5)`. Packed FP16 uses `(1e-5, 5e-4)` and BF16
`(1e-5, 4e-3)`, allowing final-format rounding and measured accumulation error.

| Output | Comparisons per seed/compiler | Primary max absolute / relative | Holdout max absolute / relative |
| --- | ---: | --- | --- |
| FP32 | 786,432 | 6.3627958297729492e-6 / 0.0028294063125030491 | 5.4836273193359375e-6 / 0.012451593684837652 |
| FP16 | 393,216 | 0.015602109953761101 / 0.0021952290355627103 | 0.015521987341344357 / 0.013722569754741336 |
| BF16 | 393,216 | 0.11626553535461426 / 0.0038905035226522119 | 0.12195014953613281 / 0.0038902225991997263 |

Primary seed 38671 and holdout 89767 pass on Clang and GCC, with identical
measurements. Near-zero cancellation explains relative peaks above `rtol`;
the absolute term covers those cases. These are bounded-domain measurements,
not promises for arbitrary exponent spreads. Separate raw capture, midpoint
and infinity/saturation tests retain their exact categorical contracts.


### Dense FP8/BF8 WMMA

`SubbyteWmma.FractionalFp8MatricesAgainstDoubleReference` uses normal signed
factors below four, fractional accumulators bounded by four, independent double
matrix multiplication, and the existing `(atol=1e-5, rtol=1e-5)` WMMA bound.
All four format pairs, both widths, all C modifiers, four destination layouts
and every available CPU selector participate. Each seed has 524,288 finite
comparisons per compiler. Seed 52813 measured maximum absolute error
`2.384185791015625e-6` and relative error `7.6754550627134739e-5`; holdout 97651
measured `1.9073486328125e-6` and `0.0001120699316373417`. Clang and GCC agree.
The absolute term covers cancellation near zero; the unchanged bound passes
both seeds. The separate offline dyadic goldens, exact integer references and
single-code decoding tests retain their raw comparisons on those domains.


### Dense FP32-output FP16/BF16 WMMA across architectures

`WmmaReplicated.FractionalMatricesAgainstDoubleReference` compares both RDNA3
and RDNA4, both widths and operand types, all 64 modifiers and cycled destination
aliases against independent logical double accumulation. RDNA3 replicas differ
by group; RDNA4 uses one distributed logical matrix. Signed normal factors use
eight exponents below four, with fractional accumulators bounded by four.
Each seed covers 524,288 finite comparisons per compiler. Seed 63827 measured
maximum absolute error `5.5860728025436401e-6` and relative error
`0.00036729950565099321`; holdout 91397 measured `6.4261257648468018e-6` and
`0.0010111704595472336`. The unchanged WMMA bound `(1e-5, 1e-5)` covers both,
including cancellation through its absolute term. Clang/GCC measurements match.

The core `wmma_test.cpp` capture and general differential comparisons now measure
128,065 finite pairs (alternate differential seeds: 128,080), with unchanged
maximum absolute/relative errors `0.03125` / `8.7627527626987843e-8` on both
compilers. Alternate seeds replace 73/107 with 479/719 in the scalar/SIMD
corpora; hardware captures are unchanged. These differential comparisons are
supplemental dispatch checks, not independent proof of hardware accuracy.
