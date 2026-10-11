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
non-generic family references. 140 explicit family-file reviews exist and 14
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

## Packed and normalized quantization

Three reviews cover packed and normalized conversions and their architecture
integration tests. These families explicitly reject strict exact semantics.
Computed integer-significand rounding and FP16 RTZ search references are now
checked directly against retained captured outputs, with source/column context
in both wave widths. Exact integer/encoded comparisons intentionally test
quantization and field preservation; passing captures do not imply strict exact
API support or arbitrary-input hardware equivalence.

## Byte packing and architecture integration

Two reviews cover byte quantization/nibble offsets and packing architecture
integration. Byte packing exercises every rounding boundary and adjacent FP32
word in each destination byte; its computed reference is now directly checked
against the stored capture columns. The DPP hardware total is retained alongside
per-lane checks. Architecture tests distinguish RDNA4 exact FP16 field packing
from unsupported strict-exact byte packing, and now include DPP failure context.

## Integer/FP32 conversion

Two reviews record the integer-only rounding reference, independent literal
rounding/saturation witnesses, retained ordinary/DPP hardware digests and
architecture OMOD policy witness. These families reject strict exact semantics;
exact encoded comparisons test their current conversion/quantization behavior.
The hardware per-output assertions now report the full configuration and lane.

## FP64 conversion and NaN scope

Two reviews document integer-significand encoding, exponent/midpoint sweeps and
split-word aliases. The midpoint sweep uses representative fractions across all
FP32 exponents, not all possible midpoints. Loose NaN checks deliberately require
quiet classification without payload/sign equality; finite-only alias checks
remain exact and separate NaN-policy tests cover both architectures. Existing
literal witnesses and per-input diagnostics are adequate here, so this review
adds no redundant fixtures or test loops.

## FP16 conversions

Two reviews cover full half-encoding sweeps, narrowing midpoint neighbors,
overflow/modifier interactions and half preservation. Hardware captures anchor
empirical OMOD/subnormal behavior. The architecture widening test explicitly
inherits rocjitsu's RDNA3 signaling-NaN preservation rule, so its raw comparison
is distinguished from ordinary loose quiet-NaN checks. Strict exact remains
unsupported for these APIs. DPP capture failures now report configuration and
per-lane values alongside the retained hardware digest.

## Rounding and fractional parts

Two reviews distinguish hardware-result corpora, literal edge witnesses and
DPP differential checks. The exact-rounding DPP test does not request optional
state; a check of an untouched local variable was removed rather than counted
as exception coverage. It now reports routing and host/guest settings on failure.
FRAC remains loose/fallback-only, with retained hardware regression digests and
literal tiny-negative, denormal, half-selection and masked-alias witnesses.

## Exponent and mantissa extraction

Two reviews record the leading-bit exponent proof and hardware/literal mantissa
evidence. FP32 exponent captures now compare each input independently before
checking the retained digest. Mantissa DPP references are explicitly differential;
literal special values and subnormal powers supplement its hardware hashes.
State-preservation assertions apply only to loose calls that pass a state pointer,
not exact calls opting out. FP64 exponent extraction remains loose/fallback-only.

## Exponent scaling

Four reviews cover FP16/32/64 scaling and architecture integration, including
boundary rounding, guest flush/scaling rules, split-word/half preservation and
strict-exact rejection. Long-double references rely on the wider x86 range used
by the validated builds; no broader portability claim is made. A missed random
pair-generation site in LDEXP was fixed by sequencing input and exponent draws.
The follow-up scan of multiple `random()` calls per statement found only
sequenced declarations and mutually exclusive conditional branches elsewhere.

## Moves and swaps

Three reviews record direct bit-field and lane-index references, before-image
alias handling and literal register-offset limits. Raw equality applies in both
semantics. The half-move capture now checks each output independently while
retaining the hardware digest that anchors ignored OMOD/CLAMP. Relative-move
coverage is explicitly baseline-CPU coverage; other move/swap tests iterate CPU
selectors. Native relative-swap captures cover both wave widths.

## Hardware registers and constant predicates

Two reviews cover independent per-bit register models, captured writable-field
behavior and simple constant comparison masks. Hardware-register captures now
check each output and report field boundaries before their retained digest.
Optional exact state rejection and preservation are distinguished from successful
state generation. Constant predicates require exact masks in both semantics;
there is no floating numerical result for a fuzzy comparison here.

## Carry and Wave64 integer arithmetic

Five reviews cover carry/borrow, wide MAD and remaining Wave64 integer families.
Carry references now compare directly with retained captured values and masks.
Carry DPP and wide-MAD hardware digests have per-output diagnostic companions.
The wide-MAD model shares the production `Uint128` utility; its independent
hardware digests and literal extended-sign witnesses remain essential evidence.
Integer results require exact comparisons under either semantics. Wave64 reviews
record all-selector coverage, supported modes, upper-lane masks and overlapping
outputs without claiming separate native hardware captures where none exist.

## Absolute differences and packed integer MAD

Three reviews record widened integer/window references, deterministic exhaustive
and sampled inputs, and independent literal overflow/selection witnesses.
The compact Wave32 literal cases now exercise strict exact as well as loose
semantics; the large loose sweeps are not mislabeled as exact coverage.
SAD DPP capture retains its hardware digest with per-lane diagnostics. Wave64
threshold cases independently test each accumulator and coincident outputs.

## Floating predicates and their exception state

Three reviews cover architecture integration, Wave64 and captured comparison
exceptions. Predicate masks require exact equality even in loose semantics.
Captured exception-pair tests now independently check predicate results as well
as reporting/nonreporting agreement. State assertions apply only to calls that
actually pass the state pointer. Records distinguish supported RDNA4 reporting
from RDNA3 atomic rejection, and reused per-pair hardware classifications from
native Wave64 captures. No new numerical model or golden fixture was introduced.

## Floating min/max selection

Four reviews record selection, signed-zero and NaN rules separately from
approximate numerical arithmetic. Raw finite comparisons are intentional for
operand selection and specified binary scaling/quantization; strict exact
semantics remains unsupported. Literal architecture checks enforce payload
rules where intended, while general half tests permit canonicalized NaNs.
Half-median hardware corpora now have per-input checks and retain their hashes,
including the original NaN canonicalization policy. Mode/route integration
sampling is explicitly distinguished from exhaustive Cartesian coverage.

## Cube and division architecture integration

Three reviews distinguish independent cube/quotient-selection references from
differential DIV_FMAS policy checks. Cube computed references now compare
directly with retained literal captures; Cartesian hardware digests retain
per-input diagnostics. This establishes captured-domain equivalence without
claiming arbitrary-input proof. Borrowed division literals cover tiny quotients,
fused scaling, midpoint and overflow behavior. Architecture integration records
stratified semantics coverage and unsupported exact reporting explicitly.

## DIV_SCALE and DIV_FMAS numerical evidence

Two reviews separate retained hardware numerical/exception digests and literal
boundary outputs from differential CPU/alias/host-mode tests. DIV_FMAS exception
corpora now check preservation of every bit outside the captured seven flags.
A proposed ordinary-FMA invalid-operation witness was rejected: DIV_FMAS has
its own suppression rules, so those flags remain hardware-digest anchored.
Digest failures distinguish the Cartesian and random corpus. DIV_SCALE condition masks are ordinary outputs, independently
hardware-anchored; broad random integration checks are not mislabeled as new
numerical evidence. DIV_FIXUP is reviewed below.

## DIV_FIXUP and half architecture behavior

Two reviews record independent classification/scaling, retained numerical and
exception captures, and borrowed half-precision ordering rules. Cartesian
captures now have per-input checks, including preserved destination halves.
Exception corpora check all bits outside the seven captured flags. Separate
flushed-input literals anchor invalid/divide-by-zero and CLAMP suppression;
DPP accumulation checks are explicitly differential. The exhaustive provisional
half sweep is distinguished from sampled guest/modifier integration.

## Trigonometric preprocessing

Two reviews distinguish TRIG_PREOP's exact table extraction from approximate
trigonometric arithmetic. Retained hardware digests and individual samples cover
RDNA4; the RDNA3 computed reference inherits rocjitsu's table and RTZ extraction.
Architecture witnesses expose table precision and scaling differences. Records
state the long-double assumption and the Wave32 even-exponent sampling limit.
Optional-state assertions now apply only when the pointer is passed.

## MULLIT loose lighting arithmetic

Both family files now compare general numerical results with measured fuzzy
bounds while retaining exact sentinel, signed-zero, untouched-storage, literal
policy and hardware-capture checks. The reference is a loose mathematical and
policy reference; strict exact remains unsupported. Calibration and an additional
deterministic random seed pass on Clang/GCC. Review records distinguish the
sampled routing product from exhaustive modifier and Cartesian input sweeps.

## Accumulating and literal FMA

Two reviews separate integer-grid exact references, captured special-value rules,
and loose numerical proximity. FMAC retains upstream half-FMA/FMAC witnesses;
FP32 literal-FMA captures now include per-input reference checks before their
digests. General loose checks use calibrated fuzzy bounds; exact payloads,
preserved halves/storage, signed zeros and finite fused-rounding literals remain
raw. Records explicitly delimit Wave32 half coverage and non-strict FP32 FMAC
requests rather than claiming strict exact coverage from those requests.

## Half FMA, routing and exceptions

Two reviews cover the ordinary and DPP half-FMA suites, including the Wave64
accumulating/literal forms that complement the preceding Wave32 reviews.
Computed integer arithmetic remains distinct from captured special-value rules.
General loose writes now use measured fuzzy comparisons; exact boundary and
payload witnesses remain raw. Exception capture tests preserve every nonflag bit,
with literal accumulation and aliased-output tests providing local diagnostics.
The existing stratified DPP mask coverage is retained.

## Mixed FMA

The mixed-FMA review covers format selection, direct half rounding, DPP, both
architectures/widths and optional reporting. General loose outputs are fuzzy;
unwritten storage, preserved halves, exact alias cases and captured witnesses
remain raw. The integer-grid numerical reference is distinguished from empirical
NaN/denormal evidence and zero-exception captures. Comparator regression tests
ensure the fuzzy allowance cannot hide storage corruption or weaken exact mode.

## Exact FP32 FMA and rounding exception captures

Two reviews distinguish complete retained hardware digests from sampled computed
reference checks and broader integration cases. One rotating lane per batch now
provides a local diagnostic in FP32 FMA captures without repeating the expensive
integer reference for every captured output. Rounding-exception tests preserve
all nonflag bits while retaining the captured seven-bit digest. Records explicitly
state the Wave32 scope and cycled CPU coverage of those exception tests.

## Output scaling and scalar guest state

Two reviews record retained conversion/scaling captures and simple scalar
bit-field/PC arithmetic. Output-scaling tests no longer use a production helper
to construct inputs; the original byte-as-word hash serialization is documented
accurately. Their aggregate witnesses complement the previously reviewed
conversion/LDEXP families. Scalar immediate coverage is exhaustive and GETPC now
checks loose as well as exact semantics, including wraparound and aliasing.

## FP64 output integration

The FP64 output review preserves the distinction between hardware SFU fuzzy
comparisons and exact captured scaling/selection witnesses. Alias tests now
check every nondestination register as well as the split output. Test decoding
is independent of production helpers; capture serialization and the Wave32,
full-EXEC scope are documented explicitly.

## FP64 arithmetic and selection

The main FP64 suite retains raw comparisons for exact dyadic mask/alias sweeps,
selection and literal boundary behavior. A new compact deterministic random
ADD/MUL/FMA test adds fuzzy numerical coverage against wider fused arithmetic,
including cancellation. An alternate seed measures a small nonzero difference
within the documented bound. The review distinguishes loose coverage from the
explicit rejection of strict exact semantics/reporting.

## FP64 unary operations

The unary review records the existing calibrated mathematical reference and
separate hardware SFU evidence. Coincident destination aliases now check the
retained high word even for NaNs, while finite mantissa extraction compares raw
bits. Approximate SFU results remain fuzzy. Numerical failure context is retained
without formatting a trace on every successful lane.

## Trigonometric exceptions

The exception review distinguishes direct retained flags from differential DPP
routing/accumulation. Both-width capture replays now preserve every nonflag bit;
DPP failures identify the relevant route, mask and lane. Exhaustive half inputs
and sampled FP32 corpora remain explicitly bounded evidence, with CPU selectors
cycled across inputs rather than a full CPU/input cross-product.

### Trigonometric architecture policy integration

Reviewed both FP32 and FP16 architecture suites, distinguishing captured FP32
anchors from the half suite's differential FP32-API-based reference. Exact
storage and categorical checks remain separate from calibrated loose arithmetic.
The half suite exhausts encodings for guest policy/OMOD; routing, selectors and
aliases use boundary inputs rather than the full Cartesian product. Core trig
numerical suites retain their own captures and remain separate review items.

### Core trigonometric suites

Reviewed FP32 captured outputs and FP16 full-encoding rocjitsu digests, retaining
independent aggregate witnesses alongside computed per-input diagnostics. The
half computed table has validation over its unmodified domain; modifier policy
coverage is documented separately rather than extrapolating that equivalence.
Loose tolerances now have alternate deterministic-seed measurements on both
compilers. Removed production bitcast dependencies and improved failure context.

### Half unary and log/exp policies

Reviewed the exhaustive mathematical half-unary corpus and the stratified
RDNA3/RDNA4 log/exp policy suite. Computed SFU references justify loose proximity,
not arbitrary bit-exactness; exact SFU fixtures remain a separate review item.
Tightened inactive-lane and nondestination preservation and added failure context
without introducing new golden data or broadening the tolerance bounds.

### Exact half unary

Added independent full-encoding rocjitsu RCP/RSQ/SQRT checkpoints, with identical
Clang/GCC generation and 4096-input failure localization. The 384-byte fixture
covers unmodified quiet-NaN behavior; existing policy and literal tests cover
other semantics without treating mathematical references as arbitrary SFU proof.
Removed vacuous state assertions and separated loose SFU tolerance from raw
inactive storage. FREXP's computed reference follows exact normalization.

### DPP half unary

Reviewed the two hardware digests, independent loose numerical references and
routing/alias/validation sweeps. Reference construction now uses saved source
words, and all source and unwritten destination words must remain identical.
Canonical NaN comparisons apply only to numerical results, not inactive storage.
The deterministic LCG has explicit sequential unsigned updates; alternate seeds
exercise the random portions without replacing fixed hardware witnesses.

### DPP FP32 unary

Added per-lane computed diagnostics for the simple captured hardware corpus,
retaining its independent digest and explicitly limited input domain. Disjoint
source words are checked bitwise. Simple rounding and fraction/mantissa extraction
now use raw results; only numerical SFU results receive the calibrated tolerance.
The fixed boundary routing suite complements core random unary coverage.

### Core FP32 unary

Reviewed mathematical, captured hardware and differential architecture-policy
references separately. Simple unary results and signed zero now remain exact in
the deterministic random tests as well as boundary tests; SFUs retain calibrated
fuzzy comparisons. Added disjoint source and out-of-wave preservation checks.
Failure-only per-lane formatting removes recurring diagnostic work from passing
random and EXP/LOG tests without reducing input coverage.

### RCP_IFLAG

Reviewed independent numeric/status captures, borrowed normalized-RCP checkpoints,
and mathematical proximity references separately. Exact captured numeric outputs
are now checked bitwise; computed mathematical comparisons remain proximity
checks outside the independently anchored domain. Null reporting equivalence now
covers both semantics and wave widths. DPP diagnostics include masks and aliases,
and state assertions no longer claim coverage when the pointer was null.

### Pseudo-scalar numerical APIs

Reviewed captured policies/modifiers, dense FP32 hardware evidence and half
mathematical checks. Reused the independent half-SFU checkpoints for the full
preserved-denormal domain, closing a gap where exact execution had only a fuzzy
full-encoding reference. Other policies retain selected independent captures and
mathematical proximity checks. Loose corpus calls now verify non-null state
preservation. No additional golden arrays were needed.

### Scalar FMA

Separated loose proximity from hardware hash equality. FP32 loose references use
the exact implementation with the retained independent capture check; FP16 uses
the existing independent integer-grid reference, now directly hash-validated over
the captured domain. Exact FP32 output and literal payload checks remain bitwise.
This reuses proven computed behavior without expanding golden fixtures.

### Scalar floating arithmetic

Added independent computed ADD/SUB/MUL references whose digests match both
existing hardware datasets, including guest flushing and saturation policies.
Loose numerical outputs now receive fuzzy comparisons, while min/max selection
and categorical/storage results remain exact. The capture check proves reference
equivalence for this corpus without requiring additional golden arrays.

### Arithmetic exception integration

Recorded the single-lane-reference dependency explicitly and linked the separate
family capture evidence. Expanded checks to all nondestination storage and every
nonflag bit. The host/opt-out test now checks active exact calls with and without
reporting, rather than treating empty EXEC as sufficient opt-out coverage.

### Dual instruction slots

Reviewed independent simple literals and RDNA3 dot captures separately from
ordinary-instruction differential references. Capture replay now covers every
CPU selector. Floating adapter checks permit loose numerical tolerance but require
raw inactive storage; source registers are preserved. Integer failures identify
the register/lane. Pair tests document caller staging rather than implying a
combined dual-issue API exists.

### Packed binary half arithmetic

Separated approximate ADD/MUL from exact selection and storage at each half.
Reviewed deterministic exhaustive-A/random-B generation, all packed modifiers,
guest policies and both widths; added missing Wave64 source checks and per-half
failure diagnostics. Literal contract witnesses remain exact. These loose-only
APIs use independent mathematical references without a new golden-data burden.


Packed FMA/FMAC now distinguishes finite loose comparisons from exact integer-grid
references, literal rocjitsu NaN/fused witnesses and raw storage checks. All
available CPU selectors cover architecture guest-policy integration. Wave64 and
FMAC modifier sweeps check nondestination words, and exception capture tests
preserve every bit outside the seven-bit arithmetic field. Both wave widths
retain captured exception digests and explicit union/alias/validation witnesses.
Computed numerical references remain justified by exact half arithmetic and the
existing exceptional-policy evidence; no new golden arrays were needed.


RDNA3 F32-output DOT2 now has independent loose fractional arithmetic checks,
in addition to its inherited gfx1100 exact captures. Direct reference-to-capture
checks establish the borrowed model on that captured domain; transformed-input
modifier tests remain explicitly differential. Half-output architecture tests
use exact small-integer references to check routing and field preservation,
plus literal policy witnesses and exhaustive accumulator encodings. They remain
loose-only and make no claim of empirical bit-exact support. Both families cover
both wave widths; new numerical loose DOT2 tests cover every CPU selector.


Half-output DOT2 now has independent fractional double-reference tests across both
architectures, widths and CPU selectors. The existing all-encoding random test
is explicitly scalar/SIMD differential, with fuzzy finite comparisons and raw
unwritten storage. Literal and integer routing assertions remain exact, as does
the retained fixed-domain hardware DPP digest. Ordinary alias sweeps now check
all nondestination words. This family remains loose-only.


FP8/BF8 DOT4's independent mathematical decoder, exhaustive repeated-byte pairs
and separately calibrated mixed-byte holdout already provide loose numerical
coverage for all four formats and both widths. Completed its review by adding
nondestination storage checks to aliases and immutable source/out-of-wave checks
to the bounded corpus. Integer DPP routing stays raw; these loose-only APIs
require no invented exact reference. Diagnostic formatting now occurs on failure.


Interpolation reviews distinguish raw routing/capture witnesses from general loose
arithmetic. FP16-input references now reproduce the retained hardware digests
directly, without using GoC output; this validates their captured domain while
the APIs remain loose-only. Finite comparisons use the output format's calibrated
bound, while inactive storage and unselected halves remain raw. FP32 references
use long-double FMA with explicit quad routing. Both families now protect random
test references against source mutation and pass alternate-seed calibration.
Wave64 capture reuse is explicitly distinguished from native hardware captures.
