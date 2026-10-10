# RDNA3/RDNA4 coverage worklist

The goal is complete arithmetic/register-operation coverage for both architectures,
with shared APIs where semantics agree and `_rdna4` variants where they differ.
The [generated inventory](ISA_INVENTORY.md) lists every canonical mnemonic in AMD's
RDNA3 and RDNA4 XML, including excluded operations and unresolved scope boundaries.
Regenerate it with `python3 tools/isa_inventory.py <machine-readable-isa/isa>`.

An API declaration is not a completed audit. Existing RDNA4 tests do not prove
RDNA3 compatibility. Equal XML descriptions do not prove identical arithmetic.
A family is complete only after reviewing its instruction results, modifiers,
wave widths, aliasing, implicit register outputs, and numerical semantics against
both targets. Exact support follows empirical evidence; unsupported exact/global
state requests must fail honestly under the documented contracts. Loose SIMD
paths must retain their existing performance and may skip optional state outputs.

The [cross-architecture review queue](ARCHITECTURE_AUDIT.md) now identifies a
confirmed broad gap: ordinary RDNA3 OMOD/IEEE policies differ from the RDNA4
policies currently used by many unsuffixed floating APIs. Those names must not
be treated as verified shared semantics. Width completeness does not close this gap.

## Evidence and starting point

- Start: `80470bd`, 682 instruction APIs, 674 tests passing with Clang and GCC.
- ISA and rocjitsu source: local TheRock `rocm-systems` revision `ffc144c564c`.
  XML content fingerprints are recorded in the inventory.
- Borrowed hardware captures establish only the targets and cases they capture.
  New RX 9070 Wave32/Wave64 DS_SWIZZLE captures are recorded below.
- Inventory initially finds 804 in-scope canonical mnemonics in the union:
  664 have matching API names; 140 lack them. Another 14 control/register
  boundaries require explicit review. 578 memory/control/scheduling mnemonics
  are excluded. These counts are neither coverage percentages nor correctness claims.

## Cross-cutting completion gates

- [x] Enumerate both XML instruction sets and identify matching public API names.
- [x] Resolve the initial boundary rows: relative addressing and hardware-register
  access remain in scope; pipeline no-ops/flushes are scheduling exclusions.
- [ ] Map XML aliases and dual-issue operations; distinguish assembly aliases
  from missing semantics and define simultaneous-read behavior for paired outputs.
- [ ] Review every shared mnemonic for architectural differences, including
  special values, rounding, denormals, exceptions, modifiers, and implicit state.
- [ ] Fill every in-scope missing API or record a concrete implementation blocker.
- [ ] Borrow all applicable rocjitsu empirical models and captures; document any
  unsupported exact behavior and optional global-state reporting.
- [ ] Audit Wave32/Wave64 contracts, especially comparisons, lane routing, and EXEC.
- [ ] Preserve existing SIMD dispatch; compare performance for affected hot paths.
- [ ] Validate complete Clang/GCC suites and targeted ASan/UBSan tests.
- [ ] Final requirement-by-requirement audit; unresolved items prevent completion.

## Family checklist

| Family | Current evidence | Remaining work |
|---|---|---|
| RDNA3 WMMA | All six mnemonics, both widths, exact rocjitsu GFX11 model; `wmma_replicated_exact_test.cpp`; 103 DOT2 captures and 28 packed matrices | Scalar implementations; verify inventory integration as other APIs evolve |
| RDNA4 WMMA/SWMMAC | Existing layouts, SIMD paths, exact FP16/BF16 tests | Audit current rocjitsu against existing implementation; FP8 exact availability and state |
| DOT2 and DOT2ACC | FP32 DOT2 split into unsuffixed RDNA3 / `_rdna4`; shared modifiers/DPP/loose SIMD; GFX1100 captures, aliases and host-state tests | DOT2ACC and dual forms reuse the same models and SIMD; RDNA3 reporting remains unsupported; ordinary forms support both wave widths |
| Floating min/max/median | RDNA3 names share existing RDNA4 NUM loose paths, including packed forms, Wave64 and SIMD dispatch | Exact IEEE-mode NaN policies and exception reporting |
| Scalar integer arithmetic | RDNA3 add/sub/addc/subb/addk spellings reuse RDNA4 implementations, as confirmed by XML aliases; all twelve CMPK forms added with exhaustive immediate tests | Continue full family audit; register/control operations tracked separately |
| Scalar/register moves | MOV/CMOV, MOVK/CMOVK, SAVEEXEC/WREXEC, relative moves, GETPC and round/denorm MODE updates; scalar-relative RX 9070 captures | General GETREG/SETREG access and continued semantic audit |
| Vector/register routing | Lane transfers, MOV/MOVREL, SWAP/SWAPREL, PERMLANE64 and DS permutations at both wave sizes; RX 9070 half-move, SWAPREL and DS_SWIZZLE captures | Broader cross-width/modifier audit; RDNA3 hardware confirmation |
| Comparisons | Existing RDNA4 results/SCC/EXEC and exception tests; integer relational predicates now have Wave64 scalar/SIMD paths; RDNA3 true/false results implemented at both wave widths | RDNA3 floating constant exception reporting and architecture-specific floating-policy audit |
| Basic floating arithmetic, FMA, conversions | Existing RDNA4 implementations and captures; confirmed OMOD/IEEE architecture differences | Split differing APIs and implement RDNA3 guest-mode policies; see ARCHITECTURE_AUDIT.md |
| Transcendentals, reciprocal, division | Existing partial exact coverage | Review newer rocjitsu models/captures, differing semantics and state outputs |
| Integer/vector arithmetic and bit operations | Existing broad RDNA4 coverage; bit-count family has exact scalar/SIMD paths and DPP at both wave sizes | Check RDNA3 spellings, modifier encodings, widths, carry outputs and all aliases |
| Interpolation | All six APIs at both wave widths; shared quad routing checked against both manuals; RDNA4 captures and independent arithmetic tests | Compare remaining RDNA3 numerical contracts and empirical behavior |
| Dual-issue VOPD | All 19 canonical operations; shared ordinary kernels, architecture-split DOT2ACC; Wave32 only | Direct VOPD hardware captures; inherited exact/global-state gaps |

Keep this checklist conservative: unchecked review work is not made complete by
a green test suite that exercises only the current implementation.

## Validated increments

- DOT2 architectural split: all 677 tests pass in Clang 21 and GCC 15 Release
  builds; all 10 DOT2 tests pass under Clang ASan/UBSan. The new API tests cover
  GFX1100 result captures, all sign/half selectors with DPP8/DPP16, EXEC masks,
  aliases and guards, all host rounding modes, and an explicit RDNA3/RDNA4
  result difference. RDNA3 optional exact exception reporting remains unsupported.
  Existing RDNA4 tests and benchmark rows now use the suffixed names. The loose
  SIMD kernel and its dispatch conditions are unchanged; no performance gain is
  claimed for this increment.
- Scalar RDNA3 spellings/immediates: 19 APIs added, reusing the existing scalar
  integer and comparison implementations. RDNA4 XML explicitly aliases all seven
  arithmetic spellings; rocjitsu's SOPK implementation confirms signed versus
  unsigned immediate extension for the twelve comparisons. Tests exhaust every
  16-bit immediate against boundary/equal operands and independently check carry,
  borrow, overflow, aliases and error atomicity. All 680 tests pass with Clang
  and GCC; the three new tests pass under ASan/UBSan. Missing in-scope API names
  are now 121; semantic audits remain distinct from name coverage.
- Basic scalar moves: six shared RDNA3/RDNA4 entry points added from rocjitsu's
  register-transfer contracts. Tests exhaust all signed immediates, every raw
  bit position in B32/B64, conditional preservation and aliases, invalid flags,
  and host FP-state preservation. All 683 tests pass with Clang and GCC; all
  three new move tests pass under ASan/UBSan. Missing in-scope API names: 115.
- Lane transfers: READLANE, WRITELANE and READFIRSTLANE implemented for both wave
  widths, using shared RDNA3/RDNA4 contracts from rocjitsu execution and register
  access. Explicit indices wrap; READFIRSTLANE selects lane 0 for zero EXEC.
  Tests cover every lane, wrapped/high-bit indices, single-bit and suffix EXEC
  masks, scalar output overlap and invalid flags. All 686 tests pass on Clang
  and GCC; the three new tests pass under ASan/UBSan. Missing API names: 112.
- Scalar EXEC operations: all 24 canonical SAVEEXEC/WREXEC B32/B64 APIs added,
  sharing validation and stores while retaining explicit Boolean expressions.
  The contracts follow rocjitsu, including B64 access to raw EXEC_HI independently
  of active wave width. Independent per-bit truth tables verify all operations,
  old/new destination distinction, zero EXEC, optional EXEC reporting and output
  alias priority. All 688 tests pass on Clang/GCC; both new tests pass under
  ASan/UBSan. Missing in-scope API names: 88.
- Constant integer comparisons: sixteen RDNA3 CMP/CMPX F/T forms added for
  I32/U32/I64/U64 operands in Wave32. Predicate evaluation does not read operands;
  true results still apply DPP output filtering. Independent DPP references check
  all new forms, CPU selections, semantics, masks and validation. All 689 tests
  pass on Clang/GCC; the new test passes under ASan/UBSan. Missing API names: 72.
  Floating F/T exception behavior and Wave64 comparison APIs remain open.

## Scope review after initial implementation increments

The functional-group labels in AMD's XML are not sufficient to decide scope.
The generator now honors explicit branch/termination flags even within SALU.
`S_CALL_B64`, `S_SETPC_B64`, `S_SWAPPC_B64`, and `S_CODE_END` are excluded as
control flow/traps. RDNA3 `LDS_DIRECT_LOAD` and `LDS_PARAM_LOAD` are labeled VALU
but their descriptions explicitly read LDS memory; they are excluded as memory
operations. `S_ALLOC_VGPR` allocates wave resources, `S_SLEEP_VAR` schedules sleep,
and `V_NOP`/`V_PIPEFLUSH` concern pipeline scheduling; these are excluded.

All twelve initially ambiguous relative-register and GETREG/SETREG operations
remain **in scope**. So do `S_GETPC_B64` (reads PC without branching),
`S_ROUND_MODE`, and `S_DENORM_MODE` (modify guest FP state, never host FP state).
This is not permission to implement guest mode updates with host fenv mutations.

All 1,396 canonical names remain listed. The revised scope contains 808 names:
732 have APIs and **76 are missing**. The 588 exclusions are explicit. The rise
from 72 missing names reflects twelve register operations admitted from the
review list and eight former in-scope exclusions; two other exclusions came
from the review list. No implementation was removed and no unresolved register
operation was silently excluded. Scope checks verified each reclassification.

## Floating min/max review in progress

At rocjitsu revision `ffc144c564c`, generated FP32 MIN/MAX, MIN3/MAX3 and
MINMAX/MAXMIN execution bodies for RDNA3 spellings and RDNA4 NUM spellings are
identical after whitespace normalization. MED3's wrappers differ in SIMD
availability; FP64 MIN/MAX have different first available encodings, requiring
like-for-like VOP3 review. FP16 and packed forms use other generated execution
paths and were not covered by that text comparison.

This supports sharing **loose** arithmetic, not a bit-exact equivalence claim.
Before adding these names, review NaNs (especially signaling NaNs), zero ties,
operand ordering in median networks, guest FP flags, output modifiers and
exception updates. Existing GoC loose/SIMD tests must remain valid, but cannot
substitute for RDNA3 hardware evidence. No min/max API was added by this audit.
- RDNA3 DOT2ACC: the VOP2 accumulator form now reuses the GFX11 exact DOT2 model
  and existing loose SIMD implementation, with D as the read/write accumulator.
  Only DPP routing is accepted; VOP3P sign/half controls are rejected as absent
  from its encoding. Captured results, DPP aliases, masks, loose SIMD and optional
  reporting contracts are tested. All 691 tests pass on Clang/GCC and all five
  architectural DOT2 tests pass under ASan/UBSan. Missing API names: 75.

Relative VGPR move range rules were checked against section 3.3.2.2 and the
VGPR-indexing tables of AMD's [RDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)
and [RDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf).
Both specify VGPR0 substitution for out-of-range VALU sources and discarded
out-of-range destination writes. Both specify an index limit of 255; rocjitsu
currently checks 1023. GoC follows the published 255 rule and records this
upstream discrepancy rather than treating the upstream code as hardware evidence.
The split instruction still extracts the documented 10-bit M0 fields before
range checking. No new hardware capture establishes behavior in that disputed
range. API views begin at VGPR0 with explicit base indices and bounds so fallback
reads are possible; 64-bit intermediate sums prevent accidental index wrapping.

- Wave32 vector-relative moves: four shared APIs implemented with bounded full
  register-file views, explicit base indices, VGPR0 fallback and DPP routing.
  Tests cover aliases, masks, empty ranges, split offsets, the documented index
  limit and widened address calculations. All 694 tests pass on Clang/GCC;
  all three relative-move tests also pass under ASan/UBSan. Missing API names: 71.
  Scalar-relative operations and Wave64 vector-relative variants remain open.

- RDNA3 DX9 FMAC: `goc_v_fmac_dx9_zero_f32` reuses DX9 FMA's scalar,
  x86-64-v3 and x86-64-v4 paths with D as the accumulator. rocjitsu's VOP2/VOP3
  execution at `ffc144c564c` confirms the accumulator is unmodified by source
  modifiers; A/B ABS/NEG, OMOD, CLAMP and DPP are supported. C modifiers are
  rejected. Tests cover all valid modifier combinations with DPP, aliases and
  masks, plus zero/NaN/infinity/subnormal cases and error preservation. The
  existing FMA hardware corpus validates the reused arithmetic on RDNA4;
  this is not a new RDNA3 FMAC hardware capture or a bit-exactness claim.
  Strict exact semantics remain unsupported. All 696 tests pass on Clang/GCC;
  the two new tests pass under ASan/UBSan. Missing API names: 70.

## Wide integer MAD architecture review

AMD's XML identifies RDNA3 `v_mad_u64_u32` as the alias of RDNA4
`v_mad_co_u64_u32`, with the same opcode, operands and unsigned carry contract.
rocjitsu at `ffc144c564c` has equivalent scalar arithmetic and saturation for
both. The new RDNA3 entry point shares GoC's implementation, including v3/v4
SIMD, exact integer arithmetic and full input/output alias handling. All six
wide-MAD tests exercise the new spelling where applicable, including an
independent 128-bit reference and the existing **RDNA4** hardware corpus.
This reuse does not turn that corpus into an RDNA3 hardware capture.

The initial signed RDNA3 review found a discrepancy (resolved by the full ISA
pseudocode below). rocjitsu's RDNA3 and RDNA4 handlers
both report signed overflow, whereas GoC's RDNA4 hardware captures establish
that the scalar output is bit 64 of the full mathematical sum (its extended
sign). These differ even without overflow: `0 * 0 + (-1)` has scalar bit 1
in the RDNA4 capture model but bit 0 in rocjitsu. XML's prose says
"overflow/carryout" and does not resolve the discrepancy. The subsequent RDNA3 pseudocode review below supplies the missing evidence;
rocjitsu's overflow result was not copied.

All 696 tests pass on Clang/GCC; all six wide-MAD tests pass under ASan/UBSan.
Missing canonical API names: 69. Wave64 MAD variants remain open.

## Register swaps and half-wave routing

`v_swap_b32` and `v_permlane64_b32` now have shared RDNA3/RDNA4 implementations
for Wave32 and Wave64. rocjitsu's two VOP1 execution files have identical swap
bodies and call the same PERMLANE64 handler. AMD's RDNA4 ISA section 16.8 confirms
that PERMLANE64 selects lane XOR 32 using destination EXEC, regardless of source
EXEC, and is a no-op in Wave32; both XML descriptions also specify that no-op.
Accordingly the Wave32 API omits EXEC and permits null operands. The Wave64
implementation snapshots both halves before any stores. Both swap operands are
read/write pointers, and whole-register aliases are supported.

Tests cover both wave sizes, full/empty/partial EXEC, lane 31/32 boundaries,
inactive sources, in-place routing, guard words and rejection of every modifier
bit. These are specification/reference tests, not new GPU captures. Scalar loops
provide exact bit operations; dedicated SIMD implementations remain open.
SWAPREL remains open because its relative-range rules need separate auditing.
The later half-register follow-up below implements SWAP_B16. Missing canonical
API names at this stage: 67.

Validation for these additions: all 700 tests pass with Clang and GCC; the four
swap/routing tests also pass under ASan/UBSan.

## DS register permutations and resolved signed MAD

Implemented DS_PERMUTE/BPERMUTE for shared Wave32, RDNA3 Wave64 and RDNA4
Wave64. AMD's RDNA3 manual section 12.5.2 explicitly restricts routing to each
32-lane half, whereas RDNA4 section 12.5.2 permits full-wave routing. The RDNA4
Wave64 APIs therefore use `_rdna4_wave64`. The RDNA3 detailed pseudocode omits
the half-wave base, but its explicit prose and rocjitsu agree on independent
halves; the implementation follows those. Both masks apply to reads and writes.
Scatter selects the highest active source on collisions, following rocjitsu and
the pseudocode, while documenting that hardware prose does not guarantee a winner.
Inputs and addresses are consumed before output stores for alias safety.

The scope audit also corrected `ds_bpermute_fi_b32`: despite its XML VMEM label,
it only routes registers and belongs in scope. Its new Wave32/Wave64 APIs read
inactive source lanes. Total in-scope canonical names are now 809, with 64 missing.
The DS_SWIZZLE follow-up below addresses rotate/FFT modes missing from
rocjitsu's simple quad/bit implementation.

The previously open signed MAD question is resolved by RDNA3 section 16.12's
explicit 65-bit concatenation: the scalar output is bit 64 of the full signed
sum, matching GoC's existing model rather than rocjitsu's signed-overflow flag.
`goc_v_mad_i64_i32` now shares the existing scalar/v3/v4 implementation. Both
signed spellings run the extended-sign witnesses, wide-reference random cases,
mask/alias tests and host-FP checks. Captures remain RDNA4-only.

Sources: AMD [RDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)
(downloaded from a [mirror](https://llm-tracker.info/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)
when AMD's endpoint returned 401), AMD RDNA4 ISA sections 12.5.2/16.15, and
rocjitsu `ffc144c564c` shared DS execution handlers. No new hardware captures.

Validation: all 705 tests pass with Clang and GCC. All five DS permutation tests
and all six wide-MAD tests pass under ASan/UBSan. The DS tests use an independent
destination-centric reference, explicit cross-half witnesses, byte-offset overflow,
collisions, masks, inactive reads, aliases and guard words. Dedicated DS SIMD
paths are not yet implemented; existing MAD SIMD implementations are unchanged.


## DS swizzle: all four encoding classes

Added shared Wave32/Wave64 DS_SWIZZLE_B32, including bit-mask, quad, masked
rotate and FFT modes. Basic selection follows rocjitsu; unlike its current
handler, inactive source lanes supply zero. Rotate and FFT follow the matching
pseudocode in both ISA manuals, with rotate arithmetic confined to each 32-lane
row. Source snapshots preserve in-place operations. No dedicated SIMD path yet.

The RDNA3 manual's masked-rotate examples disagree with its own pseudocode;
RDNA4's examples and pseudocode agree. This was checked on the local RX 9070
(gfx1201): all 2,048 rotate encodings, 32 FFT masks, 256 quad selectors and 32
bit-mode examples, each under eight EXEC masks in both wave sizes. All
1,818,624 output words match. This validates RDNA4, **not** RDNA3 hardware;
RDNA3 uses its explicit pseudocode pending hardware confirmation of the
inconsistent examples. No architectural difference was established that would
justify separate entry points.

`tests/capture_ds_swizzle.py` reproduces the captures, and
`tests/ds_swizzle_hardware.h` records word-wise hashes and raw capture SHA-256s.
Capture compiler: ROCm clang 23; Wave64 explicitly uses `-mwavefrontsize64`.
Additional tests cover all 65,536 offset encodings for both wave sizes against
an independent bit-by-bit/table reference, in-place routing, partial masks,
guard words, invalid modifiers and empty EXEC. Missing canonical API names: 63.

Validation: all 708 tests pass with Clang and GCC; all three DS_SWIZZLE tests
pass under ASan/UBSan. The final capture generator reproduces the Wave64 binary
capture byte-for-byte; regenerated Wave32 instruction bodies match the original
probe. Existing arithmetic/SIMD implementations were not changed.

## Half-register moves and swaps

Added shared `v_mov_b16` (Wave32, including DPP) and `v_swap_b16` (Wave32/Wave64).
MOV applies ABS then NEG directly to the selected half's sign bit. It preserves
NaN payloads and subnormals without FP conversion. RX 9070 raw-encoding probes
cover all ABS/NEG/OMOD/CLAMP combinations and half selections: ABS/NEG work,
while OMOD/CLAMP are ignored. LLVM rejects these modifier spellings even though
the hardware supports the input bits, and rocjitsu currently omits those input
modifiers. The implementation follows the captured behavior and the manual's
statement that MOV supports negation/absolute value. RDNA3 shares that documented
behavior, but the new capture is RDNA4-only.

SWAP accepts only HIGH_A/HIGH_D; both operands are read/write. Selected halves
use original values, and each write preserves the other half, including the
first write when exchanging halves of one VGPR. Tests cover all half selections,
whole-register aliases, both wave widths, masks, guard words, and invalid flags.
MOV tests also exhaust all 65,536 half bit patterns with every sign/half choice,
replay 4,096 captured output words, and cover DPP source filtering and aliases.
`capture_half_move.py` reproduces the raw VOP3 probe; the fixture includes its
SHA-256 and word-wise digest. The regenerated probe source matches the captured
source byte-for-byte.

Wave64 MOV, dedicated half-move/swap SIMD paths, and RDNA3 hardware validation
remain open. Missing canonical API names: 61.

Validation: all 712 tests pass on Clang/GCC; all ten tests in the move/swap
binaries pass under ASan/UBSan. Existing SIMD implementations are unchanged.

## Guest MODE updates and GETPC

Added shared S_ROUND_MODE/S_DENORM_MODE with an optional trailing `mode`
register pointer. Both manuals' scalar-control table specifies immediate[3:0];
the MODE layout puts round control at [3:0] and denormal control at [7:4].
The implementation updates only those bits in the supplied guest register,
never the host environment. Null opts out after validation. Both loose and
exact modes perform the inexpensive update when requested. The rocjitsu
handlers at the audited revision are empty, so no behavior was borrowed from
those stubs.

S_GETPC_B64 now returns PC+4 in the ordinary scalar destination, following both
manuals and rocjitsu's next-instruction address calculation. It has no EXEC
parameter and no implicit PC output because it does not write PC. Integer
addition and the existing scalar-move implementation preserve host FP state.

Tests exhaust all 65,536 immediates in both semantics modes, check preservation
of unrelated bits, null opt-out, validation and error preservation, host rounding,
sticky exceptions and x86 MXCSR. GETPC covers ordinary addresses, carry/wrap,
and input/output aliasing. No new GPU capture was required for these integer
register transformations. Missing canonical API names: 58.

Validation: all 716 tests pass with Clang/GCC; all four guest-state tests pass
under ASan/UBSan. Public standalone-header and exhaustive instruction-flag checks
include the new signatures. Existing numerical/SIMD implementations are unchanged.

## RDNA3 floating min/max spellings

Added the 18 RDNA3 binary, ternary and packed floating min/max/median names
listed as aliases of the RDNA4 `*_num_*` names in the machine-readable ISA.
Thin entry points forward to the existing implementations, including their
CPU dispatch, modifiers, DPP support, masked stores and operand alias handling.
No numerical algorithm or SIMD loop was duplicated or changed.

These paths remain loose-only. The RDNA3 manual distinguishes IEEE-mode
signaling-NaN propagation from default number selection; the new names do not
claim to implement that exact behavior. Strict exact requests fail, and exact
requests for exception reporting return `GOC_ERROR_UNSUPPORTED_GLOBAL_STATE`.
Loose calls leave the optional exception register untouched. No new RDNA3
hardware observations are claimed. Wave64 and exact variants remain open.

New tests exercise independently specified finite results, signed zeros,
NaNs, infinities and subnormals, all available CPU dispatch levels, masks,
binary destination aliasing, ternary operand order and modifiers, and unchanged
outputs on exact/reporting errors. Existing numerical and instruction-flag
suites continue to exercise the shared implementations. The inventory now has
769 canonical in-scope names with APIs and 40 absent; names alone do not establish
complete architectural coverage.

Validation: all 720 tests pass with Clang and GCC; all four new tests pass under
ASan/UBSan. Public standalone-header checks and exhaustive instruction-flag
checks include the new entry points.

## Scalar relative-register moves

Implemented S_MOVRELS_B32/B64, S_MOVRELD_B32/B64 and S_MOVRELSD_2_B32,
shared by RDNA3 and RDNA4. These are raw integer transfers in both semantics,
independent of EXEC and the host FP environment. B64 uses two consecutive
32-bit registers, with even base/index preconditions from the ISA.

Both manuals' S_MOVREL pseudocode adds DWORD offsets without scaling; RELSD_2
extracts M0[9:0] and M0[25:16]. The audited rocjitsu revision instead scales B64
offsets and extracts different fields, so those particular handlers were not
copied. A new RX 9070/gfx1201 probe covers all five instructions at five offsets
and confirms the manual interpretation. Its 450 captured register words and
SHA-256 are recorded in `tests/scalar_relative_hardware.h`; the reproducible
probe generator is `tests/capture_scalar_relative.py`. The probe saves/restores
M0 and SGPR0/1; M0 is the only reserved register it temporarily uses. No RDNA3 hardware capture
is claimed.

Indexed arrays map SGPR0..107 (including VCC) and TTMP0..15 at indices108..123.
The manuals' region rule forbids indexing across these ranges: out-of-range
reads redirect to SGPR0; writes are discarded. Explicit counts additionally
bound available storage. Destination counts must exclude TTMPs outside the
trap handler. Non-indexed scalar inputs are passed by value and non-indexed
outputs are ordinary destination pointers, allowing the caller to decode any
valid scalar operand. Widened address arithmetic prevents host overflow from
wrapping a bad index into an accessible register. Boundary behavior follows
both manuals and is separately tested; the GPU capture covers ordinary
SGPR indexing plus offset 256, confirming fallback to SGPR0/1 on reads and
discarded writes without truncating the M0 offset.

Tests cover region/count boundaries, full-width M0, split-field reserved bits,
source/destination overlap, zero-sized arrays, and flag validation before
operand access. Canonical name gaps are now 35 (774 present of 809 in scope).

Validation: all 724 tests pass with Clang/GCC; all four new scalar-relative tests
pass under ASan/UBSan. Header standalone checks and exhaustive instruction-flag
checks cover the new APIs. Existing SIMD code is unchanged.


## Relative vector swaps

Added shared V_SWAPREL_B32 for Wave32 and Wave64. Both operands are writable
register-file views. M0[9:0] and M0[25:16] select source and destination offsets;
reserved bits are ignored. Both manuals specify that an invalid operand cancels
the entire instruction, since both operands are destinations. The implementation
uses the existing swap loop and the same bounded relative-index policy as the
MOVREL family, with widened sums and flag validation before pointer arithmetic.
It copies raw bits exactly in both semantics; EXEC controls both stores.

rocjitsu's RDNA3/RDNA4 handlers agree on the valid-index exchange and on cancelling
both writes for an invalid operand. New RX 9070/gfx1201 Wave32 and Wave64 captures
also confirm valid offsets, identical resolved registers, zero/sparse/full EXEC,
reserved M0 bits and independent source/destination offset-256 cancellation.
The 18,432 captured words are represented by per-case word-wise digests in
`tests/relative_swap_hardware.h`, with raw SHA-256 fingerprints and reproducible
`tests/capture_relative_swap.py` probes. No RDNA3 GPU capture is claimed.

CPU tests additionally cover noncontiguous/reversed register views, overlapping
storage, guard words, upper Wave64 lanes, empty allocations, widened-index
overflow and all invalid instruction-flag bits. Missing canonical names: 34;
775 of 809 in-scope names have APIs, without implying complete semantic coverage.

Validation: all 727 tests pass on Clang/GCC, and all eight swap-family tests pass
under ASan/UBSan. Existing SIMD implementations are unchanged.


## RDNA3 floating constant comparisons

Added all twelve Wave32 CMP/CMPX false/true FP16/FP32/FP64 entry points. Results
follow rocjitsu's constant predicates: false returns zero, true returns EXEC
filtered by DPP16 row/bank/boundary rules. CMPX returns the new EXEC mask as its
ordinary scalar destination. FP16/FP32 accept DPP8/DPP16; FP64 rejects DPP.
ABS/NEG, half selectors, CLAMP and supported flushing settings cannot change
the constant result. Existing integer and new floating forms share the short
predicate/mask implementation; no SIMD numerical loop is necessary.

The RDNA3 manual explicitly says these predicates still signal invalid for
sNaNs and, with CLAMP, qNaNs. rocjitsu's shared result handlers do not model
those exceptions. The existing GoC comparison exception classifier is backed
by RDNA4 captures, which do not establish the behavior of these RDNA3-only
opcodes. Accordingly result masks are exact, but non-null reporting in
non-loose semantics returns `GOC_ERROR_UNSUPPORTED_GLOBAL_STATE` before any
output write. Loose mode leaves the optional register unchanged. No exception
support or new GPU observation is claimed for this slice.

The input VGPR pointers may be null because successful calls need only EXEC
and the modifiers. Tests cover all twelve entry points, all available CPU
levels, strict exact and loose semantics, EXEC/DPP filtering, accepted and
rejected modifiers, unchanged optional state in loose mode, and preservation
of both outputs on reporting errors. The generic reporting-contract test now
includes these APIs instead of treating them as implemented classifiers.

There are now 787 in-scope canonical names with APIs and 22 absent. The missing
names are VOPD operations and general scalar hardware-register access; Wave64,
exception models, and broader semantic audits remain separate open work.

Validation: all 729 tests pass with Clang/GCC. All eight constant-comparison and
exception-register contract tests pass under ASan/UBSan. Public standalone
headers and exhaustive high instruction-flag checks include the new entry points.


## Wave64 register moves and relative moves

Added Wave64 MOV_B32, MOV_B16 and all four relative MOVREL forms. All currently
accepted DPP controls stay within 16-lane rows. The wider moves therefore reuse
the existing Wave32 implementation for each half, remapping DPP16 row-enable
bits for rows 2/3. This preserves the existing SIMD source-routing paths and
keeps half/sign selection, modifier validation and raw-bit behavior in one place.
Validation runs before operand access or either half's stores. Whole-VGPR aliases
are safe because supported controls cannot read across the 32-lane boundary.
The relative-index calculation and allocation/offset checks are shared between
both wave sizes; no second indexing algorithm was introduced.

Independent routing references now accept 64-bit EXEC and directly calculate
all 64 source/destination lanes. Tests cover every row-enable combination,
representative controls from each accepted DPP class, bank filtering, FI/BOUND,
upper-lane and alternating EXEC, in-place routing, half/sign modifiers, guard
words, relative indexing and invalid descriptors. Existing Wave32 reference
users remain covered by the full suite.

The raw MOV_B16 probe now supports Wave64. Its new 8,192 RX 9070/gfx1201 outputs
match the bit-transfer/sign-modifier model, including NaNs and subnormals.
The SHA-256 and word-wise digest are in `tests/half_move_hardware.h`; the existing
hardware-corpus test now checks both wave sizes. This capture verifies raw
half-move modifiers, not Wave64 DPP routing or RDNA3 hardware behavior.

Six Wave64 APIs were added. The canonical-name count remains 787 present and
22 absent, illustrating why that count alone is not a completion measure.

Validation: all 732 tests pass on Clang/GCC, and all eleven move/relative-move
tests pass under ASan/UBSan. Standalone-header and exhaustive instruction-flag
checks include all six new entry points.


## Wave64 integer relational comparisons

Added the 72 Wave64 signed/unsigned LT/EQ/LE/GT/NE/GE CMP/CMPX entry points
for 16-, 32- and 64-bit elements. Each returns a native 64-bit scalar mask;
inactive bits are zero. A shared implementation invokes the existing comparison
kernel for both halves, remaps DPP16's upper row enables, and publishes the
combined mask only after all source reads. This last ordering is necessary when
the scalar output overlaps an upper-half source. A byte-copy publishes the mask
without introducing a typed 64-bit aliasing access to the VGPR words.

The scalar ordering logic borrowed from rocjitsu and the existing x86-64-v3/v4
kernels are reused unchanged. FP16-style half selectors apply to 16-bit integer
operands; 16/32-bit comparisons retain DPP8/DPP16, and 64-bit element comparisons
continue to reject DPP. Flags are validated before accessing any source or
writing the real output, including empty EXEC. Both semantics are exact.

Tests independently compute full 64-lane masks for all predicates, signedness,
element widths, selectors, edge values and random patterns at every available
CPU level. They cover all four upper/lower row enables, DPP filtering, empty and
upper-lane EXEC, masks overlapping each input register, guard words and error
preservation. Existing host rounding/sticky-flag/MXCSR checks now exercise both
wave sizes. No new GPU comparison capture is claimed; the existing Wave32
hardware corpus remains unchanged. Wave64 constant and floating comparisons
remain open. Canonical name counts remain 787 present and 22 absent.

The high-flag audit also exposed an existing cross-width limitation: Wave64
MBCNT_LO/HI reject DPP although Wave32 supports it. This was open at that checkpoint (resolved below); tests
must not infer modifier parity merely by stripping every `_wave64` suffix.
The shared-name check is restricted to the audited comparison/move families.

Validation: all 735 tests pass with Clang/GCC. The three new Wave64 comparison
tests, the combined-width host-FP-state test and exhaustive instruction-flag
test all pass under ASan/UBSan. Existing numerical SIMD kernels are unchanged.


## Wave64 MBCNT DPP and exact bit counts

Closed the Wave64 MBCNT_LO/HI DPP gap. The implementation stages all 64 source
lanes using existing DPP/SIMD routing, including upper row enables, then runs the
existing 64-lane counting kernel. It deliberately does not split the arithmetic
into Wave32 calls: MBCNT's prefix mask depends on the physical destination lane,
not the permuted source lane or lane number modulo 32. Whole-register aliases
remain safe, and invalid flags are rejected before source access.

The bit-count family (CLZ, CTZ, CLS, BCNT and MBCNT) now accepts strict exact
semantics. These kernels use integer bit scans/population counts and wrapping
addition, without approximation. The underlying scalar and SIMD numerical
implementations were not changed. Existing exhaustive half-word/bit-pair tests
and the Wave32 DPP GPU corpus now explicitly request strict exact semantics.

A new RX 9070/gfx1201 Wave64 capture contains 3,584 MBCNT outputs: both
instructions, seven DPP8/DPP16 descriptors and four EXEC masks, covering
FI/BOUND, row/bank selection, upper-lane behavior and accumulator wrap.
`tests/capture_mbcnt_dpp.py` reproduces the probe; per-case digests and the raw
SHA-256 are in `tests/mbcnt_dpp_hardware.h`. Every available CPU path matches.
Independent tests additionally cover all sixteen row-enable masks and aliases
with A or B. No RDNA3 hardware observation is claimed.

No canonical names or new entry points were added. Wave64 CLZ/CTZ/CLS/BCNT APIs
remain open; this slice completes the DPP support of the already existing
Wave64 MBCNT APIs and enables exact semantics for the existing bit-count family.

Validation: all 737 tests pass on Clang/GCC. All fourteen bit-count, integer DPP
and instruction-flag tests pass under ASan/UBSan. Both existing Wave32 and new
Wave64 DPP captures pass in strict exact semantics for the bit-count operations.


## Remaining Wave64 bit-count entry points

Added Wave64 CLZ, CTZ, CLS and BCNT, completing both wave sizes for the currently
exposed bit-count family. The public entry points instantiate the existing
64-lane executor and x86-64-v3/v4 templates; no numerical or routing algorithm
was duplicated. Both semantics are exact, DPP8/DPP16 are supported, and inactive
lanes/unselected storage are preserved. The existing sentinel and wrapping rules
are unchanged.

Tests cover all 65,536 half-word patterns placed low, high and repeated into both
halves of each input word, including zero/all-one sentinels. They check every
available CPU path against the independent bit-by-bit reference, along with
DPP, upper-lane masks, aliases with A or B, guards, invalid descriptors and host
rounding/exception/MXCSR preservation. The existing Wave32 bit-count and Wave64
MBCNT hardware corpora still cover the shared numerical/routing components;
no new direct GPU captures of these four Wave64 opcodes are claimed.

Four APIs were added. Canonical name counts remain 787 present and 22 absent.

Validation: all 740 tests pass on Clang/GCC. All thirteen bit-count and
instruction-flag tests pass under ASan/UBSan. Existing numerical/routing bodies
are unchanged; the new SIMD specializations vary only the lane count.


## VOPD decoded operations

Added the nineteen canonical VOPD operation names (21 APIs including the two
RDNA4 DOT2ACC variants). Each forwards to the corresponding ordinary operation;
DOT2ACC supplies the old destination as the addend. Existing scalar/SIMD dispatch
and exact models are reused without a second arithmetic implementation.

Section 7.8 of the RDNA4 ISA permits Y to overwrite an X source, but forbids X
from overwriting a Y source. Consequently, legal pairs can execute X then Y.
The caller validates slot, bank, destination parity and literal restrictions and
resolves encoded operands, including scalar DOT2 replication and out-of-range
VGPR addressing. VOPD is Wave32-only and encodes no arithmetic modifiers or DPP;
all nonzero instruction flags are rejected. LLVM's gfx1201 assembler confirms
that the ordinary ADD form is accepted and a negated VOPD operand is rejected.

The RDNA3 and RDNA4 rocjitsu VOPD handlers select different GFX11/GFX12 DOT2
models. Both FP16 and BF16 therefore have unsuffixed RDNA3 and `_rdna4` APIs.
Tests reuse the borrowed GFX1100 ordinary DOT2 captures and check an explicit
architecture-difference witness. These are not new VOPD hardware captures.
Additional tests independently check each operation's finite result, masks,
accumulator/literal/condition operands, every instruction-flag bit, state
preservation on errors, all available CPU levels and a legal reverse dependency.
Exact support and optional exception reporting retain the ordinary APIs' limits.

Canonical-name coverage is now 806 present and three absent (general scalar
GETREG/SETREG). This is not semantic completeness: the family checklist and
cross-cutting gates above remain open.

Validation: all 744 tests pass with Clang 21 and GCC 15 Release builds. All ten
VOPD, exception-register and instruction-flag tests pass under ASan/UBSan.


## Hardware-register access audit (implementation still open)

See [the detailed HWREG audit](HWREG_AUDIT.md) for the complete XML register-ID
inventory, privilege differences, snapshot-read side effects, clock-input needs
and implementation gates. The three APIs remain missing.

The three remaining canonical names, `s_getreg_b32`, `s_setreg_b32` and
`s_setreg_imm32_b32`, cannot be completed by adding a generic bitfield helper.
RDNA3 ISA section 3.4 and SOPK definitions specify a six-bit register ID,
five-bit bit offset and five-bit size-minus-one. SETREG also applies
register-specific writable bits and privilege and can trigger side effects.
The printed SETREG pseudocode's mask does not include the offset shift; it
must not be copied without checking hardware or the independent model.

rocjitsu `vm/amdgpu/hwreg.cpp` provides useful field decoding/insertion and
architecture tables, but many entries explicitly return Unsupported. Its RDNA3
STATUS descriptor is read-only even though the manual allows some privileged
writes. It is evidence for partial behavior, not an exhaustive reference.

| Register ID | RDNA3 | RDNA4 |
|---|---|---|
| 1 | MODE | WAVE_MODE |
| 2 | STATUS (includes privileged writable fields) | WAVE_STATUS (separate privileged state) |
| 3 / 4 | TRAPSTS at 3 | WAVE_STATE_PRIV at 4 |
| 17 | SHADER_TBA_HI | WAVE_EXCP_FLAG_PRIV |
| 18 | PERF_SNAPSHOT_PC_LO | WAVE_EXCP_FLAG_USER |
| 19 | PERF_SNAPSHOT_PC_HI | WAVE_TRAP_CTRL |
| 20 / 21 | SHADER_FLAT_SCRATCH_LO/HI | WAVE_SCRATCH_BASE_LO/HI |
| 29 / 30 | SHADER_CYCLES at 29 | SHADER_CYCLES_LO/HI |

A raw-encoding API therefore needs `_rdna4` variants. Implementing it requires
explicit input snapshots for readable registers and one trailing optional output
per writable architectural register, together with the existing exact/loose
state contract. Scheduler/cache actions such as FLUSH_IB must be separated from
register arithmetic rather than silently fabricated. Architectural register
write masks, privilege-dependent writes and side effects remain unchecked work.
This does not change the family's in-scope status or claim those three APIs.

## Wave64 constant comparisons

Added the 28 Wave64 F/T integer and floating CMP/CMPX entry points. One shared
mask-width template retains the Wave32 validation and result rules. DPP uses
independent 16-lane rows; the Wave64 implementation remaps the row-enable bits
for each half and combines both filtered masks before writing the result.
No operand loads or floating arithmetic are needed for these predicates.

Tests independently check both semantics, every available CPU level, masks
including lane 63, DPP routing/filtering, modifiers, and error preservation.
Floating exact non-null exception reporting still fails explicitly; loose mode
leaves the supplied state unchanged. No new hardware captures are claimed.
Canonical-name counts remain 806 present and three absent.

Validation: all 747 tests pass with Clang 21 and GCC 15 Release. All twelve
constant-comparison, exception-register and instruction-flag tests pass under
ASan/UBSan. No existing arithmetic or SIMD kernel changed.


## Wave64 floating relational comparisons

Added all 84 relational FP16/FP32/FP64 CMP/CMPX Wave64 APIs. The existing
32-lane numerical and x86-64-v3/v4 kernels remain unchanged. A Wave64 executor
validates once before reading operands, remaps DPP row enables for each half,
and stages both comparison masks and accumulated exceptions before writing
outputs. Exact non-null reporting selects the existing integer classifier;
loose reporting stays off the SIMD path's arithmetic and leaves state unchanged.

Independent tests cover predicates, signed zeros/NaNs/denormals and other edge
inputs, all sign/half selectors, flushing, every DPP row-enable mask, lane 63,
and a result overlapping the upper source half. Exception tests place the
existing RX 9070 captured operand pairs in lanes 0, 31, 32 and 63 and check masked
accumulation, signaling comparisons, input flushing and loose opt-out behavior.
These reuse the Wave32 numerical observations; no new Wave64 GPU capture or
RDNA3 floating exception validation is claimed. CLASS predicates and the
cross-architecture floating-policy audit remain open.

Validation: all 750 tests pass with Clang 21 and GCC 15 Release builds. All nine
Wave64 floating-comparison, exception-register and instruction-flag tests pass
under ASan/UBSan. Canonical counts remain 806 present and three absent.


## Wave64 CLASS and explicit width backlog

Added the six FP16/FP32/FP64 CLASS CMP/CMPX Wave64 APIs. Each reuses the existing
scalar and x86-64-v3/v4 classification kernels, with row-enable remapping for
DPP and delayed 64-bit result stores. Raw denormal classification is independent
of guest flushing flags. CLASS does not generate floating exceptions.

Tests exhaust every FP16 encoding against each of the ten class bits on every
available CPU path. Further tests cover both semantics, all three formats,
sign/half modifiers, all sixteen DPP row masks, lane 63, result/source overlap,
invalid calls and zero EXEC. All four host rounding modes and preexisting sticky
flags are preserved. Existing Wave32 classification captures continue testing
the shared kernel; no new Wave64 hardware observations are claimed.

The generated inventory now separately lists APIs taking a 32-bit EXEC mask
without a Wave64 counterpart, grouped by implementation file. There are 336
candidates after this slice, excluding architecturally Wave32-only VOPD.
This is a review queue, not a legality claim: each encoding's width must be
checked, then implemented or explicitly restricted. Scalar/pseudo-scalar/matrix
APIs are outside this signature screen and still require their own audit.
The three absent canonical names therefore substantially understate remaining
coverage work. All currently exposed comparison families now have both widths,
but the cross-architecture semantics/reporting audit remains open.

Validation: all 752 tests pass with Clang 21 and GCC 15 Release. All ten CLASS
and instruction-flag tests pass under ASan/UBSan. Sanitizer validation caught
and verified the fix for an initial FP64 adapter overread: the class-selection
operand occupies one VGPR even though the classified value occupies two.


## Wave64 Boolean operations and conditional selects

Added nine Boolean and two CNDMASK Wave64 entry points. They reuse the existing
portable and x86-64-v4 kernels, remapping DPP row-enable bits per half; CNDMASK
also consumes the corresponding half of its 64-bit condition. Whole-VGPR source
aliases remain safe because the supported DPP controls do not cross 16-lane
rows. Inactive lanes and the unwritten half of 16-bit destinations are preserved.
Existing v3 policy is unchanged: prior measurements favored the portable paths.

These operations are pure raw-bit logic, so strict exact semantics is now
accepted at both widths. No arithmetic or SIMD kernel changed. The former
unsupported-exact validation tests now exercise a genuinely reserved semantics
value instead. New independent bit-by-bit tests check all eleven operations,
both semantics, all CPU paths, all half/sign modifiers, DPP8/DPP16, all row
enables, different upper/lower condition bits, whole-source aliases, masks and
guards. Existing Wave32 hardware corpora continue to exercise shared kernels;
no new Wave64 capture or throughput measurement is claimed.

The wave-width inventory now has 325 unresolved candidates. Canonical-name
counts are unchanged at 806 present and three absent.

Validation: all 755 tests pass on Clang 21 and GCC 15 Release builds. All 24
Boolean, conditional-selection, integer DPP and instruction-flag tests pass
under ASan/UBSan, including the existing exhaustive routing suites.


## Wave64 carry/borrow arithmetic

Added the six ADD/SUB/SUBREV CO and CO_CI Wave64 APIs, with 64-bit EXEC,
input-carry and scalar output masks. Both semantics reuse the existing scalar,
x86-64-v3 and x86-64-v4 kernels. DPP row enables are remapped per half; SUBREV
continues to route B, while ADD/SUB route A. Carry-in bits retain their original
lanes. The scalar output is staged until after both halves, preserving source
aliasing, and zero EXEC writes a zero carry mask without touching VGPR pointers.

Tests independently evaluate all boundary operand pairs, carry/borrow patterns
with different upper/lower halves, saturation, both semantics and every CPU
path. They cover DPP8/DPP16 and all row masks, whole-VGPR aliases, scalar carry
output overlapping the upper source half, guards, error preservation and null
VGPRs at zero EXEC. Existing Wave32 hardware captures still validate the shared
arithmetic; no new direct Wave64 GPU capture or performance measurement is
claimed. The width-review queue now has 319 candidates.

Validation: all 758 tests pass with Clang 21 and GCC 15 Release builds. All
twelve carry/borrow and instruction-flag tests pass under ASan/UBSan.


## Wave64 shifts and bitfields

Added six reverse-shift and eight bitfield APIs. The 64-bit shift adapter takes
one count VGPR and two value/destination VGPRs, and rejects DPP. The 32-bit
shifts and all bitfield operations retain DPP on A with four Wave64 row-enable
bits. Each executes the existing kernels for both halves; whole-VGPR aliases
are safe because DPP stays within 16-lane rows. Scalar/v3/v4 dispatch policies
and numerical kernels are unchanged, including BFI's portable fallback on v3.

Strict exact semantics is now accepted at both widths: these operations use
integer bit manipulation only. Tests check shifts at every bit position and
count, sign extension, wrapped counts, bitfield offsets/widths across bit 31,
byte selectors, DPP8/DPP16, all row masks, both semantics, every CPU path,
whole-source aliases, guards and invalid/zero-EXEC calls. Existing Wave32
hardware captures continue to validate shared kernels; no new Wave64 capture
or throughput measurement is claimed. The width-review queue now contains
305 candidates; canonical-name counts remain 806 present and three absent.

Validation: all 761 tests pass on Clang 21 and GCC 15 Release. All 23 shift,
bitfield, Wave64, DPP and instruction-flag tests pass under ASan/UBSan.


## Wave64 non-carry integer addition and multiplication

Added six ADD/SUB/SUBREV/ADD3 and seven MUL Wave64 APIs, reusing existing
portable, x86-64-v3 and x86-64-v4 kernels and dispatch policy. Ordinary 32-bit
multiply still rejects DPP; 24-bit multiply supports it, and only low-product
24-bit forms accept saturation. ADD3 still wraps and rejects CLAMP; SUBREV's
DPP source remains B. Whole-VGPR aliases are preserved with row-local DPP.

Both wave sizes now accept strict exact semantics for these integer kernels.
Tests independently cover signed/unsigned 24/32-bit boundary pairs, random
words with discarded upper bytes, high/low products, saturation, ADD3 overflow,
all CPU paths, both semantics, DPP8/DPP16 and every row mask, different masks
across halves, whole-source aliases, guards and validation before operand
access. Existing Wave32 captures continue validating shared numerical/routing
logic; no new Wave64 GPU capture or throughput measurement is claimed.
The width-review queue now has 292 candidates.

Validation: all 763 tests pass with Clang 21 and GCC 15 Release. All nineteen
integer arithmetic, Wave64, DPP, instruction-flag and zero-EXEC tests pass under
ASan/UBSan. The zero-EXEC support table now records the expanded exact support.


## Wave64 integer selection and ternary operations

Added fourteen signed/unsigned min/max/median and eight ternary Wave64 APIs.
Both widths now accept strict exact semantics; existing numerical and SIMD
kernels are unchanged. Binary min/max does not access C. DPP routes A within
16-lane rows and retains all four row enables. Mixed min/max still combines
A/B before C, and byte LERP uses independent byte sums and rounding bits.

Independent tests cover all triples from signed/unsigned boundary sets, random
words, wrapping shifts/addition, byte rounding, all CPU paths, both semantics,
DPP8/DPP16 with every row-enable mask, different EXEC patterns across halves,
all whole-source destination aliases and guards. Validation checks zero EXEC
with null pointers and all unsupported low instruction-flag bits. Existing
Wave32 hardware captures remain the direct hardware evidence for the shared
kernels; no new Wave64 captures or throughput measurements are claimed.
The width-review queue now contains 270 candidates.

Validation: all 765 tests pass with Clang 21 and GCC 15 Release. All 24
selection, ternary, Wave64, DPP, zero-EXEC and instruction-flag tests pass under
ASan/UBSan. The zero-EXEC table reflects the expanded exact support.


## Wave64 ordinary and packed 16-bit integer arithmetic

Added 34 APIs: twelve ordinary binary/shift operations, eight ordinary ternary
operations, twelve packed binary/shift operations and two packed MAD forms.
The existing shared scalar/v3 kernels serve both widths. Ordinary forms preserve
the unselected D half and permit DPP on A before half selection; packed forms
compute both halves from original sources and reject DPP. Both widths now
accept strict exact semantics for this integer arithmetic.

New independent tests cover every supported half-selector and saturation
combination, signed/unsigned boundaries and random words, both semantics and
all CPU paths, whole-source aliases, masks/guards, and all ordinary DPP row
masks plus DPP8. They check packed DPP rejection and validation before operand
access. Existing exhaustive Wave32 operand tests and hardware captures remain
in place for the shared kernels; no new Wave64 captures or throughput numbers
are claimed. The width-review queue is now 236 candidates.

Validation: all 767 tests pass on Clang 21 and GCC 15 Release. All 31 ordinary
and packed integer16, Wave64, DPP, zero-EXEC and instruction-flag tests pass
under ASan/UBSan, including both exhaustive ordinary DPP suites.


## Wave64 narrow-factor and wide-result integer MAD

Added four 16/24-bit-factor MAD and four wide-result MAD APIs (including RDNA3
spellings). Narrow-factor forms now accept strict exact semantics at both
widths and reuse existing scalar/v3 kernels, DPP and selectors. Wide forms
reuse scalar/v3/v4 kernels, require one VGPR per factor and two for C/D, reject
DPP and produce a 64-bit carry mask. Their signed carry is the extended sum's
bit 64 before saturation. The scalar output is committed after all source
reads and vector writes, including zero for zero EXEC.

Independent tests cover boundary factors/addends, saturation, both semantics,
every CPU path, DPP row masks and selectors where supported, differing EXEC
halves, whole-source aliases, guards and invalid/empty calls. Wide results and
carry use the independent 128-bit reference, including negative sums and signed
overflow. A scalar-output alias in the upper source half checks delayed carry
stores against separately captured vector results. Existing Wave32 hardware
corpora remain the direct numerical evidence; no new Wave64 capture or speedup
claim is made. The width-review queue is now 228 candidates.

Validation: all 770 tests pass on Clang 21 and GCC 15 Release. All twenty
integer/wide MAD, Wave64, DPP, zero-EXEC and instruction-flag tests pass under
ASan/UBSan.

## Wave64 SAD and exact integer semantics

Added all eight SAD/MSAD/QSAD/MQSAD Wave64 APIs. Both wave widths now accept
strict exact semantics, using the existing integer arithmetic and scalar/v3
kernels. Ordinary forms route A through DPP; quad forms reject DPP, read two
A VGPRs and one B VGPR, and produce either two packed or four full-width
accumulator VGPRs. The Wave64 wrapper remaps row enables for each half without
changing the arithmetic kernels or adding sparse-mask paths.

Independent reference tests cover sliding byte windows, unsigned extremes,
zero-byte suppression, independent packed/full-width saturation, all CPU paths
and both semantics, upper/lower EXEC bits, DPP row masks, source/output aliases
and guard words. Invalid and zero-EXEC calls are tested without operands.
Existing Wave32 exhaustive arithmetic tests and the RX 9070 DPP capture remain
numerical evidence for the shared kernels; no new Wave64 hardware capture or
throughput measurement is claimed. The width-review queue is now 220 candidates.

Validation: all 772 tests pass on Clang 21 and GCC 15 Release. All thirteen
SAD, Wave64 SAD and instruction-flag tests pass under ASan/UBSan.

## Wave64 integer DOT4/DOT8

Added the four signed/mixed and unsigned byte/nibble dot-product Wave64 APIs.
RDNA3 and RDNA4 use the same packed operands and signedness controls; the
entry points remain architecture-neutral. Each operand is one VGPR, DPP is
rejected, and all modifier combinations reuse the existing scalar/v3 kernels.
Saturation remains after the whole dot plus accumulator, without intermediate
byte-pair saturation. Both widths support exact integer semantics.

The Wave64 tests use the independent radix-based reference for every
signedness/CLAMP combination, both semantics and all CPU paths, boundary and
random inputs, upper/lower masks, aliases, and guard words. Zero-EXEC and invalid
calls require no operand access. Existing cancellation and host-environment
tests continue to exercise the shared kernels. No new hardware capture or
performance measurement is claimed. The width-review queue is now 216 APIs.

Validation: all 773 tests pass on Clang 21 and GCC 15 Release. All five
integer DOT and instruction-flag tests pass under ASan/UBSan.

## Wave64 saturation and FP16 packing

Added Wave64 SAT_PK_U8_I16 and PACK_B32_F16, sharing their scalar/v3/v4
kernels and DPP routing. Both widths now accept strict exact result semantics.
SAT preserves the unselected destination half; PACK preserves payloads and
subnormals while quieting signaling NaNs. Exact PACK with non-null optional
exception output remains explicitly unsupported; loose mode skips reporting.

The Wave64 exact tests reproduce the existing RX 9070 capture digests for all
65536 encodings and 66 modifier combinations, regrouped into 64-lane calls.
This validates result bits against existing captures, not new native Wave64
hardware execution. Independent DPP tests cover every row mask, all modifiers,
CPU paths, upper EXEC bits, aliases and guards. Empty/invalid calls and optional
state opt-out/rejection are covered. The width-review queue is now 214 APIs.

Validation: all 775 tests pass on Clang 21 and GCC 15 Release. All ten packing
and instruction-flag tests pass under ASan/UBSan.

## Wave64 integer widening and saturating conversion

Added all four signed/unsigned integer widening and packed narrowing Wave64
APIs, using the existing scalar/v4 kernels and v3 packed conversion paths.
Both widths now support strict exact integer semantics. The unary widening
forms read only A; packed forms read A/B. DPP routes A before half selection
or saturation and remaps all four row-enable bits across the two halves.

Independent tests exhaust all 65536 source halves and signed/unsigned
saturation boundaries, then test whole-word operands with DPP, every row mask,
upper/lower EXEC bits, source/destination aliases and guard words on every CPU
path. Empty and invalid calls are checked before operand access. Existing
Wave32 hardware/DPP and host-FP-state tests remain in place. No new native
Wave64 capture or performance claim is made. The width-review queue is 210 APIs.

Validation: all 776 tests pass on Clang 21 and GCC 15 Release. All ten integer
conversion and instruction-flag tests pass under ASan/UBSan.

## Wave64 FP32/FP64 exponent extraction

Added both FREXP_EXP_I32 Wave64 APIs with the existing scalar/v3 execution
paths. FP32 accepts DPP and remaps all row enables; FP64 rejects DPP and reads
two source VGPRs. Both write one integer-result VGPR. Loose optional state
remains unchanged; exact non-null reporting is rejected before operand access.

The exact-semantics gate is deliberately unchanged. FP64 SIMD normalizes
subnormals through host floating-point scaling and requires host denormal
inputs enabled. A guest/host denormal-state audit is still necessary before
claiming exact support. No production FP-environment mutation was added.

Independent integer-reference tests cover every exponent field, subnormal
leading bits, all modifiers and CPU levels, DPP row masks, upper/lower EXEC,
aliases with either source half, guards, empty/invalid calls and optional state.
The existing FP32 hardware corpus remains in place; this adds no new Wave64
hardware capture. The width-review queue is 208 APIs.

Validation: all 777 tests pass on Clang 21 and GCC 15 Release. All nine
exponent-extraction and instruction-flag tests pass under ASan/UBSan.

## Host-independent FP64 exponent extraction

Removed the FP64 host-denormal dependency identified above. Scalar extraction
now uses integer exponent fields and the existing leading-zero helper. AVX2
constructs 2^52 plus the integer significand and subtracts 2^52 exactly; it never
consumes a subnormal floating-point input. Normal and exceptional results keep
their previous rules. No host control-register writes or mask-density dispatch
were added. Exact guest semantics/reporting remain a separate open gate.

Tests cross every subnormal leading-bit position, boundary significands,
normals, signed zeros, infinities and signaling NaNs with all host rounding
modes and DAZ/FTZ combinations on every CPU path through Wave64. The results
match an independent integer reference and preserve the complete MXCSR and
preexisting exception flags. Existing Wave32 and DPP tests continue to pass.

A pinned thread-CPU-time microbenchmark used five samples of four million
Wave32 calls per case, with identical saved-before and rebuilt-after harnesses.
After the test workloads finished, median ns/wave were:

| Inputs | Path | Before | After |
|---|---|---:|---:|
| Normal | Scalar | 80.141 | 33.025 |
| Normal | AVX2 | 20.530 | 20.702 |
| Subnormal | Scalar | 98.198 | 40.705 |
| Subnormal | AVX2 | 33.294 | 20.699 |

These are local focused measurements, not a whole-library performance claim.
The normal SIMD difference is below 1%; subnormal SIMD avoids the previous
floating-point denormal cost. Logs and harness are in the build directory's
`coverage-work/frexp-speed-*` files.

Validation: all 778 tests pass on Clang 21 and GCC 15 Release. All nine
exponent-extraction tests pass under ASan/UBSan.

## Wave64 byte/nibble conversion and byte packing

Added six Wave64 APIs covering four unsigned-byte-to-FP32 selections,
signed-nibble offsets, and nearest-even FP32-to-byte packing. The five
integer-to-float forms now accept strict exact semantics at both widths: their
finite values and OMOD scaling are exactly representable. Byte packing retains
loose semantics and rejects exact non-null optional exception reporting.
All forms reuse their existing scalar/v3/v4 kernels and DPP routing. Packing
routes A only, keeping byte selector B and preserved word C in their own lanes.

Independent references test all byte/nibble values and output modifiers,
rounding ties, saturation, signed infinities/NaNs, byte selections, all DPP row
masks, upper/lower EXEC, every source alias and guard words on all CPU levels.
Loose optional state remains unchanged, and exact reporting rejects before
operand access. Existing hardware captures and exhaustive Wave32 tests remain
in place; no new Wave64 capture or throughput measurement is claimed.
The width-review queue is now 202 APIs.

Validation: all 780 tests pass on Clang 21 and GCC 15 Release. All twenty
byte conversion, byte packing and instruction-flag tests pass under ASan/UBSan.

## Wave64 FP8/BF8 widening and raw NaN correction

Added all four single/packed FP8/BF8 widening Wave64 APIs and enabled exact
semantics at both widths. The existing scalar/v3/v4 algorithms are shared;
single-result forms support DPP, packed forms reject it and write two VGPRs.
Destination aliasing retains the second-result-wins rule.

Review found that prior hardware tests canonicalized NaNs before comparing,
so they could not justify an exact NaN-bit claim. A new RX 9070 gfx1201 probe
captured all 256 codes for both single instructions and both packed result
pairs. Hardware maps every NaN, regardless of input sign, to `0xffc00000`.
The former sign-preserving conversion behavior was corrected in scalar and
SIMD kernels. The shared SIMD decoder has an explicit constant NaN-policy
parameter used by conversions; existing dot/matrix callers retain their prior
policy. This changes the NaN constants in the same SIMD instruction sequence,
without adding a reporting pass or changing finite arithmetic.

`tests/capture_fp8_widen.cpp` reproduces the 1536 raw captured result words.
The regression hashes raw words without NaN normalization and checks both
wave widths on all CPU paths; Wave64 regroups the Wave32 hardware corpus.
Additional independent tests exhaust all packed byte pairs, selectors,
upper/lower masks and aliases, including overlapping packed outputs, and
exercise every DPP row-enable combination. The width-review queue is 198 APIs.

Validation: all 782 tests pass on Clang 21 and GCC 15 Release. All twelve FP8
widening and instruction-flag tests pass under ASan/UBSan. The raw HIP probe
completed successfully on the local RX 9070.

## Wave64 FP8/BF8 nearest-even and stochastic narrowing

Added all four Wave64 narrowing APIs and enabled strict exact semantics at
both widths. They share the existing integer scalar/v3/v4 models, including
captured canonical NaNs, infinity behavior, finite overflow saturation and
stochastic significand alignment. Packed forms preserve the unselected half;
stochastic forms preserve three bytes and use B's seed without modification.
DPP routes only A and remaps row enables across both halves.

The existing raw 20-column hardware fixture now runs in strict exact mode at
Wave32 and through the new Wave64 calls. The Wave64 fixture repeats captured
per-lane inputs; it is not a new native Wave64 capture. Independent numeric
references additionally cover random FP32 words/seeds, every selector and
source modifier, both saturation modes, DPP row masks, upper/lower EXEC,
whole-source aliases and guard words. Invalid and empty calls avoid operand
access. No arithmetic kernels or SIMD dispatch decisions changed.
The width-review queue is 194 APIs.

Validation: all 784 tests pass on Clang 21 and GCC 15 Release. All twelve FP8
narrowing and instruction-flag tests pass under ASan/UBSan.

## Wave64 normalized conversions

Added all six signed/unsigned normalized conversions for FP32 packed, FP16
packed and FP16 unary forms. Wave64 shares the existing scalar/v3/v4 numeric
kernels and loose semantics. DPP routes A before modifiers/selection, B remains
in its own lane, and unary forms preserve the unselected destination half.
Only packed-half APIs have optional exception outputs; loose calls skip them
and exact non-null reporting rejects before operand access.

New tests replay the raw twenty-column RX 9070 fixture across 64 lanes and
compare random inputs against the independent integer significand reference.
They cross all modifiers, CPU levels, DPP row enables, masks with both EXEC
halves, source aliases, guards and optional-state behavior. Repeating a Wave32
hardware fixture across 64 lanes is not a new native Wave64 capture. Exact
semantics and guest FP-state behavior remain open; no SIMD kernel changed.
The width-review queue is 188 APIs.

Validation: all 786 tests pass on Clang 21 and GCC 15 Release. All fourteen
normalized-conversion and instruction-flag tests pass under ASan/UBSan.

## Wave64 packed FP32 conversion

Added Wave64 FP16 RTZ and signed/unsigned integer16 packed conversions. Each
uses one VGPR per operand, routes A through DPP before modifiers, and packs A/B
into the low/high destination halves. Existing scalar/v3/v4 rounding, overflow
and NaN behavior is unchanged, as are loose semantics and optional-state rules.
Exact non-null reporting still rejects before operand access.

Tests repeat the existing nine-column raw hardware fixture across 64 lanes,
including both FP16_OVFL settings, and independently compare every modifier
combination with random FP32 inputs, all DPP row masks, upper/lower EXEC,
source aliases and guard words. Loose non-null state remains unchanged;
invalid and empty calls are checked without operands. This reuses Wave32
hardware evidence rather than claiming a new Wave64 capture. Exact semantics
and faithful optional reporting remain open. The width-review queue is 185 APIs.

Validation: all 788 tests pass on Clang 21 and GCC 15 Release. All thirteen
packed-conversion and instruction-flag tests pass under ASan/UBSan.

## Wave64 FP32/integer conversion

Added all six FP32/integer conversion Wave64 APIs. Integer-to-float, truncation,
nearest and floor reuse the existing scalar/v3/v4 dispatch choices, including
the deliberate scalar fallback for truncating v3 calls. DPP routes A with all
four row enables. Loose semantics and optional-state behavior are unchanged;
only the two truncating float-to-integer APIs have optional exception outputs.

Independent integer-bit references check signed/unsigned rounding boundaries,
positive/negative ties, saturation, infinities and NaNs, all modifiers/CPU
levels, full and upper-only masks, all DPP row enables, random source words,
aliases and guards. Loose non-null state remains unchanged and exact reporting
rejects without operand access. Exact guest rounding/denormal semantics and
faithful exception reporting remain open. No new hardware capture or SIMD
kernel change is claimed. The width-review queue is 179 APIs.

Validation: all 790 tests pass on Clang 21 and GCC 15 Release. All twelve
FP32/integer-conversion and instruction-flag tests pass under ASan/UBSan.

## Wave64 FP64 conversions

Added all six FP64/integer/FP32 conversion Wave64 APIs. FP64 operands retain
their low/high VGPR layout; integer and FP32 operands use one VGPR. The wrapper
uses the correct source/destination arity for each direction, rejects DPP and
retains existing scalar/v3/v4 dispatch, loose semantics and optional-state rules.

Independent integer-bit references check rounding and saturation witnesses,
NaNs/infinities and subnormal boundaries across every modifier and CPU path.
A separate finite-input test crosses all sixteen destination-word layouts,
source aliases, full/upper-only/sparse EXEC masks and guards, including the
high-word-wins rule when output halves alias. Unused pointer slots are null.
Loose optional state remains unchanged; exact reporting and invalid flags
reject before operand access. No new exactness, capture or throughput claim
is made. The width-review queue is 173 APIs.

Validation: all 792 tests pass on Clang 21 and GCC 15 Release. All ten FP64
conversion and instruction-flag tests pass under ASan/UBSan, including the
existing exhaustive alias suite. The new references explicitly discard unused
upper storage words for 32-bit source operands.

## Wave64 FP16 conversions

Added all six FP16/integer16/FP32 conversion Wave64 APIs, preserving the
one-VGPR operand layout, source/destination half selectors, untouched destination
half and existing overflow/output-scaling rules. DPP routes A before selection
and modifiers. Existing scalar/v3/v4 kernels and loose semantics are unchanged.
All optional exception pointers retain loose opt-out behavior and exact
non-null rejection before operand access.

The tests repeat captured FP16 rounding/overflow witnesses across both wave
halves and compare random input words against independent references for every
modifier, half selector, CPU level and overflow mode. DPP row enables, masks
with upper EXEC bits, aliases, guard words and unselected destination halves
are checked. NaN comparisons retain the loose payload policy. Existing
exhaustive Wave32 encoding and DPP hardware tests remain in place; this is not
a new native Wave64 capture or an exactness claim. The width-review queue is
167 APIs.

Validation: all 794 tests pass on Clang 21 and GCC 15 Release. All fourteen
FP16 conversion and instruction-flag tests pass under ASan/UBSan.

## Wave64 row permutations

Added Wave64 APIs for PERMLANE16, PERMLANEX16 and both variable-selector
forms. The ISA defines X row pairing as 0/1 and 2/3, so each half reuses
the existing scalar/AVX2/AVX-512 kernel without changing its numerical or
inactive-source rules. Both semantics remain supported. Whole-register
source/index/destination aliases are safe because no source crosses halves.

Independent row/index references check all four operations, FI/BOUND_CTRL
combinations, both semantics, all CPU paths, upper/lower-only and sparse EXEC,
source/index aliases and guard words. Invalid flags reject before operand
access; zero EXEC accepts null operands. Existing Wave32 raw hardware and host
FP-state tests remain in place. This slice adds no native Wave64 capture or
new throughput claim. The width-review queue is now 163 APIs.

Validation: all 796 tests pass on Clang 21 and GCC 15 Release. All six
PERMLANE and instruction-flag tests pass under ASan/UBSan.

## Wave64 FP16 exponent operations and width-inventory correction

Added Wave64 LDEXP_F16 and FREXP_EXP_I16_F16 APIs using the existing scalar
and AVX2 kernels. Half selectors, signed exponents, overflow control, DPP row
filtering and whole-register aliases retain their Wave32 behavior. Loose
semantics and exact optional-state rejection are unchanged.

Independent references check exponent boundaries, FP16 special values, both
overflow modes, each DPP routing form and all sixteen row enables, source and
exponent aliases, upper-only/sparse EXEC, guards and preserved destination
halves. Loose exception outputs remain unchanged. Validation checks null
operands on zero EXEC and on rejected modes/reporting. Existing exhaustive
Wave32 encoding, exponent, hardware-hash and host-environment tests remain.

The inventory previously misclassified twelve scalar SAVEEXEC/WREXEC APIs
as missing Wave64 forms because their explicit input data is named exec_mask.
They already have distinct `_b32`/`_b64` mnemonics and implementations; they
belong in the scalar audit, not the vector-EXEC suffix screen. Correcting
that screen and adding these two vector APIs leaves 149 width candidates.
This is not a new exactness, native Wave64 capture or performance claim.

Validation: all 799 tests pass on Clang 21 and GCC 15 Release. All fifteen
FP16 exponent and instruction-flag tests pass under ASan/UBSan. Inventory
regeneration is deterministic, and all twelve scalar exclusions have `_b64`
API counterparts.

## Wave64 FP32 FMA forms

Added Wave64 FMA, FMAC, their DX9-zero forms, and literal FMAMK/FMAAK.
The wrappers retain each form's source/output modifier restrictions, literal
placement, accumulator aliasing, optional-state contract and existing scalar,
AVX2 and AVX-512 dispatch. Ordinary and DX9 forms route A through DPP within
each half; literal forms reject DPP. Existing loose semantics and exact
reporting rejection are unchanged.

Tests combine special-value inputs from the DX9 hardware corpus with independent
source-modifier, fused arithmetic and output-scaling references. They cover all
six forms, source/destination aliases, upper EXEC bits, every DPP descriptor
and row enable, literal placement, guards and unchanged loose exception state.
Invalid modes and exact requests reject before accessing null operands. Existing
Wave32 hardware tests remain; no native Wave64 capture or exactness claim is
added. The vector width-review queue is now 143 APIs.

Validation: all 801 tests pass on Clang 21 and GCC 15 Release. All nineteen
arithmetic and instruction-flag tests pass under ASan/UBSan.

## Wave64 FP32 binary arithmetic and min/max

Added all eleven FP32 binary APIs at Wave64: ADD, SUB, SUBREV, MUL, DX9-zero
MUL, four min/max variants and the legacy MIN/MAX aliases. The existing
scalar/AVX2 operation kernels, loose semantics and optional-state rejection
are unchanged. DPP routes A and applies each half's row enables; ordinary
source/output modifiers retain their Wave32 meaning.

Tests cover every supported low modifier combination, all DPP forms plus
unpermuted execution, all row enables, upper EXEC bits, source/destination
aliases and guards. Independent references include signed zeros, subnormals,
infinities, quiet/signaling NaNs and DX9 zero-product cases. NaN payload
comparison retains the loose policy. Loose exception state stays untouched;
invalid flags and unsupported exact requests reject before operand access.
No new exactness or hardware capture claim is made. The width queue is 132.

Validation: all 803 tests pass on Clang 21 and GCC 15 Release. All eight
binary arithmetic and instruction-flag tests pass under ASan/UBSan.

## Wave64 exact FP16 FMA and optional exceptions

Added Wave64 FMA, FMAC and both literal forms using the existing exact scalar
and loose AVX2 kernels. Source/destination half selectors, accumulator selection,
DPP and overflow controls retain their Wave32 rules. Exact reporting accumulates
into a local register across both halves, then ORs the caller's EXCP_FLAG_USER
after every source read and destination write. Loose execution skips reporting.

The existing exception hardware corpus now also runs in lane 63; raw flag
hashes match, including equivalent accumulator/literal cases. Independent
integer references check all forms, exact/loose results, modifiers, half
selectors, DPP row filtering, aliases, masks and preserved halves. Directed
tests combine invalid and overflow/inexact in different halves and alias the
exception output with an unread upper source. DPP-filtered exception tests
verify host rounding and sticky-flag preservation in every rounding mode, plus
loose opt-out behavior. No new native Wave64 capture is claimed.

Corrected two stale README descriptions of FP16 FMA saving/restoring host FP
state: the implementation uses an integer intermediate and never changes host
controls. The vector width-review queue is 128 APIs.

Validation: all 807 tests pass on Clang 21 and GCC 15 Release. All fourteen
FP16 FMA and instruction-flag tests pass under ASan/UBSan, as do all five
generic exception-register tests. The generated unsupported-reporting test
list now explicitly excludes these four supported Wave64 APIs.

## Wave64 packed FP16 FMA

Added packed FP16 FMA/FMAC Wave64 forms, retaining exact scalar and loose AVX2
execution. FMA preserves all 8,192 packed selector/negation/clamp combinations;
FMAC retains its zero-instruction-flags contract. Neither form accepts DPP.
Exceptions accumulate from both packed components across both wave halves,
with the caller's register updated only after all reads and destination writes.

Tests exercise every modifier with both overflow policies and both semantics,
all CPU paths, upper EXEC bits, masks, aliases and guard words. The existing
packed exception hardware corpus also runs in lane 63. Directed witnesses
combine lower-component invalid and upper-component overflow/inexact in
different wave halves, with reporting aliased to an upper input. Rocjitsu
result witnesses check both components across the whole wave and preserve host
rounding/exception state in all four rounding modes. Loose reporting remains
untouched; validation precedes operand access.

No native Wave64 capture or new throughput claim is added. The vector width
queue is now 126 APIs.

Also corrected a stale interpolation README paragraph: RTZ P10 uses the
integer FMA intermediate instead of changing host rounding controls. Removed
the obsolete RTZ throughput claim that described the earlier implementation.

Validation: all 811 tests pass on Clang 21 and GCC 15 Release. All seventeen
packed FMA, instruction-flag and exception-register tests pass under ASan/UBSan.

## Wave64 mixed-precision FMA

Added MIX_F32, MIXLO_F16 and MIXHI_F16 Wave64 APIs using the existing kernels.
The FP32 result remains loose-only; FP16 results retain empirical exact
semantics and the measured no-exception behavior. Optional state is untouched
for supported reporting, while FP32 exact reporting remains explicitly rejected.
Source formats, half selectors, modifiers, DPP row enables, preserved destination
halves and scalar/AVX2 dispatch remain unchanged.

Independent integer references cover DPP forms, all row-enable patterns,
upper-only/sparse EXEC, shared source registers, destination aliases and guards.
The no-exception hardware corpus also runs in lane 63 across all eight source
format combinations. Exact FP16-result tests check every host rounding mode
and DAZ/FTZ setting without changing host controls or sticky exceptions.
No new native Wave64 capture is claimed. The vector width queue is 123 APIs.

Validation: all 814 tests pass on Clang 21 and GCC 15 Release. Twelve targeted
mixed-FMA hardware/host-state/Wave64, instruction-flag and exception-register
tests pass under ASan/UBSan.

## Wave64 ordinary and packed FP16 binary arithmetic

Added eighteen Wave64 APIs: ordinary ADD/SUB/SUBREV/MUL and min/max forms,
packed ADD/MUL and min/max forms, plus both sets of legacy MIN/MAX aliases.
The family-local wrapper reuses the existing scalar/AVX2 kernels. Ordinary
forms route A through DPP and preserve the unselected destination half;
packed forms read original components for both results and reject DPP.
Loose semantics and unsupported exact reporting are unchanged.

Independent references check all 1,024 ordinary modifier/half-selection
combinations with every DPP form, all row-enable patterns, both overflow
policies, all CPU paths, aliases, upper EXEC bits and guards. Packed tests
cover every valid packed modifier combination under the same overflow/CPU
choices. NaN comparisons retain loose payload semantics. All eighteen APIs
verify empty EXEC, invalid modes and unchanged optional state, including
rejection of unsupported exact requests before operand access. No new
exactness, capture or throughput claim is made. The width queue is 105 APIs.

Validation: all 818 tests pass on Clang 21 and GCC 15 Release. All five
new Wave64 ordinary/packed binary and instruction-flag tests pass under
ASan/UBSan, including the complete modifier sweeps.

## Wave64 three-input FP16/FP32 min/max

Added all 28 three-input min/max/median Wave64 APIs, including legacy
spellings. Each family reuses its existing scalar/AVX2 selection algorithm,
retaining A/B-first mixed ordering, NaN and signed-zero behavior, output
modifiers, DPP routing and (for FP16) independently selected source/destination
halves. Loose semantics and unsupported exact reporting are unchanged.

Independent references sweep every FP32 low modifier and every FP16
modifier/half-selector combination while varying DPP descriptor, row enables,
overflow policy and destination alias. Tests cover all CPU paths, upper EXEC
bits, preserved destination halves, guards and unchanged loose optional state.
Validation checks empty EXEC and rejected modes/exact requests before operand
access. Existing Wave32 numerical and hardware tests remain. This adds no new
exactness, hardware capture or throughput claim. The width queue is 77 APIs.

Validation: all 822 tests pass on Clang 21 and GCC 15 Release. The five
new Wave64 min/max and instruction-flag tests pass under ASan/UBSan.

## Wave64 FP64 arithmetic and unary operations

Added eighteen FP64 Wave64 APIs: ADD/MUL/FMA, min/max and aliases, rounding,
FRACT, FREXP mantissa, SQRT/RCP/RSQ. Each operand retains its two-VGPR low/high
layout. The family wrapper respects unary/binary/ternary arity and reuses
existing scalar/AVX2 kernels, loose semantics, modifier rules and exact-reporting
rejection. DPP remains unsupported.

Tests compare independent finite arithmetic and higher-precision unary
references across every modifier, full/upper-only/sparse EXEC, crossed source
and destination words, shared destination words and guards. The high result
word wins when output VGPRs share storage. Upper unary inputs differ from
lower inputs, and unused ternary operands are null for binary operations.
Literal rounding/exceptional witnesses are repeated across all 64 lanes;
reciprocal/square-root comparisons retain existing loose ULP tolerances.
Validation rejects before operand access, and loose exception state remains
unchanged. No new exactness or native hardware claim is made. The width queue
is 59 APIs.

Validation: all 827 tests pass on Clang 21 and GCC 15 Release. All six new
Wave64 FP64 arithmetic/unary and instruction-flag tests pass under ASan/UBSan.


## Wave64 FP16 and FP32 unary operations

Added all eleven FP16 and eleven FP32 unary Wave64 APIs. Family-local wrappers
reuse existing numerical kernels and scalar/AVX2 dispatch, including scalar
FP32 EXP/LOG. Four independent DPP rows, source/output modifiers and FP16
half selection retain their existing contracts. Exact semantics and faithful
optional exception reporting remain unsupported; loose reporting adds no work.

Independent references cover every modifier/selector combination, plain and
DPP routing, all row-enable patterns, upper EXEC bits, aliases, guards and
preserved destination halves. FP32 tests also replay the existing hardware
OMOD/CLAMP/NEG corpus across 64 lanes, checking mandatory denormal flushing,
zeros and infinities while retaining loose finite SFU tolerances. This is
reuse of existing captures, not a new native Wave64 capture. Validation rejects
unsupported modes before operand access and preserves optional output state.
The vector width queue is 37 APIs. No new throughput claim is made.

Validation: all 832 tests pass on Clang 21 and GCC 15 Release. All six new
Wave64 unary and instruction-flag tests pass under ASan/UBSan.


## Wave64 empirical SIN/COS and exceptions

Added FP16 and FP32 SIN/COS Wave64 APIs, reusing the existing rocjitsu-derived
empirical model and loose AVX2 paths. Family-local wrappers remap DPP row enables
and stage exception accumulation until all source reads and destination writes
finish. This preserves reporting-pointer aliases with upper source lanes.
Loose semantics still skip reporting and retain SIMD dispatch.

Tests replay all existing exception hardware corpora with only lane 63 active,
repeat the FP32 raw result captures with every modifier across 64 lanes, and
check every FP16 encoding against the independent rocjitsu result digests.
DPP tests vary every modifier/half selector and row-enable pattern, upper-only
and sparse EXEC, whole-register aliases and exception accumulation. Additional
tests check aliased reporting, loose opt-out, error atomicity and exact FP32
results under host rounding/denormal controls without mutating those controls.
These reuse existing hardware observations; no new native Wave64 capture or
throughput claim is made. The vector width queue is 33 APIs.

Validation: all 839 tests pass on Clang 21 and GCC 15 Release. All eight
Wave64 trig and instruction-flag tests pass under ASan/UBSan, including the
complete exception corpus and exhaustive FP16 input sweep.


## Wave64 FP32/FP64 LDEXP

Added both Wave64 LDEXP APIs using the existing scalar/v3/v4 kernels and loose
numerical semantics. FP64 A/D retain low/high VGPR pairs while B remains one
signed integer VGPR. FP32 routes A through four independent DPP rows without
permuting B; FP64 continues to reject DPP. Exact reporting remains unsupported,
and loose optional state remains unchanged.

Independent higher-precision references cover all modifiers, upper-only/sparse
EXEC, crossed source/destination words, shared destination words and guards.
Literal rounding witnesses check subnormal ties, minimum-normal boundaries and
overflow. FP32 DPP tests cover all row-enable patterns, random source words,
integer exponent extremes, source sharing and destination aliases. Validation
checks errors, empty EXEC and optional-state contracts. No new exactness,
native Wave64 capture or performance claim is made. The width queue is 31 APIs.

Validation: all 843 tests pass on Clang 21 and GCC 15 Release. All five
Wave64 LDEXP and instruction-flag tests pass under ASan/UBSan.


## Wave64 exact DIV_FMAS and condition masks

Added FP32 and FP64 DIV_FMAS Wave64 APIs. Their condition input is a 64-bit
per-lane mask, split together with EXEC when reusing the existing exact
scalar/v3/v4 kernels. Optional exception flags accumulate only after both
halves consume all operands. Loose mode skips reporting; exact null reporting
retains SIMD dispatch. FP64 operands retain two-VGPR low/high layouts.

Tests replay existing Cartesian/random result digests across 64 lanes and
exception digests with only lane 63 active. Captured rounding-boundary results
check mixed lower/upper condition bits independently. Other tests cover crossed
and shared destination words, guards, upper-only/sparse EXEC, shared inputs,
all modifiers under host rounding modes, loose opt-out and a reporting pointer
aliased with an upper input. This reuses existing hardware evidence rather
than claiming native Wave64 captures or new throughput. The width queue is
29 APIs.

Validation: all 850 tests pass on Clang 21 and GCC 15 Release. All eight
Wave64 DIV_FMAS and instruction-flag tests pass under ASan/UBSan, including
the complete result/exception corpora and reporting-alias witness.


## Wave64 exact DIV_FIXUP

Added FP16/FP32/FP64 DIV_FIXUP Wave64 APIs with existing exact scalar/v3/v4
kernels, format-specific layouts and optional exception accumulation. FP16
supports four independent DPP rows and preserves the unselected destination
half. FP32/FP64 reject DPP. A family-local wrapper consumes both halves before
writing optional state, preserving aliases with upper-lane source operands.
Loose reporting remains skipped; exact opt-out retains SIMD dispatch.

Tests replay existing Cartesian result captures across 64 lanes and exception
captures with only lane 63 active. Independent references cover every modifier
and half selector, overflow policies, random inputs, whole-register/cross-word
aliases, shared sources, guards, sparse/upper-only EXEC and host rounding modes.
FP16 DPP tests cover all row enables, source routing and masked exception
accumulation. A separate zero/zero witness checks delayed reporting with an
upper input alias. This adds no native Wave64 capture or throughput claim.
The vector width queue is 26 APIs.

Validation: all 859 tests pass on Clang 21 and GCC 15 Release. All ten
Wave64 DIV_FIXUP and instruction-flag tests pass under ASan/UBSan, including
the exhaustive modifier/FP16 DPP sweeps and result/exception corpora.


## Wave64 DIV_SCALE and scalar condition output

Added FP32/FP64 DIV_SCALE Wave64 APIs using the existing exact scalar/v3/v4
kernels. Their mandatory condition output is 64 bits, with inactive bits
cleared and zero written for empty EXEC. Both halves are staged before the
condition store, which follows all source reads and VGPR writes and supports
unaligned aliases. Errors leave both vector and condition outputs unchanged.

Tests replay the existing captured value/condition digests across 64 lanes,
check all modifiers and host rounding modes, full/upper-only/sparse EXEC,
FP64 cross-word destination aliases and zero-EXEC/error behavior. Dedicated
byte-offset aliases place the condition output in each source or destination
register, in either wave half, with guard words checked. This reuses existing
hardware observations; no native Wave64 capture or new throughput is claimed.
The vector width queue is 24 APIs.

Validation: all 864 tests pass on Clang 21 and GCC 15 Release. All six
Wave64 DIV_SCALE and instruction-flag tests pass under ASan/UBSan, including
unaligned scalar-output aliases and captured value/condition digests.


## Wave64 RCP_IFLAG

Added RCP_IFLAG Wave64 using its existing loose scalar/v3/v4 kernels and
inexpensive INT_DIV0 classification. This preserves the existing reporting
behavior even in loose mode; it does not introduce exact reciprocal support.
Four DPP rows are independent. The wrapper snapshots the prior exception
register and commits accumulated flags after all source reads and destination
writes, including when the register aliases an upper input or destination.

Independent references cover every modifier, all row-enable patterns,
full/upper-only/sparse EXEC, preserved guards, input/output/status aliases and
null reporting. Existing hardware status digests are replayed in lanes 32–63;
inactive lower zeros must not contribute. Invalid modes and strict exact
requests preserve state. No new native Wave64 capture or throughput is claimed.
The width queue is 23 APIs.

Validation: all 869 tests pass on Clang 21 and GCC 15 Release. All six
Wave64 RCP_IFLAG and instruction-flag tests pass under ASan/UBSan, including
the full upper-half hardware status corpus.


## Wave64 cube-map operations

Added CUBEID/CUBESC/CUBETC/CUBEMA Wave64 APIs with existing exact scalar/v3/v4
kernels. A family-local wrapper handles lane offsets and four DPP row enables;
only X is permuted, while Y and Z retain their lanes. Numeric selection,
modifier rules and whole-register alias support are unchanged.

Tests replay the existing raw hardware Cartesian digests across 64 lanes,
check literal axis ties at every masked lane, and compare independent references
for all modifiers, special/random words, destination/shared-source aliases,
upper-only/sparse EXEC and guards. DPP tests vary all row enables and modifier
combinations. Host tests check every rounding mode and preserved sticky flags.
Validation covers empty EXEC and error atomicity. No new native Wave64 capture
or throughput claim is made. The vector width queue is 19 APIs.

Validation: all 876 tests pass on Clang 21 and GCC 15 Release. All eight
Wave64 cube and instruction-flag tests pass under ASan/UBSan, including
the complete modifier/DPP sweeps and captured Cartesian result digests.


## Wave64 MULLIT and TRIG_PREOP

Added both Wave64 APIs using existing scalar/v3/v4 kernels. MULLIT retains loose
semantics and four independent DPP rows. TRIG_PREOP retains exact table-based
results, two destination VGPRs, A's high-word-only exponent read and one selector
VGPR; it rejects DPP. Optional exact exception reporting remains unsupported
for both, while loose pointers remain unchanged.

Tests replay MULLIT's existing hardware Cartesian digests and TRIG_PREOP's
complete exponent/selector/modifier digests across 64 lanes. Independent MULLIT
references check all modifiers, DPP row enables, source sharing, destination
aliases, upper-only/sparse EXEC and guards. TRIG_PREOP checks captured samples
with crossed/shared destination words, upper-half input rotation and all host
rounding modes. A null low-word pointer verifies its unused input, and optional
state/error tests preserve outputs. No native Wave64 capture or new throughput
claim is made. The vector width queue is 17 APIs.

Validation: all 885 tests pass on Clang 21 and GCC 15 Release. All ten
Wave64 MULLIT/TRIG_PREOP and instruction-flag tests pass under ASan/UBSan,
including the full hardware digest replays and modifier/DPP sweeps.


## Wave64 interpolation

Added all six FP32/mixed-FP16 interpolation Wave64 APIs, including RTZ forms.
Each family reuses its existing scalar/SIMD numerical kernels. Quad broadcasts
remain within each 32-lane half and read inactive source lanes; wait fields,
NEG/CLAMP, half selectors, overflow policy and destination-half preservation
retain their contracts. Numerical semantics remain loose; strict exact requests
continue to fail rather than claiming new arithmetic fidelity.

Reviewed both manuals' interpolation descriptions: RDNA3 section 12.3 gives the
same fixed DPP8 broadcasts and fetch-invalid behavior as RDNA4 sections 12.3
and 16.13. This establishes the routing shared here, not untested numerical
identity between GPUs.

Tests replay existing FP16 result digests across 64 lanes and FP32 captures in
both halves. Upper-only/single-lane capture tests verify inactive source reads.
Independent numerical/integer references cover modifiers, wait counts, half
selection, saturation, random/special values, shared inputs, destination aliases,
guards and sparse EXEC. RTZ tests preserve host rounding and pre-existing sticky
flags. No new native Wave64 capture or throughput claim is made. The vector
width queue is 11 APIs, all dot products.

Validation: all 895 tests pass on Clang 21 and GCC 15 Release. All eleven
Wave64 interpolation and instruction-flag tests pass under ASan/UBSan,
including captured digests, RTZ rounding and inactive-source reads.


## Wave64 true16 DOT2 and FP8/BF8 DOT4

Added both FP16/BF16-result DOT2 APIs and all four FP8/BF8 DOT4 combinations
at Wave64. Family-local wrappers reuse existing scalar/AVX2 numerical kernels.
True16 DOT2 retains C/D half selectors, destination-half preservation and four
independent DPP rows; DOT4 retains its C modifiers and rejects DPP. Both families
remain loose-only, and true16 exact exception reporting remains unsupported.

Independent references check all 65,536 byte pairs for each DOT4 format and
modifier, every true16 accumulator encoding, literal rounding/denormal/overflow
boundaries, mixed packed elements, all modifiers/selectors, upper-only/sparse
EXEC, destination aliases and guards. DPP tests cover every row-enable pattern
and true16 modifier combination. Validation preserves rejected outputs. No new
exactness, native Wave64 capture or throughput is claimed. The vector width
queue is five FP32-result DOT2 APIs, including architectural variants.

Validation: all 904 tests pass on Clang 21 and GCC 15 Release. All ten
Wave64 narrow-dot and instruction-flag tests pass under ASan/UBSan.

## Wave64 architecture-specific FP32-result DOT2

Added the five remaining ordinary vector width variants: FP16/BF16 DOT2 for
RDNA3 and RDNA4, plus RDNA3 DOT2ACC. A family-local two-half executor preserves
the existing GFX11/GFX12 exact models, loose AVX2 kernels, DPP row enables and
whole-VGPR aliasing. RDNA4 reporting leaves EXCP_FLAG_USER unchanged, as in the
existing capture-backed Wave32 implementation; RDNA3 exact reporting remains
explicitly unsupported. No new numerical or native Wave64 hardware claim is made.

Tests replay both GPUs' raw result fixtures, including a small finite example
that distinguishes their result bits; verify modifiers, half selectors, DPP,
upper-only/sparse EXEC, old-D accumulation and aliases; and preserve host rounding
and sticky flags. RDNA4 no-exception corpora are replayed at lane 63. Independent
integer references check the shared loose/scalar/SIMD paths and four-row routing.

The vector-EXEC signature screen now has zero missing Wave64 counterparts. This
does not close the separate scalar/pseudo-scalar/matrix width audit, encoding
legality review, architecture semantics audit or three hardware-register APIs.

Validation: all 914 tests pass with Clang 21 and GCC 15 Release. All eleven
Wave64 DOT2 and instruction-flag tests pass under ASan/UBSan.

## Matrix width audit and Wave64 FP8/BF8 WMMA

Extended the generated inventory to cover matrix APIs, which omit EXEC and
therefore escaped the ordinary vector-width screen. RDNA4 manual sections
7.12.2/7.12.3 specify Wave64 layouts for these operations. After adding all four
FP8/BF8 dense WMMA combinations, fourteen matrix width gaps remain: three integer
WMMA and eleven SWMMAC APIs. Their arithmetic stage order and sparse selectors
need separate review; a dense layout conversion alone does not prove correctness.

FP8/BF8 Wave64 uses pointer views of 32-lane register sections to reuse the
existing scalar/AVX2 algorithms, with no matrix copies or new dispatch machinery.
A/B each use one VGPR, C/D four. All inputs precede destination writes. C modifiers,
loose fallback and strict-exact rejection are unchanged. Tests cover dense
mathematical goldens, every byte encoding, all four format pairs, modifiers, CPU
levels, aliases, guards and validation.

A new native RX 9070 gfx1201 Wave64 capture,
`tests/capture_fp8_wmma_wave64.cpp`, produces the word-FNV digest
`cebccc43bff33725` for each of the four formats (256 output words each). Its
exactly representable finite sums are replayed on every CPU path and with aliases.
This confirms register use and results for that corpus, not general bit-exactness.
Rocjitsu's `gfx12_wmma_input_loc` orders FP8 lane groups differently from the
manual. Our source mapping follows the manual; a common permutation of A/B's K
axis leaves these exact finite sums invariant, so the capture does not resolve
that discrepancy or establish the GPU's floating accumulation order.

Capture source SHA-256: `e25e3e098a27660b344f341e1cb242dc772cffb0af23cf1ac7678758b6ba6967`.
Validation: all 918 tests pass with Clang 21 and GCC 15 Release. All five
Wave64 FP8 WMMA and instruction-flag tests pass under ASan/UBSan.

## Wave64 integer WMMA with native saturation evidence

Added IU8 K=16, IU4 K=16 and IU4 K=32 Wave64 APIs, preserving architectural
`_rdna4_wave64` suffixes for the two shared RDNA3/RDNA4 mnemonics. Pointer views
of 32-lane register sections reuse existing scalar, AVX2 and Zen4 VNNI kernels.
A/B each use one VGPR, C/D four; IU4 K=16 ignores upper source lanes. Validation
precedes operand access, and the existing result snapshot preserves aliases.

A native RX 9070 gfx1201 capture covers 16 randomized near-limit input sets,
all eight signedness/CLAMP modes and all three shapes: 384 digests, 98,304
output words. `capture_integer_wmma_wave64.cpp` generates the recorded
`integer_wmma_wave64_hardware.h` fixtures. Source SHA-256:
`62052c723576344a55c7c20d936fb02c8bc2e1ef1c8e7cae743fd4878911f387`.

The captures distinguish the two saturation stages (source lane groups 0+2,
then 1+3) from one final clamp and, for IU8/K=32 IU4, four clamps. All captures
match the reused kernels. Independent logical-matrix tests cover extreme
factors, cancellation and overflowing accumulators using the manual's Wave64
layout. Native digests are replayed across CPU levels and destination aliases.
Invalid flags preserve outputs. The matrix width backlog is now eleven SWMMAC
APIs; existing dense WMMA APIs all have both widths, with numerical and
architecture audits still tracked separately.

Validation: all 921 tests pass on Clang 21 and GCC 15 Release. Five targeted
integer WMMA and instruction-flag tests pass under ASan/UBSan, including the
existing Wave32 capture corpus after making its input generator width-aware.

## Wave64 FP8/BF8 SWMMAC and four index sets

Added all four FP8/BF8 sparse matrix formats at Wave64. The family decoder
now handles both register layouts and feeds the existing scalar, AVX2 and
AVX-512 accumulation kernels. Wave64 uses one A VGPR, two B VGPRs, four
in/out D VGPRs and one index VGPR. Its four 8-bit metadata sets use both
selector bits; `GOC_SWMMAC_INDEX_KEY_2` exposes the second bit. Wave32 retains
its two 16-bit sets and rejects the new bit. Arithmetic remains loose-only.

Native RX 9070 gfx1201 captures cover four format pairs, sixteen input sets
and all four keys: 256 word-FNV digests over 65,536 output words. Every digest
matches all CPU paths. The generator `capture_swmmac8_wave64.cpp` has SHA-256
`9a691f2cfd9d341b6f2e0abcbb7994bbee8d8e3562faa4a5adc9b6f43e7c2b54`.
Fixtures are in `swmmac8_wave64_hardware.h`. Exactly representable input sums
validate the routing and selectors without claiming general exact arithmetic.
The lane routing follows rocjitsu's Wave64 sparse layout; four-byte selector
handling follows the ISA and native capture rather than its 16-entry helper.

Tests additionally cover every byte encoding, source/index/destination aliases,
duplicate destinations and rejected flags. The existing Wave32 captures remain
in the suite. Seven matrix width gaps remain: FP16/BF16 and integer SWMMAC.

Validation: all 925 tests pass with Clang 21 and GCC 15 Release. All ten
SWMMAC8 and instruction-flag tests pass under ASan/UBSan.

## Wave64 integer SWMMAC with shape-specific index selection

Added all three integer SWMMAC Wave64 forms, sharing decoded matrices and
existing scalar, AVX2 and AVX-512 accumulation kernels. IU8 accepts four 8-bit
index sets; both IU4 shapes accept two 16-bit sets. Wave64 K=64 therefore permits
KEY_1, unlike Wave32 K=64. A uses one VGPR, D four, and B two (one for K=32 IU4).
K=32 IU4 ignores upper A/index lanes. Inputs remain snapshotted before stores.

Wave64 K=64 saturation groups are consecutive halves of compressed K, unlike
Wave32's interleaved halves. The decoder places these products in the existing
kernels' two-stage order. Native RX 9070 gfx1201 captures cover 1,024 combinations
of shape, randomized near-limit inputs, sign flags, CLAMP and every legal key:
262,144 result words. All recorded digests match every CPU path. Capture source:
`tests/capture_swmmac_integer_wave64.cpp`, SHA-256
`cb3a44d8c2831748ba6373d03806e72acffa2750d08a012da8fb1e13abc87208`.
Fixtures live in `tests/swmmac_integer_wave64_hardware.h`.

Additional tests cover literal overflow/cancellation that distinguishes staged
saturation from final saturation, aliases with sources and metadata, duplicate
destinations, and invalid flag output preservation. Wave32 hardware fixtures
remain unchanged. Four matrix width gaps remain, all FP16/BF16 SWMMAC.

Validation: all 929 tests pass with Clang 21 and GCC 15 Release. All eleven
integer SWMMAC and instruction-flag tests pass under ASan/UBSan.

## Wave64 FP16/BF16 SWMMAC completes the matrix width inventory

Added all four FP16/BF16 sparse forms at Wave64. Width-aware decoding handles
two A VGPRs, four B VGPRs, four FP32 or two packed D VGPRs, and four metadata
bytes per lane. The existing scalar, AVX2 and AVX-512 kernels retain pair
negation, FP32 accumulation and packed narrowing. Strict exact requests remain
unsupported; this change makes no new general numerical-exactness claim.

Native RX 9070 gfx1201 captures cover all 64 combinations of four negation bits
and two index bits for each format/output type: 256 digests over 49,152 result
words. Every CPU path matches these exactly representable sums. Generator:
`tests/capture_swmmac16_wave64.cpp`, SHA-256
`023bff6696b8698779e8ad6ca1df3bb9c7085ba9edc6e54c204e232577744f41`;
fixtures: `tests/swmmac16_wave64_hardware.h`. Additional tests cover packed
nearest-even ties, finite saturation versus infinity, literal pair negation,
aliases with inputs/metadata, duplicate destinations and validation.

The ordinary-vector and matrix API width inventories now both have zero missing
counterparts. Scalar/pseudo-scalar width contracts, encoding legality, complete
architecture comparison, hardware-register access and remaining numerical/state
gaps are separate gates and are not closed by this milestone.

Validation: all 935 tests pass with Clang 21 and GCC 15 Release. All fourteen
FP16/BF16 SWMMAC and instruction-flag tests pass under ASan/UBSan.

## Cross-architecture policy audit after width completion

Added `tools/architecture_audit.py` and the generated `ARCHITECTURE_AUDIT.md`
review queue. Every one of the 587 common in-scope canonical mnemonics is listed;
89 have direct architecture/IEEE/OMOD/NaN policy references in rocjitsu's shared
execution helpers. The report explicitly leaves transitive helpers, SIMD macros,
architecture-specific overrides and all semantic proof open. The other 222
architecture-exclusive canonical mnemonics remain covered by the ISA inventory.

A concrete finite-result incompatibility is confirmed by both manuals and
rocjitsu's `fp_mode::effective_omod`: RDNA3 ignores OMOD with output denormals
enabled or IEEE mode set, while RDNA4 applies it. At `227ced2`, a GoC scalar
probe of `goc_v_add_f32(0, 1, GOC_ALU_OMOD_2, ...)` on 1 and 2 returns success
and `0x40c00000` (6). RDNA3 with output denormals enabled requires 3. This is
a specification/model comparison, not native RDNA3 evidence. Existing RDNA4
behavior must acquire the suffix as each affected family gains RDNA3 policy;
adding a guest IEEE-mode flag is also required. Packed results and instructions
that force flushing require individual review rather than bulk renaming.

Revisited HWREG prose: RDNA4 explicitly includes the shifted SETREG mask,
whereas RDNA3's printed pseudocode omits the shift. That discrepancy must not
be mistaken for hardware semantics. Register IDs, privilege masks and side
effects still require architecture-specific API/state contracts; the three
missing HWREG APIs remain open.

Validation: four Python extraction tests cover nested scopes, comments and
strings, missing/truncated input, and architecture-exclusive inventory rows.
The report regenerates deterministically from the recorded source fingerprint.
The unchanged 935-test suites also pass on Clang 21 and GCC 15 Release;
those passing tests do not cover or resolve the newly identified RDNA3 gap.

## FP32 binary architecture-policy split

- ADD/SUB/SUBREV/MUL retain RDNA4 behavior under `_rdna4`, before `_wave64`.
  Unsuffixed APIs implement RDNA3's conditional OMOD rule. Shared arithmetic
  kernels receive the effective modifier; no duplicate SIMD implementation.
- Guest input/output denormal flushing and MODE.IEEE are accepted by these four
  families. Both scalar and eight-lane v3 kernels flush values explicitly; no
  host FP-control writes. Other families still reject the new IEEE flag.
- Tests preserve the existing RDNA4 modifier/special-value corpus and add both
  architectures, both wave sizes, guest flag combinations, OMOD, DPP, masks,
  aliases, literal signed denormal cases and unsupported exact/state contracts.
- These APIs remain loose-only. Exact NaN quieting/payloads, guest rounding modes
  and exception reporting remain unsupported; this closes an OMOD policy gap,
  not the entire arithmetic semantic audit.
- Validation: all 937 tests pass under Clang 21 and GCC 15 Release; all 10
  Binary/InstructionFlags tests pass under ASan/UBSan; four audit-tool tests pass.
  Logs: `../goc-build/coverage-work/binary-policy-*-tests.log`.

## FP32 binary selection and DX9 policy follow-up

- RDNA3 MIN/MAX now have their own explicit selection rules: in IEEE mode an
  sNaN wins over a numeric value (and is quieted); outside IEEE mode a lone NaN
  is ignored and two NaNs select A unchanged. RDNA4 MIN_NUM/MAX_NUM keep number
  preference. The manuals' V_MIN_F32/V_MAX_F32 pseudocode is stronger evidence
  here than rocjitsu's shared host `fmin`/`fmax` implementation.
- Both architectures' min/max apply input flushing before selection and ignore
  output flushing except through OMOD. This follows the instruction-specific
  denormal notes, not the different memory-atomic rules. RDNA3 additionally gates
  OMOD on guest IEEE/output-denormal mode.
- DX9 multiply is split into unsuffixed RDNA3 and `_rdna4`, in both wave widths.
  It shares the existing arithmetic kernels and adds guest denormal/IEEE flags.
- Tests extend the architecture-policy matrix to MIN/MAX and DX9 multiply, and
  use literal NaN/sign/denormal expectations across CPUs, guest flags, widths,
  masks and aliases. Existing RDNA4 tests/benchmark calls retain their behavior
  under the suffixed DX9 names. Exact arithmetic/state requests remain unsupported.
- Validation: all 939 tests pass under Clang 21 and GCC 15 Release, and all 16
  Binary/MinmaxSpellings/InstructionFlags tests pass under ASan/UBSan. Four audit
  tool tests pass. Logs: `../goc-build/coverage-work/binary-selection-*-tests.log`.

## Ordinary FP16 binary architecture policies

- ADD/SUB/SUBREV/MUL now have unsuffixed RDNA3 and `_rdna4` entry points in
  Wave32 and Wave64. RDNA3 disables OMOD with preserved output denormals or IEEE;
  RDNA4 applies OMOD regardless, matching the manuals and rocjitsu's
  `fp_mode::effective_f16_omod` policy.
- Both variants accept guest input/output denormal controls. Scalar and AVX2
  kernels flush selected input halves before arithmetic and encoded subnormal
  results after narrowing, retaining signed zero and the unselected destination
  half. Existing FP16 overflow saturation and DPP remain supported.
- Existing RDNA4 tests and benchmarks now call suffixed APIs. The new policy
  matrix spans both architectures/widths, all guest flag combinations, selectors,
  scaling factors, overflow saturation, DPP, masks and destination/source aliases.
- Packed FP16 and FP16 min/max retain their previous flags/contracts pending their
  own architecture audit; packed arithmetic does not inherit ordinary OMOD rules.
  These changes do not add exact arithmetic or faithful exception reporting.
- Validation: all 941 tests pass with Clang 21 and GCC 15 Release; all 11
  HalfBinary/InstructionFlags tests pass with ASan/UBSan. Four audit-tool tests
  pass. Logs: `../goc-build/coverage-work/half-policy-*-tests.log`.

## FP64 ADD/MUL/FMA architecture policies

- ADD/MUL/FMA now expose unsuffixed RDNA3 and `_rdna4` APIs in both wave sizes.
  RDNA3 gates OMOD on output-denormal/IEEE state; RDNA4 applies it regardless.
  This follows the manuals and rocjitsu's shared FP64 effective-OMOD policy.
- Guest input/output denormal controls are value transforms on scalar and
  four-lane AVX2 paths. FMA flushes all three selected operands before its fused
  operation. Staged low/high destination writes preserve cross-half aliases.
- Existing RDNA4 modifier, special-value and output tests retain their behavior
  under suffixed APIs. New tests combine both architectures/widths, guest flags,
  scaling/source modifiers/CLAMP, denormals and special values, EXEC masks, cross-
  half aliases and duplicate destination buffers.
- Other FP64 families still reject these guest flags pending policy review.
  Exact arithmetic, guest rounding modes and faithful optional exception reporting
  remain unsupported; this slice does not claim those numerical guarantees.
- Validation: all 942 tests pass with Clang 21 and GCC 15 Release; all 13
  Fp64/Fp64Output/InstructionFlags tests pass with ASan/UBSan. The new policy test
  also passes after the final GCC assertion-bracing correction. Four audit-tool
  tests pass. Logs: `../goc-build/coverage-work/fp64-policy-*-tests.log`.

## FP64 min/max architecture policies

- RDNA3 MIN/MAX now apply IEEE-mode signaling-NaN priority, preserve A unchanged
  when both inputs are NaNs outside IEEE mode, and gate OMOD on guest IEEE/output
  denormal state. RDNA4 MIN_NUM/MAX_NUM keep number preference and unconditional
  OMOD. The RDNA3 V_MIN_F64/V_MAX_F64 pseudocode establishes these differences.
- All six FP64 binary min/max APIs accept guest input flushing. Output flushing
  is ignored except as required by OMOD, following the instruction-specific notes.
  Both wave sizes retain four-lane AVX2 and staged cross-half alias handling.
- The RDNA3 scalar NaN rule now lives once in `minmax.h`, shared by FP32 and FP64
  using the existing FloatBits masks. Explicit vector overloads live beside the
  existing SIMD min/max functions; no policy classes, adapters or new files.
  Maintainers change the scalar selection rule in one place; SIMD intrinsics and
  the instruction-level OMOD/denormal decisions remain directly visible.
- Tests extend the FP64 policy/mask/alias matrix to min/max and add literal NaN
  payload/quieting cases, signed zero, and input/output denormal combinations.
  Existing FP32 policy tests exercise the extracted shared helper. Strict exact
  arithmetic and faithful optional exception reporting remain unsupported.
- Validation: all 943 tests pass under Clang 21 and GCC 15 Release; all 27
  Binary/Fp64/MinmaxSpellings/InstructionFlags tests pass under ASan/UBSan. Four
  audit-tool tests pass. Logs: `../goc-build/coverage-work/fp64-selection-*-tests.log`.
- A pinned-CPU-2 AVX2 MIN_NUM_F64 comparison against parent `c2a1c7f` used nine
  alternating samples with at least 50 ms thread CPU time per case. Median time
  was 40.718 -> 40.770 ns unmodified and 45.214 -> 45.645 ns with ABS/NEG/OMOD/CLAMP
  (about +0.1% and +1.0%, respectively). This limited check found no material
  regression; it does not establish performance for every instruction or host.
  Scratch driver, parent objects and raw samples are in
  `../goc-build/coverage-work/fp64-selection-*`, including `fp64-selection-timing.json`.

## Ordinary and packed FP16 binary min/max policies

- RDNA3 MIN/MAX and PK_MIN/PK_MAX now use IEEE-mode signaling-NaN priority and
  preserve A unchanged when both inputs are NaNs outside IEEE mode. Ordinary
  forms gate OMOD; packed forms do not interpret packed modifier bits as OMOD.
  All binary min/max forms accept guest input flushing and ignore output flushing
  except through ordinary OMOD, in both wave widths.
- Existing scalar and SIMD RDNA3 selection helpers now serve FP16 as well as
  FP32/FP64. FP16 SIMD conversion gains an opt-in NaN-bit-preserving path for
  legacy and propagating selection; narrowing preserves signaling bits only for
  the legacy forms. Arithmetic and RDNA4 number-selection conversion paths retain
  their existing defaults. This also fixes RDNA4 propagating FP16 min/max's sNaN
  priority when a quiet NaN competes with a signaling NaN.
- Literal tests distinguish all three selection policies using raw payloads,
  signs and quiet bits, ordinary/packed forms, both widths, guest flag combinations,
  half selectors, independent negations, clamp, masks and aliases. The ordinary
  architecture-policy matrix now includes min/max scaling/denormal combinations.
- Exact semantics and faithful exception reporting remain unsupported for these
  APIs. Ternary FP16 min/max/median are a separate policy-audit item.
- An additional all-65,536-encodings MIN/MAX test verifies that IEEE-disabled
  selection against a quiet NaN preserves every FP16 sign, payload and signaling
  bit, including through the F16C path. Conversion repair restores only the quiet
  bit changed by F16C; it does not reconstruct the entire NaN encoding.
- Validation: all 945 tests pass under Clang 21 and GCC 15 Release; all 24
  HalfBinary/PackedBinary/MinmaxSpellings/InstructionFlags tests pass under
  ASan/UBSan. Four audit-tool tests pass. Logs:
  `../goc-build/coverage-work/half-selection-*-tests.log`.
- Performance check against parent `23bbf06`: eleven alternating samples pinned
  to CPU 2, each at least 50 ms thread CPU time. AVX2 MIN_NUM_F16 medians changed
  24.607 -> 23.117 ns unmodified and 26.173 -> 25.274 ns with ABS/NEG/OMOD/CLAMP.
  MINIMUM_F16 changed 26.815 -> 26.664 ns unmodified and 28.627 -> 29.798 ns with
  modifiers (about +4.1% for that case). The measured cost buys correct sNaN
  priority; this limited finite-input check is not a whole-library performance
  result. Driver, parent objects and raw samples are in
  `../goc-build/coverage-work/half-selection-*`, including `half-selection-timing.json`.

## Architectural exception-register naming

- RDNA3 ISA section 3.4 names TRAPSTS.EXCP as the sticky exception register;
  RDNA4 uses EXCP_FLAG_USER. Arithmetic bits 0–6 agree, as also reflected by
  rocjitsu's gfx12_excp_flag_user_raw mapping. Other register fields differ.
- Renamed the final pointer to `trapsts` in 94 explicitly RDNA3 public APIs and
  their definitions: RDNA3-only canonical instructions and unsuffixed instructions
  with an explicit RDNA4 counterpart. Parameter types/order, ABI, optionality,
  accumulation/error contracts and numerical behavior are unchanged.
- Generated runtime reporting-contract tests now discover both register names.
  New `tools/global_state_audit.py` / `GLOBAL_STATE_AUDIT.md` classify every existing
  exception-output signature, with tests for architecture resolution and the real
  public declarations. Pending shared APIs are explicitly unresolved.
- This naming audit does not establish whether each instruction should expose an
  exception output at all, whether any other writable state is missing, or whether
  reporting is numerically faithful. Those remain separate completion gates.
- Validation: all 945 tests pass under both Clang 21 and GCC 15 Release; all 15
  ExceptionRegisters/Dot2Architecture tests pass under ASan/UBSan. Three new
  global-state audit tests and four architecture audit tests pass. Generated
  reporting-contract coverage remains 366 entries after accepting both parameter
  names. Logs: `../goc-build/coverage-work/register-names-*-tests.log`.

## RDNA3 FP32 ternary selection policies

- Replaced RDNA3 MIN3/MAX3/MINMAX/MAXMIN/MED3 forwarding to RDNA4 number-selection
  APIs with RDNA3 binary-policy composition, in both wave widths. RDNA3 ISA
  section 16.12 explicitly defines these nested operations; MED3 uses MIN3 for
  any NaN and otherwise removes the first maximum. IEEE mode therefore affects
  signaling NaNs at each binary selection, rather than globally propagating any
  signaling input to the final result.
- Scalar and eight-lane AVX2 paths reuse their respective binary RDNA3 selectors.
  Added input-denormal flushing and guest IEEE/output-denormal flag acceptance
  throughout FP32 ternary selection. Output denormals survive selection; RDNA3
  output scaling is disabled when output denormals are preserved or IEEE mode
  is enabled. RDNA4 scaling remains independent of those guest settings.
- Literal tests cover signaling payloads in different operand positions, all-NaN
  inputs, both IEEE settings, wave widths and optional-state preservation. Existing
  Wave64 all-modifier/DPP/mask/alias tests now use RDNA3 expectations for its APIs.
  Additional guest-mode tests distinguish input flushing, retained output
  subnormals and architecture-dependent output scaling.
- Strict exact semantics and faithful exception reporting remain unsupported for
  these APIs; loose execution still skips optional reporting. FP16 ternary
  selection remains a separate policy-audit item.
- Rocjitsu's shared MIN3/MED3 helpers confirm conditional OMOD through
  `fp_mode::effective_omod`, but use host `fmin`/`fmax` for selection. Their host
  selection is not evidence for guest IEEE-mode NaN priority; that correction
  follows the manual's nested binary pseudocode.
- Performance: eleven alternating before/after samples against `db9a8c3`, pinned
  to CPU 2, each at least 50 ms thread CPU time. RDNA4 AVX2 MIN3_NUM_F32 medians
  changed 20.384 -> 21.506 ns unmodified (+5.5%) and 24.070 -> 25.203 ns with
  ABS/NEG/OMOD/CLAMP (+4.7%). MED3_NUM_F32 changed 32.819 -> 32.883 ns and
  34.875 -> 35.102 ns (both within 1%). Input-denormal policy dispatch occurs
  outside the SIMD loop; this reduced an initial roughly 10% MIN3 regression.
  These finite-input measurements are not a whole-library performance result.
  Driver, parent objects and raw samples: `../goc-build/coverage-work/ternary-*`,
  including `ternary-timing.json`.
- Final validation: all 947 tests pass with Clang 21 and GCC 15 Release; all nine
  Minmax3 tests pass under ASan/UBSan. Seven architecture/global-state audit
  tests pass. Logs: `../goc-build/coverage-work/ternary-*-tests.log`.

## RDNA3 FP16 ternary selection policies

- Replaced RDNA3 FP16 MIN3/MAX3/MINMAX/MAXMIN/MED3 forwarding with the same
  RDNA3 nested binary selection rule used by FP32. Both wave widths use guest
  IEEE-mode signaling-NaN priority, input-denormal flushing and conditional OMOD.
  Selection preserves output subnormals except when OMOD requires flushing.
- RDNA3 MED3 follows the manual's first-maximum removal pseudocode, including
  signed-zero ties. RDNA4 MED3_NUM retains its hardware-captured ordered-zero
  median behavior. The RDNA3 result is a manual-based model, not a new native
  RDNA3 capture or a claim of empirical exact semantics.
- Reused existing scalar/SIMD RDNA3 selection and opt-in FP16 NaN-preserving
  conversion helpers. RDNA4 propagating ternary selection now also preserves
  input signaling bits until selection, fixing premature quieting by F16C.
- Literal tests check NaN signs/payloads/signaling bits, IEEE mode, first-maximum
  zero ties, all half selectors, both wave widths, masks and unchanged optional
  state. An exhaustive 65,536-encoding test checks legacy MIN3/MAX3/MED3 against
  two quiet NaNs. Existing Wave64 DPP/modifier/alias tests now use RDNA3 policy
  expectations; RDNA4 median hardware fixtures remain unchanged.
- Guest-mode tests distinguish input flush, output-subnormal preservation and
  architecture-dependent scaling. Strict exact semantics and faithful optional
  exception output remain unsupported for these selection APIs.
- Validation: all 950 tests pass under Clang 21 and GCC 15 Release; all 12
  HalfMinmax3 tests pass under ASan/UBSan. Seven architecture/global-state audit
  tests pass. Logs: `../goc-build/coverage-work/half-ternary-*-tests.log`.
- Performance against `619dd03`: eleven alternating samples pinned to CPU 2,
  each at least 50 ms thread CPU time. RDNA4 AVX2 MIN3_NUM_F16 medians changed
  29.316 -> 28.539 ns unmodified and 33.631 -> 32.500 ns with ABS/NEG/OMOD/CLAMP.
  MINIMUM3_F16 changed 33.707 -> 35.251 ns (+4.6%) and 37.016 -> 38.044 ns (+2.8%),
  including the signaling-NaN conversion correction. MED3_NUM_F16 changed
  38.987 -> 35.408 ns and 41.941 -> 39.066 ns. These finite-input timings are a
  limited regression check, not a whole-library performance claim. Artifacts:
  `../goc-build/coverage-work/half-ternary-*`, including `half-ternary-timing.json`.

## HWREG state and side-effect follow-up

- Added `HWREG_AUDIT.md`, cross-checking all 24 RDNA3 and 19 RDNA4 XML IDs plus
  the RDNA4 manual-only IB_STS2 entry. No API-presence counts changed.
- Identified RDNA4 GETREG's snapshot-PC-high read-unlock effect and sampled-wave
  visibility requirement. A read-only raw snapshot API alone cannot represent
  that complete behavior. The manual does not name a lock register or fully
  specify post-read values; the audit does not invent them.
- Recorded the RDNA3 STATUS privileged writable-bit list, scratch-write
  privilege differences from rocjitsu, exception-register replacement versus
  arithmetic accumulation, FLUSH_IB scope ambiguity, and shader-versus-host
  clock-read protocols. These are explicit implementation gates.
- Validation: extracted both OPR_HWREG maps from the machine-readable ISA XML,
  checked all table IDs/names against them, and checked the behavioral notes
  against manual sections 3.4/16.2 and rocjitsu's HWREG implementation. This was
  documentation-only; the last production gates remain 950 Clang/GCC tests and
  12 FP16 ternary sanitizer tests from `422fbfd`.

## FP32 FMA architecture and guest-mode policies

- Split FMA, FMAC and FMA_DX9_ZERO into unsuffixed RDNA3 and `_rdna4` APIs in
  both wave widths. FMAC_DX9_ZERO is RDNA3-only in the XML and keeps only its
  unsuffixed APIs. RDNA3 OMOD is disabled with preserved output denormals or
  IEEE mode; RDNA4 always applies it. This follows the manuals' OMOD rule and
  rocjitsu's effective_omod/effective_vop3_omod_f32 handling.
- Ordinary FMA/FMAC now accept guest input/output-denormal flushing and IEEE
  mode in scalar, eight-lane AVX2 and sixteen-lane AVX512 paths. DX9-zero forms
  retain their fixed input/output flushing and zero-product rule, independent
  of guest flush flags; architecture-specific OMOD gating still applies.
- Both architectures share the arithmetic kernels and DPP routing. Existing
  RDNA4 captures/tests/benchmarks now call the suffixed APIs. The RDNA3-only
  DX9 accumulator routing test explicitly enables output flushing when comparing
  against RDNA4 FMA with OMOD enabled. New tests cover every guest-mode
  combination, all four OMOD values, normal/subnormal operands through each
  input, generated subnormal outputs, masks, both widths and accumulator forms.
- Optional exception reporting remains unsupported in exact semantics; loose
  execution skips it. No host FP control writes were introduced. Literal FMA,
  FP16 FMA and dual-issue architecture policies remain separate audit items.
- Validation and DPP scratch allocation are centralized without burdening ordinary
  FMA: only the DPP helper validates routing and stages a temporary VGPR. Flags
  and low modifier bits are checked once before arithmetic, and zero-mask DPP
  calls still validate their descriptors before returning.
- Performance against `17bea5f`: eleven alternating samples pinned to CPU 2,
  each at least 50 ms thread CPU time. Ordinary RDNA4 FMA AVX2 medians changed
  17.189 -> 16.301 ns unmodified and 19.675 -> 19.250 ns with ABS/NEG/OMOD/CLAMP;
  AVX512 changed 7.848 -> 7.147 ns and 9.115 -> 8.073 ns. DX9-zero AVX2 changed
  21.242 -> 21.313 ns and 23.477 -> 23.591 ns; AVX512 changed 7.620 -> 7.701 ns
  and 10.447 -> 10.664 ns. These are limited finite-input checks, not overall
  library speedups. Raw samples and driver/parent objects are under
  `../goc-build/coverage-work/fma-architecture-*`, including
  `fma-architecture-timing.json`.
- Final validation: all 951 tests pass under Clang 21 and GCC 15 Release; all
  20 Arithmetic/InstructionFlags tests pass under ASan/UBSan. Seven audit-tool
  tests pass. Regenerated ISA/architecture/global-state inventories: the latter
  now has 616 reporting signatures, 274 resolved names, zero mismatches and
  342 unresolved shared signatures. Logs:
  `../goc-build/coverage-work/fma-architecture-*-tests.log`.

## FP16 FMA architecture and guest-mode policies

- Split FMA/FMAC/FMAMK/FMAAK into unsuffixed RDNA3 and `_rdna4` APIs, in both
  wave widths. Adapted rocjitsu's shared fp_mode::fma_f16, quiets_nan,
  effective_f16_omod and finish_fma_f16 rules: RDNA3 gates OMOD on output
  denormals/IEEE mode and preserves a selected signaling NaN when IEEE is off;
  RDNA4 quiets selected NaNs and applies OMOD independently of those settings.
- The RNE exact result model now accepts input/output-denormal flushing. Input
  flushing precedes exceptional-input classification; output flushing preserves
  zero signs and uses the pre-exponent-limit rounding boundary at the smallest
  normal. OMOD still rounds to FP16 before scaling. The existing integer
  intermediate avoids changing or depending on the host rounding controls.
- Scalar and eight-lane loose SIMD paths support the guest flush flags. Loose
  NaN payload/quiet-bit behavior remains approximate. Existing RDNA4 result and
  exception corpora were migrated to the suffixed APIs without changing their
  captured expectations. This is a rocjitsu-derived RDNA3 model, not a claim
  of a newly captured native RDNA3 hardware corpus.
- RDNA3 exception reporting and RDNA4 reporting with input/output flush enabled
  reject non-null exact requests until faithful reporting for these policies is
  verified. Null exact requests retain the result model; loose requests skip
  reporting without error. Existing RDNA4 no-flush reporting remains enabled.
- Tests cover all four operand forms, both wave widths, half selectors, all
  guest-mode combinations, all allowed OMOD settings, masks, NaN priority,
  signaling bits and host-control preservation. An independent integer oracle
  covers a Cartesian boundary corpus under both loose and exact semantics.
  Generated unsupported-reporting tests now distinguish the RDNA3 and RDNA4
  forms. Packed, mixed and dual-issue FMA policies remain separate audit items.
- Performance against `f9b7063`: eleven alternating samples pinned to CPU 2,
  each at least 50 ms thread CPU time. RDNA4 loose AVX2 FMA medians changed
  27.657 -> 28.218 ns unmodified (+2.0%) and 43.244 -> 44.717 ns with
  ABS/NEG/OMOD/CLAMP (+3.4%); literal FMAMK changed 23.228 -> 24.607 ns (+5.9%).
  Exact scalar FMA changed 792.706 -> 825.748 ns (+4.2%) and
  917.987 -> 950.263 ns (+3.5%); exact FMAMK changed 782.899 -> 803.665 ns (+2.7%).
  The shared scalar helper now carries saturation and guest policy in one flag
  word instead of adding a seventh per-lane argument. These finite-input checks
  expose modest overhead from guest controls; they are not a whole-library
  performance result. Artifacts: `../goc-build/coverage-work/half-fma-architecture-*`,
  including `half-fma-architecture-timing.json`.
- Final validation: all 954 tests pass under Clang 21 and GCC 15 Release;
  all 37 HalfFma/PackedFma/ScalarFma/ExceptionRegisters/InstructionFlags tests
  pass under ASan/UBSan. Seven audit-tool tests pass. The global-state audit
  now has 624 reporting signatures, 290 resolved names, zero mismatches and
  334 unresolved shared signatures. Logs:
  `../goc-build/coverage-work/half-fma-architecture-*-tests.log`.

## Packed FP16 FMA architecture policies

- Split `v_pk_fma_f16` and `v_pk_fmac_f16` into unsuffixed RDNA3 and
  `_rdna4` APIs for both wave sizes. rocjitsu's generated shared handlers
  (lines 17495 and 17594/17627, source hash in `ARCHITECTURE_AUDIT.md`) use
  the same `fp_mode::fma_f16` model as ordinary FP16 FMA, including
  architecture/IEEE-dependent NaN quieting. Both packed halves now use GoC's
  existing architecture-aware result helper, rather than duplicating its rules.
- Guest input/output denormal flags and IEEE mode are accepted. The existing
  eight-lane loose SIMD kernel handles flushing without falling back to scalar.
  Selected signaling NaNs remain unquieted in RDNA3 exact mode with IEEE off;
  RDNA4 quiets them. Existing RDNA4 result/exception fixtures retain their raw
  expectations and now call the suffixed APIs. No new native RDNA3 capture is
  claimed; that policy is derived from rocjitsu.
- New tests exercise both architectures, wave widths, FMA/FMAC, both semantics,
  baseline/AVX2, all eight guest-policy combinations, packed selectors,
  negation/clamp, masks and unchanged outputs on unsupported reporting errors.
  Exact RDNA3 reporting and RDNA4 reporting with flushing remain explicitly
  unsupported. Null reporting works; loose reporting is skipped. Wave64
  preflight checks these limits before any writes.
- Packed FMAC still exposes the unmodified operand form. Its VOP3 ABS/NEG/CLAMP
  support (and applicable encoding-specific behavior) remains to be implemented;
  this policy split is not a completeness claim for all encodings or FP modes.
- Against `7a33d4b`, eleven alternating CPU-2-pinned samples with at least 50 ms
  thread CPU time measured RDNA4 packed FMA loose AVX2 at 38.278 -> 39.470 ns
  (+3.1%) unmodified and 43.724 -> 44.572 ns (+1.9%) with packed selectors,
  negation and clamp. Exact scalar medians were 1633.945 -> 1642.266 ns (+0.5%)
  and 1655.012 -> 1670.885 ns (+1.0%). These finite-input measurements are not a
  whole-library performance claim. Raw samples and drivers are under
  `../goc-build/coverage-work/packed-fma-policy-*`.
- Validation: 955/955 tests pass with Clang 21 and GCC 15 Release; 17/17
  PackedFma/ExceptionRegisters tests pass with ASan/UBSan. Seven audit-tool
  tests pass. The refreshed global-state audit has 628 signatures, 298 resolved
  register names, zero mismatches and 330 still requiring architecture review.

## Packed FMAC VOP3 modifiers

- `v_pk_fmac_f16` and its RDNA4/Wave64 variants now accept ordinary ALU
  ABS/NEG for A/B and CLAMP. Each modifier applies to both packed components;
  D remains an unmodified accumulator. All OMOD values are accepted and ignored
  as specified by rocjitsu's `effective_f16_omod(..., packed_result=true, ...)`
  for GFX11/GFX12. C modifiers and half-selection flags remain invalid.
- Both manuals allow VOP2 instructions to use VOP3 controls (RDNA3 section
  16.7.1 / RDNA4 section 16.7.1). The component rules follow rocjitsu's
  `execute_v_pk_fmac_f16_vop3` (shared handler line 17627) and its FP16 model.
  This closes the VOP3 modifier gap recorded in the previous slice; DPP/other
  encoding-specific behavior is not implied by this change.
- Scalar and AVX2 paths share the existing packed FMA computation. FMA obtains
  per-component ALU modes from packed selectors; FMAC uses corresponding halves
  and applies ordinary ALU modifiers. Both compile from the same source loop,
  with a boolean template parameter distinguishing the operand forms. A single
  compiled SIMD kernel accepting arbitrary per-half modes regressed existing
  loose FMA by about 9%; the specialization removes its unused ABS checks.
  SIMD object text grows from 1,793 to 3,221 bytes. No scalar fallback is added.
- A new independent-oracle test covers all 128 modifier combinations, eight
  guest-policy combinations, both semantics/architectures/wave widths, every
  detected CPU level, masks, and destination aliases with either input.
  Validation tests still reject unsupported instruction bits, including with
  zero EXEC. Existing RDNA4 exception and result captures remain unchanged.
- A reporting test also verifies that ABS/NEG changes invalid infinity
  cancellation and that RDNA4 CLAMP suppresses its exception flags, with sticky
  state and an active uppermost lane in each wave width.
- Final performance against `f6cdd8c`: eleven alternating CPU-2-pinned samples,
  at least 50 ms thread CPU time each. Existing RDNA4 packed FMA loose AVX2
  medians were 39.549 -> 39.563 ns unmodified and 44.748 -> 44.758 ns with
  selectors/negation/clamp (both within 0.1%). Exact scalar medians were
  1645.582 -> 1651.787 ns and 1672.163 -> 1676.021 ns (both within 0.4%).
  This checks the existing kernel, not all library instructions or new FMAC
  modifier workloads. Samples/drivers: `../goc-build/coverage-work/packed-fmac-modifiers-*`.
- Validation: all 957 tests pass under Clang 21 and GCC 15 Release; all 19
  PackedFma/ExceptionRegisters tests pass under ASan/UBSan. `PLAN.md` now also
  explicitly distinguishes implemented RDNA4 no-flush FP16 reporting from
  unsupported RDNA3/flushing requests. Logs use the artifact prefix above.

## FP32 literal FMA architecture names and guest denormals

- Split FMAMK/FMAAK FP32 into unsuffixed RDNA3 and `_rdna4` entry points,
  including Wave64. Their architectural exception outputs are `trapsts` and
  `excp_flag_user`, respectively. rocjitsu's shared handlers (lines 11899 and
  12194, fingerprint in `ARCHITECTURE_AUDIT.md`) route both through its
  architecture/IEEE-aware FMA model. Existing callers now explicitly select
  the RDNA4 forms.
- Guest input/output-denormal flushing is implemented in the shared scalar,
  eight-lane AVX2 and sixteen-lane AVX-512 literal paths, including flushing
  the scalar literal before arithmetic. IEEE mode is accepted. The result
  contract remains loose: NaN payload/quieting need not match the architecture;
  both architecture APIs currently share that approximate arithmetic kernel.
  Exact arithmetic and faithful reporting remain unimplemented, rather than
  inferred from matching finite-input results. Non-null exact reporting rejects
  before writes; loose reporting leaves the state untouched.
- New tests cover all guest-policy combinations, both literal placements,
  architectures and wave sizes, every CPU level, normal/subnormal/NaN/infinity
  inputs and literals, masks, destination aliases, and unchanged results/state
  on rejected reporting. The reference flushes raw encodings around host FMA
  under the documented loose host-environment requirements. No native RDNA3
  bit-exact result claim is made by these tests.
- Performance against `c82cab2`: eleven alternating CPU-2-pinned samples,
  minimum 50 ms thread CPU time. AVX2 FMAMK/FMAAK medians changed
  13.117 -> 13.518 ns (+3.1%) and 12.602 -> 13.192 ns (+4.7%); AVX-512 changed
  4.712 -> 4.913 ns (+4.3%) and 3.615 -> 3.899 ns (+7.9%). Guest-control checks
  add a fraction of a nanosecond to these short no-modifier kernels. No additional
  duplicate fast kernel was introduced for that sub-nanosecond cost. These
  finite-input results are not a whole-library performance claim. Artifacts:
  `../goc-build/coverage-work/literal-fma32-policy-*`.
- Validation: 958/958 tests pass under Clang 21 and GCC 15 Release; 30/30
  LiteralFma/Arithmetic/ExceptionRegisters tests pass under ASan/UBSan. Seven
  audit-tool tests pass. The reporting-name audit has 632 signatures, 306
  resolved names, zero mismatches and 326 awaiting architecture review.

## Mixed FMA architecture and guest-policy audit

The denormal-policy assumptions in this initial rocjitsu-derived slice were
superseded by the native hardware correction below. In particular, MIXLO/MIXHI
do honor denormal controls, and FP16 input flushing occurs before widening in
all mixed FMA forms. The initial nondefault reporting restriction is now removed
for RDNA4 MIXLO/MIXHI following the new captures.

- `v_fma_mix_f32`, `v_fma_mixlo_f16`, and `v_fma_mixhi_f16` now distinguish
  unsuffixed RDNA3 and `_rdna4` APIs, in both wave widths. The distinction
  includes their native implicit register contracts (`trapsts` versus
  `excp_flag_user`), not just result bits. RDNA3 no longer inherits the RDNA4
  capture-derived guarantee of no MIXLO/MIXHI exception updates: its non-null
  exact reporting is explicitly unsupported.
- rocjitsu shared handlers 11750/11794/11837 (fingerprint in
  `ARCHITECTURE_AUDIT.md`) route MIX_F32 through architecture-aware FP32
  arithmetic, and MIXLO/MIXHI through `fma_f32_to_f16_nearest_environment`.
  The latter does not consume denormal or IEEE controls. GoC retains its
  integer direct-to-half result model and now accepts those ignored controls.
  Its RDNA3 results follow the shared model; no native RDNA3 corpus is claimed.
- MIX_F32 now applies guest input/output flushing in scalar and eight-lane
  AVX2 paths. FP16 inputs are widened first, so an FP16 subnormal is not an
  FP32 subnormal to be flushed. IEEE is accepted; FP32 loose NaN behavior
  remains approximate. MIXLO/MIXHI keep their four-lane loose SIMD path.
- New integer-oracle tests cover both architectures and wave widths, all eight
  source-format and guest-policy combinations, all CPU levels, half selectors,
  ABS/NEG/CLAMP, masks, both available semantics and reporting error atomicity.
  Existing RDNA4 numeric, DPP and no-exception captures keep their expectations
  and now explicitly call the suffixed APIs. DPP validation covers both names.
- RDNA4 MIXLO/MIXHI no-exception reporting remains enabled only for the captured
  default guest controls. Nondefault flags currently reject non-null exact
  reporting pending hardware verification; null reporting still computes the
  result model and loose requests always skip reporting. Existing captures do
  not by themselves establish exception behavior under every new guest setting.
- Keeping runtime guest checks in the default MIX_F32 SIMD kernel cost 7–16%
  in an initial timing check. The final implementation selects default versus
  guest-controlled SIMD before the loop. Both compile from one source template;
  the default entry retains its original six-argument interface and skips the
  new checks. SIMD object text grows from 9,499 to 10,495 bytes. The FP16-output
  kernel is unchanged.
- Final timing against `93003ea`: eleven alternating CPU-2-pinned samples with
  at least 50 ms thread CPU time. Loose AVX2 MIX_F32 medians changed
  18.725 -> 18.799 ns (+0.4%) for all-FP32 inputs and 21.191 -> 21.567 ns
  (+1.8%) for mixed inputs with NEG/CLAMP. MIXLO medians changed
  70.097 -> 70.113 ns (+0.02%) and 80.540 -> 80.897 ns (+0.4%). These are
  finite-input checks of the existing default paths, not a whole-library
  performance claim. Artifacts: `../goc-build/coverage-work/mixed-fma-policy-*`.
- Validation: all 959 tests pass on Clang 21 and GCC 15 Release; all 24
  MixedFma/ExceptionRegisters/InstructionFlags tests pass under ASan/UBSan.
  Seven audit-tool tests pass. The reporting-name audit now has 638 signatures,
  318 resolved names, zero mismatches and 320 pending architecture review.

## Native mixed-FMA denormal policies and exception reporting

- Extended `tests/capture_arithmetic_exceptions.py` so the generated probe accepts
  a runtime FP_DENORM nibble (0..15); default captures remain unchanged. Added
  the FP32 mixed-FMA family to the generator. RX 9070/gfx1201 captures exercise
  all sixteen independent FP32/FP16 input/output-denormal settings, both output
  halves, source formats/half selections, ABS/NEG/CLAMP, FP16 saturation, 4,096
  edge triples and 4,096 pseudorandom triples.
- All 16,777,216 MIXLO/MIXHI exception reads were zero. RDNA4 optional exact
  reporting therefore now accepts guest denormal flags and leaves the register
  unchanged. RDNA3 reporting remains unverified and explicitly unsupported.
- The captures contradicted rocjitsu's ignored-denormal mixed-half model. Input
  flushing uses each selected source's original FP16/FP32 type; output flushing
  uses the destination type. Corrected scalar and SIMD paths accordingly,
  including FP16 sources of MIX_F32. Output-half rounding retains the integer
  exact intermediate; the four-lane loose kernel implements the same flushing.
- `tests/mixed_fma_guest_hardware.h` stores word-wise result/exception digests
  for all 4,194,304 records with the four coupled denormal policies exposed by
  GoC. Tests reproduce those digests on exact scalar and loose SIMD paths at all
  available CPU levels. Another native probe covers both signs around the FP16
  normal boundary: flushing begins below `2^-14 - 2^-26`, rather than merely
  testing whether the final FP16 encoding is subnormal. Literal boundary tests
  and an independent integer oracle cover this distinction. Host-FP-control
  preservation tests now include guest denormal flags.
- Default SIMD kernels remain separate compiled specializations of the same
  source loops; the new checks run only for nondefault policies. Independent
  FP16 versus FP32 denormal selections are not yet exposed by the public flags.
  A further 8,388,608 MIX_F32 flag reads were zero; its general exact numerical
  and reporting contracts remain unfinished. No RDNA3 hardware claim follows
  from these RDNA4 captures.
- Raw capture summaries, generated probes and boundary values are in
  `../goc-build/coverage-work/mixed-fma-guest-captures/`; the source generator
  and compact regression fixtures are checked in. The implementation continues
  to leave host FP controls untouched; only the GPU capture probes set MODE.
- Default-path performance against `13f6238`: eleven alternating CPU-2-pinned
  samples, at least 50 ms thread CPU time. MIX_F32 AVX2 medians were
  18.688 -> 18.679 ns and 21.444 -> 21.465 ns; MIXLO medians were
  72.021 -> 71.169 ns and 80.396 -> 80.205 ns. Changes are within about 1.2%
  and do not establish a meaningful regression. These finite-input timings
  cover existing default loose paths; they are not a whole-library result.
  Timing artifacts use `../goc-build/coverage-work/mixed-guest-fix-*`.
- Validation: all 961 tests pass under Clang 21 and GCC 15 Release. All 35
  MixedFma/Interp16/ExceptionRegisters tests pass under ASan/UBSan, including
  the shared interpolation users of the modified rounding helper. Capture
  generator syntax and generated GPU probes were also exercised successfully.
