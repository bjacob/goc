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

## Evidence and starting point

- Start: `80470bd`, 682 instruction APIs, 674 tests passing with Clang and GCC.
- ISA and rocjitsu source: local TheRock `rocm-systems` revision `ffc144c564c`.
  XML content fingerprints are recorded in the inventory.
- Borrowed hardware captures establish only the targets and cases they capture.
  No new GPU captures have been made for this goal.
- Inventory initially finds 804 in-scope canonical mnemonics in the union:
  664 have matching API names; 140 lack them. Another 14 control/register
  boundaries require explicit review. 578 memory/control/scheduling mnemonics
  are excluded. These counts are neither coverage percentages nor correctness claims.

## Cross-cutting completion gates

- [x] Enumerate both XML instruction sets and identify matching public API names.
- [ ] Resolve every boundary row, including relative register addressing and
  hardware-register access, without excluding register operations for convenience.
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
| DOT2 and DOT2ACC | FP32 DOT2 split into unsuffixed RDNA3 / `_rdna4`; shared modifiers/DPP/loose SIMD; GFX1100 captures, aliases and host-state tests | Add accumulator/dual forms; RDNA3 reporting uncharacterized; Wave64 |
| Floating min/max/median | Existing RDNA4 NUM forms | RDNA3 names and NaN/zero policies, signaling NaNs, all widths and packed forms |
| Scalar integer arithmetic | RDNA3 add/sub/addc/subb/addk spellings reuse RDNA4 implementations, as confirmed by XML aliases; all twelve CMPK forms added with exhaustive immediate tests | Continue full family audit; register/control operations tracked separately |
| Scalar/register moves | MOV/CMOV B32/B64 and MOVK/CMOVK implemented for both architectures; exhaustive immediate/condition tests | Relative addressing and hardware-register access; SAVEEXEC/WREXEC implemented at both register widths |
| Vector/register routing | READLANE/WRITELANE/READFIRSTLANE now shared for Wave32/64, including lane-index wrapping and zero EXEC; existing DPP/permlane/move coverage | Swaps, relative moves, permlane64, broader cross-width behavior |
| Comparisons | Existing RDNA4 results/SCC/EXEC and exception tests | RDNA3 integer true/false forms now implemented; floating constant forms, Wave64, and architecture-specific floating policies remain |
| Basic floating arithmetic, FMA, conversions | Existing RDNA4 implementations and captures | Audit shared numerical behavior and latest rocjitsu exact models for both targets |
| Transcendentals, reciprocal, division | Existing partial exact coverage | Review newer rocjitsu models/captures, differing semantics and state outputs |
| Integer/vector arithmetic and bit operations | Existing broad RDNA4 coverage | Check RDNA3 spellings, modifier encodings, widths, carry outputs and all aliases |
| Interpolation | Existing RDNA4 paths | Compare RDNA3 contracts and empirical behavior |
| Dual-issue VOPD | No separately named APIs | Inventory independent operations and cross-operation source/output dependencies |

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
