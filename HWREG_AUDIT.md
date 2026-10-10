# Hardware-register implementation audit

This is an implementation checklist, not a coverage-completion claim.
Both architectures have `S_GETREG_B32`, `S_SETREG_B32`, and
`S_SETREG_IMM32_B32` APIs. Reads extract fields from supplied wave-visible
register values. Exact scratch writes, RDNA3 `FLUSH_IB` reporting, and RDNA4
snapshot reset/unlock reporting remain unimplemented.

Evidence inspected: AMD RDNA3 and RDNA4 ISA manuals, sections 3.4 and 16.2;
`amdgpu_isa_rdna3.xml` and `amdgpu_isa_rdna4.xml`, `OPR_HWREG`; and rocjitsu
revision `ffc144c564c`, `lib/rocjitsu/src/rocjitsu/vm/amdgpu/hwreg.cpp`.
The source copies are in the sibling TheRock checkout. Local manual text copies
are `../goc-build/coverage-work/rdna3.txt` and `/tmp/goc-rdna4-mirror.txt`.

## Register-ID coverage

The table includes every predefined ID in both XML operand maps: 24 RDNA3
entries and 19 RDNA4 entries. RDNA4 manual section 3.4 additionally identifies
ID 28 as `IB_STS2`; that prose-only entry is marked explicitly. A dash means
absent from that architecture's XML map, not permission to return zero.
Names retain XML spelling, including `WAVE_` and `SHADER_` prefixes. Manual
register names sometimes omit these prefixes.

| ID | RDNA3 XML | RDNA4 XML / prose |
|---|---|---|
| 1 | `MODE` | `WAVE_MODE` |
| 2 | `STATUS` | `WAVE_STATUS` |
| 3 | `TRAPSTS` | — |
| 4 | — | `WAVE_STATE_PRIV` |
| 5 | `GPR_ALLOC` | `WAVE_GPR_ALLOC` |
| 6 | `LDS_ALLOC` | `WAVE_LDS_ALLOC` |
| 7 | `IB_STS` | — |
| 8 | `PC_LO` | — |
| 9 | `PC_HI` | — |
| 10 | — | `PERF_SNAPSHOT_DATA` |
| 11 | — | `PERF_SNAPSHOT_PC_LO` |
| 12 | — | `PERF_SNAPSHOT_PC_HI` |
| 13 | `IB_DBG1` | — |
| 14 | `FLUSH_IB` | — |
| 15 | `SH_MEM_BASES` | `PERF_SNAPSHOT_DATA1` |
| 16 | `SHADER_TBA_LO` | `PERF_SNAPSHOT_DATA2` |
| 17 | `SHADER_TBA_HI` | `WAVE_EXCP_FLAG_PRIV` |
| 18 | `PERF_SNAPSHOT_PC_LO` | `WAVE_EXCP_FLAG_USER` |
| 19 | `PERF_SNAPSHOT_PC_HI` | `WAVE_TRAP_CTRL` |
| 20 | `SHADER_FLAT_SCRATCH_LO` | `WAVE_SCRATCH_BASE_LO` |
| 21 | `SHADER_FLAT_SCRATCH_HI` | `WAVE_SCRATCH_BASE_HI` |
| 23 | `HW_ID1` | `WAVE_HW_ID1` |
| 24 | `HW_ID2` | `WAVE_HW_ID2` |
| 25 | `POPS_PACKER` | — |
| 26 | `SCHED_MODE` | — |
| 27 | `PERF_SNAPSHOT_DATA` | — |
| 28 | `IB_STS2` | `IB_STS2` (manual only) |
| 29 | `SHADER_CYCLES` | `SHADER_CYCLES_LO` |
| 30 | — | `SHADER_CYCLES_HI` |

## Confirmed architecture differences

- The same ID can name unrelated registers: ID 18 selects
  `PERF_SNAPSHOT_PC_LO` on RDNA3 and `EXCP_FLAG_USER` on RDNA4. ID 17 selects
  `SHADER_TBA_HI` versus `EXCP_FLAG_PRIV`. All three instruction entry points
  therefore require an RDNA4 variant, with the unsuffixed name denoting RDNA3.
- RDNA3 `STATUS` permits selected privileged writes and contains `SCC` at bit
  0. RDNA4 `STATUS` is read-only; `SCC` moves to `STATE_PRIV` bit 9, which is
  writable only with `STATUS.PRIV` set. Returning raw RDNA3 status bits through
  an RDNA4 API would be incorrect.
- RDNA3 manual Table 5 marks these `STATUS` bits privileged-writable:
  0–4, 8, 13, 18, 20–21, 25, 27, 29. Other bits must be preserved. This is a
  field-specific rule, not permission to overwrite the entire register.
- RDNA3 `MODE` includes `DX10_CLAMP`, `IEEE`, trap-after-instruction and
  exception-enable fields. RDNA4 distributes exception controls into
  `TRAP_CTRL` and exception state into `EXCP_FLAG_USER`/`EXCP_FLAG_PRIV`.
  RDNA4 `MODE` instead documents FP round/denormal controls, `FP16_OVFL`,
  scalar-prefetch enable and performance disable. Do not reuse one whole-word
  writable mask across architectures.
- RDNA3 `FLAT_SCRATCH` is a 64-bit, 256-byte-aligned address with privileged
  writes. RDNA4 calls it `SCRATCH_BASE`. These are raw guest registers; updating
  them must not allocate or access host scratch memory.

## Read side effects and external state

- **RDNA4 GETREG is not universally read-only.** The performance-snapshot
  description in section 3.4.9 says to read `PERF_SNAPSHOT_PC_HI` last because
  that read resets/unlocks the snapshot registers for another capture. It also
  says only the sampled wave reads snapshot data; other waves read zero. A
  plain register snapshot without sampled-wave identity and lifecycle state
  cannot model this complete instruction behavior.
- The prose does not identify a separate architectural lock register, nor does
  it specify all post-read register values. Do not invent a named output
  register or assume that clearing `PERF_SNAPSHOT_DATA.VALID` fully models the
  unlock. Resolve this with an additional authoritative model or capture before
  claiming exact support for these reads.
- RDNA3 XML lists snapshot registers, but the inspected RDNA3 manual text does
  not describe the RDNA4 read-unlock protocol. Do not silently apply it to
  RDNA3 as though verified there.
- `SHADER_CYCLES` needs a caller-supplied guest clock value. Do not substitute
  the host wall clock or CPU cycle counter. RDNA4 shader reads of HI/LO are
  separate and may observe rollover; the documented freeze/unfreeze protocol
  in XML is for **host** access, not shader `S_GETREG_B32`.
- RDNA4 XML describes a 60-bit shader-cycle counter while the manual calls it
  64-bit. Raw caller-provided halves can preserve values without inventing
  the counter's progression or overflow model; the width discrepancy remains
  relevant to a full guest-clock implementation.
- Dynamic allocation, hardware identity, PC, status and scratch values must
  come from the emulator. GoC cannot infer them from arithmetic flags.

## Write semantics and non-register effects

- Decode `ID = imm16[5:0]`, `offset = imm16[10:6]`, and
  `size = imm16[15:11] + 1`. Sizes extend up to 32. Clip the accessed field at
  the end of the 32-bit register; avoid C/C++ shifts by 32.
- SETREG uses the shifted field mask intersected with the register's writable
  mask and privilege permissions, then merges into the old value. The RDNA3
  printed pseudocode omits the mask shift even though the operation's prose
  requires it. RDNA4 pseudocode and rocjitsu's field insertion include it.
  Treat this as a documentation discrepancy, not an architecture difference.
- A read-only field or a field unavailable at the current privilege is masked
  out by the manual's SETREG pseudocode. Rocjitsu's API-level `ReadOnly` and
  `Privileged` errors are simulator diagnostics, not proof that the hardware
  traps or that GoC should return the same error for a permitted no-op write.
- SETREG to an exception register performs masked replacement, allowing flags
  to be cleared. It must not reuse arithmetic's sticky-OR update operation.
- RDNA3 `FLUSH_IB` is write-only and has an instruction-buffer flush effect.
  Section 3.4's overview says “all waves”; its internal-register table describes
  the targeted wave and a bit-0 trigger. Its scope needs reconciliation, and
  merely storing a new integer value does not implement the instruction.
- Changing guest `MODE` must only update guest state. It must never set host
  rounding, denormal controls, exception masks or exception flags. The emulator
  remains responsible for deriving subsequent GoC call flags from guest mode.
- HALT, trap-control and scratch changes have emulator consequences beyond
  their raw bits. The library should report architectural register updates;
  the emulator applies scheduling, trap delivery and memory-model effects.

## What can and cannot be borrowed from rocjitsu

- Reuse the established descriptor decoding, clipped field extraction/insertion
  and register-ID maps. Translate its legacy internal status/exception storage
  only if the GoC API actually uses that representation; raw architectural
  register inputs need no such translation.
- Its RDNA3 `STATUS` descriptor is read-only despite the manual's privileged
  writable bits. Its RDNA3 scratch descriptors are likewise read-only despite
  privileged writes in the manual. These policies cannot be copied unchanged.
- Many register descriptors have `HwregState::Unsupported`, including snapshot
  and allocation state. Their presence in a table is not an implementation.
- Its supported write path is largely whole-value merge plus state dispatch;
  that does not establish reserved-bit write masks for every register.

## gfx1201 field-access capture

`tests/capture_hwreg.py OUTPUT_DIR` generates a HIP probe for user-mode access
to `MODE` (ID 1) and `EXCP_FLAG_USER` (ID 18). Each case saves and restores the
selected GPU register. It records the initialized full register, the updated
full register, and the field subsequently read by `S_GETREG_B32`.

The capture covers both zero/all-one initialization, four SGPR source patterns
(`0`, `0xffffffff`, `0xa5a5a5a5`, `0x5a5a5a5a`), the literal-all-one SETREG
variant, and all 32 offsets × 32 encoded widths. All **20,480 triples** match
the independent integer checks in `tests/check_hwreg_capture.py`:

| Register | Observed writable mask | Fields |
|---|---|---|
| `MODE` | `0x098000ff` | FP controls 0–7, `FP16_OVFL` 23, scalar prefetch 24, performance disable 27 |
| `EXCP_FLAG_USER` | `0xc000007f` | Arithmetic exceptions 0–6, buffer OOB 30, LOD clamped 31 |

For these registers, shifted field masks are intersected with the writable
mask; fields extending beyond bit 31 are clipped, including width-32 fields
with nonzero offsets. Reads right-shift then mask the requested width. Literal
and SGPR writes with the same source agree. Zero writes clear selected exception
flags, while unselected writable bits survive. Reserved bits read back zero
after both initializations and every update in this capture.

These results do **not** establish RDNA3 masks, privilege filtering, read-only
register writes, snapshot ownership/unlock, or instruction-buffer flush effects.
RDNA3 `MODE.EXCP_EN` also needs care: its field spans bits 12–21, but the
individual enable list omits bit 20. Section 3.4.9 explicitly calls exception-
enable index 8 reserved, resolving the ambiguity: physical MODE bit 20 is
excluded from the implemented writable mask. A union of broad field widths
alone would incorrectly include it. This is manual evidence, not an RDNA3 capture.

Local capture: `../goc-build/coverage-work/hwreg-report-captures/capture.bin`.
SHA256: `399b3ca4c4f9eacfe1bc7b94c6d3ff6c575f2ec8855566dae46d972019b0e1a8`.
Recheck with:

```sh
python3 tests/check_hwreg_capture.py ../goc-build/coverage-work/hwreg-report-captures/capture.bin
```

## API and validation gates

### Implemented reads and their input boundary

`goc_s_getreg_b32` selects RDNA3's register map;
`goc_s_getreg_b32_rdna4` selects RDNA4's. Both share clipped integer field
extraction. They take the required destination and encoded selector first, then
by-value implicit inputs as specified by PLAN.md. Logical 64-bit input pairs
such as scratch and the guest clock are passed as `uint64_t`; the descriptor
selects the relevant half. RDNA3 exposes only the low clock half. Unknown IDs,
including RDNA3's write-only `FLUSH_IB`, return an API validation error without
changing D. This is not a claim about hardware execution of invalid selectors.

Inputs are **wave-visible read values**, not a physical machine-wide snapshot
bank: the emulator supplies allocation, PC, identity, guest time, and the
snapshot values visible to the calling wave. In particular, it supplies zero
snapshot values for a non-sampled wave and zero scratch for a wave without
allocation. GoC does not create a guest clock or infer snapshot ownership from
partial identity bits. Numerical extraction is supported, while constructing
these architectural read views remains the emulator's responsibility.

RDNA4 has five trailing nullable outputs, one for each named snapshot register
(`PERF_SNAPSHOT_DATA`, `PERF_SNAPSHOT_PC_LO`, `PERF_SNAPSHOT_PC_HI`,
`PERF_SNAPSHOT_DATA1`, `PERF_SNAPSHOT_DATA2`). Reading ID 12 with any requested
output in non-loose semantics returns `GOC_ERROR_UNSUPPORTED_GLOBAL_STATE`,
before validation or D access. Null opts out; loose mode skips state updates.
Other reads leave snapshot outputs unchanged. Required numerical inputs are
separate from these pointers, so opting out cannot suppress the read result.
No invented lock register or guessed post-read values were added. The inspected
RDNA3 specification does not establish the RDNA4 read-update protocol; its API
does not silently model that protocol or claim it has been hardware-validated.

Tests cover all 64 selector IDs, all 32 offsets and widths, both semantics and
architecture maps; clock-half rollover inputs; input/output aliases; errors;
all 32 snapshot-output pointer combinations; and host FP-state preservation.
The RDNA4 MODE/EXCP_FLAG_USER read results also match all 20,480 captured field
values. These tests do not validate snapshot generation, ownership routing,
reset/unlock, or native RDNA3 reads.

### Implemented RDNA4 writes

`goc_s_setreg_b32_rdna4` and `goc_s_setreg_imm32_b32_rdna4` share one decoder.
The explicit operands are the encoded HWREG selector and scalar/literal source;
the implicit `STATUS` input is by value. Six trailing nullable outputs represent
`MODE`, `STATE_PRIV`, `EXCP_FLAG_PRIV`, `EXCP_FLAG_USER`, `TRAP_CTRL` and the
64-bit `SCRATCH_BASE`. No context struct or per-half scratch output is introduced.
The selected pointer supplies the old register value and receives a masked
replacement. Other pointers are not accessed. Cheap updates are performed in
both semantics; null opts out.

In addition to the two captured user masks above, privileged masks follow the
RDNA4 manual's named register fields: `STATE_PRIV = 0x00047e07`,
`EXCP_FLAG_PRIV = 0xc0000fff` and `TRAP_CTRL = 0x000003ff`.
`STATUS.PRIV` (bit 5) enables these writes.
These privileged masks have **manual/reference-test
evidence, not native privileged hardware captures**. The implementation reports
raw register changes; scheduling, trap delivery and scratch memory are external.

Privileged scratch writes remain unsupported in non-loose semantics when either
`scratch_base` or `state_priv` is non-null. Loose semantics skip them. The manual
requires an aligned address, but that alone does not establish a low-bit write
mask or whether writes affect `STATE_PRIV.SCRATCH_EN`. Do not silently turn the
alignment requirement into a hardware mask. Unprivileged scratch writes are
permitted no-ops. The native-width API is in place for the remaining work.

All 20 recognized IDs (including read-only IDs 5/6 and prose-only ID 28) are
handled. Read-only and privilege-denied writes succeed without touching state.
Unknown/reserved IDs return `GOC_ERROR_INVALID_FLAGS`, an API validation policy,
not a claim about how hardware executes an invalid selector. Tests cover all
64 IDs × 32 offsets × 32 widths in both privilege states, both SETREG spellings,
unselected/reserved-bit preservation, null outputs, validation and unchanged
host FP controls. The two user-register paths also match captured hardware
hashes. GETREG snapshot side effects remain outside this implemented write path.

### Implemented RDNA3 writes

`goc_s_setreg_b32` and `goc_s_setreg_imm32_b32` are the unsuffixed RDNA3 APIs.
They expose by-value `status_in` for privilege and separate trailing native
outputs `mode`, `status`, `trapsts`, `flush_ib`, and 64-bit `flat_scratch`.
The literal form delegates to the same decoder. All 24 RDNA3 XML IDs are
recognized; unknown IDs return an API validation error.

Implemented masks come from RDNA3 manual sections 3.4.1, 3.4.2 and 3.4.9:

| Register | Writable mask | Permission |
|---|---|---|
| `MODE` | `0x08afffff` | User or trap handler; bit 20 reserved |
| `STATUS` | `0x2a34211f` | Only with incoming `STATUS.PRIV` set |
| `TRAPSTS` | `0x0017fdff` | User or trap handler |

STATUS follows Table 5's explicit per-field write permissions, including a
read-only PRIV bit. TRAPSTS writes replace selected exception/status bits,
allowing clears; arithmetic's sticky-OR operation is not reused. These masks
have manual and independent field-reference tests, **not native RDNA3 hardware
capture verification**. RDNA3 TBA is treated as read-only because section 3.4.9
says TMA/TBA are read-only; rocjitsu's privileged-write descriptor is not copied.

`FLUSH_IB` is a named write-only target, so its API has a nullable output, but
no persistent value or instruction-buffer effect is guessed. Requested
non-loose reporting returns `GOC_ERROR_UNSUPPORTED_GLOBAL_STATE`; loose mode
skips it. Privileged scratch writes similarly reject non-loose requests when
either `flat_scratch` or `status` is non-null, since the possible SCRATCH_EN
update is also unresolved. Null opts out; unprivileged scratch writes are no-ops.
These errors precede other validation and leave all outputs untouched.

Tests cover all IDs, offsets, widths, both privilege levels and semantics,
SGPR/literal equivalence, per-field write permissions, reserved-bit preservation,
exception clearing, native-register opt-outs, validation ordering and host FP
control preservation. Scheduling and trap-delivery effects of updated register
bits remain the emulator's responsibility.

### Remaining gates

- Preserve one trailing nullable output pointer per architectural register that
  the instruction may write. Do not put writable state in an opaque context
  object. Required instruction results remain required; optional side-output
  fidelity follows the existing loose/exact contract.
- Keep GETREG's by-value input boundary explicit: the present API follows PLAN.md
  and does not introduce a context object. Inputs must represent the calling
  wave's architectural read values, not unfiltered physical snapshot storage.
- Before declaring coverage, test every recognized ID for each architecture,
  all 32 offsets and 32 encoded widths, size-32 and clipped fields, SGPR and
  literal SETREG equivalence, privilege filtering, unaffected-bit preservation,
  exception clearing, aliases, and unchanged outputs on errors.
- Also test MODE writes under nondefault host FP settings without host control
  changes, snapshot ownership/unlock, and externally supplied clock rollover.
  Unsupported side effects must remain explicit; do not label the three
  inventory rows complete merely because symbols or extraction helpers exist.
