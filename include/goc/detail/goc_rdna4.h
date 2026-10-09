// SPDX-License-Identifier: MIT

#ifndef GOC_RDNA4_H_
#define GOC_RDNA4_H_

#include "goc_export.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Wave32 entry points. High exec_mask bits are ignored. Each pointer names 32
// contiguous uint32_t lane words. Pointer arrays and backing storage must be valid.
// Whole VGPRs may alias; distinct VGPR addresses must not overlap. Sources are
// conceptually read before writes. Inactive destination lanes and all destinations
// on error are unchanged. Loose FP32 paths require host nearest-even rounding
// with denormals enabled. Integer arithmetic paths preserve all host FP state.

// Combined three-input integer operations, one VGPR per operand. Shift
// counts wrap modulo 32; all addition and shifting wrap to 32 bits. Instruction
// flags must be zero. Loose semantics only; preserves all host FP state.

// (A << (B & 31)) + C.
GOC_API int goc_rdna4_v_lshl_add_u32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);

// (A + B) << (C & 31).
GOC_API int goc_rdna4_v_add_lshl_u32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);

// (A << (B & 31)) | C.
GOC_API int goc_rdna4_v_lshl_or_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

// (A & B) | C.
GOC_API int goc_rdna4_v_and_or_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c);

// A | B | C.
GOC_API int goc_rdna4_v_or3_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

// A ^ B ^ C.
GOC_API int goc_rdna4_v_xor3_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

// (A ^ B) + C.
GOC_API int goc_rdna4_v_xad_u32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

// Each result byte is floor((A_byte + B_byte + (C_byte & 1)) / 2).
// Higher C-byte bits are ignored. Carries do not cross byte boundaries.
GOC_API int goc_rdna4_v_lerp_u8(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

// Sum of unsigned absolute differences between packed fields of A and B, plus
// C. SAD_U8 sums four byte differences, SAD_U16 two halfword differences, and
// SAD_U32 one full-word difference. SAD_HI_U8 shifts the byte sum left by 16
// before accumulation. MSAD_U8 omits differences where the corresponding B
// byte is zero. These forms use one VGPR for each operand.
// GOC_ALU_CLAMP saturates the final unsigned accumulation; all other instruction
// flags are invalid. Loose semantics only; preserves all host FP state.
GOC_API int goc_rdna4_v_sad_u8(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);
GOC_API int goc_rdna4_v_sad_hi_u8(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c);
GOC_API int goc_rdna4_v_sad_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);
GOC_API int goc_rdna4_v_sad_u32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);
GOC_API int goc_rdna4_v_msad_u8(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

// Quad SAD compares four overlapping four-byte windows of A (starting at
// byte offsets 0, 1, 2 and 3) against B. A has two VGPRs, low word first; B has
// one. QSAD uses all B bytes; MQSAD omits zero B bytes. PK_U16 packs four
// independent 16-bit accumulations into two C/D VGPRs. U32 uses four C/D VGPRs.
// GOC_ALU_CLAMP saturates each accumulation to its destination width; otherwise
// each wraps independently. The high byte of A's second VGPR is unused.
// All other instruction flags are invalid. Loose semantics only; preserves
// all host floating-point state.
GOC_API int goc_rdna4_v_qsad_pk_u16_u8(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c);
GOC_API int goc_rdna4_v_mqsad_pk_u16_u8(uint64_t flags, uint64_t exec_mask,
                                        uint32_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c);
GOC_API int goc_rdna4_v_mqsad_u32_u8(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);

// Reverse shifts: shift B by the count in A. A is one VGPR. B and D are one
// VGPR for 32-bit forms or two VGPRs (low word first) for 64-bit forms. Counts
// wrap modulo 32 or 64. ASHR replicates the sign bit; logical shifts insert
// zero bits. Instruction flags must be zero. Loose semantics only; preserves
// all host floating-point state.
GOC_API int goc_rdna4_v_lshlrev_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);
GOC_API int goc_rdna4_v_lshrrev_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);
GOC_API int goc_rdna4_v_ashrrev_i32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);
GOC_API int goc_rdna4_v_lshlrev_b64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);
GOC_API int goc_rdna4_v_lshrrev_b64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);
GOC_API int goc_rdna4_v_ashrrev_i64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

// FP32 sine/cosine of inputs measured in turns: sin(2*pi*A), cos(2*pi*A).
// Supports ABS_A, NEG_A, OMOD and CLAMP. Empirical exact semantics use the
// captured RDNA3/4 integer model with denormals preserved and NaNs quieted;
// all host floating-point state is preserved. Output scaling uses nearest-even
// rounding, followed by CLAMP. Active OMOD flushes subnormal outputs and both
// signed zeros to +0. Loose SIMD semantics approximate the captured polynomial
// and require host nearest-even rounding with denormals enabled.
GOC_API int goc_rdna4_v_sin_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);
GOC_API int goc_rdna4_v_cos_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);

// FP16 sine/cosine of inputs measured in turns. Supports ABS_A, NEG_A, OMOD,
// CLAMP and HIGH_A/D; preserves the unselected D half. Rounds to FP16 before
// OMOD. Active OMOD flushes tiny inputs to scaling and tiny scaled results to
// +0. Empirical exact semantics use rocjitsu's captured model with denormals
// preserved; all host floating-point state is preserved.
// Loose SIMD semantics require host nearest-even rounding with denormals enabled.
// GOC_FP16_OVFL is accepted and has no effect on the bounded finite results.
GOC_API int goc_rdna4_v_sin_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);
GOC_API int goc_rdna4_v_cos_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);

// FP16 fused multiply-add: selected halves of A/B/C/D, all ALU source/output
// modifiers and GOC_FP16_OVFL. Preserves the other D half and inactive lanes.
// Arithmetic rounds to FP16 before OMOD; active OMOD flushes tiny arithmetic
// results to +0 and newly tiny scaled results to signed zero. CLAMP is last.
// Exact empirical semantics follow rocjitsu's RNE/denormal-preserving model,
// including NaN payload priority, and preserve the host floating-point environment.
// Loose semantics require host nearest-even arithmetic with denormals enabled.
GOC_API int goc_rdna4_v_fma_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

// FMA supports all GOC_ALU source/output modifiers. Exact semantics requests
// fall back to loose unless GOC_SEMANTICS_STRICT is set.
GOC_API int goc_rdna4_v_fma_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

// FP16 fused multiply-accumulate into D. Supports A/B ABS/NEG, OMOD, CLAMP,
// HIGH_A/B/D and GOC_FP16_OVFL, with the same rounding and exact-semantics
// contract as FP16 FMA. HIGH_D selects both the accumulator and result half;
// preserves the other half. C modifiers (including HIGH_C) are invalid.
GOC_API int goc_rdna4_v_fmac_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b);

// FP32 fused multiply-accumulate into D. Supports A/B ABS/NEG, OMOD and CLAMP.
// C modifiers and half selectors are invalid. Loose semantics only.
GOC_API int goc_rdna4_v_fmac_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b);

// Literal FP16 FMA: FMAMK computes A * literal + B; FMAAK computes A * B +
// literal. The literal is a raw FP16 encoding. HIGH_A/B/D select the two VGPR
// inputs and destination half; the other D half is preserved. Other instruction
// flags are invalid. Supports GOC_FP16_OVFL and empirical exact semantics with
// the FP16 FMA rounding and host-environment contract.
GOC_API int goc_rdna4_v_fmamk_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint16_t literal,
                                  const uint32_t *const *b);

GOC_API int goc_rdna4_v_fmaak_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint16_t literal);

// Literal FP32 FMA: FMAMK computes A * literal + B; FMAAK computes A * B +
// literal. The literal is a raw FP32 encoding. No instruction flags are valid.
// Loose semantics only; requires host nearest-even rounding and denormals enabled.
GOC_API int goc_rdna4_v_fmamk_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                                  const uint32_t *const *b);

GOC_API int goc_rdna4_v_fmaak_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, uint32_t literal);

// DX9 FMA: one VGPR per operand, all ALU source/output modifiers, loose semantics.
// If either modified factor is signed zero, select modified C unchanged before
// output scaling/CLAMP; otherwise compute a fused multiply-add.
GOC_API int goc_rdna4_v_fma_dx9_zero_f32(uint64_t flags, uint64_t exec_mask,
                                         uint32_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c);

// Packed FP16 FMA: two independent fused results per lane. All GOC_PK_* flags
// below are supported. With no flags, corresponding input halves are multiplied
// and added. Each result rounds to FP16; CLAMP applies last. Supports
// GOC_FP16_OVFL and empirical exact semantics with the FP16 FMA environment
// contract. Both halves use the original inputs, including when D aliases A/B/C.
GOC_API int goc_rdna4_v_pk_fma_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c);

// Packed FP16 FMAC: component-wise A * B + D. No instruction flags are valid.
// Supports GOC_FP16_OVFL and the same loose/exact semantics as packed FMA.
GOC_API int goc_rdna4_v_pk_fmac_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

// Packed FP16 binary arithmetic: supports GOC_PK_* negation and half selectors
// for A/B, plus GOC_PK_CLAMP and GOC_FP16_OVFL. Flags for C are invalid. Each
// result rounds to FP16; the two results use original inputs even with aliases.
// Number min/max ignore a lone NaN; minimum/maximum propagate NaNs. Both order
// -0 below +0. Loose semantics require host nearest-even with denormals enabled.
GOC_API int goc_rdna4_v_pk_add_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_pk_mul_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_pk_min_num_f16(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b);

GOC_API int goc_rdna4_v_pk_max_num_f16(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b);

GOC_API int goc_rdna4_v_pk_minimum_f16(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b);

GOC_API int goc_rdna4_v_pk_maximum_f16(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b);

// Ordinary 16-bit integer binary arithmetic: one VGPR per operand. HIGH_A/B/D
// select the input and output halves; the other destination half is preserved.
// ADD/SUB accept GOC_ALU_CLAMP for signed/unsigned saturation; otherwise they
// wrap. Other forms do not accept CLAMP. Multiply retains its low 16 bits.
// Shifts use B as the value and A modulo 16 as the count. No other instruction
// flags are valid. Loose semantics only; independent of host FP state.
GOC_API int goc_rdna4_v_add_nc_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_sub_nc_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_add_nc_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_sub_nc_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_min_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

GOC_API int goc_rdna4_v_max_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

GOC_API int goc_rdna4_v_min_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

GOC_API int goc_rdna4_v_max_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

GOC_API int goc_rdna4_v_mul_lo_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_lshlrev_b16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

GOC_API int goc_rdna4_v_lshrrev_b16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

GOC_API int goc_rdna4_v_ashrrev_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

// Integer multiply-add with 32-bit C/D: one VGPR per operand. The 16-bit
// forms support HIGH_A/B; the 24-bit forms discard A/B's upper byte. Signed
// forms sign-extend the selected factors and interpret C as signed 32-bit.
// CLAMP saturates the full product plus C to the result's 32-bit range;
// otherwise results wrap. Other instruction flags are invalid. Loose semantics
// only; independent of host FP state. Whole-register aliases are supported.
GOC_API int goc_rdna4_v_mad_u32_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_mad_i32_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_mad_u32_u24(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_mad_i32_i24(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

// Ordinary 16-bit ternary arithmetic: one VGPR per operand. HIGH_A/B/C/D
// select source and destination halves; the other D half is preserved.
// MAD supports CLAMP after full-precision A * B + C, otherwise wrapping.
// MIN3/MAX3 select the smallest/largest of all three signed/unsigned inputs.
// MED3 selects the middle value. CLAMP is invalid for MIN3/MAX3/MED3.
// No other instruction flags are valid.
// Loose semantics only; independent of host floating-point state.
GOC_API int goc_rdna4_v_mad_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_mad_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_min3_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_min3_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_max3_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_max3_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_med3_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_med3_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

// Packed integer binary arithmetic: one VGPR per operand, two 16-bit results
// per lane. Supports GOC_PK_* half selectors for A/B and GOC_PK_CLAMP. CLAMP
// saturates ADD/SUB to the signed/unsigned 16-bit range; min/max and multiply
// ignore it. Without CLAMP, ADD/SUB wrap. Multiply keeps the low 16 bits.
// Negation and C flags are invalid. Both results read original inputs even
// when D aliases A/B. Loose semantics only; independent of host FP state.
GOC_API int goc_rdna4_v_pk_add_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_pk_sub_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_pk_add_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_pk_sub_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_pk_min_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_pk_max_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_pk_min_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_pk_max_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_pk_mul_lo_u16(uint64_t flags, uint64_t exec_mask,
                                      uint32_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b);

// Packed 16-bit shifts: A supplies counts modulo 16; B supplies values.
// Supports GOC_PK_* half selectors for A/B. CLAMP is accepted and ignored;
// negation and C flags are invalid. ASHR sign-extends each selected B half.
// Two results per lane, one VGPR per operand, loose semantics only.
GOC_API int goc_rdna4_v_pk_lshlrev_b16(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b);

GOC_API int goc_rdna4_v_pk_lshrrev_b16(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b);

GOC_API int goc_rdna4_v_pk_ashrrev_i16(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b);

// Packed integer multiply-add: two results per lane, one VGPR per operand.
// All GOC_PK_* half selectors for A/B/C are supported; negation is invalid.
// CLAMP saturates the full A * B + C result to the signed/unsigned 16-bit range;
// without it, results wrap. Both halves read original sources before D is written.
// Loose semantics only; independent of host floating-point state.
GOC_API int goc_rdna4_v_pk_mad_i16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_pk_mad_u16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c);

// Packed source negation: independent for the low and high result calculations.
static const uint32_t GOC_PK_NEG_LO_A = (UINT32_C(1) << 0);
static const uint32_t GOC_PK_NEG_LO_B = (UINT32_C(1) << 1);
static const uint32_t GOC_PK_NEG_LO_C = (UINT32_C(1) << 2);
static const uint32_t GOC_PK_NEG_HI_A = (UINT32_C(1) << 3);
static const uint32_t GOC_PK_NEG_HI_B = (UINT32_C(1) << 4);
static const uint32_t GOC_PK_NEG_HI_C = (UINT32_C(1) << 5);

// Packed output clamp: floats to [0, 1], with NaNs and -0 to +0;
// integer ADD/SUB/MAD saturate to their result range.
static const uint32_t GOC_PK_CLAMP = (UINT32_C(1) << 6);

// Packed half selectors flip the default choice for each result calculation.
// Zero flags use low inputs for the low result and high inputs for the high one.
// These flags can swap or replicate halves independently for each source.
static const uint32_t GOC_PK_LO_A_HIGH = (UINT32_C(1) << 7);
static const uint32_t GOC_PK_LO_B_HIGH = (UINT32_C(1) << 8);
static const uint32_t GOC_PK_LO_C_HIGH = (UINT32_C(1) << 9);
static const uint32_t GOC_PK_HI_A_LOW = (UINT32_C(1) << 10);
static const uint32_t GOC_PK_HI_B_LOW = (UINT32_C(1) << 11);
static const uint32_t GOC_PK_HI_C_LOW = (UINT32_C(1) << 12);

// Floating ALU source modifiers: ABS precedes NEG.
static const uint32_t GOC_ALU_NEG_A = (UINT32_C(1) << 0);
static const uint32_t GOC_ALU_NEG_B = (UINT32_C(1) << 1);
static const uint32_t GOC_ALU_NEG_C = (UINT32_C(1) << 2);
static const uint32_t GOC_ALU_ABS_A = (UINT32_C(1) << 3);
static const uint32_t GOC_ALU_ABS_B = (UINT32_C(1) << 4);
static const uint32_t GOC_ALU_ABS_C = (UINT32_C(1) << 5);

// Floating ALU output scaling precedes CLAMP. OMOD is a two-bit enumeration:
// none, multiply by 2, multiply by 4, divide by 2.
static const uint32_t GOC_ALU_OMOD_2 = (UINT32_C(1) << 6);
static const uint32_t GOC_ALU_OMOD_4 = (UINT32_C(2) << 6);
static const uint32_t GOC_ALU_OMOD_HALF = (UINT32_C(3) << 6);

// Result clamping where supported: floating results to [0,1] (NaNs become zero),
// integer results to the representable signed/unsigned destination range.
static const uint32_t GOC_ALU_CLAMP = (UINT32_C(1) << 8);

// Binary FP32 arithmetic: one VGPR per operand. Supports ABS/NEG for A/B,
// OMOD and CLAMP. Flags for C and half selection are invalid. Loose semantics only.
GOC_API int goc_rdna4_v_add_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

GOC_API int goc_rdna4_v_sub_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

GOC_API int goc_rdna4_v_subrev_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_mul_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

// FP32 min/max: one VGPR per operand; supports A/B ABS/NEG, OMOD and CLAMP.
// Loose semantics only. Number variants prefer numeric operands over NaNs;
// minimum/maximum propagate NaNs, preferring signaling NaNs and quieting them.
// Both families order -0 below +0.
GOC_API int goc_rdna4_v_min_num_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

GOC_API int goc_rdna4_v_max_num_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

GOC_API int goc_rdna4_v_minimum_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

GOC_API int goc_rdna4_v_maximum_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

// Three-input FP32 min/max: one VGPR per operand; supports all A/B/C ABS/NEG,
// OMOD and CLAMP, with loose semantics. First select between A/B, then between
// that result and C; output modifiers apply only after both selections.
// MIN3/MAX3 repeat the same selection; MINMAX/MAXMIN apply opposite selections.
// Number/propagating variants follow the binary NaN and signed-zero rules above.
GOC_API int goc_rdna4_v_min3_num_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_max3_num_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_minmax_num_f32(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c);

GOC_API int goc_rdna4_v_maxmin_num_f32(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c);

GOC_API int goc_rdna4_v_minimum3_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_maximum3_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_minimummaximum_f32(uint64_t flags, uint64_t exec_mask,
                                           uint32_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *c);

GOC_API int goc_rdna4_v_maximumminimum_f32(uint64_t flags, uint64_t exec_mask,
                                           uint32_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *c);

// DX9 multiplication: one VGPR per operand, with A/B ABS/NEG, OMOD and CLAMP.
// Either signed-zero input forces a positive-zero product, including with NaN
// or infinity as the other input. Loose semantics only.
GOC_API int goc_rdna4_v_mul_dx9_zero_f32(uint64_t flags, uint64_t exec_mask,
                                         uint32_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b);

// FP32 fractional part: x - floor(x), capped at the largest FP32 value below one
// before output scaling/CLAMP. Supports A ABS/NEG, OMOD and CLAMP, loose semantics.
// One VGPR per operand; infinite inputs produce NaN.
GOC_API int goc_rdna4_v_fract_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

// FP32 median: one VGPR per operand; all A/B/C ABS/NEG, OMOD and CLAMP, loose
// semantics. Any NaN selects minimumNumber across all three inputs. Otherwise
// remove the first input numerically equal to the maximum and select the maximum
// of the remaining two inputs, following the ISA's signed-zero tie behavior.
GOC_API int goc_rdna4_v_med3_num_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);

// FP64 arithmetic: two VGPRs per operand, holding each lane's low/high words.
// Supports ALU ABS/NEG on present sources, OMOD and CLAMP, with loose semantics.
// Host nearest-even rounding and enabled denormals are required. Whole VGPR
// aliases may cross operand halves; sources are read before destination writes.
GOC_API int goc_rdna4_v_add_f64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

GOC_API int goc_rdna4_v_mul_f64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

GOC_API int goc_rdna4_v_fma_f64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

// FP64 min/max uses two VGPRs per operand, with A/B ABS/NEG, OMOD and CLAMP.
// Number variants prefer numeric operands over NaNs; minimum/maximum propagate
// NaNs, preferring signaling NaNs and quieting them. Both order -0 below +0.
// Loose semantics and the same FP64 alias/host-FP-state contract apply.
GOC_API int goc_rdna4_v_min_num_f64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

GOC_API int goc_rdna4_v_max_num_f64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

GOC_API int goc_rdna4_v_minimum_f64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

GOC_API int goc_rdna4_v_maximum_f64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

// Unary FP64 arithmetic uses the same low/high VGPR layout. Supports A ABS/NEG,
// OMOD and CLAMP, with loose semantics. RNDNE rounds ties to even; FRACT computes
// x - floor(x), capped at 0x3fefffffffffffff before output modifiers.
GOC_API int goc_rdna4_v_trunc_f64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_ceil_f64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_rndne_f64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_floor_f64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_fract_f64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_sqrt_f64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_rcp_f64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_rsq_f64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);

// Binary-significand extraction: finite nonzero inputs produce magnitude in
// [0.5,1); zeros, infinities and NaN bits pass through before output modifiers.
// Supports A ABS/NEG, OMOD and CLAMP, with loose semantics. FP32 uses one VGPR
// per operand; FP64 uses low/high VGPR pairs with the FP64 alias contract.
GOC_API int goc_rdna4_v_frexp_mant_f32(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a);

GOC_API int goc_rdna4_v_frexp_mant_f64(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a);

// Binary exponent extraction into one signed 32-bit VGPR. A uses one FP32
// VGPR or an FP64 low/high pair. Returns zero for zeros, infinities and NaNs;
// finite nonzero inputs satisfy A = FREXP_MANT(A) * 2^D, including subnormals.
// A ABS/NEG, OMOD and CLAMP are accepted but do not change the integer result.
// Supports loose semantics; D may alias either whole source VGPR.
GOC_API int goc_rdna4_v_frexp_exp_i32_f32(uint64_t flags, uint64_t exec_mask,
                                          uint32_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a);

GOC_API int goc_rdna4_v_frexp_exp_i32_f64(uint64_t flags, uint64_t exec_mask,
                                          uint32_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a);

// Scale A by 2^B, with a signed 32-bit integer exponent in one B VGPR.
// FP32 A/D each use one VGPR; FP64 A/D use low/high pairs. Supports A ABS/NEG,
// OMOD and CLAMP with loose semantics and gradual underflow. Integer B has no
// sign modifiers. D may alias any whole source VGPR; FP64 writes low then high.
GOC_API int goc_rdna4_v_ldexp_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

GOC_API int goc_rdna4_v_ldexp_f64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

// Signed/unsigned 32-bit integer selection. Each operand uses one VGPR;
// D may alias any whole source VGPR. These instructions have no arithmetic
// modifiers: instruction_flags must be zero. Supports loose semantics.
// MINMAX computes max(min(A, B), C); MAXMIN computes min(max(A, B), C).
GOC_API int goc_rdna4_v_min_i32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b);

GOC_API int goc_rdna4_v_max_i32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b);

GOC_API int goc_rdna4_v_min3_i32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *const *b,
                                 const uint32_t *const *c);

GOC_API int goc_rdna4_v_max3_i32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *const *b,
                                 const uint32_t *const *c);

GOC_API int goc_rdna4_v_minmax_i32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                   const uint32_t *const *a, const uint32_t *const *b,
                                   const uint32_t *const *c);

GOC_API int goc_rdna4_v_maxmin_i32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                   const uint32_t *const *a, const uint32_t *const *b,
                                   const uint32_t *const *c);

GOC_API int goc_rdna4_v_med3_i32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *const *b,
                                 const uint32_t *const *c);

GOC_API int goc_rdna4_v_min_u32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b);

GOC_API int goc_rdna4_v_max_u32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b);

GOC_API int goc_rdna4_v_min3_u32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *const *b,
                                 const uint32_t *const *c);

GOC_API int goc_rdna4_v_max3_u32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *const *b,
                                 const uint32_t *const *c);

GOC_API int goc_rdna4_v_minmax_u32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                   const uint32_t *const *a, const uint32_t *const *b,
                                   const uint32_t *const *c);

GOC_API int goc_rdna4_v_maxmin_u32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                   const uint32_t *const *a, const uint32_t *const *b,
                                   const uint32_t *const *c);

GOC_API int goc_rdna4_v_med3_u32(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *const *b,
                                 const uint32_t *const *c);

// Integer multiplication into one 32-bit VGPR. Each source uses one VGPR;
// D may alias A or B. The 24-bit forms discard the upper byte of each input,
// then sign-extend signed inputs. High forms select bits 63:32 of the product.
// MUL_I32_I24 and MUL_U32_U24 support GOC_ALU_CLAMP to saturate to the signed
// or unsigned 32-bit range; otherwise low results wrap. Other instruction_flags
// must be zero. Supports loose semantics and preserves the host FP environment.
GOC_API int goc_rdna4_v_mul_lo_u32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_mul_hi_u32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_mul_hi_i32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_mul_i32_i24(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

GOC_API int goc_rdna4_v_mul_hi_i32_i24(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b);

GOC_API int goc_rdna4_v_mul_u32_u24(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

GOC_API int goc_rdna4_v_mul_hi_u32_u24(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b);

// Non-carry 32-bit integer addition/subtraction. Each operand uses one VGPR;
// D may alias any whole source VGPR. The two-input forms accept GOC_ALU_CLAMP
// to saturate in the signed/unsigned result domain; otherwise results wrap.
// ADD3 wraps modulo 2^32 and requires zero instruction_flags. Supports loose
// semantics and preserves the host FP environment. No carry mask is produced.
GOC_API int goc_rdna4_v_add_nc_u32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_sub_nc_u32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_subrev_nc_u32(uint64_t flags, uint64_t exec_mask,
                                      uint32_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b);

GOC_API int goc_rdna4_v_add_nc_i32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_sub_nc_i32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_add3_u32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

// True16 source/destination half selectors. Zero selects the low half.
// DOT2 consumes both A/B halves and accepts only the C and D selectors.
static const uint32_t GOC_ALU_HIGH_A = (UINT32_C(1) << 9);
static const uint32_t GOC_ALU_HIGH_B = (UINT32_C(1) << 10);
static const uint32_t GOC_ALU_HIGH_C = (UINT32_C(1) << 11);
static const uint32_t GOC_ALU_HIGH_D = (UINT32_C(1) << 12);

// Mixed FMA source formats: unset means FP32; set means FP16 in the half
// selected by GOC_ALU_HIGH_A/B/C. Half selectors are ignored for FP32 sources.
static const uint32_t GOC_MIX_F16_A = (UINT32_C(1) << 13);
static const uint32_t GOC_MIX_F16_B = (UINT32_C(1) << 14);
static const uint32_t GOC_MIX_F16_C = (UINT32_C(1) << 15);

// Wave32 mixed FMA: one VGPR per operand; supports source ABS/NEG, source
// format/half selectors, and CLAMP. OMOD and HIGH_D are invalid. MIX_F32
// produces FP32; MIXLO/MIXHI round directly to FP16 and preserve the other
// destination half. GOC_FP16_OVFL saturates finite FP16 overflow only.
// Loose semantics require host nearest-even arithmetic with denormals enabled.
// FP16-output forms also support empirical-exact scalar semantics and preserve
// host FP state in that mode. EXEC masks and whole-register aliases are supported.
GOC_API int goc_rdna4_v_fma_mix_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_fma_mixlo_f16(uint64_t flags, uint64_t exec_mask,
                                      uint32_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c);

GOC_API int goc_rdna4_v_fma_mixhi_f16(uint64_t flags, uint64_t exec_mask,
                                      uint32_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c);

// Binary FP16 arithmetic: one VGPR per operand, with independently selected
// source and destination halves. Preserves the other destination half and all
// inactive lanes; D may alias a whole source VGPR. Supports source ABS/NEG,
// OMOD then CLAMP before nearest-even FP16 narrowing, and GOC_FP16_OVFL.
// Loose semantics require host nearest-even arithmetic with denormals enabled.
GOC_API int goc_rdna4_v_add_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

GOC_API int goc_rdna4_v_sub_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

GOC_API int goc_rdna4_v_subrev_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b);

GOC_API int goc_rdna4_v_mul_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

GOC_API int goc_rdna4_v_min_num_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

GOC_API int goc_rdna4_v_max_num_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

GOC_API int goc_rdna4_v_minimum_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

GOC_API int goc_rdna4_v_maximum_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

// Three-input FP16 min/max and median: independently selected halves of A/B/C/D,
// with all source ABS/NEG, OMOD, CLAMP and GOC_FP16_OVFL. Output modifiers apply
// after the final selection. Preserves the other D half and inactive lanes;
// any whole-VGPR aliases are allowed. MINMAX/MAXMIN select A/B first, then C.
// Number variants ignore lone NaNs; MINIMUM/MAXIMUM propagate NaNs. MED3 uses
// minimumNumber(A,B,C) if any input is NaN. Loose semantics require host
// nearest-even arithmetic with denormals enabled.
GOC_API int goc_rdna4_v_min3_num_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_max3_num_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_minmax_num_f16(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c);

GOC_API int goc_rdna4_v_maxmin_num_f16(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c);

GOC_API int goc_rdna4_v_minimum3_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_maximum3_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_minimummaximum_f16(uint64_t flags, uint64_t exec_mask,
                                           uint32_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *c);

GOC_API int goc_rdna4_v_maximumminimum_f16(uint64_t flags, uint64_t exec_mask,
                                           uint32_t instruction_flags, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *c);

GOC_API int goc_rdna4_v_med3_num_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);

// FP16 LDEXP: scale the selected half of A by 2 raised to the signed int16_t
// exponent in the selected half of B. Supports A ABS/NEG, HIGH_A/B/D, OMOD,
// CLAMP and GOC_FP16_OVFL. Other source modifiers are invalid. Preserves the
// other D half and inactive lanes; D may alias a whole source VGPR. Loose
// semantics require host nearest-even arithmetic with denormals enabled.
GOC_API int goc_rdna4_v_ldexp_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

// FP16 FREXP exponent: read the selected A half and write a signed int16_t
// exponent into the selected D half. Zero, infinity and NaN return zero.
// Preserves the other D half, inactive lanes and the host FP environment;
// whole-register A/D aliasing is allowed. ABS_A/NEG_A, OMOD and CLAMP are
// accepted but do not change the result. Supports loose semantics.
GOC_API int goc_rdna4_v_frexp_exp_i16_f16(uint64_t flags, uint64_t exec_mask,
                                          uint32_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a);

// Unary FP16: one independently selected half per A/D VGPR; supports ABS_A,
// NEG_A, HIGH_A/D, OMOD, CLAMP and GOC_FP16_OVFL. Preserves the unselected D
// half and inactive lanes; whole-register A/D aliasing is allowed. EXP/LOG
// use base two, RNDNE rounds ties to even, and FRACT is capped below one.
// Loose semantics use FP32 arithmetic and output modifiers before nearest-even
// FP16 narrowing, requiring host nearest-even arithmetic with denormals enabled.
GOC_API int goc_rdna4_v_trunc_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_ceil_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_rndne_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_floor_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_sqrt_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_rcp_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_rsq_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_exp_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_log_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_fract_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_frexp_mant_f16(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a);

// True16 DOT2: A/B each hold two packed factors; C supplies one selected half.
// D replaces only its selected half, preserving the other half. Supports all
// six GOC_ALU ABS/NEG flags and HIGH_C/HIGH_D, without OMOD or CLAMP.
// Loose semantics use FP32 products and sums followed by nearest-even narrowing.
// BF16 flushes input/output denormals; F16 honors GOC_FP16_OVFL.
GOC_API int goc_rdna4_v_dot2_f16_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_dot2_bf16_bf16(uint64_t flags, uint64_t exec_mask,
                                       uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c);

// Unary FP32: one VGPR each for A/D. Supports NEG_A, ABS_A, OMOD and CLAMP;
// modifiers for absent operands are invalid. CLAMP maps NaNs to +0 and clamps
// to [0, 1]. EXP and LOG use base 2; RSQ computes reciprocal square root.
// RNDNE rounds ties to even. Only loose semantics are implemented.
GOC_API int goc_rdna4_v_trunc_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_ceil_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_rndne_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_floor_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_sqrt_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_rcp_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_rsq_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_exp_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_log_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);

// Floating DOT2 sign modifiers act after selecting each packed half.
static const uint32_t GOC_DOT_NEG_LO_A = (UINT32_C(1) << 0);
static const uint32_t GOC_DOT_NEG_LO_B = (UINT32_C(1) << 1);
static const uint32_t GOC_DOT_NEG_C = (UINT32_C(1) << 2);
static const uint32_t GOC_DOT_NEG_HI_A = (UINT32_C(1) << 3);
static const uint32_t GOC_DOT_NEG_HI_B = (UINT32_C(1) << 4);

// Floating DOT2 half selection: defaults are low for term 0 and high for term 1.
// These flags override those defaults; unlike raw op_sel_hi, zero means default.
static const uint32_t GOC_DOT_LO_A_HIGH = (UINT32_C(1) << 7);
static const uint32_t GOC_DOT_LO_B_HIGH = (UINT32_C(1) << 8);
static const uint32_t GOC_DOT_HI_A_LOW = (UINT32_C(1) << 9);
static const uint32_t GOC_DOT_HI_B_LOW = (UINT32_C(1) << 10);

// RDNA4 DOT2: A/B each hold two packed 16-bit factors in one VGPR;
// C/D each hold one FP32 value per lane. Loose and exact modes are supported.
// Supports all floating DOT2 sign/half-selection flags. GOC_DOT_CLAMP is
// accepted but has no effect on these floating DOT2 forms (as in rocjitsu).
GOC_API int goc_rdna4_v_dot2_f32_f16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_dot2_f32_bf16(uint64_t flags, uint64_t exec_mask,
                                      uint32_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      const uint32_t *const *c);

// FP8/BF8 DOT4 accepts NEG_C and ABS_C; ABS precedes NEG. A/B modifiers,
// half selection, output scaling and CLAMP are not supported.
static const uint32_t GOC_DOT_ABS_C = (UINT32_C(1) << 5);

// Wave32 DOT4: one VGPR each for A/B/C/D, four packed bytes per A/B lane;
// C/D are FP32. FP8 is OCP E4M3FN, BF8 is OCP E5M2. Loose semantics only.
GOC_API int goc_rdna4_v_dot4_f32_fp8_fp8(uint64_t flags, uint64_t exec_mask,
                                         uint32_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c);

GOC_API int goc_rdna4_v_dot4_f32_fp8_bf8(uint64_t flags, uint64_t exec_mask,
                                         uint32_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c);

GOC_API int goc_rdna4_v_dot4_f32_bf8_fp8(uint64_t flags, uint64_t exec_mask,
                                         uint32_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c);

GOC_API int goc_rdna4_v_dot4_f32_bf8_bf8(uint64_t flags, uint64_t exec_mask,
                                         uint32_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c);

// Integer DOT modifiers: SIGNED selects signed factors for I32_IU forms.
// U32_U forms accept only CLAMP. CLAMP saturates the final accumulator to its
// signed/unsigned 32-bit range; otherwise arithmetic wraps modulo 2^32.
static const uint32_t GOC_DOT_SIGNED_A = (UINT32_C(1) << 0);
static const uint32_t GOC_DOT_SIGNED_B = (UINT32_C(1) << 1);
static const uint32_t GOC_DOT_CLAMP = (UINT32_C(1) << 6);

// Integer DOT wave32: one VGPR each for A/B/C/D. A/B contain four packed
// bytes or eight packed nibbles. C/D are signed for I32_IU, unsigned for U32_U.
// Loose and exact semantics return the same integer result.
GOC_API int goc_rdna4_v_dot4_i32_iu8(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_dot4_u32_u8(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_dot8_i32_iu4(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_dot8_u32_u4(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

// WMMA modifier layout follows neg_lo[0:2], then neg_hi[0:2]. For C,
// neg_hi means absolute value, applied before neg_lo negation.
static const uint32_t GOC_WMMA_NEG_LO_A = (UINT32_C(1) << 0);
static const uint32_t GOC_WMMA_NEG_LO_B = (UINT32_C(1) << 1);
static const uint32_t GOC_WMMA_NEG_C = (UINT32_C(1) << 2);
static const uint32_t GOC_WMMA_NEG_HI_A = (UINT32_C(1) << 3);
static const uint32_t GOC_WMMA_NEG_HI_B = (UINT32_C(1) << 4);
static const uint32_t GOC_WMMA_ABS_C = (UINT32_C(1) << 5);

// Wave32 16x16x16 WMMA: A/B each contain 4 VGPRs of packed 16-bit
// elements; C/D each contain 8 VGPRs of FP32 elements. GoC applies exec_mask
// to destination writes, including WMMA. All source lanes remain readable.
// Both loose and empirical exact semantics are supported for these WMMA forms.
GOC_API int goc_rdna4_v_wmma_f32_16x16x16_f16(uint64_t flags, uint64_t exec_mask,
                                              uint32_t instruction_flags, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *c);

GOC_API int goc_rdna4_v_wmma_f32_16x16x16_bf16(uint64_t flags, uint64_t exec_mask,
                                               uint32_t instruction_flags, uint32_t *const *d,
                                               const uint32_t *const *a, const uint32_t *const *b,
                                               const uint32_t *const *c);

// Wave64 variants: 64 words per VGPR; 2 VGPRs for A/B, 4 for C/D.
// Both semantics and all WMMA modifiers are supported through scalar paths.
GOC_API int goc_rdna4w64_v_wmma_f32_16x16x16_f16(uint64_t flags, uint64_t exec_mask,
                                                 uint32_t instruction_flags, uint32_t *const *d,
                                                 const uint32_t *const *a, const uint32_t *const *b,
                                                 const uint32_t *const *c);

GOC_API int goc_rdna4w64_v_wmma_f32_16x16x16_bf16(uint64_t flags, uint64_t exec_mask,
                                                  uint32_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);

// Packed-output WMMA: A/B hold 4 VGPRs (wave32) or 2 (wave64);
// C/D hold 4 or 2 VGPRs respectively, with adjacent rows in low/high halves.
// Both semantics and all six floating WMMA modifiers are supported. Packed
// results narrow after each four-product step; GOC_FP16_OVFL controls FP16 overflow.
GOC_API int goc_rdna4_v_wmma_f16_16x16x16_f16(uint64_t flags, uint64_t exec_mask,
                                              uint32_t instruction_flags, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *c);

GOC_API int goc_rdna4_v_wmma_bf16_16x16x16_bf16(uint64_t flags, uint64_t exec_mask,
                                                uint32_t instruction_flags, uint32_t *const *d,
                                                const uint32_t *const *a, const uint32_t *const *b,
                                                const uint32_t *const *c);

GOC_API int goc_rdna4w64_v_wmma_f16_16x16x16_f16(uint64_t flags, uint64_t exec_mask,
                                                 uint32_t instruction_flags, uint32_t *const *d,
                                                 const uint32_t *const *a, const uint32_t *const *b,
                                                 const uint32_t *const *c);

GOC_API int goc_rdna4w64_v_wmma_bf16_16x16x16_bf16(uint64_t flags, uint64_t exec_mask,
                                                   uint32_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);

// Wave32 FP8/BF8 WMMA: A/B each hold 2 VGPRs, C/D each hold 8.
// FP8 is OCP E4M3FN; BF8 is OCP E5M2. Only loose semantics are implemented;
// strict exact requests return GOC_ERROR_UNSUPPORTED_SEMANTICS. Supported
// modifiers are GOC_WMMA_NEG_C and GOC_WMMA_ABS_C.
GOC_API int goc_rdna4_v_wmma_f32_16x16x16_fp8_fp8(uint64_t flags, uint64_t exec_mask,
                                                  uint32_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);

GOC_API int goc_rdna4_v_wmma_f32_16x16x16_fp8_bf8(uint64_t flags, uint64_t exec_mask,
                                                  uint32_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);

GOC_API int goc_rdna4_v_wmma_f32_16x16x16_bf8_fp8(uint64_t flags, uint64_t exec_mask,
                                                  uint32_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);

GOC_API int goc_rdna4_v_wmma_f32_16x16x16_bf8_bf8(uint64_t flags, uint64_t exec_mask,
                                                  uint32_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c);

// Integer WMMA modifiers: NEG[0:1] select signed interpretation of A/B;
// CLAMP saturates the final signed accumulator instead of wrapping modulo 2^32.
static const uint32_t GOC_WMMA_SIGNED_A = (UINT32_C(1) << 0);
static const uint32_t GOC_WMMA_SIGNED_B = (UINT32_C(1) << 1);
static const uint32_t GOC_WMMA_CLAMP = (UINT32_C(1) << 6);

// Wave32 integer WMMA: C/D each hold 8 VGPRs. A/B each hold 2 VGPRs for
// IU8 and K=32 IU4, or 1 VGPR for K=16 IU4. Supports loose and exact semantics,
// signed/unsigned factors, and CLAMP. Accumulators are signed 32-bit integers.
GOC_API int goc_rdna4_v_wmma_i32_16x16x16_iu8(uint64_t flags, uint64_t exec_mask,
                                              uint32_t instruction_flags, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *c);

GOC_API int goc_rdna4_v_wmma_i32_16x16x16_iu4(uint64_t flags, uint64_t exec_mask,
                                              uint32_t instruction_flags, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *c);

GOC_API int goc_rdna4_v_wmma_i32_16x16x32_iu4(uint64_t flags, uint64_t exec_mask,
                                              uint32_t instruction_flags, uint32_t *const *d,
                                              const uint32_t *const *a, const uint32_t *const *b,
                                              const uint32_t *const *c);

// Bit counts: every operand holds one VGPR. CLZ/CTZ return 0xffffffff for zero;
// CLS counts leading sign bits (including the sign bit) and returns 0xffffffff
// if all bits match. BCNT returns popcount(A) + B, wrapping modulo 2^32.
// MBCNT_LO counts A bits below min(lane, 32); MBCNT_HI counts A bits below
// max(lane - 32, 0). Both add B with wrapping. The lane index is physical and
// independent of EXEC. Wave64 forms use 64 lane words per VGPR.
// No instruction modifiers apply. Only loose semantics are implemented;
// all host FP state is preserved.
GOC_API int goc_rdna4_v_clz_i32_u32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_ctz_i32_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_cls_i32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_bcnt_u32_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b);

GOC_API int goc_rdna4_v_mbcnt_lo_u32_b32(uint64_t flags, uint64_t exec_mask,
                                         uint32_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b);

GOC_API int goc_rdna4_v_mbcnt_hi_u32_b32(uint64_t flags, uint64_t exec_mask,
                                         uint32_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b);

GOC_API int goc_rdna4w64_v_mbcnt_lo_u32_b32(uint64_t flags, uint64_t exec_mask,
                                            uint32_t instruction_flags, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b);

GOC_API int goc_rdna4w64_v_mbcnt_hi_u32_b32(uint64_t flags, uint64_t exec_mask,
                                            uint32_t instruction_flags, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b);

// Wave32 Boolean operations: each operand holds one VGPR. B32 forms have no
// instruction modifiers. B16 forms select A/B/D halves with HIGH_A/B/D, leave
// the other D half unchanged, and reject all other modifiers; NOT has no B.
// Only loose semantics are implemented. All host FP state is preserved.
GOC_API int goc_rdna4_v_and_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

GOC_API int goc_rdna4_v_or_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

GOC_API int goc_rdna4_v_xor_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

GOC_API int goc_rdna4_v_not_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);

GOC_API int goc_rdna4_v_and_b16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

GOC_API int goc_rdna4_v_or_b16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

GOC_API int goc_rdna4_v_xor_b16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

GOC_API int goc_rdna4_v_not_b16(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a);

// Wave32 bit operations: every operand holds one VGPR. No instruction modifiers
// are supported. Only loose semantics are implemented. BFE offsets and widths,
// and BFM widths and offsets, use their low five bits; a zero width produces zero.
// Signed BFE sign-extends A before extraction and sign-extends the extracted field.
// BFI selects B where A has a set bit, C otherwise. BFREV reverses all 32 bits.
GOC_API int goc_rdna4_v_bfe_u32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_bfe_i32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_bfi_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

GOC_API int goc_rdna4_v_bfm_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

GOC_API int goc_rdna4_v_bfrev_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
