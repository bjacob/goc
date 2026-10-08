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
// FMA supports all GOC_ALU source/output modifiers. Exact semantics requests
// fall back to loose unless GOC_SEMANTICS_STRICT is set.
GOC_API int goc_rdna4_v_fma_f32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

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
static const uint32_t GOC_ALU_CLAMP = (UINT32_C(1) << 8);

// True16 source/destination half selectors. Zero selects the low half.
// DOT2 consumes both A/B halves, so only C and D have selectors.
static const uint32_t GOC_ALU_HIGH_C = (UINT32_C(1) << 11);
static const uint32_t GOC_ALU_HIGH_D = (UINT32_C(1) << 12);

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

#ifdef __cplusplus
} // extern "C"
#endif

#endif
