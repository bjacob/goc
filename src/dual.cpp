// SPDX-License-Identifier: MIT

// One decoded VOPD operation. Arithmetic follows the corresponding ordinary
// instruction; VOPD has no instruction-modifier fields and is Wave32 only.

#include "goc/goc.h"

#include <stdint.h>

int goc_v_dual_fmac_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        uint32_t *trapsts) {
  if (instruction_flags) {
    int error = goc_v_fmac_f32(flags, 0, 0, d, a, b, trapsts);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_fmac_f32(flags, exec_mask, 0, d, a, b, trapsts);
}

int goc_v_dual_fmac_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (instruction_flags) {
    int error = goc_v_fmac_f32_rdna4(flags, 0, 0, d, a, b, excp_flag_user);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_fmac_f32_rdna4(flags, exec_mask, 0, d, a, b, excp_flag_user);
}

int goc_v_dual_fmaak_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         uint32_t literal, uint32_t *trapsts) {
  if (instruction_flags) {
    int error = goc_v_fmaak_f32(flags, 0, 0, d, a, b, literal, trapsts);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_fmaak_f32(flags, exec_mask, 0, d, a, b, literal, trapsts);
}

int goc_v_dual_fmaak_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t literal,
                               uint32_t *excp_flag_user) {
  if (instruction_flags) {
    int error = goc_v_fmaak_f32_rdna4(flags, 0, 0, d, a, b, literal, excp_flag_user);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_fmaak_f32_rdna4(flags, exec_mask, 0, d, a, b, literal, excp_flag_user);
}

int goc_v_dual_fmamk_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                         const uint32_t *const *b, uint32_t *trapsts) {
  if (instruction_flags) {
    int error = goc_v_fmamk_f32(flags, 0, 0, d, a, literal, b, trapsts);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_fmamk_f32(flags, exec_mask, 0, d, a, literal, b, trapsts);
}

int goc_v_dual_fmamk_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t literal,
                               const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (instruction_flags) {
    int error = goc_v_fmamk_f32_rdna4(flags, 0, 0, d, a, literal, b, excp_flag_user);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_fmamk_f32_rdna4(flags, exec_mask, 0, d, a, literal, b, excp_flag_user);
}

int goc_v_dual_mul_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *trapsts) {
  if (instruction_flags) {
    int error = goc_v_mul_f32(flags, 0, 0, d, a, b, trapsts);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_mul_f32(flags, exec_mask, 0, d, a, b, trapsts);
}

int goc_v_dual_mul_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *excp_flag_user) {
  if (instruction_flags) {
    int error = goc_v_mul_f32_rdna4(flags, 0, 0, d, a, b, excp_flag_user);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_mul_f32_rdna4(flags, exec_mask, 0, d, a, b, excp_flag_user);
}

int goc_v_dual_add_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *trapsts) {
  if (instruction_flags) {
    int error = goc_v_add_f32(flags, 0, 0, d, a, b, trapsts);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_add_f32(flags, exec_mask, 0, d, a, b, trapsts);
}

int goc_v_dual_add_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *excp_flag_user) {
  if (instruction_flags) {
    int error = goc_v_add_f32_rdna4(flags, 0, 0, d, a, b, excp_flag_user);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_add_f32_rdna4(flags, exec_mask, 0, d, a, b, excp_flag_user);
}

int goc_v_dual_sub_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *trapsts) {
  if (instruction_flags) {
    int error = goc_v_sub_f32(flags, 0, 0, d, a, b, trapsts);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_sub_f32(flags, exec_mask, 0, d, a, b, trapsts);
}

int goc_v_dual_sub_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             uint32_t *excp_flag_user) {
  if (instruction_flags) {
    int error = goc_v_sub_f32_rdna4(flags, 0, 0, d, a, b, excp_flag_user);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_sub_f32_rdna4(flags, exec_mask, 0, d, a, b, excp_flag_user);
}

int goc_v_dual_subrev_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                          uint32_t *trapsts) {
  if (instruction_flags) {
    int error = goc_v_subrev_f32(flags, 0, 0, d, a, b, trapsts);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_subrev_f32(flags, exec_mask, 0, d, a, b, trapsts);
}

int goc_v_dual_subrev_f32_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (instruction_flags) {
    int error = goc_v_subrev_f32_rdna4(flags, 0, 0, d, a, b, excp_flag_user);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_subrev_f32_rdna4(flags, exec_mask, 0, d, a, b, excp_flag_user);
}

int goc_v_dual_mul_dx9_zero_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *trapsts) {
  if (instruction_flags) {
    int error = goc_v_mul_dx9_zero_f32(flags, 0, 0, d, a, b, trapsts);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_mul_dx9_zero_f32(flags, exec_mask, 0, d, a, b, trapsts);
}

int goc_v_dual_mul_dx9_zero_f32_rdna4(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  if (instruction_flags) {
    int error = goc_v_mul_dx9_zero_f32_rdna4(flags, 0, 0, d, a, b, excp_flag_user);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_mul_dx9_zero_f32_rdna4(flags, exec_mask, 0, d, a, b, excp_flag_user);
}

int goc_v_dual_mov_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a) {
  if (instruction_flags) {
    int error = goc_v_mov_b32(flags, 0, 0, d, a);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_mov_b32(flags, exec_mask, 0, d, a);
}

int goc_v_dual_cndmask_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t condition) {
  if (instruction_flags) {
    int error = goc_v_cndmask_b32(flags, 0, 0, d, a, b, condition);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_cndmask_b32(flags, exec_mask, 0, d, a, b, condition);
}

int goc_v_dual_max_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *excp_flag_user) {
  if (instruction_flags) {
    int error = goc_v_max_num_f32(flags, 0, 0, d, a, b, excp_flag_user);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_max_num_f32(flags, exec_mask, 0, d, a, b, excp_flag_user);
}

int goc_v_dual_min_num_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           uint32_t *excp_flag_user) {
  if (instruction_flags) {
    int error = goc_v_min_num_f32(flags, 0, 0, d, a, b, excp_flag_user);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_min_num_f32(flags, exec_mask, 0, d, a, b, excp_flag_user);
}

int goc_v_dual_max_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *trapsts) {
  if (instruction_flags) {
    int error = goc_v_max_f32(flags, 0, 0, d, a, b, trapsts);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_max_f32(flags, exec_mask, 0, d, a, b, trapsts);
}

int goc_v_dual_min_f32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       uint32_t *trapsts) {
  if (instruction_flags) {
    int error = goc_v_min_f32(flags, 0, 0, d, a, b, trapsts);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_min_f32(flags, exec_mask, 0, d, a, b, trapsts);
}

int goc_v_dual_add_nc_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags) {
    int error = goc_v_add_nc_u32(flags, 0, 0, d, a, b);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_add_nc_u32(flags, exec_mask, 0, d, a, b);
}

int goc_v_dual_lshlrev_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags) {
    int error = goc_v_lshlrev_b32(flags, 0, 0, d, a, b);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_lshlrev_b32(flags, exec_mask, 0, d, a, b);
}

int goc_v_dual_and_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (instruction_flags) {
    int error = goc_v_and_b32(flags, 0, 0, d, a, b);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_and_b32(flags, exec_mask, 0, d, a, b);
}

int goc_v_dual_dot2acc_f32_f16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t *trapsts) {
  if (instruction_flags) {
    int error = goc_v_dot2_f32_f16(flags, 0, 0, d, a, b, d, trapsts);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_dot2_f32_f16(flags, exec_mask, 0, d, a, b, d, trapsts);
}

int goc_v_dual_dot2acc_f32_bf16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, uint32_t *trapsts) {
  if (instruction_flags) {
    int error = goc_v_dot2_f32_bf16(flags, 0, 0, d, a, b, d, trapsts);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_dot2_f32_bf16(flags, exec_mask, 0, d, a, b, d, trapsts);
}

int goc_v_dual_dot2acc_f32_f16_rdna4(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, uint32_t *excp_flag_user) {
  if (instruction_flags) {
    int error = goc_v_dot2_f32_f16_rdna4(flags, 0, 0, d, a, b, d, excp_flag_user);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_dot2_f32_f16_rdna4(flags, exec_mask, 0, d, a, b, d, excp_flag_user);
}

int goc_v_dual_dot2acc_f32_bf16_rdna4(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      uint32_t *excp_flag_user) {
  if (instruction_flags) {
    int error = goc_v_dot2_f32_bf16_rdna4(flags, 0, 0, d, a, b, d, excp_flag_user);
    return error ? error : GOC_ERROR_INVALID_FLAGS;
  }
  return goc_v_dot2_f32_bf16_rdna4(flags, exec_mask, 0, d, a, b, d, excp_flag_user);
}
