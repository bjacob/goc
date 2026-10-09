// SPDX-License-Identifier: MIT

// Semantics follow rocjitsu generated/shared/execute_shared.h. Bit spreading and
// compression implement the same per-bit operations without scalar loops.

#include "goc/goc.h"
#include "internal.h"

#include <cstring>
#include <stdint.h>

namespace {

int validate(uint64_t flags, uint64_t mode) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  return goc::validate(flags, uint32_t(mode), true,
                       GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS);
}

template <typename T, bool Compress> T quad(T a) {
  a |= a >> 1;
  a |= a >> 2;
  a &= T(UINT64_C(0x1111111111111111));
  if constexpr (!Compress)
    return a * 15;
  a = (a | (a >> 3)) & T(UINT64_C(0x0303030303030303));
  a = (a | (a >> 6)) & T(UINT64_C(0x000f000f000f000f));
  a = (a | (a >> 12)) & T(UINT64_C(0x000000ff000000ff));
  if constexpr (sizeof(T) == 8)
    a = (a | (a >> 24)) & 65535;
  return a;
}

template <bool HighA, bool HighB>
int pack(uint64_t flags, uint64_t mode, uint32_t *d, uint32_t a, uint32_t b) {
  if (int error = validate(flags, mode))
    return error;
  *d = ((a >> (HighA ? 16 : 0)) & 65535) | ((b >> (HighB ? 16 : 0)) << 16);
  return GOC_SUCCESS;
}

template <typename T>
int select(uint64_t flags, uint64_t mode, T *d, T a, T b, uint32_t input_scc) {
  if (int error = validate(flags, mode))
    return error;
  *d = input_scc & 1 ? a : b;
  return GOC_SUCCESS;
}

template <bool Compress, typename T>
int quad_write(uint64_t flags, uint64_t mode, T *d, T a, uint32_t *scc) {
  if (int error = validate(flags, mode))
    return error;
  T result = quad<T, Compress>(a);
  *d = result;
  uint32_t cc = result != 0;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_s_pack_ll_b32_b16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, uint32_t a, uint32_t b) {
  (void)exec_mask;
  return pack<false, false>(flags, instruction_flags, d, a, b);
}

int goc_rdna4_s_pack_lh_b32_b16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, uint32_t a, uint32_t b) {
  (void)exec_mask;
  return pack<false, true>(flags, instruction_flags, d, a, b);
}

int goc_rdna4_s_pack_hl_b32_b16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, uint32_t a, uint32_t b) {
  (void)exec_mask;
  return pack<true, false>(flags, instruction_flags, d, a, b);
}

int goc_rdna4_s_pack_hh_b32_b16(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *d, uint32_t a, uint32_t b) {
  (void)exec_mask;
  return pack<true, true>(flags, instruction_flags, d, a, b);
}

int goc_rdna4_s_bitreplicate_b64_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint64_t *d, uint32_t a) {
  (void)exec_mask;
  if (int error = validate(flags, instruction_flags))
    return error;
  uint64_t x = a;
  x = (x | (x << 16)) & UINT64_C(0x0000ffff0000ffff);
  x = (x | (x << 8)) & UINT64_C(0x00ff00ff00ff00ff);
  x = (x | (x << 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
  x = (x | (x << 2)) & UINT64_C(0x3333333333333333);
  x = (x | (x << 1)) & UINT64_C(0x5555555555555555);
  uint64_t result = x | (x << 1);
  *d = result;
  return GOC_SUCCESS;
}

int goc_rdna4_s_cselect_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t b, uint32_t input_scc) {
  (void)exec_mask;
  return select(flags, instruction_flags, d, a, b, input_scc);
}

int goc_rdna4_s_cselect_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, uint64_t a, uint64_t b, uint32_t input_scc) {
  (void)exec_mask;
  return select(flags, instruction_flags, d, a, b, input_scc);
}

int goc_rdna4_s_quadmask_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, uint32_t a, uint32_t *scc) {
  (void)exec_mask;
  return quad_write<true>(flags, instruction_flags, d, a, scc);
}

int goc_rdna4_s_quadmask_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, uint64_t a, uint32_t *scc) {
  (void)exec_mask;
  return quad_write<true>(flags, instruction_flags, d, a, scc);
}

int goc_rdna4_s_wqm_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t *scc) {
  (void)exec_mask;
  return quad_write<false>(flags, instruction_flags, d, a, scc);
}

int goc_rdna4_s_wqm_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint64_t *d,
                        uint64_t a, uint32_t *scc) {
  (void)exec_mask;
  return quad_write<false>(flags, instruction_flags, d, a, scc);
}
