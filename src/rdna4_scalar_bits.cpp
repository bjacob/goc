// SPDX-License-Identifier: MIT

// Models follow rocjitsu generated/shared/execute_shared.h, verified on GFX1201.

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_bits.h"

#include <cstring>
#include <stdint.h>

namespace {

enum class BitOp {
  And,
  Or,
  Xor,
  Nand,
  Nor,
  Xnor,
  AndNot,
  OrNot,
  Not,
  Reverse,
  Left,
  Right,
  SignedRight
};

template <BitOp Op, typename T>
int run(uint64_t flags, uint64_t mode, T *d, T a, uint64_t b, uint32_t *scc) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, uint32_t(mode), true))
    return error;
  T result, right = T(b);
  if constexpr (Op == BitOp::And)
    result = a & right;
  if constexpr (Op == BitOp::Or)
    result = a | right;
  if constexpr (Op == BitOp::Xor)
    result = a ^ right;
  if constexpr (Op == BitOp::Nand)
    result = ~(a & right);
  if constexpr (Op == BitOp::Nor)
    result = ~(a | right);
  if constexpr (Op == BitOp::Xnor)
    result = ~(a ^ right);
  if constexpr (Op == BitOp::AndNot)
    result = a & ~right;
  if constexpr (Op == BitOp::OrNot)
    result = a | ~right;
  if constexpr (Op == BitOp::Not)
    result = ~a;
  if constexpr (Op == BitOp::Reverse)
    result = goc::bit_reverse(a);
  if constexpr (Op == BitOp::Left || Op == BitOp::Right || Op == BitOp::SignedRight) {
    constexpr unsigned bits = sizeof(T) * 8;
    unsigned count = unsigned(b) & (bits - 1);
    if constexpr (Op == BitOp::Left)
      result = a << count;
    if constexpr (Op == BitOp::Right)
      result = a >> count;
    if constexpr (Op == BitOp::SignedRight) {
      T sign = a >> (bits - 1);
      result = (a >> count) | (count ? (T(0) - sign) << (bits - count) : 0);
    }
  }
  if constexpr (Op == BitOp::Reverse) {
    *d = result;
  } else {
    uint32_t cc = result != 0;
    *d = result;
    std::memcpy(scc, &cc, sizeof(cc));
  }
  return GOC_SUCCESS;
}

template <unsigned Shift>
int shift_add(uint64_t flags, uint64_t mode, uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, uint32_t(mode), true))
    return error;
  uint64_t wide = (uint64_t(a) << Shift) + b;
  uint32_t result = uint32_t(wide), cc = wide > UINT32_MAX;
  *d = result;
  std::memcpy(scc, &cc, sizeof(cc));
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_s_and_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::And>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_and_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint64_t *d,
                        uint64_t a, uint64_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::And>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_or_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                       uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::Or>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_or_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint64_t *d,
                       uint64_t a, uint64_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::Or>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_xor_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::Xor>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_xor_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint64_t *d,
                        uint64_t a, uint64_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::Xor>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_nand_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::Nand>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_nand_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint64_t *d, uint64_t a, uint64_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::Nand>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_nor_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::Nor>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_nor_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint64_t *d,
                        uint64_t a, uint64_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::Nor>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_xnor_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::Xnor>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_xnor_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint64_t *d, uint64_t a, uint64_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::Xnor>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_and_not1_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::AndNot>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_and_not1_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint64_t *d, uint64_t a, uint64_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::AndNot>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_or_not1_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::OrNot>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_or_not1_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, uint64_t a, uint64_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::OrNot>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_not_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::Not>(flags, instruction_flags, d, a, 0, scc);
}

int goc_rdna4_s_not_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint64_t *d,
                        uint64_t a, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::Not>(flags, instruction_flags, d, a, 0, scc);
}

int goc_rdna4_s_brev_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *d, uint32_t a) {
  (void)exec_mask;
  return run<BitOp::Reverse>(flags, instruction_flags, d, a, 0, nullptr);
}

int goc_rdna4_s_brev_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint64_t *d, uint64_t a) {
  (void)exec_mask;
  return run<BitOp::Reverse>(flags, instruction_flags, d, a, 0, nullptr);
}

int goc_rdna4_s_lshl_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::Left>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_lshl_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint64_t *d, uint64_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::Left>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_lshr_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::Right>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_lshr_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint64_t *d, uint64_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::Right>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_ashr_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::SignedRight>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_ashr_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                         uint64_t *d, uint64_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<BitOp::SignedRight>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_lshl1_add_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return shift_add<1>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_lshl2_add_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return shift_add<2>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_lshl3_add_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return shift_add<3>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_lshl4_add_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return shift_add<4>(flags, instruction_flags, d, a, b, scc);
}
