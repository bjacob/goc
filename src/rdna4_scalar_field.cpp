// SPDX-License-Identifier: MIT

// Models borrowed from rocjitsu generated/shared/execute_shared.h and verified
// against GFX1201, including clipped BFE widths and count sentinel values.

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_bits.h"

#include <algorithm>
#include <cstring>
#include <stdint.h>

namespace {

template <typename T, bool Signed> T extract(T a, uint32_t field) {
  const unsigned bits = sizeof(T) * 8;
  unsigned offset = field & (bits - 1), width = std::min((field >> 16) & 127, bits - offset);
  if (!width)
    return 0;
  T mask = width == bits ? ~T(0) : (T(1) << width) - 1;
  T result = (a >> offset) & mask;
  if constexpr (Signed)
    if (result & (T(1) << (width - 1)))
      result |= ~mask;
  return result;
}

enum class FieldOp {
  ExtractUnsigned,
  ExtractSigned,
  Mask,
  Population,
  PopulationZero,
  Leading,
  Trailing,
  Sign,
  Set,
  Clear
};

template <FieldOp Op, typename D, typename A>
int run(uint64_t flags, uint64_t mode, D *d, A a, uint32_t b, uint32_t *scc) {
  if (mode >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, uint32_t(mode), true))
    return error;
  D result;
  if constexpr (Op == FieldOp::ExtractUnsigned || Op == FieldOp::ExtractSigned)
    result = extract<A, Op == FieldOp::ExtractSigned>(a, b);
  if constexpr (Op == FieldOp::Mask) {
    constexpr unsigned bits = sizeof(D) * 8;
    unsigned width = unsigned(a) & (bits - 1), offset = b & (bits - 1);
    result = ((D(1) << width) - 1) << offset;
  }
  if constexpr (Op == FieldOp::Population || Op == FieldOp::PopulationZero)
    result = goc::bit_population(Op == FieldOp::PopulationZero ? A(~a) : a);
  if constexpr (Op == FieldOp::Leading || Op == FieldOp::Trailing || Op == FieldOp::Sign) {
    if constexpr (Op == FieldOp::Sign)
      a ^= A(0) - (a >> (sizeof(A) * 8 - 1));
    result = goc::bit_count_zero<Op == FieldOp::Trailing>(a);
  }
  if constexpr (Op == FieldOp::Set || Op == FieldOp::Clear) {
    D bit = D(1) << (b & (sizeof(D) * 8 - 1));
    result = Op == FieldOp::Set ? *d | bit : *d & ~bit;
  }
  *d = result;
  if constexpr (Op == FieldOp::ExtractUnsigned || Op == FieldOp::ExtractSigned ||
                Op == FieldOp::Population || Op == FieldOp::PopulationZero) {
    uint32_t cc = result != 0;
    std::memcpy(scc, &cc, sizeof(cc));
  }
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_s_bfe_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<FieldOp::ExtractUnsigned>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_bfe_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<FieldOp::ExtractSigned>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_bfe_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint64_t *d,
                        uint64_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<FieldOp::ExtractUnsigned>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_bfe_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint64_t *d,
                        uint64_t a, uint32_t b, uint32_t *scc) {
  (void)exec_mask;
  return run<FieldOp::ExtractSigned>(flags, instruction_flags, d, a, b, scc);
}

int goc_rdna4_s_bfm_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a, uint32_t b) {
  (void)exec_mask;
  return run<FieldOp::Mask>(flags, instruction_flags, d, a, b, nullptr);
}

int goc_rdna4_s_bfm_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint64_t *d,
                        uint32_t a, uint32_t b) {
  (void)exec_mask;
  return run<FieldOp::Mask>(flags, instruction_flags, d, a, b, nullptr);
}

int goc_rdna4_s_bcnt0_i32_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, uint32_t a, uint32_t *scc) {
  (void)exec_mask;
  return run<FieldOp::PopulationZero>(flags, instruction_flags, d, a, 0, scc);
}

int goc_rdna4_s_bcnt0_i32_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, uint64_t a, uint32_t *scc) {
  (void)exec_mask;
  return run<FieldOp::PopulationZero>(flags, instruction_flags, d, a, 0, scc);
}

int goc_rdna4_s_bcnt1_i32_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, uint32_t a, uint32_t *scc) {
  (void)exec_mask;
  return run<FieldOp::Population>(flags, instruction_flags, d, a, 0, scc);
}

int goc_rdna4_s_bcnt1_i32_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *d, uint64_t a, uint32_t *scc) {
  (void)exec_mask;
  return run<FieldOp::Population>(flags, instruction_flags, d, a, 0, scc);
}

int goc_rdna4_s_ctz_i32_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a) {
  (void)exec_mask;
  return run<FieldOp::Trailing>(flags, instruction_flags, d, a, 0, nullptr);
}

int goc_rdna4_s_ctz_i32_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint64_t a) {
  (void)exec_mask;
  return run<FieldOp::Trailing>(flags, instruction_flags, d, a, 0, nullptr);
}

int goc_rdna4_s_clz_i32_u32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t a) {
  (void)exec_mask;
  return run<FieldOp::Leading>(flags, instruction_flags, d, a, 0, nullptr);
}

int goc_rdna4_s_clz_i32_u64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint64_t a) {
  (void)exec_mask;
  return run<FieldOp::Leading>(flags, instruction_flags, d, a, 0, nullptr);
}

int goc_rdna4_s_cls_i32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *d,
                        uint32_t a) {
  (void)exec_mask;
  return run<FieldOp::Sign>(flags, instruction_flags, d, a, 0, nullptr);
}

int goc_rdna4_s_cls_i32_i64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint64_t a) {
  (void)exec_mask;
  return run<FieldOp::Sign>(flags, instruction_flags, d, a, 0, nullptr);
}

int goc_rdna4_s_bitset0_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t b) {
  (void)exec_mask;
  return run<FieldOp::Clear>(flags, instruction_flags, d, 0u, b, nullptr);
}

int goc_rdna4_s_bitset0_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, uint32_t b) {
  (void)exec_mask;
  return run<FieldOp::Clear>(flags, instruction_flags, d, 0u, b, nullptr);
}

int goc_rdna4_s_bitset1_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *d, uint32_t b) {
  (void)exec_mask;
  return run<FieldOp::Set>(flags, instruction_flags, d, 0u, b, nullptr);
}

int goc_rdna4_s_bitset1_b64(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint64_t *d, uint32_t b) {
  (void)exec_mask;
  return run<FieldOp::Set>(flags, instruction_flags, d, 0u, b, nullptr);
}
