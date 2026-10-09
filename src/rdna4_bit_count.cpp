// SPDX-License-Identifier: MIT

#include "rdna4_bit_count.h"
#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

// Low-bit masks for widths 0 through 32. A table also avoids Clang's SSE2
// vectorization of variable shifts through float-to-integer conversions,
// which can spuriously raise host FP exceptions for out-of-range counts.
static const uint32_t prefix_masks[] = {
    0x0u,       0x1u,        0x3u,        0x7u,        0xfu,       0x1fu,      0x3fu,
    0x7fu,      0xffu,       0x1ffu,      0x3ffu,      0x7ffu,     0xfffu,     0x1fffu,
    0x3fffu,    0x7fffu,     0xffffu,     0x1ffffu,    0x3ffffu,   0x7ffffu,   0xfffffu,
    0x1fffffu,  0x3fffffu,   0x7fffffu,   0xffffffu,   0x1ffffffu, 0x3ffffffu, 0x7ffffffu,
    0xfffffffu, 0x1fffffffu, 0x3fffffffu, 0x7fffffffu, 0xffffffffu};

// Returns the number of set bits in x.
uint32_t population(uint32_t x) {
  // Integer bit sums follow rocjitsu's popcount_u32_simd.
  x -= (x >> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  x = (x + (x >> 4)) & 0x0f0f0f0f;
  x += x >> 8;
  x += x >> 16;
  return x & 63;
}

template <goc::BitCount Op> uint32_t evaluate(uint32_t a, uint32_t b, unsigned lane) {
  if constexpr (Op == goc::BitCount::Sign)
    a ^= 0u - (a >> 31);
  if constexpr (Op == goc::BitCount::Leading || Op == goc::BitCount::Sign ||
                Op == goc::BitCount::Trailing) {
    if (!a)
      return UINT32_MAX;
#if defined(__GNUC__) || defined(__clang__)
    if constexpr (Op == goc::BitCount::Trailing)
      return __builtin_ctz(a);
    else
      return __builtin_clz(a);
#else
    uint32_t count = 0;
    if constexpr (Op == goc::BitCount::Trailing) {
      while (!(a & 1)) {
        ++count;
        a >>= 1;
      }
    } else {
      while (!(a & 0x80000000)) {
        ++count;
        a <<= 1;
      }
    }
    return count;
#endif
  }
  if constexpr (Op == goc::BitCount::MaskedLow)
    a &= prefix_masks[lane < 32 ? lane : 32];
  if constexpr (Op == goc::BitCount::MaskedHigh)
    a &= prefix_masks[lane < 32 ? 0 : lane - 32];
  return population(a) + b;
}

template <goc::BitCount Op, int Lanes>
int bit_count(uint64_t flags, uint64_t mask, uint32_t mode, uint32_t *const *d,
              const uint32_t *const *a, const uint32_t *const *b) {
  if (int error = goc::validate(flags, mode))
    return error;
  if constexpr (Lanes == 32)
    mask = uint32_t(mask);
  if (!mask)
    return GOC_SUCCESS;
  const uint32_t *bp = nullptr;
  if constexpr (Op == goc::BitCount::Population || Op == goc::BitCount::MaskedLow ||
                Op == goc::BitCount::MaskedHigh)
    bp = b[0];
#if defined(GOC_HAVE_X86_64_V4)
  if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V4) {
    goc::bit_count_x86_64_v4<Op, Lanes>(mask, d[0], a[0], bp);
    return GOC_SUCCESS;
  }
#endif
#if defined(GOC_HAVE_X86_64_V3)
  // Wave32 MBCNT_HI only copies B; AVX2 showed no substantial gain there.
  if constexpr (Op != goc::BitCount::MaskedHigh || Lanes != 32) {
    if ((flags & GOC_CPU_MASK) >= GOC_CPU_X86_64_V3) {
      goc::bit_count_x86_64_v3<Op, Lanes>(mask, d[0], a[0], bp);
      return GOC_SUCCESS;
    }
  }
#endif
  uint32_t result[Lanes];
  for (int lane = 0; lane < Lanes; ++lane)
    result[lane] = evaluate<Op>(a[0][lane], bp ? bp[lane] : 0, unsigned(lane));
  for (int lane = 0; lane < Lanes; ++lane)
    if ((mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_rdna4_v_clz_i32_u32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  return bit_count<goc::BitCount::Leading, 32>(flags, exec_mask, instruction_flags, d, a, nullptr);
}

int goc_rdna4_v_ctz_i32_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a) {
  return bit_count<goc::BitCount::Trailing, 32>(flags, exec_mask, instruction_flags, d, a, nullptr);
}

int goc_rdna4_v_cls_i32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *a) {
  return bit_count<goc::BitCount::Sign, 32>(flags, exec_mask, instruction_flags, d, a, nullptr);
}

int goc_rdna4_v_bcnt_u32_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             const uint32_t *const *b) {
  return bit_count<goc::BitCount::Population, 32>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_mbcnt_lo_u32_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b) {
  return bit_count<goc::BitCount::MaskedLow, 32>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4_v_mbcnt_hi_u32_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b) {
  return bit_count<goc::BitCount::MaskedHigh, 32>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4w64_v_mbcnt_lo_u32_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b) {
  return bit_count<goc::BitCount::MaskedLow, 64>(flags, exec_mask, instruction_flags, d, a, b);
}

int goc_rdna4w64_v_mbcnt_hi_u32_b32(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b) {
  return bit_count<goc::BitCount::MaskedHigh, 64>(flags, exec_mask, instruction_flags, d, a, b);
}
