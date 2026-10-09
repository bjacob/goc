// SPDX-License-Identifier: MIT

#ifndef GOC_TEST_RDNA4_SAD_REFERENCE_H_
#define GOC_TEST_RDNA4_SAD_REFERENCE_H_

#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdint.h>

namespace goc_test {

// op: SAD_U8, SAD_HI_U8, SAD_U16, SAD_U32, MSAD_U8, QSAD_PK_U16_U8,
// MQSAD_PK_U16_U8, MQSAD_U32_U8. Accumulators and results are low word first.
inline std::array<uint32_t, 4> sad_reference(int op, uint32_t a_low, uint32_t a_high, uint32_t b,
                                             const uint32_t *c, bool clamp) {
  const bool packed = op == 5 || op == 6;
  const bool masked = op == 4 || op == 6 || op == 7;
  const int width = op == 3 ? 32 : op == 2 ? 16 : 8;
  const uint64_t base = UINT64_C(1) << width;
  const uint64_t a = a_low + (uint64_t(a_high) << 32);
  std::array<uint32_t, 4> result{};
  for (int window = 0; window < (op >= 5 ? 4 : 1); ++window) {
    uint64_t x = a >> (window * 8), y = b, total = 0;
    for (int field = 0; field < 32 / width; ++field) {
      int64_t left = int64_t(x % base), right = int64_t(y % base);
      if (!masked || right != 0)
        total += uint64_t(std::abs(left - right));
      x /= base;
      y /= base;
    }
    if (op == 1)
      total *= 65536;
    uint64_t limit = packed ? 65535 : UINT32_MAX;
    total += packed ? (c[window / 2] >> (16 * (window % 2))) & 65535 : c[window];
    if (clamp)
      total = std::min(total, limit);
    else
      total %= limit + 1;
    if (packed)
      result[window / 2] |= uint32_t(total) << (16 * (window % 2));
    else
      result[window] = uint32_t(total);
  }
  return result;
}

} // namespace goc_test

#endif
