// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"

#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

// RX 9070, gfx1201, MODE=0xf0: RNE and preserved denormals.
// Four input sets, LDEXP_F32 / CVT_F32_F16 / CVT_F32_F64, all 32 modifiers,
// 32 lanes. FNV-1a over little-endian result bytes, NaNs normalized to 0x7fc00000.
const uint64_t hardware_hashes[4][3] = {
    {UINT64_C(0x1f93c115d1cdd755), UINT64_C(0xa358ade8ebb6a9d5), UINT64_C(0x1f93c115d1cdd755)},
    {UINT64_C(0xd4295ecd2f0742a5), UINT64_C(0xa358ade8ebb6a9d5), UINT64_C(0xd4295ecd2f0742a5)},
    {UINT64_C(0x54aa0c65d508fe45), UINT64_C(0xa358ade8ebb6a9d5), UINT64_C(0x54aa0c65d508fe45)},
    {UINT64_C(0x42ca079dcbd3a10d), UINT64_C(0xa358ade8ebb6a9d5), UINT64_C(0xe63df8098a3b6325)}};

const uint32_t inputs32[] = {
    0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
    0x00ffffff, 0x80ffffff, 0x01000000, 0x81000000, 0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000,
    0x40000000, 0xc0000000, 0x7e800000, 0xfe800000, 0x7f000000, 0xff000000, 0x7f7fffff, 0xff7fffff,
    0xc2fc0000, 0xc2fa0000, 0xc2fe0000, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345, 0x3f800001};

const uint32_t inputs16[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
                             0x07ff, 0x87ff, 0x0800, 0x8800, 0x3800, 0xb800, 0x3c00, 0xbc00,
                             0x4000, 0xc000, 0x7800, 0xf800, 0x7bff, 0xfbff, 0x7c00, 0xfc00,
                             0x7c01, 0xfc01, 0x7e01, 0xfe01, 0x3555, 0xb555, 0x3c01, 0xbc01};

} // namespace

TEST(OutputScaling, LdexpAndConversionsHardwareBoundaries) {
  const int exponents[] = {-200, -150, -127, -126, -1, 0, 1, 127, 128, 200, INT32_MIN, INT32_MAX};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned set = 0; set < 4; ++set)
      for (unsigned op = 0; op < 3; ++op)
        for (unsigned alias = 0; alias < 4; ++alias) {
          SCOPED_TRACE(::testing::Message() << cpu << '/' << set << '/' << op << '/' << alias);
          uint64_t hash = UINT64_C(14695981039346656037);
          for (uint32_t variant = 0; variant < 32; ++variant) {
            uint32_t words[4][32] = {};
            for (unsigned lane = 0; lane < 32; ++lane) {
              words[0][lane] = op == 1 ? inputs16[lane] : inputs32[lane];
              words[2][lane] = uint32_t(set == 0   ? 0
                                        : set == 1 ? -1
                                        : set == 2 ? 1
                                                   : exponents[lane % 12]);
              if (op == 2) {
                double value = double(goc::as_float(inputs32[lane]));
                if (set == 1)
                  value *= 0.5;
                if (set == 2)
                  value *= 2;
                if (set == 3)
                  value = lane & 1 ? -0x1p-150 : 0x1p-150;
                uint64_t raw;
                std::memcpy(&raw, &value, sizeof(raw));
                words[0][lane] = uint32_t(raw);
                words[1][lane] = uint32_t(raw >> 32);
              }
            }
            uint32_t *p[] = {words[0], words[1], words[2], words[3]};
            uint64_t mode = (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
                            ((variant >> 2 & 3) << 6) | (variant & 16 ? GOC_ALU_CLAMP : 0);
            int error = op == 0 ? goc_rdna4_v_ldexp_f32(cpu, UINT32_MAX, mode, &p[alias], p, p + 2)
                        : op == 1 ? goc_rdna4_v_cvt_f32_f16(cpu, UINT32_MAX, mode, &p[alias], p)
                                  : goc_rdna4_v_cvt_f32_f64(cpu, UINT32_MAX, mode, &p[alias], p);
            ASSERT_EQ(error, GOC_SUCCESS);
            for (uint32_t value : words[alias]) {
              if ((value & 0x7fffffff) > 0x7f800000)
                value = 0x7fc00000;
              for (unsigned shift = 0; shift < 32; shift += 8)
                hash = (hash ^ ((value >> shift) & 255)) * UINT64_C(1099511628211);
            }
          }
          EXPECT_EQ(hash, hardware_hashes[set][op]);
        }
}
