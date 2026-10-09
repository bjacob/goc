// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fma_rounding_exceptions_hardware.h"
#include "goc/goc.h"

#include <gtest/gtest.h>
#include <stdint.h>

TEST(FmaExceptions, RoundingBoundaryHardwareCorpora) {
  for (unsigned type = 0; type < 3; ++type) {
    unsigned width = 16U << type,
             fraction = type == 0   ? 10
                        : type == 1 ? 23
                                    : 52,
             bias = type == 0   ? 15
                    : type == 1 ? 127
                                : 1023;
    uint64_t sign = 1ULL << (width - 1), infinity = uint64_t(2 * bias + 1) << fraction,
             one = uint64_t(bias) << fraction,
             quarter_ulp = uint64_t(2 * bias - fraction - 2) << fraction;
    const uint64_t addends[2][8] = {{0, 1, 2, 3, sign, sign | 1, sign | 2, sign | 3},
                                    {0, 1, one, sign | one, quarter_ulp, sign | quarter_ulp,
                                     quarter_ulp + (1ULL << fraction),
                                     sign | (quarter_ulp + (1ULL << fraction))}};
    for (unsigned setting = 0; setting < 2; ++setting)
      for (unsigned variant = 0; variant < 16; ++variant) {
        uint32_t mode =
            ((variant & 3) << 6) | (variant & 4 ? GOC_ALU_CLAMP : 0) |
            (variant & 8 ? GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_ABS_B | GOC_ALU_NEG_C : 0);
        uint64_t hash = goc_test::capture_hash_seed;
        for (unsigned i = 0; i < 8192; ++i) {
          unsigned index = i & 4095;
          uint64_t inputs[] = {i < 4096 ? (1ULL << fraction) - 64 + (index >> 5)
                                        : infinity - 1 - (index >> 5),
                               one - 2 + ((index >> 3) & 3), addends[i / 4096][index & 7]};
          uint32_t words[4][2][32] = {};
          for (unsigned operand = 0; operand < 3; ++operand) {
            words[operand][0][0] = uint32_t(inputs[operand]);
            words[operand][1][0] = uint32_t(inputs[operand] >> 32);
          }
          const uint32_t *a[] = {words[0][0], words[0][1]}, *b[] = {words[1][0], words[1][1]},
                         *c[] = {words[2][0], words[2][1]};
          uint32_t *d[] = {words[3][0], words[3][1]}, exceptions = 0;
          uint64_t flags = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                           (i % (goc_init_cpu_flags() + 1));
          int status;
          if (type == 0)
            status = goc_v_fma_f16(flags | (setting ? GOC_FP16_OVFL : 0), 1, mode, d, a, b, c,
                                   &exceptions);
          else if (type == 1)
            status = goc_v_div_fmas_f32(flags, 1, mode, d, a, b, c, setting, &exceptions);
          else
            status = goc_v_div_fmas_f64(flags, 1, mode, d, a, b, c, setting, &exceptions);
          ASSERT_EQ(status, GOC_SUCCESS);
          hash = goc_test::capture_hash_word(hash, exceptions);
        }
        EXPECT_EQ(hash, goc_test::fma_rounding_exception_hashes[type][setting][variant])
            << type << "/" << setting << "/" << variant;
      }
  }
}
