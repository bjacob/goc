// SPDX-License-Identifier: MIT

#include "exec_masks.h"
#include "goc/goc.h"

#include <gtest/gtest.h>
#include <stdint.h>

TEST(Move, SharedRdna3Rdna4BitPatternsMasksAndAliases) {
  for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
    for (uint32_t exec_mask : exec_masks())
      for (bool alias : {false, true}) {
        uint32_t input[32], output[32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          input[lane] = 0x7f800001U + lane * 0x1234567U;
          output[lane] = 0xdeadbeef;
        }
        const uint32_t *a[] = {input};
        uint32_t *d[] = {alias ? input : output};
        ASSERT_EQ(goc_v_mov_b32(semantics | GOC_SEMANTICS_STRICT, exec_mask, 0, d, a), GOC_SUCCESS);
        for (unsigned lane = 0; lane < 32; ++lane)
          EXPECT_EQ(d[0][lane], alias || ((exec_mask >> lane) & 1) ? 0x7f800001U + lane * 0x1234567U
                                                                   : 0xdeadbeefU);
      }
}

TEST(Move, DppReadsBeforeAliasedStores) {
  uint32_t data[32];
  for (unsigned lane = 0; lane < 32; ++lane)
    data[lane] = lane;
  uint32_t *d[] = {data};
  const uint32_t *a[] = {data};
  uint64_t mode = GOC_DPP8;
  for (unsigned lane = 0; lane < 8; ++lane)
    mode |= uint64_t(7 - lane) << (40 + 3 * lane);
  ASSERT_EQ(goc_v_mov_b32(0, UINT32_MAX, mode, d, a), GOC_SUCCESS);
  for (unsigned lane = 0; lane < 32; ++lane)
    EXPECT_EQ(data[lane], lane ^ 7);
  EXPECT_EQ(goc_v_mov_b32(0, 0, 0, nullptr, nullptr), GOC_SUCCESS);
  EXPECT_EQ(goc_v_mov_b32(0, 0, 1, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
}
