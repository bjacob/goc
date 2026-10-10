// SPDX-License-Identifier: MIT

#include "goc/goc.h"

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

namespace {

const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
const uint64_t masks[] = {0,
                          1,
                          1ULL << 31,
                          1ULL << 32,
                          1ULL << 63,
                          0xffffffffULL,
                          0xffffffff00000000ULL,
                          0xaaaaaaaaaaaaaaaaULL,
                          UINT64_MAX};

} // namespace

TEST(Swap, BothWaveSizesMasksAndAliases) {
  for (unsigned lanes : {32U, 64U})
    for (uint64_t exec_mask : masks)
      for (bool alias : {false, true})
        for (uint64_t semantics : std::initializer_list<uint64_t>{0, exact}) {
          uint32_t data[2][66], before[2][66];
          for (unsigned reg = 0; reg < 2; ++reg)
            for (unsigned lane = 0; lane < 66; ++lane)
              data[reg][lane] = 0x7f800001U + reg * 0x80000000U + lane * 37;
          std::memcpy(before, data, sizeof(data));
          uint32_t *d[] = {data[0] + 1}, *a[] = {data[alias ? 0 : 1] + 1};
          ASSERT_EQ(lanes == 32 ? goc_v_swap_b32(semantics, uint32_t(exec_mask), 0, d, a)
                                : goc_v_swap_b32_wave64(semantics, exec_mask, 0, d, a),
                    GOC_SUCCESS);
          for (unsigned reg = 0; reg < 2; ++reg)
            for (unsigned lane = 0; lane < 66; ++lane) {
              bool active = lane > 0 && lane <= lanes && ((exec_mask >> (lane - 1)) & 1);
              EXPECT_EQ(data[reg][lane], before[active && !alias ? reg ^ 1 : reg][lane]);
            }
        }
}

TEST(Swap, Permlane64ReadsInactiveSourcesAndPreservesOriginalAliasedInput) {
  for (uint64_t exec_mask : masks)
    for (bool alias : {false, true}) {
      uint32_t data[2][66], before[2][66];
      for (unsigned reg = 0; reg < 2; ++reg)
        for (unsigned lane = 0; lane < 66; ++lane)
          data[reg][lane] = 0x80000000U + reg * 1000 + lane;
      std::memcpy(before, data, sizeof(data));
      uint32_t *d[] = {data[alias ? 0 : 1] + 1};
      const uint32_t *a[] = {data[0] + 1};
      ASSERT_EQ(goc_v_permlane64_b32_wave64(exact, exec_mask, 0, d, a), GOC_SUCCESS);
      for (unsigned reg = 0; reg < 2; ++reg)
        for (unsigned lane = 0; lane < 66; ++lane) {
          bool active =
              reg == (alias ? 0U : 1U) && lane > 0 && lane <= 64 && ((exec_mask >> (lane - 1)) & 1);
          EXPECT_EQ(data[reg][lane], active ? before[0][((lane - 1) ^ 32) + 1] : before[reg][lane]);
        }
    }
}

TEST(Swap, Wave32PermlaneIsNoOp) {
  uint32_t data[32];
  std::fill_n(data, 32, 0xdeadbeefU);
  uint32_t *d = data;
  EXPECT_EQ(goc_v_permlane64_b32(exact, 0, &d, nullptr), GOC_SUCCESS);
  EXPECT_EQ(goc_v_permlane64_b32(exact, 0, nullptr, nullptr), GOC_SUCCESS);
  for (uint32_t value : data)
    EXPECT_EQ(value, 0xdeadbeefU);
}

TEST(Swap, ValidationAndEmptyExec) {
  uint32_t data[64];
  std::fill_n(data, 64, 0xdeadbeefU);
  uint32_t *d = data;
  for (unsigned bit = 0; bit < 64; ++bit) {
    uint64_t mode = 1ULL << bit;
    EXPECT_EQ(goc_v_swap_b32(0, UINT32_MAX, mode, &d, nullptr), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_v_swap_b32_wave64(0, UINT64_MAX, mode, &d, nullptr), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_v_permlane64_b32(0, mode, &d, nullptr), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_v_permlane64_b32_wave64(0, UINT64_MAX, mode, &d, nullptr),
              GOC_ERROR_INVALID_FLAGS);
  }
  EXPECT_EQ(goc_v_swap_b32(exact, 0, 0, nullptr, nullptr), GOC_SUCCESS);
  EXPECT_EQ(goc_v_swap_b32_wave64(exact, 0, 0, nullptr, nullptr), GOC_SUCCESS);
  EXPECT_EQ(goc_v_permlane64_b32_wave64(exact, 0, 0, nullptr, nullptr), GOC_SUCCESS);
  for (uint32_t value : data)
    EXPECT_EQ(value, 0xdeadbeefU);
}
