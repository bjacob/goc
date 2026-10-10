// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "goc/goc.h"
#include "relative_swap_hardware.h"

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

TEST(Swap, HalfSelectionsBothWavesMasksAndSameRegisterExchange) {
  for (unsigned lanes : {32U, 64U})
    for (unsigned halves = 0; halves < 4; ++halves)
      for (uint64_t exec_mask : masks)
        for (bool alias : {false, true}) {
          uint32_t data[2][66], expected[2][66];
          for (unsigned reg = 0; reg < 2; ++reg)
            for (unsigned lane = 0; lane < 66; ++lane)
              data[reg][lane] = expected[reg][lane] =
                  0xabcd1234U + lane * 0x10001U + reg * 0x76543210U;
          unsigned sa = halves & 1 ? 16 : 0, sd = halves & 2 ? 16 : 0;
          unsigned ar = alias ? 0 : 1;
          for (unsigned lane = 0; lane < lanes; ++lane)
            if ((exec_mask >> lane) & 1) {
              uint32_t a = (data[ar][lane + 1] >> sa) & 65535;
              uint32_t d = (data[0][lane + 1] >> sd) & 65535;
              expected[0][lane + 1] = (expected[0][lane + 1] & ~(65535U << sd)) | (a << sd);
              expected[ar][lane + 1] = (expected[ar][lane + 1] & ~(65535U << sa)) | (d << sa);
            }
          uint32_t *d = data[0] + 1, *a = data[ar] + 1;
          uint64_t mode = (sa ? GOC_ALU_HIGH_A : 0) | (sd ? GOC_ALU_HIGH_D : 0);
          ASSERT_EQ(lanes == 32 ? goc_v_swap_b16(exact, uint32_t(exec_mask), mode, &d, &a)
                                : goc_v_swap_b16_wave64(exact, exec_mask, mode, &d, &a),
                    GOC_SUCCESS);
          EXPECT_EQ(std::memcmp(data, expected, sizeof(data)), 0);
        }
  EXPECT_EQ(goc_v_swap_b16(exact, 0, 0, nullptr, nullptr), GOC_SUCCESS);
  EXPECT_EQ(goc_v_swap_b16_wave64(exact, 0, 0, nullptr, nullptr), GOC_SUCCESS);
  for (uint64_t mode : {GOC_ALU_NEG_A, GOC_ALU_ABS_A, GOC_ALU_CLAMP, GOC_ALU_HIGH_B}) {
    EXPECT_EQ(goc_v_swap_b16(exact, UINT32_MAX, mode, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_v_swap_b16_wave64(exact, UINT64_MAX, mode, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
  }
}

TEST(Swap, RelativeBothWavesMasksAndOverlappingFiles) {
  for (unsigned lanes : {32U, 64U})
    for (uint64_t exec_mask : masks)
      for (uint64_t semantics : std::initializer_list<uint64_t>{0, exact})
        for (unsigned src_offset : {0U, 1U, 5U, 7U, 256U, 1023U})
          for (unsigned dst_offset : {0U, 1U, 5U, 7U, 256U, 1023U}) {
            uint32_t data[8][66], expected[8][66];
            uint32_t *d[8], *a[8];
            for (unsigned r = 0; r < 8; ++r) {
              d[r] = data[r] + 1;
              a[r] = data[7 - r] + 1;
              for (unsigned lane = 0; lane < 66; ++lane)
                data[r][lane] = expected[r][lane] = r * 1000 + lane;
            }
            unsigned src = 1 + src_offset, dst = 2 + dst_offset;
            if (src < 8 && dst < 8)
              for (unsigned lane = 0; lane < lanes; ++lane)
                if ((exec_mask >> lane) & 1) {
                  expected[dst][lane + 1] = data[7 - src][lane + 1];
                  expected[7 - src][lane + 1] = data[dst][lane + 1];
                }
            uint32_t m0 = src_offset | (dst_offset << 16) | 0xfc00fc00U;
            ASSERT_EQ(
                lanes == 32
                    ? goc_v_swaprel_b32(semantics, uint32_t(exec_mask), 0, d, a, 8, 8, 2, 1, m0)
                    : goc_v_swaprel_b32_wave64(semantics, exec_mask, 0, d, a, 8, 8, 2, 1, m0),
                GOC_SUCCESS);
            EXPECT_EQ(std::memcmp(data, expected, sizeof data), 0)
                << lanes << '/' << exec_mask << '/' << src_offset << '/' << dst_offset;
          }
}

TEST(Swap, RelativeInvalidOperandDiscardsBothWrites) {
  uint32_t left[64], right[64];
  std::fill(left, left + 64, 123);
  std::fill(right, right + 64, 456);
  auto d = left, a = right;
  for (uint32_t base : {1U, UINT32_MAX}) {
    ASSERT_EQ(goc_v_swaprel_b32(0, UINT32_MAX, 0, &d, &a, 1, 1, base, 0, 1U << 16), GOC_SUCCESS);
    ASSERT_EQ(goc_v_swaprel_b32_wave64(0, UINT64_MAX, 0, &d, &a, 1, 1, 0, base, 1), GOC_SUCCESS);
  }
  for (unsigned lane = 0; lane < 64; ++lane) {
    EXPECT_EQ(left[lane], 123U);
    EXPECT_EQ(right[lane], 456U);
  }
  EXPECT_EQ(goc_v_swaprel_b32(exact, UINT32_MAX, 0, nullptr, nullptr, 0, 0, 0, 0, 0), GOC_SUCCESS);
  EXPECT_EQ(goc_v_swaprel_b32_wave64(exact, UINT64_MAX, 0, nullptr, nullptr, 0, 0, 0, 0, 0),
            GOC_SUCCESS);
  EXPECT_EQ(goc_v_swaprel_b32(exact, 0, 0, nullptr, nullptr, 256, 256, 0, 0, 0), GOC_SUCCESS);
  EXPECT_EQ(goc_v_swaprel_b32_wave64(exact, 0, 0, nullptr, nullptr, 256, 256, 0, 0, 0),
            GOC_SUCCESS);
  for (unsigned bit = 0; bit < 64; ++bit) {
    EXPECT_EQ(goc_v_swaprel_b32(0, UINT32_MAX, 1ULL << bit, nullptr, nullptr, 256, 256, 2, 4, 1),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_v_swaprel_b32_wave64(0, 0, 1ULL << bit, nullptr, nullptr, 0, 0, 0, 0, 0),
              GOC_ERROR_INVALID_FLAGS);
  }
}

TEST(Swap, RelativeHardwareBothWaves) {
  const uint32_t offsets[] = {0,   2 | (4U << 16), 1U << 16,
                              256, 256U << 16,     0xfc00fc00U | 2 | (4U << 16)};
  const uint64_t exec_masks[] = {0, 1, 0xaaaaaaaaaaaaaaaaULL, UINT64_MAX};
  for (unsigned wave = 0; wave < 2; ++wave) {
    unsigned lanes = wave ? 64 : 32;
    for (unsigned op = 0; op < 6; ++op)
      for (unsigned m = 0; m < 4; ++m) {
        uint32_t data[8][64];
        uint32_t *v[8];
        for (unsigned r = 0; r < 8; ++r) {
          v[r] = data[r];
          for (unsigned lane = 0; lane < lanes; ++lane)
            data[r][lane] = 1000 * r + lane;
        }
        ASSERT_EQ(
            wave ? goc_v_swaprel_b32_wave64(exact, exec_masks[m], 0, v, v, 8, 8, 0, 1, offsets[op])
                 : goc_v_swaprel_b32(exact, uint32_t(exec_masks[m]), 0, v, v, 8, 8, 0, 1,
                                     offsets[op]),
            GOC_SUCCESS);
        uint64_t hash = goc_test::capture_hash_seed;
        for (unsigned lane = 0; lane < lanes; ++lane)
          for (unsigned r = 0; r < 8; ++r)
            hash = goc_test::capture_hash_word(hash, data[r][lane]);
        EXPECT_EQ(hash, goc_test::relative_swap_hashes[wave][op * 4 + m])
            << wave << '/' << op << '/' << m;
      }
  }
}
