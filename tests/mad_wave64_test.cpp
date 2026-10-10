// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "goc/goc.h"
#include "integer_mad_reference.h"
#include "mad64_reference.h"

#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

namespace {
const decltype(&goc_v_mad_co_u64_u32_wave64) wide[] = {
    goc_v_mad_co_u64_u32_wave64, goc_v_mad_co_i64_i32_wave64, goc_v_mad_u64_u32_wave64,
    goc_v_mad_i64_i32_wave64};
} // namespace

TEST(MadWave64, NarrowFactorsModifiersDppAndAliases) {
  const uint32_t edges[] = {0, 1, 0x7fff, 0x8000, 0xffff, 0x7fffff, 0x800000, UINT32_MAX};
  std::vector<uint64_t> routes = {0};
  for (auto route : goc_test::dpp_modes)
    routes.push_back(route);
  for (unsigned rows = 0; rows < 16; ++rows)
    routes.push_back((goc_test::dpp_modes[2] & ~GOC_DPP_ROW_MASK) |
                     (uint64_t(rows) << GOC_DPP_ROW_SHIFT));
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (unsigned m = 0; m < (op < 2 ? 8U : 2U); ++m)
          for (auto route : routes)
            for (unsigned batch = 0; batch < 8; ++batch)
              for (unsigned alias = 0; alias < 4; ++alias) {
                uint32_t initial[4][66], words[4][66];
                for (unsigned lane = 0; lane < 66; ++lane) {
                  unsigned index = batch * 64 + lane;
                  for (unsigned j = 0; j < 3; ++j) {
                    unsigned selector = (index >> (j * 3)) & 7;
                    initial[j][lane] = edges[selector] ^ (edges[7 - selector] << 16);
                  }
                  initial[3][lane] = 0xdeadbeef;
                }
                std::memcpy(words, initial, sizeof words);
                unsigned dest = alias ? alias - 1 : 3;
                const uint32_t *a = words[0] + 1, *b = words[1] + 1, *c = words[2] + 1;
                uint32_t *d = words[dest] + 1;
                uint64_t exec_mask = alias ? 0x87654321fedcba98ULL : UINT64_MAX;
                uint32_t mode = goc_test::integer_mad_mode_bits(m);
                ASSERT_EQ(goc_test::integer_mad_wave64_functions[op](
                              cpu | semantics | GOC_SEMANTICS_STRICT, exec_mask, route | mode, &d,
                              &a, &b, &c),
                          GOC_SUCCESS);
                for (unsigned lane = 0; lane < 64; ++lane) {
                  int source = int(lane);
                  bool enabled = (exec_mask >> lane) & 1;
                  if (route)
                    enabled = goc_test::dpp_source(route, exec_mask, lane, source);
                  uint32_t av = source < 0 ? 0 : initial[0][source + 1];
                  uint32_t expected =
                      enabled ? goc_test::integer_mad_reference(op, av, initial[1][lane + 1],
                                                                initial[2][lane + 1], mode)
                              : initial[dest][lane + 1];
                  ASSERT_EQ(d[lane], expected) << op << '/' << batch << '/' << lane;
                }
                EXPECT_EQ(words[dest][0], initial[dest][0]);
                EXPECT_EQ(words[dest][65], initial[dest][65]);
              }
}

TEST(MadWave64, WideExtendedSumSaturationMasksAndAliases) {
  const uint32_t factors[] = {0, 1, 2, 0x7fffffff, 0x80000000, 0x80000001, 0xfffffffe, UINT32_MAX};
  const uint64_t addends[] = {0,
                              1,
                              UINT64_MAX,
                              UINT64_MAX - 1,
                              UINT64_MAX >> 1,
                              1ULL << 63,
                              (1ULL << 63) + 1,
                              0xffffffffULL};
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (unsigned clamp = 0; clamp < 2; ++clamp)
          for (unsigned batch = 0; batch < 8; ++batch)
            for (unsigned alias = 0; alias < 3; ++alias) {
              uint32_t words[6][66], initial[6][66];
              for (unsigned lane = 0; lane < 66; ++lane) {
                initial[0][lane] = factors[lane % 8];
                initial[1][lane] = factors[(lane / 8) % 8];
                initial[2][lane] = uint32_t(addends[batch]);
                initial[3][lane] = uint32_t(addends[batch] >> 32);
                initial[4][lane] = initial[5][lane] = 0xdeadbeef;
              }
              std::memcpy(words, initial, sizeof words);
              unsigned dest = alias == 0 ? 4 : alias == 1 ? 2 : 0;
              uint32_t *d[] = {words[dest] + 1, words[dest + 1] + 1};
              const uint32_t *a = words[0] + 1, *b = words[1] + 1;
              const uint32_t *c[] = {words[2] + 1, words[3] + 1};
              uint64_t exec_mask = alias ? 0x87654321fedcba98ULL : UINT64_MAX;
              uint64_t carry = 123, expected_carry = 0;
              ASSERT_EQ(wide[op](cpu | semantics | GOC_SEMANTICS_STRICT, exec_mask,
                                 clamp ? GOC_ALU_CLAMP : 0, d, &carry, &a, &b, c),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 64; ++lane) {
                auto expected = goc_test::mad64_reference(
                    op & 1, initial[0][lane + 1], initial[1][lane + 1], addends[batch], clamp);
                bool enabled = (exec_mask >> lane) & 1;
                EXPECT_EQ(d[0][lane], enabled ? uint32_t(expected.value) : initial[dest][lane + 1]);
                EXPECT_EQ(d[1][lane],
                          enabled ? uint32_t(expected.value >> 32) : initial[dest + 1][lane + 1]);
                if (enabled && expected.carry)
                  expected_carry |= 1ULL << lane;
              }
              EXPECT_EQ(carry, expected_carry);
              for (unsigned j = dest; j < dest + 2; ++j) {
                EXPECT_EQ(words[j][0], initial[j][0]);
                EXPECT_EQ(words[j][65], initial[j][65]);
              }
            }
}

TEST(MadWave64, ScalarOutputAliasValidationAndEmptyExec) {
  for (auto fn : wide) {
    uint32_t words[6][64];
    for (unsigned j = 0; j < 6; ++j)
      for (unsigned lane = 0; lane < 64; ++lane)
        words[j][lane] = (lane * 0x9e3779b9U) ^ (0xa5a59669U + j);
    const uint32_t *a = words[0], *b = words[1], *c[] = {words[2], words[3]};
    uint32_t *d[] = {words[4], words[5]};
    uint64_t carry, expected_carry;
    ASSERT_EQ(fn(0, UINT64_MAX, 0, d, &expected_carry, &a, &b, c), GOC_SUCCESS);
    uint32_t expected[2][64];
    std::memcpy(expected, words + 4, sizeof expected);
    ASSERT_EQ(fn(0, UINT64_MAX, 0, d, reinterpret_cast<uint64_t *>(words[0] + 33), &a, &b, c),
              GOC_SUCCESS);
    std::memcpy(&carry, words[0] + 33, sizeof carry);
    EXPECT_EQ(carry, expected_carry);
    EXPECT_EQ(std::memcmp(expected, words + 4, sizeof expected), 0);
    for (auto route : goc_test::dpp_modes)
      EXPECT_EQ(fn(0, UINT64_MAX, route, nullptr, &carry, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(carry, expected_carry);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr, &carry,
                 nullptr, nullptr, nullptr),
              GOC_SUCCESS);
    EXPECT_EQ(carry, 0U);
  }
  for (auto fn : goc_test::integer_mad_wave64_functions) {
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr, nullptr,
                 nullptr, nullptr),
              GOC_SUCCESS);
    EXPECT_EQ(fn(0, UINT64_MAX, 1ULL << 63, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
  }
}
