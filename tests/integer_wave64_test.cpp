// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "goc/goc.h"
#include "integer_add_reference.h"
#include "integer_minmax_reference.h"
#include "integer_mul_reference.h"
#include "integer_ternary_reference.h"

#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

namespace {
int execute(unsigned op, uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return op < 6 ? goc_test::integer_add_wave64_functions[op](flags, exec_mask, mode, d, a, b, c)
                : goc_test::integer_mul_wave64_functions[op - 6](flags, exec_mask, mode, d, a, b);
}
} // namespace

TEST(IntegerWave64, BoundariesSaturationRoutingMasksAndAliases) {
  const uint32_t edges[] = {0,          1,          2,          0x7ffffe,  0x7fffff,   0x800000,
                            0x800001,   0xfffffe,   0xffffff,   0x1000000, 0x7ffffffe, 0x7fffffff,
                            0x80000000, 0x80000001, 0xfffffffe, UINT32_MAX};
  for (unsigned op = 0; op < 13; ++op) {
    bool can_dpp = op < 6 || op >= 9;
    bool can_clamp = op < 5 || op == 9 || op == 11;
    std::vector<uint64_t> routes = {0};
    if (can_dpp) {
      for (auto route : goc_test::dpp_modes)
        routes.push_back(route);
      for (unsigned rows = 0; rows < 16; ++rows)
        routes.push_back((goc_test::dpp_modes[2] & ~GOC_DPP_ROW_MASK) |
                         (uint64_t(rows) << GOC_DPP_ROW_SHIFT));
    }
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (unsigned clamp = 0; clamp < (can_clamp ? 2U : 1U); ++clamp)
          for (auto route : routes)
            for (unsigned batch = 0; batch < 8; ++batch)
              for (unsigned alias = 0; alias < (op == 5 ? 4U : 3U); ++alias) {
                uint32_t initial[4][66], words[4][66];
                for (unsigned lane = 0; lane < 66; ++lane) {
                  unsigned index = batch * 64 + lane;
                  initial[0][lane] = edges[(index / 16) % 16];
                  initial[1][lane] = edges[index % 16];
                  initial[2][lane] = edges[(index + batch * 3) % 16];
                  if (batch >= 4) {
                    initial[0][lane] ^= index * 0x9e3779b9U;
                    initial[1][lane] ^= index * 0x12345679U;
                  }
                  initial[3][lane] = 0xdeadbeef;
                }
                std::memcpy(words, initial, sizeof words);
                unsigned dest = alias ? alias - 1 : 3;
                const uint32_t *a = words[0] + 1, *b = words[1] + 1, *c = words[2] + 1;
                uint32_t *d = words[dest] + 1;
                uint64_t exec_mask = batch & 1 ? UINT64_MAX : 0x87654321fedcba98ULL;
                uint64_t mode = route | (clamp ? GOC_ALU_CLAMP : 0);
                ASSERT_EQ(execute(op, cpu | semantics | GOC_SEMANTICS_STRICT, exec_mask, mode, &d,
                                  &a, &b, &c),
                          GOC_SUCCESS);
                for (unsigned lane = 0; lane < 64; ++lane) {
                  int source = int(lane);
                  bool enabled = (exec_mask >> lane) & 1;
                  if (route)
                    enabled = goc_test::dpp_source(mode, exec_mask, lane, source);
                  bool reverse = op == 2;
                  uint32_t permuted = source < 0 ? 0 : initial[reverse ? 1 : 0][source + 1];
                  uint32_t av = reverse ? initial[0][lane + 1] : permuted;
                  uint32_t bv = reverse ? permuted : initial[1][lane + 1];
                  uint32_t want = initial[dest][lane + 1];
                  if (enabled)
                    want = op < 6 ? goc_test::integer_add_reference(op, av, bv,
                                                                    initial[2][lane + 1], clamp)
                                  : goc_test::integer_mul_reference(op - 6, av, bv, clamp);
                  ASSERT_EQ(d[lane], want) << op << '/' << batch << '/' << lane << '/' << mode;
                }
                EXPECT_EQ(words[dest][0], initial[dest][0]);
                EXPECT_EQ(words[dest][65], initial[dest][65]);
              }
  }
}

TEST(IntegerWave64, ZeroExecAndValidation) {
  for (unsigned op = 0; op < 13; ++op) {
    EXPECT_EQ(execute(op, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr,
                      nullptr, nullptr, nullptr),
              GOC_SUCCESS);
    EXPECT_EQ(execute(op, 0, UINT64_MAX, 1ULL << 63, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    if (op >= 6 && op < 9) {
      for (auto mode : goc_test::dpp_modes)
        EXPECT_EQ(execute(op, 0, UINT64_MAX, mode, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
    }
    if (!(op < 5 || op == 9 || op == 11)) {
      EXPECT_EQ(execute(op, 0, UINT64_MAX, GOC_ALU_CLAMP, nullptr, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
    }
  }
}

TEST(IntegerWave64, SelectionAndTernaryBoundariesRoutingAndAliases) {
  const uint32_t edges[] = {0, 1, 2, 0x7fffffff, 0x80000000, 0x80000001, 0xfffffffe, UINT32_MAX};
  std::vector<uint64_t> routes = {0};
  for (auto route : goc_test::dpp_modes)
    routes.push_back(route);
  for (unsigned rows = 0; rows < 16; ++rows)
    routes.push_back((goc_test::dpp_modes[2] & ~GOC_DPP_ROW_MASK) |
                     (uint64_t(rows) << GOC_DPP_ROW_SHIFT));
  for (unsigned op = 0; op < 22; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (uint64_t mode : routes)
          for (unsigned batch = 0; batch < 16; ++batch)
            for (unsigned alias = 0; alias < 4; ++alias) {
              uint32_t initial[4][66], words[4][66];
              for (unsigned lane = 0; lane < 66; ++lane) {
                unsigned index = batch * 64 + lane;
                for (unsigned j = 0; j < 3; ++j)
                  initial[j][lane] = batch < 8
                                         ? edges[(index >> (j * 3)) & 7]
                                         : index * (0x9e3779b9U + j * 789) ^ (0xa5a59669U >> j);
                initial[3][lane] = 0xdeadbeef;
              }
              std::memcpy(words, initial, sizeof words);
              unsigned dest = alias ? alias - 1 : 3;
              uint32_t *d = words[dest] + 1;
              const uint32_t *a = words[0] + 1, *b = words[1] + 1, *c = words[2] + 1;
              uint64_t exec_mask = alias ? 0x87654321fedcba98ULL : UINT64_MAX;
              auto fn = op < 14 ? goc_test::integer_minmax_wave64_functions[op]
                                : goc_test::integer_ternary_wave64_functions[op - 14];
              ASSERT_EQ(fn(cpu | semantics | GOC_SEMANTICS_STRICT, exec_mask, mode, &d, &a, &b, &c),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 64; ++lane) {
                int source = int(lane);
                bool enabled = (exec_mask >> lane) & 1;
                if (mode)
                  enabled = goc_test::dpp_source(mode, exec_mask, lane, source);
                uint32_t av = source < 0 ? 0 : initial[0][source + 1];
                uint32_t bv = initial[1][lane + 1], cv = initial[2][lane + 1];
                uint32_t want = initial[dest][lane + 1];
                if (enabled)
                  want = op < 14 ? goc_test::integer_minmax_reference(op, av, bv, cv)
                                 : goc_test::integer_ternary_reference(op - 14, av, bv, cv);
                ASSERT_EQ(d[lane], want) << op << '/' << batch << '/' << lane << '/' << mode;
              }
              EXPECT_EQ(words[dest][0], initial[dest][0]);
              EXPECT_EQ(words[dest][65], initial[dest][65]);
            }
}

TEST(IntegerWave64, SelectionAndTernaryValidation) {
  for (unsigned op = 0; op < 22; ++op) {
    auto fn = op < 14 ? goc_test::integer_minmax_wave64_functions[op]
                      : goc_test::integer_ternary_wave64_functions[op - 14];
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr, nullptr,
                 nullptr, nullptr),
              GOC_SUCCESS);
    for (unsigned bit = 0; bit < 32; ++bit)
      EXPECT_EQ(fn(0, UINT64_MAX, 1ULL << bit, nullptr, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
  }
}
