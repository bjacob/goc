// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "goc/goc.h"
#include "sad_reference.h"

#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <vector>

namespace {
const decltype(&goc_v_sad_u8_wave64) functions[] = {
    goc_v_sad_u8_wave64,          goc_v_sad_hi_u8_wave64,   goc_v_sad_u16_wave64,
    goc_v_sad_u32_wave64,         goc_v_msad_u8_wave64,     goc_v_qsad_pk_u16_u8_wave64,
    goc_v_mqsad_pk_u16_u8_wave64, goc_v_mqsad_u32_u8_wave64};
} // namespace

TEST(SadWave64, WindowsSaturationDppMasksAndAliases) {
  std::mt19937 random(93171);
  std::vector<uint64_t> routes = {0};
  for (auto route : goc_test::dpp_modes)
    routes.push_back(route);
  for (unsigned rows = 0; rows < 16; ++rows)
    routes.push_back((goc_test::dpp_modes[2] & ~GOC_DPP_ROW_MASK) |
                     (uint64_t(rows) << GOC_DPP_ROW_SHIFT));
  for (unsigned op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (bool clamp : {false, true})
          for (auto route : routes) {
            if (op >= 5 && route)
              continue;
            for (unsigned alias = 0; alias < 8; ++alias) {
              uint32_t initial[11][66], words[11][66];
              for (auto &reg : initial)
                for (auto &word : reg)
                  word = random();
              for (unsigned lane = 1; lane <= 64; ++lane) {
                if (lane % 4 == 0)
                  initial[2][lane] = 0;
                if (lane % 4 == 1)
                  initial[2][lane] &= 0xff00ff00;
                if (lane % 4 == 2) {
                  initial[0][lane] = UINT32_MAX;
                  initial[2][lane] = 0;
                }
                for (unsigned reg = 3; reg < 7; ++reg)
                  if (lane % 2)
                    initial[reg][lane] = UINT32_MAX - lane;
              }
              std::memcpy(words, initial, sizeof words);
              const uint32_t *a[] = {words[0] + 1, words[1] + 1};
              const uint32_t *b[] = {words[2] + 1};
              const uint32_t *c[] = {words[3] + 1, words[4] + 1, words[5] + 1, words[6] + 1};
              uint32_t *d[] = {words[7] + 1, words[8] + 1, words[9] + 1, words[10] + 1};
              const unsigned outputs = op == 7 ? 4 : op >= 5 ? 2 : 1;
              // Alias a different result register with each source VGPR in turn.
              unsigned result_reg = alias % outputs;
              if (alias)
                d[result_reg] = words[alias - 1] + 1;
              uint64_t exec_mask = alias ? 0x87654321fedcba98ULL : UINT64_MAX;
              ASSERT_EQ(functions[op](cpu | semantics | GOC_SEMANTICS_STRICT, exec_mask,
                                      route | (clamp ? GOC_ALU_CLAMP : 0), d, a, b, c),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 64; ++lane) {
                int source = int(lane);
                bool enabled = (exec_mask >> lane) & 1;
                if (route)
                  enabled = goc_test::dpp_source(route, exec_mask, lane, source);
                uint32_t accumulators[4];
                for (unsigned reg = 0; reg < 4; ++reg)
                  accumulators[reg] = initial[reg + 3][lane + 1];
                auto expected = goc_test::sad_reference(op, source < 0 ? 0 : initial[0][source + 1],
                                                        initial[1][lane + 1], initial[2][lane + 1],
                                                        accumulators, clamp);
                for (unsigned reg = 0; reg < outputs; ++reg) {
                  unsigned dest = alias && reg == result_reg ? alias - 1 : reg + 7;
                  ASSERT_EQ(d[reg][lane], enabled ? expected[reg] : initial[dest][lane + 1])
                      << op << '/' << lane << '/' << reg << '/' << alias;
                }
              }
              for (unsigned reg = 0; reg < 11; ++reg) {
                EXPECT_EQ(words[reg][0], initial[reg][0]);
                EXPECT_EQ(words[reg][65], initial[reg][65]);
              }
            }
          }
}

TEST(SadWave64, EmptyAndInvalidCallsDoNotAccessOperands) {
  for (unsigned op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint64_t flags = cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
      EXPECT_EQ(functions[op](flags, 0, GOC_ALU_CLAMP, nullptr, nullptr, nullptr, nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(functions[op](flags, UINT64_MAX, GOC_ALU_NEG_A, nullptr, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(
          functions[op](flags | (1ULL << 63), UINT64_MAX, 0, nullptr, nullptr, nullptr, nullptr),
          GOC_ERROR_INVALID_FLAGS);
      for (auto route : goc_test::dpp_modes) {
        EXPECT_EQ(functions[op](flags, 0, route, nullptr, nullptr, nullptr, nullptr),
                  op >= 5 ? GOC_ERROR_INVALID_FLAGS : GOC_SUCCESS);
      }
    }
}
