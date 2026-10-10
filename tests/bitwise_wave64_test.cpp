// SPDX-License-Identifier: MIT

#include "bitfield_reference.h"
#include "dpp_reference.h"
#include "goc/goc.h"
#include "shift_reference.h"

#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

namespace {
const decltype(&goc_v_lshlrev_b32_wave64) shifts[] = {
    goc_v_lshlrev_b32_wave64, goc_v_lshrrev_b32_wave64, goc_v_ashrrev_i32_wave64,
    goc_v_lshlrev_b64_wave64, goc_v_lshrrev_b64_wave64, goc_v_ashrrev_i64_wave64};

std::vector<uint64_t> routes() {
  std::vector<uint64_t> result = {0};
  for (auto mode : goc_test::dpp_modes)
    result.push_back(mode);
  for (unsigned rows = 0; rows < 16; ++rows)
    result.push_back((goc_test::dpp_modes[2] & ~GOC_DPP_ROW_MASK) |
                     (uint64_t(rows) << GOC_DPP_ROW_SHIFT));
  return result;
}
} // namespace

TEST(BitwiseWave64, ShiftsCountsBitsMasksRoutingAndAliases) {
  for (unsigned op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (uint64_t mode : routes()) {
          if (mode && op >= 3)
            continue;
          for (unsigned bit = 0; bit < (op < 3 ? 32U : 64U); ++bit)
            for (unsigned alias = 0; alias < 2; ++alias) {
              uint32_t a[64], b[2][64], d[2][64];
              uint64_t initial[64];
              for (unsigned lane = 0; lane < 64; ++lane) {
                a[lane] = lane | 0xffff0000U;
                initial[lane] = (lane & 1) ? ~(1ULL << bit) : 1ULL << bit;
                b[0][lane] = uint32_t(initial[lane]);
                b[1][lane] = uint32_t(initial[lane] >> 32);
                d[0][lane] = d[1][lane] = 0xdeadbeef;
              }
              const uint32_t *pa = a, *pb[] = {b[0], b[1]};
              uint32_t *pd[] = {alias ? b[0] : d[0], alias ? b[1] : d[1]};
              const uint64_t exec_mask = bit & 1 ? UINT64_MAX : 0x87654321fedcba98ULL;
              ASSERT_EQ(
                  shifts[op](cpu | semantics | GOC_SEMANTICS_STRICT, exec_mask, mode, pd, &pa, pb),
                  GOC_SUCCESS);
              for (unsigned lane = 0; lane < 64; ++lane) {
                int source = int(lane);
                bool enabled = (exec_mask >> lane) & 1;
                if (mode)
                  enabled = goc_test::dpp_source(mode, exec_mask, lane, source);
                uint32_t count = source < 0 ? 0 : a[source];
                uint64_t want = enabled ? goc_test::shift_reference(op < 3 ? 32 : 64, op % 3, count,
                                                                    initial[lane])
                                : alias ? initial[lane]
                                        : 0xdeadbeefdeadbeefULL;
                ASSERT_EQ(pd[0][lane], uint32_t(want));
                if (op >= 3) {
                  ASSERT_EQ(pd[1][lane], uint32_t(want >> 32));
                }
              }
            }
        }
}

TEST(BitwiseWave64, BitfieldsOffsetsWidthsSelectorsDppAndAliases) {
  for (unsigned op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (uint64_t mode : routes())
          for (unsigned batch = 0; batch < 16; ++batch)
            for (unsigned alias = 0; alias < 4; ++alias) {
              uint32_t words[4][66], initial[4][66];
              for (unsigned lane = 0; lane < 66; ++lane) {
                initial[0][lane] = 0x9e3779b9U * lane ^ (batch & 1 ? UINT32_MAX : 0);
                initial[1][lane] = batch * 2 + (lane / 32);
                initial[2][lane] = op == 7 ? (lane + batch * 64) * 0x01010101U : lane | 0xffff0000U;
                initial[3][lane] = 0xdeadbeef;
              }
              std::memcpy(words, initial, sizeof words);
              unsigned dest = alias ? alias - 1 : 3;
              const uint32_t *a = words[0] + 1, *b = words[1] + 1, *c = words[2] + 1;
              uint32_t *d = words[dest] + 1;
              const uint64_t exec_mask = batch & 1 ? UINT64_MAX : 0x87654321fedcba98ULL;
              ASSERT_EQ(
                  goc_test::bitfield_wave64_functions[op](cpu | semantics | GOC_SEMANTICS_STRICT,
                                                          exec_mask, mode, &d, &a, &b, &c),
                  GOC_SUCCESS);
              for (unsigned lane = 0; lane < 64; ++lane) {
                int source = int(lane);
                bool enabled = (exec_mask >> lane) & 1;
                if (mode)
                  enabled = goc_test::dpp_source(mode, exec_mask, lane, source);
                uint32_t av = source < 0 ? 0 : initial[0][source + 1];
                uint32_t want = enabled ? goc_test::bitfield_reference(op, av, initial[1][lane + 1],
                                                                       initial[2][lane + 1])
                                        : initial[dest][lane + 1];
                ASSERT_EQ(d[lane], want) << op << '/' << batch << '/' << lane;
              }
              EXPECT_EQ(words[dest][0], initial[dest][0]);
              EXPECT_EQ(words[dest][65], initial[dest][65]);
            }
}

TEST(BitwiseWave64, ValidationAndZeroExec) {
  for (unsigned op = 0; op < 6; ++op) {
    auto fn = shifts[op];
    EXPECT_EQ(
        fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr, nullptr, nullptr),
        GOC_SUCCESS);
    EXPECT_EQ(fn(0, UINT64_MAX, 1ULL << 63, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
    if (op >= 3) {
      for (auto mode : goc_test::dpp_modes)
        EXPECT_EQ(fn(0, UINT64_MAX, mode, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
    }
  }
  for (auto fn : goc_test::bitfield_wave64_functions) {
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr, nullptr,
                 nullptr, nullptr),
              GOC_SUCCESS);
    EXPECT_EQ(fn(0, UINT64_MAX, 1ULL << 63, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
  }
}
