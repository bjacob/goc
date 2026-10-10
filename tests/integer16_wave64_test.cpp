// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "goc/goc.h"
#include "integer16_reference.h"
#include "integer16_ternary_reference.h"
#include "packed_integer_reference.h"
#include "packed_mad_reference.h"

#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

namespace {
const decltype(&goc_v_pk_add_i16_wave64) packed[] = {
    goc_v_pk_add_i16_wave64,     goc_v_pk_sub_i16_wave64,     goc_v_pk_add_u16_wave64,
    goc_v_pk_sub_u16_wave64,     goc_v_pk_min_i16_wave64,     goc_v_pk_max_i16_wave64,
    goc_v_pk_min_u16_wave64,     goc_v_pk_max_u16_wave64,     goc_v_pk_mul_lo_u16_wave64,
    goc_v_pk_lshlrev_b16_wave64, goc_v_pk_lshrrev_b16_wave64, goc_v_pk_ashrrev_i16_wave64};
const decltype(&goc_v_pk_mad_i16_wave64) mad[] = {goc_v_pk_mad_u16_wave64, goc_v_pk_mad_i16_wave64};

int execute(unsigned op, uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (op < 12)
    return goc_test::integer16_wave64_functions[op](flags, exec_mask, mode, d, a, b);
  if (op < 20)
    return goc_test::integer16_ternary_wave64_functions[op - 12](flags, exec_mask, mode, d, a, b,
                                                                 c);
  if (op < 32)
    return packed[op - 20](flags, exec_mask, mode, d, a, b);
  return mad[op - 32](flags, exec_mask, mode, d, a, b, c);
}

uint32_t mode_bits(unsigned op, unsigned m) {
  if (op < 12)
    return goc_test::integer16_mode_bits(m);
  if (op < 20)
    return goc_test::integer16_ternary_mode_bits(m);
  if (op < 32)
    return (m & 1 ? GOC_PK_LO_A_HIGH : 0) | (m & 2 ? GOC_PK_LO_B_HIGH : 0) |
           (m & 4 ? GOC_PK_HI_A_LOW : 0) | (m & 8 ? GOC_PK_HI_B_LOW : 0) |
           (m & 16 ? GOC_PK_CLAMP : 0);
  return (m & 63) << 7 | (m & 64 ? GOC_PK_CLAMP : 0);
}
} // namespace

TEST(Integer16Wave64, HalfSelectorsSaturationRoutingAndAliases) {
  const uint32_t edge[] = {0, 1, 2, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff};
  for (unsigned op = 0; op < 34; ++op) {
    std::vector<uint64_t> routes = {0};
    if (op < 20) {
      for (auto route : goc_test::dpp_modes)
        routes.push_back(route);
      for (unsigned rows = 0; rows < 16; ++rows)
        routes.push_back((goc_test::dpp_modes[2] & ~GOC_DPP_ROW_MASK) |
                         (uint64_t(rows) << GOC_DPP_ROW_SHIFT));
    }
    unsigned mode_count = op < 12 ? 16 : op < 14 ? 32 : op < 20 ? 16 : op < 32 ? 32 : 128;
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (unsigned m = 0; m < mode_count; ++m)
          for (auto route : routes)
            for (unsigned alias = 0; alias < 4; ++alias) {
              uint32_t initial[4][66], words[4][66];
              for (unsigned lane = 0; lane < 66; ++lane) {
                for (unsigned j = 0; j < 3; ++j) {
                  unsigned index = lane + m * 64;
                  initial[j][lane] =
                      edge[(index >> (j * 3)) & 7] | (edge[7 - ((index >> (j * 3)) & 7)] << 16);
                  if (lane >= 48)
                    initial[j][lane] ^= index * (0x9e3779b9U + j * 123);
                }
                initial[3][lane] = 0xfacecafe;
              }
              std::memcpy(words, initial, sizeof words);
              unsigned dest = alias ? alias - 1 : 3;
              const uint32_t *a = words[0] + 1, *b = words[1] + 1, *c = words[2] + 1;
              uint32_t *d = words[dest] + 1;
              uint64_t exec_mask = alias ? 0x87654321fedcba98ULL : UINT64_MAX;
              uint32_t low_mode = mode_bits(op, m);
              ASSERT_EQ(execute(op, cpu | semantics | GOC_SEMANTICS_STRICT, exec_mask,
                                route | low_mode, &d, &a, &b, &c),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 64; ++lane) {
                int source = int(lane);
                bool enabled = (exec_mask >> lane) & 1;
                if (route)
                  enabled = goc_test::dpp_source(route, exec_mask, lane, source);
                uint32_t av = source < 0 ? 0 : initial[0][source + 1];
                uint32_t bv = initial[1][lane + 1], cv = initial[2][lane + 1];
                uint32_t want = initial[dest][lane + 1];
                if (enabled) {
                  if (op < 12)
                    want = goc_test::integer16_reference(op, av, bv, want, low_mode);
                  else if (op < 20)
                    want =
                        goc_test::integer16_ternary_reference(op - 12, av, bv, cv, low_mode, want);
                  else if (op < 32)
                    want = goc_test::packed_integer_reference(op - 20, av, bv, low_mode);
                  else
                    want = goc_test::packed_mad_reference(op - 32, av, bv, cv, low_mode);
                }
                ASSERT_EQ(d[lane], want) << op << '/' << m << '/' << lane << '/' << route;
              }
              EXPECT_EQ(words[dest][0], initial[dest][0]);
              EXPECT_EQ(words[dest][65], initial[dest][65]);
            }
  }
}

TEST(Integer16Wave64, ValidationBeforeOperandAccess) {
  for (unsigned op = 0; op < 34; ++op) {
    EXPECT_EQ(execute(op, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr,
                      nullptr, nullptr, nullptr),
              GOC_SUCCESS);
    EXPECT_EQ(execute(op, 0, UINT64_MAX, 1ULL << 63, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    if (op >= 20) {
      for (auto route : goc_test::dpp_modes)
        EXPECT_EQ(execute(op, 0, UINT64_MAX, route, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
    }
  }
}
