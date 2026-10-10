// SPDX-License-Identifier: MIT

#include "boolean_reference.h"
#include "cndmask_reference.h"
#include "dpp_reference.h"
#include "goc/goc.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

namespace {
int execute(unsigned op, uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, uint64_t condition) {
  switch (op) {
  case 0:
    return goc_v_and_b32_wave64(flags, exec_mask, mode, d, a, b);
  case 1:
    return goc_v_or_b32_wave64(flags, exec_mask, mode, d, a, b);
  case 2:
    return goc_v_xor_b32_wave64(flags, exec_mask, mode, d, a, b);
  case 3:
    return goc_v_not_b32_wave64(flags, exec_mask, mode, d, a);
  case 4:
    return goc_v_and_b16_wave64(flags, exec_mask, mode, d, a, b);
  case 5:
    return goc_v_or_b16_wave64(flags, exec_mask, mode, d, a, b);
  case 6:
    return goc_v_xor_b16_wave64(flags, exec_mask, mode, d, a, b);
  case 7:
    return goc_v_not_b16_wave64(flags, exec_mask, mode, d, a);
  case 8:
    return goc_v_xnor_b32_wave64(flags, exec_mask, mode, d, a, b);
  case 9:
    return goc_v_cndmask_b32_wave64(flags, exec_mask, mode, d, a, b, condition);
  default:
    return goc_v_cndmask_b16_wave64(flags, exec_mask, mode, d, a, b, condition);
  }
}
} // namespace

TEST(BooleanWave64, ModifiersRoutingAliasingAndExactSemantics) {
  std::vector<uint64_t> routes = {0};
  for (auto mode : goc_test::dpp_modes)
    routes.push_back(mode);
  for (unsigned rows = 0; rows < 16; ++rows)
    routes.push_back((goc_test::dpp_modes[2] & ~GOC_DPP_ROW_MASK) |
                     (uint64_t(rows) << GOC_DPP_ROW_SHIFT));
  for (unsigned op = 0; op < 11; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (unsigned m = 0; m < (op == 10 ? 128U : op == 9 ? 16U : 8U); ++m)
          for (auto route : routes)
            for (unsigned alias = 0; alias < 3; ++alias) {
              uint32_t words[3][66], initial[3][66];
              for (unsigned j = 0; j < 3; ++j)
                for (unsigned lane = 0; lane < 66; ++lane)
                  words[j][lane] = initial[j][lane] =
                      (lane * (0x9e3779b9U + j * 123)) ^ (0xa5a59669U >> j);
              unsigned dst = alias == 0 ? 2 : alias - 1;
              uint32_t *d = words[dst] + 1;
              const uint32_t *a = words[0] + 1, *b = words[1] + 1;
              uint64_t mode =
                  route | (op < 9 ? goc_test::boolean_mode(op, m) : goc_test::cndmask_mode(m));
              const uint64_t exec_mask = alias ? UINT64_MAX : 0x87654321fedcba98ULL;
              const uint64_t condition = 0x5a5a5a5aa5a5a5a5ULL;
              ASSERT_EQ(execute(op, cpu | semantics | GOC_SEMANTICS_STRICT, exec_mask, mode, &d, &a,
                                &b, condition),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 64; ++lane) {
                int source = int(lane);
                bool enabled = (exec_mask >> lane) & 1;
                if (route)
                  enabled = goc_test::dpp_source(mode, exec_mask, lane, source);
                uint32_t av = source < 0 ? 0 : initial[0][source + 1];
                uint32_t bv = initial[1][lane + 1], old = initial[dst][lane + 1];
                uint32_t want = old;
                if (enabled)
                  want = op < 9 ? goc_test::boolean_reference(op, av, bv, old, uint32_t(mode))
                                : goc_test::cndmask_reference(op == 10, av, bv, old, m,
                                                              (condition >> lane) & 1);
                ASSERT_EQ(d[lane], want) << op << '/' << m << '/' << lane << '/' << route;
              }
              EXPECT_EQ(words[dst][0], initial[dst][0]);
              EXPECT_EQ(words[dst][65], initial[dst][65]);
            }
}

TEST(BooleanWave64, ZeroExecAndValidation) {
  for (unsigned op = 0; op < 11; ++op) {
    EXPECT_EQ(execute(op, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr,
                      nullptr, nullptr, UINT64_MAX),
              GOC_SUCCESS);
    EXPECT_EQ(execute(op, 0, UINT64_MAX, 1ULL << 63, nullptr, nullptr, nullptr, 0),
              GOC_ERROR_INVALID_FLAGS);
  }
}

TEST(BooleanWave64, Wave32StrictExactResults) {
  for (unsigned op = 0; op < 11; ++op) {
    uint32_t a[32], b[32], d[32];
    for (unsigned lane = 0; lane < 32; ++lane) {
      a[lane] = (1U << lane) ^ 0x7f800001;
      b[lane] = ~a[lane];
      d[lane] = 0xa5a59669;
    }
    const uint32_t *pa = a, *pb = b;
    uint32_t *pd = d;
    uint64_t flags = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
    const uint32_t condition = 0x55555555;
    int status = op < 9    ? goc_test::boolean_functions[op](flags, UINT32_MAX, 0, &pd, &pa, &pb)
                 : op == 9 ? goc_v_cndmask_b32(flags, UINT32_MAX, 0, &pd, &pa, &pb, condition)
                           : goc_v_cndmask_b16(flags, UINT32_MAX, 0, &pd, &pa, &pb, condition);
    ASSERT_EQ(status, GOC_SUCCESS);
    for (unsigned lane = 0; lane < 32; ++lane)
      EXPECT_EQ(d[lane], op < 9
                             ? goc_test::boolean_reference(op, a[lane], b[lane], 0xa5a59669, 0)
                             : goc_test::cndmask_reference(op == 10, a[lane], b[lane], 0xa5a59669,
                                                           0, (condition >> lane) & 1));
  }
}
