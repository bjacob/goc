// SPDX-License-Identifier: MIT

#include "compare_exceptions_hardware.h"
#include "dpp_reference.h"
#include "float_compare_reference.h"
#include "goc/goc.h"

#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

TEST(FloatCompareWave64, PredicatesModifiersDppAndAliases) {
  for (unsigned op = 0; op < 84; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned flush = 0; flush < 2; ++flush) {
        std::vector<uint64_t> routing = {0};
        if (op < 56) {
          for (auto mode : goc_test::dpp_modes)
            routing.push_back(mode);
          for (unsigned rows = 0; rows < 16; ++rows)
            routing.push_back((goc_test::dpp_modes[2] & ~GOC_DPP_ROW_MASK) |
                              (uint64_t(rows) << GOC_DPP_ROW_SHIFT));
        }
        for (auto route : routing)
          for (unsigned m = 0; m < (op < 28 ? 64U : 16U); ++m) {
            uint32_t words[4][64];
            for (unsigned lane = 0; lane < 64; ++lane) {
              uint32_t w[4];
              goc_test::float_compare_inputs(op / 28, (lane + m * 64) % 4096, w);
              for (unsigned j = 0; j < 4; ++j)
                words[j][lane] = w[j];
            }
            const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]};
            const uint64_t exec_mask = m & 1 ? 0x87654321fedcba98ULL : UINT64_MAX;
            uint64_t expected = 0;
            for (unsigned lane = 0; lane < 64; ++lane) {
              int source = int(lane);
              bool enabled = (exec_mask >> lane) & 1;
              if (route)
                enabled = goc_test::dpp_source(route, exec_mask, lane, source);
              uint32_t w[] = {source < 0 ? 0 : words[0][source], source < 0 ? 0 : words[1][source],
                              words[2][lane], words[3][lane]};
              if (enabled && goc_test::float_compare_reference(op, m, flush, w))
                expected |= 1ULL << lane;
            }
            uint64_t flags = cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                             (flush ? GOC_FP_FLUSH_INPUT_DENORMALS : 0);
            uint64_t result = 123;
            auto fn = goc_test::float_compare_wave64_functions[op];
            ASSERT_EQ(fn(flags, exec_mask, route | goc_test::float_compare_mode(m), &result, a, b,
                         nullptr),
                      GOC_SUCCESS);
            ASSERT_EQ(result, expected) << op << '/' << m << '/' << route;
            // A result in the upper source half must not overwrite unread input.
            ASSERT_EQ(fn(flags, exec_mask, route | goc_test::float_compare_mode(m),
                         reinterpret_cast<uint64_t *>(words[0] + 33), a, b, nullptr),
                      GOC_SUCCESS);
            std::memcpy(&result, words[0] + 33, sizeof result);
            ASSERT_EQ(result, expected);
          }
      }
}

TEST(FloatCompareWave64, HardwareExceptionPairsInUpperAndLowerLanes) {
  for (unsigned op = 0; op < 84; ++op)
    for (unsigned signaling = 0; signaling < 2; ++signaling)
      for (unsigned flush = 0; flush < 2; ++flush)
        for (unsigned pair = 0; pair < 144; ++pair) {
          uint64_t av = goc_test::compare_exception_inputs[op / 28][pair / 12];
          uint64_t bv = goc_test::compare_exception_inputs[op / 28][pair % 12];
          uint32_t words[4][64] = {};
          for (unsigned lane : {0U, 31U, 32U, 63U}) {
            words[0][lane] = uint32_t(av);
            words[1][lane] = uint32_t(av >> 32);
            words[2][lane] = uint32_t(bv);
            words[3][lane] = uint32_t(bv >> 32);
          }
          const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]};
          for (uint64_t exec_mask :
               {uint64_t(0), uint64_t(1), uint64_t(1) << 32, uint64_t(1) << 63, UINT64_MAX}) {
            SCOPED_TRACE(::testing::Message()
                         << "op=" << op << " pair=" << pair << " signaling=" << signaling
                         << " flush=" << flush << " exec=" << exec_mask);
            uint64_t d = 123;
            uint32_t state = 0x80000000;
            uint64_t flags = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                             (flush ? GOC_FP_FLUSH_INPUT_DENORMALS : 0);
            auto fn = goc_test::float_compare_wave64_functions[op];
            ASSERT_EQ(fn(flags, exec_mask, signaling ? GOC_ALU_CLAMP : 0, &d, a, b, &state),
                      GOC_SUCCESS);
            uint32_t expected =
                goc_test::compare_exception_hardware[(!flush) + 2 * signaling][pair];
            ASSERT_EQ(state, 0x80000000U | (exec_mask ? expected : 0)) << op << '/' << pair;
            state = 0x80000000;
            ASSERT_EQ(fn(goc_init_cpu_flags(), exec_mask, 0, &d, a, b, &state), GOC_SUCCESS);
            EXPECT_EQ(state, 0x80000000U);
          }
        }
}

TEST(FloatCompareWave64, InvalidFlagsAndZeroExec) {
  for (auto fn : goc_test::float_compare_wave64_functions) {
    uint64_t d = 123;
    uint32_t state = 0x80000000;
    EXPECT_EQ(fn(0, UINT64_MAX, 1ULL << 63, &d, nullptr, nullptr, &state), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(d, 123U);
    EXPECT_EQ(state, 0x80000000U);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, &d, nullptr, nullptr, &state), GOC_SUCCESS);
    EXPECT_EQ(d, 0U);
    EXPECT_EQ(state, 0x80000000U);
  }
}
