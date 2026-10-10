// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "exec_masks.h"
#include "goc/goc.h"
#include "half_binary_reference.h"
#include "half_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {

const decltype(&goc_v_add_f16_rdna4_wave64) wave64[] = {
    goc_v_add_f16_rdna4_wave64, goc_v_sub_f16_rdna4_wave64, goc_v_subrev_f16_rdna4_wave64,
    goc_v_mul_f16_rdna4_wave64, goc_v_min_num_f16_wave64,   goc_v_max_num_f16_wave64,
    goc_v_minimum_f16_wave64,   goc_v_maximum_f16_wave64,   goc_v_min_f16_wave64,
    goc_v_max_f16_wave64};

void check(uint32_t actual, uint32_t before, uint16_t want, uint32_t mode) {
  int shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  EXPECT_EQ((actual ^ before) & ~(0xffffU << shift), 0u);
  uint16_t got = uint16_t(actual >> shift);
  if ((want & 0x7fff) > 0x7c00) {
    EXPECT_GT(got & 0x7fff, 0x7c00);
  } else {
    EXPECT_EQ(got, want);
  }
}

} // namespace

TEST(HalfBinary, CanonicalNanPreservesUnselectedHalf) {
  for (uint32_t bits = 0; bits < 65536; ++bits) {
    const bool nan = (bits & 0x7c00) == 0x7c00 && (bits & 0x3ff) != 0;
    const uint32_t expected = nan ? 0x7e00 : bits;
    // A signaling NaN in the other half must remain untouched.
    EXPECT_EQ(goc_test::canonical_half_nan(0xfc010000U | bits, 0), 0xfc010000U | expected);
    EXPECT_EQ(goc_test::canonical_half_nan((bits << 16) | 0x7c01, GOC_ALU_HIGH_D),
              (expected << 16) | 0x7c01);
  }
}

TEST(HalfBinary, EveryEncodingAndRandomInputs) {
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool saturate : {false, true}) {
        std::mt19937 random(427);
        for (unsigned base = 0; base < 131072; base += 32) {
          SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << saturate << '/' << base);
          uint32_t words[3][32];
          uint32_t *p[] = {words[0], words[1], words[2]};
          for (int lane = 0; lane < 32; ++lane) {
            unsigned i = base + lane;
            words[0][lane] = i < 65536 ? i : random();
            words[1][lane] = random();
            words[2][lane] = 0xdeadbeef;
          }
          ASSERT_EQ(goc_test::half_binary_functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                        UINT32_MAX, 0, p + 2, p, p + 1, nullptr),
                    GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            check(words[2][lane], 0xdeadbeef,
                  goc_test::half_binary_reference(op, words[0][lane], words[1][lane], 0, saturate),
                  0);
        }
      }
}

TEST(HalfBinary, AllModifiersAndHalfSelectors) {
  const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3bff, 0x3c00,
                             0xbc00, 0x4001, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe00};
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 1024; ++variant)
        for (bool saturate : {false, true}) {
          auto mode = goc_test::half_binary_modifiers(variant);
          SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << mode);
          uint32_t words[3][32], before[3][32];
          for (int reg = 0; reg < 3; ++reg)
            for (int lane = 0; lane < 32; ++lane)
              words[reg][lane] = values[(lane + reg * 3) % 16] |
                                 (uint32_t(values[(lane / 2 + reg * 5) % 16]) << 16);
          std::memcpy(before, words, sizeof(words));
          int dest = variant % 3;
          uint32_t exec_mask = UINT32_MAX;
          uint32_t *p[] = {words[0], words[1], words[2]};
          ASSERT_EQ(goc_test::half_binary_functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                        exec_mask, mode, p + dest, p, p + 1,
                                                        nullptr),
                    GOC_SUCCESS);
          for (int reg = 0; reg < 3; ++reg)
            for (int lane = 0; lane < 32; ++lane) {
              if (reg == dest && ((exec_mask >> lane) & 1)) {
                check(words[reg][lane], before[reg][lane],
                      goc_test::half_binary_reference(op, before[0][lane], before[1][lane], mode,
                                                      saturate),
                      mode);
              } else {
                EXPECT_EQ(words[reg][lane], before[reg][lane]);
              }
            }
        }
}

TEST(HalfBinary, MasksAliasesAndUntouchedHalves) {
  const uint32_t arithmetic_modes[] = {0, GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_NEG_B,
                                       GOC_ALU_OMOD_HALF, GOC_ALU_OMOD_4 | GOC_ALU_CLAMP};
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned selectors = 0; selectors < 8; ++selectors)
        for (uint32_t arithmetic : arithmetic_modes)
          for (uint32_t exec_mask : exec_masks())
            for (int b = 0; b < 2; ++b)
              for (int dest = 0; dest < 3; ++dest) {
                uint32_t mode = goc_test::half_binary_modifiers(selectors << 7) | arithmetic;
                SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << mode << '/'
                                                  << exec_mask << '/' << b << '/' << dest);
                uint32_t words[3][34], before[3][34];
                std::mt19937 random(12);
                for (auto &reg : words)
                  for (auto &word : reg)
                    word = random();
                std::memcpy(before, words, sizeof(words));
                uint32_t *p[] = {words[0] + 1, words[1] + 1, words[2] + 1};
                ASSERT_EQ(goc_test::half_binary_functions[op](cpu, exec_mask, mode, p + dest, p,
                                                              p + b, nullptr),
                          GOC_SUCCESS);
                for (int reg = 0; reg < 3; ++reg)
                  for (int lane = 0; lane < 34; ++lane) {
                    if (reg == dest && lane >= 1 && lane <= 32 && ((exec_mask >> (lane - 1)) & 1)) {
                      check(words[reg][lane], before[reg][lane],
                            goc_test::half_binary_reference(op, before[0][lane], before[b][lane],
                                                            mode, false),
                            mode);
                    } else {
                      EXPECT_EQ(words[reg][lane], before[reg][lane]);
                    }
                  }
              }
}

TEST(HalfBinary, LiteralRoundingOverflowAndNanRules) {
  struct Case {
    int op;
    uint16_t a, b, result;
    uint32_t mode;
    bool saturate;
  };

  const Case cases[] = {{0, 0x3c00, 0x1000, 0x3c00, 0, false}, // Even tie rounds down.
                        {0, 0x3c01, 0x1000, 0x3c02, 0, false}, // Odd tie rounds up.
                        {1, 0x0400, 0x03ff, 1, 0, false},
                        {2, 0x3c00, 0x4000, 0x3c00, 0, false},
                        {3, 1, 0x3800, 0, 0, false},
                        {3, 3, 0x3800, 2, 0, false},
                        {3, 0x8001, 0x3800, 0x8000, 0, false},
                        {0, 0x7bff, 0x7bff, 0x7c00, 0, false},
                        {0, 0x7bff, 0x7bff, 0x7bff, 0, true},
                        {0, 0xfbff, 0xfbff, 0xfbff, 0, true},
                        {0, 0x7bff, 0x7bff, 0x7bff, GOC_ALU_OMOD_HALF, false},
                        {3, 0x7c00, 0x3c00, 0x7c00, 0, true},
                        {4, 0x8000, 0, 0x8000, 0, false},
                        {5, 0x8000, 0, 0, 0, false},
                        {6, 0x8000, 0, 0x8000, 0, false},
                        {7, 0x8000, 0, 0, 0, false},
                        {4, 0x7c01, 0x3c00, 0x3c00, 0, false},
                        {5, 0x3c00, 0xfc01, 0x3c00, 0, false},
                        {6, 0x7c01, 0x3c00, 0x7e00, 0, false},
                        {7, 0x3c00, 0x7e00, 0, GOC_ALU_CLAMP, false},
                        {3, 0, 0x7c00, 0, GOC_ALU_CLAMP, false}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases) {
      uint32_t words[3][32];
      uint32_t *p[] = {words[0], words[1], words[2]};
      std::fill(words[0], words[0] + 32, test.a);
      std::fill(words[1], words[1] + 32, test.b);
      std::fill(words[2], words[2] + 32, 0xdeadbeef);
      ASSERT_EQ(goc_test::half_binary_functions[test.op](cpu | (test.saturate ? GOC_FP16_OVFL : 0),
                                                         UINT32_MAX, test.mode, p + 2, p, p + 1,
                                                         nullptr),
                GOC_SUCCESS);
      for (auto word : words[2])
        check(word, 0xdeadbeef, test.result, test.mode);
    }
}

TEST(HalfBinary, ValidationAndSemantics) {
  for (auto fn : goc_test::half_binary_functions) {
    uint32_t data[32];
    std::fill(data, data + 32, 0xdeadbeef);
    auto p = data;
    for (uint32_t exec_mask : {0U, UINT32_MAX}) {
      for (int bit = 0; bit < 32; ++bit)
        if (!(goc_test::half_binary_known & (1U << bit))) {
          EXPECT_EQ(fn(0, exec_mask, 1U << bit, &p, &p, &p, nullptr), GOC_ERROR_INVALID_FLAGS);
        }
      EXPECT_EQ(fn(1ULL << 63, exec_mask, 0, &p, &p, &p, nullptr), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, exec_mask, 0, &p, &p, &p,
                   nullptr),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    for (auto word : data)
      EXPECT_EQ(word, 0xdeadbeef);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &p, &p, &p, nullptr), GOC_SUCCESS);
  }
}

TEST(HalfBinary, Wave64ModifiersSelectorsDppAndAliases) {
  const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3bff, 0x3c00,
                             0xbc00, 0x4001, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe00};
  for (unsigned op = 0; op < 10; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool saturate : {false, true})
        for (unsigned variant = 0; variant < 1024; ++variant)
          for (unsigned route_index = 0; route_index < 8; ++route_index) {
            uint32_t mode = goc_test::half_binary_modifiers(variant);
            uint64_t descriptor = route_index ? goc_test::dpp_modes[route_index - 1] : 0;
            if (descriptor & GOC_DPP16)
              descriptor = (descriptor & ~GOC_DPP_ROW_MASK) |
                           (uint64_t((variant >> 6) & 15) << GOC_DPP_ROW_SHIFT);
            uint32_t words[3][66], before[3][66];
            for (unsigned reg = 0; reg < 3; ++reg)
              for (unsigned lane = 0; lane < 66; ++lane)
                words[reg][lane] = values[(lane + 5 * reg) % 16] |
                                   (uint32_t(values[(lane * 7 + 3 * reg) % 16]) << 16);
            std::memcpy(before, words, sizeof words);
            unsigned dest = variant % 3;
            uint32_t *d = words[dest] + 1;
            const uint32_t *a = words[0] + 1, *b = words[1] + 1;
            uint64_t exec_mask = 0x87654321fedcba98ULL;
            uint32_t exceptions = 0x80000055;
            ASSERT_EQ(wave64[op](cpu | (saturate ? GOC_FP16_OVFL : 0), exec_mask, descriptor | mode,
                                 &d, &a, &b, &exceptions),
                      GOC_SUCCESS);
            EXPECT_EQ(exceptions, 0x80000055U);
            for (unsigned lane = 0; lane < 64; ++lane) {
              int source = int(lane);
              bool enabled = descriptor ? goc_test::dpp_source(descriptor, exec_mask, lane, source)
                                        : ((exec_mask >> lane) & 1);
              if (enabled)
                check(d[lane], before[dest][lane + 1],
                      goc_test::half_binary_reference(
                          op >= 8 ? op - 4 : op, source < 0 ? 0 : before[0][source + 1],
                          before[1][lane + 1], op >= 8 ? mode & ~GOC_ALU_OMOD_HALF : mode,
                          saturate),
                      mode);
              else
                EXPECT_EQ(d[lane], before[dest][lane + 1]);
            }
            for (unsigned reg = 0; reg < 3; ++reg) {
              EXPECT_EQ(words[reg][0], before[reg][0]);
              EXPECT_EQ(words[reg][65], before[reg][65]);
            }
          }
}

TEST(HalfBinary, Wave64ValidationAndOptionalState) {
  for (auto fn : wave64) {
    EXPECT_EQ(fn(0, 0, 0, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(fn(0, UINT64_MAX, 1ULL << 36, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT64_MAX, 0, nullptr,
                 nullptr, nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    uint32_t exceptions = 0x80000055;
    EXPECT_EQ(
        fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT64_MAX, 0, nullptr, nullptr, nullptr, &exceptions),
        GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
    EXPECT_EQ(exceptions, 0x80000055U);
  }
}

TEST(HalfBinary, ArchitectureScalingDenormalsSelectorsDppAndAliases) {
  const goc_test::HalfBinaryFn rdna3[] = {goc_v_add_f16, goc_v_sub_f16, goc_v_subrev_f16,
                                          goc_v_mul_f16, goc_v_min_f16, goc_v_max_f16};
  const decltype(&goc_v_add_f16_wave64) rdna3_wide[] = {
      goc_v_add_f16_wave64, goc_v_sub_f16_wave64, goc_v_subrev_f16_wave64,
      goc_v_mul_f16_wave64, goc_v_min_f16_wave64, goc_v_max_f16_wave64};
  const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x03ff, 0x83ff, 0x0400,
                             0x8400, 0x3800, 0x3c00, 0xc000, 0x7bff, 0x7c00, 0x7e01};
  const uint64_t routes[] = {0, goc_test::dpp_modes[0], goc_test::dpp_modes[6]};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned arch = 0; arch < 2; ++arch)
      for (unsigned op = 0; op < 6; ++op)
        for (unsigned settings = 0; settings < 16; ++settings)
          for (unsigned selector = 0; selector < 8; ++selector)
            for (unsigned scale = 0; scale < 4; ++scale)
              for (uint64_t route : routes)
                for (unsigned width : {32U, 64U})
                  for (unsigned alias = 0; alias < 3; ++alias) {
                    uint64_t flags = cpu | (settings & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                                     (settings & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                                     (settings & 4 ? GOC_FP_IEEE_MODE : 0) |
                                     (settings & 8 ? GOC_FP16_OVFL : 0);
                    uint32_t mode = (scale << 6) | GOC_ALU_ABS_A | GOC_ALU_NEG_B |
                                    (selector & 1 ? GOC_ALU_HIGH_A : 0) |
                                    (selector & 2 ? GOC_ALU_HIGH_B : 0) |
                                    (selector & 4 ? GOC_ALU_HIGH_D : 0);
                    uint32_t words[3][66], before[3][66];
                    for (unsigned reg = 0; reg < 3; ++reg)
                      for (unsigned lane = 0; lane < 66; ++lane)
                        words[reg][lane] = uint32_t(values[(lane + 3 * reg) % 14]) |
                                           (uint32_t(values[(lane * 3 + 5 * reg + 1) % 14]) << 16);
                    std::memcpy(before, words, sizeof words);
                    uint32_t *d = words[alias] + 1;
                    const uint32_t *a = words[0] + 1, *b = words[1] + 1;
                    uint64_t exec_mask = width == 32 ? 0xfedcba98ULL : 0x87654321fedcba98ULL;
                    uint32_t state = 0x12345678;
                    int status =
                        width == 32
                            ? (arch ? goc_test::half_binary_functions[op] : rdna3[op])(
                                  flags, uint32_t(exec_mask), mode | route, &d, &a, &b, &state)
                            : (arch ? wave64[op] : rdna3_wide[op])(flags, exec_mask, mode | route,
                                                                   &d, &a, &b, &state);
                    ASSERT_EQ(status, GOC_SUCCESS);
                    EXPECT_EQ(state, 0x12345678U);
                    uint32_t effective_mode = mode;
                    if (!arch && (!(settings & 2) || (settings & 4)))
                      effective_mode &= ~GOC_ALU_OMOD_HALF;
                    for (unsigned lane = 0; lane < width; ++lane) {
                      int source = int(lane);
                      bool enabled = route ? goc_test::dpp_source(route, exec_mask, lane, source)
                                           : ((exec_mask >> lane) & 1);
                      if (!enabled) {
                        EXPECT_EQ(d[lane], before[alias][lane + 1]);
                        continue;
                      }
                      uint32_t x = source < 0 ? 0 : before[0][source + 1];
                      uint32_t y = before[1][lane + 1];
                      if (settings & 1) {
                        // Flush each raw half before the independent reference chooses one.
                        auto flush = [](uint32_t word) {
                          uint32_t lo = word & 0xffff, hi = word >> 16;
                          if ((lo & 0x7fff) < 0x400)
                            lo &= 0x8000;
                          if ((hi & 0x7fff) < 0x400)
                            hi &= 0x8000;
                          return lo | (hi << 16);
                        };
                        x = flush(x);
                        y = flush(y);
                      }
                      uint16_t expected =
                          goc_test::half_binary_reference(op, x, y, effective_mode, settings & 8);
                      if (op < 4 && (settings & 2) && (expected & 0x7fff) < 0x400)
                        expected &= 0x8000;
                      check(d[lane], before[alias][lane + 1], expected, mode);
                    }
                    for (unsigned reg = 0; reg < 3; ++reg) {
                      EXPECT_EQ(words[reg][0], before[reg][0]);
                      EXPECT_EQ(words[reg][width + 1], before[reg][width + 1]);
                    }
                  }
}

TEST(HalfBinary, Rdna3GuestFlagsDoNotEnableExactReporting) {
  const goc_test::HalfBinaryFn functions[] = {goc_v_add_f16, goc_v_sub_f16, goc_v_subrev_f16,
                                              goc_v_mul_f16};
  uint64_t flags = GOC_FP_IEEE_MODE | GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS |
                   GOC_SEMANTICS_EXACT_EMPIRICAL;
  for (auto fn : functions) {
    uint32_t state = 0x12345678;
    EXPECT_EQ(fn(flags, UINT32_MAX, 0, nullptr, nullptr, nullptr, &state),
              GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
    EXPECT_EQ(state, 0x12345678U);
    EXPECT_EQ(fn(flags | GOC_SEMANTICS_STRICT, UINT32_MAX, 0, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(fn(flags, 0, 0, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
  }
}

TEST(HalfBinary, ArithmeticOptionalStateAndExactFallback) {
  const goc_test::HalfBinaryFn narrow[][4] = {
      {goc_v_add_f16, goc_v_sub_f16, goc_v_subrev_f16, goc_v_mul_f16},
      {goc_v_add_f16_rdna4, goc_v_sub_f16_rdna4, goc_v_subrev_f16_rdna4, goc_v_mul_f16_rdna4}};
  const decltype(&goc_v_add_f16_wave64) wide[][4] = {
      {goc_v_add_f16_wave64, goc_v_sub_f16_wave64, goc_v_subrev_f16_wave64, goc_v_mul_f16_wave64},
      {goc_v_add_f16_rdna4_wave64, goc_v_sub_f16_rdna4_wave64, goc_v_subrev_f16_rdna4_wave64,
       goc_v_mul_f16_rdna4_wave64}};
  // Exact integer-valued inputs keep this contract test independent of rounding.
  const uint16_t results[] = {0x4500, 0xbc00, 0x3c00, 0x4600}; // 2+3, 2-3, 3-2, 2*3.
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned arch = 0; arch < 2; ++arch)
      for (unsigned op = 0; op < 4; ++op)
        for (unsigned width : {32U, 64U})
          for (uint64_t exec_mask : {0ULL, 0xa55aa55a5aa55aa5ULL, ~0ULL}) {
            SCOPED_TRACE(::testing::Message()
                         << cpu << '/' << arch << '/' << op << '/' << width << '/' << exec_mask);
            uint32_t a[64], b[64], d[64];
            std::fill(a, a + 64, 0x4000);
            std::fill(b, b + 64, 0x4200);
            const uint32_t *pa = a, *pb = b;
            uint32_t *pd = d;
            uint32_t state = 0x81234567;
            auto call = [&](uint64_t flags, uint64_t mode, uint32_t *output) {
              return width == 32
                         ? narrow[arch][op](flags, uint32_t(exec_mask), mode, &pd, &pa, &pb, output)
                         : wide[arch][op](flags, exec_mask, mode, &pd, &pa, &pb, output);
            };
            for (bool strict : {false, true}) {
              std::fill(d, d + 64, 0xdeadbeef);
              uint64_t flags =
                  cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | (strict ? GOC_SEMANTICS_STRICT : 0);
              // Reporting errors take precedence even over invalid modifiers and empty EXEC.
              for (uint64_t mode : {0ULL, 1ULL << 63}) {
                EXPECT_EQ(call(flags, mode, &state), GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
                EXPECT_EQ(state, 0x81234567U);
                for (auto word : d)
                  EXPECT_EQ(word, 0xdeadbeefU);
              }
              EXPECT_EQ(call(flags, 0, nullptr),
                        strict ? GOC_ERROR_UNSUPPORTED_SEMANTICS : GOC_SUCCESS);
              for (unsigned lane = 0; lane < 64; ++lane)
                EXPECT_EQ(d[lane], !strict && lane < width && ((exec_mask >> lane) & 1)
                                       ? 0xdead0000U | results[op]
                                       : 0xdeadbeefU);
            }
            std::fill(d, d + 64, 0xdeadbeef);
            EXPECT_EQ(call(cpu, 0, &state), GOC_SUCCESS);
            EXPECT_EQ(state, 0x81234567U);
            for (unsigned lane = 0; lane < 64; ++lane)
              EXPECT_EQ(d[lane], lane < width && ((exec_mask >> lane) & 1)
                                     ? 0xdead0000U | results[op]
                                     : 0xdeadbeefU);
          }
}

TEST(HalfBinary, MinmaxRawNanBitsGuestModesAndPackedSelectors) {
  // A/B, then selected source for legacy IEEE=0, legacy IEEE=1, RDNA4 number,
  // and RDNA4 propagating forms. 0/1 select A/B; 2/3 quiet the selected A/B NaN.
  struct Case {
    uint16_t a, b;
    unsigned selected[8];
  };

  const Case cases[] = {
      {0x7c11, 0x3c00, {1, 1, 2, 2, 1, 1, 2, 2}}, {0xbc00, 0xfc23, {0, 0, 3, 3, 0, 0, 3, 3}},
      {0x7e11, 0xfc23, {0, 0, 3, 3, 0, 0, 3, 3}}, {0x7c11, 0xfc23, {0, 0, 2, 2, 2, 2, 2, 2}},
      {0x7c11, 0xfe23, {0, 0, 2, 2, 2, 2, 2, 2}}, {0x7e11, 0xfe23, {0, 0, 0, 0, 0, 0, 0, 0}},
      {0x7e11, 0x3c00, {1, 1, 1, 1, 1, 1, 0, 0}}, {0x8000, 0, {0, 1, 0, 1, 0, 1, 0, 1}},
      {0x8001, 1, {0, 1, 0, 1, 0, 1, 0, 1}},
  };
  const goc_test::HalfBinaryFn narrow[][6] = {
      {goc_v_min_f16, goc_v_max_f16, goc_v_min_num_f16, goc_v_max_num_f16, goc_v_minimum_f16,
       goc_v_maximum_f16},
      {goc_v_pk_min_f16, goc_v_pk_max_f16, goc_v_pk_min_num_f16, goc_v_pk_max_num_f16,
       goc_v_pk_minimum_f16, goc_v_pk_maximum_f16}};
  const decltype(&goc_v_min_f16_wave64) wide[][6] = {
      {goc_v_min_f16_wave64, goc_v_max_f16_wave64, goc_v_min_num_f16_wave64,
       goc_v_max_num_f16_wave64, goc_v_minimum_f16_wave64, goc_v_maximum_f16_wave64},
      {goc_v_pk_min_f16_wave64, goc_v_pk_max_f16_wave64, goc_v_pk_min_num_f16_wave64,
       goc_v_pk_max_num_f16_wave64, goc_v_pk_minimum_f16_wave64, goc_v_pk_maximum_f16_wave64}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned packed = 0; packed < 2; ++packed)
      for (unsigned op = 0; op < 6; ++op)
        for (unsigned settings = 0; settings < 8; ++settings)
          for (unsigned selector = 0; selector < 8; ++selector)
            for (unsigned width : {32U, 64U})
              for (unsigned alias = 0; alias < 3; ++alias) {
                uint32_t words[3][64], before[3][64];
                // Each input's two halves use the same case but independently
                // negated payloads. Selectors therefore expose sign/half mistakes.
                for (unsigned lane = 0; lane < 64; ++lane) {
                  const auto &test = cases[lane % 9];
                  words[0][lane] = test.a | (uint32_t(test.a ^ 0x8000) << 16);
                  words[1][lane] = test.b | (uint32_t(test.b ^ 0x8000) << 16);
                  words[2][lane] = 0xdeadbeef;
                }
                // Select both inputs from the same half so the literal order table
                // remains valid; undo the upper-half sign with NEG when selected.
                bool upper = selector & 1;
                uint32_t mode =
                    packed
                        ? (upper ? GOC_PK_LO_A_HIGH | GOC_PK_LO_B_HIGH | GOC_PK_NEG_LO_A |
                                       GOC_PK_NEG_LO_B
                                 : 0) |
                              (selector & 2 ? GOC_PK_HI_A_LOW | GOC_PK_HI_B_LOW
                                            : GOC_PK_NEG_HI_A | GOC_PK_NEG_HI_B) |
                              (selector & 4 ? GOC_PK_CLAMP : 0)
                        : (upper ? GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_NEG_A | GOC_ALU_NEG_B
                                 : 0) |
                              (selector & 2 ? GOC_ALU_HIGH_D : 0) |
                              (selector & 4 ? GOC_ALU_CLAMP : 0);
                std::memcpy(before, words, sizeof words);
                uint32_t *d = words[alias];
                const uint32_t *a = words[0], *b = words[1];
                uint64_t flags = cpu | (settings & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                                 (settings & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                                 (settings & 4 ? GOC_FP_IEEE_MODE : 0);
                uint64_t exec_mask = 0x87654321fedcba98ULL;
                uint32_t state = 0x12345678;
                ASSERT_EQ(
                    width == 32
                        ? narrow[packed][op](flags, uint32_t(exec_mask), mode, &d, &a, &b, &state)
                        : wide[packed][op](flags, exec_mask, mode, &d, &a, &b, &state),
                    GOC_SUCCESS);
                EXPECT_EQ(state, 0x12345678U);
                unsigned column = (op >= 4 ? 6 : op >= 2 ? 4 : settings & 4 ? 2 : 0) + (op & 1);
                for (unsigned lane = 0; lane < width; ++lane) {
                  const auto &test = cases[lane % 9];
                  unsigned choice = test.selected[column];
                  uint16_t expected = choice & 1 ? test.b : test.a;
                  if (choice >= 2)
                    expected |= 0x0200;
                  if ((settings & 1) && lane % 9 == 8)
                    expected &= 0x8000;
                  if (selector & 4) {
                    if ((expected & 0x8000) || (expected & 0x7fff) > 0x7c00)
                      expected = 0;
                    else if (expected > 0x3c00)
                      expected = 0x3c00;
                  }
                  unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
                  uint32_t want = packed ? uint32_t(expected) * 0x10001U
                                         : (before[alias][lane] & ~(0xffffU << shift)) |
                                               (uint32_t(expected) << shift);
                  if (!((exec_mask >> lane) & 1))
                    want = before[alias][lane];
                  EXPECT_EQ(d[lane], want) << cpu << '/' << packed << '/' << op << '/' << settings
                                           << '/' << selector << '/' << lane;
                }
              }
}

TEST(HalfBinary, LegacySelectionPreservesEveryHalfEncoding) {
  // With IEEE=0 and B a quiet NaN, MIN/MAX must select A unchanged. This also
  // checks every signaling-NaN sign/payload across F16C promotion and narrowing.
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (auto fn : {goc_v_min_f16, goc_v_max_f16})
      for (unsigned base = 0; base < 65536; base += 32) {
        uint32_t a[32], b[32], d[32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          a[lane] = base + lane;
          b[lane] = 0x7e01;
          d[lane] = 0x7c110000;
        }
        uint32_t *pd = d;
        const uint32_t *pa = a, *pb = b;
        ASSERT_EQ(fn(cpu, UINT32_MAX, 0, &pd, &pa, &pb, nullptr), GOC_SUCCESS);
        for (unsigned lane = 0; lane < 32; ++lane)
          EXPECT_EQ(d[lane], 0x7c110000U | (base + lane)) << cpu;
      }
}
