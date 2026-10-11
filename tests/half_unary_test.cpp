// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "exec_masks.h"
#include "fuzzy.h"
#include "goc/goc.h"
#include "half_reference.h"
#include "half_unary_reference.h"
#include "loose_tolerances.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

namespace {
const decltype(&goc_v_trunc_f16_rdna4_wave64) wave64[] = {
    goc_v_trunc_f16_rdna4_wave64, goc_v_ceil_f16_rdna4_wave64,      goc_v_rndne_f16_rdna4_wave64,
    goc_v_floor_f16_rdna4_wave64, goc_v_sqrt_f16_rdna4_wave64,      goc_v_rcp_f16_rdna4_wave64,
    goc_v_rsq_f16_rdna4_wave64,   goc_v_exp_f16_rdna4_wave64,       goc_v_log_f16_rdna4_wave64,
    goc_v_fract_f16_rdna4_wave64, goc_v_frexp_mant_f16_rdna4_wave64};

::testing::AssertionResult check(int op, uint32_t actual, uint32_t before, uint16_t want,
                                 uint32_t mode) {
  int shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  if ((actual ^ before) & ~(0xffffU << shift))
    return ::testing::AssertionFailure()
           << "unselected half changed: before=" << before << ", actual=" << actual;
  uint16_t got = uint16_t(actual >> shift);
  if ((want & 0x7fff) > 0x7c00) {
    if ((got & 0x7fff) > 0x7c00)
      return ::testing::AssertionSuccess();
  } else if (op >= 4 && op <= 8 && (want & 0x7fff) != 0 && (want & 0x7fff) < 0x7c00 &&
             (got & 0x8000) == (want & 0x8000)) {
    return goc_test::fuzzy(goc_test::half_value(got), goc_test::half_value(want),
                           goc_test::half_sfu_tolerance);
  } else if (got == want) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << "actual bits=" << got << ", expected bits=" << want;
}

} // namespace

TEST(HalfUnary, EveryEncodingAndCpuLevel) {
  for (int op = 0; op < 11; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool saturate : {false, true})
        for (uint32_t mode : {0U, GOC_ALU_HIGH_A | GOC_ALU_HIGH_D | GOC_ALU_OMOD_4})
          for (unsigned base = 0; base < 65536; base += 32) {
            SCOPED_TRACE(::testing::Message()
                         << op << '/' << cpu << '/' << saturate << '/' << mode << '/' << base);
            uint32_t input[32], output[32];
            for (int lane = 0; lane < 32; ++lane) {
              auto bits = base + lane;
              input[lane] = mode ? (bits << 16) | 0xbeef : 0xdead0000 | bits;
              output[lane] = 0xdeadbeef;
            }
            auto a = input, d = output;
            ASSERT_EQ(goc_test::half_unary_functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                         UINT32_MAX, mode, &d, &a, nullptr),
                      GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane)
              EXPECT_TRUE(check(op, output[lane], 0xdeadbeef,
                                goc_test::half_unary_reference(op, input[lane], mode, saturate),
                                mode))
                  << "lane=" << lane;
          }
}

TEST(HalfUnary, AllModifiersMasksAndAliases) {
  const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3800, 0xb800,
                             0x3bff, 0x3c00, 0x3e00, 0xbe00, 0x4100, 0xc100, 0x4200, 0x4400,
                             0x4c00, 0xcc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe00};
  for (int op = 0; op < 11; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 128; ++variant)
        for (bool saturate : {false, true})
          for (uint32_t exec_mask : exec_masks())
            for (bool alias : {false, true}) {
              auto mode = goc_test::half_unary_modifiers(variant);
              SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << mode << '/'
                                                << saturate << '/' << exec_mask << '/' << alias);
              uint32_t words[2][34], before[2][34];
              for (int reg = 0; reg < 2; ++reg) {
                std::fill(words[reg], words[reg] + 34, 0xdeadbeef);
                for (int lane = 1; lane <= 32; ++lane)
                  words[reg][lane] = values[(lane + reg * 5) % 24] |
                                     (uint32_t(values[(lane * 7 + reg) % 24]) << 16);
              }
              std::memcpy(before, words, sizeof(words));
              auto a = words[0] + 1, d = words[alias ? 0 : 1] + 1;
              ASSERT_EQ(goc_test::half_unary_functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                           exec_mask, mode, &d, &a, nullptr),
                        GOC_SUCCESS);
              for (int reg = 0; reg < 2; ++reg)
                for (int lane = 0; lane < 34; ++lane) {
                  if (reg == (alias ? 0 : 1) && lane >= 1 && lane <= 32 &&
                      ((exec_mask >> (lane - 1)) & 1)) {
                    EXPECT_TRUE(check(
                        op, words[reg][lane], before[reg][lane],
                        goc_test::half_unary_reference(op, before[0][lane], mode, saturate), mode))
                        << "lane=" << lane;
                  } else {
                    EXPECT_EQ(words[reg][lane], before[reg][lane]);
                  }
                }
            }
}

TEST(HalfUnary, LiteralBoundaries) {
  struct Case {
    int op;
    uint16_t input, result;
    uint32_t mode;
    bool saturate;
  };

  const Case cases[] = {{0, 0xbe00, 0xbc00, 0, false},
                        {1, 0xb800, 0x8000, 0, false},
                        {2, 0x3800, 0, 0, false},
                        {2, 0xb800, 0x8000, 0, false},
                        {2, 0x3e00, 0x4000, 0, false},
                        {2, 0x4100, 0x4000, 0, false},
                        {3, 0x8001, 0xbc00, 0, false},
                        {4, 0x4400, 0x4000, 0, false},
                        {5, 0, 0x7bff, 0, true},
                        {5, 1, 0x7bff, 0, true},
                        {5, 1, 0x7c00, 0, false},
                        {5, 0x8000, 0xfbff, 0, true},
                        {6, 0, 0x7bff, 0, true},
                        {6, 0x8000, 0xfbff, 0, true},
                        {6, 0x4400, 0x3800, 0, false},
                        {7, 0x4c00, 0x7c00, 0, false},
                        {7, 0x4c00, 0x7bff, 0, true},
                        {7, 0x7bff, 0x7bff, 0, true},
                        {7, 0x7c00, 0x7c00, 0, true},
                        {7, 0x4c00, 0x7c00, GOC_ALU_OMOD_HALF, false},
                        {7, 0x4c00, 0x77ff, GOC_ALU_OMOD_HALF, true},
                        {8, 0, 0xfbff, 0, true},
                        {8, 0x8000, 0xfbff, 0, true},
                        {8, 0, 0xf7ff, GOC_ALU_OMOD_HALF, true},
                        {7, 0xce00, 1, 0, false},
                        {7, 0xce40, 0, 0, false},
                        {8, 0, 0xfc00, 0, false},
                        {8, 0x4400, 0x4000, 0, false},
                        {9, 0x8001, 0x3bff, 0, false},
                        {9, 0x8000, 0, 0, false},
                        {9, 0xbd00, 0x3a00, 0, false},
                        {10, 1, 0x3800, 0, false},
                        {10, 0x3ff, 0x3bfe, 0, false},
                        {10, 0x8000, 0x8000, 0, false},
                        {10, 0xfc00, 0xfc00, 0, false}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (const auto &test : cases) {
      SCOPED_TRACE(::testing::Message() << test.op << '/' << cpu << '/' << test.input);
      uint32_t input[32], output[32];
      std::fill(input, input + 32, test.input);
      std::fill(output, output + 32, 0xdeadbeef);
      auto a = input, d = output;
      ASSERT_EQ(goc_test::half_unary_functions[test.op](cpu | (test.saturate ? GOC_FP16_OVFL : 0),
                                                        UINT32_MAX, test.mode, &d, &a, nullptr),
                GOC_SUCCESS);
      for (auto word : output)
        EXPECT_EQ(word, 0xdead0000U | test.result);
    }
}

TEST(HalfUnary, ValidationAndSemantics) {
  for (unsigned op = 0; op < 11; ++op) {
    auto fn = goc_test::half_unary_functions[op];
    uint32_t data[32];
    std::fill(data, data + 32, 0xdeadbeef);
    auto p = data;
    for (uint32_t exec_mask : {0U, UINT32_MAX}) {
      for (int bit = 0; bit < 32; ++bit) {
        if (!(goc_test::half_unary_known & (1U << bit))) {
          EXPECT_EQ(fn(0, exec_mask, 1U << bit, &p, &p, nullptr), GOC_ERROR_INVALID_FLAGS);
        }
      }
      EXPECT_EQ(fn(1ULL << 63, exec_mask, 0, &p, &p, nullptr), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(
          fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, exec_mask, 0, &p, &p, nullptr),
          op < 7 || op == 10 ? GOC_SUCCESS : GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    if (op == 7 || op == 8) {
      for (auto word : data)
        EXPECT_EQ(word, 0xdeadbeef);
    }
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &p, &p, nullptr), GOC_SUCCESS);
  }
}

TEST(HalfUnary, ExpLogEveryEncodingAndOutputModifier) {
  for (unsigned op : {7u, 8u})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned omod = 0; omod < 4; ++omod)
        for (bool clamp : {false, true})
          for (bool saturate : {false, true})
            for (unsigned base = 0; base < 65536; base += 32) {
              uint32_t mode =
                  GOC_ALU_HIGH_A | GOC_ALU_HIGH_D | (omod << 6) | (clamp ? GOC_ALU_CLAMP : 0);
              SCOPED_TRACE(::testing::Message()
                           << op << '/' << cpu << '/' << mode << '/' << saturate << '/' << base);
              uint32_t a[32], d[32];
              for (unsigned lane = 0; lane < 32; ++lane) {
                a[lane] = ((base + lane) << 16) | 0xdead;
                d[lane] = 0xfacecafe;
              }
              auto pa = a, pd = d;
              ASSERT_EQ(goc_test::half_unary_functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                           UINT32_MAX, mode, &pd, &pa, nullptr),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 32; ++lane)
                EXPECT_TRUE(check(op, d[lane], 0xfacecafe,
                                  goc_test::half_unary_reference(op, a[lane], mode, saturate),
                                  mode))
                    << "lane=" << lane;
            }
}

TEST(HalfUnary, HardwareExpLogRoundingAndOverflow) {
  const uint32_t values[] = {0x4c004bff, 0x00008000, 0x7c00fc00, 0x7bfffbff,
                             0xce40ce00, 0x00018001, 0x3bff3c01, 0x40003c00};
  const uint32_t modes[] = {0, 1, 8, 9, 64, 128, 192, 256, 512, 4096, 4608, 5065, 576, 640, 704};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool saturate : {false, true}) {
      uint64_t hash = goc_test::capture_hash_seed;
      for (uint32_t exec_mask :
           {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
        for (unsigned op : {7u, 8u})
          for (uint64_t descriptor : goc_test::dpp_modes)
            for (uint32_t mode : modes) {
              uint32_t a[32], d[32];
              for (unsigned lane = 0; lane < 32; ++lane) {
                a[lane] = values[lane % 8];
                d[lane] = 0xdead0000u + lane;
              }
              auto pa = a, pd = d;
              ASSERT_EQ(goc_test::half_unary_functions[op](cpu | (saturate ? GOC_FP16_OVFL : 0),
                                                           exec_mask, descriptor | mode, &pd, &pa,
                                                           nullptr),
                        GOC_SUCCESS);
              unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
              for (unsigned lane = 0; lane < 32; ++lane) {
                int source;
                uint32_t want = 0xdead0000u + lane;
                bool written = goc_test::dpp_source(descriptor | mode, exec_mask, lane, source);
                if (written) {
                  uint32_t value = goc_test::half_unary_reference(op, source < 0 ? 0 : a[source],
                                                                  mode, saturate);
                  want = (want & ~(65535U << shift)) | (value << shift);
                }
                if (written) {
                  EXPECT_TRUE(check(op, d[lane], 0xdead0000u + lane, uint16_t(want >> shift), mode))
                      << op << "/" << cpu << "/" << mode << "/" << descriptor << "/" << exec_mask
                      << "/" << lane;
                } else {
                  EXPECT_EQ(d[lane], want) << "inactive lane=" << lane;
                }
                uint32_t word = d[lane];
                if (((word >> shift) & 0x7fff) > 0x7c00)
                  word = (word & ~(65535U << shift)) | (0x7e00U << shift);
                hash = goc_test::capture_hash_word(hash, word);
              }
            }
      EXPECT_EQ(hash, saturate ? 0x7e3c62853aa82d6cULL : 0xdeda4067db286486ULL);
    }
}

TEST(HalfUnary, Wave64ModifiersDppSelectorsAndAliases) {
  const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3bff, 0x3c00,
                             0xbc00, 0x4001, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe00};
  for (unsigned op = 0; op < 11; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 128; ++variant)
        for (unsigned route_index = 0; route_index < 8; ++route_index)
          for (unsigned dest = 0; dest < 2; ++dest) {
            uint32_t mode = (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
                            ((variant >> 2 & 3) << 6) | (variant & 16 ? GOC_ALU_CLAMP : 0) |
                            (variant & 32 ? GOC_ALU_HIGH_A : 0) |
                            (variant & 64 ? GOC_ALU_HIGH_D : 0);
            uint64_t descriptor = route_index ? goc_test::dpp_modes[route_index - 1] : 0;
            if (descriptor & GOC_DPP16)
              descriptor =
                  (descriptor & ~GOC_DPP_ROW_MASK) | (uint64_t(variant & 15) << GOC_DPP_ROW_SHIFT);
            uint32_t words[2][66], before[2][66];
            for (unsigned reg = 0; reg < 2; ++reg)
              for (unsigned lane = 0; lane < 66; ++lane)
                words[reg][lane] = values[(lane + reg * 5) % 16] |
                                   (uint32_t(values[(lane * 7 + reg * 3) % 16]) << 16);
            std::memcpy(before, words, sizeof words);
            uint32_t *d = words[dest] + 1;
            const uint32_t *a = words[0] + 1;
            uint64_t exec_mask = 0x87654321fedcba98ULL;
            bool saturate = variant & 1;
            uint32_t exceptions = 0x80000055;
            ASSERT_EQ(wave64[op](cpu | (saturate ? GOC_FP16_OVFL : 0), exec_mask, descriptor | mode,
                                 &d, &a, &exceptions),
                      GOC_SUCCESS);
            EXPECT_EQ(exceptions, 0x80000055U);
            for (unsigned lane = 0; lane < 64; ++lane) {
              int source = int(lane);
              bool enabled = descriptor ? goc_test::dpp_source(descriptor, exec_mask, lane, source)
                                        : ((exec_mask >> lane) & 1);
              if (enabled) {
                EXPECT_TRUE(check(op, d[lane], before[dest][lane + 1],
                                  goc_test::half_unary_reference(
                                      op, source < 0 ? 0 : before[0][source + 1], mode, saturate),
                                  mode))
                    << op << "/" << cpu << "/" << variant << "/" << route_index << "/" << dest
                    << "/" << lane;
              } else {
                EXPECT_EQ(d[lane], before[dest][lane + 1]);
              }
            }
            for (unsigned reg = 0; reg < 2; ++reg) {
              EXPECT_EQ(words[reg][0], before[reg][0]);
              EXPECT_EQ(words[reg][65], before[reg][65]);
              if (reg != dest) {
                for (unsigned lane = 1; lane <= 64; ++lane)
                  EXPECT_EQ(words[reg][lane], before[reg][lane]) << "source lane=" << lane - 1;
              }
            }
          }
}

TEST(HalfUnary, Wave64ValidationAndOptionalState) {
  for (unsigned op = 0; op < 11; ++op) {
    auto fn = wave64[op];
    bool supported = op < 7 || op == 10;
    uint32_t words[64] = {};
    uint32_t *d = words;
    const uint32_t *a = words;
    EXPECT_EQ(fn(0, 0, 0, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(fn(0, UINT64_MAX, GOC_ALU_NEG_B, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(
        fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT64_MAX, 0, &d, &a, nullptr),
        supported ? GOC_SUCCESS : GOC_ERROR_UNSUPPORTED_SEMANTICS);
    uint32_t exceptions = 0x80000055;
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT64_MAX, 0, &d, &a, &exceptions),
              GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
    EXPECT_EQ(exceptions, 0x80000055U);
  }
}

TEST(HalfUnary, Rdna3Dx10ClampPolicies) {
  const goc_test::HalfUnaryFn narrow[] = {goc_v_trunc_f16, goc_v_ceil_f16,      goc_v_rndne_f16,
                                          goc_v_floor_f16, goc_v_sqrt_f16,      goc_v_rcp_f16,
                                          goc_v_rsq_f16,   goc_v_exp_f16,       goc_v_log_f16,
                                          goc_v_fract_f16, goc_v_frexp_mant_f16};
  const decltype(&goc_v_trunc_f16_wave64) wide[] = {
      goc_v_trunc_f16_wave64, goc_v_ceil_f16_wave64,      goc_v_rndne_f16_wave64,
      goc_v_floor_f16_wave64, goc_v_sqrt_f16_wave64,      goc_v_rcp_f16_wave64,
      goc_v_rsq_f16_wave64,   goc_v_exp_f16_wave64,       goc_v_log_f16_wave64,
      goc_v_fract_f16_wave64, goc_v_frexp_mant_f16_wave64};
  const uint32_t inputs[] = {0x7c01, 0xfc55, 0x7e23, 0xfe7f, 0,      0x8000, 1,      0x8001,
                             0x3800, 0xb800, 0x4000, 0xc000, 0x7c00, 0xfc00, 0x3c00, 0x7bff};
  for (unsigned op = 0; op < 11; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool precise : {false, true}) {
        if (precise && op >= 7 && op <= 9) // EXP/LOG/FRACT have no borrowed exact model.
          continue;
        for (bool wave64 : {false, true})
          for (unsigned policy = 0; policy < 16; ++policy)
            for (unsigned halves = 0; halves < 4; ++halves)
              for (unsigned source_mode = 0; source_mode < 8; ++source_mode)
                for (uint64_t route : {uint64_t(0), goc_test::dpp_modes[1]}) {
                  uint64_t flags =
                      cpu | (precise ? GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT : 0) |
                      (policy & 1 ? GOC_FP_IEEE_MODE : 0) |
                      (policy & 2 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                      (policy & 4 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                      (policy & 8 ? GOC_FP16_OVFL : 0);
                  unsigned a_shift = halves & 1 ? 16 : 0, d_shift = halves & 2 ? 16 : 0;
                  uint64_t mode = (a_shift ? GOC_ALU_HIGH_A : 0) | (d_shift ? GOC_ALU_HIGH_D : 0) |
                                  route | (source_mode & 1 ? GOC_ALU_NEG_A : 0) |
                                  (source_mode & 2 ? GOC_ALU_ABS_A : 0) |
                                  (source_mode & 4 ? GOC_ALU_OMOD_2 : 0);
                  uint64_t exec_mask = wave64 ? 0x9171abcd8912a5a5ULL : 0x8912a5a5U;
                  uint32_t input[64], unclamped[64], actual[64];
                  for (unsigned lane = 0; lane < 64; ++lane)
                    input[lane] =
                        (0xdeadbeefU & ~(65535U << a_shift)) | (inputs[lane % 16] << a_shift);
                  auto call = [&](uint64_t f, uint64_t m, uint32_t *d, uint32_t *state,
                                  bool alias = false) {
                    const uint32_t *a = alias ? d : input;
                    return wave64 ? wide[op](f, exec_mask, m, &d, &a, state)
                                  : narrow[op](f, uint32_t(exec_mask), m, &d, &a, state);
                  };
                  std::fill(unclamped, unclamped + 64, 0xdeadbeefU);
                  ASSERT_EQ(call(flags, mode, unclamped, nullptr), GOC_SUCCESS);
                  for (bool alias : {false, true})
                    for (bool preserve : {false, true}) {
                      if (alias)
                        std::copy(input, input + 64, actual);
                      else
                        std::fill(actual, actual + 64, 0xdeadbeefU);
                      uint32_t state = 0x80000000;
                      ASSERT_EQ(call(flags | (preserve ? GOC_FP_DX10_CLAMP_OFF : 0),
                                     mode | GOC_ALU_CLAMP, actual, precise ? nullptr : &state,
                                     alias),
                                GOC_SUCCESS);
                      if (!precise) {
                        EXPECT_EQ(state, 0x80000000U);
                      }
                      for (unsigned lane = 0; lane < 64; ++lane) {
                        if (!((exec_mask >> lane) & 1)) {
                          EXPECT_EQ(actual[lane], alias ? input[lane] : 0xdeadbeefU);
                          continue;
                        }
                        uint32_t value = (unclamped[lane] >> d_shift) & 0xffff;
                        uint32_t magnitude = value & 0x7fff;
                        uint32_t expected = value;
                        if (!(magnitude > 0x7c00U && preserve))
                          expected = (value >> 15) || magnitude > 0x7c00U ? 0U
                                     : value > 0x3c00U                    ? 0x3c00U
                                                                          : value;
                        uint32_t before = alias ? input[lane] : 0xdeadbeefU;
                        if (precise)
                          EXPECT_EQ(actual[lane],
                                    (before & ~(65535U << d_shift)) | (expected << d_shift));
                        else
                          EXPECT_TRUE(
                              check(op, actual[lane], before, uint16_t(expected), uint32_t(mode)))
                              << "lane=" << lane;
                      }
                    }
                }
      }
  for (unsigned op = 0; op < 11; ++op) {
    EXPECT_EQ(goc_test::half_unary_functions[op](GOC_FP_DX10_CLAMP_OFF, 1, GOC_ALU_CLAMP, nullptr,
                                                 nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(
        wave64[op](GOC_FP_DX10_CLAMP_OFF, 1ULL << 63, GOC_ALU_CLAMP, nullptr, nullptr, nullptr),
        GOC_ERROR_INVALID_FLAGS);
    uint32_t state = 0x12345678;
    EXPECT_EQ(narrow[op](GOC_FP_DX10_CLAMP_OFF | GOC_SEMANTICS_EXACT_EMPIRICAL, 0, UINT64_MAX,
                         nullptr, nullptr, &state),
              GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
    EXPECT_EQ(state, 0x12345678U);
  }
}
