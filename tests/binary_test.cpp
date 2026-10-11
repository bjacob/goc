// SPDX-License-Identifier: MIT

#include "binary_reference.h"
#include "dpp_reference.h"
#include "exec_masks.h"
#include "fp_environment.h"
#include "fuzzy.h"
#include "goc/goc.h"
#include "internal.h"
#include "loose_tolerances.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

using Fn = decltype(&goc_v_add_f32_rdna4);
const Fn functions[] = {goc_v_add_f32_rdna4, goc_v_sub_f32_rdna4, goc_v_subrev_f32_rdna4,
                        goc_v_mul_f32_rdna4, goc_v_min_num_f32,   goc_v_max_num_f32,
                        goc_v_minimum_f32,   goc_v_maximum_f32,   goc_v_mul_dx9_zero_f32_rdna4};

const decltype(&goc_v_add_f32_rdna4_wave64) wave64[] = {
    goc_v_add_f32_rdna4_wave64, goc_v_sub_f32_rdna4_wave64, goc_v_subrev_f32_rdna4_wave64,
    goc_v_mul_f32_rdna4_wave64, goc_v_min_num_f32_wave64,   goc_v_max_num_f32_wave64,
    goc_v_minimum_f32_wave64,   goc_v_maximum_f32_wave64,   goc_v_mul_dx9_zero_f32_rdna4_wave64,
    goc_v_min_f32_wave64,       goc_v_max_f32_wave64};

::testing::AssertionResult compare(uint32_t actual, float expected, bool arithmetic) {
  uint32_t want = goc::as_bits(expected);
  unsigned magnitude = want & 0x7fffffff;
  if (magnitude > 0x7f800000) {
    if (std::isnan(goc::as_float(actual)))
      return ::testing::AssertionSuccess();
  } else if (arithmetic && magnitude && magnitude < 0x7f800000) {
    return goc_test::fuzzy(goc::as_float(actual), expected,
                           goc_test::scalar_arithmetic_f32_tolerance);
  } else if (actual == want) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << "actual bits=" << actual << " expected bits=" << want;
}

} // namespace

TEST(Binary, AllModifiersMasksAliasesAndSpecialValues) {
  const uint32_t values[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x00800000,
                             0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000, 0x3f800001, 0x3f7fffff,
                             0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345,
                             0x3e800000, 0x40000000, 0xc0400000};
  for (int op = 0; op < 9; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (uint32_t variant = 0; variant < 128; ++variant) {
          if (semantics && op != 0 && op != 3)
            continue;
          uint32_t mode = (variant & 3) | ((variant & 12) << 1) | ((variant & 112) << 2);
          for (uint32_t exec_mask : exec_masks())
            for (int alias = 0; alias < 3; ++alias) {
              SCOPED_TRACE(::testing::Message()
                           << op << "/" << cpu << "/" << mode << "/" << exec_mask << "/" << alias);
              uint32_t storage[3][34], before[32], saved[3][34];
              uint32_t *v[3];
              for (int reg = 0; reg < 3; ++reg) {
                std::fill(storage[reg], storage[reg] + 34, 0xdeadbeef);
                v[reg] = storage[reg] + 1;
              }
              for (int lane = 0; lane < 32; ++lane) {
                v[0][lane] = values[lane % 21];
                v[1][lane] = values[(lane * 5 + 3) % 21];
                before[lane] = v[alias][lane];
              }
              std::memcpy(saved, storage, sizeof(storage));
              ASSERT_EQ(functions[op](cpu | semantics | GOC_SEMANTICS_STRICT, exec_mask, mode,
                                      &v[alias], &v[0], &v[1], nullptr),
                        GOC_SUCCESS);
              for (int lane = 0; lane < 32; ++lane) {
                if (!((exec_mask >> lane) & 1)) {
                  EXPECT_EQ(v[alias][lane], before[lane]);
                  continue;
                }
                float want =
                    goc_test::binary_reference(op, goc::as_float(values[lane % 21]),
                                               goc::as_float(values[(lane * 5 + 3) % 21]), mode);
                EXPECT_TRUE(compare(v[alias][lane], want, !semantics && (op < 4 || op == 8)))
                    << "lane=" << lane;
              }
              for (unsigned reg = 0; reg < 3; ++reg) {
                EXPECT_EQ(storage[reg][0], 0xdeadbeef);
                EXPECT_EQ(storage[reg][33], 0xdeadbeef);
                if (reg != unsigned(alias)) {
                  EXPECT_EQ(std::memcmp(storage[reg], saved[reg], sizeof(storage[reg])), 0);
                }
              }
            }
        }
}

TEST(Binary, Validation) {
  for (auto fn : functions) {
    uint32_t a[32] = {}, d[32];
    std::fill(d, d + 32, 0xdeadbeef);
    auto pa = a, pd = d;
    EXPECT_EQ(fn(0, UINT32_MAX, GOC_ALU_NEG_C, &pd, &pa, &pa, nullptr), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(0, 0, GOC_ALU_HIGH_C, &pd, &pa, &pa, nullptr), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(
        fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, &pd, &pa, &pa, nullptr),
        fn == goc_v_add_f32_rdna4 || fn == goc_v_mul_f32_rdna4 ? GOC_SUCCESS
                                                               : GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, 0U, 0, &pd, &pa, &pa, nullptr), GOC_SUCCESS);
    for (auto value : d)
      EXPECT_EQ(value, 0xdeadbeef);
  }
}

TEST(Binary, MinMaxLiteralNaNsAndSignedZeros) {
  // Expected columns: minimumNumber, maximumNumber, minimum, maximum.
  const uint32_t cases[][6] = {
      {0, 0x80000000, 0x80000000, 0, 0x80000000, 0},
      {0x80000000, 0, 0x80000000, 0, 0x80000000, 0},
      {0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000},
      {0x7fc12345, 0xbf800000, 0xbf800000, 0xbf800000, 0x7fc12345, 0x7fc12345},
      {0xbf800000, 0xff812345, 0xbf800000, 0xbf800000, 0xffc12345, 0xffc12345},
      {0x7f812345, 0x3f800000, 0x3f800000, 0x3f800000, 0x7fc12345, 0x7fc12345},
      {0x3f800000, 0xffc12345, 0x3f800000, 0x3f800000, 0xffc12345, 0xffc12345},
      {0x7fc12345, 0xff812346, 0x7fc12345, 0x7fc12345, 0xffc12346, 0xffc12346},
      {0x7f812345, 0xff812346, 0x7fc12345, 0x7fc12345, 0x7fc12345, 0x7fc12345},
      {0xffc12345, 0x7fc12346, 0xffc12345, 0xffc12345, 0xffc12345, 0xffc12345},
      {0xff800000, 0x7f800000, 0xff800000, 0x7f800000, 0xff800000, 0x7f800000},
      {1, 0x80000001, 0x80000001, 1, 0x80000001, 1},
  };
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int op = 0; op < 4; ++op)
      for (const auto &test : cases) {
        SCOPED_TRACE(::testing::Message() << cpu << "/" << op << "/" << test[0] << "/" << test[1]);
        uint32_t a[32], b[32], d[32];
        std::fill(a, a + 32, test[0]);
        std::fill(b, b + 32, test[1]);
        auto pa = a, pb = b, pd = d;
        ASSERT_EQ(functions[op + 4](cpu, UINT32_MAX, 0, &pd, &pa, &pb, nullptr), GOC_SUCCESS);
        for (uint32_t value : d)
          EXPECT_EQ(value, test[op + 2]);
      }
}

TEST(Binary, Dx9ZeroOverridesEveryOtherOperand) {
  const uint32_t values[] = {0,          0x80000000, 1,          0x80000001, 0x3f800000,
                             0xbf800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
                             0x7fc12345, 0xffc12345, 0x7f812345, 0xff812345};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint32_t variant = 0; variant < 128; ++variant)
      for (uint32_t zero : {0U, 0x80000000U})
        for (bool reverse : {false, true}) {
          uint32_t a[32], b[32], d[32];
          uint32_t mode = (variant & 3) | ((variant & 12) << 1) | ((variant & 112) << 2);
          for (int lane = 0; lane < 32; ++lane) {
            a[lane] = reverse ? values[lane % 14] : zero;
            b[lane] = reverse ? zero : values[lane % 14];
          }
          auto pa = a, pb = b, pd = d;
          ASSERT_EQ(goc_v_mul_dx9_zero_f32_rdna4(cpu, UINT32_MAX, mode, &pd, &pa, &pb, nullptr),
                    GOC_SUCCESS);
          for (uint32_t value : d)
            EXPECT_EQ(value, 0u) << cpu << "/" << mode << "/" << zero << "/" << reverse;
        }
}

TEST(Binary, Dx9NonzeroUnderflowRetainsSign) {
  uint32_t a[32], b[32], d[32];
  std::fill(a, a + 32, 0x80000001);
  std::fill(b, b + 32, 1);
  auto pa = a, pb = b, pd = d;
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    ASSERT_EQ(goc_v_mul_dx9_zero_f32_rdna4(cpu, UINT32_MAX, 0, &pd, &pa, &pb, nullptr),
              GOC_SUCCESS);
    for (uint32_t value : d)
      EXPECT_EQ(value, 0x80000000);
  }
}

TEST(Binary, Wave64ModifiersDppMasksAndAliases) {
  const uint32_t values[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x00800000,
                             0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000, 0x3f800001, 0x3f7fffff,
                             0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345,
                             0x7f812345, 0xff812345, 0xc0400000};
  for (unsigned op = 0; op < 11; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < 128; ++variant)
        for (unsigned route_index = 0; route_index < 8; ++route_index)
          for (unsigned alias = 0; alias < 3; ++alias) {
            uint32_t mode = (variant & 3) | ((variant & 12) << 1) | ((variant & 112) << 2);
            uint64_t descriptor = route_index ? goc_test::dpp_modes[route_index - 1] : 0;
            if (descriptor & GOC_DPP16)
              descriptor =
                  (descriptor & ~GOC_DPP_ROW_MASK) | (uint64_t(variant & 15) << GOC_DPP_ROW_SHIFT);
            uint32_t words[3][66], before[3][66];
            for (unsigned reg = 0; reg < 3; ++reg)
              for (unsigned lane = 0; lane < 66; ++lane)
                words[reg][lane] = values[(lane * (reg + 1) + 7 * reg) % 21];
            std::memcpy(before, words, sizeof words);
            uint32_t *d = words[alias] + 1;
            const uint32_t *a = words[0] + 1, *b = words[1] + 1;
            uint64_t exec_mask = 0x87654321fedcba98ULL;
            uint32_t excp_flag_user = 0x12345678;
            ASSERT_EQ(wave64[op](cpu, exec_mask, mode | descriptor, &d, &a, &b, &excp_flag_user),
                      GOC_SUCCESS);
            EXPECT_EQ(excp_flag_user, 0x12345678U);
            for (unsigned lane = 0; lane < 64; ++lane) {
              int source = int(lane);
              bool enabled = descriptor ? goc_test::dpp_source(descriptor, exec_mask, lane, source)
                                        : ((exec_mask >> lane) & 1);
              if (!enabled) {
                EXPECT_EQ(d[lane], before[alias][lane + 1]);
                continue;
              }
              float expected = goc_test::binary_reference(
                  op >= 9 ? op - 5 : op, goc::as_float(source < 0 ? 0 : before[0][source + 1]),
                  goc::as_float(before[1][lane + 1]), op >= 9 ? mode & ~GOC_ALU_OMOD_HALF : mode);
              EXPECT_TRUE(compare(d[lane], expected, op < 4 || op == 8))
                  << cpu << '/' << op << '/' << mode << '/' << lane;
            }
            for (unsigned reg = 0; reg < 3; ++reg) {
              EXPECT_EQ(words[reg][0], before[reg][0]);
              EXPECT_EQ(words[reg][65], before[reg][65]);
              if (reg != alias) {
                EXPECT_EQ(std::memcmp(words[reg], before[reg], sizeof(words[reg])), 0);
              }
            }
          }
}

TEST(Binary, Wave64ValidationAndOptionalState) {
  for (auto fn : wave64) {
    EXPECT_EQ(fn(0, 0, 0, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(fn(0, UINT64_MAX, GOC_ALU_NEG_C, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr, nullptr,
                 nullptr, nullptr),
              fn == goc_v_add_f32_rdna4_wave64 || fn == goc_v_mul_f32_rdna4_wave64
                  ? GOC_SUCCESS
                  : GOC_ERROR_UNSUPPORTED_SEMANTICS);
    uint32_t excp_flag_user = 0x12345678;
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT64_MAX, 0, nullptr, nullptr, nullptr,
                 &excp_flag_user),
              GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
    EXPECT_EQ(excp_flag_user, 0x12345678U);
  }
}

TEST(Binary, ArchitectureOutputScalingGuestModesAndDpp) {
  // RDNA3 ISA 7.2.3.1: OMOD is disabled by preserved output denormals or IEEE.
  // RDNA4 applies OMOD regardless. Finite powers of two avoid rounding ambiguity.
  const Fn rdna3[] = {goc_v_add_f32, goc_v_sub_f32, goc_v_subrev_f32,      goc_v_mul_f32,
                      goc_v_min_f32, goc_v_max_f32, goc_v_mul_dx9_zero_f32};
  const unsigned rdna4_index[] = {0, 1, 2, 3, 4, 5, 8};
  const decltype(&goc_v_add_f32_wave64) rdna3_wide[] = {
      goc_v_add_f32_wave64,         goc_v_sub_f32_wave64, goc_v_subrev_f32_wave64,
      goc_v_mul_f32_wave64,         goc_v_min_f32_wave64, goc_v_max_f32_wave64,
      goc_v_mul_dx9_zero_f32_wave64};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned arch = 0; arch < 2; ++arch)
      for (unsigned op = 0; op < 7; ++op)
        for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
          for (unsigned settings = 0; settings < 8; ++settings)
            for (unsigned scale = 0; scale < 4; ++scale)
              for (unsigned route = 0; route < 8; ++route)
                for (unsigned width : {32U, 64U})
                  for (unsigned alias = 0; alias < 3; ++alias) {
                    if (semantics && op != 0 && op != 3)
                      continue;
                    SCOPED_TRACE(::testing::Message()
                                 << cpu << '/' << arch << '/' << op << '/' << settings << '/'
                                 << scale << '/' << route << '/' << width << '/' << alias);
                    uint64_t flags = cpu | semantics | GOC_SEMANTICS_STRICT |
                                     (settings & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                                     (settings & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                                     (settings & 4 ? GOC_FP_IEEE_MODE : 0);
                    uint32_t words[3][66], before[3][66];
                    for (unsigned reg = 0; reg < 3; ++reg)
                      for (unsigned lane = 0; lane < 66; ++lane)
                        words[reg][lane] = goc::as_bits(float(1 + (lane + reg) % 8));
                    std::memcpy(before, words, sizeof words);
                    uint32_t *d = words[alias] + 1;
                    const uint32_t *a = words[0] + 1, *b = words[1] + 1;
                    uint64_t exec_mask = width == 32 ? 0xfedcba98ULL : 0x87654321fedcba98ULL;
                    uint64_t descriptor = route ? goc_test::dpp_modes[route - 1] : 0;
                    uint64_t mode = descriptor | (uint64_t(scale) << 6) | GOC_ALU_NEG_B;
                    uint32_t state = 0x12345678;
                    int status =
                        width == 32
                            ? (arch ? functions[rdna4_index[op]]
                                    : rdna3[op])(flags, uint32_t(exec_mask), mode, &d, &a, &b,
                                                 semantics ? nullptr : &state)
                            : (arch ? wave64[rdna4_index[op]] : rdna3_wide[op])(
                                  flags, exec_mask, mode, &d, &a, &b, semantics ? nullptr : &state);
                    ASSERT_EQ(status, GOC_SUCCESS);
                    if (!semantics) {
                      EXPECT_EQ(state, 0x12345678U);
                    }
                    const float scales[] = {1, 2, 4, 0.5f};
                    float factor = scales[arch || ((settings & 2) && !(settings & 4)) ? scale : 0];
                    for (unsigned lane = 0; lane < width; ++lane) {
                      int source = int(lane);
                      bool enabled = descriptor
                                         ? goc_test::dpp_source(descriptor, exec_mask, lane, source)
                                         : ((exec_mask >> lane) & 1);
                      uint32_t expected = before[alias][lane + 1];
                      if (enabled) {
                        float x = source < 0 ? 0 : goc::as_float(before[0][source + 1]);
                        float y = -goc::as_float(before[1][lane + 1]);
                        float value = op == 0   ? x + y
                                      : op == 1 ? x - y
                                      : op == 2 ? y - x
                                      : op == 4 ? std::min(x, y)
                                      : op == 5 ? std::max(x, y)
                                                : x * y;
                        if (op == 6 && (x == 0 || y == 0))
                          value = 0;
                        // Active OMOD maps an unscaled zero (including -0) to +0.
                        bool active_omod = scale && (arch || ((settings & 2) && !(settings & 4)));
                        expected = active_omod && value == 0 ? 0 : goc::as_bits(value * factor);
                      }
                      EXPECT_EQ(d[lane], expected) << lane;
                    }
                    for (unsigned reg = 0; reg < 3; ++reg) {
                      EXPECT_EQ(words[reg][0], before[reg][0]);
                      for (unsigned index = 1; index < 66; ++index) {
                        if (reg != alias || index > width) {
                          EXPECT_EQ(words[reg][index], before[reg][index]);
                        }
                      }
                    }
                  }
}

TEST(Binary, GuestDenormalFlushingAndUnsupportedExactState) {
  const Fn all[] = {goc_v_add_f32,          goc_v_sub_f32,       goc_v_subrev_f32,
                    goc_v_mul_f32,          goc_v_add_f32_rdna4, goc_v_sub_f32_rdna4,
                    goc_v_subrev_f32_rdna4, goc_v_mul_f32_rdna4};
  // Signed subnormal inputs; multiplication by 2 makes the result normal unless
  // the inputs were flushed first. Add/sub retain subnormal magnitudes.
  const uint32_t expected[][4] = {{0x807ffffe, 0x80000000, 0x80000000, 0x80000000},
                                  {0x807ffffc, 0, 0x80000000, 0},
                                  {0x007ffffc, 0, 0, 0},
                                  {0x80fffffc, 0x80000000, 0x80fffffc, 0x80000000}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned fn = 0; fn < 8; ++fn)
      for (unsigned settings = 0; settings < 4; ++settings) {
        uint32_t a[32], b[32], d[32];
        std::fill(a, a + 32, 0x807ffffeU);
        std::fill(b, b + 32, fn % 4 == 3 ? 0x40000000U : fn % 4 == 0 ? 0x80000000U : 0x80000002U);
        uint32_t *pd = d;
        const uint32_t *pa = a, *pb = b;
        uint64_t flags = cpu | (settings & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                         (settings & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0);
        ASSERT_EQ(all[fn](flags, UINT32_MAX, 0, &pd, &pa, &pb, nullptr), GOC_SUCCESS);
        for (uint32_t value : d)
          EXPECT_EQ(value, expected[fn % 4][settings]) << fn << '/' << settings << '/' << cpu;
        uint32_t state = 0x12345678;
        EXPECT_EQ(all[fn](flags | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, nullptr, nullptr,
                          nullptr, &state),
                  GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
        EXPECT_EQ(state, 0x12345678U);
        EXPECT_EQ(all[fn](flags | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0,
                          nullptr, nullptr, nullptr, nullptr),
                  fn % 4 == 0 || fn % 4 == 3 ? GOC_SUCCESS : GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
}

TEST(Binary, MinmaxGuestIeeeNaNsAndDenormalSelection) {
  // Columns A, B, RDNA3 IEEE=0 min/max, RDNA3 IEEE=1 min/max, RDNA4 min/max.
  // Manual V_MIN_F32/V_MAX_F32 pseudocode gives sNaNs priority only in IEEE mode.
  const uint32_t cases[][8] = {
      {0x7f812345, 0x3f800000, 0x3f800000, 0x3f800000, 0x7fc12345, 0x7fc12345, 0x3f800000,
       0x3f800000},
      {0xbf800000, 0xff812346, 0xbf800000, 0xbf800000, 0xffc12346, 0xffc12346, 0xbf800000,
       0xbf800000},
      {0x7fc12345, 0xff812346, 0x7fc12345, 0x7fc12345, 0xffc12346, 0xffc12346, 0x7fc12345,
       0x7fc12345},
      {0x7f812345, 0xff812346, 0x7f812345, 0x7f812345, 0x7fc12345, 0x7fc12345, 0x7fc12345,
       0x7fc12345},
      {0x7f812345, 0xffc12346, 0x7f812345, 0x7f812345, 0x7fc12345, 0x7fc12345, 0x7fc12345,
       0x7fc12345},
      {0x7fc12345, 0xffc12346, 0x7fc12345, 0x7fc12345, 0x7fc12345, 0x7fc12345, 0x7fc12345,
       0x7fc12345},
      {0x7fc12345, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
       0x3f800000},
      {0, 0x80000000, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0},
      {0x80000001, 1, 0x80000001, 1, 0x80000001, 1, 0x80000001, 1},
  };
  const Fn narrow[] = {goc_v_min_f32,          goc_v_max_f32,         goc_v_min_num_f32,
                       goc_v_max_num_f32,      goc_v_dual_min_f32,    goc_v_dual_max_f32,
                       goc_v_dual_min_num_f32, goc_v_dual_max_num_f32};
  const decltype(&goc_v_min_f32_wave64) wide[] = {goc_v_min_f32_wave64, goc_v_max_f32_wave64,
                                                  goc_v_min_num_f32_wave64,
                                                  goc_v_max_num_f32_wave64};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned op = 0; op < 8; ++op)
      for (unsigned settings = 0; settings < 8; ++settings)
        for (unsigned width : {32U, 64U})
          for (unsigned alias = 0; alias < 3; ++alias) {
            if (op >= 4 && width == 64)
              continue;
            uint32_t words[3][64], before[3][64];
            for (unsigned lane = 0; lane < 64; ++lane) {
              words[0][lane] = cases[lane % 9][0];
              words[1][lane] = cases[lane % 9][1];
              words[2][lane] = 0xdeadbeef;
            }
            std::memcpy(before, words, sizeof words);
            uint32_t *d = words[alias];
            const uint32_t *a = words[0], *b = words[1];
            uint64_t flags = cpu | (settings & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                             (settings & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                             (settings & 4 ? GOC_FP_IEEE_MODE : 0);
            uint64_t exec_mask = 0x87654321fedcba98ULL;
            uint32_t state = 0x12345678;
            ASSERT_EQ(width == 32 ? narrow[op](flags, uint32_t(exec_mask), 0, &d, &a, &b, &state)
                                  : wide[op](flags, exec_mask, 0, &d, &a, &b, &state),
                      GOC_SUCCESS);
            EXPECT_EQ(state, 0x12345678U);
            for (unsigned reg = 0; reg < 3; ++reg) {
              if (reg != alias) {
                EXPECT_EQ(std::memcmp(words[reg], before[reg], sizeof(words[reg])), 0);
              }
            }
            unsigned column = 2 + (op % 4 >= 2 ? 4 : settings & 4 ? 2 : 0) + (op & 1);
            for (unsigned lane = 0; lane < 64; ++lane) {
              if (lane >= width) {
                EXPECT_EQ(d[lane], before[alias][lane]);
                continue;
              }
              uint32_t expected = cases[lane % 9][column];
              if ((settings & 1) && lane % 9 == 8)
                expected &= 0x80000000U;
              if (!((exec_mask >> lane) & 1))
                expected = before[alias][lane];
              EXPECT_EQ(d[lane], expected) << cpu << '/' << op << '/' << settings << '/' << lane;
            }
          }
}

TEST(Binary, Dx9GuestInputFlushPrecedesZeroOverride) {
  const Fn narrow[] = {goc_v_mul_dx9_zero_f32, goc_v_mul_dx9_zero_f32_rdna4};
  const decltype(&goc_v_mul_dx9_zero_f32_wave64) wide[] = {goc_v_mul_dx9_zero_f32_wave64,
                                                           goc_v_mul_dx9_zero_f32_rdna4_wave64};
  // A negative subnormal becomes -0 when flushed; DX9 then overrides infinity
  // and NaN with +0, rather than producing an invalid multiplication result.
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned arch = 0; arch < 2; ++arch)
      for (unsigned width : {32U, 64U})
        for (uint32_t other : {0x7f800000U, 0xff800000U, 0x7f812345U, 0xffc12345U}) {
          uint32_t a[64], b[64], d[64];
          std::fill(a, a + 64, 0x80000001U);
          std::fill(b, b + 64, other);
          uint32_t *pd = d;
          const uint32_t *pa = a, *pb = b;
          uint64_t flags = cpu | GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_IEEE_MODE;
          ASSERT_EQ(width == 32 ? narrow[arch](flags, UINT32_MAX, 0, &pd, &pa, &pb, nullptr)
                                : wide[arch](flags, UINT64_MAX, 0, &pd, &pa, &pb, nullptr),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < width; ++lane)
            EXPECT_EQ(d[lane], 0U);
        }
}

TEST(Binary, ExactOptOutFallbackAndErrorOrdering) {
  const struct {
    Fn narrow;
    decltype(&goc_v_add_f32_wave64) wide;
    uint32_t expected;
    bool exact;
  } cases[] = {
      {goc_v_min_f32, goc_v_min_f32_wave64, 0xc0000000U, false},
      {goc_v_max_f32, goc_v_max_f32_wave64, 0x40400000U, false},
      {goc_v_min_num_f32, goc_v_min_num_f32_wave64, 0xc0000000U, false},
      {goc_v_max_num_f32, goc_v_max_num_f32_wave64, 0x40400000U, false},
      {goc_v_minimum_f32, goc_v_minimum_f32_wave64, 0xc0000000U, false},
      {goc_v_maximum_f32, goc_v_maximum_f32_wave64, 0x40400000U, false},
      {goc_v_add_f32, goc_v_add_f32_wave64, 0x3f800000U, true},
      {goc_v_add_f32_rdna4, goc_v_add_f32_rdna4_wave64, 0x3f800000U, true},
      {goc_v_sub_f32, goc_v_sub_f32_wave64, 0xc0a00000U, false},
      {goc_v_sub_f32_rdna4, goc_v_sub_f32_rdna4_wave64, 0xc0a00000U, false},
      {goc_v_subrev_f32, goc_v_subrev_f32_wave64, 0x40a00000U, false},
      {goc_v_subrev_f32_rdna4, goc_v_subrev_f32_rdna4_wave64, 0x40a00000U, false},
      {goc_v_mul_f32, goc_v_mul_f32_wave64, 0xc0c00000U, true},
      {goc_v_mul_f32_rdna4, goc_v_mul_f32_rdna4_wave64, 0xc0c00000U, true},
      {goc_v_mul_dx9_zero_f32, goc_v_mul_dx9_zero_f32_wave64, 0xc0c00000U, false},
      {goc_v_mul_dx9_zero_f32_rdna4, goc_v_mul_dx9_zero_f32_rdna4_wave64, 0xc0c00000U, false},
      {goc_v_dual_min_f32, nullptr, 0xc0000000U, false},
      {goc_v_dual_max_f32, nullptr, 0x40400000U, false},
      {goc_v_dual_min_num_f32, nullptr, 0xc0000000U, false},
      {goc_v_dual_max_num_f32, nullptr, 0x40400000U, false},
      {goc_v_dual_add_f32, nullptr, 0x3f800000U, true},
      {goc_v_dual_add_f32_rdna4, nullptr, 0x3f800000U, true},
      {goc_v_dual_sub_f32, nullptr, 0xc0a00000U, false},
      {goc_v_dual_sub_f32_rdna4, nullptr, 0xc0a00000U, false},
      {goc_v_dual_subrev_f32, nullptr, 0x40a00000U, false},
      {goc_v_dual_subrev_f32_rdna4, nullptr, 0x40a00000U, false},
      {goc_v_dual_mul_f32, nullptr, 0xc0c00000U, true},
      {goc_v_dual_mul_f32_rdna4, nullptr, 0xc0c00000U, true},
      {goc_v_dual_mul_dx9_zero_f32, nullptr, 0xc0c00000U, false},
      {goc_v_dual_mul_dx9_zero_f32_rdna4, nullptr, 0xc0c00000U, false},
  };

  for (const auto &test : cases)
    for (unsigned width : {32U, 64U}) {
      if (width == 64 && !test.wide)
        continue;
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (bool strict : {false, true}) {
          uint32_t a[64], b[64], output[64];
          std::fill_n(a, 64, 0xc0000000U); // -2
          std::fill_n(b, 64, 0x40400000U); // 3
          std::fill_n(output, 64, 0xdeadbeefU);
          const uint32_t *pa = a, *pb = b;
          uint32_t *d = output, state = 0x12345678U;
          auto call = [&](uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *report) {
            return width == 32 ? test.narrow(flags, uint32_t(exec_mask), mode, &d, &pa, &pb, report)
                               : test.wide(flags, exec_mask, mode, &d, &pa, &pb, report);
          };
          uint64_t flags =
              cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | (strict ? GOC_SEMANTICS_STRICT : 0);
          // Requested exact state must fail even with empty EXEC, invalid
          // instruction flags, or non-strict numerical fallback allowed.
          for (uint64_t exec_mask : {0ULL, 0xffffffffffffffffULL}) {
            EXPECT_EQ(call(flags, exec_mask, 0, &state), GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
            EXPECT_EQ(call(flags, exec_mask, 1ULL << 63, &state),
                      GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
          }
          EXPECT_EQ(state, 0x12345678U);
          for (auto word : output)
            EXPECT_EQ(word, 0xdeadbeefU);
          int result = call(flags, 0xffffffffffffffffULL, 0, nullptr);
          EXPECT_EQ(result, strict && !test.exact ? GOC_ERROR_UNSUPPORTED_SEMANTICS : GOC_SUCCESS);
          for (unsigned lane = 0; lane < width; ++lane)
            EXPECT_EQ(output[lane], strict && !test.exact ? 0xdeadbeefU : test.expected);
          // Loose reporting is optional and never blocks the SIMD calculation.
          EXPECT_EQ(call(cpu, 0xffffffffffffffffULL, 0, &state), GOC_SUCCESS);
          EXPECT_EQ(state, 0x12345678U);
          for (unsigned lane = 0; lane < width; ++lane)
            EXPECT_EQ(output[lane], test.expected);
        }
    }
}

TEST(Binary, BorrowedAddMulPolicyVectorsAcrossHostModes) {
  // rocjitsu tests/valu_fp_mode_test.cpp, binary_f32_policy_cases: captured
  // gfx1100/gfx1201 NaN, signed-zero and pre-packing tininess cases, round=RNE.
  const uint32_t inputs[][2] = {{0x7f801abc, 0xffc00222}, {0x3f800000, 0x7f801abc},
                                {0x80000000, 0x80000000}, {0, 0x80000000},
                                {0x00800000, 0x3f7fffff}, {0x80800000, 0x3f7fffff}};
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(rounding), 0);
    for (bool host_flush : {false, true}) {
#if defined(__x86_64__) || defined(_M_X64)
      unsigned controls = _mm_getcsr();
      controls = host_flush ? controls | 0x8040U : controls & ~0x8040U;
      _mm_setcsr(controls);
#else
      (void)host_flush;
#endif
      for (bool multiply : {false, true})
        for (bool rdna4 : {false, true})
          for (unsigned policy = 0; policy < 8; ++policy)
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
              for (unsigned form = 0; form < 3; ++form)
                for (unsigned target = 0; target < 3; ++target) {
                  uint32_t words[3][64], before[3][64], expected[64];
                  for (unsigned lane = 0; lane < 64; ++lane) {
                    unsigned row = lane % (multiply ? 6 : 4);
                    words[0][lane] = inputs[row][0];
                    words[1][lane] = multiply && row == 2 ? 0x3f800000U : inputs[row][1];
                    words[2][lane] = 0xdeadbeefU;
                    expected[lane] = row < 2 ? (rdna4 || (policy & 4) ? 0x7fc01abcU : 0x7f801abcU)
                                     : row == 2 || (row == 3 && multiply) ? 0x80000000U
                                     : row == 3                           ? 0U
                                     : row == 4 ? (policy & 2 ? 0U : 0x00800000U)
                                                : (policy & 2 ? 0x80000000U : 0x80800000U);
                  }
                  std::memcpy(before, words, sizeof words);
                  uint32_t *d = words[target];
                  const uint32_t *a = words[0], *b = words[1];
                  uint64_t flags = cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                                   (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                                   (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                                   (policy & 4 ? GOC_FP_IEEE_MODE : 0);
                  Fn ordinary = multiply ? (rdna4 ? goc_v_mul_f32_rdna4 : goc_v_mul_f32)
                                         : (rdna4 ? goc_v_add_f32_rdna4 : goc_v_add_f32);
                  Fn dual = multiply ? (rdna4 ? goc_v_dual_mul_f32_rdna4 : goc_v_dual_mul_f32)
                                     : (rdna4 ? goc_v_dual_add_f32_rdna4 : goc_v_dual_add_f32);
                  auto wide = multiply
                                  ? (rdna4 ? goc_v_mul_f32_rdna4_wave64 : goc_v_mul_f32_wave64)
                                  : (rdna4 ? goc_v_add_f32_rdna4_wave64 : goc_v_add_f32_wave64);
                  uint64_t exec_mask = 0xa5a55a5afedcba98ULL;
                  ASSERT_EQ(form == 1 ? wide(flags, exec_mask, 0, &d, &a, &b, nullptr)
                                      : (form == 2 ? dual : ordinary)(flags, uint32_t(exec_mask), 0,
                                                                      &d, &a, &b, nullptr),
                            GOC_SUCCESS);
                  for (unsigned reg = 0; reg < 3; ++reg) {
                    if (reg != target) {
                      EXPECT_EQ(std::memcmp(words[reg], before[reg], sizeof(words[reg])), 0);
                    }
                  }
                  for (unsigned lane = 0; lane < 64; ++lane) {
                    bool active = (form == 1 || lane < 32) && ((exec_mask >> lane) & 1);
                    EXPECT_EQ(d[lane], active ? expected[lane] : before[target][lane]);
                  }
                }
      EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
      EXPECT_EQ(_mm_getcsr(), controls);
#endif
    }
  }
}

// Independent binary64 arithmetic over deterministic arbitrary FP32 encodings.
// Raw literal tests separately constrain NaN payloads and signed-zero policies.
TEST(Binary, RandomInputsAgainstDoubleReference) {
  const Fn rdna3[] = {goc_v_add_f32,     goc_v_sub_f32,     goc_v_subrev_f32,
                      goc_v_mul_f32,     goc_v_min_f32,     goc_v_max_f32,
                      goc_v_minimum_f32, goc_v_maximum_f32, goc_v_mul_dx9_zero_f32};
  const decltype(&goc_v_add_f32_wave64) rdna3_wide[] = {
      goc_v_add_f32_wave64,     goc_v_sub_f32_wave64,     goc_v_subrev_f32_wave64,
      goc_v_mul_f32_wave64,     goc_v_min_f32_wave64,     goc_v_max_f32_wave64,
      goc_v_minimum_f32_wave64, goc_v_maximum_f32_wave64, goc_v_mul_dx9_zero_f32_wave64};
  for (bool older : {false, true})
    for (bool wide : {false, true})
      for (unsigned op = 0; op < 9; ++op)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          std::mt19937 random(68147);
          for (unsigned batch = 0; batch < 64; ++batch) {
            uint32_t words[3][64], before[3][64];
            for (auto &reg : words)
              for (auto &word : reg)
                word = random();
            std::memcpy(before, words, sizeof(words));
            unsigned variant = random() & 127;
            uint32_t mode = (variant & 3) | ((variant & 12) << 1) | ((variant & 112) << 2);
            uint32_t effective_mode =
                older && op != 6 && op != 7 ? mode & ~GOC_ALU_OMOD_HALF : mode;
            unsigned target = batch % 3;
            uint64_t exec_mask = batch & 1 ? 0x87654321fedcba98ULL : UINT64_MAX;
            const uint32_t *a = words[0], *b = words[1];
            uint32_t *d = words[target], state = 0xffffff80;
            ASSERT_EQ(wide ? (older ? rdna3_wide[op] : wave64[op])(cpu, exec_mask, mode, &d, &a, &b,
                                                                   &state)
                           : (older ? rdna3[op] : functions[op])(cpu, uint32_t(exec_mask), mode, &d,
                                                                 &a, &b, &state),
                      GOC_SUCCESS);
            ASSERT_EQ(state, 0xffffff80U);
            for (unsigned reg = 0; reg < 3; ++reg)
              for (unsigned lane = 0; lane < 64; ++lane) {
                if (reg != target || lane >= (wide ? 64U : 32U) || !((exec_mask >> lane) & 1)) {
                  ASSERT_EQ(words[reg][lane], before[reg][lane]);
                  continue;
                }
                float expected =
                    goc_test::binary_reference(op, goc::as_float(before[0][lane]),
                                               goc::as_float(before[1][lane]), effective_mode);
                ASSERT_TRUE(compare(words[reg][lane], expected, op < 4 || op == 8))
                    << older << "/" << wide << "/" << op << "/" << cpu << "/" << batch << "/"
                    << mode << "/" << lane;
              }
          }
        }
}
