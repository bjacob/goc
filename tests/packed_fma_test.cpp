// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "exec_masks.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "half_fma_reference.h"
#include "packed_fma_exceptions_hardware.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {

const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3c00,
                           0x3c01, 0x3bff, 0x3800, 0xb800, 0x4000, 0xc000, 0xbc00,
                           0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe12};

int call(bool accumulate, uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  return accumulate ? goc_v_pk_fmac_f16(flags, exec_mask, mode, d, a, b, nullptr)
                    : goc_v_pk_fma_f16(flags, exec_mask, mode, d, a, b, c, nullptr);
}

uint32_t reference(uint32_t a, uint32_t b, uint32_t c, uint32_t mode, bool saturate) {
  const uint32_t words[] = {a, b, c};
  const uint32_t low_select[] = {GOC_PK_LO_A_HIGH, GOC_PK_LO_B_HIGH, GOC_PK_LO_C_HIGH};
  const uint32_t high_select[] = {GOC_PK_HI_A_LOW, GOC_PK_HI_B_LOW, GOC_PK_HI_C_LOW};
  const uint32_t low_neg[] = {GOC_PK_NEG_LO_A, GOC_PK_NEG_LO_B, GOC_PK_NEG_LO_C};
  const uint32_t high_neg[] = {GOC_PK_NEG_HI_A, GOC_PK_NEG_HI_B, GOC_PK_NEG_HI_C};
  uint32_t result = 0;
  for (int half = 0; half < 2; ++half) {
    uint16_t inputs[3];
    for (int reg = 0; reg < 3; ++reg) {
      bool high = half ? !(mode & high_select[reg]) : bool(mode & low_select[reg]);
      inputs[reg] = uint16_t(words[reg] >> (high ? 16 : 0));
      if (mode & (half ? high_neg[reg] : low_neg[reg]))
        inputs[reg] ^= 0x8000;
    }
    uint16_t value =
        goc_test::half_fma_reference::evaluate(inputs[0], inputs[1], inputs[2], 0, saturate);
    if (mode & GOC_PK_CLAMP)
      value = (value & 0x8000) || (value & 0x7fff) > 0x7c00 ? 0 : std::min<uint16_t>(value, 0x3c00);
    result |= uint32_t(value) << (16 * half);
  }
  return result;
}

void check(uint32_t actual, uint32_t expected, bool exact) {
  for (int half = 0; half < 2; ++half) {
    uint16_t got = uint16_t(actual >> (16 * half)), want = uint16_t(expected >> (16 * half));
    if (!exact && (want & 0x7fff) > 0x7c00) {
      EXPECT_GT(got & 0x7fff, 0x7c00);
    } else {
      EXPECT_EQ(got, want);
    }
  }
}

void fill(uint32_t (&words)[4][34]) {
  for (int reg = 0; reg < 4; ++reg) {
    std::fill(words[reg], words[reg] + 34, 0xfacecafe);
    for (int lane = 1; lane <= 32; ++lane)
      words[reg][lane] =
          values[(lane + reg * 7) % 20] | (uint32_t(values[(lane * 3 + reg * 5) % 20]) << 16);
  }
}

void run(bool accumulate, uint64_t flags, uint32_t exec_mask, uint64_t mode, int a, int b, int c,
         int d, uint32_t (&words)[4][34]) {
  uint32_t before[4][34];
  std::memcpy(before, words, sizeof(before));
  uint32_t *p[] = {words[0] + 1, words[1] + 1, words[2] + 1, words[3] + 1};
  ASSERT_EQ(call(accumulate, flags, exec_mask, mode, p + d, p + a, p + b, p + c), GOC_SUCCESS);
  for (int reg = 0; reg < 4; ++reg)
    for (int lane = 0; lane < 34; ++lane) {
      if (reg == d && lane >= 1 && lane <= 32 && ((exec_mask >> (lane - 1)) & 1)) {
        check(words[reg][lane],
              reference(before[a][lane], before[b][lane], before[accumulate ? d : c][lane], mode,
                        flags & GOC_FP16_OVFL),
              flags & GOC_SEMANTICS_MASK);
      } else {
        EXPECT_EQ(words[reg][lane], before[reg][lane]);
      }
    }
}

} // namespace

TEST(PackedFma, AllModifiersOverflowPoliciesAndCpuLevels) {
  for (bool accumulate : {false, true})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool exact : {false, true})
        for (bool saturate : {false, true})
          for (uint32_t mode = 0; mode < (accumulate ? 1u : 8192u); ++mode) {
            SCOPED_TRACE(::testing::Message() << accumulate << '/' << cpu << '/' << exact << '/'
                                              << saturate << '/' << mode);
            uint64_t flags = cpu | (saturate ? GOC_FP16_OVFL : 0) |
                             (exact ? GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT : 0);
            uint32_t words[4][34];
            fill(words);
            run(accumulate, flags, UINT32_MAX, mode, 0, 1, 2, mode % 4, words);
          }
}

TEST(PackedFma, EveryEncodingAndRandomTriples) {
  for (bool accumulate : {false, true})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool exact : {false, true}) {
        std::mt19937 random(2371);
        for (unsigned base = 0; base < 131072; base += 32) {
          SCOPED_TRACE(::testing::Message()
                       << accumulate << '/' << cpu << '/' << exact << '/' << base);
          uint32_t words[4][34];
          fill(words);
          for (int lane = 1; lane <= 32; ++lane) {
            uint32_t code = uint16_t(base + lane - 1);
            words[0][lane] = base < 65536 ? code | ((65535 - code) << 16) : random();
            words[1][lane] = random();
            words[2][lane] = random();
            words[3][lane] = random();
          }
          uint32_t mode = accumulate ? 0 : random() % 8192;
          uint64_t flags = cpu | ((base & 32) ? GOC_FP16_OVFL : 0) |
                           (exact ? GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT : 0);
          run(accumulate, flags, UINT32_MAX, mode, 0, 1, 2, 3, words);
        }
      }
}

TEST(PackedFma, MasksAndEveryWholeRegisterAlias) {
  const int layouts[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  const uint32_t modes[] = {0,
                            63,
                            8191,
                            GOC_PK_HI_A_LOW,
                            GOC_PK_HI_B_LOW,
                            GOC_PK_HI_C_LOW,
                            GOC_PK_LO_A_HIGH | GOC_PK_HI_A_LOW | GOC_PK_NEG_HI_C,
                            GOC_PK_LO_B_HIGH | GOC_PK_HI_B_LOW | GOC_PK_NEG_LO_A | GOC_PK_CLAMP,
                            GOC_PK_LO_C_HIGH | GOC_PK_HI_C_LOW | GOC_PK_NEG_HI_B};
  for (bool accumulate : {false, true})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool exact : {false, true})
        for (auto mode : modes) {
          if (accumulate && mode)
            continue;
          for (auto exec_mask : exec_masks())
            for (const auto &layout : layouts)
              for (int d = 0; d < 4; ++d) {
                SCOPED_TRACE(::testing::Message() << accumulate << '/' << cpu << '/' << exact << '/'
                                                  << mode << '/' << exec_mask << '/' << d);
                uint32_t words[4][34];
                fill(words);
                run(accumulate, cpu | (exact ? GOC_SEMANTICS_EXACT_EMPIRICAL : 0), exec_mask, mode,
                    layout[0], layout[1], layout[2], d, words);
              }
        }
}

TEST(PackedFma, BothResultsReadOriginalHalves) {
  struct Case {
    uint32_t mode, want;
  };

  const Case cases[] = {{0, 0x4c004800},
                        {GOC_PK_LO_A_HIGH | GOC_PK_HI_A_LOW, 0x4a004980},
                        {GOC_PK_HI_A_LOW, 0x4a004800},
                        {GOC_PK_HI_B_LOW, 0x4b004800},
                        {GOC_PK_HI_C_LOW, 0x4a804800},
                        {GOC_PK_LO_A_HIGH | GOC_PK_LO_B_HIGH | GOC_PK_LO_C_HIGH, 0x4c004c00},
                        {GOC_PK_HI_A_LOW | GOC_PK_HI_B_LOW | GOC_PK_HI_C_LOW, 0x48004800}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (auto test : cases)
      for (int d = 0; d < 4; ++d) {
        uint32_t words[4][32];
        uint32_t *p[] = {words[0], words[1], words[2], words[3]};
        std::fill(words[0], words[0] + 32, 0x40003c00);
        std::fill(words[1], words[1] + 32, 0x44004200);
        std::fill(words[2], words[2] + 32, 0x48004500);
        std::fill(words[3], words[3] + 32, 0xfacecafe);
        ASSERT_EQ(goc_v_pk_fma_f16(cpu, UINT32_MAX, test.mode, p + d, p, p + 1, p + 2, nullptr),
                  GOC_SUCCESS);
        for (auto word : words[d])
          EXPECT_EQ(word, test.want);
      }
}

TEST(PackedFma, HardwareWitnessesAndExactHostEnvironment) {
  // Packed witnesses from rocjitsu's f16_fma_nan_cases, plus fused tie cases.
  struct Case {
    uint16_t a, b, c, want;
  };

  const Case cases[] = {{0x7fc1, 0xff80, 0xff80, 0x7fc1}, {0x7c01, 0x3c00, 0, 0x7e01},
                        {0x3c00, 0xfc12, 0x7e01, 0xfe12}, {0, 0x7c00, 0x7e01, 0xfe00},
                        {0x7c00, 0x3c00, 0xfc00, 0xfe00}, {0x3c01, 0x3e00, 0x8001, 0x3e01},
                        {0x3c03, 0x3e00, 1, 0x3e05}};
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (bool accumulate : {false, true})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO})
        for (auto test : cases) {
          std::fesetround(rounding);
          std::feclearexcept(FE_ALL_EXCEPT);
          std::feraiseexcept(FE_DIVBYZERO);
          uint32_t words[4][32];
          uint32_t *p[] = {words[0], words[1], words[2], words[3]};
          std::fill(words[0], words[0] + 32, uint32_t(test.a) * 0x10001);
          std::fill(words[1], words[1] + 32, uint32_t(test.b) * 0x10001);
          std::fill(words[2], words[2] + 32, uint32_t(test.c) * 0x10001);
          std::fill(words[3], words[3] + 32, uint32_t(test.c) * 0x10001);
          EXPECT_EQ(call(accumulate, cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                         UINT32_MAX, 0, p + 3, p, p + 1, p + 2),
                    GOC_SUCCESS);
          for (auto word : words[3])
            EXPECT_EQ(word, uint32_t(test.want) * 0x10001);
          EXPECT_EQ(std::fegetround(), rounding);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
        }
}

TEST(PackedFma, ValidationAndZeroMasks) {
  for (bool accumulate : {false, true}) {
    uint32_t words[32];
    std::fill(words, words + 32, 0xfacecafe);
    auto p = words;
    for (uint32_t exec_mask : {0U, UINT32_MAX}) {
      for (unsigned bit = accumulate ? 0 : 13; bit < 32; ++bit)
        EXPECT_EQ(call(accumulate, 0, exec_mask, 1U << bit, &p, &p, &p, &p),
                  GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(call(accumulate, 1ULL << 63, exec_mask, 0, &p, &p, &p, &p),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(call(accumulate, (2ULL << 16) | GOC_SEMANTICS_STRICT, exec_mask, 0, &p, &p, &p, &p),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    for (auto word : words)
      EXPECT_EQ(word, 0xfacecafe);
    EXPECT_EQ(call(accumulate, GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr,
                   nullptr, nullptr, nullptr),
              GOC_SUCCESS);
    EXPECT_EQ(call(accumulate, 0, 0U, 0, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(call(accumulate, 2ULL << 16, UINT32_MAX, 0, &p, &p, &p, &p), GOC_SUCCESS);
  }
}

TEST(PackedFma, ExceptionHardwareCorpus) {
  const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3c00, 0xbc00,
                             0x4000, 0x7bff, 0x77ff, 0x7c00, 0xfc00, 0x7c01, 0xfe03, 0x800};
  for (unsigned op = 0; op < 2; ++op)
    for (unsigned variant = 0; variant < (op ? 1U : 16U); ++variant)
      for (bool saturate : {false, true}) {
        uint32_t mode = (variant & 1 ? 5 : 0) | (variant & 2 ? 48 : 0) |
                        (variant & 4 ? GOC_PK_CLAMP : 0) | (variant & 8 ? 0x1f80 : 0);
        uint64_t hash = goc_test::capture_hash_seed;
        for (unsigned i = 0; i < 8192; ++i) {
          uint32_t words[4][32] = {};
          uint64_t state = uint64_t(i) * 0x9e3779b97f4a7c15ULL;
          for (unsigned operand = 0; operand < 3; ++operand) {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            unsigned index = (i >> (8 - operand * 4)) & 15;
            uint32_t lo = i < 4096 ? values[index] : uint16_t(state * 0x2545f4914f6cdd1dULL);
            uint32_t hi = i < 4096 ? values[(index + 5) % 16] : ((lo * 0x9e37 + 0x1234) & 65535);
            words[operand][0] = lo | (hi << 16);
            words[operand][1] = 0x7c017c01;
          }
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
          uint32_t *d[] = {words[op ? 2 : 3]};
          uint32_t exceptions = 0x80000000;
          uint64_t flags = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                           (saturate ? GOC_FP16_OVFL : 0) | (i % (goc_init_cpu_flags() + 1));
          int status = op ? goc_v_pk_fmac_f16(flags, 1, 0, d, a, b, &exceptions)
                          : goc_v_pk_fma_f16(flags, 1, mode, d, a, b, c, &exceptions);
          ASSERT_EQ(status, GOC_SUCCESS);
          ASSERT_TRUE(exceptions & 0x80000000);
          hash = goc_test::capture_hash_word(hash, exceptions & 127);
        }
        EXPECT_EQ(hash, goc_test::packed_fma_exception_hashes[op][variant])
            << op << "/" << variant << "/" << saturate;
      }
}
