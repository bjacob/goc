// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "exec_masks.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "half_fma_exceptions_hardware.h"
#include "half_fma_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {

const uint64_t semantics[] = {GOC_SEMANTICS_LOOSE,
                              GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT};
const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x401,  0x7ff,
                           0x800,  0x3800, 0xb800, 0x3bff, 0x3c00, 0xbc00, 0x3c01, 0x4200,
                           0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfc12, 0x7fc1, 0xff80};

bool nan(uint16_t bits) { return (bits & 0x7fff) > 0x7c00; }

void check(uint32_t actual, uint32_t before, uint16_t want, uint32_t mode, bool exact) {
  int shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  EXPECT_EQ((actual ^ before) & ~(0xffffU << shift), 0u);
  uint16_t got = uint16_t(actual >> shift);
  if (!exact && nan(want)) {
    EXPECT_TRUE(nan(got));
  } else {
    EXPECT_EQ(got, want);
  }
}

void run(uint64_t flags, uint32_t mode, uint32_t (&words)[4][32]) {
  uint32_t before[4][32];
  std::memcpy(before, words, sizeof(before));
  uint32_t *p[] = {words[0], words[1], words[2], words[3]};
  ASSERT_EQ(goc_v_fma_f16(flags, UINT32_MAX, mode, p + 3, p, p + 1, p + 2, nullptr), GOC_SUCCESS);
  for (int lane = 0; lane < 32; ++lane) {
    SCOPED_TRACE(lane);
    check(words[3][lane], before[3][lane],
          goc_test::half_fma_reference::evaluate(before[0][lane], before[1][lane], before[2][lane],
                                                 mode, flags & GOC_FP16_OVFL),
          mode, flags & GOC_SEMANTICS_MASK);
  }
}

} // namespace

TEST(HalfFma, BoundaryCartesianProductsEveryEncodingAndRandomTriples) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (auto sem : semantics) {
      std::mt19937 random(84712);
      for (unsigned base = 0; base < 13824 + 262144; base += 32) {
        uint32_t words[4][32];
        uint32_t mode = ((base / 32) % 4) << 6;
        uint64_t flags = cpu | sem | ((base & 32) ? GOC_FP16_OVFL : 0);
        SCOPED_TRACE(::testing::Message() << cpu << '/' << sem << '/' << base << '/' << mode);
        for (int lane = 0; lane < 32; ++lane) {
          unsigned i = base + lane;
          for (int reg = 0; reg < 3; ++reg) {
            uint16_t code = base < 13824                       ? values[i % 24]
                            : reg == 0 && base < 13824 + 65536 ? uint16_t(i - 13824)
                                                               : uint16_t(random());
            if (base < 13824)
              i /= 24;
            words[reg][lane] = 0xdead0000 | code;
          }
          words[3][lane] = 0xfacecafe;
        }
        run(flags, mode, words);
      }
    }
}

TEST(HalfFma, AllModifiersAndHalfSelectors) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (auto sem : semantics)
      for (uint32_t mode = 0; mode < 8192; ++mode)
        for (bool saturate : {false, true}) {
          SCOPED_TRACE(::testing::Message() << cpu << '/' << sem << '/' << mode << '/' << saturate);
          uint32_t words[4][32];
          for (int reg = 0; reg < 4; ++reg)
            for (int lane = 0; lane < 32; ++lane)
              words[reg][lane] = values[(lane + reg * 5) % 24] |
                                 (uint32_t(values[(lane * 7 + reg * 3) % 24]) << 16);
          run(cpu | sem | (saturate ? GOC_FP16_OVFL : 0), mode, words);
        }
}

TEST(HalfFma, MasksAndAllWholeRegisterAliases) {
  const int layouts[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  const uint32_t modes[] = {
      0, 8191, GOC_ALU_HIGH_A | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF,
      GOC_ALU_HIGH_B | GOC_ALU_ABS_C | GOC_ALU_OMOD_4};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (auto sem : semantics)
      for (auto mode : modes)
        for (auto exec_mask : exec_masks())
          for (const auto &layout : layouts)
            for (int dest = 0; dest < 4; ++dest) {
              SCOPED_TRACE(::testing::Message()
                           << cpu << '/' << sem << '/' << mode << '/' << exec_mask << '/' << dest
                           << '/' << layout[0] << layout[1] << layout[2]);
              uint32_t words[4][34], before[4][34];
              for (int reg = 0; reg < 4; ++reg) {
                std::fill(words[reg], words[reg] + 34, 0xdeadbeef);
                for (int lane = 1; lane <= 32; ++lane)
                  words[reg][lane] = values[(lane + reg * 5) % 24] |
                                     (uint32_t(values[(lane * 7 + reg * 3) % 24]) << 16);
              }
              std::memcpy(before, words, sizeof(before));
              uint32_t *p[] = {words[0] + 1, words[1] + 1, words[2] + 1, words[3] + 1};
              ASSERT_EQ(goc_v_fma_f16(cpu | sem, exec_mask, mode, p + dest, p + layout[0],
                                      p + layout[1], p + layout[2], nullptr),
                        GOC_SUCCESS);
              for (int reg = 0; reg < 4; ++reg)
                for (int lane = 0; lane < 34; ++lane) {
                  if (reg == dest && lane >= 1 && lane <= 32 && ((exec_mask >> (lane - 1)) & 1)) {
                    check(words[reg][lane], before[reg][lane],
                          goc_test::half_fma_reference::evaluate(
                              before[layout[0]][lane], before[layout[1]][lane],
                              before[layout[2]][lane], mode, false),
                          mode, sem != 0);
                  } else {
                    EXPECT_EQ(words[reg][lane], before[reg][lane]);
                  }
                }
            }
}

TEST(HalfFma, RocjitsuHardwareWitnesses) {
  // From rocjitsu tests/valu_fp_mode_test.cpp's gfx1201 FMA captures.
  struct Case {
    uint16_t a, b, c, want;
    uint32_t mode;
    bool saturate;
  };

  const Case cases[] = {{0x7fc1, 0xff80, 0xff80, 0x7fc1, 0, false},
                        {0x7c01, 0x3c00, 0, 0x7e01, 0, false},
                        {0x3c00, 0xfc12, 0x7e01, 0xfe12, 0, false},
                        {0, 0x7c00, 0x7e01, 0xfe00, 0, false},
                        {0x7c00, 0x3c00, 0xfc00, 0xfe00, 0, false},
                        {0x400, 0xbc00, 0, 0x8000, GOC_ALU_OMOD_HALF, false},
                        {0x3c00, 0x7bff, 0x7bff, 0x7c00, GOC_ALU_OMOD_HALF, false},
                        {0x3c00, 0x7bff, 0x7bff, 0x77ff, GOC_ALU_OMOD_HALF, true},
                        {0x400, 0x3800, 0, 0, GOC_ALU_OMOD_4, false},
                        {0x400, 0xb800, 0, 0, GOC_ALU_OMOD_4, false}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (auto sem : semantics)
      for (auto test : cases)
        for (bool clamp : {false, true}) {
          uint32_t words[4][32];
          uint32_t *p[] = {words[0], words[1], words[2], words[3]};
          std::fill(words[0], words[0] + 32, test.a);
          std::fill(words[1], words[1] + 32, test.b);
          std::fill(words[2], words[2] + 32, test.c);
          std::fill(words[3], words[3] + 32, 0xdeadbeef);
          uint32_t mode = test.mode | (clamp ? GOC_ALU_CLAMP : 0);
          uint16_t want = test.want;
          if (clamp)
            want = nan(want) || (want & 0x8000) ? 0 : std::min<uint16_t>(want, 0x3c00);
          ASSERT_EQ(goc_v_fma_f16(cpu | sem | (test.saturate ? GOC_FP16_OVFL : 0), UINT32_MAX, mode,
                                  p + 3, p, p + 1, p + 2, nullptr),
                    GOC_SUCCESS);
          for (auto word : words[3])
            check(word, 0xdeadbeef, want, mode, sem != 0);
        }
}

TEST(HalfFma, DoubleRoundingAndTininessBoundaries) {
  struct Case {
    uint16_t a, b, c, want;
    uint32_t mode;
  };

  const Case cases[] = {// FP32 FMA followed by FP16 narrowing would round these the wrong way.
                        {0x3c01, 0x3e00, 0x8001, 0x3e01, 0},
                        {0x3c03, 0x3e00, 0x0001, 0x3e05, 0},
                        {0xbc01, 0x3e00, 0x0001, 0xbe01, 0},
                        {0xbc03, 0x3e00, 0x8001, 0xbe05, 0},
                        // All three round to min-normal, but active OMOD tests tininess before
                        // exponent packing at 2^-14 - 2^-26, not at the usual half midpoint.
                        {1, 0x39ff, 0x03ff, 0, GOC_ALU_OMOD_2},
                        {1, 0x3a00, 0x03ff, 0x0800, GOC_ALU_OMOD_2},
                        {1, 0x3a01, 0x03ff, 0x0800, GOC_ALU_OMOD_2},
                        {1, 0x39ff, 0x03ff, 0x0400, 0},
                        {0x8001, 0x39ff, 0x83ff, 0, GOC_ALU_OMOD_HALF},
                        {0x8001, 0x3a00, 0x83ff, 0x8000, GOC_ALU_OMOD_HALF},
                        {0x8001, 0x3a01, 0x83ff, 0x8000, GOC_ALU_OMOD_HALF}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (auto sem : semantics)
      for (auto test : cases) {
        uint32_t words[4][32];
        uint32_t *p[] = {words[0], words[1], words[2], words[3]};
        std::fill(words[0], words[0] + 32, test.a);
        std::fill(words[1], words[1] + 32, test.b);
        std::fill(words[2], words[2] + 32, test.c);
        std::fill(words[3], words[3] + 32, 0xdeadbeef);
        ASSERT_EQ(goc_test::half_fma_reference::evaluate(test.a, test.b, test.c, test.mode, false),
                  test.want);
        ASSERT_EQ(goc_v_fma_f16(cpu | sem, UINT32_MAX, test.mode, p + 3, p, p + 1, p + 2, nullptr),
                  GOC_SUCCESS);
        for (auto word : words[3])
          check(word, 0xdeadbeef, test.want, test.mode, true);
      }
}

TEST(HalfFma, ExactPreservesHostEnvironment) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
      std::fesetround(rounding);
      std::feclearexcept(FE_ALL_EXCEPT);
      std::feraiseexcept(FE_DIVBYZERO);
      uint32_t words[4][32];
      for (int reg = 0; reg < 4; ++reg)
        for (int lane = 0; lane < 32; ++lane)
          words[reg][lane] = values[(lane * 5 + reg * 3) % 24];
      run(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, words);
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
    }
}

TEST(HalfFma, ValidationAndSemantics) {
  uint32_t data[32];
  std::fill(data, data + 32, 0xdeadbeef);
  auto p = data;
  for (uint32_t exec_mask : {0U, UINT32_MAX}) {
    for (int bit = 13; bit < 32; ++bit)
      EXPECT_EQ(goc_v_fma_f16(0, exec_mask, 1U << bit, &p, &p, &p, &p, nullptr),
                GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_v_fma_f16(1ULL << 63, exec_mask, 0, &p, &p, &p, &p, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(
        goc_v_fma_f16((2ULL << 16) | GOC_SEMANTICS_STRICT, exec_mask, 0, &p, &p, &p, &p, nullptr),
        GOC_ERROR_UNSUPPORTED_SEMANTICS);
  }
  for (auto word : data)
    EXPECT_EQ(word, 0xdeadbeef);
  EXPECT_EQ(goc_v_fma_f16(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr,
                          nullptr, nullptr, nullptr, nullptr),
            GOC_SUCCESS);
  EXPECT_EQ(goc_v_fma_f16(2ULL << 16, UINT32_MAX, 0, &p, &p, &p, &p, nullptr), GOC_SUCCESS);
}

TEST(HalfFma, ExceptionHardwareCorpus) {
  const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3c00, 0xbc00,
                             0x4000, 0x7bff, 0x77ff, 0x7c00, 0xfc00, 0x7c01, 0xfe03, 0x800};
  for (unsigned variant = 0; variant < 16; ++variant)
    for (bool saturate : {false, true}) {
      uint32_t mode =
          ((variant & 3) << 6) | (variant & 4 ? GOC_ALU_CLAMP : 0) |
          (variant & 8 ? GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_ABS_B | GOC_ALU_NEG_C : 0);
      uint64_t hash = goc_test::capture_hash_seed;
      for (unsigned i = 0; i < 8192; ++i) {
        uint32_t words[4][32] = {};
        uint64_t state = uint64_t(i) * 0x9e3779b97f4a7c15ULL;
        for (unsigned operand = 0; operand < 3; ++operand) {
          state ^= state >> 12;
          state ^= state << 25;
          state ^= state >> 27;
          words[operand][0] = i < 4096 ? values[(i >> (8 - operand * 4)) & 15]
                                       : uint16_t(state * 0x2545f4914f6cdd1dULL);
          // Inactive lanes would raise invalid if accidentally included.
          words[operand][1] = 0x7c01;
        }
        const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
        uint32_t *d[] = {words[3]};
        uint32_t exceptions = 0x80000000;
        uint64_t flags = GOC_SEMANTICS_EXACT_EMPIRICAL | (i & 1 ? GOC_SEMANTICS_STRICT : 0) |
                         (saturate ? GOC_FP16_OVFL : 0) | (i % (goc_init_cpu_flags() + 1));
        ASSERT_EQ(goc_v_fma_f16(flags, 1, mode, d, a, b, c, &exceptions), GOC_SUCCESS);
        ASSERT_TRUE(exceptions & 0x80000000);
        hash = goc_test::capture_hash_word(hash, exceptions & 127);
        // Literal and accumulator forms must generate the same flags, selecting
        // the same half, even when the destination aliases an input.
        if (!mode) {
          uint32_t expected = words[3][0];
          uint32_t mk_flags = 0x80000000, ak_flags = mk_flags, mac_flags = mk_flags;
          ASSERT_EQ(goc_v_fmamk_f16(flags, 1, 0, d, a, uint16_t(words[1][0]), c, &mk_flags),
                    GOC_SUCCESS);
          EXPECT_EQ(words[3][0], expected);
          ASSERT_EQ(goc_v_fmaak_f16(flags, 1, 0, d, a, b, uint16_t(words[2][0]), &ak_flags),
                    GOC_SUCCESS);
          EXPECT_EQ(words[3][0], expected);
          words[3][0] = words[2][0];
          ASSERT_EQ(goc_v_fmac_f16(flags, 1, 0, d, a, b, &mac_flags), GOC_SUCCESS);
          EXPECT_EQ(words[3][0], expected);
          EXPECT_EQ(mk_flags, exceptions);
          EXPECT_EQ(ak_flags, exceptions);
          EXPECT_EQ(mac_flags, exceptions);
        }
      }
      EXPECT_EQ(hash, goc_test::half_fma_exception_hashes[variant]) << variant << "/" << saturate;
    }
}

TEST(HalfFma, AccumulatorAndLiteralExceptionHardwareCorpora) {
  const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3c00, 0xbc00,
                             0x4000, 0x7bff, 0x77ff, 0x7c00, 0xfc00, 0x7c01, 0xfe03, 0x800};
  for (unsigned op = 0; op < 3; ++op)
    for (unsigned variant = 0; variant < (op ? 1U : 16U); ++variant)
      for (bool saturate : {false, true}) {
        uint32_t mode = ((variant & 3) << 6) | (variant & 4 ? GOC_ALU_CLAMP : 0) |
                        (variant & 8 ? GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_ABS_B : 0);
        uint64_t hash = goc_test::capture_hash_seed;
        for (unsigned i = 0; i < 8192; ++i) {
          uint32_t words[4][32] = {};
          uint64_t state = uint64_t(i) * 0x9e3779b97f4a7c15ULL;
          for (unsigned operand = 0; operand < 3; ++operand) {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            words[operand][0] = i < 4096 ? values[(i >> (8 - operand * 4)) & 15]
                                         : uint16_t(state * 0x2545f4914f6cdd1dULL);
          }
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
          uint32_t *d[] = {words[op ? 3 : 2]}, exceptions = 0x80000000;
          uint64_t flags = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                           (saturate ? GOC_FP16_OVFL : 0) | (i % (goc_init_cpu_flags() + 1));
          int status = op == 0   ? goc_v_fmac_f16(flags, 1, mode, d, a, b, &exceptions)
                       : op == 1 ? goc_v_fmamk_f16(flags, 1, 0, d, a, 0x3c01, c, &exceptions)
                                 : goc_v_fmaak_f16(flags, 1, 0, d, a, b, 0x3c01, &exceptions);
          ASSERT_EQ(status, GOC_SUCCESS);
          ASSERT_TRUE(exceptions & 0x80000000);
          hash = goc_test::capture_hash_word(hash, exceptions & 127);
        }
        EXPECT_EQ(hash, goc_test::other_half_fma_exception_hashes[op][variant]);
      }
}
