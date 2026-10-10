// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
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
  ASSERT_EQ(goc_v_fma_f16_rdna4(flags, UINT32_MAX, mode, p + 3, p, p + 1, p + 2, nullptr),
            GOC_SUCCESS);
  for (int lane = 0; lane < 32; ++lane) {
    SCOPED_TRACE(lane);
    check(words[3][lane], before[3][lane],
          goc_test::half_fma_reference::evaluate(before[0][lane], before[1][lane], before[2][lane],
                                                 mode, flags & GOC_FP16_OVFL),
          mode, flags & GOC_SEMANTICS_MASK);
  }
}

int wave64_call(unsigned op, uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                uint16_t literal, uint32_t *excp_flag_user) {
  switch (op) {
  case 0:
    return goc_v_fma_f16_rdna4_wave64(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
  case 1:
    return goc_v_fmac_f16_rdna4_wave64(flags, exec_mask, mode, d, a, b, excp_flag_user);
  case 2:
    return goc_v_fmamk_f16_rdna4_wave64(flags, exec_mask, mode, d, a, literal, b, excp_flag_user);
  default:
    return goc_v_fmaak_f16_rdna4_wave64(flags, exec_mask, mode, d, a, b, literal, excp_flag_user);
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
              ASSERT_EQ(goc_v_fma_f16_rdna4(cpu | sem, exec_mask, mode, p + dest, p + layout[0],
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
          ASSERT_EQ(goc_v_fma_f16_rdna4(cpu | sem | (test.saturate ? GOC_FP16_OVFL : 0), UINT32_MAX,
                                        mode, p + 3, p, p + 1, p + 2, nullptr),
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
        ASSERT_EQ(
            goc_v_fma_f16_rdna4(cpu | sem, UINT32_MAX, test.mode, p + 3, p, p + 1, p + 2, nullptr),
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
      EXPECT_EQ(goc_v_fma_f16_rdna4(0, exec_mask, 1U << bit, &p, &p, &p, &p, nullptr),
                GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_v_fma_f16_rdna4(1ULL << 63, exec_mask, 0, &p, &p, &p, &p, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_v_fma_f16_rdna4((2ULL << 16) | GOC_SEMANTICS_STRICT, exec_mask, 0, &p, &p, &p, &p,
                                  nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
  }
  for (auto word : data)
    EXPECT_EQ(word, 0xdeadbeef);
  EXPECT_EQ(goc_v_fma_f16_rdna4(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr,
                                nullptr, nullptr, nullptr, nullptr),
            GOC_SUCCESS);
  EXPECT_EQ(goc_v_fma_f16_rdna4(2ULL << 16, UINT32_MAX, 0, &p, &p, &p, &p, nullptr), GOC_SUCCESS);
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
        ASSERT_EQ(goc_v_fma_f16_rdna4(flags, 1, mode, d, a, b, c, &exceptions), GOC_SUCCESS);
        ASSERT_TRUE(exceptions & 0x80000000);
        hash = goc_test::capture_hash_word(hash, exceptions & 127);
        // Literal and accumulator forms must generate the same flags, selecting
        // the same half, even when the destination aliases an input.
        if (!mode) {
          uint32_t expected = words[3][0];
          uint32_t mk_flags = 0x80000000, ak_flags = mk_flags, mac_flags = mk_flags;
          ASSERT_EQ(goc_v_fmamk_f16_rdna4(flags, 1, 0, d, a, uint16_t(words[1][0]), c, &mk_flags),
                    GOC_SUCCESS);
          EXPECT_EQ(words[3][0], expected);
          ASSERT_EQ(goc_v_fmaak_f16_rdna4(flags, 1, 0, d, a, b, uint16_t(words[2][0]), &ak_flags),
                    GOC_SUCCESS);
          EXPECT_EQ(words[3][0], expected);
          words[3][0] = words[2][0];
          ASSERT_EQ(goc_v_fmac_f16_rdna4(flags, 1, 0, d, a, b, &mac_flags), GOC_SUCCESS);
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
          int status = op == 0   ? goc_v_fmac_f16_rdna4(flags, 1, mode, d, a, b, &exceptions)
                       : op == 1 ? goc_v_fmamk_f16_rdna4(flags, 1, 0, d, a, 0x3c01, c, &exceptions)
                                 : goc_v_fmaak_f16_rdna4(flags, 1, 0, d, a, b, 0x3c01, &exceptions);
          ASSERT_EQ(status, GOC_SUCCESS);
          ASSERT_TRUE(exceptions & 0x80000000);
          hash = goc_test::capture_hash_word(hash, exceptions & 127);
        }
        EXPECT_EQ(hash, goc_test::other_half_fma_exception_hashes[op][variant]);
      }
}

TEST(HalfFma, Wave64UpperLaneExceptionHardwareCorpus) {
  const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3c00, 0xbc00,
                             0x4000, 0x7bff, 0x77ff, 0x7c00, 0xfc00, 0x7c01, 0xfe03, 0x800};
  for (unsigned variant = 0; variant < 16; ++variant)
    for (bool saturate : {false, true}) {
      uint32_t mode =
          ((variant & 3) << 6) | (variant & 4 ? GOC_ALU_CLAMP : 0) |
          (variant & 8 ? GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_ABS_B | GOC_ALU_NEG_C : 0);
      uint64_t hash = goc_test::capture_hash_seed;
      for (unsigned i = 0; i < 8192; ++i) {
        uint32_t words[4][64] = {};
        uint64_t state = uint64_t(i) * 0x9e3779b97f4a7c15ULL;
        for (unsigned operand = 0; operand < 3; ++operand) {
          state ^= state >> 12;
          state ^= state << 25;
          state ^= state >> 27;
          words[operand][63] = i < 4096 ? values[(i >> (8 - operand * 4)) & 15]
                                        : uint16_t(state * 0x2545f4914f6cdd1dULL);
          // Inactive lanes would raise invalid if accidentally included.
          words[operand][1] = 0x7c01;
        }
        const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
        uint32_t *d[] = {words[3]};
        uint32_t exceptions = 0x80000000;
        uint64_t flags = GOC_SEMANTICS_EXACT_EMPIRICAL | (i & 1 ? GOC_SEMANTICS_STRICT : 0) |
                         (saturate ? GOC_FP16_OVFL : 0) | (i % (goc_init_cpu_flags() + 1));
        ASSERT_EQ(goc_v_fma_f16_rdna4_wave64(flags, 1ULL << 63, mode, d, a, b, c, &exceptions),
                  GOC_SUCCESS);
        ASSERT_TRUE(exceptions & 0x80000000);
        hash = goc_test::capture_hash_word(hash, exceptions & 127);
        // Literal and accumulator forms must generate the same flags, selecting
        // the same half, even when the destination aliases an input.
        if (!mode) {
          uint32_t expected = words[3][63];
          uint32_t mk_flags = 0x80000000, ak_flags = mk_flags, mac_flags = mk_flags;
          ASSERT_EQ(goc_v_fmamk_f16_rdna4_wave64(flags, 1ULL << 63, 0, d, a, uint16_t(words[1][63]),
                                                 c, &mk_flags),
                    GOC_SUCCESS);
          EXPECT_EQ(words[3][63], expected);
          ASSERT_EQ(goc_v_fmaak_f16_rdna4_wave64(flags, 1ULL << 63, 0, d, a, b,
                                                 uint16_t(words[2][63]), &ak_flags),
                    GOC_SUCCESS);
          EXPECT_EQ(words[3][63], expected);
          words[3][63] = words[2][63];
          ASSERT_EQ(goc_v_fmac_f16_rdna4_wave64(flags, 1ULL << 63, 0, d, a, b, &mac_flags),
                    GOC_SUCCESS);
          EXPECT_EQ(words[3][63], expected);
          EXPECT_EQ(mk_flags, exceptions);
          EXPECT_EQ(ak_flags, exceptions);
          EXPECT_EQ(mac_flags, exceptions);
        }
      }
      EXPECT_EQ(hash, goc_test::half_fma_exception_hashes[variant]) << variant << "/" << saturate;
    }
}

TEST(HalfFma, Wave64FormsDppHalfSelectorsAndAliases) {
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (auto sem : semantics)
        for (unsigned variant = 0; variant < 128; ++variant)
          for (unsigned route_index = 0; route_index < (op < 2 ? 8U : 1U); ++route_index)
            for (unsigned alias = 0; alias < 4; ++alias) {
              uint32_t mode = (variant * 65) & 8191;
              if (op == 1)
                mode &= ~(GOC_ALU_NEG_C | GOC_ALU_ABS_C | GOC_ALU_HIGH_C);
              if (op >= 2)
                mode &= GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D;
              uint64_t descriptor = route_index ? goc_test::dpp_modes[route_index - 1] : 0;
              if (descriptor & GOC_DPP16)
                descriptor = (descriptor & ~GOC_DPP_ROW_MASK) |
                             (uint64_t(variant & 15) << GOC_DPP_ROW_SHIFT);
              uint32_t words[4][66], before[4][66];
              for (unsigned reg = 0; reg < 4; ++reg)
                for (unsigned lane = 0; lane < 66; ++lane)
                  words[reg][lane] = values[(lane + 5 * reg) % 24] |
                                     (uint32_t(values[(lane * 7 + 3 * reg) % 24]) << 16);
              std::memcpy(before, words, sizeof words);
              uint32_t *d = words[alias] + 1;
              const uint32_t *a = words[0] + 1, *b = words[1] + 1, *c = words[2] + 1;
              uint64_t exec_mask = 0x87654321fedcba98ULL;
              bool saturate = variant & 1;
              ASSERT_EQ(wave64_call(op, cpu | sem | (saturate ? GOC_FP16_OVFL : 0), exec_mask,
                                    descriptor | mode, &d, &a, &b, &c, 0x3c01, nullptr),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 64; ++lane) {
                int source = int(lane);
                bool enabled = descriptor
                                   ? goc_test::dpp_source(descriptor, exec_mask, lane, source)
                                   : ((exec_mask >> lane) & 1);
                if (!enabled) {
                  EXPECT_EQ(d[lane], before[alias][lane + 1]);
                  continue;
                }
                uint32_t x = source < 0 ? 0 : before[0][source + 1];
                uint32_t y = before[1][lane + 1], z = before[op == 1 ? alias : 2][lane + 1];
                x = uint16_t(x >> (mode & GOC_ALU_HIGH_A ? 16 : 0));
                y = uint16_t(y >> (mode & GOC_ALU_HIGH_B ? 16 : 0));
                z = uint16_t(z >> (mode & (op == 1 ? GOC_ALU_HIGH_D : GOC_ALU_HIGH_C) ? 16 : 0));
                if (op == 2) {
                  z = y;
                  y = 0x3c01;
                }
                if (op == 3)
                  z = 0x3c01;
                uint16_t expected = goc_test::half_fma_reference::evaluate(
                    x, y, z, mode & ~(GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_C), saturate);
                check(d[lane], before[alias][lane + 1], expected, mode, sem != GOC_SEMANTICS_LOOSE);
              }
              for (unsigned reg = 0; reg < 4; ++reg) {
                EXPECT_EQ(words[reg][0], before[reg][0]);
                EXPECT_EQ(words[reg][65], before[reg][65]);
              }
            }
}

TEST(HalfFma, Wave64ReportingAccumulationAndAliasedUpperSource) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool alias : {false, true}) {
      uint32_t a[64] = {}, b[64] = {}, c[64] = {}, d[64] = {};
      // Lane 0 raises invalid; lane 62 overflows. Reporting into a[63] early
      // would change the upper input before it is read.
      a[0] = 0x7c01;
      b[0] = 0x3c00;
      a[62] = 0x7bff;
      b[62] = 0x4000;
      a[63] = b[63] = 0x3c00;
      uint32_t exceptions = 0x80000000;
      uint32_t *out = alias ? &a[63] : &exceptions;
      uint32_t original = *out;
      uint32_t *pd = d;
      const uint32_t *pa = a, *pb = b, *pc = c;
      ASSERT_EQ(
          goc_v_fma_f16_rdna4_wave64(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                     1ULL | (3ULL << 62), 0, &pd, &pa, &pb, &pc, out),
          GOC_SUCCESS);
      EXPECT_EQ(d[62], 0x7c00U);
      EXPECT_EQ(d[63], 0x3c00U);
      EXPECT_EQ(*out,
                original | GOC_EXCEPTION_INVALID | GOC_EXCEPTION_OVERFLOW | GOC_EXCEPTION_INEXACT);
    }
  for (unsigned op = 0; op < 4; ++op) {
    uint32_t exceptions = 0x80000000;
    EXPECT_EQ(wave64_call(op, GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, nullptr, nullptr, nullptr,
                          nullptr, 0, &exceptions),
              GOC_SUCCESS);
    EXPECT_EQ(wave64_call(op, GOC_SEMANTICS_EXACT_EMPIRICAL, UINT64_MAX, 1ULL << 36, nullptr,
                          nullptr, nullptr, nullptr, 0, &exceptions),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(exceptions, 0x80000000U);
  }
}

TEST(HalfFma, Wave64ReportingDppAndHostEnvironment) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned op = 0; op < 4; ++op)
        for (unsigned route_index = 0; route_index < (op < 2 ? 8U : 1U); ++route_index)
          for (unsigned rows = 0; rows < 16; ++rows) {
            uint64_t descriptor = route_index ? goc_test::dpp_modes[route_index - 1] : 0;
            if (descriptor & GOC_DPP16)
              descriptor = (descriptor & ~GOC_DPP_ROW_MASK) | (uint64_t(rows) << GOC_DPP_ROW_SHIFT);
            uint32_t a[64], b[64], c[64] = {}, d[64] = {};
            std::fill_n(a, 64, 0x7c01);
            std::fill_n(b, 64, 0x3c00);
            const uint32_t *pa = a, *pb = b, *pc = c;
            uint32_t *pd = d;
            uint64_t exec_mask = 0x87654321fedcba98ULL;
            uint32_t expected = 0x80000000, exceptions = expected;
            for (unsigned lane = 0; lane < 64; ++lane) {
              int source = int(lane);
              bool enabled = descriptor ? goc_test::dpp_source(descriptor, exec_mask, lane, source)
                                        : ((exec_mask >> lane) & 1);
              if (enabled && source >= 0)
                expected |= GOC_EXCEPTION_INVALID;
            }
            ASSERT_EQ(std::fesetround(rounding), 0);
            std::feclearexcept(FE_ALL_EXCEPT);
            std::feraiseexcept(FE_DIVBYZERO);
            ASSERT_EQ(wave64_call(op, cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                  exec_mask, descriptor, &pd, &pa, &pb, &pc, 0x3c00, &exceptions),
                      GOC_SUCCESS);
            EXPECT_EQ(exceptions, expected);
            EXPECT_EQ(std::fegetround(), rounding);
            EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
            exceptions = 0x80000000;
            ASSERT_EQ(wave64_call(op, cpu, exec_mask, descriptor, &pd, &pa, &pb, &pc, 0x3c00,
                                  &exceptions),
                      GOC_SUCCESS);
            EXPECT_EQ(exceptions, 0x80000000U);
          }
}

namespace {

int architecture_fma(bool rdna3, bool wide, unsigned form, uint64_t flags, uint64_t exec_mask,
                     uint32_t mode, uint32_t *const *d, const uint32_t *const *a,
                     const uint32_t *const *b, const uint32_t *const *c, uint16_t literal_b,
                     uint16_t literal_c, uint32_t *state = nullptr) {
  if (form == 0) {
    auto fn = rdna3 ? goc_v_fma_f16 : goc_v_fma_f16_rdna4;
    auto fn64 = rdna3 ? goc_v_fma_f16_wave64 : goc_v_fma_f16_rdna4_wave64;
    return wide ? fn64(flags, exec_mask, mode, d, a, b, c, state)
                : fn(flags, uint32_t(exec_mask), mode, d, a, b, c, state);
  }
  if (form == 1) {
    auto fn = rdna3 ? goc_v_fmac_f16 : goc_v_fmac_f16_rdna4;
    auto fn64 = rdna3 ? goc_v_fmac_f16_wave64 : goc_v_fmac_f16_rdna4_wave64;
    return wide ? fn64(flags, exec_mask, mode, d, a, b, state)
                : fn(flags, uint32_t(exec_mask), mode, d, a, b, state);
  }
  if (form == 2) {
    auto fn = rdna3 ? goc_v_fmamk_f16 : goc_v_fmamk_f16_rdna4;
    auto fn64 = rdna3 ? goc_v_fmamk_f16_wave64 : goc_v_fmamk_f16_rdna4_wave64;
    return wide ? fn64(flags, exec_mask, mode, d, a, literal_b, c, state)
                : fn(flags, uint32_t(exec_mask), mode, d, a, literal_b, c, state);
  }
  auto fn = rdna3 ? goc_v_fmaak_f16 : goc_v_fmaak_f16_rdna4;
  auto fn64 = rdna3 ? goc_v_fmaak_f16_wave64 : goc_v_fmaak_f16_rdna4_wave64;
  return wide ? fn64(flags, exec_mask, mode, d, a, b, literal_c, state)
              : fn(flags, uint32_t(exec_mask), mode, d, a, b, literal_c, state);
}

} // namespace

TEST(HalfFma, ArchitectureGuestModesAndLiteralOperands) {
  struct Case {
    uint16_t a, b, c, result[4];
  };

  // Results indexed by {flush_output, flush_input}, before IEEE quieting/OMOD.
  const Case cases[] = {
      {0x3c00, 0x4000, 0x3c00, {0x4200, 0x4200, 0x4200, 0x4200}},
      {1, 0x3c00, 0, {1, 0, 0, 0}},
      {0x3c00, 1, 0, {1, 0, 0, 0}},
      {0, 0x3c00, 1, {1, 0, 0, 0}},
      {0x0400, 0x3800, 0, {0x0200, 0x0200, 0, 0}},
      {0x8400, 0x3800, 0x8000, {0x8200, 0x8200, 0x8000, 0x8000}},
      {0x7c15, 0x3c00, 0, {0x7c15, 0x7c15, 0x7c15, 0x7c15}},
      {0x7e22, 0x7c15, 0, {0x7e22, 0x7e22, 0x7e22, 0x7e22}},
      {1, 0x7c00, 0x7e22, {0x7e22, 0xfe00, 0x7e22, 0xfe00}},
  };
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t semantics : {uint64_t(0), GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT})
      for (bool rdna3 : {false, true})
        for (bool wide : {false, true})
          for (unsigned form = 0; form < 4; ++form)
            for (bool high : {false, true})
              for (unsigned settings = 0; settings < 8; ++settings)
                for (unsigned omod = 0; omod < (form < 2 ? 4U : 1U); ++omod)
                  for (const auto &test : cases) {
                    uint64_t flags = cpu | semantics |
                                     ((settings & 1) ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                                     ((settings & 2) ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                                     ((settings & 4) ? GOC_FP_IEEE_MODE : 0);
                    uint32_t mode = omod * GOC_ALU_OMOD_2;
                    if (high)
                      mode |= GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D |
                              (form == 0 ? GOC_ALU_HIGH_C : 0);
                    auto word = [high](uint16_t value) {
                      return high ? (uint32_t(value) << 16) | 0xbeefU : 0xdead0000U | value;
                    };
                    uint32_t a[64], b[64], c[64], d[64];
                    std::fill_n(a, 64, word(test.a));
                    std::fill_n(b, 64, word(test.b));
                    std::fill_n(c, 64, word(test.c));
                    uint32_t before = form == 1 ? word(test.c) : 0xdeadbeefU;
                    std::fill_n(d, 64, before);
                    const uint32_t *pa = a, *pb = b, *pc = c;
                    uint32_t *pd = d;
                    uint64_t exec_mask = wide ? 0x55555555aaaaaaaaULL : 0xaaaaaaaaULL;
                    ASSERT_EQ(architecture_fma(rdna3, wide, form, flags, exec_mask, mode, &pd, &pa,
                                               &pb, &pc, test.b, test.c),
                              GOC_SUCCESS);
                    uint16_t expected = test.result[settings & 3];
                    if (nan(expected) && (!rdna3 || (settings & 4)))
                      expected |= 0x200;
                    unsigned scale = rdna3 && (!(settings & 2) || (settings & 4)) ? 0 : omod;
                    if (scale && !nan(expected)) {
                      if ((expected & 0x7fff) < 0x400)
                        expected = 0;
                      else if (scale == 3 && (expected & 0x7fff) < 0x800)
                        expected &= 0x8000;
                      else
                        expected =
                            uint16_t(int(expected) + (scale == 3 ? -1024 : int(scale) * 1024));
                    }
                    for (unsigned lane = 0; lane < 64; ++lane) {
                      if ((exec_mask >> lane) & 1)
                        check(d[lane], before, expected, mode, semantics != 0);
                      else
                        EXPECT_EQ(d[lane], before);
                    }
                  }
}

TEST(HalfFma, NewArchitecturePolicyPreservesHostStateAndRejectsUnverifiedReporting) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO})
    for (bool host_flush : {false, true}) {
      ASSERT_EQ(std::fesetround(rounding), 0);
#if defined(__x86_64__) || defined(_M_X64)
      unsigned host_mode = _mm_getcsr();
      _mm_setcsr(host_flush ? host_mode | 0x8040U : host_mode & ~0x8040U);
      host_mode = _mm_getcsr();
#else
      (void)host_flush;
#endif
      for (bool rdna3 : {false, true})
        for (bool wide : {false, true})
          for (unsigned form = 0; form < 4; ++form) {
            uint32_t a[64], b[64], c[64], d[64];
            std::fill_n(a, 64, 0x0400U);
            std::fill_n(b, 64, 0x3800U);
            std::fill_n(c, 64, 0U);
            std::fill_n(d, 64, 0U);
            const uint32_t *pa = a, *pb = b, *pc = c;
            uint32_t *pd = d;
            uint64_t flags = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                             GOC_FP_FLUSH_OUTPUT_DENORMALS;
            uint64_t exec_mask = wide ? UINT64_MAX : UINT32_MAX;
            ASSERT_EQ(architecture_fma(rdna3, wide, form, flags, exec_mask, 0, &pd, &pa, &pb, &pc,
                                       0x3800, 0),
                      GOC_SUCCESS);
            for (uint32_t value : d)
              EXPECT_EQ(value, 0U);
            uint32_t state = 0x80000055;
            std::fill_n(d, 64, 0xdeadbeefU);
            EXPECT_EQ(architecture_fma(rdna3, wide, form, flags, exec_mask, 0, &pd, &pa, &pb, &pc,
                                       0x3800, 0, &state),
                      GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
            EXPECT_EQ(state, 0x80000055U);
            for (uint32_t value : d)
              EXPECT_EQ(value, 0xdeadbeefU);
          }
      EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
      EXPECT_EQ(_mm_getcsr(), host_mode);
#endif
    }
}

TEST(HalfFma, ArchitectureIntegerOracleBoundaryCases) {
  const uint16_t inputs[] = {0,      0x8000, 1,      0x8001, 0x03ff, 0x0400, 0x0401, 0x3800,
                             0x3bff, 0x3c00, 0x3c01, 0xbc00, 0x7bff, 0x7c00, 0x7c15, 0xfe22};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t semantics : {uint64_t(0), GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT})
      for (bool rdna3 : {false, true})
        for (unsigned settings = 0; settings < 8; ++settings)
          for (uint32_t mode : {0U, GOC_ALU_OMOD_2, GOC_ALU_OMOD_HALF}) {
            uint64_t flags = cpu | semantics | ((settings & 1) ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                             ((settings & 2) ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                             ((settings & 4) ? GOC_FP_IEEE_MODE : 0);
            for (unsigned base = 0; base < 4096; base += 32) {
              uint32_t a[32], b[32], c[32], d[32];
              for (unsigned lane = 0; lane < 32; ++lane) {
                unsigned index = base + lane;
                a[lane] = inputs[index % 16];
                b[lane] = inputs[(index / 16) % 16];
                c[lane] = inputs[index / 256];
                d[lane] = 0xdead0000U;
              }
              const uint32_t *pa = a, *pb = b, *pc = c;
              uint32_t *pd = d;
              ASSERT_EQ(architecture_fma(rdna3, false, 0, flags, UINT32_MAX, mode, &pd, &pa, &pb,
                                         &pc, 0, 0),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 32; ++lane)
                check(d[lane], 0xdead0000U,
                      goc_test::half_fma_reference::evaluate(a[lane], b[lane], c[lane], mode, false,
                                                             flags, rdna3),
                      mode, semantics != 0);
            }
          }
}
