// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "exec_masks.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "half_fma_reference.h"
#include "literal_fma32_hardware.h"
#include "mixed_fma_reference.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {

const uint32_t known16 = GOC_ALU_HIGH_A | GOC_ALU_HIGH_B | GOC_ALU_HIGH_D;
const uint32_t special32[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff,
                              0x00800000, 0x3f800000, 0x3f800001, 0x3f7fffff, 0x3f000000,
                              0xbf000000, 0x40000000, 0xc0000000, 0xbf800000, 0x7f7fffff,
                              0xff7fffff, 0x7f800000, 0xff800000, 0x7f800001, 0xffc01234};
const uint16_t special16[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3c00,
                              0x3c01, 0x3bff, 0x3800, 0xb800, 0x4000, 0xc000, 0xbc00,
                              0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe12};

int call(bool half, bool multiply, uint64_t flags, uint32_t exec_mask, uint64_t mode,
         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b, uint32_t literal) {
  if (half)
    return multiply
               ? goc_v_fmamk_f16_rdna4(flags, exec_mask, mode, d, a, uint16_t(literal), b, nullptr)
               : goc_v_fmaak_f16_rdna4(flags, exec_mask, mode, d, a, b, uint16_t(literal), nullptr);
  return multiply ? goc_v_fmamk_f32_rdna4(flags, exec_mask, mode, d, a, literal, b, nullptr)
                  : goc_v_fmaak_f32_rdna4(flags, exec_mask, mode, d, a, b, literal, nullptr);
}

float as_float(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint32_t as_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

uint32_t reference(bool half, bool multiply, uint32_t a, uint32_t b, uint32_t literal,
                   uint32_t mode, bool saturate) {
  if (half) {
    a = uint16_t(a >> (mode & GOC_ALU_HIGH_A ? 16 : 0));
    b = uint16_t(b >> (mode & GOC_ALU_HIGH_B ? 16 : 0));
    return multiply ? goc_test::half_fma_reference::evaluate(a, literal, b, 0, saturate)
                    : goc_test::half_fma_reference::evaluate(a, b, literal, 0, saturate);
  }
  return goc_test::mixed_fma_reference::evaluate(0, a, multiply ? literal : b,
                                                 multiply ? b : literal, 0, 0, false);
}

void check(bool half, uint32_t actual, uint32_t before, uint32_t want, uint32_t mode, bool exact) {
  unsigned shift = half && (mode & GOC_ALU_HIGH_D) ? 16 : 0;
  uint32_t mask = half ? 0xffffU << shift : UINT32_MAX;
  EXPECT_EQ(actual & ~mask, before & ~mask);
  uint32_t got = (actual & mask) >> shift;
  uint32_t magnitude = half ? 0x7fff : 0x7fffffff, infinity = half ? 0x7c00 : 0x7f800000;
  if (!exact && (want & magnitude) > infinity) {
    EXPECT_GT(got & magnitude, infinity);
  } else {
    EXPECT_EQ(got, want);
  }
}

uint32_t selectors(unsigned bits) { return ((bits & 3) << 9) | ((bits & 4) ? GOC_ALU_HIGH_D : 0); }

void run(bool half, bool multiply, uint64_t flags, uint32_t exec_mask, uint64_t mode, int a, int b,
         int d, uint32_t literal, uint32_t (&words)[3][34]) {
  uint32_t before[3][34];
  std::memcpy(before, words, sizeof(before));
  uint32_t *p[] = {words[0] + 1, words[1] + 1, words[2] + 1};
  ASSERT_EQ(call(half, multiply, flags, exec_mask, mode, p + d, p + a, p + b, literal),
            GOC_SUCCESS);
  for (int reg = 0; reg < 3; ++reg)
    for (int lane = 0; lane < 34; ++lane) {
      if (reg == d && lane >= 1 && lane <= 32 && ((exec_mask >> (lane - 1)) & 1)) {
        check(half, words[reg][lane], before[reg][lane],
              reference(half, multiply, before[a][lane], before[b][lane], literal, mode,
                        flags & GOC_FP16_OVFL),
              mode, (flags & GOC_SEMANTICS_MASK) == GOC_SEMANTICS_EXACT_EMPIRICAL);
      } else {
        EXPECT_EQ(words[reg][lane], before[reg][lane]);
      }
    }
}

} // namespace

TEST(LiteralFma, AllHalfLiteralsAndRandomInputs) {
  std::mt19937 random(431);
  for (unsigned literal = 0; literal < 65536; ++literal) {
    uint32_t words[3][32], expected[2][32];
    uint32_t mode = selectors(literal % 8);
    bool saturate = literal & 8;
    for (int lane = 0; lane < 32; ++lane) {
      words[0][lane] = random();
      words[1][lane] = random();
      for (bool multiply : {false, true})
        expected[multiply][lane] =
            reference(true, multiply, words[0][lane], words[1][lane], literal, mode, saturate);
    }
    uint32_t *p[] = {words[0], words[1], words[2]};
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool exact : {false, true})
        for (bool multiply : {false, true}) {
          SCOPED_TRACE(::testing::Message()
                       << literal << '/' << cpu << '/' << exact << '/' << multiply);
          std::fill(words[2], words[2] + 32, 0xfacecafe);
          uint64_t flags = cpu | (saturate ? GOC_FP16_OVFL : 0) |
                           (exact ? GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT : 0);
          ASSERT_EQ(call(true, multiply, flags, UINT32_MAX, mode, p + 2, p, p + 1, literal),
                    GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            check(true, words[2][lane], 0xfacecafe, expected[multiply][lane], mode, exact);
        }
  }
}

TEST(LiteralFma, SpecialValuesSelectorsMasksAliasesAndOverflow) {
  for (bool half : {false, true})
    for (bool multiply : {false, true})
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (bool exact : {false, true})
          for (unsigned selection = 0; selection < (half ? 8u : 1u); ++selection)
            for (bool saturate : {false, true})
              for (auto exec_mask : exec_masks())
                for (int b : {0, 1})
                  for (int d = 0; d < 3; ++d) {
                    uint32_t mode = selectors(selection);
                    SCOPED_TRACE(::testing::Message()
                                 << half << '/' << multiply << '/' << cpu << '/' << exact << '/'
                                 << mode << '/' << exec_mask << '/' << b << '/' << d);
                    uint32_t words[3][34];
                    for (int reg = 0; reg < 3; ++reg) {
                      std::fill(words[reg], words[reg] + 34, 0xfacecafe);
                      for (int lane = 1; lane <= 32; ++lane) {
                        unsigned i = (lane + reg * 7) % 20;
                        words[reg][lane] =
                            half ? special16[i] | (uint32_t(special16[(i + 3) % 20]) << 16)
                                 : special32[i];
                      }
                    }
                    uint64_t flags = cpu | (saturate ? GOC_FP16_OVFL : 0) |
                                     (exact ? GOC_SEMANTICS_EXACT_EMPIRICAL : 0) |
                                     (exact ? GOC_SEMANTICS_STRICT : 0);
                    // Rotate literal categories across mask and alias cases.
                    unsigned index = unsigned(exec_mask + d + b + selection) % 20;
                    run(half, multiply, flags, exec_mask, mode, 0, b, d,
                        half ? special16[index] : special32[index], words);
                  }
}

TEST(LiteralFma, Fp32RandomTriplesAndLiteralFusedRounding) {
  std::mt19937 random(9341);
  for (bool multiply : {false, true})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool exact : {false, true})
        for (unsigned batch = 0; batch < 4096; ++batch) {
          uint32_t words[3][34];
          for (auto &reg : words)
            for (auto &word : reg)
              word = random();
          run(false, multiply,
              cpu | (exact ? GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT : 0), UINT32_MAX,
              0, 0, 1, batch % 3, random(), words);
        }

  struct Case {
    bool half, multiply;
    uint32_t a, b, literal, want;
  };

  const Case cases[] = {{false, false, 0x3f800001, 0x3f7ffffe, 0xbf800000, 0xa8800000},
                        {false, true, 0x3f800001, 0xbf800000, 0x3f7ffffe, 0xa8800000},
                        {true, false, 0x3c01, 0x3e00, 0x8001, 0x3e01},
                        {true, true, 0x3c01, 0x8001, 0x3e00, 0x3e01},
                        {true, false, 0x3c03, 0x3e00, 1, 0x3e05},
                        {true, true, 0x3c03, 1, 0x3e00, 0x3e05},
                        {true, false, 0, 0x7c00, 0x7c01, 0xfe00},
                        {true, true, 0, 0x7c01, 0x7c00, 0xfe00}};
  for (auto test : cases)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t words[3][32];
      uint32_t *p[] = {words[0], words[1], words[2]};
      std::fill(words[0], words[0] + 32, test.a);
      std::fill(words[1], words[1] + 32, test.b);
      std::fill(words[2], words[2] + 32, 0xfacecafe);
      ASSERT_EQ(call(test.half, test.multiply, cpu, UINT32_MAX, 0, p + 2, p, p + 1, test.literal),
                GOC_SUCCESS);
      for (auto word : words[2])
        check(test.half, word, 0xfacecafe, test.want, 0, false);
    }
}

TEST(LiteralFma, ExactHalfPayloadPriorityAndHostEnvironment) {
  struct Case {
    uint16_t a, b, literal, mk, ak;
  };

  const Case cases[] = {{0x7c01, 0xfc12, 0x7d23, 0x7e01, 0x7e01},
                        {0x3c00, 0xfc12, 0x7d23, 0x7f23, 0xfe12},
                        {0, 0x7c00, 0x7c01, 0x7e01, 0xfe00},
                        {0, 0x7c01, 0x7c00, 0xfe00, 0x7e01},
                        {0x3c01, 0x3e00, 0x8001, 0x3e00, 0x3e01}};
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO})
      for (bool multiply : {false, true})
        for (unsigned selection = 0; selection < 8; ++selection)
          for (auto test : cases) {
            std::fesetround(rounding);
            std::feclearexcept(FE_ALL_EXCEPT);
            std::feraiseexcept(FE_DIVBYZERO);
            uint32_t words[3][32];
            uint32_t *p[] = {words[0], words[1], words[2]};
            std::fill(words[0], words[0] + 32, uint32_t(test.a) * 0x10001);
            std::fill(words[1], words[1] + 32, uint32_t(test.b) * 0x10001);
            std::fill(words[2], words[2] + 32, 0xfacecafe);
            uint32_t mode = selectors(selection);
            EXPECT_EQ(call(true, multiply,
                           cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX,
                           mode, p + 2, p, p + 1, test.literal),
                      GOC_SUCCESS);
            for (auto word : words[2])
              check(true, word, 0xfacecafe, multiply ? test.mk : test.ak, mode, true);
            EXPECT_EQ(std::fegetround(), rounding);
            EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
          }
}

TEST(LiteralFma, ValidationAndZeroMasks) {
  for (bool half : {false, true})
    for (bool multiply : {false, true}) {
      uint32_t words[32];
      std::fill(words, words + 32, 0xfacecafe);
      auto p = words;
      for (uint32_t exec_mask : {0U, UINT32_MAX}) {
        for (unsigned bit = 0; bit < 32; ++bit) {
          if ((1U << bit) & ~(half ? known16 : 0)) {
            EXPECT_EQ(call(half, multiply, 0, exec_mask, 1U << bit, &p, &p, &p, 0),
                      GOC_ERROR_INVALID_FLAGS);
          }
        }
        EXPECT_EQ(call(half, multiply, 1ULL << 63, exec_mask, 0, &p, &p, &p, 0),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(
            call(half, multiply, (2ULL << 16) | GOC_SEMANTICS_STRICT, exec_mask, 0, &p, &p, &p, 0),
            GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
      for (auto word : words)
        EXPECT_EQ(word, 0xfacecafe);
      EXPECT_EQ(call(half, multiply, 0, 0, 0, nullptr, nullptr, nullptr, 0), GOC_SUCCESS);
      EXPECT_EQ(call(half, multiply, 0, 0U, 0, nullptr, nullptr, nullptr, 0), GOC_SUCCESS);
      EXPECT_EQ(call(half, multiply, 2ULL << 16, UINT32_MAX, 0, &p, &p, &p, 0), GOC_SUCCESS);
    }
}

TEST(LiteralFma, Fp32ArchitectureGuestModes) {
  auto flush = [](uint32_t bits) { return (bits & 0x7f800000) ? bits : bits & 0x80000000; };
  for (bool exact : {false, true})
    for (bool rdna3 : {false, true})
      for (bool wide : {false, true})
        for (bool multiply : {false, true})
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
            for (unsigned policy = 0; policy < 8; ++policy)
              for (uint32_t literal : special32)
                for (int alias = 0; alias < 3; ++alias) {
                  uint64_t flags =
                      (exact ? GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT : 0) | cpu |
                      (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                      (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                      (policy & 4 ? GOC_FP_IEEE_MODE : 0);
                  uint32_t words[3][64], before[3][64];
                  for (int reg = 0; reg < 3; ++reg)
                    for (int lane = 0; lane < 64; ++lane)
                      words[reg][lane] = special32[(lane * (reg + 1) + 7 * reg) % 20];
                  std::memcpy(before, words, sizeof(words));
                  const uint32_t *a = words[0], *b = words[1];
                  uint32_t *d = words[alias], state = 0x12345678;
                  uint64_t exec_mask = 0xa7109efae12ce36dULL;
                  auto mk32 = rdna3 ? goc_v_fmamk_f32 : goc_v_fmamk_f32_rdna4;
                  auto ak32 = rdna3 ? goc_v_fmaak_f32 : goc_v_fmaak_f32_rdna4;
                  auto mk64 = rdna3 ? goc_v_fmamk_f32_wave64 : goc_v_fmamk_f32_rdna4_wave64;
                  auto ak64 = rdna3 ? goc_v_fmaak_f32_wave64 : goc_v_fmaak_f32_rdna4_wave64;
                  auto invoke = [&](uint64_t f, bool report = true) {
                    uint32_t *state_output = report ? &state : nullptr;
                    if (multiply)
                      return wide ? mk64(f, exec_mask, 0, &d, &a, literal, &b, state_output)
                                  : mk32(f, uint32_t(exec_mask), 0, &d, &a, literal, &b,
                                         state_output);
                    return wide
                               ? ak64(f, exec_mask, 0, &d, &a, &b, literal, state_output)
                               : ak32(f, uint32_t(exec_mask), 0, &d, &a, &b, literal, state_output);
                  };
                  ASSERT_EQ(invoke(flags, !exact), GOC_SUCCESS);
                  EXPECT_EQ(state, 0x12345678U);
                  for (int lane = 0; lane < 64; ++lane) {
                    if (lane >= (wide ? 64 : 32) || !((exec_mask >> lane) & 1)) {
                      EXPECT_EQ(d[lane], before[alias][lane]);
                      continue;
                    }
                    float x = as_float(policy & 1 ? flush(before[0][lane]) : before[0][lane]);
                    float y = as_float(policy & 1 ? flush(before[1][lane]) : before[1][lane]);
                    float k = as_float(policy & 1 ? flush(literal) : literal);
                    uint32_t expected = as_bits(multiply ? std::fma(x, k, y) : std::fma(x, y, k));
                    if (policy & 2)
                      expected = flush(expected);
                    if (exact)
                      expected = goc_test::mixed_fma_reference::evaluate(
                          0, before[0][lane], multiply ? literal : before[1][lane],
                          multiply ? before[1][lane] : literal, 0, 0, false, false, flags, rdna3);
                    if (!exact && (expected & 0x7fffffff) > 0x7f800000)
                      EXPECT_GT(d[lane] & 0x7fffffff, 0x7f800000U);
                    else
                      EXPECT_EQ(d[lane], expected);
                  }
                  std::memcpy(before, words, sizeof(words));
                  EXPECT_EQ(invoke(flags | GOC_SEMANTICS_EXACT_EMPIRICAL),
                            GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
                  EXPECT_EQ(std::memcmp(before, words, sizeof(words)), 0);
                  EXPECT_EQ(state, 0x12345678U);
                }
}

TEST(LiteralFma, Fp32HardwareCorpus) {
  const uint32_t edge[] = {0,          0x80000000, 1,          0x80000001, 0x7fffff,   0x800000,
                           0x3f800000, 0xbf800000, 0x40000000, 0x7f7fffff, 0x7effffff, 0x7f800000,
                           0xff800000, 0x7f800001, 0xffc00003, 0x1000000};
  for (bool multiply : {true, false})
    for (bool wide : {false, true})
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned policy = 0; policy < 4; ++policy) {
          uint64_t flags = cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                           (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                           (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0);
          uint64_t hash = goc_test::capture_hash_seed;
          unsigned width = wide ? 64 : 32;
          for (unsigned base = 0; base < 8192; base += width) {
            uint32_t input[3][64], d[64];
            for (unsigned lane = 0; lane < width; ++lane) {
              unsigned i = base + lane;
              uint64_t state = uint64_t(i) * 0x9e3779b97f4a7c15ULL;
              for (unsigned operand = 0; operand < 3; ++operand) {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                input[operand][lane] = i < 4096 ? edge[(i >> (8 - operand * 4)) & 15]
                                                : uint32_t(state * 0x2545f4914f6cdd1dULL);
              }
            }
            const uint32_t *a = input[0], *b = input[multiply ? 2 : 1];
            uint32_t *dest = d;
            int result =
                wide ? (multiply ? goc_v_fmamk_f32_rdna4_wave64(flags, UINT64_MAX, 0, &dest, &a,
                                                                0x3f800001, &b, nullptr)
                                 : goc_v_fmaak_f32_rdna4_wave64(flags, UINT64_MAX, 0, &dest, &a, &b,
                                                                0x3f800001, nullptr))
                     : call(false, multiply, flags, UINT32_MAX, 0, &dest, &a, &b, 0x3f800001);
            ASSERT_EQ(result, GOC_SUCCESS);
            for (unsigned lane = 0; lane < width; ++lane)
              hash = goc_test::capture_hash_word(hash, d[lane]);
          }
          EXPECT_EQ(hash, goc_test::literal_fma32_hashes[multiply ? 0 : 1][policy])
              << multiply << '/' << wide << '/' << cpu << '/' << policy;
        }
}

TEST(LiteralFma, ExactFp32HostEnvironment) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO})
    for (bool flush : {false, true})
      for (bool multiply : {false, true})
        for (uint32_t literal : special32)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            ASSERT_EQ(std::fesetround(rounding), 0);
            std::feclearexcept(FE_ALL_EXCEPT);
            std::feraiseexcept(FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
            _mm_setcsr((_mm_getcsr() & ~0x8040U) | (flush ? 0x8040U : 0));
            unsigned controls = _mm_getcsr();
#endif
            uint32_t words[3][34];
            for (unsigned reg = 0; reg < 3; ++reg)
              for (unsigned lane = 0; lane < 34; ++lane)
                words[reg][lane] = special32[(lane + reg * 7) % 20];
            run(false, multiply, cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                UINT32_MAX, 0, 0, 1, 2, literal, words);
            EXPECT_EQ(std::fegetround(), rounding);
            EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
            EXPECT_EQ(_mm_getcsr(), controls);
#endif
          }
}
