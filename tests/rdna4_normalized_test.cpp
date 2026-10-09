// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_normalized_hardware.h"
#include "rdna4_normalized_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_cvt_pk_norm_i16_f32);
const Fn functions[] = {
    goc_rdna4_v_cvt_pk_norm_i16_f32,
    goc_rdna4_v_cvt_pk_norm_u16_f32,
    goc_rdna4_v_cvt_pk_norm_i16_f16,
    goc_rdna4_v_cvt_pk_norm_u16_f16,
    [](uint64_t f, uint32_t m, uint64_t i, uint32_t *const *d, const uint32_t *const *a,
       const uint32_t *const *) { return goc_rdna4_v_cvt_norm_i16_f16(f, m, i, d, a); },
    [](uint64_t f, uint32_t m, uint64_t i, uint32_t *const *d, const uint32_t *const *a,
       const uint32_t *const *) { return goc_rdna4_v_cvt_norm_u16_f16(f, m, i, d, a); }};

::testing::AssertionResult check(unsigned op, uint64_t flags, uint32_t mode, const uint32_t av[32],
                                 const uint32_t bv[32]) {
  uint32_t output[32];
  std::fill_n(output, 32, 0xa5a5a5a5);
  const uint32_t *a[] = {av}, *b[] = {bv};
  uint32_t *d[] = {output};
  int status = functions[op](flags, UINT32_MAX, mode, d, a, b);
  if (status != GOC_SUCCESS)
    return ::testing::AssertionFailure() << "status " << status;
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t expected = goc_test::normalized_reference(op, av[lane], bv[lane], 0xa5a5a5a5, mode);
    if (output[lane] != expected)
      return ::testing::AssertionFailure()
             << op << "/" << flags << "/" << mode << "/" << lane << std::hex << " A " << av[lane]
             << " B " << bv[lane] << " expected " << expected << " actual " << output[lane];
  }
  return ::testing::AssertionSuccess();
}

} // namespace

TEST(Normalized, HardwareCapturedTiesSaturationAndModifiers) {
  for (const auto &capture : goc_test::normalized_captures)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned column = 0; column < 20; ++column) {
        unsigned op = column < 18 ? column / 3 : column - 14;
        unsigned variant = column < 18 ? column % 3 : 3;
        uint32_t mode = variant == 1 ? GOC_ALU_CLAMP
                        : variant == 2
                            ? GOC_ALU_ABS_A | GOC_ALU_NEG_A | (op < 4 ? GOC_ALU_NEG_B : 0)
                        : variant == 3 ? GOC_ALU_CLAMP | GOC_ALU_OMOD_2
                                       : 0;
        uint32_t av[32], bv[32], output[32];
        uint32_t raw = op < 2 ? capture.source_float : capture.source_half;
        std::fill_n(av, 32, raw);
        std::fill_n(bv, 32, raw ^ (op < 2 ? 0x80000000 : 0x8000));
        std::fill_n(output, 32, 0xa5a5a5a5);
        const uint32_t *a[] = {av}, *b[] = {bv};
        uint32_t *d[] = {output};
        ASSERT_EQ(functions[op](cpu, UINT32_MAX, mode, d, a, b), GOC_SUCCESS);
        for (auto word : output)
          ASSERT_EQ(word, capture.expected[column])
              << op << "/" << column << "/" << cpu << std::hex << "/" << capture.source_float;
      }
}

TEST(Normalized, EveryHalfEncoding) {
  for (unsigned op = 2; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant : {0u, goc_test::normalized_modes(op) - 1})
        for (unsigned start = 0; start < 65536; start += 32) {
          uint32_t av[32], bv[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            av[lane] = (start + lane) | ((65535 - start - lane) << 16);
            bv[lane] = ~av[lane];
          }
          ASSERT_TRUE(check(op, cpu, goc_test::normalized_mode(op, variant), av, bv));
        }
}

TEST(Normalized, EveryIntegerRoundingBoundaryAndAdjacentFloat) {
  for (unsigned op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      unsigned scale = op ? 65535 : 32767;
      for (unsigned start = 0; start < scale * 3; start += 32) {
        uint32_t av[32], bv[32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          unsigned i = (start + lane) % (scale * 3);
          float midpoint = float((double(i / 3) + 0.5) / scale);
          uint32_t bits;
          std::memcpy(&bits, &midpoint, sizeof(bits));
          av[lane] = bits + i % 3 - 1;
          bv[lane] = av[lane] ^ 0x80000000;
        }
        ASSERT_TRUE(check(op, cpu, 0, av, bv));
      }
    }
}

TEST(Normalized, EveryModifierAndRandomFullWords) {
  for (unsigned op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < goc_test::normalized_modes(op); ++variant) {
        std::mt19937 random(8161);
        for (unsigned batch = 0; batch < 32; ++batch) {
          uint32_t av[32], bv[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            av[lane] = random();
            bv[lane] = random();
          }
          ASSERT_TRUE(check(op, cpu | ((batch & 1) ? GOC_FP16_OVFL : 0),
                            goc_test::normalized_mode(op, variant), av, bv));
        }
      }
}

TEST(Normalized, EveryModifierMaskAndWholeRegisterAlias) {
  std::mt19937 random(8147);
  for (unsigned op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < goc_test::normalized_modes(op); ++variant)
        for (uint32_t mask : rdna4_exec_masks())
          for (unsigned breg = 0; breg < (op < 4 ? 2u : 1u); ++breg)
            for (unsigned dreg = 0; dreg < 3; ++dreg) {
              uint32_t storage[3][34], expected[3][34];
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned word = 0; word < 34; ++word)
                  storage[reg][word] = expected[reg][word] = random();
              uint32_t mode = goc_test::normalized_mode(op, variant);
              for (unsigned lane = 0; lane < 32; ++lane)
                if ((mask >> lane) & 1)
                  expected[dreg][lane + 1] = goc_test::normalized_reference(
                      op, storage[0][lane + 1], storage[breg][lane + 1], storage[dreg][lane + 1],
                      mode);
              const uint32_t *a[] = {storage[0] + 1}, *b[] = {storage[breg] + 1};
              uint32_t *d[] = {storage[dreg] + 1};
              ASSERT_EQ(functions[op](cpu, mask, mode, d, a, b), GOC_SUCCESS);
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned word = 0; word < 34; ++word)
                  ASSERT_EQ(storage[reg][word], expected[reg][word])
                      << op << "/" << cpu << "/" << variant;
            }
}

TEST(Normalized, HostRoundingAndFalseMidpoints) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    EXPECT_EQ(std::fesetround(rounding), 0);
    for (unsigned op = 0; op < 6; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (const auto &capture : goc_test::normalized_captures) {
          uint32_t av[32], bv[32];
          uint32_t raw = op < 2 ? capture.source_float : capture.source_half;
          std::fill_n(av, 32, raw);
          std::fill_n(bv, 32, raw ^ (op < 2 ? 0x80000000 : 0x8000));
          EXPECT_TRUE(check(op, cpu, 0, av, bv));
          EXPECT_EQ(std::fegetround(), rounding);
        }
  }
}

TEST(Normalized, ValidationAndSemanticFallback) {
  for (unsigned op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[32] = {}, output[32];
      std::fill_n(output, 32, 0xdeadbeef);
      const uint32_t *a[] = {input};
      uint32_t *d[] = {output};
      uint32_t known = goc_test::normalized_mode(op, goc_test::normalized_modes(op) - 1);
      for (unsigned bit = 0; bit < 32; ++bit)
        if (!(known & (1u << bit))) {
          EXPECT_EQ(functions[op](cpu, UINT32_MAX, 1u << bit, d, a, a), GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(functions[op](cpu, 0, 1u << bit, nullptr, nullptr, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
        }
      EXPECT_EQ(functions[op](cpu | (UINT64_C(1) << 63), UINT32_MAX, 0, d, a, a),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                              UINT32_MAX, 0, d, a, a),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
      for (auto word : output)
        EXPECT_EQ(word, 0xdeadbeefu);
      EXPECT_EQ(functions[op](cpu, UINT32_C(0), 0, nullptr, nullptr, nullptr), GOC_SUCCESS);
      for (unsigned sem = 0; sem < 4; ++sem)
        ASSERT_TRUE(check(op, cpu | (uint64_t(sem) << 16), 0, input, input));
    }
}

TEST(Normalized, DppMasksAndWholeRegisterAliases) {
  std::mt19937 random(8147);
  for (auto descriptor : goc_test::dpp_modes)
    for (unsigned op = 0; op < 6; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned variant :
             {0u, goc_test::normalized_modes(op) / 2, goc_test::normalized_modes(op) - 1})
          for (uint32_t mask : rdna4_exec_masks())
            for (unsigned breg = 0; breg < (op < 4 ? 2u : 1u); ++breg)
              for (unsigned dreg = 0; dreg < 3; ++dreg) {
                uint32_t storage[3][34], expected[3][34];
                for (unsigned reg = 0; reg < 3; ++reg)
                  for (unsigned word = 0; word < 34; ++word)
                    storage[reg][word] = expected[reg][word] = random();
                uint64_t mode = descriptor | goc_test::normalized_mode(op, variant);
                for (unsigned lane = 0; lane < 32; ++lane) {
                  int source = 0;
                  if (goc_test::dpp_source(mode, mask, lane, source))
                    expected[dreg][lane + 1] = goc_test::normalized_reference(
                        op, source < 0 ? 0 : storage[0][source + 1], storage[breg][lane + 1],
                        storage[dreg][lane + 1], uint32_t(mode));
                }
                const uint32_t *a[] = {storage[0] + 1}, *b[] = {storage[breg] + 1};
                uint32_t *d[] = {storage[dreg] + 1};
                ASSERT_EQ(functions[op](cpu, mask, mode, d, a, b), GOC_SUCCESS);
                for (unsigned reg = 0; reg < 3; ++reg)
                  for (unsigned word = 0; word < 34; ++word)
                    ASSERT_EQ(storage[reg][word], expected[reg][word])
                        << op << "/" << cpu << "/" << variant;
              }
}

TEST(Normalized, DppEveryModifier) {
  std::mt19937 random(9431);
  for (unsigned op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (auto descriptor : goc_test::dpp_modes)
        for (unsigned variant = 0; variant < goc_test::normalized_modes(op); ++variant) {
          uint32_t av[32], bv[32], original[32], output[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            av[lane] = random();
            bv[lane] = random();
            output[lane] = original[lane] = random();
          }
          auto mode = descriptor | goc_test::normalized_mode(op, variant);
          const uint32_t *a[] = {av}, *b[] = {bv};
          uint32_t *d[] = {output};
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, mode, d, a, b), GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            int source = 0;
            uint32_t expected = original[lane];
            if (goc_test::dpp_source(mode, UINT32_MAX, lane, source))
              expected = goc_test::normalized_reference(op, source < 0 ? 0 : av[source], bv[lane],
                                                        original[lane], uint32_t(mode));
            ASSERT_EQ(output[lane], expected) << op << "/" << cpu << "/" << mode;
          }
        }
}

TEST(Normalized, DppValidation) {
  for (auto fn : functions)
    for (auto descriptor : goc_test::dpp_modes) {
      EXPECT_EQ(fn(0, 0, descriptor, nullptr, nullptr, nullptr), GOC_SUCCESS);
      EXPECT_EQ(fn(0, 0, descriptor | (UINT64_C(1) << 36), nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
    }
}

// RX 9070: six forms, source/destination selectors, modifiers and eight EXEC masks.
TEST(Normalized, DppHardwareCorpus) {
  const uint32_t values[] = {0x3800b800, 0x3bffbc00, 0x00018001, 0x7e00fc00,
                             0x3f000000, 0xbf000000, 0x3f7fffff, 0x7f800001};
  const uint32_t masks[] = {0xffffffff, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  const uint32_t packed_modes[] = {0, 1, 8, 9, 2, 16, 27, 283, 512, 1024, 1536, 1819};
  const uint32_t unary_modes[] = {0, 1, 8, 9, 256, 64, 128, 192, 512, 4096, 4608, 5065};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (auto mask : masks)
      for (unsigned op = 0; op < 6; ++op)
        for (auto descriptor : goc_test::dpp_modes)
          for (unsigned variant = 0; variant < (op < 2 ? 8u : 12u); ++variant) {
            uint32_t av[32], bv[32], output[32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              av[lane] = values[lane % 8];
              bv[lane] = values[(lane + 3) % 8];
              output[lane] = 0xdead0000u + lane;
            }
            const uint32_t *a[] = {av}, *b[] = {bv};
            uint32_t *d[] = {output};
            auto mode = descriptor | (op < 4 ? packed_modes[variant] : unary_modes[variant]);
            ASSERT_EQ(functions[op](cpu, mask, mode, d, a, b), GOC_SUCCESS);
            for (auto word : output)
              hash = (hash ^ word) * UINT64_C(1099511628211);
          }
    EXPECT_EQ(hash, UINT64_C(0x711fcd0f84aa5615)) << cpu;
  }
}
