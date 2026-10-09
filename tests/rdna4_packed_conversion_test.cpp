// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_conversion64_reference.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_packed_conversion_hardware.h"
#include "rdna4_packed_conversion_reference.h"

#include <algorithm>
#include <cfenv>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_cvt_pk_rtz_f16_f32);
const Fn functions[] = {goc_rdna4_v_cvt_pk_rtz_f16_f32, goc_rdna4_v_cvt_pk_i16_f32,
                        goc_rdna4_v_cvt_pk_u16_f32};

::testing::AssertionResult check(unsigned op, uint64_t flags, uint32_t mode, const uint32_t av[32],
                                 const uint32_t bv[32]) {
  uint32_t output[32];
  const uint32_t *a[] = {av}, *b[] = {bv};
  uint32_t *d[] = {output};
  int status = functions[op](flags, UINT32_MAX, mode, d, a, b);
  if (status != GOC_SUCCESS)
    return ::testing::AssertionFailure() << "status " << status;
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t expected = goc_test::packed_conversion_reference(op, av[lane], bv[lane], mode);
    if (output[lane] != expected)
      return ::testing::AssertionFailure()
             << op << "/" << flags << "/" << mode << "/" << lane << std::hex << " A " << av[lane]
             << " B " << bv[lane] << " expected " << expected << " actual " << output[lane];
  }
  return ::testing::AssertionSuccess();
}

} // namespace

TEST(PackedConversion, HardwareCapturedRoundingSaturationAndModifiers) {
  const unsigned ops[] = {0, 0, 1, 1, 2, 2, 0, 1, 2};
  const uint32_t modes[] = {0,
                            GOC_ALU_CLAMP,
                            0,
                            GOC_ALU_CLAMP,
                            0,
                            GOC_ALU_CLAMP,
                            GOC_ALU_OMOD_2,
                            GOC_ALU_ABS_A | GOC_ALU_NEG_A | GOC_ALU_NEG_B,
                            GOC_ALU_ABS_A | GOC_ALU_NEG_B};
  for (const auto &capture : goc_test::packed_conversion_captures)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool sat : {false, true})
        for (unsigned column = 0; column < 9; ++column) {
          uint32_t av[32], bv[32], output[32];
          std::fill_n(av, 32, capture.source);
          std::fill_n(bv, 32, capture.source ^ 0x80000000);
          const uint32_t *a[] = {av}, *b[] = {bv};
          uint32_t *d[] = {output};
          ASSERT_EQ(functions[ops[column]](cpu | (sat ? GOC_FP16_OVFL : 0), UINT32_MAX,
                                           modes[column], d, a, b),
                    GOC_SUCCESS);
          for (auto word : output)
            ASSERT_EQ(word, capture.expected[column])
                << column << "/" << cpu << std::hex << "/" << capture.source;
        }
}

TEST(PackedConversion, EveryExponentAndSourceModifier) {
  const uint32_t fractions[] = {0,      1,        0xfff,    0x1000,   0x1fff,
                                0x2000, 0x3fffff, 0x400000, 0x7ffffe, 0x7fffff};
  for (unsigned op = 0; op < 3; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < goc_test::packed_conversion_modes(op); ++variant)
        for (unsigned start = 0; start < 512; start += 32) {
          uint32_t av[32], bv[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            unsigned i = start + lane;
            av[lane] = ((i / 2) << 23) | ((i & 1) << 31) | fractions[(lane + variant) % 10];
            bv[lane] = (av[lane] ^ 0x80000000) ^ 0x007fffff;
          }
          ASSERT_TRUE(check(op, cpu, goc_test::packed_conversion_mode(op, variant), av, bv));
        }
}

TEST(PackedConversion, EveryHalfBoundaryAndAdjacentFloat) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool sat : {false, true})
      for (unsigned start = 0; start < 0x7c00 * 3; start += 32) {
        uint32_t av[32], bv[32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          unsigned index = (start + lane) % (0x7c00 * 3), half = index / 3;
          uint32_t value = uint32_t(goc_test::conversion_reencode(half, 10, 15, 23, 127));
          av[lane] = value + (value == 0 && index % 3 == 0 ? 0 : index % 3 - 1);
          bv[lane] = av[lane] ^ 0x80000000;
        }
        ASSERT_TRUE(check(0, cpu | (sat ? GOC_FP16_OVFL : 0), 0, av, bv));
      }
}

TEST(PackedConversion, RandomWordsAndEveryModifier) {
  for (unsigned op = 0; op < 3; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < goc_test::packed_conversion_modes(op); ++variant) {
        std::mt19937 random(6521);
        for (unsigned batch = 0; batch < 32; ++batch) {
          uint32_t av[32], bv[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            av[lane] = random();
            bv[lane] = random();
          }
          ASSERT_TRUE(check(op, cpu, goc_test::packed_conversion_mode(op, variant), av, bv));
        }
      }
}

TEST(PackedConversion, EveryModifierMaskAndWholeRegisterAlias) {
  std::mt19937 random(1447);
  for (unsigned op = 0; op < 3; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned variant = 0; variant < goc_test::packed_conversion_modes(op); ++variant)
        for (uint32_t mask : rdna4_exec_masks())
          for (unsigned breg = 0; breg < 2; ++breg)
            for (unsigned dreg = 0; dreg < 3; ++dreg) {
              uint32_t storage[3][34], expected[3][34];
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned word = 0; word < 34; ++word)
                  storage[reg][word] = expected[reg][word] = random();
              uint32_t mode = goc_test::packed_conversion_mode(op, variant);
              for (unsigned lane = 0; lane < 32; ++lane)
                if ((mask >> lane) & 1)
                  expected[dreg][lane + 1] = goc_test::packed_conversion_reference(
                      op, storage[0][lane + 1], storage[breg][lane + 1], mode);
              const uint32_t *a[] = {storage[0] + 1}, *b[] = {storage[breg] + 1};
              uint32_t *d[] = {storage[dreg] + 1};
              ASSERT_EQ(functions[op](cpu, mask, mode, d, a, b), GOC_SUCCESS);
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned word = 0; word < 34; ++word)
                  ASSERT_EQ(storage[reg][word], expected[reg][word])
                      << op << "/" << cpu << "/" << variant;
            }
}

TEST(PackedConversion, HostRoundingDoesNotAffectResults) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  const uint32_t values[] = {0x3fc00000, 0xbfc00000, 0x3f801fff, 0xbf801fff,
                             0x477fff00, 0x7f800001, 0xff800000, 0x33800000};
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    EXPECT_EQ(std::fesetround(rounding), 0);
    for (unsigned op = 0; op < 3; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
        uint32_t av[32], bv[32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          av[lane] = values[lane % 8];
          bv[lane] = values[(lane + 3) % 8];
        }
        EXPECT_TRUE(check(op, cpu, 0, av, bv));
        EXPECT_EQ(std::fegetround(), rounding);
      }
  }
}

TEST(PackedConversion, ValidationAndSemanticFallback) {
  for (unsigned op = 0; op < 3; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[32] = {}, output[32];
      std::fill_n(output, 32, 0xdeadbeef);
      const uint32_t *a[] = {input};
      uint32_t *d[] = {output};
      uint32_t known =
          goc_test::packed_conversion_mode(op, goc_test::packed_conversion_modes(op) - 1);
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

TEST(PackedConversion, DppMasksAliasesAndGuards) {
  std::mt19937 random(1447);
  for (auto descriptor : goc_test::dpp_modes)
    for (unsigned op = 0; op < 3; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned variant : {0u, goc_test::packed_conversion_modes(op) / 2,
                                 goc_test::packed_conversion_modes(op) - 1})
          for (uint32_t mask : rdna4_exec_masks())
            for (unsigned breg = 0; breg < 2; ++breg)
              for (unsigned dreg = 0; dreg < 3; ++dreg) {
                uint32_t storage[3][34], expected[3][34];
                for (unsigned reg = 0; reg < 3; ++reg)
                  for (unsigned word = 0; word < 34; ++word)
                    storage[reg][word] = expected[reg][word] = random();
                uint64_t mode = descriptor | goc_test::packed_conversion_mode(op, variant);
                for (unsigned lane = 0; lane < 32; ++lane) {
                  int source = 0;
                  if (goc_test::dpp_source(mode, mask, lane, source))
                    expected[dreg][lane + 1] = goc_test::packed_conversion_reference(
                        op, source < 0 ? 0 : storage[0][source + 1], storage[breg][lane + 1],
                        uint32_t(mode));
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

// RX 9070: all modifiers for all three forms, seven descriptors and eight masks.
// Canonicalize FP16 NaNs only; integer results retain all bits.
TEST(PackedConversion, DppHardwareCorpus) {
  const uint32_t values[] = {0x387fffff, 0x80000001, 0x477ff000, 0xc7000080,
                             0x7f800001, 0xff800000, 0x3fc00000, 0x47800000};
  const uint32_t masks[] = {0xffffffff, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (auto mask : masks)
      for (unsigned op = 0; op < 3; ++op)
        for (auto descriptor : goc_test::dpp_modes)
          for (unsigned variant = 0; variant < goc_test::packed_conversion_modes(op); ++variant) {
            uint32_t av[32], bv[32], output[32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              av[lane] = values[lane % 8];
              bv[lane] = values[(lane + 3) % 8];
              output[lane] = 0xdead0000u + lane;
            }
            const uint32_t *a[] = {av}, *b[] = {bv};
            uint32_t *d[] = {output};
            ASSERT_EQ(functions[op](cpu, mask,
                                    descriptor | goc_test::packed_conversion_mode(op, variant), d,
                                    a, b),
                      GOC_SUCCESS);
            for (auto word : output) {
              if (op == 0) {
                uint32_t low = word & 65535, high = word >> 16;
                if ((low & 0x7fff) > 0x7c00)
                  low = 0x7e00;
                if ((high & 0x7fff) > 0x7c00)
                  high = 0x7e00;
                word = low | (high << 16);
              }
              hash = (hash ^ word) * UINT64_C(1099511628211);
            }
          }
    EXPECT_EQ(hash, UINT64_C(0xa70566992e7f5ce5)) << cpu;
  }
}

TEST(PackedConversion, DppValidation) {
  for (auto fn : functions)
    for (auto descriptor : goc_test::dpp_modes) {
      EXPECT_EQ(fn(0, 0, descriptor, nullptr, nullptr, nullptr), GOC_SUCCESS);
      for (auto invalid : {UINT64_C(1) << 36, uint64_t(GOC_ALU_HIGH_A)})
        EXPECT_EQ(fn(0, 0, descriptor | invalid, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
    }
}
