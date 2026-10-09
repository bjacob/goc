// SPDX-License-Identifier: MIT

#include "exec_masks.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "packed_mad_reference.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_v_pk_mad_i16);
const Fn functions[] = {goc_v_pk_mad_u16, goc_v_pk_mad_i16};

uint32_t mode_bits(int mode) { return uint32_t(mode & 63) << 7 | (mode & 64 ? GOC_PK_CLAMP : 0); }

} // namespace

TEST(PackedMad, EveryModifierBoundaryTriplesAndRandomInputs) {
  const uint32_t edge[] = {0, 1, 2, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff};
  for (int sign = 0; sign < 2; ++sign)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int mode = 0; mode < 128; ++mode) {
        SCOPED_TRACE(::testing::Message() << sign << "/" << cpu << "/" << mode);
        std::mt19937 random(995);
        for (int start = 0; start < 1536; start += 32) {
          uint32_t words[4][32];
          uint32_t *p[] = {words[0], words[1], words[2], words[3]};
          for (int lane = 0; lane < 32; ++lane) {
            int index = start + lane;
            for (int reg = 0; reg < 3; ++reg) {
              words[reg][lane] =
                  start < 512 ? edge[index % 8] | (edge[7 - index % 8] << 16) : random();
              index /= 8;
            }
          }
          ASSERT_EQ(functions[sign](cpu, UINT32_MAX, mode_bits(mode), p + 3, p, p + 1, p + 2),
                    GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(words[3][lane],
                      goc_test::packed_mad_reference(sign, words[0][lane], words[1][lane],
                                                     words[2][lane], mode_bits(mode)));
        }
      }
}

TEST(PackedMad, EverySourceEncoding) {
  for (int sign = 0; sign < 2; ++sign)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int source = 0; source < 3; ++source)
        for (uint32_t mode : {uint32_t(0), GOC_PK_CLAMP | GOC_PK_LO_A_HIGH | GOC_PK_HI_C_LOW}) {
          std::mt19937 random(401);
          for (uint32_t start = 0; start < 65536; start += 32) {
            uint32_t words[4][32];
            uint32_t *p[] = {words[0], words[1], words[2], words[3]};
            for (uint32_t lane = 0; lane < 32; ++lane) {
              for (int reg = 0; reg < 3; ++reg)
                words[reg][lane] = random();
              words[source][lane] = (start + lane) | ((65535 - start - lane) << 16);
            }
            ASSERT_EQ(functions[sign](cpu, UINT32_MAX, mode, p + 3, p, p + 1, p + 2), GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane)
              ASSERT_EQ(words[3][lane],
                        goc_test::packed_mad_reference(sign, words[0][lane], words[1][lane],
                                                       words[2][lane], mode));
          }
        }
}

TEST(PackedMad, AllMasksModifiersAndWholeRegisterAliases) {
  for (int sign = 0; sign < 2; ++sign)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int mode = 0; mode < 128; ++mode)
        for (uint32_t exec_mask : exec_masks())
          // Restricted growth indices enumerate all 15 partitions of A/B/C/D.
          for (int bi = 0; bi <= 1; ++bi)
            for (int ci = 0; ci <= bi + 1; ++ci)
              for (int di = 0; di <= std::max(bi, ci) + 1; ++di) {
                uint32_t words[4][32], saved[4][32];
                uint32_t *p[] = {words[0], words[1], words[2], words[3]};
                for (int reg = 0; reg < 4; ++reg)
                  for (int lane = 0; lane < 32; ++lane)
                    words[reg][lane] = 0x7af551d3u * (lane + reg * 32 + 1);
                std::memcpy(saved, words, sizeof(words));
                ASSERT_EQ(
                    functions[sign](cpu, exec_mask, mode_bits(mode), p + di, p, p + bi, p + ci),
                    GOC_SUCCESS);
                for (int reg = 0; reg < 4; ++reg)
                  for (int lane = 0; lane < 32; ++lane) {
                    uint32_t expected =
                        reg == di && (exec_mask >> lane & 1)
                            ? goc_test::packed_mad_reference(sign, saved[0][lane], saved[bi][lane],
                                                             saved[ci][lane], mode_bits(mode))
                            : saved[reg][lane];
                    ASSERT_EQ(words[reg][lane], expected)
                        << sign << "/" << cpu << "/" << mode << "/" << exec_mask;
                  }
              }
}

TEST(PackedMad, LiteralFullPrecisionSaturationAndCrossHalfAliases) {
  struct Witness {
    bool sign;
    uint32_t a, b, c, mode, expected;
  };

  const Witness cases[] = {
      {false, 0xffffffff, 0xffffffff, 0xffffffff, 0, 0},
      {false, 0xffffffff, 0xffffffff, 0xffffffff, GOC_PK_CLAMP, 0xffffffff},
      {true, 0x80007fff, 0x00020002, 0x7fff8000, GOC_PK_CLAMP, 0x80007ffe},
      {true, 0x80008000, 0x80008000, 0x80008000, GOC_PK_CLAMP, 0x7fff7fff},
      {true, 0x80008000, 0x00010001, 0xffffffff, GOC_PK_CLAMP, 0x80008000},
      {true, 0xffffffff, 0xffffffff, 0xffffffff, GOC_PK_CLAMP, 0},
      {false, 0x00020003, 0x00050007, 0x000b000d,
       GOC_PK_LO_A_HIGH | GOC_PK_HI_A_LOW | GOC_PK_LO_C_HIGH | GOC_PK_HI_C_LOW, 0x001c0019},
  };
  for (const auto &w : cases)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t words[3][32];
      uint32_t *p[] = {words[0], words[1], words[2]};
      for (int lane = 0; lane < 32; ++lane) {
        words[0][lane] = w.a;
        words[1][lane] = w.b;
        words[2][lane] = w.c;
      }
      ASSERT_EQ(functions[w.sign](cpu, UINT32_MAX, w.mode, p, p, p + 1, p + 2), GOC_SUCCESS);
      for (uint32_t value : words[0])
        EXPECT_EQ(value, w.expected);
    }
}

TEST(PackedMad, ValidationAndFloatingEnvironment) {
  for (Fn fn : functions) {
    uint32_t words[4][32] = {}, saved[32];
    uint32_t *p[] = {words[0], words[1], words[2], words[3]};
    for (int i = 0; i < 32; ++i)
      words[3][i] = 0x76543210;
    std::memcpy(saved, words[3], sizeof(saved));
    for (uint32_t invalid : {GOC_PK_NEG_LO_A, GOC_PK_NEG_LO_B, GOC_PK_NEG_LO_C, GOC_PK_NEG_HI_A,
                             GOC_PK_NEG_HI_B, GOC_PK_NEG_HI_C, 0x80000000U}) {
      EXPECT_EQ(fn(0, UINT32_MAX, invalid, p + 3, p, p + 1, p + 2), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(0, 0, invalid, nullptr, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
    }
    EXPECT_EQ(std::memcmp(saved, words[3], sizeof(saved)), 0);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr, nullptr,
                 nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(fn(1ULL << 63, 0, 0, nullptr, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(0, 0U, 0, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
    goc_test::ScopedFpEnvironment environment;
    ASSERT_TRUE(environment.saved());
    std::fesetround(FE_UPWARD);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      EXPECT_EQ(
          fn(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, GOC_PK_CLAMP, p + 3, p, p + 1, p + 2),
          GOC_SUCCESS);
      EXPECT_EQ(std::fegetround(), FE_UPWARD);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
    }
  }
}
