// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_packed_integer_reference.h"

#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_pk_add_i16);
const Fn functions[] = {
    goc_rdna4_v_pk_add_i16,     goc_rdna4_v_pk_sub_i16,     goc_rdna4_v_pk_add_u16,
    goc_rdna4_v_pk_sub_u16,     goc_rdna4_v_pk_min_i16,     goc_rdna4_v_pk_max_i16,
    goc_rdna4_v_pk_min_u16,     goc_rdna4_v_pk_max_u16,     goc_rdna4_v_pk_mul_lo_u16,
    goc_rdna4_v_pk_lshlrev_b16, goc_rdna4_v_pk_lshrrev_b16, goc_rdna4_v_pk_ashrrev_i16};

uint32_t mode_bits(int mode) {
  return (mode & 1 ? GOC_PK_LO_A_HIGH : 0) | (mode & 2 ? GOC_PK_LO_B_HIGH : 0) |
         (mode & 4 ? GOC_PK_HI_A_LOW : 0) | (mode & 8 ? GOC_PK_HI_B_LOW : 0) |
         (mode & 16 ? GOC_PK_CLAMP : 0);
}

} // namespace

TEST(PackedInteger, AllModifiersBoundaryPairsAndRandomInputs) {
  const uint32_t values[] = {0, 1, 2, 0x7ffe, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff};
  for (int op = 0; op < 12; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int mode = 0; mode < 32; ++mode) {
        SCOPED_TRACE(::testing::Message() << op << "/" << cpu << "/" << mode);
        std::mt19937 random(921);
        for (int start = 0; start < 1152; start += 32) {
          uint32_t a[32], b[32], d[32];
          const uint32_t *ap[] = {a}, *bp[] = {b};
          uint32_t *dp[] = {d};
          for (int lane = 0; lane < 32; ++lane) {
            int i = start + lane;
            a[lane] = i < 128 ? values[i % 9] | (values[(i / 9) % 9] << 16) : random();
            b[lane] = i < 128 ? values[(i / 9) % 9] | (values[8 - i % 9] << 16) : random();
          }
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, mode_bits(mode), dp, ap, bp), GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(d[lane],
                      goc_test::packed_integer_reference(op, a[lane], b[lane], mode_bits(mode)));
        }
      }
}

TEST(PackedInteger, EveryHalfEncoding) {
  for (int op = 0; op < 12; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint32_t mode : {uint32_t(0), GOC_PK_CLAMP | GOC_PK_LO_A_HIGH | GOC_PK_HI_B_LOW}) {
        SCOPED_TRACE(::testing::Message() << op << "/" << cpu << "/" << mode);
        std::mt19937 random(475);
        for (uint32_t start = 0; start < 65536; start += 32) {
          uint32_t a[32], b[32], d[32];
          const uint32_t *ap[] = {a}, *bp[] = {b};
          uint32_t *dp[] = {d};
          for (uint32_t lane = 0; lane < 32; ++lane) {
            a[lane] = (start + lane) | ((65535 - start - lane) << 16);
            b[lane] = random();
            if (op >= 9) {
              uint32_t count = b[lane];
              b[lane] = a[lane];
              a[lane] = count;
            }
          }
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, mode, dp, ap, bp), GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(d[lane], goc_test::packed_integer_reference(op, a[lane], b[lane], mode));
        }
      }
}

TEST(PackedInteger, MasksAndAllWholeRegisterAliases) {
  const int layouts[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  for (int op = 0; op < 12; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int mode = 0; mode < 32; ++mode)
        for (uint32_t mask : rdna4_exec_masks())
          for (const auto &layout : layouts) {
            uint32_t words[3][32], saved[3][32];
            for (int r = 0; r < 3; ++r)
              for (int lane = 0; lane < 32; ++lane)
                words[r][lane] = uint32_t(0x73be19a5u * (lane + 17 * r + 1));
            std::memcpy(saved, words, sizeof(words));
            const uint32_t *a[] = {words[layout[0]]}, *b[] = {words[layout[1]]};
            uint32_t *d[] = {words[layout[2]]};
            ASSERT_EQ(functions[op](cpu, mask, mode_bits(mode), d, a, b), GOC_SUCCESS);
            for (int r = 0; r < 3; ++r)
              for (int lane = 0; lane < 32; ++lane) {
                uint32_t expected =
                    r == layout[2] && (mask >> lane & 1)
                        ? goc_test::packed_integer_reference(
                              op, saved[layout[0]][lane], saved[layout[1]][lane], mode_bits(mode))
                        : saved[r][lane];
                ASSERT_EQ(words[r][lane], expected)
                    << op << "/" << cpu << "/" << mode << "/" << mask << "/" << lane;
              }
          }
}

TEST(PackedInteger, LiteralSaturationAndHalfSelectionWitnesses) {
  struct Witness {
    int op;
    uint32_t a, b, mode, expected;
  };

  const Witness cases[] = {
      {9, 0x0001000f, 0x80010001, 0, 0x00028000},
      {10, 0x0001000f, 0x80018000, 0, 0x40000001},
      {11, 0x0001000f, 0x80018000, 0, 0xc000ffff},
      {9, 0xfff00010, 0x1234abcd, GOC_PK_CLAMP, 0x1234abcd},
      {10, 0x00000001, 0xffff0000, 0, 0xffff0000},
      {11, 0x000f000f, 0xffff7fff, GOC_PK_CLAMP, 0xffff0000},
      {0, 0x80007fff, 0xffff0001, 0, 0x7fff8000},
      {0, 0x80007fff, 0xffff0001, GOC_PK_CLAMP, 0x80007fff},
      {1, 0x7fff8000, 0xffff0001, GOC_PK_CLAMP, 0x7fff8000},
      {2, 0x80007fff, 0xffff0001, GOC_PK_CLAMP, 0xffff8000},
      {3, 0x00010000, 0x00020001, 0, 0xffffffff},
      {3, 0x00010000, 0x00020001, GOC_PK_CLAMP, 0},
      {4, 0x80007fff, 0x7fff8000, GOC_PK_CLAMP, 0x80008000},
      {6, 0x80007fff, 0x7fff8000, GOC_PK_CLAMP, 0x7fff7fff},
      {8, 0xffffffff, 0xffffffff, GOC_PK_CLAMP, 0x00010001},
      {2, 0x11112222, 0x33334444, GOC_PK_LO_A_HIGH | GOC_PK_HI_B_LOW, 0x55555555},
  };
  for (const auto &w : cases)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t a[32], b[32];
      for (int lane = 0; lane < 32; ++lane) {
        a[lane] = w.a;
        b[lane] = w.b;
      }
      const uint32_t *ap[] = {a}, *bp[] = {b};
      uint32_t *dp[] = {a};
      ASSERT_EQ(functions[w.op](cpu, UINT32_MAX, w.mode, dp, ap, bp), GOC_SUCCESS);
      for (uint32_t value : a)
        EXPECT_EQ(value, w.expected);
    }
}

TEST(PackedInteger, ShiftCountsAndDiscardedCountBits) {
  const uint32_t values[] = {0, 1, 0x7fff, 0x8000, 0x8001, 0xffff, 0x55aa, 0xaa55};
  for (int op = 9; op < 12; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int mode = 0; mode < 32; ++mode)
        for (uint32_t start = 0; start < 65536; start += 32) {
          uint32_t a[32], b[32], d[32];
          for (uint32_t lane = 0; lane < 32; ++lane) {
            a[lane] = (start + lane) | ((65535 - start - lane) << 16);
            b[lane] = values[lane % 8] | (values[(lane / 8 + start / 32) % 8] << 16);
          }
          const uint32_t *ap[] = {a}, *bp[] = {b};
          uint32_t *dp[] = {d};
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, mode_bits(mode), dp, ap, bp), GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            ASSERT_EQ(d[lane],
                      goc_test::packed_integer_reference(op, a[lane], b[lane], mode_bits(mode)))
                << op << "/" << cpu << "/" << mode << "/" << start << "/" << lane;
        }
}

TEST(PackedInteger, ValidationAndFloatingEnvironment) {
  for (Fn fn : functions) {
    uint32_t a[32] = {}, b[32] = {}, d[32], saved[32];
    for (int i = 0; i < 32; ++i)
      d[i] = 0x12345678;
    std::memcpy(saved, d, sizeof(d));
    const uint32_t *ap[] = {a}, *bp[] = {b};
    uint32_t *dp[] = {d};
    for (uint32_t invalid :
         {GOC_PK_NEG_LO_A, GOC_PK_NEG_LO_B, GOC_PK_NEG_HI_A, GOC_PK_NEG_HI_B, GOC_PK_NEG_LO_C,
          GOC_PK_NEG_HI_C, GOC_PK_LO_C_HIGH, GOC_PK_HI_C_LOW, UINT32_C(0x80000000)}) {
      EXPECT_EQ(fn(0, UINT32_MAX, invalid, dp, ap, bp), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(0, 0, invalid, nullptr, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
    }
    EXPECT_EQ(std::memcmp(saved, d, sizeof(d)), 0);
    EXPECT_EQ(
        fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr, nullptr, nullptr),
        GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(fn(0, UINT32_C(0), 0, nullptr, nullptr, nullptr), GOC_SUCCESS);
    goc_test::ScopedFpEnvironment environment;
    ASSERT_TRUE(environment.saved());
    std::fesetround(FE_DOWNWARD);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INVALID);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      EXPECT_EQ(fn(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, GOC_PK_CLAMP, dp, ap, bp),
                GOC_SUCCESS);
      EXPECT_EQ(std::fegetround(), FE_DOWNWARD);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID);
    }
  }
}
