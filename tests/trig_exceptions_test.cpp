// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "exec_masks.h"
#include "goc/goc.h"
#include "trig_exceptions_hardware.h"

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_v_sin_f16);
const Fn functions[] = {goc_v_sin_f16, goc_v_cos_f16, goc_v_sin_f32, goc_v_cos_f32};
const decltype(&goc_v_sin_f16_wave64) wave64_functions[] = {
    goc_v_sin_f16_wave64, goc_v_cos_f16_wave64, goc_v_sin_f32_wave64, goc_v_cos_f32_wave64};
const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

} // namespace

TEST(TrigExceptions, HardwareCorpora) {
  const uint32_t anchors[] = {0,          0x00145f30, 0x00800000, 0x3e800000,
                              0x3f000000, 0x3f800000, 0x4a000000, 0x7f800000,
                              0x80000000, 0x80145f30, 0x80800000, 0xbe800000,
                              0xbf000000, 0xbf800000, 0x7fc00000, 0xff800000};
  for (unsigned corpus = 0; corpus < 6; ++corpus)
    for (unsigned variant = 0; variant < 16; ++variant) {
      uint32_t mode = ((variant & 3) << 6) | (variant & 4 ? GOC_ALU_CLAMP : 0) |
                      (variant & 8 ? GOC_ALU_ABS_A | GOC_ALU_NEG_A : 0);
      uint64_t hash = goc_test::capture_hash_seed;
      for (unsigned i = 0; i < 65536; ++i) {
        uint32_t a[32], d[32] = {};
        // Nonparticipating signaling NaNs must not contribute exceptions.
        std::fill(a, a + 32, corpus < 2 ? 0x7c01 : 0x7f800001);
        a[0] = corpus < 2   ? i
               : corpus < 4 ? (i << 16) | ((i * 0x9e37) & 65535)
                            : anchors[i / 4096] + (i % 4096) - 2048;
        const uint32_t *pa = a;
        uint32_t *pd = d, exceptions = 0x80000000;
        ASSERT_EQ(functions[corpus < 4 ? corpus : corpus - 2](
                      exact | (i % (goc_init_cpu_flags() + 1)), 1, mode, &pd, &pa, &exceptions),
                  GOC_SUCCESS);
        ASSERT_TRUE(exceptions & 0x80000000);
        hash = goc_test::capture_hash_word(hash, exceptions & 127);
      }
      EXPECT_EQ(hash, goc_test::trig_exception_hashes[corpus][variant]) << corpus << "/" << variant;
    }
}

TEST(TrigExceptions, DppMasksAliasesAndHalfSelectors) {
  const uint32_t values[2][8] = {
      {0x7c013800, 0x00010000, 0x04007c00, 0x7e003c00, 0x35558001, 0x3fff3a00, 0x3c013400,
       0x80000000},
      {0x7f800001, 1, 0x7f800000, 0x7fc00000, 0x80000001, 0x3f000001, 0x3e800000, 0x80000000}};
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t descriptor : goc_test::dpp_modes)
      for (uint32_t exec_mask : exec_masks())
        for (unsigned variant = 0; variant < (op < 2 ? 4U : 1U); ++variant) {
          uint32_t mode = (variant & 1 ? GOC_ALU_HIGH_A : 0) | (variant & 2 ? GOC_ALU_HIGH_D : 0);
          uint32_t initial[32], expected[32], expected_flags = 0x80000000;
          for (unsigned lane = 0; lane < 32; ++lane)
            initial[lane] = expected[lane] = values[op / 2][lane % 8];
          for (unsigned lane = 0; lane < 32; ++lane) {
            int source;
            if (!goc_test::dpp_source(descriptor, exec_mask, lane, source))
              continue;
            uint32_t a[32] = {}, d[32] = {};
            a[0] = source < 0 ? 0 : initial[source];
            d[0] = initial[lane];
            const uint32_t *pa = a;
            uint32_t *pd = d;
            ASSERT_EQ(functions[op](exact, 1, mode, &pd, &pa, &expected_flags), GOC_SUCCESS);
            expected[lane] = d[0];
          }
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t actual[32], exceptions = 0x80000000;
            std::memcpy(actual, initial, sizeof(actual));
            uint32_t *p = actual;
            ASSERT_EQ(functions[op](cpu | exact, exec_mask, descriptor | mode, &p, &p, &exceptions),
                      GOC_SUCCESS);
            EXPECT_EQ(exceptions, expected_flags);
            for (unsigned lane = 0; lane < 32; ++lane)
              EXPECT_EQ(actual[lane], expected[lane]);
          }
        }
}

TEST(TrigExceptions, LooseOptOutAndErrorAtomicity) {
  for (Fn fn : functions)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t a[32], d[32], expected[32];
      std::fill(a, a + 32, 0x7f817c01);
      std::fill(d, d + 32, 0x12345678);
      std::fill(expected, expected + 32, 0x12345678);
      const uint32_t *pa = a;
      uint32_t *pd = d, *pe = expected, exceptions = 0x80000000;
      ASSERT_EQ(fn(cpu, UINT32_MAX, 0, &pd, &pa, &exceptions), GOC_SUCCESS);
      ASSERT_EQ(fn(cpu, UINT32_MAX, 0, &pe, &pa, nullptr), GOC_SUCCESS);
      EXPECT_EQ(exceptions, 0x80000000);
      for (unsigned lane = 0; lane < 32; ++lane)
        EXPECT_EQ(d[lane], expected[lane]);
      EXPECT_EQ(fn(cpu | exact, 0, 0, nullptr, nullptr, &exceptions), GOC_SUCCESS);
      EXPECT_EQ(fn(cpu | exact, UINT32_MAX, 1ULL << 31, nullptr, nullptr, &exceptions),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(exceptions, 0x80000000);
    }
}

TEST(TrigExceptions, Wave64HardwareCorpora) {
  const uint32_t anchors[] = {0,          0x00145f30, 0x00800000, 0x3e800000,
                              0x3f000000, 0x3f800000, 0x4a000000, 0x7f800000,
                              0x80000000, 0x80145f30, 0x80800000, 0xbe800000,
                              0xbf000000, 0xbf800000, 0x7fc00000, 0xff800000};
  for (unsigned corpus = 0; corpus < 6; ++corpus)
    for (unsigned variant = 0; variant < 16; ++variant) {
      uint32_t mode = ((variant & 3) << 6) | (variant & 4 ? GOC_ALU_CLAMP : 0) |
                      (variant & 8 ? GOC_ALU_ABS_A | GOC_ALU_NEG_A : 0);
      uint64_t hash = goc_test::capture_hash_seed;
      for (unsigned i = 0; i < 65536; ++i) {
        uint32_t a[64], d[64] = {};
        // Nonparticipating signaling NaNs must not contribute exceptions.
        std::fill(a, a + 64, corpus < 2 ? 0x7c01 : 0x7f800001);
        a[63] = corpus < 2   ? i
                : corpus < 4 ? (i << 16) | ((i * 0x9e37) & 65535)
                             : anchors[i / 4096] + (i % 4096) - 2048;
        const uint32_t *pa = a;
        uint32_t *pd = d, exceptions = 0x80000000;
        ASSERT_EQ(
            wave64_functions[corpus < 4 ? corpus : corpus - 2](
                exact | (i % (goc_init_cpu_flags() + 1)), 1ULL << 63, mode, &pd, &pa, &exceptions),
            GOC_SUCCESS);
        ASSERT_TRUE(exceptions & 0x80000000);
        hash = goc_test::capture_hash_word(hash, exceptions & 127);
      }
      EXPECT_EQ(hash, goc_test::trig_exception_hashes[corpus][variant]) << corpus << "/" << variant;
    }
}

TEST(TrigExceptions, Wave64DppMasksAliasesAndHalfSelectors) {
  const uint32_t values[2][8] = {
      {0x7c013800, 0x00010000, 0x04007c00, 0x7e003c00, 0x35558001, 0x3fff3a00, 0x3c013400,
       0x80000000},
      {0x7f800001, 1, 0x7f800000, 0x7fc00000, 0x80000001, 0x3f000001, 0x3e800000, 0x80000000}};
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t descriptor_base : goc_test::dpp_modes)
      for (uint64_t exec_mask :
           {uint64_t(UINT64_MAX), uint64_t(0xffffffff00000000ULL), uint64_t(0x87654321fedcba98ULL)})
        for (unsigned variant = 0; variant < (op < 2 ? 128U : 32U); ++variant) {
          uint64_t descriptor = descriptor_base;
          if (descriptor & GOC_DPP16)
            descriptor =
                (descriptor & ~GOC_DPP_ROW_MASK) | (uint64_t(variant & 15) << GOC_DPP_ROW_SHIFT);
          uint32_t mode = (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
                          (((variant >> 2) & 3) << 6) | (variant & 16 ? GOC_ALU_CLAMP : 0) |
                          (variant & 32 ? GOC_ALU_HIGH_A : 0) | (variant & 64 ? GOC_ALU_HIGH_D : 0);
          uint32_t initial[64], expected[64], expected_flags = 0x80000000;
          for (unsigned lane = 0; lane < 64; ++lane)
            initial[lane] = expected[lane] = values[op / 2][(lane + 3 * (lane / 32)) % 8];
          for (unsigned lane = 0; lane < 64; ++lane) {
            int source;
            if (!goc_test::dpp_source(descriptor, exec_mask, lane, source))
              continue;
            uint32_t a[32] = {}, d[32] = {};
            a[0] = source < 0 ? 0 : initial[source];
            d[0] = initial[lane];
            const uint32_t *pa = a;
            uint32_t *pd = d;
            ASSERT_EQ(functions[op](exact, 1, mode, &pd, &pa, &expected_flags), GOC_SUCCESS);
            expected[lane] = d[0];
          }
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t actual[64], exceptions = 0x80000000;
            std::memcpy(actual, initial, sizeof(actual));
            uint32_t *p = actual;
            ASSERT_EQ(wave64_functions[op](cpu | exact, exec_mask, descriptor | mode, &p, &p,
                                           &exceptions),
                      GOC_SUCCESS);
            EXPECT_EQ(exceptions, expected_flags);
            for (unsigned lane = 0; lane < 64; ++lane)
              EXPECT_EQ(actual[lane], expected[lane]);
          }
        }
}

TEST(TrigExceptions, Wave64LooseOptOutAndErrorAtomicity) {
  for (auto fn : wave64_functions)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t a[64], d[64], expected[64];
      std::fill(a, a + 64, 0x7f817c01);
      std::fill(d, d + 64, 0x12345678);
      std::fill(expected, expected + 64, 0x12345678);
      const uint32_t *pa = a;
      uint32_t *pd = d, *pe = expected, exceptions = 0x80000000;
      ASSERT_EQ(fn(cpu, UINT64_MAX, 0, &pd, &pa, &exceptions), GOC_SUCCESS);
      ASSERT_EQ(fn(cpu, UINT64_MAX, 0, &pe, &pa, nullptr), GOC_SUCCESS);
      EXPECT_EQ(exceptions, 0x80000000);
      for (unsigned lane = 0; lane < 64; ++lane)
        EXPECT_EQ(d[lane], expected[lane]);
      EXPECT_EQ(fn(cpu | exact, 0, 0, nullptr, nullptr, &exceptions), GOC_SUCCESS);
      EXPECT_EQ(fn(cpu | exact, UINT64_MAX, 1ULL << 31, nullptr, nullptr, &exceptions),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(exceptions, 0x80000000);
    }
}

TEST(TrigExceptions, Wave64ReportingAliasesUpperSource) {
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t a[64] = {}, d[64], expected[64];
      // Lower invalid must not be written into lane 63 before it is read.
      a[0] = op < 2 ? 0x7c01 : 0x7f800001;
      a[63] = op < 2 ? 0x3400 : 0x3e000000;
      std::fill(d, d + 64, 0xdeadbeef);
      std::fill(expected, expected + 64, 0xdeadbeef);
      const uint32_t *pa = a;
      uint32_t *pd = d, *pe = expected, exceptions = 0;
      uint32_t initial = a[63];
      ASSERT_EQ(wave64_functions[op](cpu | exact, 1ULL | (1ULL << 63), 0, &pe, &pa, &exceptions),
                GOC_SUCCESS);
      ASSERT_NE(exceptions, 0U);
      ASSERT_EQ(wave64_functions[op](cpu | exact, 1ULL | (1ULL << 63), 0, &pd, &pa, &a[63]),
                GOC_SUCCESS);
      EXPECT_EQ(a[63], initial | exceptions);
      for (unsigned lane = 0; lane < 64; ++lane)
        EXPECT_EQ(d[lane], expected[lane]);
    }
}
