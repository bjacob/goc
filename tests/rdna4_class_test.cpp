// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_class_hardware.h"
#include "rdna4_class_reference.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"

#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {
using Fn = decltype(&goc_rdna4_v_cmp_class_f16);
const Fn functions[] = {goc_rdna4_v_cmp_class_f16, goc_rdna4_v_cmpx_class_f16,
                        goc_rdna4_v_cmp_class_f32, goc_rdna4_v_cmpx_class_f32,
                        goc_rdna4_v_cmp_class_f64, goc_rdna4_v_cmpx_class_f64};
} // namespace

TEST(Class, DppModifiersMasksAliasesAndFpState) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    std::fesetround(rounding);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INVALID | FE_INEXACT);
    for (unsigned op = 0; op < 4; ++op)
      for (unsigned m = 0; m < (op < 2 ? 16u : 4u); ++m)
        for (uint64_t descriptor : goc_test::dpp_modes)
          for (bool shared : {false, true}) {
            uint32_t initial[2][34];
            for (unsigned lane = 0; lane < 34; ++lane) {
              initial[0][lane] =
                  op < 2 ? uint32_t(goc_test::class_edges16[lane % 16]) |
                               (uint32_t(goc_test::class_edges16[(lane + 7) % 16]) << 16)
                         : goc_test::class_edges32[lane % 16];
              initial[1][lane] = (lane * 0x9e3779b9u) ^ 0xa5a59669u;
            }
            for (uint32_t mask : rdna4_exec_masks()) {
              uint32_t want = 0;
              for (unsigned lane = 0; lane < 32; ++lane) {
                int source;
                if (goc_test::dpp_source(descriptor, mask, lane, source))
                  want |= uint32_t(goc_test::class_reference(
                              op / 2, source < 0 ? 0 : initial[0][source + 1], 0,
                              initial[shared ? 0 : 1][lane + 1], m))
                          << lane;
              }
              for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
                for (unsigned target = 0; target < 5; ++target) {
                  uint32_t words[2][34], outside = 0xdeadbeef;
                  std::memcpy(words, initial, sizeof(words));
                  unsigned reg = target / 2, word = target & 1 ? 32 : 1;
                  uint32_t *d = target == 4 ? &outside : &words[reg][word];
                  const uint32_t *a[] = {words[0] + 1}, *b[] = {words[shared ? 0 : 1] + 1};
                  uint64_t flags =
                      cpu | (shared ? GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                                          GOC_FP_FLUSH_INPUT_DENORMALS
                                    : 0);
                  ASSERT_EQ(
                      functions[op](flags, mask, descriptor | goc_test::class_mode(m), d, a, b),
                      GOC_SUCCESS);
                  ASSERT_EQ(*d, want) << op << "/" << m << "/" << descriptor << "/" << cpu;
                  for (unsigned r = 0; r < 2; ++r)
                    for (unsigned w = 0; w < 34; ++w)
                      ASSERT_EQ(words[r][w],
                                target != 4 && r == reg && w == word ? want : initial[r][w]);
                }
            }
          }
    EXPECT_EQ(std::fegetround(), rounding);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID | FE_INEXACT);
  }
}

TEST(Class, DppValidationAndZeroExec) {
  for (unsigned op = 0; op < 6; ++op)
    for (uint64_t descriptor : goc_test::dpp_modes) {
      uint32_t result = 0xdeadbeef;
      EXPECT_EQ(functions[op](0, UINT32_C(0), descriptor, &result, nullptr, nullptr),
                op < 4 ? GOC_SUCCESS : GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(result, op < 4 ? 0u : 0xdeadbeef);
      for (uint64_t invalid : {UINT64_C(1) << 36, uint64_t(GOC_ALU_NEG_B)}) {
        result = 0xdeadbeef;
        EXPECT_EQ(functions[op](0, UINT32_MAX, descriptor | invalid, &result, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(result, 0xdeadbeef);
      }
    }
}

TEST(Class, DppHardwareCorpus) {
  // RX 9070/gfx1201: 17,920 masks, MODE 0 and 0xf0, CMP/CMPX,
  // every source modifier and seven DPP descriptors. FNV hashes raw masks.
  const uint32_t values[][8] = {
      {0x80000000, 0x80010001, 0x83ff03ff, 0x84000400, 0xbc003c00, 0xfc007c00, 0xfe007e00,
       0xfc017c01},
      {0, 0x80000000, 1, 0x80000001, 0x3f800000, 0xbf800000, 0x7f800000, 0x7f800001}};
  const uint32_t masks[] = {UINT32_MAX, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t semantics : {UINT64_C(0), GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT}) {
      uint64_t hash = UINT64_C(14695981039346656037);
      for (bool flush : {true, false})
        for (unsigned batch = 0; batch < 4; ++batch)
          for (uint32_t mask : masks)
            for (unsigned op = 0; op < 4; ++op)
              for (unsigned m = 0; m < (op < 2 ? 16u : 4u); ++m)
                for (uint64_t descriptor : goc_test::dpp_modes) {
                  uint32_t av[32], bv[32], result = 0xdeadbeef;
                  for (unsigned lane = 0; lane < 32; ++lane) {
                    av[lane] = values[op / 2][(lane + batch) % 8];
                    bv[lane] = ((lane + batch * 32) * 0x9e3779b9u) ^ 0xa5a59669u;
                  }
                  const uint32_t *a[] = {av}, *b[] = {bv};
                  ASSERT_EQ(
                      functions[op](cpu | semantics | (flush ? GOC_FP_FLUSH_INPUT_DENORMALS : 0),
                                    mask, descriptor | goc_test::class_mode(m), &result, a, b),
                      GOC_SUCCESS);
                  hash = (hash ^ result) * UINT64_C(1099511628211);
                }
      EXPECT_EQ(hash, UINT64_C(0xb8f5447c269df6e5)) << cpu << "/" << semantics;
    }
}

TEST(Class, HardwareFormatsModifiersExecAndCmpx) {
  const uint32_t masks[] = {UINT32_MAX, 0, 0x55555555, 0xaaaaaaaa, 1};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned mi = 0; mi < 5; ++mi) {
      unsigned variant = 0;
      for (unsigned op = 0; op < 6; ++op)
        for (unsigned m = 0; m < (op < 2 ? 16u : 4u); ++m, ++variant) {
          uint64_t digest = UINT64_C(14695981039346656037);
          for (unsigned start = 0; start < 65536; start += 32) {
            uint32_t words[3][32], result = 0x12345678;
            goc_test::class_capture_inputs(op / 2, start, words);
            const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2]};
            ASSERT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                                        GOC_FP_FLUSH_OUTPUT_DENORMALS |
                                        GOC_FP_FLUSH_INPUT_DENORMALS,
                                    masks[mi], goc_test::class_mode(m), &result, a, b),
                      GOC_SUCCESS);
            for (unsigned byte = 0; byte < 4; ++byte) {
              digest ^= (result >> (8 * byte)) & 255;
              digest *= UINT64_C(1099511628211);
            }
          }
          EXPECT_EQ(digest, goc_test::class_digests[mi * 48 + variant])
              << cpu << "/" << mi << "/" << op << "/" << m;
        }
    }
}

TEST(Class, EveryClassMaskAndKnownClass) {
  const unsigned indices[] = {6, 5, 7, 4, 7, 4, 8, 3, 8, 3, 8, 3, 9, 2, 0, 1};
  for (unsigned op = 0; op < 6; ++op)
    for (unsigned m = 0; m < (op < 2 ? 16u : 4u); ++m)
      for (unsigned classes = 0; classes < 1024; ++classes) {
        uint32_t words[3][32], want = 0;
        for (unsigned lane = 0; lane < 32; ++lane) {
          uint64_t value = op < 2 ? uint64_t(goc_test::class_edges16[lane % 16]) |
                                        (uint64_t(goc_test::class_edges16[(lane + 7) % 16]) << 16)
                           : op < 4 ? goc_test::class_edges32[lane % 16]
                                    : goc_test::class_edges64[lane % 16];
          words[0][lane] = uint32_t(value);
          words[1][lane] = uint32_t(value >> 32);
          words[2][lane] = classes | ((1023u - classes) << 16) | 0xfc00fc00;
          unsigned index = indices[(lane + (op < 2 && (m & 4) ? 7 : 0)) % 16];
          if (index >= 2) {
            if ((m & 1) && index < 6)
              index = 11 - index;
            if (m & 2)
              index = 11 - index;
          }
          uint32_t selected = op < 2 && (m & 8) ? 1023 - classes : classes;
          want |= ((selected >> index) & 1) << lane;
        }
        const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2]};
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          uint32_t result = 0;
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, goc_test::class_mode(m), &result, a, b),
                    GOC_SUCCESS);
          ASSERT_EQ(result, want) << cpu << "/" << op << "/" << m << "/" << classes;
        }
      }
}

TEST(Class, ScalarOutputAliasesMasksAndUnalignedStorage) {
  for (unsigned op = 0; op < 6; ++op)
    for (unsigned m = 0; m < (op < 2 ? 16u : 4u); ++m)
      for (unsigned source_alias = 0; source_alias < 2; ++source_alias) {
        uint32_t initial[3][35];
        for (unsigned reg = 0; reg < 3; ++reg)
          for (unsigned lane = 0; lane < 35; ++lane)
            initial[reg][lane] = (lane * 0x7395a831u) ^ (reg * 0xa7925163u);
        uint32_t expected = 0;
        for (unsigned lane = 0; lane < 32; ++lane)
          expected |= uint32_t(goc_test::class_reference(
                          op / 2, initial[0][lane + 1], initial[source_alias ? 0 : 1][lane + 1],
                          initial[source_alias ? 0 : 2][lane + 1], m))
                      << lane;
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (uint32_t mask : rdna4_exec_masks())
            for (unsigned target = 0; target < 7; ++target) {
              uint32_t words[3][35], outside = 0x87654321;
              std::memcpy(words, initial, sizeof(words));
              unsigned reg_target = target / 2, lane_target = (target & 1) ? 32 : 1;
              uint32_t *result = target == 6 ? &outside : &words[reg_target][lane_target];
              const uint32_t *a[] = {words[0] + 1, words[source_alias ? 0 : 1] + 1},
                             *b[] = {words[source_alias ? 0 : 2] + 1};
              ASSERT_EQ(functions[op](cpu, mask, goc_test::class_mode(m), result, a, b),
                        GOC_SUCCESS);
              ASSERT_EQ(*result, expected & mask) << cpu << "/" << op << "/" << m << "/" << target;
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned lane = 0; lane < 35; ++lane) {
                  uint32_t want = target != 6 && reg == reg_target && lane == lane_target
                                      ? expected & mask
                                      : initial[reg][lane];
                  ASSERT_EQ(words[reg][lane], want);
                }
            }
      }
}

TEST(Class, ValidationAndCompleteHostFpState) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (unsigned op = 0; op < 6; ++op) {
    uint32_t known = goc_test::class_mode(op < 2 ? 15 : 3), result = 1;
    EXPECT_EQ(functions[op](0, UINT32_C(0), known, &result, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(result, 0u);
    for (unsigned bit = 0; bit < 32; ++bit) {
      if (!(known & (1u << bit))) {
        result = 0xdeadbeef;
        EXPECT_EQ(functions[op](0, 0, 1u << bit, &result, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(result, 0xdeadbeef);
      }
    }
    for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
      for (unsigned flush = 0; flush < 2; ++flush) {
        std::fesetround(rounding);
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
        _mm_setcsr((_mm_getcsr() & ~0x8040u) | (flush ? 0x8040u : 0));
        unsigned before = _mm_getcsr();
#endif
        int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
        uint32_t words[3][32];
        goc_test::class_capture_inputs(op / 2, 0, words);
        const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2]};
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          EXPECT_EQ(functions[op](cpu, UINT32_MAX, known, &result, a, b), GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
#if defined(__x86_64__) || defined(_M_X64)
        EXPECT_EQ(_mm_getcsr(), before);
#endif
      }
  }
}
