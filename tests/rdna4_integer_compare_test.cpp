// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_integer_compare_hardware.h"
#include "rdna4_integer_compare_reference.h"

#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

TEST(IntegerCompare, DppPredicatesSelectorsMasksAndAliases) {
  for (unsigned op = 0; op < 48; ++op)
    for (unsigned m = 0; m < (op < 24 ? 4u : 1u); ++m)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (auto descriptor : goc_test::dpp_modes)
          for (auto mask : rdna4_exec_masks())
            for (unsigned shared = 0; shared < 2; ++shared)
              for (unsigned target = 0; target < 5; ++target) {
                uint32_t words[2][34], expected[2][34], outside = 0xdeadbeef, want = 0;
                for (unsigned word = 0; word < 34; ++word) {
                  uint32_t w[4];
                  goc_test::integer_compare_inputs(op / 24, 256 + word, w);
                  words[0][word] = expected[0][word] = w[0];
                  words[1][word] = expected[1][word] = w[2];
                }
                unsigned br = shared ? 0 : 1;
                for (unsigned lane = 0; lane < 32; ++lane) {
                  int source = 0;
                  if (goc_test::dpp_source(descriptor, mask, lane, source)) {
                    uint32_t w[] = {source < 0 ? 0 : words[0][source + 1], 0, words[br][lane + 1],
                                    0};
                    want |= uint32_t(goc_test::integer_compare_reference(op, m, w)) << lane;
                  }
                }
                unsigned reg = target / 2, word = target % 2 ? 32 : 1;
                uint32_t *d = target < 4 ? words[reg] + word : &outside;
                if (target < 4)
                  expected[reg][word] = want;
                const uint32_t *a[] = {words[0] + 1}, *b[] = {words[br] + 1};
                uint64_t semantics =
                    op & 1 ? GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT : 0;
                ASSERT_EQ(goc_test::integer_compare_functions[op](
                              cpu | semantics, mask, descriptor | goc_test::integer_compare_mode(m),
                              d, a, b),
                          GOC_SUCCESS);
                ASSERT_EQ(*d, want)
                    << op << "/" << m << "/" << cpu << "/" << descriptor << "/" << mask;
                ASSERT_EQ(std::memcmp(words, expected, sizeof(words)), 0);
              }
}

TEST(IntegerCompare, DppValidationAndZeroExec) {
  for (unsigned op = 0; op < 72; ++op)
    for (auto descriptor : goc_test::dpp_modes) {
      uint32_t d = 0xdeadbeef;
      auto fn = goc_test::integer_compare_functions[op];
      for (auto invalid : {UINT64_C(1) << 36, UINT64_C(1)}) {
        EXPECT_EQ(fn(0, 0, descriptor | invalid, &d, nullptr, nullptr), GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(d, 0xdeadbeefu);
      }
      EXPECT_EQ(fn(0, UINT32_C(0), descriptor, &d, nullptr, nullptr),
                op < 48 ? GOC_SUCCESS : GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(d, op < 48 ? 0u : 0xdeadbeefu);
    }
}

// RX 9070: four input batches, eight EXEC masks, every integer CMP/CMPX
// predicate and half selector for 16/32-bit operands, seven DPP descriptors.
TEST(IntegerCompare, DppHardwareCorpusAndHostFpState) {
  const uint32_t values[] = {0,          UINT32_MAX, 0x00018000, 0x80000001,
                             0x7fffffff, 0x80008000, 0x7fff7fff, 0xffff0000};
  const uint32_t masks[] = {0xffffffff, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  std::fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    std::fesetround(rounding);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INVALID | FE_INEXACT);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL}) {
        uint64_t hash = UINT64_C(14695981039346656037);
        for (unsigned batch = 0; batch < 4; ++batch)
          for (auto mask : masks)
            for (unsigned op = 0; op < 48; ++op)
              for (unsigned m = 0; m < (op < 24 ? 4u : 1u); ++m)
                for (auto descriptor : goc_test::dpp_modes) {
                  uint32_t av[32], bv[32], output = 0xdeadbeef;
                  for (unsigned lane = 0; lane < 32; ++lane) {
                    av[lane] = values[(lane + batch) % 8];
                    bv[lane] = values[(lane * 3 + batch * 5) % 8];
                  }
                  const uint32_t *a[] = {av}, *b[] = {bv};
                  EXPECT_EQ(goc_test::integer_compare_functions[op](
                                cpu | semantics, mask,
                                descriptor | goc_test::integer_compare_mode(m), &output, a, b),
                            GOC_SUCCESS);
                  hash = (hash ^ output) * UINT64_C(1099511628211);
                }
        EXPECT_EQ(hash, UINT64_C(0xb530b3f6ab40c985)) << cpu << "/" << semantics;
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID | FE_INEXACT);
      }
  }
  std::fesetenv(&saved);
}

TEST(IntegerCompare, HardwarePredicatesWidthsModifiersAndExec) {
  const uint32_t masks[] = {UINT32_MAX, 0, 0x55555555, 0xaaaaaaaa, 1};
  const uint64_t max_cpu = goc_init_cpu_flags();
  for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu)
    for (unsigned mi = 0; mi < 5; ++mi) {
      unsigned variant = 0;
      for (unsigned op = 0; op < 72; ++op)
        for (unsigned m = 0; m < (op < 24 ? 4u : 1u); ++m, ++variant) {
          uint64_t digest = UINT64_C(14695981039346656037);
          for (unsigned start = 0; start < 65536; start += 32) {
            uint32_t words[4][32], d;
            for (unsigned lane = 0; lane < 32; ++lane) {
              uint32_t w[4];
              goc_test::integer_compare_inputs(op / 24, start + lane, w);
              for (unsigned j = 0; j < 4; ++j)
                words[j][lane] = w[j];
            }
            const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]};
            ASSERT_EQ(goc_test::integer_compare_functions[op](
                          cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, masks[mi],
                          goc_test::integer_compare_mode(m), &d, a, b),
                      GOC_SUCCESS);
            for (unsigned byte = 0; byte < 4; ++byte) {
              digest ^= (d >> (8 * byte)) & 255;
              digest *= UINT64_C(1099511628211);
            }
          }
          EXPECT_EQ(digest, goc_test::integer_compare_digests[mi * 144 + variant])
              << cpu << "/" << op << "/" << m;
        }
    }
}

TEST(IntegerCompare, IndependentBoundaryPairsAndRandomInputs) {
  const uint64_t max_cpu = goc_init_cpu_flags();
  for (unsigned op = 0; op < 72; ++op)
    for (unsigned m = 0; m < (op < 24 ? 4u : 1u); ++m)
      for (unsigned start :
           {0u, 32u, 64u, 96u, 128u, 160u, 192u, 224u, 256u, 1024u, 32768u, 65504u}) {
        uint32_t words[4][32], expected = 0;
        for (unsigned lane = 0; lane < 32; ++lane) {
          uint32_t w[4];
          goc_test::integer_compare_inputs(op / 24, start + lane, w);
          for (unsigned j = 0; j < 4; ++j)
            words[j][lane] = w[j];
          expected |= uint32_t(goc_test::integer_compare_reference(op, m, w)) << lane;
        }
        const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]};
        for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu) {
          uint32_t d;
          ASSERT_EQ(goc_test::integer_compare_functions[op](
                        cpu, UINT32_MAX, goc_test::integer_compare_mode(m), &d, a, b),
                    GOC_SUCCESS);
          ASSERT_EQ(d, expected) << op << "/" << m << "/" << start << "/" << cpu;
        }
      }
}

TEST(IntegerCompare, ScalarOutputAliasesSourcesUnalignedAndMasks) {
  const uint64_t max_cpu = goc_init_cpu_flags();
  for (unsigned op = 0; op < 72; ++op)
    for (unsigned m = 0; m < (op < 24 ? 4u : 1u); ++m)
      for (unsigned alias = 0; alias < 3; ++alias) {
        uint32_t initial[4][35];
        for (unsigned j = 0; j < 4; ++j)
          for (unsigned lane = 0; lane < 35; ++lane)
            initial[j][lane] = 0x87654321;
        uint32_t expected = 0;
        for (unsigned lane = 0; lane < 32; ++lane) {
          uint32_t w[4];
          goc_test::integer_compare_inputs(op / 24, 128 + lane, w);
          if (alias == 1) {
            w[2] = w[0];
            w[3] = w[1];
          } else if (alias == 2) {
            w[1] = w[0];
            w[3] = w[2];
          }
          for (unsigned j = 0; j < 4; ++j)
            initial[j][lane + 1] = w[j];
          expected |= uint32_t(goc_test::integer_compare_reference(op, m, w)) << lane;
        }
        for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu)
          for (uint32_t mask : rdna4_exec_masks())
            for (unsigned target = 0; target < 9; ++target) {
              uint32_t words[4][35], outside = 0xdeadbeef;
              std::memcpy(words, initial, sizeof(words));
              unsigned reg = target / 2, lane = target % 2 ? 32 : 1;
              uint32_t *d = target == 8 ? &outside : &words[reg][lane];
              const uint32_t *a[] = {words[0] + 1, words[alias == 2 ? 0 : 1] + 1};
              const uint32_t *b[] = {words[alias == 1 ? 0 : 2] + 1, words[alias == 1   ? 1
                                                                          : alias == 2 ? 2
                                                                                       : 3] +
                                                                        1};
              ASSERT_EQ(goc_test::integer_compare_functions[op](
                            cpu, mask, goc_test::integer_compare_mode(m), d, a, b),
                        GOC_SUCCESS);
              ASSERT_EQ(*d, expected & mask) << op << "/" << m << "/" << cpu << "/" << alias;
              for (unsigned j = 0; j < 4; ++j)
                for (unsigned l = 0; l < 35; ++l)
                  ASSERT_EQ(words[j][l],
                            target != 8 && j == reg && l == lane ? expected & mask : initial[j][l]);
            }
      }
}

TEST(IntegerCompare, ValidationAndHostFpState) {
  fenv_t saved;
  std::fegetenv(&saved);
#if defined(__x86_64__) || defined(_M_X64)
  unsigned original_mxcsr = _mm_getcsr();
#endif
  const uint64_t max_cpu = goc_init_cpu_flags();
  for (unsigned op = 0; op < 72; ++op) {
    uint32_t known = goc_test::integer_compare_mode(op < 24 ? 3 : 0), d = 1;
    EXPECT_EQ(goc_test::integer_compare_functions[op](0, UINT32_C(0), known, &d, nullptr, nullptr),
              GOC_SUCCESS);
    EXPECT_EQ(d, 0u);
    for (unsigned bit = 0; bit < 32; ++bit)
      if (!(known & (1u << bit))) {
        d = 0xdeadbeef;
        EXPECT_EQ(goc_test::integer_compare_functions[op](0, 0, 1u << bit, &d, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(d, 0xdeadbeef);
      }
    for (int round : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
      for (unsigned flush = 0; flush < 2; ++flush) {
        uint32_t words[4][32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          uint32_t w[4];
          goc_test::integer_compare_inputs(op / 24, lane + 32768, w);
          for (unsigned j = 0; j < 4; ++j)
            words[j][lane] = w[j];
        }
        const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]};
        std::fesetround(round);
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
        _mm_setcsr((_mm_getcsr() & ~0x8040u) | (flush ? 0x8040u : 0));
        unsigned before = _mm_getcsr();
#endif
        int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
        for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu)
          EXPECT_EQ(goc_test::integer_compare_functions[op](cpu, UINT32_MAX, known, &d, a, b),
                    GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), round);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
#if defined(__x86_64__) || defined(_M_X64)
        EXPECT_EQ(_mm_getcsr(), before);
#endif
      }
  }
  std::fesetenv(&saved);
#if defined(__x86_64__) || defined(_M_X64)
  _mm_setcsr(original_mxcsr);
#endif
}
