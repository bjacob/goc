// SPDX-License-Identifier: MIT

#include "goc/goc.h"
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
                          cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                          UINT64_C(0xffffffff00000000) | masks[mi],
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
          for (uint64_t mask : rdna4_exec_masks())
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
              ASSERT_EQ(*d, expected & uint32_t(mask))
                  << op << "/" << m << "/" << cpu << "/" << alias;
              for (unsigned j = 0; j < 4; ++j)
                for (unsigned l = 0; l < 35; ++l)
                  ASSERT_EQ(words[j][l], target != 8 && j == reg && l == lane
                                             ? expected & uint32_t(mask)
                                             : initial[j][l]);
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
    EXPECT_EQ(goc_test::integer_compare_functions[op](0, UINT64_C(0xffffffff00000000), known, &d,
                                                      nullptr, nullptr),
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
