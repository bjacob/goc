// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "exec_masks.h"
#include "float_compare_hardware.h"
#include "float_compare_reference.h"
#include "fp_environment.h"
#include "goc/goc.h"

#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {
void inputs(unsigned op, unsigned start, uint32_t words[4][32]) {
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t w[4];
    goc_test::float_compare_inputs(op / 28, start + lane, w);
    for (unsigned j = 0; j < 4; ++j)
      words[j][lane] = w[j];
  }
}

uint32_t expected(unsigned op, unsigned m, bool flush, const uint32_t words[4][32]) {
  uint32_t result = 0;
  for (unsigned lane = 0; lane < 32; ++lane) {
    uint32_t w[] = {words[0][lane], words[1][lane], words[2][lane], words[3][lane]};
    result |= uint32_t(goc_test::float_compare_reference(op, m, flush, w)) << lane;
  }
  return result;
}
} // namespace

TEST(FloatCompare, HardwarePredicatesModifiersExecAndDenormalModes) {
  const uint32_t masks[] = {UINT32_MAX, 0, 0x55555555, 0xaaaaaaaa, 1};
  const uint64_t max_cpu = goc_init_cpu_flags();
  for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu)
    for (unsigned q = 0; q < 2; ++q)
      for (unsigned mi = 0; mi < 5; ++mi)
        for (unsigned op = 0; op < 84; ++op) {
          uint64_t flags = cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                           GOC_FP_FLUSH_OUTPUT_DENORMALS | (q ? 0 : GOC_FP_FLUSH_INPUT_DENORMALS),
                   digest = goc_test::capture_hash_seed;
          for (unsigned m = 0; m < (op < 28 ? 64u : 16u); ++m)
            for (unsigned start = 0; start < 4096; start += 32) {
              uint32_t words[4][32], d;
              inputs(op, start, words);
              const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]};
              ASSERT_EQ(goc_test::float_compare_functions[op](
                            flags, masks[mi], goc_test::float_compare_mode(m), &d, a, b, nullptr),
                        GOC_SUCCESS);
              digest = goc_test::capture_hash_bytes(digest, d, 4);
            }
          EXPECT_EQ(digest, goc_test::float_compare_digests[(q * 5 + mi) * 84 + op])
              << cpu << "/" << q << "/" << mi << "/" << op;
        }
}

TEST(FloatCompare, IndependentBoundariesAndEveryModifier) {
  const uint64_t max_cpu = goc_init_cpu_flags();
  for (unsigned op = 0; op < 84; ++op)
    for (unsigned m = 0; m < (op < 28 ? 64u : 16u); ++m)
      for (unsigned flush = 0; flush < 2; ++flush)
        for (unsigned start :
             {0u, 32u, 64u, 96u, 128u, 160u, 192u, 224u, 256u, 1024u, 32768u, 65504u}) {
          uint32_t words[4][32];
          inputs(op, start, words);
          uint32_t want = expected(op, m, flush, words);
          const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]};
          for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu) {
            uint32_t d;
            ASSERT_EQ(goc_test::float_compare_functions[op](
                          cpu | (flush ? GOC_FP_FLUSH_INPUT_DENORMALS : 0), UINT32_MAX,
                          goc_test::float_compare_mode(m), &d, a, b, nullptr),
                      GOC_SUCCESS);
            ASSERT_EQ(d, want) << op << "/" << m << "/" << flush << "/" << start << "/" << cpu;
          }
        }
}

TEST(FloatCompare, EveryFp16EncodingAndPredicate) {
  const uint64_t max_cpu = goc_init_cpu_flags();
  for (unsigned op = 0; op < 28; ++op)
    for (unsigned m : {0u, 63u})
      for (unsigned flush = 0; flush < 2; ++flush)
        for (unsigned start = 0; start < 65536; start += 32) {
          uint32_t words[4][32] = {};
          for (unsigned lane = 0; lane < 32; ++lane) {
            unsigned i = start + lane;
            words[0][lane] = i | ((65535 - i) << 16);
            words[2][lane] = i % 2 ? words[0][lane] : (uint32_t(0x7c01) | (i << 16));
          }
          uint32_t want = expected(op, m, flush, words);
          const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]};
          for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu) {
            uint32_t d;
            ASSERT_EQ(goc_test::float_compare_functions[op](
                          cpu | (flush ? GOC_FP_FLUSH_INPUT_DENORMALS : 0), UINT32_MAX,
                          goc_test::float_compare_mode(m), &d, a, b, nullptr),
                      GOC_SUCCESS);
            ASSERT_EQ(d, want) << op << "/" << m << "/" << flush << "/" << start << "/" << cpu;
          }
        }
}

TEST(FloatCompare, ScalarOutputAliasesSourcesUnalignedAndMasks) {
  const uint64_t max_cpu = goc_init_cpu_flags();
  for (unsigned op = 0; op < 84; ++op)
    for (unsigned m : {0u, op < 28 ? 63u : 15u})
      for (unsigned flush = 0; flush < 2; ++flush)
        for (unsigned alias = 0; alias < 3; ++alias) {
          uint32_t initial[4][35], plain[4][32];
          inputs(op, 128, plain);
          for (unsigned j = 0; j < 4; ++j)
            for (unsigned l = 0; l < 35; ++l)
              initial[j][l] = 0xdeadbeef;
          for (unsigned l = 0; l < 32; ++l) {
            if (alias == 1) {
              plain[2][l] = plain[0][l];
              plain[3][l] = plain[1][l];
            } else if (alias == 2) {
              plain[1][l] = plain[0][l];
              plain[3][l] = plain[2][l];
            }
            for (unsigned j = 0; j < 4; ++j)
              initial[j][l + 1] = plain[j][l];
          }
          uint32_t want = expected(op, m, flush, plain);
          for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu)
            for (uint32_t exec_mask : exec_masks())
              for (unsigned target = 0; target < 9; ++target) {
                uint32_t words[4][35], after[4][35], outside = 0;
                std::memcpy(words, initial, sizeof(words));
                std::memcpy(after, initial, sizeof(after));
                unsigned reg = target / 2, lane = target % 2 ? 32 : 1;
                uint32_t *d = target == 8 ? &outside : &words[reg][lane];
                if (target != 8)
                  after[reg][lane] = want & exec_mask;
                const uint32_t *a[] = {words[0] + 1, words[alias == 2 ? 0 : 1] + 1},
                               *b[] = {words[alias == 1 ? 0 : 2] + 1, words[alias == 1   ? 1
                                                                            : alias == 2 ? 2
                                                                                         : 3] +
                                                                          1};
                ASSERT_EQ(goc_test::float_compare_functions[op](
                              cpu | (flush ? GOC_FP_FLUSH_INPUT_DENORMALS : 0), exec_mask,
                              goc_test::float_compare_mode(m), d, a, b, nullptr),
                          GOC_SUCCESS);
                ASSERT_EQ(*d, want & exec_mask)
                    << op << "/" << m << "/" << flush << "/" << alias << "/" << cpu;
                ASSERT_EQ(std::memcmp(words, after, sizeof(words)), 0);
              }
        }
}

TEST(FloatCompare, ValidationAndCompleteHostFpState) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  const uint64_t max_cpu = goc_init_cpu_flags();
  uint32_t result = 0xdeadbeef;
  EXPECT_EQ(goc_v_cmp_lt_i32(GOC_FP_FLUSH_INPUT_DENORMALS, 0, 0, &result, nullptr, nullptr),
            GOC_ERROR_INVALID_FLAGS);
  EXPECT_EQ(result, 0xdeadbeef);
  for (unsigned op = 0; op < 84; ++op) {
    uint32_t known = goc_test::float_compare_mode(op < 28 ? 63 : 15) | GOC_ALU_CLAMP, d = 1;
    EXPECT_EQ(goc_test::float_compare_functions[op](GOC_FP_FLUSH_INPUT_DENORMALS, 0U, known, &d,
                                                    nullptr, nullptr, nullptr),
              GOC_SUCCESS);
    EXPECT_EQ(d, 0u);
    for (unsigned bit = 0; bit < 32; ++bit)
      if (!(known & (1u << bit))) {
        d = 0xdeadbeef;
        EXPECT_EQ(
            goc_test::float_compare_functions[op](0, 0, 1u << bit, &d, nullptr, nullptr, nullptr),
            GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(d, 0xdeadbeef);
      }
    for (unsigned flush = 0; flush < 2; ++flush)
      for (int round : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
        for (unsigned host_flush = 0; host_flush < 2; ++host_flush) {
          uint32_t words[4][32];
          inputs(op, 224, words);
          uint32_t want = expected(op, op < 28 ? 63 : 15, flush, words);
          const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]};
          std::fesetround(round);
          std::feclearexcept(FE_ALL_EXCEPT);
          std::feraiseexcept(FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
          _mm_setcsr((_mm_getcsr() & ~0x8040u) | (host_flush ? 0x8040u : 0));
          unsigned before = _mm_getcsr();
#endif
          int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
          for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu) {
            EXPECT_EQ(goc_test::float_compare_functions[op](
                          cpu | (flush ? GOC_FP_FLUSH_INPUT_DENORMALS : 0), UINT32_MAX, known, &d,
                          a, b, nullptr),
                      GOC_SUCCESS);
            EXPECT_EQ(d, want);
          }
          EXPECT_EQ(std::fegetround(), round);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
#if defined(__x86_64__) || defined(_M_X64)
          EXPECT_EQ(_mm_getcsr(), before);
#endif
        }
  }
}

TEST(FloatCompare, DppPredicatesSelectorsMasksAndAliases) {
  for (unsigned op = 0; op < 56; ++op)
    for (unsigned m = 0; m < (op < 28 ? 64u : 16u); ++m)
      for (bool flush : {false, true})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (auto descriptor : goc_test::dpp_modes)
            for (auto exec_mask :
                 (m == 0 || m == (op < 28 ? 63u : 15u) ? exec_masks()
                                                       : std::vector<uint32_t>{UINT32_MAX}))
              for (unsigned shared = 0; shared < 2; ++shared)
                for (unsigned target = 0; target < 5; ++target) {
                  uint32_t words[2][34], expected[2][34], outside = 0xdeadbeef, want = 0;
                  for (unsigned word = 0; word < 34; ++word) {
                    uint32_t w[4];
                    goc_test::float_compare_inputs(op / 28, m & 1 ? 256 + word : (word * 7) % 256,
                                                   w);
                    words[0][word] = expected[0][word] = w[0];
                    words[1][word] = expected[1][word] = w[2];
                  }
                  unsigned br = shared ? 0 : 1;
                  for (unsigned lane = 0; lane < 32; ++lane) {
                    int source = 0;
                    if (goc_test::dpp_source(descriptor, exec_mask, lane, source)) {
                      uint32_t w[] = {source < 0 ? 0 : words[0][source + 1], 0, words[br][lane + 1],
                                      0};
                      want |= uint32_t(goc_test::float_compare_reference(op, m, flush, w)) << lane;
                    }
                  }
                  unsigned reg = target / 2, word = target % 2 ? 32 : 1;
                  uint32_t *d = target < 4 ? words[reg] + word : &outside;
                  if (target < 4)
                    expected[reg][word] = want;
                  const uint32_t *a[] = {words[0] + 1}, *b[] = {words[br] + 1};
                  uint64_t semantics =
                      op & 1 ? GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT : 0;
                  ASSERT_EQ(goc_test::float_compare_functions[op](
                                cpu | semantics | (flush ? GOC_FP_FLUSH_INPUT_DENORMALS : 0),
                                exec_mask, descriptor | goc_test::float_compare_mode(m), d, a, b,
                                nullptr),
                            GOC_SUCCESS);
                  ASSERT_EQ(*d, want)
                      << op << "/" << m << "/" << cpu << "/" << descriptor << "/" << exec_mask;
                  ASSERT_EQ(std::memcmp(words, expected, sizeof(words)), 0);
                }
}

TEST(FloatCompare, DppValidationAndZeroExec) {
  for (unsigned op = 0; op < 84; ++op)
    for (auto descriptor : goc_test::dpp_modes) {
      uint32_t d = 0xdeadbeef;
      auto fn = goc_test::float_compare_functions[op];
      for (auto invalid : {1ULL << 36, 1ULL << 2}) {
        EXPECT_EQ(fn(0, 0, descriptor | invalid, &d, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(d, 0xdeadbeefu);
      }
      EXPECT_EQ(fn(0, 0U, descriptor, &d, nullptr, nullptr, nullptr),
                op < 56 ? GOC_SUCCESS : GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(d, op < 56 ? 0u : 0xdeadbeefu);
    }
}

// RX 9070: both denormal modes, four input batches, eight EXEC masks, all
// FP16/FP32 CMP/CMPX predicates, source modifier cases and seven DPP descriptors.
TEST(FloatCompare, DppHardwareCorpusAndHostFpState) {
  const uint32_t values[2][8] = {
      {0x80000000, 0x80010001, 0x83ff03ff, 0x84000400, 0xbc003c00, 0xfc007c00, 0xfe007e00,
       0xfc017c01},
      {0, 0x80000000, 1, 0x80000001, 0x3f800000, 0xbf800000, 0x7f800000, 0x7f800001}};
  const uint32_t masks[] = {0xffffffff, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  const unsigned modes[] = {0, 1, 2, 4, 8, 15, 16, 32, 48, 49, 50, 63};
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    std::fesetround(rounding);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INVALID | FE_INEXACT);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL}) {
        uint64_t hash = goc_test::capture_hash_seed;
        for (bool flush : {true, false})
          for (unsigned batch = 0; batch < 4; ++batch)
            for (auto exec_mask : masks)
              for (unsigned op = 0; op < 56; ++op)
                for (unsigned variant = 0; variant < (op < 28 ? 12u : 6u); ++variant)
                  for (auto descriptor : goc_test::dpp_modes) {
                    uint32_t av[32], bv[32], output = 0xdeadbeef;
                    for (unsigned lane = 0; lane < 32; ++lane) {
                      av[lane] = values[op / 28][(lane + batch) % 8];
                      bv[lane] = values[op / 28][(lane * 3 + batch * 5) % 8];
                    }
                    const uint32_t *a[] = {av}, *b[] = {bv};
                    EXPECT_EQ(goc_test::float_compare_functions[op](
                                  cpu | semantics | (flush ? GOC_FP_FLUSH_INPUT_DENORMALS : 0),
                                  exec_mask,
                                  descriptor | goc_test::float_compare_mode(modes[variant]),
                                  &output, a, b, nullptr),
                              GOC_SUCCESS);
                    hash = goc_test::capture_hash_word(hash, output);
                  }
        EXPECT_EQ(hash, 0xbe15e5a37f47d025ULL) << cpu << "/" << semantics;
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID | FE_INEXACT);
      }
  }
}
