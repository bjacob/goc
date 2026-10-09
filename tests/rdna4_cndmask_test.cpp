// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_cndmask_hardware.h"
#include "rdna4_cndmask_reference.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using Fn = decltype(&goc_rdna4_v_cndmask_b32);
const Fn functions[] = {goc_rdna4_v_cndmask_b32, goc_rdna4_v_cndmask_b16};

} // namespace

TEST(Cndmask, HardwareAllEncodingsAndModifiers) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned variant = 0; variant < 288; ++variant) {
      bool half = variant >= 32;
      unsigned m = (half ? variant - 32 : variant) / 2;
      uint32_t condition = variant & 1 ? UINT32_MAX : 0;
      uint64_t digest = UINT64_C(14695981039346656037);
      for (unsigned start = 0; start < 65536; start += 32) {
        uint32_t words[3][32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          unsigned i = start + lane;
          words[0][lane] = i | ((65535 - i) << 16);
          words[1][lane] = (i * 0x7395a831u) ^ 0xa7925163u;
          words[2][lane] = 0xcafebeef;
        }
        const uint32_t *a[] = {words[0]}, *b[] = {words[1]};
        uint32_t *d[] = {words[2]};
        ASSERT_EQ(functions[half](cpu, UINT32_MAX, goc_test::cndmask_mode(m), d, a, b, condition),
                  GOC_SUCCESS);
        for (uint32_t value : words[2])
          for (unsigned byte = 0; byte < 4; ++byte) {
            digest ^= (value >> (8 * byte)) & 255;
            digest *= UINT64_C(1099511628211);
          }
      }
      EXPECT_EQ(digest, goc_test::cndmask_digests[variant]) << cpu << "/" << variant;
    }
}

TEST(Cndmask, EveryModifierMaskAliasAndUnalignedStorage) {
  for (unsigned half = 0; half < 2; ++half)
    for (unsigned m = 0; m < (half ? 128u : 16u); ++m)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned alias = 0; alias < 2; ++alias)
          for (unsigned target = 0; target < 3; ++target)
            for (uint64_t mask : rdna4_exec_masks()) {
              uint32_t words[3][35], expected[3][35];
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned lane = 0; lane < 35; ++lane)
                  words[reg][lane] = expected[reg][lane] =
                      (lane * 0x7395a831u) ^ (reg * 0xa7925163u);
              uint32_t condition = (m * 0x9e3779b9u) ^ 0x96969696u;
              unsigned source_b = alias ? 0 : 1;
              for (unsigned lane = 0; lane < 32; ++lane)
                if ((mask >> lane) & 1)
                  expected[target][lane + 1] = goc_test::cndmask_reference(
                      half, words[0][lane + 1], words[source_b][lane + 1], words[target][lane + 1],
                      m, (condition >> lane) & 1);
              const uint32_t *a[] = {words[0] + 1}, *b[] = {words[source_b] + 1};
              uint32_t *d[] = {words[target] + 1};
              ASSERT_EQ(functions[half](cpu, mask, goc_test::cndmask_mode(m), d, a, b, condition),
                        GOC_SUCCESS);
              for (unsigned reg = 0; reg < 3; ++reg)
                ASSERT_TRUE(std::equal(words[reg], words[reg] + 35, expected[reg]))
                    << half << "/" << m << "/" << cpu << "/" << target;
            }
}

TEST(Cndmask, EveryConditionBitIndependentOfExec) {
  for (unsigned half = 0; half < 2; ++half)
    for (unsigned m = 0; m < (half ? 128u : 16u); ++m)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint64_t condition : rdna4_exec_masks()) {
          uint32_t words[3][32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            words[0][lane] = 0x7c017f81u + lane;
            words[1][lane] = 0xff818123u - lane;
            words[2][lane] = 0x12345678;
          }
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]};
          uint32_t *d[] = {words[2]};
          uint32_t mask = (m & 1) ? 0xaaaaaaaa : 0x55555555;
          ASSERT_EQ(
              functions[half](cpu, mask, goc_test::cndmask_mode(m), d, a, b, uint32_t(condition)),
              GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            uint32_t want =
                (mask >> lane) & 1
                    ? goc_test::cndmask_reference(half, words[0][lane], words[1][lane], 0x12345678,
                                                  m, (condition >> lane) & 1)
                    : 0x12345678;
            ASSERT_EQ(words[2][lane], want) << half << "/" << m << "/" << cpu << "/" << lane;
          }
        }
}

TEST(Cndmask, InvalidFlagsAndHostFpState) {
  fenv_t saved;
  std::fegetenv(&saved);
  for (unsigned half = 0; half < 2; ++half) {
    uint32_t known = goc_test::cndmask_mode(half ? 127 : 15);
    EXPECT_EQ(functions[half](0, UINT64_C(0xffffffff00000000), known, nullptr, nullptr, nullptr, 0),
              GOC_SUCCESS);
    for (unsigned bit = 0; bit < 32; ++bit) {
      if (!(known & (1u << bit))) {
        EXPECT_EQ(functions[half](0, 0, 1u << bit, nullptr, nullptr, nullptr, 0),
                  GOC_ERROR_INVALID_FLAGS);
      }
    }
    EXPECT_EQ(functions[half](GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr,
                              nullptr, nullptr, 0),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        uint32_t a0[32], b0[32], d0[32];
        std::fill_n(a0, 32, 0x7f800001);
        std::fill_n(b0, 32, 0x7c018001);
        std::fill_n(d0, 32, 0xaabbccdd);
        const uint32_t *a[] = {a0}, *b[] = {b0};
        uint32_t *d[] = {d0};
        std::fesetround(rounding);
        std::feclearexcept(FE_ALL_EXCEPT);
        std::feraiseexcept(FE_INVALID | FE_DIVBYZERO);
        int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
        EXPECT_EQ(functions[half](cpu, UINT32_MAX, known, d, a, b, 0xaaaaaaaa), GOC_SUCCESS);
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
      }
  }
  std::fesetenv(&saved);
}
