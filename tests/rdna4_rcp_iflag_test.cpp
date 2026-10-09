// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_rcp_iflag_hardware.h"
#include "rdna4_rcp_iflag_reference.h"

#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

TEST(RcpIflag, HardwareNumericAndStickyExceptions) {
  const uint32_t masks[] = {UINT32_MAX, 0, 0x55555555, 0xaaaaaaaa, 1, 0x80000000},
                 seeds[] = {0, 0x15, 0x55};
  const uint64_t max_cpu = goc_init_cpu_flags();
  for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu)
    for (unsigned si = 0; si < 3; ++si)
      for (unsigned mi = 0; mi < 6; ++mi)
        for (unsigned m = 0; m < 32; ++m) {
          uint64_t digest = UINT64_C(14695981039346656037),
                   flags = cpu | (mi & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                           (mi & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0);
          for (unsigned wave = 0; wave < 128; ++wave) {
            uint32_t a[32], d[32], status;
            for (unsigned lane = 0; lane < 32; ++lane) {
              a[lane] = goc_test::rcp_iflag_input(wave * 32 + lane);
              d[lane] = 0xcafebeef;
            }
            const uint32_t *ap[] = {a};
            uint32_t *dp[] = {d};
            ASSERT_EQ(goc_rdna4_v_rcp_iflag_f32(flags, masks[mi], goc_test::rcp_iflag_mode(m), dp,
                                                ap, &status, seeds[si]),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 32; ++lane) {
              if ((masks[mi] >> lane) & 1)
                ASSERT_TRUE(
                    goc_test::rcp_iflag_close(d[lane], goc_test::rcp_iflag_reference(a[lane], m)));
              else
                ASSERT_EQ(d[lane], 0xcafebeef);
            }
            for (unsigned shift = 0; shift < 32; shift += 8) {
              digest ^= (status >> shift) & 255;
              digest *= UINT64_C(1099511628211);
            }
          }
          EXPECT_EQ(digest, goc_test::rcp_iflag_status_digests[(si * 6 + mi) * 32 + m])
              << cpu << "/" << si << "/" << mi << "/" << m;
        }
  for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu)
    for (unsigned m = 0; m < 32; ++m)
      for (unsigned group = 0; group < 2; ++group) {
        uint32_t a[32], d[32], status;
        for (unsigned lane = 0; lane < 32; ++lane)
          a[lane] = goc_test::rcp_iflag_input(group ? 2048 + lane : lane * 32);
        const uint32_t *ap[] = {a};
        uint32_t *dp[] = {d};
        ASSERT_EQ(goc_rdna4_v_rcp_iflag_f32(cpu, UINT32_MAX, goc_test::rcp_iflag_mode(m), dp, ap,
                                            &status, 0),
                  GOC_SUCCESS);
        for (unsigned lane = 0; lane < 32; ++lane)
          ASSERT_TRUE(
              goc_test::rcp_iflag_close(d[lane], goc_test::rcp_iflag_numeric[m][group * 32 + lane]))
              << cpu << "/" << m << "/" << group << "/" << lane;
      }
}

TEST(RcpIflag, ModifiersExecAliasesAndUnalignedStorage) {
  const uint64_t max_cpu = goc_init_cpu_flags();
  for (unsigned m = 0; m < 32; ++m)
    for (unsigned alias = 0; alias < 2; ++alias)
      for (uint32_t mask : rdna4_exec_masks())
        for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu)
          for (unsigned target = 0; target < 5; ++target) {
            uint32_t words[2][35], initial[2][35], outside = 0;
            for (unsigned j = 0; j < 2; ++j)
              for (unsigned lane = 0; lane < 35; ++lane)
                words[j][lane] = 0xdeadbeef;
            for (unsigned lane = 0; lane < 32; ++lane)
              words[0][lane + 1] = goc_test::rcp_iflag_input(lane * 32);
            std::memcpy(initial, words, sizeof(words));
            uint32_t seed = target & 1 ? 0xa5a50055 : 0xa5a50015;
            uint32_t wanted_status = goc_test::rcp_iflag_status(words[0] + 1, mask, m, seed);
            unsigned status_reg = target / 2, status_lane = target % 2 ? 32 : 1;
            uint32_t *status = target == 4 ? &outside : &words[status_reg][status_lane],
                     *dp[] = {words[alias ? 0 : 1] + 1};
            const uint32_t *ap[] = {words[0] + 1};
            ASSERT_EQ(goc_rdna4_v_rcp_iflag_f32(cpu, mask, goc_test::rcp_iflag_mode(m), dp, ap,
                                                status, seed),
                      GOC_SUCCESS);
            ASSERT_EQ(*status, wanted_status);
            for (unsigned j = 0; j < 2; ++j)
              for (unsigned lane = 0; lane < 35; ++lane) {
                if (target != 4 && j == status_reg && lane == status_lane) {
                  ASSERT_EQ(words[j][lane], wanted_status);
                  continue;
                }
                if (j == (alias ? 0u : 1u) && lane >= 1 && lane <= 32 && ((mask >> (lane - 1)) & 1))
                  ASSERT_TRUE(goc_test::rcp_iflag_close(
                      words[j][lane], goc_test::rcp_iflag_reference(initial[0][lane], m)));
                else
                  ASSERT_EQ(words[j][lane], initial[j][lane]);
              }
          }
}

TEST(RcpIflag, ActiveSubnormalsClampAndStickyState) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint32_t a[32], d[32], status;
    for (auto &v : a)
      v = 0x3f800000;
    a[1] = 1;
    const uint32_t *ap[] = {a};
    uint32_t *dp[] = {d};
    ASSERT_EQ(goc_rdna4_v_rcp_iflag_f32(cpu, 1, 0, dp, ap, &status, 0), GOC_SUCCESS);
    EXPECT_EQ(status, 0u);
    ASSERT_EQ(goc_rdna4_v_rcp_iflag_f32(cpu, 2, 0, dp, ap, &status, 0), GOC_SUCCESS);
    EXPECT_EQ(status, GOC_RDNA4_EXCEPTION_INT_DIV0);
    EXPECT_EQ(d[1], 0x7f800000u);
    ASSERT_EQ(goc_rdna4_v_rcp_iflag_f32(cpu, 2, GOC_ALU_CLAMP, dp, ap, &status, 0), GOC_SUCCESS);
    EXPECT_EQ(status, 0u);
    EXPECT_EQ(d[1], 0x3f800000u);
    ASSERT_EQ(goc_rdna4_v_rcp_iflag_f32(cpu, 2, GOC_ALU_CLAMP, dp, ap, &status, 0x55), GOC_SUCCESS);
    EXPECT_EQ(status, 0x55u);
  }
}

TEST(RcpIflag, ValidationAndHostRounding) {
  uint32_t status = 0xdeadbeef, dwords[32] = {};
  uint32_t *dp[] = {dwords};
  EXPECT_EQ(goc_rdna4_v_rcp_iflag_f32(0, UINT32_C(0), 0, nullptr, nullptr, &status, 0x12345678),
            GOC_SUCCESS);
  EXPECT_EQ(status, 0x12345678u);
  const uint32_t known = goc_test::rcp_iflag_mode(31);
  for (unsigned bit = 0; bit < 32; ++bit)
    if (!(known & (1u << bit))) {
      status = 0xdeadbeef;
      EXPECT_EQ(goc_rdna4_v_rcp_iflag_f32(0, 0, 1u << bit, dp, nullptr, &status, 0),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(status, 0xdeadbeef);
      EXPECT_EQ(dwords[0], 0u);
    }
  EXPECT_EQ(goc_rdna4_v_rcp_iflag_f32(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0,
                                      dp, nullptr, &status, 0),
            GOC_ERROR_UNSUPPORTED_SEMANTICS);
  EXPECT_EQ(status, 0xdeadbeef);
  EXPECT_EQ(goc_rdna4_v_rcp_iflag_f32(UINT64_C(1) << 63, 0, 0, dp, nullptr, &status, 0),
            GOC_ERROR_INVALID_FLAGS);
  EXPECT_EQ(status, 0xdeadbeef);
  fenv_t saved;
  std::fegetenv(&saved);
  std::fesetround(FE_TONEAREST);
  uint32_t a[32];
  for (auto &v : a)
    v = 0x40400000;
  const uint32_t *ap[] = {a};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    EXPECT_EQ(goc_rdna4_v_rcp_iflag_f32(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, known, dp,
                                        ap, &status, 0),
              GOC_SUCCESS);
    EXPECT_EQ(std::fegetround(), FE_TONEAREST);
  }
  std::fesetenv(&saved);
}

TEST(RcpIflag, DppModifiersMasksAndStatusAliases) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned m = 0; m < 32; ++m)
      for (auto descriptor : goc_test::dpp_modes)
        for (auto mask : rdna4_exec_masks())
          for (unsigned target = 0; target < 2; ++target)
            for (unsigned status_target = 0; status_target < 3; ++status_target) {
              uint32_t words[2][34], expected[2][34], permuted[32] = {}, status = 0xdeadbeef;
              for (unsigned reg = 0; reg < 2; ++reg)
                for (unsigned word = 0; word < 34; ++word)
                  words[reg][word] = expected[reg][word] =
                      reg ? 0xdead0000u + word : goc_test::rcp_iflag_input((word - 1) * 32);
              uint32_t effective = 0;
              for (unsigned lane = 0; lane < 32; ++lane) {
                int source = 0;
                if (goc_test::dpp_source(descriptor, mask, lane, source)) {
                  effective |= uint32_t(1) << lane;
                  permuted[lane] = source < 0 ? 0 : words[0][source + 1];
                  expected[target][lane + 1] = goc_test::rcp_iflag_reference(permuted[lane], m);
                }
              }
              uint32_t seed = m & 1 ? 0x55 : 0x15;
              uint32_t want = goc_test::rcp_iflag_status(permuted, effective, m, seed);
              if (status_target < 2)
                expected[status_target][14] = want;
              const uint32_t *a[] = {words[0] + 1};
              uint32_t *d[] = {words[target] + 1};
              ASSERT_EQ(goc_rdna4_v_rcp_iflag_f32(
                            cpu, mask, descriptor | goc_test::rcp_iflag_mode(m), d, a,
                            status_target < 2 ? words[status_target] + 14 : &status, seed),
                        GOC_SUCCESS);
              EXPECT_EQ(status, status_target < 2 ? 0xdeadbeef : want);
              for (unsigned reg = 0; reg < 2; ++reg)
                for (unsigned word = 0; word < 34; ++word) {
                  if (reg == status_target && word == 14)
                    ASSERT_EQ(words[reg][word], want);
                  else if (reg == target && word >= 1 && word <= 32 &&
                           ((effective >> (word - 1)) & 1))
                    ASSERT_TRUE(goc_test::rcp_iflag_close(words[reg][word], expected[reg][word]));
                  else
                    ASSERT_EQ(words[reg][word], expected[reg][word]);
                }
            }
}

TEST(RcpIflag, DppValidationAndZeroExec) {
  for (auto descriptor : goc_test::dpp_modes) {
    uint32_t status = 0xdeadbeef;
    for (auto invalid : {UINT64_C(1) << 36, UINT64_C(1) << 1})
      for (uint32_t mask : {UINT32_C(0), UINT32_MAX}) {
        EXPECT_EQ(goc_rdna4_v_rcp_iflag_f32(0, mask, descriptor | invalid, nullptr, nullptr,
                                            &status, 0x15),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(status, 0xdeadbeefu);
      }
    EXPECT_EQ(goc_rdna4_v_rcp_iflag_f32(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0,
                                        descriptor, nullptr, nullptr, &status, 0x15),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(status, 0xdeadbeefu);
    EXPECT_EQ(goc_rdna4_v_rcp_iflag_f32(0, UINT32_C(0), descriptor, nullptr, nullptr, &status,
                                        0x12345678),
              GOC_SUCCESS);
    EXPECT_EQ(status, 0x12345678u);
  }
}

// RX 9070: two denormal modes, three sticky-status seeds, eight EXEC masks,
// every modifier and seven DPP descriptors. Powers of two give exact finite
// reciprocals; NaN payloads are canonicalized, while status and zeros are exact.
TEST(RcpIflag, DppHardwareCorpus) {
  const uint32_t values[] = {0,          0x80000000, 1,          0x807fffff,
                             0x3f800000, 0xc0000000, 0x7f800000, 0x7fc00000};
  const uint32_t masks[] = {0xffffffff, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint64_t fp : {GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS, UINT64_C(0)})
      for (uint32_t seed : {0u, 0x15u, 0x55u})
        for (auto mask : masks)
          for (unsigned m = 0; m < 32; ++m)
            for (auto descriptor : goc_test::dpp_modes) {
              uint32_t av[32], output[32], status = 0;
              for (unsigned lane = 0; lane < 32; ++lane) {
                av[lane] = values[lane % 8];
                output[lane] = 0xdead0000u + lane;
              }
              const uint32_t *a[] = {av};
              uint32_t *d[] = {output};
              ASSERT_EQ(goc_rdna4_v_rcp_iflag_f32(cpu | fp, mask,
                                                  descriptor | goc_test::rcp_iflag_mode(m), d, a,
                                                  &status, seed),
                        GOC_SUCCESS);
              for (auto word : output) {
                if ((word & 0x7fffffff) > 0x7f800000)
                  word = 0x7fc00000;
                hash = (hash ^ word) * UINT64_C(1099511628211);
              }
              hash = (hash ^ status) * UINT64_C(1099511628211);
            }
    EXPECT_EQ(hash, UINT64_C(0x0c1abc142a4dc525)) << cpu;
  }
}
