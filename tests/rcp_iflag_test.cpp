// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "exec_masks.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "rcp_iflag_hardware.h"
#include "rcp_iflag_reference.h"
#include "unary_exact_hardware.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

TEST(RcpIflag, HardwareNumericAndStickyExceptions) {
  const uint32_t masks[] = {UINT32_MAX, 0, 0x55555555, 0xaaaaaaaa, 1, 0x80000000},
                 seeds[] = {0, 0x15, 0x55};
  const uint64_t max_cpu = goc_init_cpu_flags();
  for (uint64_t semantics : {uint64_t(0), GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT})
    for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu)
      for (unsigned si = 0; si < 3; ++si)
        for (unsigned mi = 0; mi < 6; ++mi)
          for (unsigned m = 0; m < 32; ++m) {
            uint64_t digest = goc_test::capture_hash_seed,
                     flags = cpu | semantics | (mi & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                             (mi & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0);
            for (unsigned wave = 0; wave < 128; ++wave) {
              uint32_t a[32], d[32], status = seeds[si];
              for (unsigned lane = 0; lane < 32; ++lane) {
                a[lane] = goc_test::rcp_iflag_input(wave * 32 + lane);
                d[lane] = 0xcafebeef;
              }
              const uint32_t *ap[] = {a};
              uint32_t *dp[] = {d};
              ASSERT_EQ(goc_v_rcp_iflag_f32_rdna4(flags, masks[mi], goc_test::rcp_iflag_mode(m), dp,
                                                  ap, &status),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 32; ++lane) {
                if ((masks[mi] >> lane) & 1)
                  ASSERT_TRUE(goc_test::rcp_iflag_close(d[lane],
                                                        goc_test::rcp_iflag_reference(a[lane], m)));
                else
                  ASSERT_EQ(d[lane], 0xcafebeef);
              }
              digest = goc_test::capture_hash_bytes(digest, status, 4);
            }
            EXPECT_EQ(digest, goc_test::rcp_iflag_status_digests[(si * 6 + mi) * 32 + m])
                << cpu << "/" << si << "/" << mi << "/" << m;
          }
  for (uint64_t semantics : {uint64_t(0), GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT})
    for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu)
      for (unsigned m = 0; m < 32; ++m)
        for (unsigned group = 0; group < 2; ++group) {
          uint32_t a[32], d[32], status = 0;
          for (unsigned lane = 0; lane < 32; ++lane)
            a[lane] = goc_test::rcp_iflag_input(group ? 2048 + lane : lane * 32);
          const uint32_t *ap[] = {a};
          uint32_t *dp[] = {d};
          ASSERT_EQ(goc_v_rcp_iflag_f32_rdna4(cpu | semantics, UINT32_MAX,
                                              goc_test::rcp_iflag_mode(m), dp, ap, &status),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane)
            ASSERT_TRUE(goc_test::rcp_iflag_close(
                d[lane], goc_test::rcp_iflag_numeric[m][group * 32 + lane]))
                << cpu << "/" << m << "/" << group << "/" << lane;
        }
}

TEST(RcpIflag, ModifiersExecAliasesAndUnalignedStorage) {
  const uint64_t max_cpu = goc_init_cpu_flags();
  for (unsigned m = 0; m < 32; ++m)
    for (unsigned alias = 0; alias < 2; ++alias)
      for (uint32_t exec_mask : exec_masks())
        for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu)
          for (unsigned target = 0; target < 5; ++target) {
            uint32_t words[2][35], initial[2][35], outside = 0;
            for (unsigned j = 0; j < 2; ++j)
              for (unsigned lane = 0; lane < 35; ++lane)
                words[j][lane] = 0xdeadbeef;
            for (unsigned lane = 0; lane < 32; ++lane)
              words[0][lane + 1] = goc_test::rcp_iflag_input(lane * 32);
            std::memcpy(initial, words, sizeof(words));
            unsigned status_reg = target / 2, status_lane = target % 2 ? 32 : 1;
            uint32_t *status = target == 4 ? &outside : &words[status_reg][status_lane],
                     *dp[] = {words[alias ? 0 : 1] + 1};
            uint32_t wanted_status =
                goc_test::rcp_iflag_status(words[0] + 1, exec_mask, m, *status);
            const uint32_t *ap[] = {words[0] + 1};
            ASSERT_EQ(goc_v_rcp_iflag_f32_rdna4(cpu, exec_mask, goc_test::rcp_iflag_mode(m), dp, ap,
                                                status),
                      GOC_SUCCESS);
            ASSERT_EQ(*status, wanted_status);
            for (unsigned j = 0; j < 2; ++j)
              for (unsigned lane = 0; lane < 35; ++lane) {
                if (target != 4 && j == status_reg && lane == status_lane) {
                  ASSERT_EQ(words[j][lane], wanted_status);
                  continue;
                }
                if (j == (alias ? 0u : 1u) && lane >= 1 && lane <= 32 &&
                    ((exec_mask >> (lane - 1)) & 1))
                  ASSERT_TRUE(goc_test::rcp_iflag_close(
                      words[j][lane], goc_test::rcp_iflag_reference(initial[0][lane], m)));
                else
                  ASSERT_EQ(words[j][lane], initial[j][lane]);
              }
          }
}

TEST(RcpIflag, ActiveSubnormalsClampAndStickyState) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint32_t a[32], d[32], status = 0;
    for (auto &v : a)
      v = 0x3f800000;
    a[1] = 1;
    const uint32_t *ap[] = {a};
    uint32_t *dp[] = {d};
    ASSERT_EQ(goc_v_rcp_iflag_f32_rdna4(cpu, 1, 0, dp, ap, &status), GOC_SUCCESS);
    EXPECT_EQ(status, 0u);
    ASSERT_EQ(goc_v_rcp_iflag_f32_rdna4(cpu, 2, 0, dp, ap, &status), GOC_SUCCESS);
    EXPECT_EQ(status, GOC_EXCEPTION_INT_DIV0);
    EXPECT_EQ(d[1], 0x7f800000u);
    ASSERT_EQ(goc_v_rcp_iflag_f32_rdna4(cpu, 2, GOC_ALU_CLAMP, dp, ap, &status), GOC_SUCCESS);
    EXPECT_EQ(status, GOC_EXCEPTION_INT_DIV0);
    EXPECT_EQ(d[1], 0x3f800000u);
    status = 0x55;
    ASSERT_EQ(goc_v_rcp_iflag_f32_rdna4(cpu, 2, GOC_ALU_CLAMP, dp, ap, &status), GOC_SUCCESS);
    EXPECT_EQ(status, 0x55u);
  }
}

TEST(RcpIflag, ValidationAndHostRounding) {
  uint32_t status = 0xdeadbeef, dwords[32] = {};
  uint32_t *dp[] = {dwords};
  EXPECT_EQ(goc_v_rcp_iflag_f32_rdna4(0, 0U, 0, nullptr, nullptr, &status), GOC_SUCCESS);
  EXPECT_EQ(status, 0xdeadbeefu);
  const uint32_t known = goc_test::rcp_iflag_mode(31);
  for (unsigned bit = 0; bit < 32; ++bit)
    if (!(known & (1u << bit))) {
      status = 0xdeadbeef;
      EXPECT_EQ(goc_v_rcp_iflag_f32_rdna4(0, 0, 1u << bit, dp, nullptr, &status),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(status, 0xdeadbeef);
      EXPECT_EQ(dwords[0], 0u);
    }
  EXPECT_EQ(goc_v_rcp_iflag_f32_rdna4(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0,
                                      dp, nullptr, &status),
            GOC_SUCCESS);
  EXPECT_EQ(status, 0xdeadbeef);
  EXPECT_EQ(goc_v_rcp_iflag_f32_rdna4(1ULL << 63, 0, 0, dp, nullptr, &status),
            GOC_ERROR_INVALID_FLAGS);
  EXPECT_EQ(status, 0xdeadbeef);
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  std::fesetround(FE_TONEAREST);
  uint32_t a[32];
  for (auto &v : a)
    v = 0x40400000;
  const uint32_t *ap[] = {a};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    EXPECT_EQ(goc_v_rcp_iflag_f32_rdna4(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, known, dp,
                                        ap, &status),
              GOC_SUCCESS);
    EXPECT_EQ(std::fegetround(), FE_TONEAREST);
  }
}

TEST(RcpIflag, DppModifiersMasksAndStatusAliases) {
  for (uint64_t semantics : {uint64_t(0), GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned m = 0; m < 32; ++m)
        for (auto descriptor : goc_test::dpp_modes)
          for (auto exec_mask : exec_masks())
            for (unsigned target = 0; target < 2; ++target)
              for (unsigned status_target = 0; status_target < 3; ++status_target) {
                uint32_t words[2][34], expected[2][34], permuted[32] = {}, status = 0xdeadbeef;
                for (unsigned reg = 0; reg < 2; ++reg)
                  for (unsigned word = 0; word < 34; ++word)
                    words[reg][word] = expected[reg][word] =
                        reg ? 0xdead0000u + word : goc_test::rcp_iflag_input((word - 1) * 32);
                uint32_t reference_exec_mask = 0;
                for (unsigned lane = 0; lane < 32; ++lane) {
                  int source = 0;
                  if (goc_test::dpp_source(descriptor, exec_mask, lane, source)) {
                    reference_exec_mask |= uint32_t(1) << lane;
                    permuted[lane] = source < 0 ? 0 : words[0][source + 1];
                    expected[target][lane + 1] = goc_test::rcp_iflag_reference(permuted[lane], m);
                  }
                }
                uint32_t seed = status_target < 2 ? words[status_target][14] : status;
                uint32_t want = goc_test::rcp_iflag_status(permuted, reference_exec_mask, m, seed);
                if (status_target < 2)
                  expected[status_target][14] = want;
                const uint32_t *a[] = {words[0] + 1};
                uint32_t *d[] = {words[target] + 1};
                ASSERT_EQ(goc_v_rcp_iflag_f32_rdna4(
                              cpu | semantics, exec_mask, descriptor | goc_test::rcp_iflag_mode(m),
                              d, a, status_target < 2 ? words[status_target] + 14 : &status),
                          GOC_SUCCESS);
                EXPECT_EQ(status, status_target < 2 ? 0xdeadbeef : want);
                for (unsigned reg = 0; reg < 2; ++reg)
                  for (unsigned word = 0; word < 34; ++word) {
                    if (reg == status_target && word == 14)
                      ASSERT_EQ(words[reg][word], want);
                    else if (reg == target && word >= 1 && word <= 32 &&
                             ((reference_exec_mask >> (word - 1)) & 1))
                      ASSERT_TRUE(goc_test::rcp_iflag_close(words[reg][word], expected[reg][word]));
                    else
                      ASSERT_EQ(words[reg][word], expected[reg][word]);
                  }
              }
}

TEST(RcpIflag, DppValidationAndZeroExec) {
  for (auto descriptor : goc_test::dpp_modes) {
    uint32_t status = 0xdeadbeef;
    for (auto invalid : {1ULL << 36, 1ULL << 1})
      for (uint32_t exec_mask : {0U, UINT32_MAX}) {
        EXPECT_EQ(goc_v_rcp_iflag_f32_rdna4(0, exec_mask, descriptor | invalid, nullptr, nullptr,
                                            &status),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(status, 0xdeadbeefu);
      }
    EXPECT_EQ(goc_v_rcp_iflag_f32_rdna4(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0,
                                        descriptor, nullptr, nullptr, &status),
              GOC_SUCCESS);
    EXPECT_EQ(status, 0xdeadbeefu);
    EXPECT_EQ(goc_v_rcp_iflag_f32_rdna4(0, 0U, descriptor, nullptr, nullptr, &status), GOC_SUCCESS);
    EXPECT_EQ(status, 0xdeadbeefu);
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
    uint64_t hash = goc_test::capture_hash_seed;
    for (uint64_t fp : std::initializer_list<uint64_t>{
             GOC_FP_FLUSH_INPUT_DENORMALS | GOC_FP_FLUSH_OUTPUT_DENORMALS, 0ULL})
      for (uint32_t seed : {0u, 0x15u, 0x55u})
        for (auto exec_mask : masks)
          for (unsigned m = 0; m < 32; ++m)
            for (auto descriptor : goc_test::dpp_modes) {
              uint32_t av[32], output[32], status = seed;
              for (unsigned lane = 0; lane < 32; ++lane) {
                av[lane] = values[lane % 8];
                output[lane] = 0xdead0000u + lane;
              }
              const uint32_t *a[] = {av};
              uint32_t *d[] = {output};
              ASSERT_EQ(goc_v_rcp_iflag_f32_rdna4(cpu | fp, exec_mask,
                                                  descriptor | goc_test::rcp_iflag_mode(m), d, a,
                                                  &status),
                        GOC_SUCCESS);
              for (auto word : output) {
                if ((word & 0x7fffffff) > 0x7f800000)
                  word = 0x7fc00000;
                hash = goc_test::capture_hash_word(hash, word);
              }
              hash = goc_test::capture_hash_word(hash, status);
            }
    EXPECT_EQ(hash, 0x0c1abc142a4dc525ULL) << cpu;
  }
}

// Reporting opt-out must preserve the numerical result on every CPU path.
TEST(RcpIflag, NullExceptionOutput) {
  uint32_t a[32] = {}, reported[32], unreported[32];
  for (unsigned lane = 0; lane < 32; ++lane)
    a[lane] = lane % 2 ? 0x40000000 : 0;
  const uint32_t *ap[] = {a};
  uint32_t *dp[] = {reported}, *np[] = {unreported};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint32_t status = 0x80000001;
    ASSERT_EQ(goc_v_rcp_iflag_f32_rdna4(cpu, UINT32_MAX, 0, dp, ap, &status), GOC_SUCCESS);
    ASSERT_EQ(goc_v_rcp_iflag_f32_rdna4(cpu, UINT32_MAX, 0, np, ap, nullptr), GOC_SUCCESS);
    EXPECT_EQ(status, 0x80000001 | GOC_EXCEPTION_INT_DIV0);
    for (unsigned lane = 0; lane < 32; ++lane)
      EXPECT_EQ(reported[lane], unreported[lane]);
  }
}

TEST(RcpIflag, Wave64ModifiersExecAliasesAndUnalignedStorage) {
  const uint64_t max_cpu = goc_init_cpu_flags();
  for (unsigned m = 0; m < 32; ++m)
    for (unsigned alias = 0; alias < 2; ++alias)
      for (uint64_t exec_mask : {uint64_t(UINT64_MAX), uint64_t(0), uint64_t(0xffffffff00000000ULL),
                                 uint64_t(0x87654321fedcba98ULL)})
        for (uint64_t cpu = 0; cpu <= max_cpu; ++cpu)
          for (unsigned target = 0; target < 5; ++target) {
            uint32_t words[2][67], initial[2][67], outside = 0;
            for (unsigned j = 0; j < 2; ++j)
              for (unsigned lane = 0; lane < 67; ++lane)
                words[j][lane] = 0xdeadbeef;
            for (unsigned lane = 0; lane < 64; ++lane)
              words[0][lane + 1] = goc_test::rcp_iflag_input(lane * 32);
            std::memcpy(initial, words, sizeof(words));
            unsigned status_reg = target / 2, status_lane = target % 2 ? 64 : 1;
            uint32_t *status = target == 4 ? &outside : &words[status_reg][status_lane],
                     *dp[] = {words[alias ? 0 : 1] + 1};
            uint32_t wanted_status =
                goc_test::rcp_iflag_status(words[0] + 1, exec_mask, m, *status);
            const uint32_t *ap[] = {words[0] + 1};
            ASSERT_EQ(goc_v_rcp_iflag_f32_rdna4_wave64(cpu, exec_mask, goc_test::rcp_iflag_mode(m),
                                                       dp, ap, status),
                      GOC_SUCCESS);
            ASSERT_EQ(*status, wanted_status);
            for (unsigned j = 0; j < 2; ++j)
              for (unsigned lane = 0; lane < 67; ++lane) {
                if (target != 4 && j == status_reg && lane == status_lane) {
                  ASSERT_EQ(words[j][lane], wanted_status);
                  continue;
                }
                if (j == (alias ? 0u : 1u) && lane >= 1 && lane <= 64 &&
                    ((exec_mask >> (lane - 1)) & 1))
                  ASSERT_TRUE(goc_test::rcp_iflag_close(
                      words[j][lane], goc_test::rcp_iflag_reference(initial[0][lane], m)));
                else
                  ASSERT_EQ(words[j][lane], initial[j][lane]);
              }
          }
}

TEST(RcpIflag, Wave64DppModifiersMasksAndStatusAliases) {
  for (uint64_t semantics : {uint64_t(0), GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned m = 0; m < 32; ++m)
        for (auto descriptor_base : goc_test::dpp_modes)
          for (uint64_t exec_mask : {uint64_t(UINT64_MAX), uint64_t(0xffffffff00000000ULL),
                                     uint64_t(0x87654321fedcba98ULL)})
            for (unsigned target = 0; target < 2; ++target)
              for (unsigned status_target = 0; status_target < 3; ++status_target) {
                uint64_t descriptor = descriptor_base;
                if (descriptor & GOC_DPP16)
                  descriptor =
                      (descriptor & ~GOC_DPP_ROW_MASK) | (uint64_t(m & 15) << GOC_DPP_ROW_SHIFT);
                uint32_t words[2][66], expected[2][66], permuted[64] = {}, status = 0xdeadbeef;
                for (unsigned reg = 0; reg < 2; ++reg)
                  for (unsigned word = 0; word < 66; ++word)
                    words[reg][word] = expected[reg][word] =
                        reg ? 0xdead0000u + word : goc_test::rcp_iflag_input((word - 1) * 32);
                uint64_t reference_exec_mask = 0;
                for (unsigned lane = 0; lane < 64; ++lane) {
                  int source = 0;
                  if (goc_test::dpp_source(descriptor, exec_mask, lane, source)) {
                    reference_exec_mask |= 1ULL << lane;
                    permuted[lane] = source < 0 ? 0 : words[0][source + 1];
                    expected[target][lane + 1] = goc_test::rcp_iflag_reference(permuted[lane], m);
                  }
                }
                uint32_t seed = status_target < 2 ? words[status_target][46] : status;
                uint32_t want = goc_test::rcp_iflag_status(permuted, reference_exec_mask, m, seed);
                if (status_target < 2)
                  expected[status_target][46] = want;
                const uint32_t *a[] = {words[0] + 1};
                uint32_t *d[] = {words[target] + 1};
                ASSERT_EQ(goc_v_rcp_iflag_f32_rdna4_wave64(
                              cpu | semantics, exec_mask, descriptor | goc_test::rcp_iflag_mode(m),
                              d, a, status_target < 2 ? words[status_target] + 46 : &status),
                          GOC_SUCCESS);
                EXPECT_EQ(status, status_target < 2 ? 0xdeadbeef : want);
                for (unsigned reg = 0; reg < 2; ++reg)
                  for (unsigned word = 0; word < 66; ++word) {
                    if (reg == status_target && word == 46)
                      ASSERT_EQ(words[reg][word], want);
                    else if (reg == target && word >= 1 && word <= 64 &&
                             ((reference_exec_mask >> (word - 1)) & 1))
                      ASSERT_TRUE(goc_test::rcp_iflag_close(words[reg][word], expected[reg][word]));
                    else
                      ASSERT_EQ(words[reg][word], expected[reg][word]);
                  }
              }
}

TEST(RcpIflag, Wave64DppValidationAndZeroExec) {
  for (auto descriptor : goc_test::dpp_modes) {
    uint32_t status = 0xdeadbeef;
    for (auto invalid : {1ULL << 36, 1ULL << 1})
      for (uint64_t exec_mask : {uint64_t(0), uint64_t(UINT64_MAX)}) {
        EXPECT_EQ(goc_v_rcp_iflag_f32_rdna4_wave64(0, exec_mask, descriptor | invalid, nullptr,
                                                   nullptr, &status),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(status, 0xdeadbeefu);
      }
    EXPECT_EQ(goc_v_rcp_iflag_f32_rdna4_wave64(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                               0, descriptor, nullptr, nullptr, &status),
              GOC_SUCCESS);
    EXPECT_EQ(status, 0xdeadbeefu);
    EXPECT_EQ(goc_v_rcp_iflag_f32_rdna4_wave64(0, 0U, descriptor, nullptr, nullptr, &status),
              GOC_SUCCESS);
    EXPECT_EQ(status, 0xdeadbeefu);
  }
}

TEST(RcpIflag, Wave64NullExceptionOutput) {
  uint32_t a[64] = {}, reported[64], unreported[64];
  for (unsigned lane = 0; lane < 64; ++lane)
    a[lane] = lane % 2 ? 0x40000000 : 0;
  const uint32_t *ap[] = {a};
  uint32_t *dp[] = {reported}, *np[] = {unreported};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint32_t status = 0x80000001;
    ASSERT_EQ(goc_v_rcp_iflag_f32_rdna4_wave64(cpu, UINT64_MAX, 0, dp, ap, &status), GOC_SUCCESS);
    ASSERT_EQ(goc_v_rcp_iflag_f32_rdna4_wave64(cpu, UINT64_MAX, 0, np, ap, nullptr), GOC_SUCCESS);
    EXPECT_EQ(status, 0x80000001 | GOC_EXCEPTION_INT_DIV0);
    for (unsigned lane = 0; lane < 64; ++lane)
      EXPECT_EQ(reported[lane], unreported[lane]);
  }
}

TEST(RcpIflag, Wave64UpperHalfHardwareStatusCorpus) {
  const uint32_t masks[] = {UINT32_MAX, 0, 0x55555555, 0xaaaaaaaa, 1, 0x80000000};
  const uint32_t seeds[] = {0, 0x15, 0x55};
  // Replay the captured Wave32 inputs in lanes 32..63; inactive zeros below
  // must not contribute INT_DIV0. The original status digests remain unchanged.
  for (uint64_t semantics : {uint64_t(0), GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned si = 0; si < 3; ++si)
        for (unsigned mi = 0; mi < 6; ++mi)
          for (unsigned m = 0; m < 32; ++m) {
            uint64_t digest = goc_test::capture_hash_seed;
            uint64_t flags = cpu | semantics | (mi & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                             (mi & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0);
            for (unsigned wave = 0; wave < 128; ++wave) {
              uint32_t a[64] = {}, d[64], status = seeds[si];
              for (unsigned lane = 0; lane < 64; ++lane) {
                if (lane >= 32)
                  a[lane] = goc_test::rcp_iflag_input(wave * 32 + lane - 32);
                d[lane] = 0xcafebeef;
              }
              const uint32_t *ap = a;
              uint32_t *dp = d;
              uint64_t exec_mask = uint64_t(masks[mi]) << 32;
              ASSERT_EQ(goc_v_rcp_iflag_f32_rdna4_wave64(
                            flags, exec_mask, goc_test::rcp_iflag_mode(m), &dp, &ap, &status),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 64; ++lane) {
                if ((exec_mask >> lane) & 1) {
                  ASSERT_TRUE(goc_test::rcp_iflag_close(d[lane],
                                                        goc_test::rcp_iflag_reference(a[lane], m)));
                } else {
                  ASSERT_EQ(d[lane], 0xcafebeef);
                }
              }
              digest = goc_test::capture_hash_bytes(digest, status, 4);
            }
            EXPECT_EQ(digest, goc_test::rcp_iflag_status_digests[(si * 6 + mi) * 32 + m]);
          }
}

TEST(RcpIflag, ArchitectureOmodGuestPolicyAcrossWaveSizesAndDpp) {
  for (uint64_t semantics : {uint64_t(0), GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT})
    for (bool rdna3 : {false, true})
      for (bool wide : {false, true})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (unsigned policy = 0; policy < 8; ++policy)
            for (unsigned m = 0; m < 32; ++m)
              for (auto descriptor : goc_test::dpp_modes)
                for (bool alias : {false, true}) {
                  uint32_t input[64], output[64], original[64];
                  for (unsigned lane = 0; lane < 64; ++lane) {
                    // Powers of two make the reciprocal and OMOD exact on every CPU.
                    input[lane] = original[lane] = 0x3f800000 + (lane % 4) * 0x800000;
                    output[lane] = 0xdeadbeef;
                  }
                  const uint32_t *a = input;
                  uint32_t *d = alias ? input : output, state = 0x80000001;
                  uint64_t flags = cpu | semantics |
                                   (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                                   (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                                   (policy & 4 ? GOC_FP_IEEE_MODE : 0);
                  uint64_t exec_mask = wide ? 0x8192a5a571abcdefULL : 0x71abcdef;
                  uint64_t mode = descriptor | goc_test::rcp_iflag_mode(m);
                  int error = wide ? (rdna3 ? goc_v_rcp_iflag_f32_wave64
                                            : goc_v_rcp_iflag_f32_rdna4_wave64)(
                                         flags, exec_mask, mode, &d, &a,
                                         rdna3 && semantics ? nullptr : &state)
                                   : (rdna3 ? goc_v_rcp_iflag_f32 : goc_v_rcp_iflag_f32_rdna4)(
                                         flags, uint32_t(exec_mask), mode, &d, &a,
                                         rdna3 && semantics ? nullptr : &state);
                  ASSERT_EQ(error, GOC_SUCCESS);
                  unsigned effective_m = rdna3 && (!(policy & 2) || (policy & 4)) ? m & ~12U : m;
                  for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane) {
                    int source;
                    bool active = goc_test::dpp_source(descriptor, exec_mask, lane, source);
                    uint32_t expected = active ? goc_test::rcp_iflag_reference(
                                                     source < 0 ? 0 : original[source], effective_m)
                                        : alias ? original[lane]
                                                : 0xdeadbeef;
                    EXPECT_EQ(d[lane], expected)
                        << rdna3 << '/' << wide << '/' << policy << '/' << m;
                  }
                  if (rdna3) {
                    EXPECT_EQ(state, 0x80000001U);
                  }
                }
}

TEST(RcpIflag, Rdna3ReportingValidationAndLooseOptOut) {
  for (bool wide : {false, true})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[64] = {}, output[64];
      std::fill(output, output + 64, 0xdeadbeef);
      const uint32_t *a = input;
      uint32_t *d = output, state = 0x80000001;
      auto invoke = [&](uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *status) {
        return wide ? goc_v_rcp_iflag_f32_wave64(flags, exec_mask, mode, &d, &a, status)
                    : goc_v_rcp_iflag_f32(flags, uint32_t(exec_mask), mode, &d, &a, status);
      };
      for (uint64_t exec_mask : {0ULL, ~0ULL}) {
        EXPECT_EQ(invoke(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, exec_mask, 0, &state),
                  GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
        EXPECT_EQ(invoke(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, exec_mask, 0,
                         &state),
                  GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
        EXPECT_EQ(invoke(cpu, exec_mask, GOC_ALU_NEG_B, &state), GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(state, 0x80000001U);
        for (auto word : output)
          EXPECT_EQ(word, 0xdeadbeefU);
      }
      ASSERT_EQ(invoke(cpu, ~0ULL, 0, &state), GOC_SUCCESS);
      EXPECT_EQ(state, 0x80000001U);
      for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane)
        EXPECT_EQ(output[lane], 0x7f800000U);
      EXPECT_EQ(invoke(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, ~0ULL, 0, nullptr), GOC_SUCCESS);
      EXPECT_EQ(
          invoke(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, ~0ULL, 0, nullptr),
          GOC_SUCCESS);
    }
}

TEST(RcpIflag, ExactBorrowedNormalizedCapture) {
  const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
  uint64_t hash = goc_test::capture_hash_seed;
  for (unsigned base = 0; base < (1U << 23); base += 64) {
    uint32_t input[64], output[64], state = 0x80000001;
    for (unsigned lane = 0; lane < 64; ++lane)
      input[lane] = 0x3f800000 + base + lane;
    bool wide = (base >> 6) & 1, rdna3 = (base >> 7) & 1;
    const uint64_t flags = exact | ((base >> 8) % (goc_init_cpu_flags() + 1));
    const uint32_t *a = input;
    uint32_t *d = output;
    if (wide) {
      auto fn = rdna3 ? goc_v_rcp_iflag_f32_wave64 : goc_v_rcp_iflag_f32_rdna4_wave64;
      ASSERT_EQ(fn(flags, UINT64_MAX, 0, &d, &a, rdna3 ? nullptr : &state), GOC_SUCCESS);
    } else {
      auto fn = rdna3 ? goc_v_rcp_iflag_f32 : goc_v_rcp_iflag_f32_rdna4;
      ASSERT_EQ(fn(flags, UINT32_MAX, 0, &d, &a, rdna3 ? nullptr : &state), GOC_SUCCESS);
      a += 32;
      d += 32;
      ASSERT_EQ(fn(flags, UINT32_MAX, 0, &d, &a, rdna3 ? nullptr : &state), GOC_SUCCESS);
    }
    EXPECT_EQ(state, 0x80000001U);
    for (auto word : output)
      hash = goc_test::capture_hash_word(hash, word);
  }
  // rocjitsu uses the same empirical reciprocal for RCP and RCP_IFLAG.
  // Reuse the retained RCP mapping fixture; this is not a new IFLAG capture.
  EXPECT_EQ(hash, goc_test::unary_normalized_hashes[0]);
}

TEST(RcpIflag, ExactNanQuietingAndStateErrorPrecedence) {
  for (bool rdna3 : {false, true})
    for (bool wide : {false, true})
      for (unsigned policy = 0; policy < 8; ++policy) {
        uint32_t values[64];
        std::fill(values, values + 64, 0xff800123U);
        const uint32_t *a = values;
        uint32_t *d = values, state = 0x80000001;
        const uint64_t flags = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                               (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                               (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                               (policy & 4 ? GOC_FP_IEEE_MODE : 0);
        auto call = [&](uint64_t f, uint64_t exec_mask, uint64_t mode, uint32_t *report) {
          return wide ? (rdna3 ? goc_v_rcp_iflag_f32_wave64 : goc_v_rcp_iflag_f32_rdna4_wave64)(
                            f, exec_mask, mode, &d, &a, report)
                      : (rdna3 ? goc_v_rcp_iflag_f32 : goc_v_rcp_iflag_f32_rdna4)(
                            f, uint32_t(exec_mask), mode, &d, &a, report);
        };
        ASSERT_EQ(call(flags, 1, 0, rdna3 ? nullptr : &state), GOC_SUCCESS);
        EXPECT_EQ(values[0], 0xffc00123U);
        EXPECT_EQ(state, 0x80000001U);
        EXPECT_EQ(call(2 * GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
        for (uint64_t strict : {uint64_t(0), uint64_t(GOC_SEMANTICS_STRICT)}) {
          const uint64_t unavailable =
              (rdna3 ? GOC_SEMANTICS_EXACT_EMPIRICAL : 2 * GOC_SEMANTICS_EXACT_EMPIRICAL) | strict;
          EXPECT_EQ(call(unavailable, 0, GOC_ALU_NEG_B, &state),
                    GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
          EXPECT_EQ(state, 0x80000001U);
          EXPECT_EQ(values[0], 0xffc00123U);
        }
      }
}
