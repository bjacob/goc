// SPDX-License-Identifier: MIT

#include "byte_conversion_reference.h"
#include "dpp_reference.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "pack_reference.h"
#include "test_instruction.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

using Pack = goc_test::WaveInstruction<int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *,
                                               const uint32_t *const *, const uint32_t *const *)>;
using Byte = goc_test::WaveInstruction<decltype(&goc_v_mad_u32_u24)>;
const Pack pack[2][2] = {{goc_v_pack_b32_f16, goc_v_pack_b32_f16_wave64},
                         {goc_v_pack_b32_f16_rdna4, goc_v_pack_b32_f16_rdna4_wave64}};
const Byte byte[2][2] = {{goc_v_cvt_pk_u8_f32, goc_v_cvt_pk_u8_f32_wave64},
                         {goc_v_cvt_pk_u8_f32_rdna4, goc_v_cvt_pk_u8_f32_rdna4_wave64}};

uint64_t policy(unsigned guest) {
  return (guest & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (guest & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (guest & 4 ? GOC_FP_IEEE_MODE : 0);
}

void inputs(unsigned op, uint32_t words[4][66]) {
  const uint32_t edges[] = {0,          1,          0x80000001, 0x3effffff, 0x3f000000,
                            0x3f000001, 0x3fc00000, 0x40200000, 0x437effff, 0x437f0000,
                            0x7f800000, 0xff800000, 0x7f800001, 0x7fc00001};
  const uint16_t half[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3c00,
                           0xbc00, 0x7bff, 0x7c00, 0xfc00, 0x7c01, 0x7e00, 0xffff};
  for (unsigned lane = 0; lane < 66; ++lane) {
    words[0][lane] =
        op ? edges[lane % 14] : half[lane % 14] | (uint32_t(half[(lane * 3 + 1) % 14]) << 16);
    words[1][lane] =
        op ? lane : half[(lane * 5 + 2) % 14] | (uint32_t(half[(lane * 7 + 3) % 14]) << 16);
    words[2][lane] = lane * 0x01030709U;
    words[3][lane] = 0xdeadbeef;
  }
}

} // namespace

TEST(PackArchitecture, PoliciesModifiersMasksAndAliases) {
  for (unsigned op = 0; op < 2; ++op)
    for (unsigned arch = 0; arch < 2; ++arch)
      for (unsigned wide = 0; wide < 2; ++wide)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
            for (unsigned m = 0; m < (op ? 8u : 64u); ++m)
              for (unsigned target = 0; target < 4; ++target)
                for (uint64_t exec_mask :
                     {uint64_t(0), uint64_t(0x8181818155555555ULL), UINT64_MAX}) {
                  uint32_t words[4][66], expected[4][66];
                  inputs(op, words);
                  std::memcpy(expected, words, sizeof(words));
                  unsigned lanes = wide ? 64 : 32;
                  uint64_t mode = op ? goc_test::byte_pack_mode(m) : goc_test::pack_mode(m + 2);
                  for (unsigned lane = 0; lane < lanes; ++lane)
                    if ((exec_mask >> lane) & 1)
                      expected[target][lane + 1] =
                          op ? goc_test::byte_pack_reference(words[0][lane + 1], words[1][lane + 1],
                                                             words[2][lane + 1], mode)
                             : goc_test::pack_reference(m + 2, words[0][lane + 1],
                                                        words[1][lane + 1], 0);
                  const uint32_t *a[] = {words[0] + 1}, *b[] = {words[1] + 1},
                                 *c[] = {words[2] + 1};
                  uint32_t *d[] = {words[target] + 1}, state = 0x12345678;
                  uint64_t flags = cpu | policy(guest);
                  int status = op ? byte[arch][wide](flags, exec_mask, mode, d, a, b, c, &state)
                                  : pack[arch][wide](flags, exec_mask, mode, d, a, b, &state);
                  ASSERT_EQ(status, GOC_SUCCESS);
                  ASSERT_EQ(state, 0x12345678U);
                  ASSERT_EQ(std::memcmp(words, expected, sizeof(words)), 0)
                      << op << "/" << arch << "/" << wide << "/" << guest << "/" << m;
                }
}

TEST(PackArchitecture, DppAndGuestPolicies) {
  for (unsigned op = 0; op < 2; ++op)
    for (unsigned arch = 0; arch < 2; ++arch)
      for (unsigned wide = 0; wide < 2; ++wide)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (unsigned guest = 0; guest < 8; ++guest)
            for (uint64_t descriptor : goc_test::dpp_modes) {
              uint32_t words[4][66], expected[66];
              inputs(op, words);
              std::copy_n(words[3], 66, expected);
              uint64_t exec_mask = wide ? 0xa5a55a5af0f00f0fULL : 0xf0f00f0fU;
              uint64_t mode =
                  descriptor | (op ? goc_test::byte_pack_mode(3) : goc_test::pack_mode(55));
              for (unsigned lane = 0; lane < (wide ? 64u : 32u); ++lane) {
                int source;
                if (!goc_test::dpp_source(descriptor, exec_mask, lane, source))
                  continue;
                uint32_t a = source < 0 ? 0 : words[0][source + 1];
                expected[lane + 1] =
                    op ? goc_test::byte_pack_reference(a, words[1][lane + 1], words[2][lane + 1],
                                                       uint32_t(mode))
                       : goc_test::pack_reference(55, a, words[1][lane + 1], 0);
              }
              const uint32_t *a[] = {words[0] + 1}, *b[] = {words[1] + 1}, *c[] = {words[2] + 1};
              uint32_t *d[] = {words[3] + 1};
              uint64_t flags = cpu | policy(guest);
              ASSERT_EQ(op ? byte[arch][wide](flags, exec_mask, mode, d, a, b, c)
                           : pack[arch][wide](flags, exec_mask, mode, d, a, b),
                        GOC_SUCCESS);
              ASSERT_EQ(std::memcmp(words[3], expected, sizeof(expected)), 0);
            }
}

TEST(PackArchitecture, ExactSupportAndReportingErrors) {
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned guest = 0; guest < 8; ++guest) {
        uint32_t av[64], bv[64], output[64];
        std::fill_n(av, 64, 0x7c01);
        std::fill_n(bv, 64, 0x8001);
        std::fill_n(output, 64, 0xdeadbeef);
        const uint32_t *a[] = {av}, *b[] = {bv};
        uint32_t *d[] = {output}, state = 0x12345678;
        for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT})
          for (uint64_t exec_mask : {uint64_t(0), UINT64_MAX}) {
            uint64_t flags = policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL | strict;
            EXPECT_EQ(pack[arch][wide](flags, exec_mask, 0, d, nullptr, nullptr, &state),
                      GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
            EXPECT_EQ(byte[arch][wide](flags, exec_mask, 0, d, nullptr, nullptr, nullptr, &state),
                      GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
            EXPECT_EQ(state, 0x12345678U);
            for (auto word : output)
              EXPECT_EQ(word, 0xdeadbeefU);
          }
        uint64_t flags = policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;
        EXPECT_EQ(byte[arch][wide](flags, 0, 0, d, nullptr, nullptr, nullptr),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
        EXPECT_EQ(pack[arch][wide](flags, UINT64_MAX, 0, d, a, b),
                  arch ? GOC_SUCCESS : GOC_ERROR_UNSUPPORTED_SEMANTICS);
        for (unsigned lane = 0; lane < 64; ++lane)
          EXPECT_EQ(output[lane], arch && lane < (wide ? 64u : 32u) ? 0x80017e01U : 0xdeadbeefU);
      }
}

TEST(PackArchitecture, HostControlsArePreserved) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    for (bool host_flush : {false, true}) {
      ASSERT_EQ(std::fesetround(rounding), 0);
#if defined(__x86_64__) || defined(_M_X64)
      _mm_setcsr((_mm_getcsr() & ~0x8040U) | (host_flush ? 0x8040U : 0));
      const unsigned controls = _mm_getcsr() & ~0x3fU;
#endif
      for (unsigned op = 0; op < 2; ++op)
        for (unsigned arch = 0; arch < 2; ++arch)
          for (unsigned wide = 0; wide < 2; ++wide)
            for (unsigned guest = 0; guest < 8; ++guest)
              for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
                uint32_t words[4][66];
                inputs(op, words);
                const uint32_t *a[] = {words[0] + 1}, *b[] = {words[1] + 1}, *c[] = {words[2] + 1};
                uint32_t *d[] = {words[3] + 1}, state = 0xabcdef01;
                uint64_t flags = cpu | policy(guest);
                ASSERT_EQ(op ? byte[arch][wide](flags, UINT64_MAX, 0, d, a, b, c, &state)
                             : pack[arch][wide](flags, UINT64_MAX, 0, d, a, b, &state),
                          GOC_SUCCESS);
                EXPECT_EQ(state, 0xabcdef01U);
                for (unsigned lane = 1; lane <= (wide ? 64u : 32u); ++lane)
                  EXPECT_EQ(words[3][lane],
                            op ? goc_test::byte_pack_reference(words[0][lane], words[1][lane],
                                                               words[2][lane], 0)
                               : goc_test::pack_reference(2, words[0][lane], words[1][lane], 0));
                EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
                EXPECT_EQ(_mm_getcsr() & ~0x3fU, controls);
#endif
              }
    }
}
