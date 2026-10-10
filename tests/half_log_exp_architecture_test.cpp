// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "half_unary_reference.h"
#include "test_instruction.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {
using Fn = goc_test::WaveInstruction<decltype(&goc_v_not_b32)>;
const Fn functions[2][2][2] = {
    {{goc_v_exp_f16, goc_v_log_f16}, {goc_v_exp_f16_wave64, goc_v_log_f16_wave64}},
    {{goc_v_exp_f16_rdna4, goc_v_log_f16_rdna4},
     {goc_v_exp_f16_rdna4_wave64, goc_v_log_f16_rdna4_wave64}}};

uint64_t policy(unsigned guest) {
  return (guest & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (guest & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (guest & 4 ? GOC_FP_IEEE_MODE : 0);
}

uint32_t effective_mode(unsigned arch, unsigned guest, uint32_t mode) {
  if (!arch && (!(guest & 2) || (guest & 4)))
    mode &= ~GOC_ALU_OMOD_HALF;
  return mode;
}

void check(uint16_t got, uint16_t expected) {
  if ((expected & 0x7fff) > 0x7c00) {
    ASSERT_GT(got & 0x7fff, 0x7c00);
  } else if ((expected & 0x7fff) == 0 || (expected & 0x7fff) == 0x7c00) {
    ASSERT_EQ(got, expected);
  } else {
    ASSERT_EQ(got & 0x8000, expected & 0x8000);
    ASSERT_LE(std::abs(int(got) - int(expected)), 1);
  }
}
} // namespace

TEST(HalfLogExpArchitecture, PoliciesModifiersDppAndAliases) {
  const uint16_t edges[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3c00, 0x4000,
                            0xbc00, 0xcb80, 0xcc00, 0x7bff, 0x7c00, 0xfc00, 0x7c01, 0x7e12};
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned op = 0; op < 2; ++op)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
            for (unsigned variant = 0; variant < 128; ++variant) {
              uint32_t initial[66], words[66], output[66];
              for (unsigned i = 0; i < 66; ++i)
                initial[i] = edges[i % 16] | (uint32_t(edges[(i * 5 + 7) % 16]) << 16);
              std::memcpy(words, initial, sizeof(words));
              std::fill_n(output, 66, 0xdeadbeef);
              bool alias = variant & 1;
              uint32_t *target = alias ? words : output;
              uint64_t mode = goc_test::half_unary_modifiers(variant);
              if (variant & 2)
                mode |= goc_test::dpp_modes[(variant / 2) % 7];
              uint64_t exec_mask = variant & 4 ? 0xa5a55a5af0f00f0fULL : UINT64_MAX;
              if (!wide)
                exec_mask = uint32_t(exec_mask);
              const uint32_t *a[] = {words + 1};
              uint32_t *d[] = {target + 1};
              uint32_t state = 0x12345678;
              uint64_t flags = cpu | policy(guest) | (variant & 8 ? GOC_FP16_OVFL : 0);
              ASSERT_EQ(functions[arch][wide][op](flags, exec_mask, mode, d, a, &state),
                        GOC_SUCCESS);
              ASSERT_EQ(state, 0x12345678U);
              unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
              for (unsigned i = 0; i < 66; ++i) {
                int source = int(i) - 1;
                bool write = i >= 1 && i <= (wide ? 64u : 32u) && ((exec_mask >> (i - 1)) & 1);
                if (write && (mode >> 32))
                  write = goc_test::dpp_source(mode, exec_mask, i - 1, source);
                uint32_t before = alias ? initial[i] : 0xdeadbeef;
                if (!write) {
                  ASSERT_EQ(target[i], before);
                  continue;
                }
                uint32_t input = source < 0 ? 0 : initial[source + 1];
                uint16_t expected = goc_test::half_unary_reference(
                    op + 7, input, effective_mode(arch, guest, uint32_t(mode)),
                    flags & GOC_FP16_OVFL, flags);
                check(uint16_t(target[i] >> shift), expected);
                ASSERT_EQ(target[i] & (shift ? 0xffffU : 0xffff0000U),
                          before & (shift ? 0xffffU : 0xffff0000U));
              }
            }
}

TEST(HalfLogExpArchitecture, LiteralScalingAndDenormalWitnesses) {
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned guest = 0; guest < 8; ++guest) {
          uint32_t input[64], output[64] = {};
          const uint32_t *a[] = {input};
          uint32_t *d[] = {output};
          std::fill_n(input, 64, 0x3c00);
          ASSERT_EQ(functions[arch][wide][0](cpu | policy(guest), UINT64_MAX, GOC_ALU_OMOD_2, d, a),
                    GOC_SUCCESS);
          uint16_t expected = (arch || ((guest & 2) && !(guest & 4))) ? 0x4400 : 0x4000;
          for (unsigned lane = 0; lane < (wide ? 64u : 32u); ++lane)
            ASSERT_EQ(output[lane] & 65535, expected);
          std::fill_n(input, 64, 1);
          ASSERT_EQ(functions[arch][wide][1](cpu | policy(guest), UINT64_MAX, 0, d, a),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < (wide ? 64u : 32u); ++lane)
            ASSERT_EQ(output[lane] & 65535, guest & 1 ? 0xfc00U : 0xce00U);
          std::fill_n(input, 64, 0xcb80); // exp2(-15) is a half subnormal.
          ASSERT_EQ(functions[arch][wide][0](cpu | policy(guest), UINT64_MAX, 0, d, a),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < (wide ? 64u : 32u); ++lane)
            ASSERT_EQ(output[lane] & 65535, guest & 2 ? 0U : 0x200U);
        }
}

TEST(HalfLogExpArchitecture, ErrorsAndHostControls) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(rounding), 0);
#if defined(__x86_64__) || defined(_M_X64)
    _mm_setcsr(_mm_getcsr() | 0x8040U);
    unsigned controls = _mm_getcsr() & ~0x3fU;
#endif
    for (unsigned arch = 0; arch < 2; ++arch)
      for (unsigned wide = 0; wide < 2; ++wide)
        for (unsigned op = 0; op < 2; ++op)
          for (unsigned guest = 0; guest < 8; ++guest) {
            uint32_t data[64], input[64];
            std::fill_n(data, 64, 0xdeadbeef);
            std::fill_n(input, 64, 1);
            uint32_t *d[] = {data}, state = 0x12345678;
            const uint32_t *a[] = {input};
            for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT})
              for (uint64_t exec_mask : {uint64_t(0), UINT64_MAX}) {
                uint64_t flags = policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL | strict;
                EXPECT_EQ(functions[arch][wide][op](flags, exec_mask, 0, d, nullptr, &state),
                          GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
                if (strict) {
                  EXPECT_EQ(functions[arch][wide][op](flags, exec_mask, 0, d, nullptr),
                            GOC_ERROR_UNSUPPORTED_SEMANTICS);
                }
                for (auto word : data)
                  EXPECT_EQ(word, 0xdeadbeefU);
                EXPECT_EQ(state, 0x12345678U);
              }
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
              ASSERT_EQ(functions[arch][wide][op](cpu | policy(guest), UINT64_MAX, 0, d, a, &state),
                        GOC_SUCCESS);
              EXPECT_EQ(state, 0x12345678U);
              EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
              EXPECT_EQ(_mm_getcsr() & ~0x3fU, controls);
#endif
            }
          }
  }
}
