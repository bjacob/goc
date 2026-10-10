// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "div_fixup_reference.h"
#include "dpp_reference.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "half_reference.h"
#include "test_instruction.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

using Fn = goc_test::WaveInstruction<int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *,
                                             const uint32_t *const *, const uint32_t *const *,
                                             const uint32_t *const *)>;
const Fn functions[2][2] = {{goc_v_div_fixup_f16, goc_v_div_fixup_f16_wave64},
                            {goc_v_div_fixup_f16_rdna4, goc_v_div_fixup_f16_rdna4_wave64}};
const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

uint64_t policy(unsigned guest) {
  return (guest & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (guest & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (guest & 4 ? GOC_FP_IEEE_MODE : 0);
}

uint16_t reference(unsigned arch, unsigned guest, uint32_t mode, bool saturate, uint32_t a,
                   uint32_t b, uint32_t c) {
  uint32_t raw[] = {a, b, c};
  for (unsigned i = 0; i < 3; ++i) {
    raw[i] = (raw[i] >> (mode & (GOC_ALU_HIGH_A << i) ? 16 : 0)) & 65535;
    if (i && (guest & 1) && !(raw[i] & 0x7c00))
      raw[i] &= 0x8000;
  }
  if (!arch && (!(guest & 2) || (guest & 4)))
    mode &= ~GOC_ALU_OMOD_HALF;
  if (arch)
    return uint16_t(goc_test::fixup_reference(16, raw[0], raw[1], raw[2], mode, saturate));
  // rocjitsu ffc144c564c: RDNA3 VDivFixupF16Vop3's promoted result is scaled
  // and clamped before narrowing, then finalize_omod_f16 flushes tiny results.
  uint16_t base = uint16_t(goc_test::fixup_reference(
      16, raw[0], raw[1], raw[2], mode & ~(GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP), false));
  if ((base & 0x7fff) > 0x7c00)
    return mode & GOC_ALU_CLAMP ? 0 : base;
  const int powers[] = {0, 1, 2, -1};
  unsigned omod = (mode >> 6) & 3;
  double value = std::ldexp(goc_test::half_value(base), powers[omod]);
  if (mode & GOC_ALU_CLAMP)
    value = !(value > 0) ? 0 : value > 1 ? 1 : value;
  uint16_t result = goc_test::half_bits(value, saturate);
  return omod && !(result & 0x7c00) ? 0 : result;
}

} // namespace

TEST(HalfDivFixupArchitecture, EveryProvisionalEncodingAndOutputScale) {
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned scale = 0; scale < 4; ++scale)
      for (unsigned saturate = 0; saturate < 2; ++saturate)
        for (unsigned first = 0; first < 65536; first += 32) {
          uint32_t words[4][32], expected[32];
          const uint32_t mode = scale << 6;
          for (unsigned lane = 0; lane < 32; ++lane) {
            words[0][lane] = first + lane;
            words[1][lane] = 0x3c00;
            words[2][lane] = 0x3c00 | ((first + lane) & 0x8000);
            words[3][lane] = 0xa5a50000;
            expected[lane] = 0xa5a50000 | reference(arch, 2, mode, saturate, words[0][lane],
                                                    words[1][lane], words[2][lane]);
          }
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
          uint32_t *d[] = {words[3]};
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            ASSERT_EQ(functions[arch][0](cpu | exact | policy(2) | (saturate ? GOC_FP16_OVFL : 0),
                                         UINT32_MAX, mode, d, a, b, c),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 32; ++lane)
              ASSERT_EQ(words[3][lane], expected[lane])
                  << arch << '/' << scale << '/' << cpu << '/' << first + lane;
          }
        }
}

TEST(HalfDivFixupArchitecture, GuestPoliciesDppModifiersAndAliases) {
  const uint16_t edges[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x7ff,  0x3c00,
                            0xbc00, 0x4000, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe12};
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned guest = 0; guest < 8; ++guest)
        for (unsigned variant = 0; variant < 512; ++variant)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t initial[4][66], words[4][66];
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned lane = 0; lane < 66; ++lane)
                initial[reg][lane] = edges[(lane * (reg * 2 + 1) + variant) % 16] |
                                     uint32_t(edges[(lane * 7 + reg + variant / 16) % 16]) << 16;
            std::memcpy(words, initial, sizeof(words));
            uint64_t mode = (variant * 37) & 8191;
            if (variant & 2)
              mode |= goc_test::dpp_modes[(variant / 2) % 7];
            uint64_t exec_mask = variant & 4 ? 0xa5a55a5af0f00f0fULL : UINT64_MAX;
            if (!wide)
              exec_mask = uint32_t(exec_mask);
            const uint32_t *a[] = {words[0] + 1}, *b[] = {words[1] + 1}, *c[] = {words[2] + 1};
            unsigned dest = variant % 4;
            uint32_t *d[] = {words[dest] + 1}, state = 0x12345678;
            bool saturate = variant & 8;
            ASSERT_EQ(functions[arch][wide](cpu | policy(guest) | (saturate ? GOC_FP16_OVFL : 0),
                                            exec_mask, mode, d, a, b, c, &state),
                      GOC_SUCCESS);
            ASSERT_EQ(state, 0x12345678U);
            for (unsigned lane = 0; lane < 66; ++lane) {
              uint32_t expected = initial[dest][lane];
              int source = int(lane) - 1;
              bool write =
                  lane >= 1 && lane <= (wide ? 64U : 32U) && ((exec_mask >> (lane - 1)) & 1);
              if (write && (mode >> 32))
                write = goc_test::dpp_source(mode, exec_mask, lane - 1, source);
              if (write) {
                uint16_t value = reference(arch, guest, uint32_t(mode), saturate,
                                           source < 0 ? 0 : initial[0][source + 1],
                                           initial[1][lane], initial[2][lane]);
                unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
                expected = (expected & ~(65535U << shift)) | (uint32_t(value) << shift);
              }
              ASSERT_EQ(words[dest][lane], expected) << arch << '/' << wide << '/' << guest << '/'
                                                     << variant << '/' << lane << '/' << cpu;
            }
          }
}

TEST(HalfDivFixupArchitecture, LiteralArchitectureDifferences) {
  struct Case {
    uint16_t p;
    uint32_t mode;
    bool saturate;
    uint16_t rdna3, rdna4;
  };

  const Case cases[] = {{0x3ff, GOC_ALU_OMOD_2, false, 0x7fe, 0},
                        {0x7ff, GOC_ALU_OMOD_HALF, false, 0x400, 0},
                        {0x7c00, 0, true, 0x7c00, 0x7bff},
                        {0x7bff, GOC_ALU_OMOD_2, true, 0x7bff, 0x7bff}};
  for (const auto &entry : cases)
    for (unsigned arch = 0; arch < 2; ++arch)
      for (unsigned wide = 0; wide < 2; ++wide)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          uint32_t input[64], one[64], output[64] = {};
          std::fill_n(input, 64, entry.p);
          std::fill_n(one, 64, 0x3c00);
          const uint32_t *a[] = {input}, *b[] = {one};
          uint32_t *d[] = {output};
          ASSERT_EQ(
              functions[arch][wide](cpu | exact | policy(2) | (entry.saturate ? GOC_FP16_OVFL : 0),
                                    UINT64_MAX, entry.mode, d, a, b, b),
              GOC_SUCCESS);
          for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane)
            EXPECT_EQ(output[lane], arch ? entry.rdna4 : entry.rdna3);
        }
}

TEST(HalfDivFixupArchitecture, ErrorsAndHostEnvironment) {
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
        for (unsigned guest = 0; guest < 8; ++guest) {
          SCOPED_TRACE(std::to_string(arch) + "/" + std::to_string(wide) + "/" +
                       std::to_string(guest));
          uint32_t data[64], state = 0x12345678;
          std::fill_n(data, 64, 0x3c0007ff);
          const uint32_t *a[] = {data};
          uint32_t *d[] = {data};
          if (!arch)
            for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT})
              for (uint64_t exec_mask : {uint64_t(0), UINT64_MAX}) {
                EXPECT_EQ(
                    functions[arch][wide](policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL | strict,
                                          exec_mask, 0, d, nullptr, nullptr, nullptr, &state),
                    GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
                EXPECT_EQ(state, 0x12345678U);
                for (auto value : data)
                  EXPECT_EQ(value, 0x3c0007ffU);
              }
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            ASSERT_EQ(functions[arch][wide](cpu | exact | policy(guest), UINT64_MAX,
                                            GOC_ALU_OMOD_HALF | GOC_ALU_HIGH_B | GOC_ALU_HIGH_C, d,
                                            a, a, a),
                      GOC_SUCCESS);
            EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
            EXPECT_EQ(_mm_getcsr() & ~0x3fU, controls);
#endif
          }
        }
  }
}
