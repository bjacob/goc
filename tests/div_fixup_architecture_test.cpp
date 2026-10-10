// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "div_fixup_reference.h"
#include "fp_environment.h"
#include "goc/goc.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

using Fn32 = decltype(&goc_v_div_fixup_f32);
using Fn64 = decltype(&goc_v_div_fixup_f32_wave64);
const Fn32 functions[2][2] = {{goc_v_div_fixup_f32, goc_v_div_fixup_f64},
                              {goc_v_div_fixup_f32_rdna4, goc_v_div_fixup_f64_rdna4}};
const Fn64 wide_functions[2][2] = {
    {goc_v_div_fixup_f32_wave64, goc_v_div_fixup_f64_wave64},
    {goc_v_div_fixup_f32_rdna4_wave64, goc_v_div_fixup_f64_rdna4_wave64}};
const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

uint64_t policy(unsigned guest) {
  return (guest & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (guest & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (guest & 4 ? GOC_FP_IEEE_MODE : 0);
}

int call(unsigned arch, unsigned wide, unsigned op, uint64_t flags, uint64_t exec_mask,
         uint64_t mode, uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
         const uint32_t *const *c, uint32_t *state = nullptr) {
  if (wide)
    return wide_functions[arch][op](flags, exec_mask, mode, d, a, b, c, state);
  return functions[arch][op](flags, uint32_t(exec_mask), mode, d, a, b, c, state);
}

} // namespace

TEST(DivFixupArchitecture, PoliciesModifiersMasksAndAliases) {
  std::mt19937 random(930271);
  for (unsigned op = 0; op < 2; ++op)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned arch = 0; arch < 2; ++arch)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (unsigned mode = 0; mode < 512; ++mode) {
            uint32_t initial[6][64], words[6][64], expected[6][64];
            const uint64_t sign = op ? 0x8000000000000000ULL : 0x80000000ULL;
            const uint64_t inf = op ? 0x7ff0000000000000ULL : 0x7f800000ULL;
            const uint64_t one = op ? 0x3ff0000000000000ULL : 0x3f800000ULL;
            const uint64_t minimum = op ? 0x0010000000000000ULL : 0x00800000ULL;
            const uint64_t edges[] = {0,   sign,       1,       sign | 1, minimum - 1, minimum,
                                      one, sign | one, inf - 1, inf,      inf | 1,     inf | sign};
            for (unsigned operand = 0; operand < 3; ++operand)
              for (unsigned lane = 0; lane < 64; ++lane) {
                uint64_t value = uint64_t(random()) << 32 | random();
                if (lane < 16)
                  value = edges[(lane * (operand * 2 + 1) + mode) % 12];
                initial[2 * operand][lane] = uint32_t(value);
                initial[2 * operand + 1][lane] = uint32_t(value >> 32);
              }
            std::memcpy(expected, initial, sizeof(expected));
            unsigned first = mode % 6, second = (mode / 6) % 6;
            uint64_t exec_mask = mode & 1 ? 0xa5a55a5af0f00f0fULL : UINT64_MAX;
            if (!wide)
              exec_mask = uint32_t(exec_mask);
            unsigned effective =
                !arch && (!(guest & 2) || (guest & 4)) ? mode & ~GOC_ALU_OMOD_HALF : mode;
            for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane)
              if ((exec_mask >> lane) & 1) {
                uint64_t raw[3];
                for (unsigned operand = 0; operand < 3; ++operand) {
                  raw[operand] = initial[operand * 2][lane];
                  if (op)
                    raw[operand] |= uint64_t(initial[operand * 2 + 1][lane]) << 32;
                  if (operand && (guest & 1) && !(raw[operand] & inf))
                    raw[operand] &= sign;
                }
                uint64_t value = goc_test::fixup_reference(op ? 64 : 32, raw[0], raw[1], raw[2],
                                                           effective, false);
                expected[first][lane] = uint32_t(value);
                if (op)
                  expected[second][lane] = uint32_t(value >> 32);
              }
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
              std::memcpy(words, initial, sizeof(words));
              const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]},
                             *c[] = {words[4], words[5]};
              uint32_t *d[] = {words[first], words[second]}, state = 0x12345678;
              const bool loose = mode & 1;
              ASSERT_EQ(call(arch, wide, op, cpu | policy(guest) | (loose ? 0 : exact), exec_mask,
                             mode, d, a, b, c, loose ? &state : nullptr),
                        GOC_SUCCESS);
              ASSERT_EQ(state, 0x12345678U);
              ASSERT_EQ(std::memcmp(words, expected, sizeof(words)), 0)
                  << arch << '/' << wide << '/' << op << '/' << guest << '/' << mode << '/' << cpu;
            }
          }
}

TEST(DivFixupArchitecture, ProvisionalQuotientAndFlushedOperands) {
  // rocjitsu division_test.cpp preserves a tiny provisional quotient while
  // applying input flushing to the denominator and numerator only.
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned op = 0; op < 2; ++op)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            const uint64_t sign = op ? 0x8000000000000000ULL : 0x80000000ULL;
            const uint64_t inf = op ? 0x7ff0000000000000ULL : 0x7f800000ULL;
            const uint64_t one = op ? 0x3ff0000000000000ULL : 0x3f800000ULL;
            const uint64_t quiet = op ? 0x0008000000000000ULL : 0x00400000ULL;
            const uint64_t av[] = {1, one, one, one};
            const uint64_t bv[] = {one, sign | 1, 1, one};
            const uint64_t cv[] = {one, one, 1, sign | 1};
            const uint64_t expected[] = {1, (guest & 1) ? sign | inf : sign | one,
                                         (guest & 1) ? sign | inf | quiet : one,
                                         (guest & 1) ? sign : sign | one};
            uint32_t words[8][64] = {};
            for (unsigned lane = 0; lane < 64; ++lane) {
              const uint64_t inputs[] = {av[lane % 4], bv[lane % 4], cv[lane % 4]};
              for (unsigned operand = 0; operand < 3; ++operand) {
                words[operand * 2][lane] = uint32_t(inputs[operand]);
                words[operand * 2 + 1][lane] = uint32_t(inputs[operand] >> 32);
              }
            }
            const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]},
                           *c[] = {words[4], words[5]};
            uint32_t *d[] = {words[6], words[7]};
            ASSERT_EQ(call(arch, wide, op, exact | cpu | policy(guest), UINT64_MAX, 0, d, a, b, c),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane) {
              uint64_t value = words[6][lane];
              if (op)
                value |= uint64_t(words[7][lane]) << 32;
              ASSERT_EQ(value, expected[lane % 4]);
            }
          }
}

TEST(DivFixupArchitecture, ErrorsAndHostEnvironment) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(rounding), 0);
#if defined(__x86_64__) || defined(_M_X64)
    _mm_setcsr(_mm_getcsr() | 0x8040U);
    unsigned controls = _mm_getcsr();
#endif
    for (unsigned arch = 0; arch < 2; ++arch)
      for (unsigned wide = 0; wide < 2; ++wide)
        for (unsigned op = 0; op < 2; ++op)
          for (unsigned guest = 0; guest < 8; ++guest) {
            uint32_t data[2][64], state = 0x12345678;
            std::fill_n(data[0], 64, 1);
            std::fill_n(data[1], 64, 0);
            const uint32_t *a[] = {data[0], data[1]};
            uint32_t *d[] = {data[0], data[1]};
            if (!arch)
              for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT})
                for (uint64_t exec_mask : {uint64_t(0), UINT64_MAX}) {
                  ASSERT_EQ(call(arch, wide, op,
                                 policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL | strict, exec_mask,
                                 0, d, nullptr, nullptr, nullptr, &state),
                            GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
                  EXPECT_EQ(state, 0x12345678U);
                  for (unsigned lane = 0; lane < 64; ++lane) {
                    EXPECT_EQ(data[0][lane], 1U);
                    EXPECT_EQ(data[1][lane], 0U);
                  }
                }
            EXPECT_EQ(
                call(arch, wide, op, policy(guest), 0, GOC_DPP8, d, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
              ASSERT_EQ(
                  call(arch, wide, op, cpu | policy(guest) | exact, UINT64_MAX, 0, d, a, a, a),
                  GOC_SUCCESS);
              EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
              EXPECT_EQ(_mm_getcsr(), controls);
#endif
            }
          }
  }
}
