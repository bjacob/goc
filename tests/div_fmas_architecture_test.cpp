// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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

using Fn32 = decltype(&goc_v_div_fmas_f32);
using Fn64 = decltype(&goc_v_div_fmas_f32_wave64);
const Fn32 functions[2][2] = {{goc_v_div_fmas_f32, goc_v_div_fmas_f64},
                              {goc_v_div_fmas_f32_rdna4, goc_v_div_fmas_f64_rdna4}};
const Fn64 wide_functions[2][2] = {
    {goc_v_div_fmas_f32_wave64, goc_v_div_fmas_f64_wave64},
    {goc_v_div_fmas_f32_rdna4_wave64, goc_v_div_fmas_f64_rdna4_wave64}};
const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

uint64_t policy(unsigned guest) {
  return (guest & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (guest & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (guest & 4 ? GOC_FP_IEEE_MODE : 0);
}

int call(unsigned arch, unsigned wide, unsigned op, uint64_t flags, uint64_t exec_mask,
         uint64_t mode, uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
         const uint32_t *const *c, uint64_t condition, uint32_t *state = nullptr) {
  if (wide)
    return wide_functions[arch][op](flags, exec_mask, mode, d, a, b, c, condition, state);
  return functions[arch][op](flags, uint32_t(exec_mask), mode, d, a, b, c, uint32_t(condition),
                             state);
}

template <class T> uint64_t bits(T value) {
  if constexpr (sizeof(T) == 4) {
    uint32_t raw;
    std::memcpy(&raw, &value, sizeof(raw));
    return raw;
  } else {
    uint64_t raw;
    std::memcpy(&raw, &value, sizeof(raw));
    return raw;
  }
}

} // namespace

TEST(DivFmasArchitecture, PoliciesModifiersMasksAndAliases) {
  std::mt19937 random(902147);
  for (unsigned op = 0; op < 2; ++op)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned arch = 0; arch < 2; ++arch)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (unsigned mode = 0; mode < 512; ++mode) {
            uint32_t initial[6][64], words[6][64], expected[2][64];
            for (auto &reg : initial)
              for (auto &word : reg)
                word = random();
            std::fill_n(expected[0], 64, 0);
            std::fill_n(expected[1], 64, 0);
            const uint32_t *a[] = {initial[0], initial[1]}, *b[] = {initial[2], initial[3]},
                           *c[] = {initial[4], initial[5]};
            uint32_t *gold[] = {expected[0], expected[1]};
            unsigned effective =
                !arch && (!(guest & 2) || (guest & 4)) ? mode & ~GOC_ALU_OMOD_HALF : mode;
            const uint32_t high = random();
            const uint32_t low = random();
            uint64_t condition = uint64_t(high) << 32 | low;
            ASSERT_EQ(call(1, wide, op, exact, UINT64_MAX, effective, gold, a, b, c, condition),
                      GOC_SUCCESS);
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
              std::memcpy(words, initial, sizeof(words));
              const uint32_t *av[] = {words[0], words[1]}, *bv[] = {words[2], words[3]},
                             *cv[] = {words[4], words[5]};
              unsigned first = mode % 6, second = (mode / 6) % 6;
              uint32_t *d[] = {words[first], words[second]}, state = 0x12345678;
              uint64_t exec_mask = mode & 1 ? 0xa5a55a5af0f00f0fULL : UINT64_MAX;
              if (!wide)
                exec_mask = uint32_t(exec_mask);
              ASSERT_EQ(call(arch, wide, op, cpu | policy(guest), exec_mask, mode, d, av, bv, cv,
                             condition, &state),
                        GOC_SUCCESS);
              ASSERT_EQ(state, 0x12345678U);
              // Guest output flushing acts on the rounded result. Active OMOD
              // already eliminates subnormals in the established RDNA4 model.
              uint32_t wanted[6][64];
              std::memcpy(wanted, initial, sizeof(wanted));
              for (unsigned reg = 0; reg <= op; ++reg)
                for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane)
                  if ((exec_mask >> lane) & 1) {
                    uint64_t value = expected[0][lane];
                    if (op)
                      value |= uint64_t(expected[1][lane]) << 32;
                    uint64_t exponent = op ? 0x7ff0000000000000ULL : 0x7f800000ULL;
                    uint64_t sign = op ? 0x8000000000000000ULL : 0x80000000ULL;
                    if ((guest & 2) && !(value & exponent))
                      value &= sign;
                    wanted[reg ? second : first][lane] = uint32_t(value >> (32 * reg));
                  }
              ASSERT_EQ(std::memcmp(words, wanted, sizeof(words)), 0)
                  << arch << '/' << wide << '/' << op << '/' << guest << '/' << mode << '/' << cpu;
            }
          }
}

TEST(DivFmasArchitecture, LiteralDenormalsAndOutputScaling) {
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned op = 0; op < 2; ++op)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t words[8][64] = {};
            const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]},
                           *c[] = {words[4], words[5]};
            uint32_t *d[] = {words[6], words[7]};
            const uint64_t one = op ? 0x3ff0000000000000ULL : 0x3f800000ULL;
            const uint64_t sign = op ? 0x8000000000000000ULL : 0x80000000ULL;
            const uint64_t minimum = op ? 0x0010000000000000ULL : 0x00800000ULL;
            // Tiny signed products, subnormal inputs yielding a normal result,
            // and a normal result that exposes RDNA3's OMOD gating.
            const uint64_t av[] = {1, sign | 1, minimum / 2, one};
            const uint64_t bv[] = {one, one, one + minimum, one};
            for (unsigned lane = 0; lane < 64; ++lane) {
              words[0][lane] = uint32_t(av[lane % 4]);
              words[1][lane] = uint32_t(av[lane % 4] >> 32);
              words[2][lane] = uint32_t(bv[lane % 4]);
              words[3][lane] = uint32_t(bv[lane % 4] >> 32);
            }
            for (unsigned scaled = 0; scaled < 2; ++scaled) {
              uint64_t mode = scaled ? GOC_ALU_OMOD_2 : 0;
              bool active = scaled && (arch || ((guest & 2) && !(guest & 4)));
              uint64_t semantics = !arch && active ? GOC_SEMANTICS_LOOSE : exact;
              ASSERT_EQ(call(arch, wide, op, cpu | policy(guest) | semantics, UINT64_MAX, mode, d,
                             a, b, c, 0),
                        GOC_SUCCESS);
              const uint64_t expected[] = {active || (guest & 2) ? 0ULL : 1ULL,
                                           active        ? 0
                                           : (guest & 2) ? sign
                                                         : sign | 1,
                                           active ? minimum * 2 : minimum,
                                           active ? one + minimum : one};
              for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane) {
                uint64_t value = words[6][lane];
                if (op)
                  value |= uint64_t(words[7][lane]) << 32;
                ASSERT_EQ(value, expected[lane % 4]);
              }
            }
          }
}

TEST(DivFmasArchitecture, ReportingErrorsAndHostEnvironment) {
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
            if (!arch || (guest & 2))
              for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT})
                for (uint64_t exec_mask : {uint64_t(0), UINT64_MAX}) {
                  ASSERT_EQ(call(arch, wide, op,
                                 policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL | strict, exec_mask,
                                 0, d, nullptr, nullptr, nullptr, 0, &state),
                            GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
                  EXPECT_EQ(state, 0x12345678U);
                  for (unsigned lane = 0; lane < 64; ++lane) {
                    EXPECT_EQ(data[0][lane], 1U);
                    EXPECT_EQ(data[1][lane], 0U);
                  }
                }
            if (!arch && (guest & 2) && !(guest & 4)) {
              EXPECT_EQ(call(arch, wide, op, policy(guest) | exact, 0, GOC_ALU_OMOD_2, d, nullptr,
                             nullptr, nullptr, 0),
                        GOC_ERROR_UNSUPPORTED_SEMANTICS);
            }
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
              ASSERT_EQ(call(arch, wide, op, cpu | policy(guest) | exact, UINT64_MAX, 0, d, a, a, a,
                             UINT64_MAX),
                        GOC_SUCCESS);
              EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
              EXPECT_EQ(_mm_getcsr(), controls);
#endif
            }
          }
  }
}

TEST(DivFmasArchitecture, BorrowedFusedScalingWitnesses) {
  // rocjitsu tests/division_test.cpp at ffc144c564c: scale direction,
  // subnormal midpoint, avoided overflow, and exact cancellation witnesses.
  struct Case {
    uint64_t a, b, c, result;
    bool post;
  };

  const Case cases[2][7] = {
      {{bits(0.f), bits(0.f), bits(1.f), bits(0x1p-64f), true},
       {bits(0.f), bits(0.f), bits(2.f), bits(0x1p65f), true},
       {bits(1.f), bits(2.f), bits(0.f), bits(0x1p-63f), true},
       {bits(0x1p-100f), bits(0x1p-100f), bits(0x1p-86f), 1, true},
       {bits(-0x1p-100f), bits(0x1p-100f), bits(0x1p-86f), 0, true},
       {bits(0x1p100f), bits(0x1p50f), bits(0.f), bits(0x1p86f), true},
       {bits(0x1.000002p0f), bits(0x1.fffffcp-1f), bits(-1.f), bits(-0x1p-46f), false}},
      {{bits(0.), bits(0.), bits(1.), bits(0x1p-128), true},
       {bits(0.), bits(0.), bits(2.), bits(0x1p129), true},
       {bits(1.), bits(2.), bits(0.), bits(0x1p-127), true},
       {bits(0x1p-600), bits(0x1p-600), bits(0x1p-947), 1, true},
       {bits(-0x1p-600), bits(0x1p-600), bits(0x1p-947), 0, true},
       {bits(0x1p800), bits(0x1p300), bits(0.), bits(0x1p972), true},
       {bits(0x1.0000000000001p0), bits(0x1.ffffffffffffep-1), bits(-1.), bits(-0x1p-104), false}}};
  for (unsigned op = 0; op < 2; ++op)
    for (unsigned arch = 0; arch < 2; ++arch)
      for (unsigned wide = 0; wide < 2; ++wide)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t words[8][64] = {};
            uint64_t condition = 0;
            for (unsigned lane = 0; lane < 64; ++lane) {
              const auto &entry = cases[op][lane % 7];
              uint64_t inputs[] = {entry.a, entry.b, entry.c};
              for (unsigned operand = 0; operand < 3; ++operand) {
                words[operand * 2][lane] = uint32_t(inputs[operand]);
                words[operand * 2 + 1][lane] = uint32_t(inputs[operand] >> 32);
              }
              condition |= uint64_t(entry.post) << lane;
            }
            const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2], words[3]},
                           *c[] = {words[4], words[5]};
            uint32_t *d[] = {words[6], words[7]};
            ASSERT_EQ(call(arch, wide, op, exact | cpu | policy(guest), UINT64_MAX, 0, d, a, b, c,
                           condition),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane) {
              uint64_t value = words[6][lane];
              if (op)
                value |= uint64_t(words[7][lane]) << 32;
              uint64_t expected = cases[op][lane % 7].result;
              if ((guest & 2) && expected == 1)
                expected = 0;
              ASSERT_EQ(value, expected) << op << '/' << lane;
            }
          }
}
