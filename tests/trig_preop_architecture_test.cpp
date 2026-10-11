// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "fp_environment.h"
#include "goc/goc.h"
#include "trig_preop_hardware.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

uint64_t policy(unsigned guest) {
  return (guest & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (guest & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (guest & 4 ? GOC_FP_IEEE_MODE : 0);
}

int call(bool rdna4, bool wide, uint64_t flags, uint64_t exec_mask, uint64_t mode,
         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
         uint32_t *state = nullptr) {
  if (wide)
    return (rdna4 ? goc_v_trig_preop_f64_rdna4_wave64
                  : goc_v_trig_preop_f64_wave64)(flags, exec_mask, mode, d, a, b, state);
  return (rdna4 ? goc_v_trig_preop_f64_rdna4 : goc_v_trig_preop_f64)(flags, uint32_t(exec_mask),
                                                                     mode, d, a, b, state);
}

// rocjitsu ffc144c564c, generated/shared/execute_shared.h,
// execute_v_trig_preop_f64_vop3: 1201 valid bits of 2/pi, RTZ lookup,
// followed by finish_f64. Use wider floating arithmetic as an independent
// reference for GoC's integer-only construction and scaling. The caller uses
// nearest-even host rounding with denormals preserved.
uint64_t reference(unsigned exponent, unsigned selector, unsigned mode, unsigned guest,
                   bool rdna4 = false) {
  constexpr char digits[] = "a2f9836e4e441529fc2757d1f534ddc0db6295993c439041fe5163abdebb"
                            "c561b7246e3a424dd2e006492eea09d1921cfe1deb1cb129a73ee88235f5"
                            "2ebb4484e99c7026b45f7e413991d639835339f49c845f8bbdf9283b1ff8"
                            "97ffde05980fef2f118b5a0a6d1f6d367ecf27cb09b74f463f669e5fea2d"
                            "7527bac7ebe5f17b3d0739f78a5292ea6bfb5fb11f8d5d0856033046fc7b6babf0";
  unsigned shift = 53 * (selector & 31) + (exponent > 1077 ? exponent - 1077 : 0);
  uint64_t segment = 0;
  for (unsigned i = 0; i < 53; ++i) {
    unsigned bit = shift + i;
    unsigned nibble = 0;
    if (bit < (rdna4 ? 1184U : 1201U)) {
      char c = digits[bit / 4];
      nibble = c <= '9' ? c - '0' : c - 'a' + 10;
    }
    segment = (segment << 1) | ((nibble >> (3 - bit % 4)) & 1);
  }
  long double unrounded = std::ldexp(static_cast<long double>(segment),
                                     -53 - int(shift) + (exponent >= 1968 ? 128 : 0));
  double value = static_cast<double>(unrounded);
  if (static_cast<long double>(value) > unrounded)
    value = std::nextafter(value, 0.0);
  unsigned omod = (mode >> 6) & 3;
  if (!rdna4 && (!(guest & 2) || (guest & 4)))
    omod = 0;
  if (omod) {
    if (rdna4 && std::fpclassify(value) == FP_SUBNORMAL)
      value = 0;
    value = static_cast<double>(
        std::ldexp(static_cast<long double>(value), omod == 3 ? -1 : int(omod)));
    if (std::fpclassify(value) == FP_SUBNORMAL)
      value = 0;
  }
  if (mode & GOC_ALU_CLAMP)
    value = std::min(value, 1.0);
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

} // namespace

TEST(TrigPreopArchitecture, BorrowedTableEveryExponentSelectorAndModifier) {
  for (unsigned mode = 0; mode < 8; ++mode) {
    unsigned instruction = (mode & 3) << 6 | (mode & 4 ? GOC_ALU_CLAMP : 0);
    for (unsigned e = 0; e < 2048; e += 2) {
      uint32_t high[64], selector[64], output[2][64];
      uint64_t want[64];
      for (unsigned lane = 0; lane < 64; ++lane) {
        high[lane] = (e + lane / 32) << 20 | (lane & 1) << 31 | 0xabcde;
        selector[lane] = 0xabcdefe0 | (lane & 31);
        want[lane] = reference(e + lane / 32, lane, instruction, 2);
      }
      const uint32_t *a[] = {nullptr, high}, *b[] = {selector};
      uint32_t *d[] = {output[0], output[1]};
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (bool wide : {false, true}) {
          ASSERT_EQ(call(false, wide, cpu | exact | policy(2), UINT64_MAX,
                         instruction | GOC_ALU_ABS_A | GOC_ALU_NEG_A, d, a, b),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane)
            ASSERT_EQ(uint64_t(output[0][lane]) | uint64_t(output[1][lane]) << 32, want[lane])
                << cpu << "/" << wide << "/" << e << "/" << lane << "/" << mode;
        }
    }
  }
}

TEST(TrigPreopArchitecture, GuestPoliciesMasksAliasesAndGuards) {
  for (bool rdna4 : {false, true})
    for (bool wide : {false, true})
      for (unsigned guest = 0; guest < 8; ++guest)
        for (unsigned mode = 0; mode < 32; ++mode) {
          const unsigned instruction = goc_test::trig_preop_mode(mode);
          for (unsigned target = 0; target < 16; ++target) {
            uint32_t initial[4][66], words[4][66], expected[4][66];
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned lane = 0; lane < 66; ++lane)
                initial[reg][lane] = 0xbad00000U + 73 * reg + lane;
            for (unsigned lane = 0; lane < 64; ++lane) {
              initial[0][lane + 1] = goc_test::trig_preop_exponent(lane % 32) << 20;
              initial[1][lane + 1] = goc_test::trig_preop_selector(lane % 32);
            }
            std::memcpy(expected, initial, sizeof(expected));
            uint64_t exec_mask = mode & 1 ? 0x87654321fedcba98ULL : UINT64_MAX;
            if (!wide)
              exec_mask = uint32_t(exec_mask);
            for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane)
              if ((exec_mask >> lane) & 1) {
                uint64_t value = reference(initial[0][lane + 1] >> 20, initial[1][lane + 1],
                                           instruction, guest, rdna4);
                expected[target / 4][lane + 1] = uint32_t(value);
                expected[target % 4][lane + 1] = uint32_t(value >> 32);
              }
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
              std::memcpy(words, initial, sizeof(words));
              const uint32_t *a[] = {nullptr, words[0] + 1}, *b[] = {words[1] + 1};
              uint32_t *d[] = {words[target / 4] + 1, words[target % 4] + 1}, state = 0x12345678;
              ASSERT_EQ(call(rdna4, wide, cpu | policy(guest) | (mode & 1 ? 0 : exact), exec_mask,
                             instruction, d, a, b, mode & 1 ? &state : nullptr),
                        GOC_SUCCESS);
              if (mode & 1) {
                ASSERT_EQ(state, 0x12345678U);
              }
              ASSERT_EQ(std::memcmp(words, expected, sizeof(words)), 0)
                  << rdna4 << "/" << wide << "/" << guest << "/" << mode << "/" << target << "/"
                  << cpu;
            }
          }
        }
}

TEST(TrigPreopArchitecture, LiteralArchitectureDifferences) {
  // Table-tail precision, tiny results normalized by OMOD, and a one-bit
  // rounding difference. Values follow the borrowed RDNA3 model and retained
  // RDNA4 table; no new hardware model is inferred from these witnesses.
  const unsigned exponent[] = {1968, 1091, 1091, 1997};
  const unsigned selector[] = {5, 19, 19, 4};
  const uint64_t expected[2][4] = {
      {0x180cc11bf1ecULL, 0x19e839cfbc5294ULL, 0x29e839cfbc5294ULL, 0x11aba10ac06608dULL},
      {0x180cc1180000ULL, 0, 0, 0x11aba10ac06608cULL}};
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned omod = 0; omod < 4; ++omod) {
      uint32_t high[64], selectors[64], output[2][64];
      std::fill_n(high, 64, exponent[omod] << 20);
      std::fill_n(selectors, 64, selector[omod]);
      const uint32_t *a[] = {nullptr, high}, *b[] = {selectors};
      uint32_t *d[] = {output[0], output[1]};
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (bool wide : {false, true}) {
          ASSERT_EQ(call(arch, wide, cpu | exact | policy(2), UINT64_MAX, omod << 6, d, a, b),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane)
            EXPECT_EQ(uint64_t(output[0][lane]) | uint64_t(output[1][lane]) << 32,
                      expected[arch][omod]);
        }
    }
}

TEST(TrigPreopArchitecture, ReportingErrorsAndHostControls) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  uint32_t high[64], selector[64], output[2][64];
  const uint32_t *a[] = {nullptr, high}, *b[] = {selector};
  uint32_t *d[] = {output[0], output[1]};
  for (unsigned lane = 0; lane < 64; ++lane) {
    high[lane] = goc_test::trig_preop_exponent(lane % 32) << 20;
    selector[lane] = goc_test::trig_preop_selector(lane % 32);
  }
  for (bool rdna4 : {false, true})
    for (bool wide : {false, true}) {
      uint32_t state = 0x12345678;
      for (uint64_t semantics : {GOC_SEMANTICS_EXACT_EMPIRICAL, exact})
        for (uint64_t exec_mask : {0ULL, ~0ULL})
          EXPECT_EQ(call(rdna4, wide, semantics, exec_mask, 0, nullptr, nullptr, nullptr, &state),
                    GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
      EXPECT_EQ(state, 0x12345678U);
      EXPECT_EQ(call(rdna4, wide, 0, 0, 0, nullptr, nullptr, nullptr, &state), GOC_SUCCESS);
      for (uint64_t mode : {uint64_t(GOC_ALU_NEG_B), uint64_t(1) << 32})
        EXPECT_EQ(call(rdna4, wide, 0, 0, mode, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(call(rdna4, wide, 1ULL << 63, 0, 0, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
          for (unsigned denorm = 0; denorm < 4; ++denorm) {
            ASSERT_EQ(std::fesetround(rounding), 0);
#if defined(__x86_64__) || defined(_M_X64)
            _mm_setcsr((_mm_getcsr() & ~0x8040U) | (denorm & 1 ? 0x40U : 0) |
                       (denorm & 2 ? 0x8000U : 0));
            unsigned controls = _mm_getcsr() & ~0x3fU;
#endif
            ASSERT_EQ(
                call(rdna4, wide, cpu | exact | policy(2), UINT64_MAX, GOC_ALU_OMOD_4, d, a, b),
                GOC_SUCCESS);
            EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
            EXPECT_EQ(_mm_getcsr() & ~0x3fU, controls);
#endif
          }
    }
}
