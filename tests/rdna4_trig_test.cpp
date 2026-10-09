// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"
#include "rdna4_exec_masks.h"
#include "rdna4_trig_golden.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

using Fn = decltype(&goc_rdna4_v_sin_f32);
const Fn functions[] = {goc_rdna4_v_sin_f32, goc_rdna4_v_cos_f32};

uint32_t modifiers(int mode) {
  return (mode & 1 ? GOC_ALU_NEG_A : 0) | (mode & 2 ? GOC_ALU_ABS_A : 0) |
         (uint32_t((mode >> 2) & 3) << 6) | (mode & 16 ? GOC_ALU_CLAMP : 0);
}

// Double scaling is exact for every finite captured result. Narrow once to
// FP32, then apply the captured active-OMOD output flush and clamp policy.
uint32_t scale_reference(uint32_t bits, int mode) {
  const double scales[] = {1, 2, 4, 0.5};
  float result = float(double(goc::as_float(bits)) * scales[(mode >> 2) & 3]);
  if (mode & 16)
    result = !(result > 0) ? 0 : result > 1 ? 1 : result;
  bits = goc::as_bits(result);
  if ((mode & 12) && (bits & 0x7fffffff) < 0x800000)
    return 0;
  return bits;
}

} // namespace

TEST(Trig, CapturedResultsWithEveryModifier) {
  for (int op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int mode = 0; mode < 32; ++mode)
        for (const auto &golden : goc_test::trig_golden) {
          // Construct a source whose modified value has the captured sign.
          // ABS constrains that sign, so omit negative captures in that case.
          uint32_t input = golden[0];
          if (mode & 1)
            input ^= 0x80000000;
          if ((mode & 2) && (input >> 31))
            continue;
          uint32_t a[32], d[32];
          for (int lane = 0; lane < 32; ++lane)
            a[lane] = input | ((mode & 2) && (lane & 1) ? 0x80000000 : 0);
          const uint32_t *ap[] = {a};
          uint32_t *dp[] = {d};
          ASSERT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                  UINT32_MAX, modifiers(mode), dp, ap),
                    GOC_SUCCESS);
          uint32_t expected = scale_reference(golden[op + 1], mode);
          for (uint32_t value : d)
            ASSERT_EQ(value, expected) << op << "/" << cpu << "/" << mode << "/" << input;
        }
}

TEST(Trig, FullRangeRandomInputsAndIntervalBoundaries) {
  for (int op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      std::mt19937 random(781);
      for (int start = 0; start < 65536; start += 32) {
        uint32_t a[32], d[32];
        const uint32_t *ap[] = {a};
        uint32_t *dp[] = {d};
        for (int lane = 0; lane < 32; ++lane) {
          int index = start + lane;
          a[lane] = index < 1024 ? goc::as_bits(float(index / 8) / 128) + index % 8 - 4
                                 : uint32_t(random());
        }
        ASSERT_EQ(functions[op](cpu, UINT32_MAX, 0, dp, ap), GOC_SUCCESS);
        for (int lane = 0; lane < 32; ++lane) {
          uint32_t magnitude = a[lane] & 0x7fffffff;
          if (magnitude >= 0x7f800000) {
            EXPECT_EQ(d[lane], magnitude == 0x7f800000 ? 0xffc00000 : a[lane] | 0x400000);
          } else {
            double phase = std::remainder(double(goc::as_float(a[lane])), 1.0);
            double angle = phase * 6.283185307179586476925286766559;
            double expected = op ? std::cos(angle) : std::sin(angle);
            ASSERT_NEAR(double(goc::as_float(d[lane])), expected, 3e-7)
                << op << "/" << cpu << "/" << a[lane];
          }
        }
      }
    }
}

TEST(Trig, LooseMatchesScalarWithEveryModifier) {
  for (int op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int mode = 0; mode < 32; ++mode) {
        std::mt19937 random(8287);
        for (int start = 0; start < 8192; start += 32) {
          uint32_t a[32], reference[32], actual[32];
          for (int lane = 0; lane < 32; ++lane)
            a[lane] = start == 0 ? goc_test::trig_golden[lane][0] : uint32_t(random());
          const uint32_t *ap[] = {a};
          uint32_t *rp[] = {reference}, *dp[] = {actual};
          ASSERT_EQ(
              functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, modifiers(mode), rp, ap),
              GOC_SUCCESS);
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, modifiers(mode), dp, ap), GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane) {
            uint32_t magnitude = reference[lane] & 0x7fffffff;
            if (!magnitude || magnitude >= 0x7f800000) {
              ASSERT_EQ(actual[lane], reference[lane]) << op << "/" << cpu << "/" << mode;
            } else {
              const float scales[] = {1, 2, 4, 0.5f};
              ASSERT_NEAR(goc::as_float(actual[lane]), goc::as_float(reference[lane]),
                          3e-7f * scales[(mode >> 2) & 3])
                  << op << "/" << cpu << "/" << mode << "/" << a[lane];
            }
          }
        }
      }
}

TEST(Trig, EveryMaskAliasesAndModifier) {
  for (int op = 0; op < 2; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int mode = 0; mode < 32; ++mode) {
        uint32_t source[32], full[32];
        for (int lane = 0; lane < 32; ++lane)
          source[lane] = goc_test::trig_golden[lane][0];
        const uint32_t *ap[] = {source};
        uint32_t *fp[] = {full};
        ASSERT_EQ(functions[op](cpu, UINT32_MAX, modifiers(mode), fp, ap), GOC_SUCCESS);
        for (uint64_t mask : rdna4_exec_masks())
          for (bool alias : {false, true}) {
            uint32_t a[32], d[32];
            std::copy_n(source, 32, a);
            std::fill_n(d, 32, 0xdeadbeef);
            const uint32_t *input[] = {a};
            uint32_t *output[] = {alias ? a : d};
            ASSERT_EQ(functions[op](cpu, mask, modifiers(mode), output, input), GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane)
              ASSERT_EQ(output[0][lane], (mask >> lane) & 1 ? full[lane]
                                         : alias            ? source[lane]
                                                            : 0xdeadbeef);
          }
      }
}

TEST(Trig, ExactPreservesHostEnvironment) {
  uint32_t source[32], expected[2][32][32];
  for (int lane = 0; lane < 32; ++lane)
    source[lane] = goc_test::trig_golden[lane][0];
  const uint32_t *source_pointer[] = {source};
  for (int op = 0; op < 2; ++op)
    for (int mode = 0; mode < 32; ++mode) {
      uint32_t *destination[] = {expected[op][mode]};
      ASSERT_EQ(functions[op](GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, modifiers(mode),
                              destination, source_pointer),
                GOC_SUCCESS);
    }
  std::fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
#if defined(__x86_64__) || defined(_M_X64)
  const unsigned saved_mxcsr = _mm_getcsr();
#endif
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    for (int flush = 0; flush < 2; ++flush) {
      std::fesetround(rounding);
      std::feclearexcept(FE_ALL_EXCEPT);
      std::feraiseexcept(FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
      _mm_setcsr((_mm_getcsr() & ~0x8040u) | (flush ? 0x8040u : 0));
      unsigned before = _mm_getcsr();
#endif
      for (int op = 0; op < 2; ++op)
        for (int mode = 0; mode < 32; ++mode) {
          uint32_t d[32];
          uint32_t *dp[] = {d};
          EXPECT_EQ(functions[op](goc_init_cpu_flags() | GOC_SEMANTICS_EXACT_EMPIRICAL |
                                      GOC_SEMANTICS_STRICT,
                                  UINT32_MAX, modifiers(mode), dp, source_pointer),
                    GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(d[lane], expected[op][mode][lane]);
        }
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
      EXPECT_EQ(_mm_getcsr(), before);
#endif
    }
  std::fesetenv(&saved);
#if defined(__x86_64__) || defined(_M_X64)
  _mm_setcsr(saved_mxcsr);
#endif
}

TEST(Trig, InvalidFlagsAndReservedSemanticsPreserveDestination) {
  for (Fn fn : functions) {
    uint32_t a[32] = {}, d[32];
    std::fill_n(d, 32, 0xdeadbeef);
    const uint32_t *ap[] = {a};
    uint32_t *dp[] = {d};
    for (int bit = 0; bit < 32; ++bit) {
      uint32_t mode = uint32_t(1) << bit;
      if (mode & (GOC_ALU_NEG_A | GOC_ALU_ABS_A | GOC_ALU_OMOD_HALF | GOC_ALU_CLAMP))
        continue;
      for (uint64_t mask : {UINT64_C(0), UINT64_MAX})
        EXPECT_EQ(fn(0, mask, mode, dp, ap), GOC_ERROR_INVALID_FLAGS);
    }
    EXPECT_EQ(fn(UINT64_C(1) << 63, 0, 0, dp, ap), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(GOC_SEMANTICS_MASK | GOC_SEMANTICS_STRICT, 0, 0, dp, ap),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    for (uint32_t value : d)
      EXPECT_EQ(value, 0xdeadbeef);
    EXPECT_EQ(fn(GOC_SEMANTICS_MASK, UINT32_MAX, 0, dp, ap), GOC_SUCCESS);
  }
}
