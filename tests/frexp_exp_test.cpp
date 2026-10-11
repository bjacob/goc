// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "dpp_reference.h"
#include "exec_masks.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "test_instruction.h"
#include "unary_capture_input.h"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>
#include <vector>

namespace {

using Fn = decltype(&goc_v_frexp_exp_i32_f32);
const Fn functions[] = {goc_v_frexp_exp_i32_f32, goc_v_frexp_exp_i32_f64_rdna4};

uint32_t mode(int variant) {
  return (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
         ((variant >> 2 & 3) << 6) | (variant & 16 ? GOC_ALU_CLAMP : 0);
}

// Independent integer reference: find the leading significand bit and combine
// its position with the exponent field. Never interpret a NaN as a host float.
uint32_t reference(uint64_t bits, bool fp64) {
  int width = fp64 ? 52 : 23, bias = fp64 ? 1023 : 127;
  uint64_t fraction = bits & ((1ULL << width) - 1);
  int field = int(bits >> width) & (2 * bias + 1);
  if (field == 2 * bias + 1 || (field == 0 && fraction == 0))
    return 0;
  if (field)
    return uint32_t(field - bias + 1);
  int leading = 0;
  while (fraction >>= 1)
    ++leading;
  return uint32_t(leading + 2 - bias - width);
}

} // namespace

TEST(FrexpExp, AllModifiersMasksAndAliases) {
  for (int fp64 = 0; fp64 < 2; ++fp64) {
    const int width = fp64 ? 52 : 23, bias = fp64 ? 1023 : 127;
    const uint64_t sign = 1ULL << (fp64 ? 63 : 31);
    const uint64_t min_normal = 1ULL << width;
    const uint64_t inf = uint64_t(2 * bias + 1) << width;
    const uint64_t inputs[] = {0,
                               sign,
                               1,
                               sign | 1,
                               min_normal - 1,
                               sign | (min_normal - 1),
                               min_normal,
                               sign | min_normal,
                               min_normal + 1,
                               sign | (min_normal + 1),
                               inf - 1,
                               sign | (inf - 1),
                               inf,
                               sign | inf,
                               inf | 1,
                               sign | inf | 1,
                               inf | (min_normal >> 1),
                               sign | inf | (min_normal >> 1),
                               uint64_t(bias) << width,
                               sign | (uint64_t(bias) << width),
                               uint64_t(bias - 1) << width,
                               sign | (uint64_t(bias - 1) << width),
                               (uint64_t(bias + 5) << width) | 31,
                               (uint64_t(bias - 5) << width) | 31,
                               2,
                               3,
                               4,
                               7,
                               min_normal / 2,
                               min_normal / 2 - 1,
                               inf | 12345,
                               sign | inf | 12345};
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int variant = 0; variant < 32; ++variant)
        for (uint32_t exec_mask : exec_masks())
          for (int alias = 0; alias < (fp64 ? 3 : 2); ++alias) {
            SCOPED_TRACE(::testing::Message() << fp64 << "/" << cpu << "/" << variant << "/"
                                              << exec_mask << "/" << alias);
            uint32_t storage[3][34];
            for (auto &reg : storage)
              std::fill(reg, reg + 34, 0xdeadbeef);
            uint32_t *a[] = {storage[0] + 1, storage[1] + 1};
            uint32_t *d = storage[alias == 0 ? 2 : alias - 1] + 1;
            for (int lane = 0; lane < 32; ++lane) {
              a[0][lane] = uint32_t(inputs[lane]);
              a[1][lane] = uint32_t(inputs[lane] >> 32);
            }
            std::array<uint32_t, 32> before;
            std::copy(d, d + 32, before.begin());
            ASSERT_EQ(functions[fp64](cpu, exec_mask, mode(variant), &d, a, nullptr), GOC_SUCCESS);
            for (int lane = 0; lane < 32; ++lane)
              EXPECT_EQ(d[lane],
                        (exec_mask >> lane & 1) ? reference(inputs[lane], fp64) : before[lane]);
            for (const auto &reg : storage) {
              EXPECT_EQ(reg[0], 0xdeadbeef);
              EXPECT_EQ(reg[33], 0xdeadbeef);
            }
          }
  }
}

TEST(FrexpExp, EveryExponentAndSubnormalLeadingBit) {
  for (int fp64 = 0; fp64 < 2; ++fp64) {
    int width = fp64 ? 52 : 23, max_field = fp64 ? 2047 : 255;
    uint64_t sign = 1ULL << (fp64 ? 63 : 31);
    uint64_t fraction = (1ULL << width) - 1;
    std::vector<uint64_t> inputs;
    for (int field = 0; field <= max_field; ++field)
      for (uint64_t tail : std::initializer_list<uint64_t>{0ULL, 1ULL, fraction / 2, fraction}) {
        uint64_t value = (uint64_t(field) << width) | tail;
        inputs.push_back(value);
        inputs.push_back(value | sign);
      }
    for (int bit = 0; bit < width; ++bit)
      for (uint64_t value : {1ULL << bit, (1ULL << bit) - 1}) {
        inputs.push_back(value);
        inputs.push_back(value | sign);
      }
    std::mt19937_64 random(417);
    for (int i = 0; i < 1024; ++i)
      inputs.push_back(random() & (fp64 ? UINT64_MAX : UINT32_MAX));
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (int variant = 0; variant < 32; ++variant)
        for (size_t start = 0; start < inputs.size(); start += 32) {
          SCOPED_TRACE(::testing::Message()
                       << fp64 << "/" << cpu << "/" << variant << "/" << start);
          uint32_t a[2][32], d[32];
          uint32_t *pa[] = {a[0], a[1]}, *pd = d;
          for (int lane = 0; lane < 32; ++lane) {
            uint64_t value = inputs[(start + lane) % inputs.size()];
            a[0][lane] = uint32_t(value);
            a[1][lane] = uint32_t(value >> 32);
          }
          ASSERT_EQ(functions[fp64](cpu, UINT32_MAX, mode(variant), &pd, pa, nullptr), GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(d[lane], reference(inputs[(start + lane) % inputs.size()], fp64));
        }
  }
}

TEST(FrexpExp, LiteralValuesAndValidation) {
  const uint64_t inputs[][8] = {
      {1, 0x007fffff, 0x00800000, 0x3f000000, 0x3f800000, 0x7f7fffff, 0, 0x7f800001},
      {1, 0x000fffffffffffff, 0x0010000000000000, 0x3fe0000000000000, 0x3ff0000000000000,
       0x7fefffffffffffff, 0, 0x7ff0000000000001}};
  const int expected[][8] = {{-148, -126, -125, 0, 1, 128, 0, 0},
                             {-1073, -1022, -1021, 0, 1, 1024, 0, 0}};
  for (int fp64 = 0; fp64 < 2; ++fp64)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t a[2][32], d[32];
      uint32_t *pa[] = {a[0], a[1]}, *pd = d;
      for (int lane = 0; lane < 32; ++lane) {
        a[0][lane] = uint32_t(inputs[fp64][lane % 8]);
        a[1][lane] = uint32_t(inputs[fp64][lane % 8] >> 32);
      }
      ASSERT_EQ(functions[fp64](cpu, UINT32_MAX, 0, &pd, pa, nullptr), GOC_SUCCESS);
      for (int lane = 0; lane < 32; ++lane)
        EXPECT_EQ(d[lane], uint32_t(expected[fp64][lane % 8]));
      std::fill(d, d + 32, 0xdeadbeef);
      for (uint32_t exec_mask : {0U, UINT32_MAX}) {
        for (uint32_t invalid : {GOC_ALU_NEG_B, GOC_ALU_ABS_B, GOC_ALU_HIGH_D, 1U << 31})
          EXPECT_EQ(functions[fp64](cpu, exec_mask, invalid, &pd, pa, nullptr),
                    GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(functions[fp64](cpu | (1ULL << 63), exec_mask, 0, &pd, pa, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(functions[fp64](cpu | (2ULL << 16) | GOC_SEMANTICS_STRICT, exec_mask, 0, &pd, pa,
                                  nullptr),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
      for (uint32_t value : d)
        EXPECT_EQ(value, 0xdeadbeef);
      EXPECT_EQ(
          functions[fp64](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &pd, pa, nullptr),
          GOC_SUCCESS);
    }
}

TEST(FrexpExp, DppHardwareCorpus) {
  const uint32_t values[] = {1,          0x807fffff, 0x00800000, 0x80000000,
                             0x7f800001, 0xff800000, 0x3f800000, 0xff7fffff};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (uint32_t exec_mask :
         {0xffffffffu, 0u, 0xaaaaaaaau, 0x55555555u, 1u, 0x80000000u, 0xffffu, 0xffff0000u})
      for (uint64_t descriptor : goc_test::dpp_modes)
        for (uint32_t low : {0u, 1u, 8u, 9u, 256u, 257u, 264u, 265u}) {
          uint32_t a[32], d[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            a[lane] = values[lane % 8];
            d[lane] = 0xdead0000u + lane;
          }
          auto pa = a, pd = d;
          ASSERT_EQ(functions[0](cpu, exec_mask, descriptor | low, &pd, &pa, nullptr), GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane) {
            int source;
            uint32_t want = 0xdead0000u + lane;
            if (goc_test::dpp_source(descriptor, exec_mask, lane, source))
              want = reference(source < 0 ? 0 : a[source], false);
            EXPECT_EQ(d[lane], want);
            hash = goc_test::capture_hash_word(hash, d[lane]);
          }
        }
    EXPECT_EQ(hash, 0xb96b26fadd63a1e5ULL);
  }
}

TEST(FrexpExp, DppModifiersMasksAliasesAndRandomWords) {
  std::mt19937 random(812);
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t descriptor : goc_test::dpp_modes)
      for (int variant = 0; variant < 32; ++variant)
        for (uint32_t exec_mask : exec_masks())
          for (bool alias : {false, true}) {
            uint32_t words[2][34], before[2][34];
            for (auto &reg : words)
              for (auto &word : reg)
                word = random();
            std::memcpy(before, words, sizeof(words));
            auto a = words[0] + 1, d = words[alias ? 0 : 1] + 1;
            ASSERT_EQ(functions[0](cpu, exec_mask, descriptor | mode(variant), &d, &a, nullptr),
                      GOC_SUCCESS);
            for (unsigned reg = 0; reg < 2; ++reg)
              for (unsigned lane = 0; lane < 34; ++lane) {
                int source;
                uint32_t want = before[reg][lane];
                if (reg == (alias ? 0u : 1u) && lane >= 1 && lane <= 32 &&
                    goc_test::dpp_source(descriptor, exec_mask, lane - 1, source))
                  want = reference(source < 0 ? 0 : before[0][source + 1], false);
                EXPECT_EQ(words[reg][lane], want);
              }
          }
}

TEST(FrexpExp, Fp32PreservesHostEnvironment) {
  const uint32_t values[] = {1,          0x807fffff, 0x00800000, 0x80000000,
                             0x7f800001, 0xff800000, 0x3f800000, 0xff7fffff};
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
      for (unsigned descriptor = 0; descriptor < 8; ++descriptor)
        for (unsigned variant = 0; variant < 32; ++variant) {
          uint64_t flags = descriptor ? goc_test::dpp_modes[descriptor - 1] : 0;
          uint32_t a[32], d[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            a[lane] = values[lane % 8];
            d[lane] = 0xdeadbeef;
          }
          auto pa = a, pd = d;
          std::fesetround(rounding);
          std::feclearexcept(FE_ALL_EXCEPT);
          std::feraiseexcept(FE_DIVBYZERO);
          EXPECT_EQ(functions[0](cpu, 0xaaaaaaaau, flags | mode(variant), &pd, &pa, nullptr),
                    GOC_SUCCESS);
          EXPECT_EQ(std::fegetround(), rounding);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
          for (unsigned lane = 0; lane < 32; ++lane) {
            int source = lane;
            bool active = descriptor ? goc_test::dpp_source(flags, 0xaaaaaaaau, lane, source)
                                     : (lane % 2 == 1);
            EXPECT_EQ(d[lane], active ? reference(source < 0 ? 0 : a[source], false) : 0xdeadbeef);
          }
        }
}

TEST(FrexpExp, DppValidation) {
  for (uint64_t descriptor : goc_test::dpp_modes) {
    EXPECT_EQ(functions[0](0, 0, descriptor, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(functions[0](0, UINT32_MAX, descriptor | GOC_ALU_HIGH_A, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(functions[0](0, UINT32_MAX, descriptor | (1ULL << 36), nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(functions[0]((2ULL << 16) | GOC_SEMANTICS_STRICT, UINT32_MAX, descriptor, nullptr,
                           nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
  }
}

TEST(FrexpExp, Wave64ExponentFieldsModifiersDppAndAliases) {
  const decltype(&goc_v_frexp_exp_i32_f32_wave64) wave64[] = {goc_v_frexp_exp_i32_f32_wave64,
                                                              goc_v_frexp_exp_i32_f64_rdna4_wave64};
  for (unsigned fp64 = 0; fp64 < 2; ++fp64)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      unsigned width = fp64 ? 52 : 23;
      unsigned fields = fp64 ? 2048 : 256;
      for (unsigned start = 0; start < fields; start += 64)
        for (unsigned variant = 0; variant < 32; ++variant) {
          uint32_t words[3][64];
          for (unsigned lane = 0; lane < 64; ++lane) {
            uint64_t bits = uint64_t(start + lane) << width;
            bits |= lane & 1 ? (1ULL << width) - 1 : 0;
            words[0][lane] = uint32_t(bits);
            words[1][lane] = uint32_t(bits >> 32);
          }
          const uint32_t *a[] = {words[0], words[1]};
          uint32_t *d = words[2];
          uint32_t excp_flag_user = 0x12345678;
          ASSERT_EQ(wave64[fp64](cpu, UINT64_MAX, mode(variant), &d, a, &excp_flag_user),
                    GOC_SUCCESS);
          EXPECT_EQ(excp_flag_user, 0x12345678U);
          for (unsigned lane = 0; lane < 64; ++lane)
            ASSERT_EQ(d[lane],
                      reference(uint64_t(words[0][lane]) | (uint64_t(words[1][lane]) << 32), fp64));
        }
      for (unsigned variant = 0; variant < 32; ++variant)
        for (auto route : goc_test::dpp_modes) {
          if (fp64 && route != goc_test::dpp_modes[0])
            continue;
          for (unsigned rows = 0; rows < 16; ++rows)
            for (unsigned target = 0; target < 3; ++target) {
              uint32_t initial[3][66], words[3][66];
              for (unsigned lane = 0; lane < 66; ++lane) {
                uint64_t bits = lane < width ? 1ULL << lane : uint64_t(lane) << width;
                initial[0][lane] = uint32_t(bits);
                initial[1][lane] = uint32_t(bits >> 32);
                initial[2][lane] = 0xdeadbeef;
              }
              std::memcpy(words, initial, sizeof words);
              uint64_t descriptor = fp64 ? 0 : route;
              if (descriptor & GOC_DPP16)
                descriptor =
                    (descriptor & ~GOC_DPP_ROW_MASK) | (uint64_t(rows) << GOC_DPP_ROW_SHIFT);
              const uint32_t *a[] = {words[0] + 1, words[1] + 1};
              uint32_t *d = words[target] + 1;
              uint64_t exec_mask = 0x87654321fedcba98ULL;
              ASSERT_EQ(wave64[fp64](cpu, exec_mask, descriptor | mode(variant), &d, a, nullptr),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 64; ++lane) {
                int source = int(lane);
                bool enabled = exec_mask >> lane & 1;
                if (descriptor)
                  enabled = goc_test::dpp_source(descriptor, exec_mask, lane, source);
                uint64_t bits = source < 0
                                    ? 0
                                    : uint64_t(initial[0][source + 1]) |
                                          (fp64 ? uint64_t(initial[1][source + 1]) << 32 : 0);
                EXPECT_EQ(d[lane], enabled ? reference(bits, fp64) : initial[target][lane + 1]);
              }
              EXPECT_EQ(words[target][0], initial[target][0]);
              EXPECT_EQ(words[target][65], initial[target][65]);
            }
        }
      EXPECT_EQ(wave64[fp64](cpu, 0, 0, nullptr, nullptr, nullptr), GOC_SUCCESS);
      EXPECT_EQ(wave64[fp64](cpu, UINT64_MAX, 1ULL << 63, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
      uint32_t excp_flag_user = 0x12345678;
      EXPECT_EQ(wave64[fp64](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL, UINT64_MAX, 0, nullptr, nullptr,
                             &excp_flag_user),
                GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
      EXPECT_EQ(excp_flag_user, 0x12345678U);
    }
}

TEST(FrexpExp, Fp64HostDenormalControlsAndRoundingDoNotAffectResults) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    for (unsigned controls = 0; controls < 4; ++controls) {
      ASSERT_EQ(std::fesetround(rounding), 0);
      ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
      ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
#if defined(__x86_64__) || defined(_M_X64)
      unsigned csr =
          (_mm_getcsr() & ~0x8040U) | (controls & 1 ? 0x40U : 0) | (controls & 2 ? 0x8000U : 0);
      _mm_setcsr(csr);
#endif
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned bit = 0; bit < 52; ++bit) {
          uint32_t input[2][64], output[64];
          uint64_t cases[] = {1ULL << bit,
                              (1ULL << (bit + 1)) - 1,
                              0x000fffffffffffffULL,
                              0x0010000000000000ULL,
                              0x7ff0000000000001ULL,
                              0x7ff0000000000000ULL,
                              0,
                              0x7fefffffffffffffULL};
          for (unsigned lane = 0; lane < 64; ++lane) {
            uint64_t bits = cases[lane % 8] | (lane & 8 ? 1ULL << 63 : 0);
            input[0][lane] = uint32_t(bits);
            input[1][lane] = uint32_t(bits >> 32);
          }
          const uint32_t *a[] = {input[0], input[1]};
          uint32_t *d = output;
          ASSERT_EQ(goc_v_frexp_exp_i32_f64_rdna4_wave64(cpu, UINT64_MAX, 0, &d, a, nullptr),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 64; ++lane)
            ASSERT_EQ(output[lane], reference(cases[lane % 8], true));
          EXPECT_EQ(std::fegetround(), rounding);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
          EXPECT_EQ(_mm_getcsr(), csr);
#endif
        }
    }
}

namespace {

int invoke(bool rdna3, bool wide, uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *d,
           const uint32_t *a, uint32_t *state) {
  if (wide)
    return (rdna3 ? goc_v_frexp_exp_i32_f32_wave64
                  : goc_v_frexp_exp_i32_f32_rdna4_wave64)(flags, exec_mask, mode, &d, &a, state);
  return (rdna3 ? goc_v_frexp_exp_i32_f32
                : goc_v_frexp_exp_i32_f32_rdna4)(flags, uint32_t(exec_mask), mode, &d, &a, state);
}

uint32_t modifiers(unsigned m) {
  return (m & 1 ? GOC_ALU_NEG_A : 0) | (m & 2 ? GOC_ALU_ABS_A : 0) | ((m >> 2 & 3) << 6) |
         (m & 16 ? GOC_ALU_CLAMP : 0);
}

} // namespace

TEST(FrexpExpExact, HardwareNumericsAndGuestPolicy) {
  for (bool exact : {false, true})
    for (bool rdna3 : {false, true})
      for (bool wide : {false, true})
        for (unsigned policy = 0; policy < 8; ++policy)
          for (unsigned m = 0; m < 32; ++m) {
            unsigned lanes = wide ? 64 : 32;
            uint64_t hash = goc_test::capture_hash_seed;
            for (unsigned base = 0; base < 8192; base += lanes) {
              uint32_t input[64], output[64];
              for (unsigned lane = 0; lane < lanes; ++lane)
                input[lane] = goc_test::unary_capture_input(0, base + lane);
              uint64_t flags = ((base / lanes) % (goc_init_cpu_flags() + 1)) |
                               (exact ? GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT : 0) |
                               (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                               (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                               (policy & 4 ? GOC_FP_IEEE_MODE : 0);
              ASSERT_EQ(
                  invoke(rdna3, wide, flags, UINT64_MAX, modifiers(m), output, input, nullptr),
                  GOC_SUCCESS);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                uint32_t magnitude = input[lane] & 0x7fffffff;
                uint32_t expected =
                    (policy & 1) && magnitude < 0x800000 ? 0 : reference(input[lane], false);
                ASSERT_EQ(output[lane], expected)
                    << "exact=" << exact << " rdna3=" << rdna3 << " wide=" << wide
                    << " flags=" << flags << " modifier=" << m << " input_index=" << base + lane;
                hash = goc_test::capture_hash_word(hash, output[lane]);
              }
            }
            const uint64_t hashes[] = {0xcc12cb39440259e1ULL, 0xcf729b6b9bdb7fbaULL};
            EXPECT_EQ(hash, hashes[policy & 1])
                << exact << '/' << rdna3 << '/' << wide << '/' << policy << '/' << m;
          }
}

TEST(FrexpExpExact, DppAliasesGuestModesAndHostControls) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (bool exact : {false, true})
    for (bool rdna3 : {false, true})
      for (bool wide : {false, true})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (unsigned policy = 0; policy < 8; ++policy)
            for (auto descriptor : goc_test::dpp_modes) {
              unsigned lanes = wide ? 64 : 32;
              uint64_t exec_mask = wide ? 0x9192a5a571abcdefULL : 0x71abcdef;
              uint64_t flags = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT |
                               (policy & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                               (policy & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                               (policy & 4 ? GOC_FP_IEEE_MODE : 0);
              unsigned m = (policy * 5) % 32;
              uint32_t original[64], expected[64];
              for (unsigned lane = 0; lane < lanes; ++lane)
                original[lane] = goc_test::unary_capture_input(0, lane % 32);
              for (unsigned lane = 0; lane < lanes; ++lane) {
                expected[lane] = original[lane];
                int source;
                if (goc_test::dpp_source(descriptor, exec_mask, lane, source)) {
                  uint32_t input = source < 0 ? 0 : original[source];
                  uint32_t magnitude = input & 0x7fffffff;
                  expected[lane] =
                      (policy & 1) && magnitude < 0x800000 ? 0 : reference(input, false);
                }
              }
              flags = (exact ? flags : flags & ~(GOC_SEMANTICS_MASK | GOC_SEMANTICS_STRICT)) | cpu;
              for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO})
                for (unsigned controls = 0; controls < 4; ++controls) {
                  ASSERT_EQ(std::fesetround(rounding), 0);
                  std::feclearexcept(FE_ALL_EXCEPT);
                  std::feraiseexcept(FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
                  _mm_setcsr((_mm_getcsr() & ~0x8040U) | (controls & 1 ? 0x40U : 0) |
                             (controls & 2 ? 0x8000U : 0));
                  unsigned host_controls = _mm_getcsr();
#endif
                  uint32_t input[64], state = 0x80000040;
                  std::memcpy(input, original, lanes * sizeof(uint32_t));
                  ASSERT_EQ(invoke(rdna3, wide, flags, exec_mask, modifiers(m) | descriptor, input,
                                   input, exact ? nullptr : &state),
                            GOC_SUCCESS);
                  for (unsigned lane = 0; lane < lanes; ++lane)
                    EXPECT_EQ(input[lane], expected[lane]);
                  if (!exact) {
                    EXPECT_EQ(state, 0x80000040U);
                  }
                  EXPECT_EQ(std::fegetround(), rounding);
                  EXPECT_EQ(std::fetestexcept(FE_DIVBYZERO), FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
                  EXPECT_EQ(_mm_getcsr() & ~0x3fU, host_controls & ~0x3fU);
#endif
                }
            }
}

TEST(FrexpExpExact, ValidationAndEmptyExec) {
  for (bool rdna3 : {false, true})
    for (bool wide : {false, true}) {
      uint32_t input[64] = {}, output[64], state = 0x80000040;
      std::fill(output, output + 64, 0xdeadbeef);
      EXPECT_EQ(invoke(rdna3, wide, 0, UINT64_MAX, GOC_ALU_NEG_B, output, input, &state),
                GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(invoke(rdna3, wide, 2ULL << 16, UINT64_MAX, 0, output, input, &state),
                GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
      EXPECT_EQ(invoke(rdna3, wide, GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, nullptr, nullptr, &state),
                GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
      EXPECT_EQ(invoke(rdna3, wide, GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, nullptr, nullptr, nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(state, 0x80000040U);
      for (auto value : output)
        EXPECT_EQ(value, 0xdeadbeefU);
    }
}

namespace {

using ExponentInstruction = goc_test::WaveInstruction<decltype(&goc_v_cvt_f64_i32)>;
const ExponentInstruction fp64_instructions[2][2] = {
    {goc_v_frexp_exp_i32_f64, goc_v_frexp_exp_i32_f64_wave64},
    {goc_v_frexp_exp_i32_f64_rdna4, goc_v_frexp_exp_i32_f64_rdna4_wave64}};

uint64_t fp64_policy(unsigned guest) {
  return (guest & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (guest & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (guest & 4 ? GOC_FP_IEEE_MODE : 0);
}

} // namespace

TEST(FrexpExp, Fp64ArchitecturePoliciesWidthsModifiersAndAliases) {
  const uint64_t values[] = {0,
                             0x8000000000000000ULL,
                             1,
                             0x8000000000000001ULL,
                             0xfffffffffffffULL,
                             0x10000000000000ULL,
                             0x10000000000001ULL,
                             0x3ff0000000000000ULL,
                             0xbfefffffffffffffULL,
                             0x7fefffffffffffffULL,
                             0x7ff0000000000000ULL,
                             0xfff0000000000000ULL,
                             0x7ff0000000000001ULL,
                             0xfff0000000000001ULL,
                             0x7ff8000000004321ULL,
                             0xfff8000000004321ULL};
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned guest = 0; guest < 8; ++guest)
        for (unsigned m = 0; m < 32; ++m)
          for (unsigned alias = 0; alias < 3; ++alias)
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
              uint32_t words[3][66], before[3][66];
              for (unsigned lane = 0; lane < 66; ++lane) {
                uint64_t bits = values[(lane + lane / 32) % 16];
                words[0][lane] = uint32_t(bits);
                words[1][lane] = uint32_t(bits >> 32);
                words[2][lane] = 0xdeadbeef;
              }
              std::memcpy(before, words, sizeof(words));
              const uint32_t *a[] = {words[0] + 1, words[1] + 1};
              uint32_t *d = words[alias] + 1, state = 0x80000055;
              const uint64_t exec_mask = 0xf13579bde2468acfULL;
              ASSERT_EQ(fp64_instructions[arch][wide](cpu | fp64_policy(guest), exec_mask,
                                                      modifiers(m), &d, a, &state),
                        GOC_SUCCESS);
              EXPECT_EQ(state, 0x80000055U);
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned lane = 0; lane < 66; ++lane) {
                  uint32_t want = before[reg][lane];
                  if (reg == alias && lane > 0 && lane <= (wide ? 64U : 32U) &&
                      ((exec_mask >> (lane - 1)) & 1))
                    want = reference(uint64_t(before[0][lane]) | (uint64_t(before[1][lane]) << 32),
                                     true);
                  ASSERT_EQ(words[reg][lane], want)
                      << arch << '/' << wide << '/' << guest << '/' << m;
                }
            }
}

TEST(FrexpExp, Fp64ArchitectureValidationAndHostEnvironment) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (const auto &architecture : fp64_instructions)
    for (auto instruction : architecture)
      for (unsigned guest = 0; guest < 8; ++guest)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          uint32_t words[2][64], output[64], state = 0x80000055;
          for (unsigned lane = 0; lane < 64; ++lane) {
            uint64_t bits = 1ULL << (lane % 52);
            words[0][lane] = uint32_t(bits);
            words[1][lane] = uint32_t(bits >> 32);
            output[lane] = 0xdeadbeef;
          }
          const uint32_t *a[] = {words[0], words[1]};
          uint32_t *d = output;
          for (uint64_t exec_mask : {0ULL, ~0ULL}) {
            EXPECT_EQ(instruction(cpu | fp64_policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL,
                                  exec_mask, 0, &d, a, &state),
                      GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
            EXPECT_EQ(instruction(cpu | fp64_policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL |
                                      GOC_SEMANTICS_STRICT,
                                  exec_mask, 0, &d, a),
                      GOC_ERROR_UNSUPPORTED_SEMANTICS);
            EXPECT_EQ(instruction(cpu | fp64_policy(guest), exec_mask, GOC_DPP8, &d, a, &state),
                      GOC_ERROR_INVALID_FLAGS);
            EXPECT_EQ(
                instruction(cpu | fp64_policy(guest), exec_mask, GOC_ALU_NEG_B, &d, a, &state),
                GOC_ERROR_INVALID_FLAGS);
          }
          for (auto word : output)
            EXPECT_EQ(word, 0xdeadbeefU);
          for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
            for (unsigned controls = 0; controls < 4; ++controls) {
              ASSERT_EQ(std::fesetround(rounding), 0);
              ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
              ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
#if defined(__x86_64__) || defined(_M_X64)
              unsigned csr = (_mm_getcsr() & ~0x8040U) | (controls & 1 ? 0x40U : 0) |
                             (controls & 2 ? 0x8000U : 0);
              _mm_setcsr(csr);
#endif
              EXPECT_EQ(instruction(cpu | fp64_policy(guest), UINT64_MAX, 0, &d, a, &state),
                        GOC_SUCCESS);
              for (unsigned lane = 0; lane < 64; ++lane) {
                // Wave32 leaves the upper half untouched even after repeated calls.
                uint32_t want = instruction.reporting_wave64 || lane < 32
                                    ? reference(1ULL << (lane % 52), true)
                                    : 0xdeadbeefU;
                EXPECT_EQ(output[lane], want);
              }
              EXPECT_EQ(std::fegetround(), rounding);
              EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64)
              EXPECT_EQ(_mm_getcsr(), csr);
#endif
            }
          EXPECT_EQ(state, 0x80000055U);
        }
}
