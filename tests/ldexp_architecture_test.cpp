// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "omod_reference.h"
#include "test_instruction.h"

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

using Binary = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *, const uint32_t *const *,
                       const uint32_t *const *);
using Instruction = goc_test::WaveInstruction<Binary>;
const Instruction instructions[2][2][2] = {
    {{goc_v_ldexp_f32, goc_v_ldexp_f64}, {goc_v_ldexp_f32_wave64, goc_v_ldexp_f64_wave64}},
    {{goc_v_ldexp_f32_rdna4, goc_v_ldexp_f64_rdna4},
     {goc_v_ldexp_f32_rdna4_wave64, goc_v_ldexp_f64_rdna4_wave64}}};

uint64_t policy(unsigned guest) {
  return (guest & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (guest & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (guest & 4 ? GOC_FP_IEEE_MODE : 0);
}

uint32_t modifiers(unsigned variant) {
  return (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
         ((variant >> 2 & 3) << 6) | (variant & 16 ? GOC_ALU_CLAMP : 0);
}

template <typename T, typename U> U reference(U bits, int exponent, uint32_t mode, unsigned guest) {
  const U sign = U(1) << (sizeof(U) * 8 - 1);
  const U normal = U(1) << (sizeof(U) == 8 ? 52 : 23);
  if ((guest & 1) && (bits & ~sign) < normal)
    bits &= sign;
  if (mode & GOC_ALU_ABS_A)
    bits &= ~sign;
  if (mode & GOC_ALU_NEG_A)
    bits ^= sign;
  T input;
  std::memcpy(&input, &bits, sizeof(input));
  long double wide = std::ldexp(static_cast<long double>(input), std::clamp(exponent, -4096, 4096));
  if ((mode & GOC_ALU_OMOD_HALF) && std::abs(wide) < (sizeof(T) == 8 ? 0x1p-1022L : 0x1p-126L))
    wide = 0;
  T result = T(wide);
  if constexpr (sizeof(T) == 8)
    result = goc_test::omod_f64_reference(result, mode);
  else
    result = goc_test::omod_f32_reference(result, mode);
  if (mode & GOC_ALU_CLAMP)
    result = !(result > 0) ? T(0) : result > 1 ? T(1) : result;
  std::memcpy(&bits, &result, sizeof(bits));
  if ((guest & 2) && (bits & ~sign) < normal)
    bits &= sign;
  return bits;
}

uint64_t expected(bool fp64, uint64_t input, int exponent, uint32_t mode, unsigned guest) {
  return fp64 ? reference<double>(input, exponent, mode, guest)
              : reference<float>(uint32_t(input), exponent, mode, guest);
}

} // namespace

TEST(LdexpArchitecture, LiteralScalingAndFlushing) {
  // Values chosen to separate input flushing, output flushing, and the RDNA3
  // IEEE/denormal OMOD gate without depending on rounding at an FP boundary.
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wave64 = 0; wave64 < 2; ++wave64)
      for (unsigned fp64 = 0; fp64 < 2; ++fp64)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t input[2][64], power[64], output[2][64];
            const unsigned fraction = fp64 ? 52 : 23;
            const uint64_t normal = 1ULL << fraction;
            const uint64_t three = fp64 ? 0x4008000000000000ULL : 0x40400000;
            for (unsigned lane = 0; lane < 64; ++lane) {
              uint64_t bits = lane % 3 == 0 ? three : lane % 3 == 1 ? 1 : normal;
              input[0][lane] = uint32_t(bits);
              input[1][lane] = uint32_t(bits >> 32);
              power[lane] = lane % 3 == 0 ? 0 : lane % 3 == 1 ? fraction : uint32_t(-2);
              output[0][lane] = output[1][lane] = 0xdeadbeef;
            }
            const uint32_t *a[] = {input[0], input[1]}, *b = power;
            uint32_t *d[] = {output[0], output[1]}, state = 0x80000055;
            ASSERT_EQ(instructions[arch][wave64][fp64](cpu | policy(guest), UINT64_MAX,
                                                       GOC_ALU_OMOD_2, d, a, &b, &state),
                      GOC_SUCCESS);
            const bool scale = arch || ((guest & 2) && !(guest & 4));
            for (unsigned lane = 0; lane < (wave64 ? 64U : 32U); ++lane) {
              uint64_t want = lane % 3 == 0
                                  ? (scale ? (fp64 ? 0x4018000000000000ULL : 0x40c00000) : three)
                              : lane % 3 == 1 ? (guest & 1 ? 0 : normal * (scale ? 2 : 1))
                                              : (scale || (guest & 2) ? 0 : normal / 4);
              EXPECT_EQ(output[0][lane], uint32_t(want));
              if (fp64) {
                EXPECT_EQ(output[1][lane], uint32_t(want >> 32));
              }
            }
            EXPECT_EQ(state, 0x80000055U);
          }
}

TEST(LdexpArchitecture, GuestPoliciesModifiersWidthsAndAliases) {
  const int powers[] = {INT32_MIN, -1074, -149, -1, 0, 1, 127, 1023, INT32_MAX};
  const unsigned aliases[][2] = {{3, 4}, {0, 1}, {2, 0}, {3, 3}};
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wave64 = 0; wave64 < 2; ++wave64)
      for (unsigned fp64 = 0; fp64 < 2; ++fp64)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (unsigned variant = 0; variant < 32; ++variant)
            for (const auto &alias : aliases)
              for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
                const unsigned fraction = fp64 ? 52 : 23;
                const uint64_t sign = 1ULL << (fp64 ? 63 : 31), normal = 1ULL << fraction;
                const uint64_t inf = uint64_t(fp64 ? 2047 : 255) << fraction;
                const uint64_t values[] = {0,
                                           sign,
                                           1,
                                           sign | 3,
                                           normal - 1,
                                           normal,
                                           2 * normal - 1,
                                           sign | (2 * normal - 1),
                                           inf - 1,
                                           inf,
                                           sign | inf,
                                           fp64 ? 0x3ff8000000000000ULL : 0x3fc00000};
                uint32_t words[5][66], want[5][66];
                for (auto &reg : words)
                  for (auto &word : reg)
                    word = 0xdeadbeef;
                for (unsigned lane = 0; lane < 64; ++lane) {
                  uint64_t value = values[(lane + lane / 32) % 12];
                  words[0][lane + 1] = uint32_t(value);
                  words[1][lane + 1] = uint32_t(value >> 32);
                  words[2][lane + 1] = uint32_t(powers[lane % 9]);
                }
                std::memcpy(want, words, sizeof(want));
                const uint64_t exec_mask = 0xf13579bde2468acfULL;
                const unsigned lanes = wave64 ? 64 : 32;
                uint32_t mode = modifiers(variant);
                uint32_t effective =
                    !arch && (!(guest & 2) || (guest & 4)) ? mode & ~GOC_ALU_OMOD_HALF : mode;
                for (unsigned reg = 0; reg <= fp64; ++reg)
                  for (unsigned lane = 0; lane < lanes; ++lane)
                    if ((exec_mask >> lane) & 1) {
                      uint64_t value = values[(lane + lane / 32) % 12];
                      want[alias[reg]][lane + 1] = uint32_t(
                          expected(fp64, value, powers[lane % 9], effective, guest) >> (32 * reg));
                    }
                const uint32_t *a[] = {words[0] + 1, words[1] + 1}, *b = words[2] + 1;
                uint32_t *d[] = {words[alias[0]] + 1, words[alias[1]] + 1};
                ASSERT_EQ(instructions[arch][wave64][fp64](cpu | policy(guest), exec_mask, mode, d,
                                                           a, &b),
                          GOC_SUCCESS);
                ASSERT_EQ(std::memcmp(words, want, sizeof(words)), 0)
                    << arch << '/' << wave64 << '/' << fp64 << '/' << guest << '/' << variant << '/'
                    << cpu;
              }
}

TEST(LdexpArchitecture, DppGuestPolicies) {
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wave64 = 0; wave64 < 2; ++wave64)
      for (unsigned guest = 0; guest < 8; ++guest)
        for (uint64_t descriptor : goc_test::dpp_modes)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t words[66], original[66], powers[64];
            for (unsigned lane = 0; lane < 66; ++lane)
              words[lane] = lane % 2 ? 0x00800000 + lane : lane;
            std::memcpy(original, words, sizeof(words));
            for (unsigned lane = 0; lane < 64; ++lane)
              powers[lane] = uint32_t(int(lane % 7) - 3);
            const uint64_t exec_mask = 0xf13579bde2468acfULL;
            const uint32_t mode = GOC_ALU_NEG_A | GOC_ALU_OMOD_2;
            const uint32_t effective =
                !arch && (!(guest & 2) || (guest & 4)) ? mode & ~GOC_ALU_OMOD_HALF : mode;
            uint32_t *d = words + 1;
            const uint32_t *a = d, *b = powers;
            ASSERT_EQ(instructions[arch][wave64][0](cpu | policy(guest), exec_mask,
                                                    descriptor | mode, &d, &a, &b),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 64; ++lane) {
              int source;
              bool active =
                  wave64 ? goc_test::dpp_source(descriptor, exec_mask, lane, source)
                         : lane < 32 &&
                               goc_test::dpp_source(descriptor, uint32_t(exec_mask), lane, source);
              uint32_t want = original[lane + 1];
              if (active)
                want = uint32_t(expected(false, source < 0 ? 0 : original[source + 1],
                                         int(lane % 7) - 3, effective, guest));
              ASSERT_EQ(words[lane + 1], want);
            }
            EXPECT_EQ(words[0], original[0]);
            EXPECT_EQ(words[65], original[65]);
          }
}

TEST(LdexpArchitecture, ErrorsAndHostControls) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (const auto &architecture : instructions)
    for (const auto &wave : architecture)
      for (auto instruction : wave)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (unsigned guest = 0; guest < 8; ++guest) {
            uint32_t words[2][64] = {}, output[2][64], state = 0x80000055;
            const uint32_t *a[] = {words[0], words[1]}, *b = words[0];
            uint32_t *d[] = {output[0], output[1]};
            std::memset(output, 0x5a, sizeof(output));
            for (uint64_t exec_mask : {0ULL, ~0ULL}) {
              EXPECT_EQ(instruction(cpu | policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL, exec_mask,
                                    0, d, a, &b, &state),
                        GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
              EXPECT_EQ(instruction(cpu | policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL |
                                        GOC_SEMANTICS_STRICT,
                                    exec_mask, 0, d, a, &b),
                        GOC_ERROR_UNSUPPORTED_SEMANTICS);
              EXPECT_EQ(
                  instruction(cpu | policy(guest), exec_mask, GOC_ALU_NEG_B, d, a, &b, &state),
                  GOC_ERROR_INVALID_FLAGS);
            }
            for (auto &reg : output)
              for (auto word : reg)
                EXPECT_EQ(word, 0x5a5a5a5aU);
            for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
              ASSERT_EQ(std::fesetround(rounding), 0);
              EXPECT_EQ(instruction(cpu | policy(guest), UINT64_MAX, 0, d, a, &b, &state),
                        GOC_SUCCESS);
              EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
              for (unsigned denorm : {0U, 0x40U, 0x8000U, 0x8040U}) {
                unsigned control = (_mm_getcsr() & ~0x8040U) | denorm;
                _mm_setcsr(control | 0x20U);
                EXPECT_EQ(instruction(cpu | policy(guest), UINT64_MAX, 0, d, a, &b, &state),
                          GOC_SUCCESS);
                EXPECT_EQ(_mm_getcsr() & ~0x3fU, control & ~0x3fU);
                EXPECT_NE(_mm_getcsr() & 0x20U, 0U);
              }
#endif
            }
            EXPECT_EQ(state, 0x80000055U);
          }
}

TEST(LdexpArchitecture, NaNsRemainNaNsUnlessClamped) {
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wave64 = 0; wave64 < 2; ++wave64)
      for (unsigned fp64 = 0; fp64 < 2; ++fp64)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (unsigned variant = 0; variant < 32; ++variant)
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
              uint32_t input[2][64], output[2][64], power[64] = {};
              const uint64_t inf = fp64 ? 0x7ff0000000000000ULL : 0x7f800000;
              const uint64_t sign = fp64 ? 0x8000000000000000ULL : 0x80000000;
              for (unsigned lane = 0; lane < 64; ++lane) {
                uint64_t bits = inf | (lane & 1 ? sign : 0) |
                                (lane & 2 ? (fp64 ? 0x8000000000000ULL : 0x400000) : 1);
                input[0][lane] = uint32_t(bits);
                input[1][lane] = uint32_t(bits >> 32);
              }
              const uint32_t *a[] = {input[0], input[1]}, *b = power;
              uint32_t *d[] = {output[0], output[1]};
              uint32_t mode = modifiers(variant);
              ASSERT_EQ(
                  instructions[arch][wave64][fp64](cpu | policy(guest), UINT64_MAX, mode, d, a, &b),
                  GOC_SUCCESS);
              for (unsigned lane = 0; lane < (wave64 ? 64U : 32U); ++lane) {
                uint64_t value = output[0][lane];
                if (fp64)
                  value |= uint64_t(output[1][lane]) << 32;
                if (mode & GOC_ALU_CLAMP)
                  EXPECT_EQ(value, 0U);
                else
                  EXPECT_GT(value & ~sign, inf);
              }
            }
}
