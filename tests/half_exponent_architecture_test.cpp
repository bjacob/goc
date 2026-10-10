// SPDX-License-Identifier: MIT

#include "dpp_reference.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "half_exponent_reference.h"
#include "test_instruction.h"

#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

using Binary = goc_test::WaveInstruction<decltype(&goc_v_min_u32)>;
using Unary = goc_test::WaveInstruction<decltype(&goc_v_cvt_f64_i32)>;
const Binary ldexp_functions[2][2] = {{goc_v_ldexp_f16, goc_v_ldexp_f16_wave64},
                                      {goc_v_ldexp_f16_rdna4, goc_v_ldexp_f16_rdna4_wave64}};
const Unary frexp_functions[2][2] = {
    {goc_v_frexp_exp_i16_f16, goc_v_frexp_exp_i16_f16_wave64},
    {goc_v_frexp_exp_i16_f16_rdna4, goc_v_frexp_exp_i16_f16_rdna4_wave64}};

int invoke(unsigned arch, unsigned wide, unsigned op, uint64_t flags, uint64_t exec_mask,
           uint64_t mode, uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
           uint32_t *state = nullptr) {
  return op ? frexp_functions[arch][wide](flags, exec_mask, mode, d, a, state)
            : ldexp_functions[arch][wide](flags, exec_mask, mode, d, a, b, state);
}

uint64_t policy(unsigned guest) {
  return (guest & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (guest & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (guest & 4 ? GOC_FP_IEEE_MODE : 0) |
         (guest & 8 ? GOC_FP16_OVFL : 0);
}

uint32_t modifiers(unsigned variant) {
  return (variant & 1 ? GOC_ALU_NEG_A : 0) | (variant & 2 ? GOC_ALU_ABS_A : 0) |
         ((variant & 28) << 4) | (variant & 32 ? GOC_ALU_HIGH_A : 0) |
         (variant & 64 ? GOC_ALU_HIGH_D : 0) | (variant & 128 ? GOC_ALU_HIGH_B : 0);
}

uint16_t expected(unsigned arch, unsigned op, unsigned guest, uint32_t a, uint32_t b,
                  uint32_t mode) {
  if (!op) {
    if (!arch && (!(guest & 2) || (guest & 4)))
      mode &= ~GOC_ALU_OMOD_HALF;
    unsigned shift = mode & GOC_ALU_HIGH_A ? 16 : 0;
    uint16_t bits = uint16_t(a >> shift);
    if ((guest & 1) && (bits & 0x7fff) < 0x400) {
      bits &= 0x8000;
      a = (a & ~(0xffffU << shift)) | (uint32_t(bits) << shift);
    }
  }
  uint16_t result = goc_test::half_exponent_reference(op, a, b, mode, guest & 8);
  if (!op && (guest & 2) && (result & 0x7fff) < 0x400)
    result &= 0x8000;
  return result;
}

bool equal(unsigned op, uint32_t actual, uint32_t expected, uint32_t mode) {
  if (op)
    return actual == expected;
  unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  const uint32_t untouched = ~(0xffffU << shift);
  if ((actual & untouched) != (expected & untouched))
    return false;
  uint16_t want = uint16_t(expected >> shift), got = uint16_t(actual >> shift);
  return (want & 0x7fff) > 0x7c00 ? (got & 0x7fff) > 0x7c00 : want == got;
}

} // namespace

TEST(HalfExponentArchitecture, LiteralPolicies) {
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned guest = 0; guest < 16; ++guest)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          uint32_t a[64], b[64], d[64];
          for (unsigned lane = 0; lane < 64; ++lane) {
            a[lane] = lane % 3 == 0 ? 0x4200 : lane % 3 == 1 ? 1 : 0x400;
            b[lane] = lane % 3 == 0 ? 0 : lane % 3 == 1 ? 10 : uint16_t(-2);
            d[lane] = 0xdeadbeef;
          }
          const uint32_t *pa = a, *pb = b;
          uint32_t *pd = d, state = 0x80000055;
          ASSERT_EQ(ldexp_functions[arch][wide](cpu | policy(guest), UINT64_MAX, GOC_ALU_OMOD_2,
                                                &pd, &pa, &pb, &state),
                    GOC_SUCCESS);
          bool scale = arch || ((guest & 2) && !(guest & 4));
          for (unsigned lane = 0; lane < (wide ? 64U : 32U); ++lane) {
            uint32_t want = lane % 3 == 0          ? (scale ? 0x4600 : 0x4200)
                            : lane % 3 == 1        ? (guest & 1 ? 0
                                                      : scale   ? 0x800
                                                                : 0x400)
                            : scale || (guest & 2) ? 0
                                                   : 0x100;
            EXPECT_EQ(d[lane], 0xdead0000U | want);
          }
          EXPECT_EQ(state, 0x80000055U);
        }
}

TEST(HalfExponentArchitecture, ModifiersGuestModesDppAndAliases) {
  const uint16_t values[] = {0,      0x8000, 1,      0x8003, 0x3ff,  0x400,  0x401,  0x7ff,
                             0x3c01, 0xbc01, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe00};
  const int powers[] = {-32768, -25, -15, -1, 0, 1, 16, 32767};
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned op = 0; op < 2; ++op)
        for (unsigned guest = 0; guest < 16; ++guest)
          for (unsigned variant = 0; variant < (op ? 128U : 256U); ++variant)
            for (unsigned alias = 0; alias < 3; ++alias)
              for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
                uint32_t words[3][66], before[3][66];
                for (unsigned lane = 0; lane < 66; ++lane) {
                  words[0][lane] = values[lane % 16] | (uint32_t(values[(lane + 3) % 16]) << 16);
                  words[1][lane] = uint16_t(powers[lane % 8]) |
                                   (uint32_t(uint16_t(powers[(lane + 2) % 8])) << 16);
                  words[2][lane] = 0xdeadbeef;
                }
                std::memcpy(before, words, sizeof(before));
                uint32_t mode = modifiers(variant);
                // Cycle all DPP descriptors and an ordinary form through the
                // modifier/alias matrix. Routing is checked against the independent reference.
                uint64_t descriptor = variant % 8 ? goc_test::dpp_modes[variant % 8 - 1] : 0;
                uint64_t exec_mask = 0xf13579bde2468acfULL;
                const uint32_t *a = words[0] + 1, *b = words[1] + 1;
                uint32_t *d = words[alias] + 1;
                ASSERT_EQ(invoke(arch, wide, op, cpu | policy(guest), exec_mask, descriptor | mode,
                                 &d, &a, &b),
                          GOC_SUCCESS);
                for (unsigned lane = 0; lane < 64; ++lane) {
                  int source = int(lane);
                  bool active = lane < (wide ? 64U : 32U) && ((exec_mask >> lane) & 1);
                  if (active && descriptor)
                    active =
                        wide ? goc_test::dpp_source(descriptor, exec_mask, lane, source)
                             : goc_test::dpp_source(descriptor, uint32_t(exec_mask), lane, source);
                  uint32_t want = before[alias][lane + 1];
                  if (active) {
                    unsigned shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
                    uint16_t result =
                        expected(arch, op, guest, source < 0 ? 0 : before[0][source + 1],
                                 before[1][lane + 1], mode);
                    want = (want & ~(0xffffU << shift)) | (uint32_t(result) << shift);
                  }
                  ASSERT_TRUE(equal(op, words[alias][lane + 1], want, mode))
                      << arch << '/' << wide << '/' << op << '/' << guest << '/' << variant << '/'
                      << alias << '/' << cpu << '/' << lane;
                }
                for (unsigned reg = 0; reg < 3; ++reg) {
                  EXPECT_EQ(words[reg][0], before[reg][0]);
                  EXPECT_EQ(words[reg][65], before[reg][65]);
                  if (reg != alias) {
                    EXPECT_EQ(std::memcmp(words[reg], before[reg], sizeof(words[reg])), 0);
                  }
                }
              }
}

TEST(HalfExponentArchitecture, EveryHalfUnderGuestPolicies) {
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned op = 0; op < 2; ++op)
      for (unsigned guest = 0; guest < 8; ++guest)
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
          for (unsigned base = 0; base < 65536; base += 64) {
            uint32_t a[64], b[64], d[64];
            for (unsigned lane = 0; lane < 64; ++lane) {
              a[lane] = (base + lane) << 16;
              b[lane] = uint16_t(int(lane % 31) - 15);
              d[lane] = 0xdeadbeef;
            }
            const uint32_t *pa = a, *pb = b;
            uint32_t *pd = d;
            const uint32_t mode = GOC_ALU_HIGH_A | GOC_ALU_HIGH_D | GOC_ALU_OMOD_HALF;
            ASSERT_EQ(invoke(arch, 1, op, cpu | policy(guest), UINT64_MAX, mode, &pd, &pa, &pb),
                      GOC_SUCCESS);
            for (unsigned lane = 0; lane < 64; ++lane) {
              uint32_t want =
                  0xbeef | (uint32_t(expected(arch, op, guest, a[lane], b[lane], mode)) << 16);
              ASSERT_TRUE(equal(op, d[lane], want, mode))
                  << arch << '/' << op << '/' << guest << '/' << base + lane;
            }
          }
}

TEST(HalfExponentArchitecture, ValidationAndHostControls) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned op = 0; op < 2; ++op)
        for (unsigned guest = 0; guest < 16; ++guest)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t a[64] = {}, b[64] = {}, d[64], state = 0x80000055;
            for (auto &word : d)
              word = 0xdeadbeef;
            const uint32_t *pa = a, *pb = b;
            uint32_t *pd = d;
            for (uint64_t exec_mask : {0ULL, ~0ULL}) {
              EXPECT_EQ(invoke(arch, wide, op, cpu | policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL,
                               exec_mask, 0, &pd, &pa, &pb, &state),
                        GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
              EXPECT_EQ(
                  invoke(arch, wide, op,
                         cpu | policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                         exec_mask, 0, &pd, &pa, &pb),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
              EXPECT_EQ(invoke(arch, wide, op, cpu | policy(guest), exec_mask, GOC_ALU_NEG_B, &pd,
                               &pa, &pb),
                        GOC_ERROR_INVALID_FLAGS);
            }
            for (auto word : d)
              EXPECT_EQ(word, 0xdeadbeefU);
            for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
              ASSERT_EQ(std::fesetround(rounding), 0);
              EXPECT_EQ(
                  invoke(arch, wide, op, cpu | policy(guest), UINT64_MAX, 0, &pd, &pa, &pb, &state),
                  GOC_SUCCESS);
              EXPECT_EQ(std::fegetround(), rounding);
#if defined(__x86_64__) || defined(_M_X64)
              for (unsigned denorm : {0U, 0x40U, 0x8000U, 0x8040U}) {
                unsigned control = (_mm_getcsr() & ~0x8040U) | denorm;
                _mm_setcsr(control | 0x20U);
                EXPECT_EQ(invoke(arch, wide, op, cpu | policy(guest), UINT64_MAX, 0, &pd, &pa, &pb,
                                 &state),
                          GOC_SUCCESS);
                EXPECT_EQ(_mm_getcsr() & ~0x3fU, control & ~0x3fU);
                EXPECT_NE(_mm_getcsr() & 0x20U, 0U);
              }
#endif
            }
            EXPECT_EQ(state, 0x80000055U);
          }
}
