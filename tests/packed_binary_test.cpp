// SPDX-License-Identifier: MIT

#include "exec_masks.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "half_reference.h"
#include "test_instruction.h"

#include <algorithm>
#include <cfenv>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {

const decltype(&goc_v_pk_add_f16_rdna4_wave64) wave64[] = {
    goc_v_pk_add_f16_rdna4_wave64, goc_v_pk_mul_f16_rdna4_wave64, goc_v_pk_min_num_f16_wave64,
    goc_v_pk_max_num_f16_wave64,   goc_v_pk_minimum_f16_wave64,   goc_v_pk_maximum_f16_wave64,
    goc_v_pk_min_f16_wave64,       goc_v_pk_max_f16_wave64};

using Fn = decltype(&goc_v_pk_add_f16_rdna4);
const Fn functions[] = {goc_v_pk_add_f16_rdna4, goc_v_pk_mul_f16_rdna4, goc_v_pk_min_num_f16,
                        goc_v_pk_max_num_f16,   goc_v_pk_minimum_f16,   goc_v_pk_maximum_f16};
const uint32_t known = GOC_PK_NEG_LO_A | GOC_PK_NEG_LO_B | GOC_PK_NEG_HI_A | GOC_PK_NEG_HI_B |
                       GOC_PK_CLAMP | GOC_PK_LO_A_HIGH | GOC_PK_LO_B_HIGH | GOC_PK_HI_A_LOW |
                       GOC_PK_HI_B_LOW;
const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x3c00,
                           0x3c01, 0x3bff, 0x3800, 0xb800, 0x4000, 0xc000, 0xbc00,
                           0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe12};

uint32_t reference(int op, uint32_t a, uint32_t b, uint32_t mode, bool saturate,
                   uint64_t fp_flags = 0) {
  uint32_t result = 0;
  for (int half = 0; half < 2; ++half) {
    bool a_high = half ? !(mode & GOC_PK_HI_A_LOW) : bool(mode & GOC_PK_LO_A_HIGH);
    bool b_high = half ? !(mode & GOC_PK_HI_B_LOW) : bool(mode & GOC_PK_LO_B_HIGH);
    uint16_t ab = uint16_t(a >> (a_high ? 16 : 0)), bb = uint16_t(b >> (b_high ? 16 : 0));
    if (fp_flags & GOC_FP_FLUSH_INPUT_DENORMALS) {
      if ((ab & 0x7fff) < 0x400)
        ab &= 0x8000;
      if ((bb & 0x7fff) < 0x400)
        bb &= 0x8000;
    }
    double x = goc_test::half_value(ab);
    double y = goc_test::half_value(bb);
    if (mode & (half ? GOC_PK_NEG_HI_A : GOC_PK_NEG_LO_A))
      x = -x;
    if (mode & (half ? GOC_PK_NEG_HI_B : GOC_PK_NEG_LO_B))
      y = -y;
    double value = goc_test::half_binary(op == 0 ? 0 : op == 1 ? 3 : op + 2, x, y);
    if (mode & GOC_PK_CLAMP)
      value = !(value > 0) ? 0 : std::min(value, 1.0);
    uint16_t encoded = goc_test::half_bits(value, saturate);
    if (op < 2 && (fp_flags & GOC_FP_FLUSH_OUTPUT_DENORMALS) && (encoded & 0x7fff) < 0x400)
      encoded &= 0x8000;
    result |= uint32_t(encoded) << (16 * half);
  }
  return result;
}

void check(uint32_t got, uint32_t want) {
  for (int half = 0; half < 2; ++half) {
    uint16_t actual = uint16_t(got >> (16 * half)), expected = uint16_t(want >> (16 * half));
    if ((expected & 0x7fff) > 0x7c00) {
      EXPECT_GT(actual & 0x7fff, 0x7c00);
    } else {
      EXPECT_EQ(actual, expected);
    }
  }
}

void fill(uint32_t (&words)[3][34]) {
  for (int reg = 0; reg < 3; ++reg) {
    std::fill(words[reg], words[reg] + 34, 0xfacecafe);
    for (int lane = 1; lane <= 32; ++lane)
      words[reg][lane] =
          values[(lane + reg * 7) % 20] | (uint32_t(values[(lane * 3 + reg * 5) % 20]) << 16);
  }
}

void run(int op, uint64_t flags, uint32_t exec_mask, uint64_t mode, int b, int d,
         uint32_t (&words)[3][34]) {
  uint32_t before[3][34];
  std::memcpy(before, words, sizeof(before));
  uint32_t *p[] = {words[0] + 1, words[1] + 1, words[2] + 1};
  ASSERT_EQ(functions[op](flags, exec_mask, mode, p + d, p, p + b, nullptr), GOC_SUCCESS);
  for (int reg = 0; reg < 3; ++reg)
    for (int lane = 0; lane < 34; ++lane) {
      if (reg == d && lane >= 1 && lane <= 32 && ((exec_mask >> (lane - 1)) & 1)) {
        check(words[reg][lane],
              reference(op, before[0][lane], before[b][lane], mode, flags & GOC_FP16_OVFL));
      } else {
        EXPECT_EQ(words[reg][lane], before[reg][lane]);
      }
    }
}

} // namespace

TEST(PackedBinary, AllModifiersAndOverflowPolicies) {
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool saturate : {false, true})
        for (uint32_t mode = 0; mode < 8192; ++mode) {
          if (mode & ~known)
            continue;
          SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << saturate << '/' << mode);
          uint32_t words[3][34];
          fill(words);
          run(op, cpu | (saturate ? GOC_FP16_OVFL : 0), UINT32_MAX, mode, 1, mode % 3, words);
        }
}

TEST(PackedBinary, SpecialCartesianPairsAndEveryEncoding) {
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      std::mt19937 random(7231);
      for (unsigned base = 0; base < 512 + 65536; base += 32) {
        SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << base);
        uint32_t words[3][34];
        fill(words);
        for (int lane = 1; lane <= 32; ++lane) {
          unsigned i = base + lane - 1;
          if (base < 512) {
            words[0][lane] = values[i % 20] | (uint32_t(values[(i / 20) % 20]) << 16);
            words[1][lane] = values[(i / 20) % 20] | (uint32_t(values[i % 20]) << 16);
          } else {
            uint32_t code = uint16_t(i - 512);
            words[0][lane] = code | ((65535 - code) << 16);
            words[1][lane] = random();
          }
        }
        run(op, cpu, UINT32_MAX, 0, 1, 2, words);
      }
    }
}

TEST(PackedBinary, MasksAndAllWholeRegisterAliases) {
  const uint32_t modes[] = {0,
                            known,
                            GOC_PK_HI_A_LOW,
                            GOC_PK_HI_B_LOW,
                            GOC_PK_LO_A_HIGH | GOC_PK_HI_A_LOW,
                            GOC_PK_LO_B_HIGH | GOC_PK_HI_B_LOW,
                            GOC_PK_NEG_LO_B | GOC_PK_NEG_HI_A,
                            GOC_PK_LO_A_HIGH | GOC_PK_NEG_HI_B | GOC_PK_CLAMP};
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (auto mode : modes)
        for (auto exec_mask : exec_masks())
          for (int b : {0, 1})
            for (int d = 0; d < 3; ++d) {
              SCOPED_TRACE(::testing::Message() << op << '/' << cpu << '/' << mode << '/'
                                                << exec_mask << '/' << b << '/' << d);
              uint32_t words[3][34];
              fill(words);
              run(op, cpu, exec_mask, mode, b, d, words);
            }
}

TEST(PackedBinary, LiteralHalfSelectionSignedZeroAndNanRules) {
  struct Case {
    uint32_t a, b, mode, want[6];
  };

  const Case cases[] = {
      {0x44004000,
       0x3c004200,
       0,
       {0x45004500, 0x44004600, 0x3c004000, 0x44004200, 0x3c004000, 0x44004200}},
      {0x44004000,
       0x3c004200,
       GOC_PK_HI_A_LOW,
       {0x42004500, 0x40004600, 0x3c004000, 0x40004200, 0x3c004000, 0x40004200}},
      {0x44004000,
       0x3c004200,
       GOC_PK_HI_B_LOW,
       {0x47004500, 0x4a004600, 0x42004000, 0x44004200, 0x42004000, 0x44004200}},
      {0x00008000, 0x80000000, 0, {0, 0x80008000, 0x80008000, 0, 0x80008000, 0}},
      {0x3c007c01,
       0xfe124000,
       0,
       {0x7e007e00, 0x7e007e00, 0x3c004000, 0x3c004000, 0x7e007e00, 0x7e007e00}},
      {0x3c007c01, 0xfe124000, GOC_PK_CLAMP, {0, 0, 0x3c003c00, 0x3c003c00, 0, 0}}};
  for (int op = 0; op < 6; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (const auto &test : cases)
        for (int d = 0; d < 3; ++d) {
          uint32_t words[3][32];
          uint32_t *p[] = {words[0], words[1], words[2]};
          std::fill(words[0], words[0] + 32, test.a);
          std::fill(words[1], words[1] + 32, test.b);
          std::fill(words[2], words[2] + 32, 0xfacecafe);
          ASSERT_EQ(functions[op](cpu, UINT32_MAX, test.mode, p + d, p, p + 1, nullptr),
                    GOC_SUCCESS);
          for (auto word : words[d])
            check(word, test.want[op]);
        }
}

TEST(PackedBinary, ValidationAndSemantics) {
  for (auto fn : functions) {
    uint32_t words[32];
    std::fill(words, words + 32, 0xfacecafe);
    auto p = words;
    for (uint32_t exec_mask : {0U, UINT32_MAX}) {
      for (unsigned bit = 0; bit < 32; ++bit) {
        if ((1U << bit) & ~known) {
          EXPECT_EQ(fn(0, exec_mask, 1U << bit, &p, &p, &p, nullptr), GOC_ERROR_INVALID_FLAGS);
        }
      }
      EXPECT_EQ(fn(1ULL << 63, exec_mask, 0, &p, &p, &p, nullptr), GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, exec_mask, 0, &p, &p, &p,
                   nullptr),
                GOC_ERROR_UNSUPPORTED_SEMANTICS);
    }
    for (auto word : words)
      EXPECT_EQ(word, 0xfacecafe);
    EXPECT_EQ(fn(0, 0, 0, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(fn(0, 0U, 0, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT32_MAX, 0, &p, &p, &p, nullptr), GOC_SUCCESS);
  }
}

TEST(PackedBinary, Wave64ModifiersMasksAndAliases) {
  for (unsigned op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool saturate : {false, true})
        for (uint32_t mode = 0; mode < 8192; ++mode) {
          if (mode & ~known)
            continue;
          uint32_t words[3][66], before[3][66];
          for (unsigned reg = 0; reg < 3; ++reg)
            for (unsigned lane = 0; lane < 66; ++lane)
              words[reg][lane] = values[(lane + 7 * reg) % 20] |
                                 (uint32_t(values[(lane * 3 + 5 * reg) % 20]) << 16);
          std::memcpy(before, words, sizeof words);
          unsigned dest = mode % 3;
          uint32_t *d = words[dest] + 1;
          const uint32_t *a = words[0] + 1, *b = words[1] + 1;
          uint64_t exec_mask = 0x87654321fedcba98ULL;
          uint32_t exceptions = 0x80000055;
          ASSERT_EQ(wave64[op](cpu | (saturate ? GOC_FP16_OVFL : 0), exec_mask, mode, &d, &a, &b,
                               &exceptions),
                    GOC_SUCCESS);
          EXPECT_EQ(exceptions, 0x80000055U);
          for (unsigned lane = 0; lane < 64; ++lane) {
            if ((exec_mask >> lane) & 1)
              check(d[lane], reference(op >= 6 ? op - 4 : op, before[0][lane + 1],
                                       before[1][lane + 1], mode, saturate));
            else
              EXPECT_EQ(d[lane], before[dest][lane + 1]);
          }
          for (unsigned reg = 0; reg < 3; ++reg) {
            EXPECT_EQ(words[reg][0], before[reg][0]);
            EXPECT_EQ(words[reg][65], before[reg][65]);
          }
        }
}

TEST(PackedBinary, Wave64ValidationAndOptionalState) {
  for (auto fn : wave64) {
    EXPECT_EQ(fn(0, 0, 0, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
    EXPECT_EQ(fn(0, UINT64_MAX, 1ULL << 36, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT64_MAX, 0, nullptr,
                 nullptr, nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    uint32_t exceptions = 0x80000055;
    EXPECT_EQ(
        fn(GOC_SEMANTICS_EXACT_EMPIRICAL, UINT64_MAX, 0, nullptr, nullptr, nullptr, &exceptions),
        GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
    EXPECT_EQ(exceptions, 0x80000055U);
  }
}

namespace {
using Arithmetic =
    goc_test::WaveInstruction<int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *,
                                      const uint32_t *const *, const uint32_t *const *)>;
const Arithmetic architecture[2][2][2] = {
    {{goc_v_pk_add_f16, goc_v_pk_mul_f16}, {goc_v_pk_add_f16_wave64, goc_v_pk_mul_f16_wave64}},
    {{goc_v_pk_add_f16_rdna4, goc_v_pk_mul_f16_rdna4},
     {goc_v_pk_add_f16_rdna4_wave64, goc_v_pk_mul_f16_rdna4_wave64}}};

uint64_t guest_policy(unsigned guest) {
  return (guest & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
         (guest & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) | (guest & 4 ? GOC_FP_IEEE_MODE : 0);
}
} // namespace

TEST(PackedBinary, ArchitecturePoliciesModifiersAndAliases) {
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (unsigned op = 0; op < 2; ++op)
        for (unsigned guest = 0; guest < 8; ++guest)
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
            for (uint32_t mode = 0; mode < 8192; ++mode) {
              if (mode & ~known)
                continue;
              uint32_t words[3][66], before[3][66];
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned lane = 0; lane < 66; ++lane)
                  words[reg][lane] = values[(lane + reg * 7) % 20] |
                                     (uint32_t(values[(lane * 3 + reg * 5) % 20]) << 16);
              std::memcpy(before, words, sizeof(words));
              unsigned target = mode % 3;
              uint64_t exec_mask = mode & 1 ? 0x8181818155555555ULL : UINT64_MAX;
              uint64_t flags = cpu | guest_policy(guest) | (mode & 2 ? GOC_FP16_OVFL : 0);
              uint32_t state = 0x12345678;
              const uint32_t *a[] = {words[0] + 1}, *b[] = {words[1] + 1};
              uint32_t *d[] = {words[target] + 1};
              ASSERT_EQ(architecture[arch][wide][op](flags, exec_mask, mode, d, a, b, &state),
                        GOC_SUCCESS);
              ASSERT_EQ(state, 0x12345678U);
              for (unsigned reg = 0; reg < 3; ++reg)
                for (unsigned lane = 0; lane < 66; ++lane) {
                  if (reg == target && lane >= 1 && lane <= (wide ? 64u : 32u) &&
                      ((exec_mask >> (lane - 1)) & 1))
                    check(words[reg][lane], reference(op, before[0][lane], before[1][lane], mode,
                                                      flags & GOC_FP16_OVFL, flags));
                  else
                    ASSERT_EQ(words[reg][lane], before[reg][lane]);
                }
            }
}

TEST(PackedBinary, LiteralGuestDenormalWitnesses) {
  for (unsigned arch = 0; arch < 2; ++arch)
    for (unsigned wide = 0; wide < 2; ++wide)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (unsigned guest = 0; guest < 8; ++guest) {
          uint32_t a[64], b[64], result[64];
          const uint32_t *ap[] = {a}, *bp[] = {b};
          uint32_t *d[] = {result};
          // Low half isolates input flushing; high half produces a subnormal from normal inputs.
          std::fill_n(a, 64, 0x04010001U);
          std::fill_n(b, 64, 0x84000000U);
          ASSERT_EQ(
              architecture[arch][wide][0](cpu | guest_policy(guest), UINT64_MAX, 0, d, ap, bp),
              GOC_SUCCESS);
          const uint32_t add_expected = (guest & 3 ? 0U : 1U) | (guest & 2 ? 0U : 0x10000U);
          for (unsigned lane = 0; lane < (wide ? 64u : 32u); ++lane)
            ASSERT_EQ(result[lane], add_expected);
          std::fill_n(a, 64, 0x04000400U);
          std::fill_n(b, 64, 0xb8003800U);
          ASSERT_EQ(
              architecture[arch][wide][1](cpu | guest_policy(guest), UINT64_MAX, 0, d, ap, bp),
              GOC_SUCCESS);
          for (unsigned lane = 0; lane < (wide ? 64u : 32u); ++lane)
            ASSERT_EQ(result[lane], guest & 2 ? 0x80000000U : 0x82000200U);
        }
}

TEST(PackedBinary, ArchitectureErrorsAndHostControls) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int round : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(round), 0);
#if defined(__x86_64__) || defined(_M_X64)
    _mm_setcsr(_mm_getcsr() | 0x8040U);
    const unsigned controls = _mm_getcsr() & ~0x3fU;
#endif
    for (unsigned arch = 0; arch < 2; ++arch)
      for (unsigned wide = 0; wide < 2; ++wide)
        for (unsigned op = 0; op < 2; ++op)
          for (unsigned guest = 0; guest < 8; ++guest) {
            uint32_t a[64], b[64], result[64], state = 0x12345678;
            std::fill_n(a, 64, 0x00010001U);
            std::fill_n(b, 64, 0x3c003c00U);
            const uint32_t *ap[] = {a}, *bp[] = {b};
            uint32_t *d[] = {result};
            for (uint64_t strict : {uint64_t(0), GOC_SEMANTICS_STRICT})
              for (uint64_t exec_mask : {uint64_t(0), UINT64_MAX}) {
                std::fill_n(result, 64, 0xdeadbeef);
                const auto fn = architecture[arch][wide][op];
                uint64_t flags = guest_policy(guest) | GOC_SEMANTICS_EXACT_EMPIRICAL | strict;
                EXPECT_EQ(fn(flags, exec_mask, 0, d, nullptr, nullptr, &state),
                          GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
                EXPECT_EQ(state, 0x12345678U);
                for (auto word : result)
                  EXPECT_EQ(word, 0xdeadbeefU);
                if (strict) {
                  EXPECT_EQ(fn(flags, exec_mask, 0, d, nullptr, nullptr),
                            GOC_ERROR_UNSUPPORTED_SEMANTICS);
                }
              }
            for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
              ASSERT_EQ(architecture[arch][wide][op](cpu | guest_policy(guest), UINT64_MAX, 0, d,
                                                     ap, bp, &state),
                        GOC_SUCCESS);
              EXPECT_EQ(state, 0x12345678U);
              EXPECT_EQ(std::fegetround(), round);
#if defined(__x86_64__) || defined(_M_X64)
              EXPECT_EQ(_mm_getcsr() & ~0x3fU, controls);
#endif
            }
          }
  }
}

TEST(PackedBinary, SelectionGuestPoliciesAllModifiersAndAliases) {
  const Fn narrow[] = {goc_v_pk_min_f16,     goc_v_pk_max_f16,     goc_v_pk_min_num_f16,
                       goc_v_pk_max_num_f16, goc_v_pk_minimum_f16, goc_v_pk_maximum_f16};
  const decltype(&goc_v_pk_min_f16_wave64) wide[] = {
      goc_v_pk_min_f16_wave64,     goc_v_pk_max_f16_wave64,     goc_v_pk_min_num_f16_wave64,
      goc_v_pk_max_num_f16_wave64, goc_v_pk_minimum_f16_wave64, goc_v_pk_maximum_f16_wave64};
  // Select raw half encodings: keep payloads and signaling bits out of host arithmetic.
  auto select = [](uint16_t a, uint16_t b, unsigned op, unsigned guest) {
    if (guest & 1) {
      if ((a & 0x7fff) < 0x400)
        a &= 0x8000;
      if ((b & 0x7fff) < 0x400)
        b &= 0x8000;
    }
    bool an = (a & 0x7fff) > 0x7c00, bn = (b & 0x7fff) > 0x7c00;
    if (op >= 4 || (op < 2 && (guest & 4))) {
      if (an && !(a & 0x200))
        return uint16_t(a | 0x200);
      if (bn && !(b & 0x200))
        return uint16_t(b | 0x200);
    }
    if (an || bn) {
      if (op >= 4)
        return an ? a : b;
      if (an && bn)
        return op < 2 ? a : uint16_t(a | 0x200);
      return an ? b : a;
    }
    auto order = [](uint16_t bits) {
      return bits & 0x8000 ? uint16_t(~bits) : uint16_t(bits ^ 0x8000);
    };
    return ((op & 1) ? order(a) > order(b) : order(a) < order(b)) ? a : b;
  };
  for (unsigned op = 0; op < 6; ++op)
    for (unsigned guest = 0; guest < 8; ++guest)
      for (unsigned variant = 0; variant < 512; ++variant) {
        // Nine independent bits: four negations, clamp, and four half selectors.
        const uint32_t bits[] = {GOC_PK_NEG_LO_A,  GOC_PK_NEG_LO_B, GOC_PK_NEG_HI_A,
                                 GOC_PK_NEG_HI_B,  GOC_PK_CLAMP,    GOC_PK_LO_A_HIGH,
                                 GOC_PK_LO_B_HIGH, GOC_PK_HI_A_LOW, GOC_PK_HI_B_LOW};
        uint32_t mode = 0;
        for (unsigned bit = 0; bit < 9; ++bit)
          if ((variant >> bit) & 1)
            mode |= bits[bit];
        uint64_t flags = (guest & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                         (guest & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                         (guest & 4 ? GOC_FP_IEEE_MODE : 0);
        for (unsigned width : {32U, 64U}) {
          uint32_t before[3][66], expected[3][66];
          for (unsigned reg = 0; reg < 3; ++reg)
            for (unsigned lane = 0; lane < 66; ++lane)
              before[reg][lane] = values[(lane + 7 * reg + variant) % 20] |
                                  (uint32_t(values[(3 * lane + 5 * reg + variant) % 20]) << 16);
          std::memcpy(expected, before, sizeof expected);
          unsigned dest = variant % 3;
          uint64_t exec_mask = variant & 1 ? 0x87654321fedcba98ULL : UINT64_MAX;
          if (width == 32)
            exec_mask = uint32_t(exec_mask);
          for (unsigned lane = 0; lane < width; ++lane)
            if ((exec_mask >> lane) & 1) {
              uint32_t result = 0;
              for (unsigned half = 0; half < 2; ++half) {
                bool ah = half ? !(mode & GOC_PK_HI_A_LOW) : bool(mode & GOC_PK_LO_A_HIGH);
                bool bh = half ? !(mode & GOC_PK_HI_B_LOW) : bool(mode & GOC_PK_LO_B_HIGH);
                uint16_t a = uint16_t(before[0][lane + 1] >> (ah ? 16 : 0));
                uint16_t b = uint16_t(before[1][lane + 1] >> (bh ? 16 : 0));
                if (mode & (half ? GOC_PK_NEG_HI_A : GOC_PK_NEG_LO_A))
                  a ^= 0x8000;
                if (mode & (half ? GOC_PK_NEG_HI_B : GOC_PK_NEG_LO_B))
                  b ^= 0x8000;
                uint16_t chosen = select(a, b, op, guest);
                if (mode & GOC_PK_CLAMP) {
                  if ((chosen & 0x8000) || (chosen & 0x7fff) > 0x7c00)
                    chosen = 0;
                  else if (chosen > 0x3c00)
                    chosen = 0x3c00;
                }
                result |= uint32_t(chosen) << (16 * half);
              }
              expected[dest][lane + 1] = result;
            }
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
            uint32_t words[3][66], state = 0x12345678;
            std::memcpy(words, before, sizeof words);
            uint32_t *d = words[dest] + 1;
            const uint32_t *a = words[0] + 1, *b = words[1] + 1;
            ASSERT_EQ(width == 32
                          ? narrow[op](flags | cpu, uint32_t(exec_mask), mode, &d, &a, &b, &state)
                          : wide[op](flags | cpu, exec_mask, mode, &d, &a, &b, &state),
                      GOC_SUCCESS);
            ASSERT_EQ(state, 0x12345678U);
            for (unsigned reg = 0; reg < 3; ++reg)
              for (unsigned lane = 0; lane < 66; ++lane)
                ASSERT_EQ(words[reg][lane], expected[reg][lane])
                    << op << '/' << guest << '/' << variant << '/' << width << '/' << cpu << '/'
                    << reg << '/' << lane;
          }
        }
      }
}
