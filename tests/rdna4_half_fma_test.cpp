// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"

#include <algorithm>
#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {

const uint64_t semantics[] = {GOC_SEMANTICS_LOOSE,
                              GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT};
const uint16_t values[] = {0,      0x8000, 1,      0x8001, 0x3ff,  0x400,  0x401,  0x7ff,
                           0x800,  0x3800, 0xb800, 0x3bff, 0x3c00, 0xbc00, 0x3c01, 0x4200,
                           0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfc12, 0x7fc1, 0xff80};

bool nan(uint16_t bits) { return (bits & 0x7fff) > 0x7c00; }

// Independent exact integer oracle: magnitude * 2^exponent. All finite half
// products and aligned half addends fit in uint64_t, including their sum.
struct Number {
  uint64_t magnitude;
  int exponent;
  bool negative;
};

Number decode(uint16_t bits) {
  int e = (bits >> 10) & 31;
  return {uint64_t((bits & 1023) + (e ? 1024 : 0)), (e ? e : 1) - 25, bool(bits & 0x8000)};
}

uint64_t rounded_shift(uint64_t bits, int shift) {
  if (shift <= 0)
    return bits << -shift;
  uint64_t tail = bits & ((UINT64_C(1) << shift) - 1);
  uint64_t midpoint = UINT64_C(1) << (shift - 1);
  return (bits >> shift) + (tail > midpoint || (tail == midpoint && ((bits >> shift) & 1)));
}

uint16_t pack(Number value, bool saturate) {
  uint16_t sign = value.negative ? 0x8000 : 0;
  if (!value.magnitude)
    return sign;
  int top = 0;
  for (uint64_t bits = value.magnitude; bits >>= 1;)
    ++top;
  int exponent = top + value.exponent;
  if (exponent < -14)
    return sign | uint16_t(rounded_shift(value.magnitude, -24 - value.exponent));
  uint64_t sig = rounded_shift(value.magnitude, top - 10);
  if (sig == 2048) {
    sig = 1024;
    ++exponent;
  }
  if (exponent > 15)
    return sign | (saturate ? 0x7bff : 0x7c00);
  return sign | uint16_t(((exponent + 15) << 10) + sig - 1024);
}

uint16_t reference(uint32_t a, uint32_t b, uint32_t c, uint32_t mode, bool saturate) {
  uint32_t words[] = {a, b, c};
  uint16_t h[3];
  for (int i = 0; i < 3; ++i) {
    h[i] = uint16_t(words[i] >> (mode & (GOC_ALU_HIGH_A << i) ? 16 : 0));
    if (mode & (GOC_ALU_ABS_A << i))
      h[i] &= 0x7fff;
    if (mode & (GOC_ALU_NEG_A << i))
      h[i] ^= 0x8000;
  }
  unsigned ma = h[0] & 0x7fff, mb = h[1] & 0x7fff, mc = h[2] & 0x7fff;
  uint16_t result;
  if ((!ma && mb == 0x7c00) || (!mb && ma == 0x7c00))
    result = 0xfe00;
  else if (nan(h[0]) || nan(h[1]) || nan(h[2]))
    result = (nan(h[0]) ? h[0] : nan(h[1]) ? h[1] : h[2]) | 0x200;
  else if (ma == 0x7c00 || mb == 0x7c00) {
    result = ((h[0] ^ h[1]) & 0x8000) | 0x7c00;
    if (mc == 0x7c00 && ((result ^ h[2]) & 0x8000))
      result = 0xfe00;
  } else if (mc == 0x7c00)
    result = h[2];
  else {
    auto x = decode(h[0]), y = decode(h[1]), z = decode(h[2]);
    Number p = {x.magnitude * y.magnitude, x.exponent + y.exponent, x.negative != y.negative};
    int grid = std::min(p.exponent, z.exponent);
    uint64_t pm = p.magnitude << (p.exponent - grid), zm = z.magnitude << (z.exponent - grid);
    Number sum = {0, grid, false};
    if (p.negative == z.negative) {
      sum.magnitude = pm + zm;
      sum.negative = p.negative;
    } else {
      sum.magnitude = pm > zm ? pm - zm : zm - pm;
      sum.negative = pm > zm ? p.negative : zm > pm ? z.negative : false;
    }
    result = pack(sum, saturate);
    unsigned omod = (mode >> 6) & 3;
    if (omod) {
      // Active OMOD flushes before packing at 2^-14 - 2^-26.
      bool tiny = grid < -26 ? sum.magnitude < (UINT64_C(4095) << (-26 - grid))
                             : (sum.magnitude << (grid + 26)) < 4095;
      if (tiny)
        result = 0;
      else if (omod == 3 && (result & 0x7fff) < 0x0800)
        result &= 0x8000;
      else if ((result & 0x7fff) < 0x7c00) {
        auto scaled = decode(result);
        scaled.exponent += omod == 3 ? -1 : int(omod);
        result = pack(scaled, saturate);
      }
    }
  }
  if (mode & GOC_ALU_CLAMP)
    result = nan(result) || (result & 0x8000) ? 0 : std::min<uint16_t>(result, 0x3c00);
  return result;
}

void check(uint32_t actual, uint32_t before, uint16_t want, uint32_t mode, bool exact) {
  int shift = mode & GOC_ALU_HIGH_D ? 16 : 0;
  EXPECT_EQ((actual ^ before) & ~(UINT32_C(0xffff) << shift), 0u);
  uint16_t got = uint16_t(actual >> shift);
  if (!exact && nan(want)) {
    EXPECT_TRUE(nan(got));
  } else {
    EXPECT_EQ(got, want);
  }
}

void run(uint64_t flags, uint32_t mode, uint32_t (&words)[4][32]) {
  uint32_t before[4][32];
  std::memcpy(before, words, sizeof(before));
  uint32_t *p[] = {words[0], words[1], words[2], words[3]};
  ASSERT_EQ(goc_rdna4_v_fma_f16(flags, UINT32_MAX, mode, p + 3, p, p + 1, p + 2), GOC_SUCCESS);
  for (int lane = 0; lane < 32; ++lane) {
    SCOPED_TRACE(lane);
    check(words[3][lane], before[3][lane],
          reference(before[0][lane], before[1][lane], before[2][lane], mode, flags & GOC_FP16_OVFL),
          mode, flags & GOC_SEMANTICS_MASK);
  }
}

} // namespace

TEST(HalfFma, BoundaryCartesianProductsEveryEncodingAndRandomTriples) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (auto sem : semantics) {
      std::mt19937 random(84712);
      for (unsigned base = 0; base < 13824 + 262144; base += 32) {
        uint32_t words[4][32];
        uint32_t mode = ((base / 32) % 4) << 6;
        uint64_t flags = cpu | sem | ((base & 32) ? GOC_FP16_OVFL : 0);
        SCOPED_TRACE(::testing::Message() << cpu << '/' << sem << '/' << base << '/' << mode);
        for (int lane = 0; lane < 32; ++lane) {
          unsigned i = base + lane;
          for (int reg = 0; reg < 3; ++reg) {
            uint16_t code = base < 13824                       ? values[i % 24]
                            : reg == 0 && base < 13824 + 65536 ? uint16_t(i - 13824)
                                                               : uint16_t(random());
            if (base < 13824)
              i /= 24;
            words[reg][lane] = 0xdead0000 | code;
          }
          words[3][lane] = 0xfacecafe;
        }
        run(flags, mode, words);
      }
    }
}

TEST(HalfFma, AllModifiersAndHalfSelectors) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (auto sem : semantics)
      for (uint32_t mode = 0; mode < 8192; ++mode)
        for (bool saturate : {false, true}) {
          SCOPED_TRACE(::testing::Message() << cpu << '/' << sem << '/' << mode << '/' << saturate);
          uint32_t words[4][32];
          for (int reg = 0; reg < 4; ++reg)
            for (int lane = 0; lane < 32; ++lane)
              words[reg][lane] = values[(lane + reg * 5) % 24] |
                                 (uint32_t(values[(lane * 7 + reg * 3) % 24]) << 16);
          run(cpu | sem | (saturate ? GOC_FP16_OVFL : 0), mode, words);
        }
}

TEST(HalfFma, MasksAndAllWholeRegisterAliases) {
  const int layouts[][3] = {{0, 1, 2}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 0, 0}};
  const uint32_t modes[] = {
      0, 8191, GOC_ALU_HIGH_A | GOC_ALU_HIGH_C | GOC_ALU_HIGH_D | GOC_ALU_NEG_A | GOC_ALU_OMOD_HALF,
      GOC_ALU_HIGH_B | GOC_ALU_ABS_C | GOC_ALU_OMOD_4};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (auto sem : semantics)
      for (auto mode : modes)
        for (auto mask : rdna4_exec_masks())
          for (const auto &layout : layouts)
            for (int dest = 0; dest < 4; ++dest) {
              SCOPED_TRACE(::testing::Message()
                           << cpu << '/' << sem << '/' << mode << '/' << mask << '/' << dest << '/'
                           << layout[0] << layout[1] << layout[2]);
              uint32_t words[4][34], before[4][34];
              for (int reg = 0; reg < 4; ++reg) {
                std::fill(words[reg], words[reg] + 34, 0xdeadbeef);
                for (int lane = 1; lane <= 32; ++lane)
                  words[reg][lane] = values[(lane + reg * 5) % 24] |
                                     (uint32_t(values[(lane * 7 + reg * 3) % 24]) << 16);
              }
              std::memcpy(before, words, sizeof(before));
              uint32_t *p[] = {words[0] + 1, words[1] + 1, words[2] + 1, words[3] + 1};
              ASSERT_EQ(goc_rdna4_v_fma_f16(cpu | sem, mask, mode, p + dest, p + layout[0],
                                            p + layout[1], p + layout[2]),
                        GOC_SUCCESS);
              for (int reg = 0; reg < 4; ++reg)
                for (int lane = 0; lane < 34; ++lane) {
                  if (reg == dest && lane >= 1 && lane <= 32 && ((mask >> (lane - 1)) & 1)) {
                    check(words[reg][lane], before[reg][lane],
                          reference(before[layout[0]][lane], before[layout[1]][lane],
                                    before[layout[2]][lane], mode, false),
                          mode, sem != 0);
                  } else {
                    EXPECT_EQ(words[reg][lane], before[reg][lane]);
                  }
                }
            }
}

TEST(HalfFma, RocjitsuHardwareWitnesses) {
  // From rocjitsu tests/valu_fp_mode_test.cpp's gfx1201 FMA captures.
  struct Case {
    uint16_t a, b, c, want;
    uint32_t mode;
    bool saturate;
  };

  const Case cases[] = {{0x7fc1, 0xff80, 0xff80, 0x7fc1, 0, false},
                        {0x7c01, 0x3c00, 0, 0x7e01, 0, false},
                        {0x3c00, 0xfc12, 0x7e01, 0xfe12, 0, false},
                        {0, 0x7c00, 0x7e01, 0xfe00, 0, false},
                        {0x7c00, 0x3c00, 0xfc00, 0xfe00, 0, false},
                        {0x400, 0xbc00, 0, 0x8000, GOC_ALU_OMOD_HALF, false},
                        {0x3c00, 0x7bff, 0x7bff, 0x7c00, GOC_ALU_OMOD_HALF, false},
                        {0x3c00, 0x7bff, 0x7bff, 0x77ff, GOC_ALU_OMOD_HALF, true},
                        {0x400, 0x3800, 0, 0, GOC_ALU_OMOD_4, false},
                        {0x400, 0xb800, 0, 0, GOC_ALU_OMOD_4, false}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (auto sem : semantics)
      for (auto test : cases)
        for (bool clamp : {false, true}) {
          uint32_t words[4][32];
          uint32_t *p[] = {words[0], words[1], words[2], words[3]};
          std::fill(words[0], words[0] + 32, test.a);
          std::fill(words[1], words[1] + 32, test.b);
          std::fill(words[2], words[2] + 32, test.c);
          std::fill(words[3], words[3] + 32, 0xdeadbeef);
          uint32_t mode = test.mode | (clamp ? GOC_ALU_CLAMP : 0);
          uint16_t want = test.want;
          if (clamp)
            want = nan(want) || (want & 0x8000) ? 0 : std::min<uint16_t>(want, 0x3c00);
          ASSERT_EQ(goc_rdna4_v_fma_f16(cpu | sem | (test.saturate ? GOC_FP16_OVFL : 0), UINT32_MAX,
                                        mode, p + 3, p, p + 1, p + 2),
                    GOC_SUCCESS);
          for (auto word : words[3])
            check(word, 0xdeadbeef, want, mode, sem != 0);
        }
}

TEST(HalfFma, DoubleRoundingAndTininessBoundaries) {
  struct Case {
    uint16_t a, b, c, want;
    uint32_t mode;
  };

  const Case cases[] = {// FP32 FMA followed by FP16 narrowing would round these the wrong way.
                        {0x3c01, 0x3e00, 0x8001, 0x3e01, 0},
                        {0x3c03, 0x3e00, 0x0001, 0x3e05, 0},
                        {0xbc01, 0x3e00, 0x0001, 0xbe01, 0},
                        {0xbc03, 0x3e00, 0x8001, 0xbe05, 0},
                        // All three round to min-normal, but active OMOD tests tininess before
                        // exponent packing at 2^-14 - 2^-26, not at the usual half midpoint.
                        {1, 0x39ff, 0x03ff, 0, GOC_ALU_OMOD_2},
                        {1, 0x3a00, 0x03ff, 0x0800, GOC_ALU_OMOD_2},
                        {1, 0x3a01, 0x03ff, 0x0800, GOC_ALU_OMOD_2},
                        {1, 0x39ff, 0x03ff, 0x0400, 0},
                        {0x8001, 0x39ff, 0x83ff, 0, GOC_ALU_OMOD_HALF},
                        {0x8001, 0x3a00, 0x83ff, 0x8000, GOC_ALU_OMOD_HALF},
                        {0x8001, 0x3a01, 0x83ff, 0x8000, GOC_ALU_OMOD_HALF}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (auto sem : semantics)
      for (auto test : cases) {
        uint32_t words[4][32];
        uint32_t *p[] = {words[0], words[1], words[2], words[3]};
        std::fill(words[0], words[0] + 32, test.a);
        std::fill(words[1], words[1] + 32, test.b);
        std::fill(words[2], words[2] + 32, test.c);
        std::fill(words[3], words[3] + 32, 0xdeadbeef);
        ASSERT_EQ(reference(test.a, test.b, test.c, test.mode, false), test.want);
        ASSERT_EQ(goc_rdna4_v_fma_f16(cpu | sem, UINT32_MAX, test.mode, p + 3, p, p + 1, p + 2),
                  GOC_SUCCESS);
        for (auto word : words[3])
          check(word, 0xdeadbeef, test.want, test.mode, true);
      }
}

TEST(HalfFma, ExactPreservesHostEnvironment) {
  std::fenv_t saved;
  std::fegetenv(&saved);
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
      std::fesetround(rounding);
      std::feclearexcept(FE_ALL_EXCEPT);
      std::feraiseexcept(FE_DIVBYZERO);
      uint32_t words[4][32];
      for (int reg = 0; reg < 4; ++reg)
        for (int lane = 0; lane < 32; ++lane)
          words[reg][lane] = values[(lane * 5 + reg * 3) % 24];
      run(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, words);
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
    }
  std::fesetenv(&saved);
}

TEST(HalfFma, ValidationAndSemantics) {
  uint32_t data[32];
  std::fill(data, data + 32, 0xdeadbeef);
  auto p = data;
  for (uint64_t mask : {UINT64_C(0), UINT64_C(0xffffffff00000000), UINT64_MAX}) {
    for (int bit = 13; bit < 32; ++bit)
      EXPECT_EQ(goc_rdna4_v_fma_f16(0, mask, UINT32_C(1) << bit, &p, &p, &p, &p),
                GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(goc_rdna4_v_fma_f16(UINT64_C(1) << 63, mask, 0, &p, &p, &p, &p),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(
        goc_rdna4_v_fma_f16((UINT64_C(2) << 16) | GOC_SEMANTICS_STRICT, mask, 0, &p, &p, &p, &p),
        GOC_ERROR_UNSUPPORTED_SEMANTICS);
  }
  for (auto word : data)
    EXPECT_EQ(word, 0xdeadbeef);
  EXPECT_EQ(goc_rdna4_v_fma_f16(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 0, nullptr,
                                nullptr, nullptr, nullptr),
            GOC_SUCCESS);
  EXPECT_EQ(goc_rdna4_v_fma_f16(UINT64_C(2) << 16, UINT32_MAX, 0, &p, &p, &p, &p), GOC_SUCCESS);
}
