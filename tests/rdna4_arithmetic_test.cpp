// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "goc/goc.h"
#include "internal.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_dx9_hardware.h"
#include "rdna4_exec_masks.h"
#include "rdna4_fma_omod_hardware.h"
#include "rdna4_omod_reference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

extern "C" int goc_test_c_api(void);

TEST(Api, C99LinksAgainstCppImplementation) { EXPECT_EQ(goc_test_c_api(), 1); }

TEST(Arithmetic, ConstInputs) {
  const uint32_t a[32] = {0x40000000}, b[32] = {0x40400000}, c[32] = {0x40800000};
  const uint32_t *const pa = a, *const pb = b, *const pc = c;
  uint32_t d[32] = {};
  uint32_t *pd = d;
  ASSERT_EQ(goc_rdna4_v_fma_f32(0, 1, 0, &pd, &pa, &pb, &pc), GOC_SUCCESS);
  EXPECT_EQ(d[0], 0x41200000u);
}

TEST(Arithmetic, DeterministicFmaMaskAndAliasing) {
  std::mt19937 rng(42);
  std::array<uint32_t, 32> a, b, c;
  std::array<uint32_t, 32> expected;
  for (int i = 0; i < 32; ++i) {
    int x = int(rng() % 129) - 64, y = int(rng() % 129) - 64, z = int(rng() % 129) - 64;
    a[i] = goc::as_bits(float(x));
    b[i] = goc::as_bits(float(y));
    c[i] = goc::as_bits(float(z));
    expected[i] = goc::as_bits(float(x * y + z));
  }
  const auto original = a;
  auto pa = a.data(), pb = b.data(), pc = c.data();
  ASSERT_EQ(goc_rdna4_v_fma_f32(0, 0xaaaaaaaa, 0, &pa, &pa, &pb, &pc), 0);
  for (int i = 0; i < 32; ++i)
    EXPECT_EQ(a[i], (i % 2) ? expected[i] : original[i]);
}

TEST(Arithmetic, ErrorsPreserveDestination) {
  uint32_t a[32] = {}, d[32];
  auto pa = a, pd = d;
  for (auto &v : d)
    v = 0xdeadbeef;
  EXPECT_EQ(goc_rdna4_v_fma_f32(GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, UINT32_MAX, 0,
                                &pd, &pa, &pa, &pa),
            GOC_ERROR_UNSUPPORTED_SEMANTICS);
  EXPECT_EQ(goc_rdna4_v_log_f32(0, UINT32_MAX, GOC_ALU_NEG_B, &pd, &pa), GOC_ERROR_INVALID_FLAGS);
  EXPECT_EQ(goc_rdna4_v_log_f32(1ULL << 63, UINT32_MAX, 0, &pd, &pa), GOC_ERROR_INVALID_FLAGS);
  for (auto v : d)
    EXPECT_EQ(v, 0xdeadbeef);
  EXPECT_EQ(goc_rdna4_v_fma_f32(GOC_SEMANTICS_EXACT_EMPIRICAL, 0, 0, &pd, &pa, &pa, &pa), 0);
  for (auto v : d)
    EXPECT_EQ(v, 0xdeadbeef);
}

TEST(Arithmetic, LogPowersOfTwoAndEmptyMask) {
  uint32_t a[32];
  for (int i = 0; i < 32; ++i)
    a[i] = goc::as_bits(std::ldexp(1.0f, i - 16));
  auto pa = a;
  ASSERT_EQ(goc_rdna4_v_log_f32(0, 0U, 0, &pa, &pa), 0);
  EXPECT_EQ(a[0], goc::as_bits(std::ldexp(1.0f, -16)));
  ASSERT_EQ(goc_rdna4_v_log_f32(0, UINT32_MAX, 0, &pa, &pa), 0);
  for (int i = 0; i < 32; ++i)
    EXPECT_EQ(a[i], goc::as_bits(float(i - 16)));
}

TEST(Arithmetic, AllCpuLevelsFmaGoldenAndAliasing) {
  // Literal IEEE inputs/outputs cover cancellation, fused rounding, subnormals,
  // infinities and signed zero. The fused witness is (1+2^-23)*(1-2^-23)-1.
  const uint32_t a_bits[] = {0x3f800001, 0x00000001, 0x7f800000, 0x80000000,
                             0x3fc00000, 0xc0000000, 0x00800000, 0x3f800000};
  const uint32_t b_bits[] = {0x3f7ffffe, 0x3f800000, 0x40000000, 0x40000000,
                             0x40000000, 0x40400000, 0x3f000000, 0x3f800000};
  const uint32_t c_bits[] = {0xbf800000, 0x00000001, 0x00000000, 0x80000000,
                             0xbf800000, 0x40c00000, 0x00000000, 0xbf800000};
  const uint32_t golden[] = {0xa8800000, 0x00000002, 0x7f800000, 0x80000000,
                             0x40000000, 0x00000000, 0x00400000, 0x00000000};
  for (uint64_t level = 0; level <= goc_init_cpu_flags(); ++level)
    for (int alias = 0; alias < 4; ++alias)
      for (uint32_t mask : rdna4_exec_masks()) {
        SCOPED_TRACE(::testing::Message()
                     << "level=" << level << " alias=" << alias << " mask=" << mask);
        uint32_t storage[4][34]; // Offsets avoid requiring SIMD alignment.
        uint32_t *ptrs[4];
        for (int j = 0; j < 4; ++j) {
          ptrs[j] = storage[j] + 1;
          storage[j][0] = storage[j][33] = 0xdeadbeef;
        }
        for (int i = 0; i < 32; ++i) {
          ptrs[0][i] = a_bits[i % 8];
          ptrs[1][i] = b_bits[i % 8];
          ptrs[2][i] = c_bits[i % 8];
          ptrs[3][i] = 0x12345678;
        }
        std::array<uint32_t, 32> before;
        std::copy(ptrs[alias], ptrs[alias] + 32, before.begin());
        ASSERT_EQ(goc_rdna4_v_fma_f32(level, mask, 0, &ptrs[alias], &ptrs[0], &ptrs[1], &ptrs[2]),
                  0);
        for (int i = 0; i < 32; ++i)
          EXPECT_EQ(ptrs[alias][i], ((mask >> i) & 1) ? golden[i % 8] : before[i]);
        for (int j = 0; j < 4; ++j) {
          EXPECT_EQ(storage[j][0], 0xdeadbeef);
          EXPECT_EQ(storage[j][33], 0xdeadbeef);
        }
      }
}

namespace {

void check_fma_modifiers(bool dx9) {
  auto fn = dx9 ? goc_rdna4_v_fma_dx9_zero_f32 : goc_rdna4_v_fma_f32;
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint32_t modifiers = 0; modifiers < 512; ++modifiers)
      for (uint32_t mask : rdna4_exec_masks())
        for (int alias = 0; alias < 4; ++alias) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << "/" << modifiers << "/" << mask << "/" << alias);
          uint32_t storage[4][34], original[32], expected[32];
          uint32_t *v[4];
          for (int reg = 0; reg < 4; ++reg) {
            std::fill(storage[reg], storage[reg] + 34, 0xdeadbeef);
            v[reg] = storage[reg] + 1;
          }
          for (int lane = 0; lane < 32; ++lane) {
            double inputs[3] = {(lane - 17) * 0.25, (lane % 7 - 3) * 0.5, (lane % 11 - 5) * 0.25};
            for (int source = 0; source < 3; ++source) {
              v[source][lane] = goc::as_bits(float(inputs[source]));
              if (modifiers & (8u << source))
                inputs[source] = std::abs(inputs[source]);
              if (modifiers & (1u << source))
                inputs[source] = -inputs[source];
            }
            // Dyadic inputs have an exactly representable double-precision
            // product/sum, independent of the implementation's FP32 FMA.
            float want = float(inputs[0] * inputs[1] + inputs[2]);
            if (dx9 && (inputs[0] == 0 || inputs[1] == 0))
              want = float(0.0 + inputs[2]);
            want = goc_test::omod_f32_reference(want, modifiers);
            if (modifiers & GOC_ALU_CLAMP)
              want = std::min(1.0f, std::max(0.0f, want));
            expected[lane] = goc::as_bits(want);
            original[lane] = v[alias][lane];
          }
          ASSERT_EQ(fn(cpu, mask, modifiers, &v[alias], &v[0], &v[1], &v[2]), GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(v[alias][lane], ((mask >> lane) & 1) ? expected[lane] : original[lane]);
          for (const auto &reg : storage) {
            EXPECT_EQ(reg[0], 0xdeadbeef);
            EXPECT_EQ(reg[33], 0xdeadbeef);
          }
        }
}

} // namespace

TEST(Arithmetic, FmaAllModifiersCpuLevelsMasksAndAliases) { check_fma_modifiers(false); }

TEST(Arithmetic, Dx9FmaAllModifiersCpuLevelsMasksAndAliases) { check_fma_modifiers(true); }

TEST(Arithmetic, FmaModifierSpecialValues) {
  const uint32_t a_bits[] = {0x3f800001, 0x7fc12345, 0x7f800000, 0x80000000,
                             0x00000001, 0x7f7fffff, 0x3fc00000, 0x3fc00000};
  const uint32_t b_bits[] = {0x3f7ffffe, 0x3f800000, 0x00000000, 0x40000000,
                             0x3f800000, 0x40000000, 0x3fc00000, 0x3fc00000};
  const uint32_t c_bits[] = {0xbf800000, 0, 0, 0x80000000, 1, 0, 0xbe800000, 0xbe800000};
  const uint32_t modes[] = {
      GOC_ALU_OMOD_2,    GOC_ALU_CLAMP, GOC_ALU_CLAMP, GOC_ALU_OMOD_HALF,
      GOC_ALU_OMOD_HALF, GOC_ALU_CLAMP, GOC_ALU_ABS_C, GOC_ALU_ABS_C | GOC_ALU_NEG_C};
  const uint32_t golden[] = {0xa9000000, 0, 0, 0, 0, 0x3f800000, 0x40200000, 0x40000000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int test = 0; test < 8; ++test) {
      uint32_t a[32], b[32], c[32], d[32];
      std::fill(a, a + 32, a_bits[test]);
      std::fill(b, b + 32, b_bits[test]);
      std::fill(c, c + 32, c_bits[test]);
      auto pa = a, pb = b, pc = c, pd = d;
      ASSERT_EQ(goc_rdna4_v_fma_f32(cpu, UINT32_MAX, modes[test], &pd, &pa, &pb, &pc), GOC_SUCCESS);
      for (uint32_t value : d)
        EXPECT_EQ(value, golden[test]);
    }
}

TEST(Arithmetic, Dx9FmaZeroProductAdditionWithEveryModifierAndAlias) {
  const uint32_t values[] = {0,          0x80000000, 1,          0x80000001, 0x3f800000, 0xbf800000,
                             0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345,
                             0x7f812345, 0xff812345, 0x3f000000, 0xbf000000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint32_t mode = 0; mode < 512; ++mode)
      for (bool reverse : {false, true})
        for (int alias = 0; alias < 4; ++alias) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << "/" << mode << "/" << reverse << "/" << alias);
          uint32_t storage[4][32], expected[32];
          uint32_t *v[4] = {storage[0], storage[1], storage[2], storage[3]};
          for (int lane = 0; lane < 32; ++lane) {
            uint32_t zero = lane & 1 ? 0x80000000 : 0;
            uint32_t other = values[(lane * 5 + 3) % 16];
            v[0][lane] = reverse ? other : zero;
            v[1][lane] = reverse ? zero : other;
            v[2][lane] = values[lane % 16];
            uint32_t want = v[2][lane];
            if (mode & GOC_ALU_ABS_C)
              want &= 0x7fffffff;
            if (mode & GOC_ALU_NEG_C)
              want ^= 0x80000000;
            // The zero product is positive; adding a flushed signed zero
            // gives +0 in RNE. An addend NaN is quieted by the addition.
            if ((want & 0x7fffffff) < 0x00800000)
              want = 0;
            else if ((want & 0x7fffffff) > 0x7f800000)
              want |= 0x00400000;
            want = goc::as_bits(goc_test::omod_f32_reference(goc::as_float(want), mode));
            if (mode & GOC_ALU_CLAMP) {
              float value = goc::as_float(want);
              want = !(value > 0) ? 0 : value > 1 ? 0x3f800000 : want;
            }
            expected[lane] = want;
          }
          ASSERT_EQ(
              goc_rdna4_v_fma_dx9_zero_f32(cpu, UINT32_MAX, mode, &v[alias], &v[0], &v[1], &v[2]),
              GOC_SUCCESS);
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(v[alias][lane], expected[lane]);
        }
}

TEST(Arithmetic, Dx9FmaFusedRoundingAndValidation) {
  // The ordinary fused-rounding witness must survive the DX9 extension.
  uint32_t a[32], b[32], c[32], d[32];
  std::fill(a, a + 32, 0x3f800001);
  std::fill(b, b + 32, 0x3f7ffffe);
  std::fill(c, c + 32, 0xbf800000);
  auto pa = a, pb = b, pc = c, pd = d;
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    std::fill(d, d + 32, 0xdeadbeef);
    EXPECT_EQ(goc_rdna4_v_fma_dx9_zero_f32(cpu, UINT32_MAX, GOC_ALU_HIGH_C, &pd, &pa, &pb, &pc),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(
        goc_rdna4_v_fma_dx9_zero_f32(cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT,
                                     UINT32_MAX, 0, &pd, &pa, &pb, &pc),
        GOC_ERROR_UNSUPPORTED_SEMANTICS);
    for (uint32_t value : d)
      EXPECT_EQ(value, 0xdeadbeef);
    ASSERT_EQ(goc_rdna4_v_fma_dx9_zero_f32(cpu, UINT32_MAX, 0, &pd, &pa, &pb, &pc), GOC_SUCCESS);
    for (uint32_t value : d)
      EXPECT_EQ(value, 0xa8800000);
  }
}

TEST(Arithmetic, FmaAndFmacOmodHardwareBoundaries) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool fmac : {false, true})
      for (unsigned omod = 0; omod < 4; ++omod)
        for (unsigned clamp = 0; clamp < 2; ++clamp)
          for (unsigned neg = 0; neg < 2; ++neg)
            for (uint32_t mask : rdna4_exec_masks())
              for (int alias = 0; alias < (fmac ? 1 : 4); ++alias) {
                SCOPED_TRACE(::testing::Message()
                             << cpu << '/' << fmac << '/' << omod << '/' << clamp << '/' << neg
                             << '/' << mask << '/' << alias);
                uint32_t words[4][34];
                uint32_t *p[4];
                for (int reg = 0; reg < 4; ++reg) {
                  std::fill(words[reg], words[reg] + 34, 0xdeadbeef);
                  p[reg] = words[reg] + 1;
                }
                for (int lane = 0; lane < 32; ++lane) {
                  p[0][lane] = 0x80000000;
                  p[1][lane] = goc_test::fma_omod_inputs[lane];
                  p[2][lane] = 0x3f800000;
                  p[3][lane] = 0x80000000;
                }
                uint32_t before[32];
                std::copy(p[alias], p[alias] + 32, before);
                uint64_t mode = (omod << 6) | (clamp ? GOC_ALU_CLAMP : 0) | neg;
                int error =
                    fmac ? goc_rdna4_v_fmac_f32(cpu, mask, mode, &p[alias], &p[1], &p[2])
                         : goc_rdna4_v_fma_f32(cpu, mask, mode, &p[alias], &p[1], &p[2], &p[3]);
                ASSERT_EQ(error, GOC_SUCCESS);
                for (int lane = 0; lane < 32; ++lane) {
                  uint32_t want = (mask >> lane) & 1
                                      ? goc_test::fma_omod_hardware[omod][clamp][neg][lane]
                                      : before[lane];
                  if ((want & 0x7fffffff) > 0x7f800000)
                    EXPECT_GT(p[alias][lane] & 0x7fffffff, 0x7f800000u);
                  else
                    EXPECT_EQ(p[alias][lane], want);
                }
                for (const auto &reg : words) {
                  EXPECT_EQ(reg[0], 0xdeadbeefu);
                  EXPECT_EQ(reg[33], 0xdeadbeefu);
                }
              }
}

TEST(Arithmetic, Dx9HardwareAllModifiersAndDenormalModes) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned fp = 0; fp < 4; ++fp)
      for (unsigned set = 0; set < 4; ++set)
        for (unsigned alias = 0; alias < 4; ++alias) {
          SCOPED_TRACE(::testing::Message() << cpu << '/' << fp << '/' << set << '/' << alias);
          uint64_t hash = goc_test::capture_hash_seed;
          uint64_t flags = cpu | (fp & 1 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                           (fp & 2 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0);
          for (uint32_t mode = 0; mode < 512; ++mode) {
            uint32_t words[4][32] = {};
            for (unsigned lane = 0; lane < 32; ++lane)
              goc_test::dx9_hardware_inputs(set, lane, words[0][lane], words[1][lane],
                                            words[2][lane]);
            uint32_t *p[] = {words[0], words[1], words[2], words[3]};
            ASSERT_EQ(goc_rdna4_v_fma_dx9_zero_f32(flags, UINT32_MAX, mode, &p[alias], &p[0], &p[1],
                                                   &p[2]),
                      GOC_SUCCESS);
            for (uint32_t value : words[alias]) {
              if ((value & 0x7fffffff) > 0x7f800000)
                value = 0x7fc00000;
              for (unsigned shift = 0; shift < 32; shift += 8)
                hash = goc_test::capture_hash_word(hash, ((value >> shift) & 255));
            }
          }
          EXPECT_EQ(hash, goc_test::dx9_hardware_hashes[set]);
        }
}

TEST(Arithmetic, Dx9DppHardwareCorpus) {
  // RX 9070/gfx1201, raw ISA opcode 0x209 (LLVM lacks DPP assembly), MODE 0xf0: all 512 modifiers,
  // seven DPP descriptors, and eight EXEC masks. FNV hashes words with NaN payloads canonicalized.
  const uint32_t values[] = {1,          0x807fffff, 0x00800000, 0x80000000,
                             0x7f800001, 0xff800000, 0x3f800000, 0xff7fffff};
  const uint32_t masks[] = {UINT32_MAX, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (uint32_t mask : masks)
      for (unsigned mode = 0; mode < 512; ++mode)
        for (uint64_t descriptor : goc_test::dpp_modes) {
          uint32_t words[4][32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            words[0][lane] = values[lane % 8];
            words[1][lane] = values[(lane + 3) % 8];
            words[2][lane] = values[(lane + 5) % 8];
            words[3][lane] = 0xdead0000u + lane;
          }
          const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
          uint32_t *d[] = {words[3]};
          ASSERT_EQ(goc_rdna4_v_fma_dx9_zero_f32(cpu, mask, descriptor | mode, d, a, b, c),
                    GOC_SUCCESS);
          for (uint32_t word : words[3]) {
            if ((word & 0x7fffffff) > 0x7f800000)
              word = 0x7fc00000;
            hash = goc_test::capture_hash_word(hash, word);
          }
        }
    EXPECT_EQ(hash, 0xda037611da867b25ULL);
  }
}

TEST(Arithmetic, Dx9DppModifiersMasksAliasesAndGuards) {
  for (uint32_t mode = 0; mode < 512; ++mode)
    for (uint64_t descriptor : goc_test::dpp_modes)
      for (uint32_t mask : rdna4_exec_masks()) {
        if (mode != 0 && mode != 511 && mask != UINT32_MAX)
          continue;
        for (bool shared : {false, true}) {
          uint32_t initial[4][34], expected[32], writes = 0;
          for (unsigned reg = 0; reg < 4; ++reg)
            for (unsigned lane = 0; lane < 34; ++lane)
              initial[reg][lane] = goc::as_bits(float(int((lane * (reg + 1)) % 17) - 8) * 0.25f);
          for (unsigned lane = 0; lane < 32; ++lane) {
            int source;
            if (!goc_test::dpp_source(descriptor, mask, lane, source))
              continue;
            writes |= 1u << lane;
            double input[] = {source < 0 ? 0.0 : goc::as_float(initial[0][source + 1]),
                              goc::as_float(initial[shared ? 0 : 1][lane + 1]),
                              goc::as_float(initial[shared ? 0 : 2][lane + 1])};
            for (unsigned i = 0; i < 3; ++i) {
              if (mode & (8u << i))
                input[i] = std::abs(input[i]);
              if (mode & (1u << i))
                input[i] = -input[i];
            }
            float value =
                float((input[0] == 0 || input[1] == 0 ? 0.0 : input[0] * input[1]) + input[2]);
            value = goc_test::omod_f32_reference(value, mode);
            if (mode & GOC_ALU_CLAMP)
              value = std::min(1.0f, std::max(0.0f, value));
            expected[lane] = goc::as_bits(value);
          }
          for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
            for (unsigned target = 0; target < 4; ++target) {
              uint32_t words[4][34];
              for (unsigned reg = 0; reg < 4; ++reg)
                std::copy_n(initial[reg], 34, words[reg]);
              const uint32_t *a[] = {words[0] + 1}, *b[] = {words[shared ? 0 : 1] + 1},
                             *c[] = {words[shared ? 0 : 2] + 1};
              uint32_t *d[] = {words[target] + 1};
              ASSERT_EQ(goc_rdna4_v_fma_dx9_zero_f32(cpu, mask, descriptor | mode, d, a, b, c),
                        GOC_SUCCESS);
              for (unsigned reg = 0; reg < 4; ++reg)
                for (unsigned lane = 0; lane < 34; ++lane) {
                  uint32_t want = initial[reg][lane];
                  if (reg == target && lane > 0 && lane < 33 && ((writes >> (lane - 1)) & 1))
                    want = expected[lane - 1];
                  ASSERT_EQ(words[reg][lane], want)
                      << mode << "/" << descriptor << "/" << cpu << "/" << lane;
                }
            }
        }
      }
}
