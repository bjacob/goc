// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_dpp_reference.h"
#include "rdna4_exec_masks.h"
#include "rdna4_sad_reference.h"

#include <algorithm>
#include <array>
#include <cfenv>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <vector>

namespace {

using Fn = decltype(&goc_rdna4_v_sad_u8);
const Fn functions[] = {goc_rdna4_v_sad_u8,          goc_rdna4_v_sad_hi_u8,
                        goc_rdna4_v_sad_u16,         goc_rdna4_v_sad_u32,
                        goc_rdna4_v_msad_u8,         goc_rdna4_v_qsad_pk_u16_u8,
                        goc_rdna4_v_mqsad_pk_u16_u8, goc_rdna4_v_mqsad_u32_u8};

int outputs(int op) { return op == 7 ? 4 : op >= 5 ? 2 : 1; }

::testing::AssertionResult check(int op, uint64_t flags, bool clamp, uint32_t words[12][32]) {
  uint32_t expected[12][32];
  for (int reg = 0; reg < 12; ++reg)
    std::copy_n(words[reg], 32, expected[reg]);
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t c[] = {words[3][lane], words[4][lane], words[5][lane], words[6][lane]};
    auto result =
        goc_test::sad_reference(op, words[0][lane], words[1][lane], words[2][lane], c, clamp);
    for (int reg = 0; reg < outputs(op); ++reg)
      expected[7 + reg][lane] = result[reg];
  }
  const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2]},
                 *c[] = {words[3], words[4], words[5], words[6]};
  uint32_t *d[] = {words[7], words[8], words[9], words[10]};
  int status = functions[op](flags, UINT32_MAX, clamp ? GOC_ALU_CLAMP : 0, d, a, b, c);
  if (status != GOC_SUCCESS)
    return ::testing::AssertionFailure() << "status " << status;
  for (int reg = 0; reg < 12; ++reg)
    for (int lane = 0; lane < 32; ++lane)
      if (words[reg][lane] != expected[reg][lane])
        return ::testing::AssertionFailure()
               << op << "/" << flags << "/" << clamp << "/" << reg << "/" << lane << ": "
               << words[reg][lane] << " != " << expected[reg][lane];
  return ::testing::AssertionSuccess();
}

} // namespace

TEST(Sad, BoundaryTriplesRandomInputsAndSaturation) {
  const uint32_t edges[] = {0,          1,          0xff,       0x100,      0xffff,     0x10000,
                            0x7fffffff, 0x80000000, 0xffffffff, 0xfffffffe, 0xffffff00, 0xffff0000,
                            0x01020304, 0x04030201, 0xff00ff00, 0x00ff00ff};
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool clamp : {false, true}) {
        std::mt19937 random(6205);
        for (int start = 0; start < 8192; start += 32) {
          uint32_t words[12][32];
          for (int lane = 0; lane < 32; ++lane) {
            int index = start + lane;
            for (auto &reg : words)
              reg[lane] = random();
            if (start < 4096) {
              words[0][lane] = edges[index % 16];
              words[1][lane] = edges[(index + 9) % 16];
              words[2][lane] = edges[(index / 16) % 16];
              for (int reg = 0; reg < 4; ++reg)
                words[3 + reg][lane] = edges[(index / 256 + reg * 3) % 16];
            }
          }
          ASSERT_TRUE(check(op, cpu, clamp, words));
        }
      }
}

TEST(Sad, EveryBytePair) {
  for (int op : {0, 1, 4, 5, 6, 7})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool clamp : {false, true})
        for (unsigned start = 0; start < 65536; start += 32) {
          uint32_t words[12][32] = {};
          for (unsigned lane = 0; lane < 32; ++lane) {
            unsigned x = (start + lane) / 256, y = (start + lane) % 256;
            words[0][lane] = x * 0x01010101u;
            words[1][lane] = (255 - x) * 0x01010101u;
            words[2][lane] = y * 0x01010101u;
            for (int reg = 0; reg < 4; ++reg)
              words[3 + reg][lane] = UINT32_MAX - 85 * (lane + reg);
          }
          ASSERT_TRUE(check(op, cpu, clamp, words));
        }
}

TEST(Sad, EveryHalfwordAndUnsignedExtremes) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (bool clamp : {false, true})
      for (uint32_t against : {0u, 1u, 32767u, 32768u, 65534u, 65535u})
        for (unsigned start = 0; start < 65536; start += 32) {
          uint32_t words[12][32] = {};
          for (unsigned lane = 0; lane < 32; ++lane) {
            unsigned value = start + lane;
            words[0][lane] = value | ((65535 - value) << 16);
            words[2][lane] = against | (against << 16);
            words[3][lane] = UINT32_MAX - value;
          }
          ASSERT_TRUE(check(2, cpu, clamp, words));
        }
}

TEST(Sad, LiteralWindowsAndIndependentOverflow) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (int op = 0; op < 8; ++op)
      for (bool clamp : {false, true}) {
        uint32_t words[12][32] = {};
        const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2]},
                       *c[] = {words[3], words[4], words[5], words[6]};
        uint32_t *d[] = {words[7], words[8], words[9], words[10]};
        for (int lane = 0; lane < 32; ++lane) {
          words[2][lane] = UINT32_MAX;
          for (int reg = 3; reg < 7; ++reg)
            words[reg][lane] = UINT32_MAX;
        }
        ASSERT_EQ(functions[op](cpu, UINT32_MAX, clamp ? GOC_ALU_CLAMP : 0, d, a, b, c),
                  GOC_SUCCESS);
        const uint32_t wrapping[] = {1019, 0x03fbffff, 0x1fffd,    0xfffffffe,
                                     1019, 0x03fb03fb, 0x03fb03fb, 1019};
        for (int reg = 0; reg < outputs(op); ++reg)
          for (int lane = 0; lane < 32; ++lane)
            ASSERT_EQ(d[reg][lane], clamp ? UINT32_MAX : wrapping[op]);
        if (op < 5)
          continue;
        for (int lane = 0; lane < 32; ++lane) {
          words[0][lane] = 0x04030201;
          words[1][lane] = 0x08070605;
          words[2][lane] = op == 5 ? 0x04030201 : 0x00030001;
          if (op == 7) {
            for (int reg = 0; reg < 4; ++reg)
              words[3 + reg][lane] = 100 * (reg + 1);
          } else {
            words[3][lane] = 0x0014000a;
            words[4][lane] = 0x0028001e;
          }
        }
        ASSERT_EQ(functions[op](cpu, UINT32_MAX, clamp ? GOC_ALU_CLAMP : 0, d, a, b, c),
                  GOC_SUCCESS);
        const uint32_t golden[][4] = {
            {0x0018000a, 0x00340026, 0, 0}, {0x0016000a, 0x002e0022, 0, 0}, {100, 202, 304, 406}};
        for (int reg = 0; reg < outputs(op); ++reg)
          for (int lane = 0; lane < 32; ++lane)
            EXPECT_EQ(d[reg][lane], golden[op - 5][reg]);
      }
}

TEST(Sad, MaskedBytesAndUnusedSourceByte) {
  for (int op : {4, 5, 6, 7})
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool clamp : {false, true})
        for (int byte_mask = 0; byte_mask < 16; ++byte_mask)
          for (unsigned start = 0; start < 256; start += 32) {
            uint32_t words[12][32] = {};
            for (unsigned lane = 0; lane < 32; ++lane) {
              words[0][lane] = 0x04030201;
              words[1][lane] = ((start + lane) << 24) | 0x00070605;
              for (int byte = 0; byte < 4; ++byte)
                if ((byte_mask >> byte) & 1)
                  words[2][lane] |= uint32_t(20 + byte) << (8 * byte);
              for (int reg = 0; reg < 4; ++reg)
                words[3 + reg][lane] = 0xfff0fff0u + reg;
            }
            ASSERT_TRUE(check(op, cpu, clamp, words));
            for (int reg = 0; reg < outputs(op); ++reg)
              for (int lane = 1; lane < 32; ++lane)
                EXPECT_EQ(words[7 + reg][lane], words[7 + reg][0]);
          }
}

TEST(Sad, MasksAndEveryDestinationSourceAlias) {
  for (int op = 0; op < 8; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (bool clamp : {false, true})
        for (bool shared_inputs : {false, true}) {
          std::mt19937 random(1214);
          uint32_t original[12][32], result[4][32];
          for (auto &reg : original)
            for (auto &word : reg)
              word = random();
          int source[7];
          for (int i = 0; i < 7; ++i)
            source[i] = shared_inputs ? 0 : i;
          for (int lane = 0; lane < 32; ++lane) {
            uint32_t c[4];
            for (int reg = 0; reg < 4; ++reg)
              c[reg] = original[source[3 + reg]][lane];
            auto value =
                goc_test::sad_reference(op, original[source[0]][lane], original[source[1]][lane],
                                        original[source[2]][lane], c, clamp);
            for (int reg = 0; reg < outputs(op); ++reg)
              result[reg][lane] = value[reg];
          }
          std::vector<std::array<int, 4>> aliases = {
              {7, 8, 9, 10}, {0, 0, 0, 0}, {6, 5, 4, 3}, {1, 0, 2, 4}};
          for (int slot = 0; slot < outputs(op); ++slot)
            for (int target = 0; target < 11; ++target) {
              std::array<int, 4> map = {7, 8, 9, 10};
              map[slot] = target;
              aliases.push_back(map);
            }
          for (uint32_t mask : rdna4_exec_masks())
            for (const auto &map : aliases) {
              uint32_t words[12][32], expected[12][32];
              for (int reg = 0; reg < 12; ++reg) {
                std::copy_n(original[reg], 32, words[reg]);
                std::copy_n(original[reg], 32, expected[reg]);
              }
              const uint32_t *a[] = {words[source[0]], words[source[1]]}, *b[] = {words[source[2]]},
                             *c[] = {words[source[3]], words[source[4]], words[source[5]],
                                     words[source[6]]};
              uint32_t *d[] = {words[map[0]], words[map[1]], words[map[2]], words[map[3]]};
              for (int reg = 0; reg < outputs(op); ++reg)
                for (int lane = 0; lane < 32; ++lane)
                  if ((mask >> lane) & 1)
                    expected[map[reg]][lane] = result[reg][lane];
              ASSERT_EQ(functions[op](cpu, mask, clamp ? GOC_ALU_CLAMP : 0, d, a, b, c),
                        GOC_SUCCESS);
              for (int reg = 0; reg < 12; ++reg)
                ASSERT_TRUE(std::equal(words[reg], words[reg] + 32, expected[reg]))
                    << op << "/" << cpu << "/" << clamp << "/" << mask << "/" << reg;
            }
        }
}

TEST(Sad, ValidationAndHostFpState) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    std::fesetround(rounding);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
    for (int op = 0; op < 8; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
        uint32_t words[12][32];
        for (auto &reg : words)
          std::fill_n(reg, 32, 0xdeadbeef);
        const uint32_t *a[] = {words[0], words[1]}, *b[] = {words[2]},
                       *c[] = {words[3], words[4], words[5], words[6]};
        uint32_t *d[] = {words[7], words[8], words[9], words[10]};
        for (uint32_t mask : {UINT32_C(0), UINT32_MAX}) {
          for (int bit = 0; bit < 32; ++bit)
            if ((uint32_t(1) << bit) != GOC_ALU_CLAMP) {
              EXPECT_EQ(functions[op](cpu, mask, uint32_t(1) << bit, d, a, b, c),
                        GOC_ERROR_INVALID_FLAGS);
            }
          EXPECT_EQ(functions[op](cpu | (UINT64_C(1) << 63), mask, 0, d, a, b, c),
                    GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(functions[op](cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, mask,
                                  0, d, a, b, c),
                    GOC_ERROR_UNSUPPORTED_SEMANTICS);
        }
        for (auto &reg : words)
          for (uint32_t word : reg)
            EXPECT_EQ(word, 0xdeadbeef);
        for (bool clamp : {false, true})
          EXPECT_TRUE(check(op, cpu | GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_FP16_OVFL, clamp, words));
      }
    EXPECT_EQ(std::fegetround(), rounding);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
  }
}

TEST(Sad, DppClampMasksAliasesAndGuards) {
  std::mt19937 random(7351);
  for (int op = 0; op < 5; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (auto descriptor : goc_test::dpp_modes)
        for (bool clamp : {false, true})
          for (auto mask : rdna4_exec_masks())
            for (unsigned breg = 0; breg < 2; ++breg)
              for (unsigned creg = 0; creg < 3; ++creg)
                for (unsigned dreg = 0; dreg < 4; ++dreg) {
                  uint32_t storage[4][34], expected[4][34];
                  for (unsigned reg = 0; reg < 4; ++reg)
                    for (unsigned word = 0; word < 34; ++word)
                      storage[reg][word] = expected[reg][word] = random();
                  uint64_t mode = descriptor | (clamp ? GOC_ALU_CLAMP : 0);
                  for (unsigned lane = 0; lane < 32; ++lane) {
                    int source = 0;
                    if (goc_test::dpp_source(mode, mask, lane, source)) {
                      uint32_t accumulator = storage[creg][lane + 1];
                      expected[dreg][lane + 1] =
                          goc_test::sad_reference(op, source < 0 ? 0 : storage[0][source + 1], 0,
                                                  storage[breg][lane + 1], &accumulator, clamp)[0];
                    }
                  }
                  const uint32_t *a[] = {storage[0] + 1}, *b[] = {storage[breg] + 1},
                                 *c[] = {storage[creg] + 1};
                  uint32_t *d[] = {storage[dreg] + 1};
                  ASSERT_EQ(functions[op](cpu, mask, mode, d, a, b, c), GOC_SUCCESS);
                  for (unsigned reg = 0; reg < 4; ++reg)
                    for (unsigned word = 0; word < 34; ++word)
                      ASSERT_EQ(storage[reg][word], expected[reg][word])
                          << op << "/" << cpu << "/" << mode;
                }
}

TEST(Sad, DppValidation) {
  for (auto descriptor : goc_test::dpp_modes) {
    for (unsigned op = 0; op < 5; ++op) {
      EXPECT_EQ(functions[op](0, 0, descriptor, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
      for (auto invalid : {UINT64_C(1) << 36, uint64_t(GOC_ALU_NEG_A), uint64_t(GOC_ALU_HIGH_A)})
        EXPECT_EQ(functions[op](0, 0, descriptor | invalid, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
    }
    for (unsigned op = 5; op < 8; ++op)
      EXPECT_EQ(functions[op](0, 0, descriptor, nullptr, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
  }
}

// RX 9070: five operations, seven descriptors, CLAMP off/on and eight EXEC masks.
TEST(Sad, DppHardwareCorpus) {
  const uint32_t values[] = {0,          0xffffffff, 0x00ff00ff, 0xff00ff00,
                             0x80008000, 0x7fff7fff, 0xfffffffe, 1};
  const uint32_t masks[] = {0xffffffff, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    uint64_t hash = goc_test::capture_hash_seed;
    for (auto mask : masks)
      for (unsigned op = 0; op < 5; ++op)
        for (auto descriptor : goc_test::dpp_modes)
          for (bool clamp : {false, true}) {
            uint32_t av[32], bv[32], cv[32], output[32];
            for (unsigned lane = 0; lane < 32; ++lane) {
              av[lane] = values[lane % 8];
              bv[lane] = values[(lane + 3) % 8];
              cv[lane] = values[(lane + 5) % 8];
              output[lane] = 0xdead0000u + lane;
            }
            const uint32_t *a[] = {av}, *b[] = {bv}, *c[] = {cv};
            uint32_t *d[] = {output};
            ASSERT_EQ(
                functions[op](cpu, mask, descriptor | (clamp ? GOC_ALU_CLAMP : 0), d, a, b, c),
                GOC_SUCCESS);
            for (auto word : output)
              hash = goc_test::capture_hash_word(hash, word);
          }
    EXPECT_EQ(hash, UINT64_C(0x9a854a2e5f977214)) << cpu;
  }
}
