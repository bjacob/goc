// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "cube_hardware.h"
#include "cube_reference.h"
#include "dpp_reference.h"
#include "exec_masks.h"
#include "fp_environment.h"
#include "goc/goc.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <vector>

namespace {

using Fn = decltype(&goc_v_cubeid_f32_rdna4);
const Fn functions[] = {goc_v_cubeid_f32_rdna4, goc_v_cubesc_f32_rdna4, goc_v_cubetc_f32_rdna4,
                        goc_v_cubema_f32_rdna4};
const decltype(&goc_v_cubeid_f32_rdna4_wave64) wave64_functions[] = {
    goc_v_cubeid_f32_rdna4_wave64, goc_v_cubesc_f32_rdna4_wave64, goc_v_cubetc_f32_rdna4_wave64,
    goc_v_cubema_f32_rdna4_wave64};
const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

} // namespace

TEST(Cube, HardwareCartesianCorpus) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned column = 0; column < 20; ++column) {
      uint64_t digest = goc_test::capture_hash_seed;
      for (unsigned start = 0; start < 4096; start += 32) {
        uint32_t av[32], bv[32], cv[32], dv[32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          unsigned index = start + lane;
          av[lane] = goc_test::cube_capture_values[index / 256];
          bv[lane] = goc_test::cube_capture_values[(index / 16) % 16];
          cv[lane] = goc_test::cube_capture_values[index % 16];
        }
        const uint32_t *a[] = {av}, *b[] = {bv}, *c[] = {cv};
        uint32_t *d[] = {dv};
        ASSERT_EQ(functions[column / 5](cpu | exact, UINT32_MAX,
                                        goc_test::cube_capture_modes[column % 5], d, a, b, c),
                  GOC_SUCCESS);
        for (auto word : dv)
          digest = goc_test::capture_hash_bytes(digest, word, 4);
      }
      EXPECT_EQ(digest, goc_test::cube_capture_digests[column]) << cpu << "/" << column;
    }
}

TEST(Cube, AxisTiesAtEveryMaskedLaneWithAlias) {
  // Equal magnitudes select Z before Y before X, including negative faces.
  const uint32_t inputs[][3] = {{0x3f800000, 0xbf800000, 0x3f800000},
                                {0xbf800000, 0x3f800000, 0xbf800000},
                                {0x3f800000, 0xbf800000, 0},
                                {0xbf800000, 0, 0}};
  const uint32_t faces[] = {0x40800000, 0x40a00000, 0x40400000, 0x3f800000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned selected = 0; selected < 32; ++selected) {
      uint32_t words[3][32];
      for (unsigned lane = 0; lane < 32; ++lane)
        for (unsigned reg = 0; reg < 3; ++reg)
          words[reg][lane] = inputs[lane % 4][reg];
      const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
      uint32_t *d[] = {words[0]};
      ASSERT_EQ(goc_v_cubeid_f32_rdna4(cpu | exact, 1U << selected, 0, d, a, b, c), GOC_SUCCESS);
      for (unsigned lane = 0; lane < 32; ++lane)
        EXPECT_EQ(words[0][lane], lane == selected ? faces[lane % 4] : inputs[lane % 4][0]);
    }
}

TEST(Cube, HardwareLiteralCases) {
  for (const auto &capture : goc_test::cube_captures)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned column = 0; column < 20; ++column) {
        uint32_t av[32], bv[32], cv[32], dv[32];
        std::fill_n(av, 32, capture.source[0]);
        std::fill_n(bv, 32, capture.source[1]);
        std::fill_n(cv, 32, capture.source[2]);
        const uint32_t *a[] = {av}, *b[] = {bv}, *c[] = {cv};
        uint32_t *d[] = {dv};
        ASSERT_EQ(functions[column / 5](cpu, UINT32_MAX, goc_test::cube_capture_modes[column % 5],
                                        d, a, b, c),
                  GOC_SUCCESS);
        for (auto word : dv)
          ASSERT_EQ(word, capture.expected[column]) << cpu << "/" << column;
      }
}

TEST(Cube, EveryModifierSpecialValuesAndRandomBits) {
  std::mt19937 random(998143);
  for (unsigned op = 0; op < 4; ++op)
    for (uint32_t mode = 0; mode < 512; ++mode)
      for (unsigned block = 0; block < 16; ++block) {
        uint32_t av[32], bv[32], cv[32], dv[32], expected[32];
        for (unsigned lane = 0; lane < 32; ++lane) {
          av[lane] = block < 8 ? goc_test::cube_capture_values[(lane + block) % 16] : random();
          bv[lane] =
              block < 8 ? goc_test::cube_capture_values[(lane * 3 + block * 2) % 16] : random();
          cv[lane] =
              block < 8 ? goc_test::cube_capture_values[(lane * 7 + block * 3) % 16] : random();
          expected[lane] = goc_test::cube_reference(op, av[lane], bv[lane], cv[lane], mode);
        }
        const uint32_t *a[] = {av}, *b[] = {bv}, *c[] = {cv};
        uint32_t *d[] = {dv};
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          ASSERT_EQ(functions[op](cpu | exact, UINT32_MAX, mode, d, a, b, c), GOC_SUCCESS);
          for (unsigned lane = 0; lane < 32; ++lane)
            ASSERT_EQ(dv[lane], expected[lane])
                << op << "/" << mode << "/" << cpu << "/" << std::hex << av[lane] << "/" << bv[lane]
                << "/" << cv[lane];
        }
      }
}

TEST(Cube, EveryModifierMaskAndDestinationAlias) {
  std::mt19937 random(55919);
  for (unsigned op = 0; op < 4; ++op)
    for (uint32_t mode = 0; mode < 512; ++mode) {
      uint32_t original[4][34], result[32];
      for (auto &reg : original)
        for (auto &word : reg)
          word = random();
      for (unsigned lane = 0; lane < 32; ++lane)
        result[lane] = goc_test::cube_reference(op, original[0][lane + 1], original[1][lane + 1],
                                                original[2][lane + 1], mode);
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint32_t exec_mask : exec_masks())
          for (unsigned alias = 0; alias < 4; ++alias) {
            uint32_t storage[4][34];
            std::memcpy(storage, original, sizeof(storage));
            const uint32_t *a[] = {storage[0] + 1}, *b[] = {storage[1] + 1},
                           *c[] = {storage[2] + 1};
            uint32_t *d[] = {storage[alias] + 1};
            ASSERT_EQ(functions[op](cpu | GOC_FP16_OVFL | (mode & 1 ? exact : 0), exec_mask, mode,
                                    d, a, b, c),
                      GOC_SUCCESS);
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned word = 0; word < 34; ++word) {
                uint32_t want =
                    reg == alias && word > 0 && word < 33 && ((exec_mask >> (word - 1)) & 1)
                        ? result[word - 1]
                        : original[reg][word];
                ASSERT_EQ(storage[reg][word], want)
                    << op << "/" << mode << "/" << cpu << "/" << alias;
              }
          }
    }
}

TEST(Cube, SharedSourceAliases) {
  for (unsigned op = 0; op < 4; ++op)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (unsigned breg = 0; breg < 2; ++breg)
        for (unsigned creg = 0; creg < 3; ++creg)
          for (unsigned dreg = 0; dreg < 4; ++dreg)
            for (uint32_t mode : goc_test::cube_capture_modes) {
              uint32_t data[4][32], expected[32];
              for (unsigned reg = 0; reg < 4; ++reg)
                for (unsigned lane = 0; lane < 32; ++lane)
                  data[reg][lane] = goc_test::cube_capture_values[(lane + reg * 5) % 16];
              for (unsigned lane = 0; lane < 32; ++lane)
                expected[lane] = goc_test::cube_reference(op, data[0][lane], data[breg][lane],
                                                          data[creg][lane], mode);
              const uint32_t *a[] = {data[0]}, *b[] = {data[breg]}, *c[] = {data[creg]};
              uint32_t *d[] = {data[dreg]};
              ASSERT_EQ(functions[op](cpu | exact, UINT32_MAX, mode, d, a, b, c), GOC_SUCCESS);
              for (unsigned lane = 0; lane < 32; ++lane)
                ASSERT_EQ(data[dreg][lane], expected[lane]);
            }
}

TEST(Cube, HostFloatingPointState) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    EXPECT_EQ(std::fesetround(rounding), 0);
    for (unsigned op = 0; op < 4; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint32_t mode = 0; mode < 512; ++mode) {
          uint32_t av[32], bv[32], cv[32], dv[32], expected[32];
          for (unsigned lane = 0; lane < 32; ++lane) {
            av[lane] = goc_test::cube_capture_values[lane % 16];
            bv[lane] = goc_test::cube_capture_values[(lane + 5) % 16];
            cv[lane] = goc_test::cube_capture_values[(lane + 9) % 16];
            expected[lane] = goc_test::cube_reference(op, av[lane], bv[lane], cv[lane], mode);
          }
          const uint32_t *a[] = {av}, *b[] = {bv}, *c[] = {cv};
          uint32_t *d[] = {dv};
          EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
          EXPECT_EQ(std::feraiseexcept(FE_INEXACT | FE_INVALID), 0);
          int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
          EXPECT_EQ(functions[op](cpu | exact, UINT32_MAX, mode, d, a, b, c), GOC_SUCCESS);
          EXPECT_EQ(std::fegetround(), rounding);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
          for (unsigned lane = 0; lane < 32; ++lane)
            EXPECT_EQ(dv[lane], expected[lane]);
        }
  }
}

TEST(Cube, ValidationAndSemanticFallback) {
  for (Fn fn : functions)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[32] = {}, output[32];
      std::fill_n(output, 32, 0xdeadbeef);
      const uint32_t *a[] = {input};
      uint32_t *d[] = {output};
      for (unsigned bit = 9; bit < 32; ++bit) {
        EXPECT_EQ(fn(cpu, UINT32_MAX, 1u << bit, d, a, a, a), GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(fn(cpu, 0, 1u << bit, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
      }
      EXPECT_EQ(fn(cpu | (1ull << 63), UINT32_MAX, 0, d, a, a, a), GOC_ERROR_INVALID_FLAGS);
      for (uint64_t semantics :
           {2 * GOC_SEMANTICS_EXACT_EMPIRICAL, 3 * GOC_SEMANTICS_EXACT_EMPIRICAL}) {
        EXPECT_EQ(fn(cpu | semantics | GOC_SEMANTICS_STRICT, UINT32_MAX, 0, d, a, a, a),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
      for (auto word : output)
        EXPECT_EQ(word, 0xdeadbeef);
      EXPECT_EQ(fn(cpu | exact, 0U, 511, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
      EXPECT_EQ(fn(cpu, 0, 0, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
      EXPECT_EQ(fn(cpu | (2 * GOC_SEMANTICS_EXACT_EMPIRICAL), UINT32_MAX, 0, d, a, a, a),
                GOC_SUCCESS);
    }
}

// DPP permutes X before the source modifiers; Y and Z retain their lanes.
TEST(Cube, DppModifiersMasksAliasesAndGuards) {
  std::mt19937 random(615739);
  uint32_t initial[4][34];
  for (auto &reg : initial)
    for (auto &word : reg)
      word = random();
  unsigned op = 0;
  for (auto fn : functions) {
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint32_t m = 0; m < 512; ++m)
        for (auto descriptor : goc_test::dpp_modes) {
          bool endpoints = m == 0 || m == 511;
          auto masks = endpoints ? exec_masks() : std::vector<uint32_t>{UINT32_MAX};
          for (auto exec_mask : masks)
            for (unsigned sharing = 0; sharing < (endpoints ? 4u : 1u); ++sharing)
              for (unsigned target = 0; target < 4; ++target) {
                unsigned br = sharing == 1 || sharing == 2 ? 0 : 1;
                unsigned cr = sharing == 2 ? 0 : sharing == 3 ? 1 : 2;
                uint32_t words[4][34], expected[4][34];
                std::memcpy(words, initial, sizeof(words));
                std::memcpy(expected, initial, sizeof(expected));
                for (unsigned lane = 0; lane < 32; ++lane) {
                  int source = 0;
                  if (goc_test::dpp_source(descriptor, exec_mask, lane, source))
                    expected[target][lane + 1] =
                        goc_test::cube_reference(op, source < 0 ? 0 : initial[0][source + 1],
                                                 initial[br][lane + 1], initial[cr][lane + 1], m);
                }
                const uint32_t *a[] = {words[0] + 1}, *b[] = {words[br] + 1},
                               *c[] = {words[cr] + 1};
                uint32_t *d[] = {words[target] + 1};
                uint64_t semantics =
                    m & 1 ? GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT : 0;
                ASSERT_EQ(fn(cpu | semantics, exec_mask, descriptor | m, d, a, b, c), GOC_SUCCESS);
                ASSERT_EQ(std::memcmp(words, expected, sizeof(words)), 0)
                    << op << "/" << cpu << "/" << m << "/" << descriptor << "/" << exec_mask;
              }
        }
    ++op;
  }
}

TEST(Cube, DppValidation) {
  for (auto fn : functions)
    for (auto descriptor : goc_test::dpp_modes) {
      EXPECT_EQ(fn(0, 0, descriptor, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
      for (auto invalid : {1ULL << 36, 1ULL << 9})
        EXPECT_EQ(fn(0, 0, descriptor | invalid, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
    }
}

// RX 9070: eight EXEC masks, four operations, 64 modifier combinations and
// seven DPP descriptors. Raw-word digest preserves NaN payloads and zero signs.
TEST(Cube, DppHardwareCorpusAndHostFpState) {
  const uint32_t values[] = {0x00000001, 0x807fffff, 0x00800000, 0x80000000,
                             0x7f800001, 0xff800000, 0x3f800000, 0xff7fffff};
  const uint32_t masks[] = {0xffffffff, 0,          0xaaaaaaaa, 0x55555555,
                            1,          0x80000000, 0xffff,     0xffff0000};
  const uint32_t signs[] = {0, 1, 2, 4, 8, 16, 32, 63};
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    std::fesetround(rounding);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INVALID | FE_INEXACT);
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, exact}) {
        uint64_t hash = goc_test::capture_hash_seed;
        for (auto exec_mask : masks)
          for (auto fn : functions)
            for (unsigned variant = 0; variant < 64; ++variant)
              for (auto descriptor : goc_test::dpp_modes) {
                uint32_t av[32], bv[32], cv[32], output[32];
                uint32_t mode = signs[variant % 8] | ((variant / 8) << 6);
                for (unsigned lane = 0; lane < 32; ++lane) {
                  av[lane] = values[lane % 8];
                  bv[lane] = values[(lane + 3) % 8];
                  cv[lane] = values[(lane + 5) % 8];
                  output[lane] = 0xdead0000u + lane;
                }
                const uint32_t *a[] = {av}, *b[] = {bv}, *c[] = {cv};
                uint32_t *d[] = {output};
                EXPECT_EQ(fn(cpu | semantics, exec_mask, descriptor | mode, d, a, b, c),
                          GOC_SUCCESS);
                for (auto word : output)
                  hash = goc_test::capture_hash_word(hash, word);
              }
        EXPECT_EQ(hash, 0xb1d0cbb608c5c005ULL) << cpu << "/" << semantics;
        EXPECT_EQ(std::fegetround(), rounding);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID | FE_INEXACT);
      }
  }
}

TEST(Cube, Wave64HardwareCartesianCorpus) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned column = 0; column < 20; ++column) {
      uint64_t digest = goc_test::capture_hash_seed;
      for (unsigned start = 0; start < 4096; start += 64) {
        uint32_t av[64], bv[64], cv[64], dv[64];
        for (unsigned lane = 0; lane < 64; ++lane) {
          unsigned index = start + lane;
          av[lane] = goc_test::cube_capture_values[index / 256];
          bv[lane] = goc_test::cube_capture_values[(index / 16) % 16];
          cv[lane] = goc_test::cube_capture_values[index % 16];
        }
        const uint32_t *a[] = {av}, *b[] = {bv}, *c[] = {cv};
        uint32_t *d[] = {dv};
        ASSERT_EQ(wave64_functions[column / 5](cpu | exact, UINT64_MAX,
                                               goc_test::cube_capture_modes[column % 5], d, a, b,
                                               c),
                  GOC_SUCCESS);
        for (auto word : dv)
          digest = goc_test::capture_hash_bytes(digest, word, 4);
      }
      EXPECT_EQ(digest, goc_test::cube_capture_digests[column]) << cpu << "/" << column;
    }
}

TEST(Cube, Wave64AxisTiesAtEveryMaskedLaneWithAlias) {
  // Equal magnitudes select Z before Y before X, including negative faces.
  const uint32_t inputs[][3] = {{0x3f800000, 0xbf800000, 0x3f800000},
                                {0xbf800000, 0x3f800000, 0xbf800000},
                                {0x3f800000, 0xbf800000, 0},
                                {0xbf800000, 0, 0}};
  const uint32_t faces[] = {0x40800000, 0x40a00000, 0x40400000, 0x3f800000};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (unsigned selected = 0; selected < 64; ++selected) {
      uint32_t words[3][64];
      for (unsigned lane = 0; lane < 64; ++lane)
        for (unsigned reg = 0; reg < 3; ++reg)
          words[reg][lane] = inputs[lane % 4][reg];
      const uint32_t *a[] = {words[0]}, *b[] = {words[1]}, *c[] = {words[2]};
      uint32_t *d[] = {words[0]};
      ASSERT_EQ(goc_v_cubeid_f32_rdna4_wave64(cpu | exact, 1ULL << selected, 0, d, a, b, c),
                GOC_SUCCESS);
      for (unsigned lane = 0; lane < 64; ++lane)
        EXPECT_EQ(words[0][lane], lane == selected ? faces[lane % 4] : inputs[lane % 4][0]);
    }
}

TEST(Cube, Wave64EveryModifierSpecialValuesAndRandomBits) {
  std::mt19937 random(998143);
  for (unsigned op = 0; op < 4; ++op)
    for (uint32_t mode = 0; mode < 512; ++mode)
      for (unsigned block = 0; block < 16; ++block) {
        uint32_t av[64], bv[64], cv[64], dv[64], expected[64];
        for (unsigned lane = 0; lane < 64; ++lane) {
          av[lane] = block < 8 ? goc_test::cube_capture_values[(lane + block) % 16] : random();
          bv[lane] =
              block < 8 ? goc_test::cube_capture_values[(lane * 3 + block * 2) % 16] : random();
          cv[lane] =
              block < 8 ? goc_test::cube_capture_values[(lane * 7 + block * 3) % 16] : random();
          expected[lane] = goc_test::cube_reference(op, av[lane], bv[lane], cv[lane], mode);
        }
        const uint32_t *a[] = {av}, *b[] = {bv}, *c[] = {cv};
        uint32_t *d[] = {dv};
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          ASSERT_EQ(wave64_functions[op](cpu | exact, UINT64_MAX, mode, d, a, b, c), GOC_SUCCESS);
          for (unsigned lane = 0; lane < 64; ++lane)
            ASSERT_EQ(dv[lane], expected[lane])
                << op << "/" << mode << "/" << cpu << "/" << std::hex << av[lane] << "/" << bv[lane]
                << "/" << cv[lane];
        }
      }
}

TEST(Cube, Wave64EveryModifierMaskAndDestinationAlias) {
  std::mt19937 random(55919);
  for (unsigned op = 0; op < 4; ++op)
    for (uint32_t mode = 0; mode < 512; ++mode) {
      uint32_t original[4][66], result[64];
      for (auto &reg : original)
        for (auto &word : reg)
          word = random();
      for (unsigned lane = 0; lane < 64; ++lane)
        result[lane] = goc_test::cube_reference(op, original[0][lane + 1], original[1][lane + 1],
                                                original[2][lane + 1], mode);
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint64_t exec_mask : {uint64_t(UINT64_MAX), uint64_t(0xffffffff00000000ULL),
                                   uint64_t(0x87654321fedcba98ULL)})
          for (unsigned alias = 0; alias < 4; ++alias) {
            uint32_t storage[4][66];
            std::memcpy(storage, original, sizeof(storage));
            const uint32_t *a[] = {storage[0] + 1}, *b[] = {storage[1] + 1},
                           *c[] = {storage[2] + 1};
            uint32_t *d[] = {storage[alias] + 1};
            ASSERT_EQ(wave64_functions[op](cpu | GOC_FP16_OVFL | (mode & 1 ? exact : 0), exec_mask,
                                           mode, d, a, b, c),
                      GOC_SUCCESS);
            for (unsigned reg = 0; reg < 4; ++reg)
              for (unsigned word = 0; word < 66; ++word) {
                uint32_t want =
                    reg == alias && word > 0 && word < 65 && ((exec_mask >> (word - 1)) & 1)
                        ? result[word - 1]
                        : original[reg][word];
                ASSERT_EQ(storage[reg][word], want)
                    << op << "/" << mode << "/" << cpu << "/" << alias;
              }
          }
    }
}

TEST(Cube, Wave64HostFloatingPointState) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    EXPECT_EQ(std::fesetround(rounding), 0);
    for (unsigned op = 0; op < 4; ++op)
      for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
        for (uint32_t mode = 0; mode < 512; ++mode) {
          uint32_t av[64], bv[64], cv[64], dv[64], expected[64];
          for (unsigned lane = 0; lane < 64; ++lane) {
            av[lane] = goc_test::cube_capture_values[lane % 16];
            bv[lane] = goc_test::cube_capture_values[(lane + 5) % 16];
            cv[lane] = goc_test::cube_capture_values[(lane + 9) % 16];
            expected[lane] = goc_test::cube_reference(op, av[lane], bv[lane], cv[lane], mode);
          }
          const uint32_t *a[] = {av}, *b[] = {bv}, *c[] = {cv};
          uint32_t *d[] = {dv};
          EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
          EXPECT_EQ(std::feraiseexcept(FE_INEXACT | FE_INVALID), 0);
          int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
          EXPECT_EQ(wave64_functions[op](cpu | exact, UINT64_MAX, mode, d, a, b, c), GOC_SUCCESS);
          EXPECT_EQ(std::fegetround(), rounding);
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
          for (unsigned lane = 0; lane < 64; ++lane)
            EXPECT_EQ(dv[lane], expected[lane]);
        }
  }
}

TEST(Cube, Wave64ValidationAndSemanticFallback) {
  for (auto fn : wave64_functions)
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
      uint32_t input[64] = {}, output[64];
      std::fill_n(output, 64, 0xdeadbeef);
      const uint32_t *a[] = {input};
      uint32_t *d[] = {output};
      for (unsigned bit = 9; bit < 32; ++bit) {
        EXPECT_EQ(fn(cpu, UINT64_MAX, 1u << bit, d, a, a, a), GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(fn(cpu, 0, 1u << bit, nullptr, nullptr, nullptr, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
      }
      EXPECT_EQ(fn(cpu | (1ull << 63), UINT64_MAX, 0, d, a, a, a), GOC_ERROR_INVALID_FLAGS);
      for (uint64_t semantics :
           {2 * GOC_SEMANTICS_EXACT_EMPIRICAL, 3 * GOC_SEMANTICS_EXACT_EMPIRICAL}) {
        EXPECT_EQ(fn(cpu | semantics | GOC_SEMANTICS_STRICT, UINT64_MAX, 0, d, a, a, a),
                  GOC_ERROR_UNSUPPORTED_SEMANTICS);
      }
      for (auto word : output)
        EXPECT_EQ(word, 0xdeadbeef);
      EXPECT_EQ(fn(cpu | exact, 0U, 511, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
      EXPECT_EQ(fn(cpu, 0, 0, nullptr, nullptr, nullptr, nullptr), GOC_SUCCESS);
      EXPECT_EQ(fn(cpu | (2 * GOC_SEMANTICS_EXACT_EMPIRICAL), UINT64_MAX, 0, d, a, a, a),
                GOC_SUCCESS);
    }
}

// DPP permutes X before the source modifiers; Y and Z retain their lanes.
TEST(Cube, Wave64DppModifiersMasksAliasesAndGuards) {
  std::mt19937 random(615739);
  uint32_t initial[4][66];
  for (auto &reg : initial)
    for (auto &word : reg)
      word = random();
  unsigned op = 0;
  for (auto fn : wave64_functions) {
    for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
      for (uint32_t m = 0; m < 512; ++m)
        for (auto descriptor_base : goc_test::dpp_modes) {
          uint64_t descriptor = descriptor_base;
          if (descriptor & GOC_DPP16)
            descriptor = (descriptor & ~GOC_DPP_ROW_MASK) | (uint64_t(m & 15) << GOC_DPP_ROW_SHIFT);
          bool endpoints = m == 0 || m == 511;
          std::vector<uint64_t> masks =
              endpoints
                  ? std::vector<uint64_t>{UINT64_MAX, 0xffffffff00000000ULL, 0x87654321fedcba98ULL}
                  : std::vector<uint64_t>{UINT64_MAX};
          for (auto exec_mask : masks)
            for (unsigned sharing = 0; sharing < (endpoints ? 4u : 1u); ++sharing)
              for (unsigned target = 0; target < 4; ++target) {
                unsigned br = sharing == 1 || sharing == 2 ? 0 : 1;
                unsigned cr = sharing == 2 ? 0 : sharing == 3 ? 1 : 2;
                uint32_t words[4][66], expected[4][66];
                std::memcpy(words, initial, sizeof(words));
                std::memcpy(expected, initial, sizeof(expected));
                for (unsigned lane = 0; lane < 64; ++lane) {
                  int source = 0;
                  if (goc_test::dpp_source(descriptor, exec_mask, lane, source))
                    expected[target][lane + 1] =
                        goc_test::cube_reference(op, source < 0 ? 0 : initial[0][source + 1],
                                                 initial[br][lane + 1], initial[cr][lane + 1], m);
                }
                const uint32_t *a[] = {words[0] + 1}, *b[] = {words[br] + 1},
                               *c[] = {words[cr] + 1};
                uint32_t *d[] = {words[target] + 1};
                uint64_t semantics =
                    m & 1 ? GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT : 0;
                ASSERT_EQ(fn(cpu | semantics, exec_mask, descriptor | m, d, a, b, c), GOC_SUCCESS);
                ASSERT_EQ(std::memcmp(words, expected, sizeof(words)), 0)
                    << op << "/" << cpu << "/" << m << "/" << descriptor << "/" << exec_mask;
              }
        }
    ++op;
  }
}

TEST(Cube, ArchitectureIeeeDenormalAndNanPolicies) {
  // Literal unmodified results from rocjitsu shared/cube.h's selection rules.
  // Columns are ID, SC, TC, MA, before NaN quieting, OMOD and CLAMP.
  struct Case {
    uint32_t a, b, c, result[4];
  };

  const Case cases[] = {
      {0x7f800123, 0x3f800000, 0x40000000, {0, 0xc0000000, 0xbf800000, 0x7f800123}},
      {0x3f800000, 0x7f800123, 0x40000000, {0, 0xc0000000, 0xff800123, 0x40000000}},
      {0x3f800000, 0x40000000, 0x7f800123, {0x40000000, 0x3f800000, 0x7f800123, 0x40800000}},
      {0, 1, 0, {0x40800000, 0, 0x80000001, 0}},
      {0x3f800000, 0x40000000, 0xc0400000, {0x40a00000, 0xbf800000, 0xc0000000, 0xc0c00000}},
      {0x7f7fffff, 0, 0, {0, 0x80000000, 0x80000000, 0x7f800000}},
  };
  const Fn narrow[][4] = {{goc_v_cubeid_f32, goc_v_cubesc_f32, goc_v_cubetc_f32, goc_v_cubema_f32},
                          {goc_v_cubeid_f32_rdna4, goc_v_cubesc_f32_rdna4, goc_v_cubetc_f32_rdna4,
                           goc_v_cubema_f32_rdna4}};
  const decltype(&goc_v_cubeid_f32_wave64) wide[][4] = {
      {goc_v_cubeid_f32_wave64, goc_v_cubesc_f32_wave64, goc_v_cubetc_f32_wave64,
       goc_v_cubema_f32_wave64},
      {goc_v_cubeid_f32_rdna4_wave64, goc_v_cubesc_f32_rdna4_wave64, goc_v_cubetc_f32_rdna4_wave64,
       goc_v_cubema_f32_rdna4_wave64}};
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t semantics : {uint64_t(0), exact})
      for (unsigned arch = 0; arch < 2; ++arch)
        for (unsigned op = 0; op < 4; ++op)
          for (unsigned width : {32U, 64U})
            for (unsigned settings = 0; settings < (arch ? 8U : 16U); ++settings)
              for (unsigned scale = 0; scale < 4; ++scale)
                for (bool clamp : {false, true})
                  for (unsigned alias : {0U, 3U})
                    for (uint64_t route : {uint64_t(0), goc_test::dpp_modes[1]})
                      for (const auto &test : cases) {
                        SCOPED_TRACE(::testing::Message()
                                     << cpu << '/' << semantics << '/' << arch << '/' << op << '/'
                                     << width << '/' << settings << '/' << scale << '/' << clamp
                                     << '/' << alias << '/' << route << '/' << test.a << '/'
                                     << test.b << '/' << test.c);
                        uint64_t flags = cpu | semantics | (settings & 1 ? GOC_FP_IEEE_MODE : 0) |
                                         (settings & 2 ? GOC_FP_FLUSH_INPUT_DENORMALS : 0) |
                                         (settings & 4 ? GOC_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                                         (settings & 8 ? GOC_FP_DX10_CLAMP_OFF : 0);
                        uint64_t mode = route | (scale << 6) | (clamp ? GOC_ALU_CLAMP : 0);
                        uint64_t exec_mask = 0xa55aa55a5aa55aa5ULL;
                        uint32_t words[4][64];
                        std::fill(words[0], words[0] + 64, test.a);
                        std::fill(words[1], words[1] + 64, test.b);
                        std::fill(words[2], words[2] + 64, test.c);
                        std::fill(words[3], words[3] + 64, 0xdeadbeefU);
                        const uint32_t *a = words[0], *b = words[1], *c = words[2];
                        uint32_t *d = words[alias];
                        ASSERT_EQ(
                            width == 32
                                ? narrow[arch][op](flags, uint32_t(exec_mask), mode, &d, &a, &b, &c)
                                : wide[arch][op](flags, exec_mask, mode, &d, &a, &b, &c),
                            GOC_SUCCESS);
                        uint32_t want = test.result[op];
                        if ((arch || (settings & 1)) && (want & 0x7fffffffU) > 0x7f800000U)
                          want |= 0x00400000;
                        unsigned effective_scale = !arch && (settings & 1) ? 0 : scale;
                        if (effective_scale) {
                          if ((want & 0x7fffffffU) < 0x00800000U)
                            want = 0;
                          else if ((want & 0x7fffffffU) < 0x7f800000U)
                            want = goc_test::cube_bits(
                                std::ldexp(goc_test::cube_float(want),
                                           effective_scale == 3 ? -1 : int(effective_scale)));
                        }
                        if (clamp && (!(settings & 8) || (want & 0x7fffffffU) <= 0x7f800000U)) {
                          if ((want >> 31) || want > 0x7f800000U)
                            want = 0;
                          else if (want > 0x3f800000U)
                            want = 0x3f800000U;
                        }
                        for (unsigned lane = 0; lane < 64; ++lane)
                          EXPECT_EQ(d[lane], lane < width && ((exec_mask >> lane) & 1) ? want
                                             : alias == 0                              ? test.a
                                                          : 0xdeadbeefU);
                      }
}

TEST(Cube, Dx10ClampFlagAvailability) {
  // RDNA4 has no DX10_CLAMP control; reject the RDNA3-only flag before loads.
  for (auto fn : functions)
    EXPECT_EQ(fn(GOC_FP_DX10_CLAMP_OFF, 1, GOC_ALU_CLAMP, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
  for (auto fn : wave64_functions)
    EXPECT_EQ(
        fn(GOC_FP_DX10_CLAMP_OFF, 1ULL << 63, GOC_ALU_CLAMP, nullptr, nullptr, nullptr, nullptr),
        GOC_ERROR_INVALID_FLAGS);
}
