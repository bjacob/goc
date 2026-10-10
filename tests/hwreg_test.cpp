// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "hwreg_hardware.h"

#include <array>
#include <cfenv>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

uint16_t descriptor(unsigned id, unsigned offset, unsigned width) {
  return uint16_t(id | (offset << 6) | ((width - 1) << 11));
}

// Reference inserts individual source bits; no shifted field-mask formula.
uint64_t insert(uint64_t old, uint32_t source, unsigned offset, unsigned width, uint64_t writable) {
  for (unsigned bit = 0; bit < width && bit + offset < 32; ++bit) {
    uint64_t target = 1ULL << (bit + offset);
    if (writable & target)
      old = (old & ~target) | (((source >> bit) & 1) ? target : 0);
  }
  return old;
}

} // namespace

TEST(Hwreg, CapturedUserWrites) {
  const uint32_t sources[] = {0, UINT32_MAX, 0xa5a5a5a5, 0x5a5a5a5a, UINT32_MAX};
  const unsigned ids[] = {1, 18};
  const uint32_t writable[] = {0x098000ff, 0xc000007f};
  for (uint64_t flags : std::initializer_list<uint64_t>{0, exact})
    for (unsigned reg = 0; reg < 2; ++reg)
      for (unsigned seed = 0; seed < 2; ++seed)
        for (unsigned pattern = 0; pattern < 5; ++pattern) {
          auto fn = pattern == 4 ? goc_s_setreg_imm32_b32_rdna4 : goc_s_setreg_b32_rdna4;
          uint64_t hash = goc_test::capture_hash_seed;
          for (unsigned offset = 0; offset < 32; ++offset)
            for (unsigned width = 1; width <= 32; ++width) {
              uint32_t value = seed ? writable[reg] : 0;
              ASSERT_EQ(fn(flags, 0, descriptor(ids[reg], offset, width), sources[pattern], 0,
                           reg == 0 ? &value : nullptr, nullptr, nullptr,
                           reg == 1 ? &value : nullptr, nullptr, nullptr),
                        GOC_SUCCESS);
              hash = goc_test::capture_hash_word(hash, value);
            }
          EXPECT_EQ(hash, goc_test::hwreg_write_hashes[reg][seed][pattern]);
        }
}

TEST(Hwreg, EveryIdFieldPrivilegeAndPreservedBit) {
  for (auto fn : {goc_s_setreg_b32_rdna4, goc_s_setreg_imm32_b32_rdna4})
    for (bool privileged : {false, true})
      for (unsigned id = 0; id < 64; ++id)
        for (unsigned offset = 0; offset < 32; ++offset)
          for (unsigned width = 1; width <= 32; ++width) {
            SCOPED_TRACE(::testing::Message()
                         << id << '/' << offset << '/' << width << '/' << privileged);
            std::array<uint32_t, 5> state = {0x12345678, 0x89abcdef, 0x55555555, 0xaaaaaaaa,
                                             0xfedcba98};
            auto expected = state;
            uint64_t scratch = 0x123456789abcdef0ULL, expected_scratch = scratch;
            uint32_t source = (id * 0x1234567U + offset * 0x7654321U) ^ (width * 0xdeadbeefU);
            bool known = true, unsupported = false;
            switch (id) {
            case 1:
              expected[0] = insert(state[0], source, offset, width, 0x098000ff);
              break;
            case 4:
              expected[1] = insert(state[1], source, offset, width, privileged ? 0x47e07 : 0);
              break;
            case 17:
              expected[2] = insert(state[2], source, offset, width, privileged ? 0xc0000fff : 0);
              break;
            case 18:
              expected[3] = insert(state[3], source, offset, width, 0xc000007f);
              break;
            case 19:
              expected[4] = insert(state[4], source, offset, width, privileged ? 0x3ff : 0);
              break;
            case 20:
            case 21:
              unsupported = privileged;
              break;
            case 2:
            case 5:
            case 6:
            case 10:
            case 11:
            case 12:
            case 15:
            case 16:
            case 23:
            case 24:
            case 28:
            case 29:
            case 30:
              break;
            default:
              known = false;
              break;
            }
            EXPECT_EQ(fn(exact, 0, descriptor(id, offset, width), source, privileged ? 32 : 0,
                         &state[0], &state[1], &state[2], &state[3], &state[4], &scratch),
                      unsupported ? GOC_ERROR_UNSUPPORTED_GLOBAL_STATE
                      : known     ? GOC_SUCCESS
                                  : GOC_ERROR_INVALID_FLAGS);
            EXPECT_EQ(state, expected);
            EXPECT_EQ(scratch, expected_scratch);
          }
}

TEST(Hwreg, NullOutputsAndValidation) {
  for (auto fn : {goc_s_setreg_b32_rdna4, goc_s_setreg_imm32_b32_rdna4}) {
    for (uint64_t flags : std::initializer_list<uint64_t>{0, exact})
      for (unsigned id :
           {1, 2, 4, 5, 6, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 23, 24, 28, 29, 30})
        EXPECT_EQ(fn(flags, 0, descriptor(id, 0, 32), UINT32_MAX, 32, nullptr, nullptr, nullptr,
                     nullptr, nullptr, nullptr),
                  GOC_SUCCESS);
    uint32_t mode = 0xdeadbeef;
    for (unsigned bit = 0; bit < 64; ++bit)
      EXPECT_EQ(fn(exact, 1ULL << bit, 1, 0, 0, &mode, nullptr, nullptr, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(1ULL << 63, 0, 1, 0, 0, &mode, nullptr, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(2 * GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 1, 0, 0, &mode,
                 nullptr, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(mode, 0xdeadbeefU);
  }
}

TEST(Hwreg, UnsupportedScratchReportingIsOptionalAndNeverGuessed) {
  for (auto fn : {goc_s_setreg_b32_rdna4, goc_s_setreg_imm32_b32_rdna4})
    for (uint64_t flags : std::initializer_list<uint64_t>{0, exact, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (unsigned id : {20, 21})
        for (uint32_t status : {0U, 32U})
          for (unsigned pointers = 0; pointers < 4; ++pointers) {
            uint32_t state_priv = 0x12345678, mode = 0xdeadbeef;
            uint64_t scratch = 0x123456789abcde00ULL;
            bool unsupported = flags && status && pointers;
            EXPECT_EQ(fn(flags, 0, descriptor(id, 0, 32), UINT32_MAX, status, &mode,
                         pointers & 1 ? &state_priv : nullptr, nullptr, nullptr, nullptr,
                         pointers & 2 ? &scratch : nullptr),
                      unsupported ? GOC_ERROR_UNSUPPORTED_GLOBAL_STATE : GOC_SUCCESS);
            EXPECT_EQ(state_priv, 0x12345678U);
            EXPECT_EQ(mode, 0xdeadbeefU);
            EXPECT_EQ(scratch, 0x123456789abcde00ULL);
          }
}

TEST(Hwreg, GuestModeDoesNotChangeHostEnvironment) {
  goc_test::ScopedFpEnvironment saved;
  ASSERT_TRUE(saved.saved());
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(rounding), 0);
    ASSERT_EQ(std::feraiseexcept(FE_INVALID | FE_INEXACT), 0);
    int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
#if defined(__x86_64__) || defined(_M_X64)
    unsigned controls = _mm_getcsr();
#endif
    for (uint32_t value = 0; value < 256; ++value) {
      uint32_t mode = 0;
      ASSERT_EQ(goc_s_setreg_b32_rdna4(exact, 0, descriptor(1, 0, 32), value, 0, &mode, nullptr,
                                       nullptr, nullptr, nullptr, nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(mode, value);
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
#if defined(__x86_64__) || defined(_M_X64)
      EXPECT_EQ(_mm_getcsr(), controls);
#endif
    }
  }
}
