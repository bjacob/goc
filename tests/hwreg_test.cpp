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

uint64_t pair(const std::array<uint32_t, 64> &values, unsigned low) {
  return values[low] | (uint64_t(values[low + 1]) << 32);
}

int read(bool rdna4, uint64_t flags, uint64_t instruction_flags, uint16_t hwreg, uint32_t *d,
         const std::array<uint32_t, 64> &v, std::array<uint32_t, 5> *snapshot = nullptr,
         unsigned report = 31) {
  if (rdna4)
    return goc_s_getreg_b32_rdna4(flags, instruction_flags, d, hwreg, v[1], v[2], v[4], v[5], v[6],
                                  v[10], v[11], v[12], v[15], v[16], v[17], v[18], v[19],
                                  pair(v, 20), v[23], v[24], v[28], pair(v, 29),
                                  snapshot && (report & 1) ? &(*snapshot)[0] : nullptr,
                                  snapshot && (report & 2) ? &(*snapshot)[1] : nullptr,
                                  snapshot && (report & 4) ? &(*snapshot)[2] : nullptr,
                                  snapshot && (report & 8) ? &(*snapshot)[3] : nullptr,
                                  snapshot && (report & 16) ? &(*snapshot)[4] : nullptr);
  return goc_s_getreg_b32(flags, instruction_flags, d, hwreg, v[1], v[2], v[3], v[5], v[6], v[7],
                          pair(v, 8), v[13], v[15], pair(v, 16), pair(v, 18), pair(v, 20), v[23],
                          v[24], v[25], v[26], v[27], v[28], pair(v, 29));
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
      ASSERT_EQ(goc_s_setreg_b32(exact, 0, descriptor(1, 0, 32), value, 0, &mode, nullptr, nullptr,
                                 nullptr, nullptr),
                GOC_SUCCESS);
      EXPECT_EQ(mode, value);
      std::array<uint32_t, 64> registers{};
      registers[1] = mode;
      for (bool rdna4 : {false, true}) {
        uint32_t output = 0;
        ASSERT_EQ(read(rdna4, exact, 0, descriptor(1, 0, 32), &output, registers), GOC_SUCCESS);
        EXPECT_EQ(output, mode);
      }
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
#if defined(__x86_64__) || defined(_M_X64)
      EXPECT_EQ(_mm_getcsr(), controls);
#endif
    }
  }
}

TEST(Hwreg, GetregEveryIdAndField) {
  // IDs come from each architecture's XML plus RDNA4's prose-only IB_STS2.
  // RDNA3 FLUSH_IB (14) is write-only and is deliberately not a read selector.
  const unsigned rdna3_ids[] = {1,  2,  3,  5,  6,  7,  8,  9,  13, 15, 16, 17,
                                18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29};
  const unsigned rdna4_ids[] = {1,  2,  4,  5,  6,  10, 11, 12, 15, 16,
                                17, 18, 19, 20, 21, 23, 24, 28, 29, 30};
  for (bool rdna4 : {false, true}) {
    std::array<bool, 64> valid{};
    if (rdna4) {
      for (unsigned id : rdna4_ids)
        valid[id] = true;
    } else {
      for (unsigned id : rdna3_ids)
        valid[id] = true;
    }
    for (uint64_t flags : std::initializer_list<uint64_t>{0, exact})
      for (unsigned id = 0; id < 64; ++id)
        for (unsigned offset = 0; offset < 32; ++offset)
          for (unsigned width = 1; width <= 32; ++width) {
            std::array<uint32_t, 64> values;
            for (unsigned reg = 0; reg < values.size(); ++reg)
              values[reg] = (0xdeadbeefU * (reg + 1)) ^ (offset * 0x1234567U + width * 0x87654321U);
            uint32_t expected = 0;
            for (unsigned bit = 0; bit < width && offset + bit < 32; ++bit)
              expected |= ((values[id] >> (offset + bit)) & 1) << bit;
            uint32_t output = 0xfeedface;
            ASSERT_EQ(read(rdna4, flags, 0, descriptor(id, offset, width), &output, values),
                      valid[id] ? GOC_SUCCESS : GOC_ERROR_INVALID_FLAGS);
            EXPECT_EQ(output, valid[id] ? expected : 0xfeedfaceU)
                << rdna4 << '/' << id << '/' << offset << '/' << width;
          }
  }
}

TEST(Hwreg, GetregHardwareCapture) {
  const uint32_t sources[] = {0, UINT32_MAX, 0xa5a5a5a5, 0x5a5a5a5a, UINT32_MAX};
  const unsigned ids[] = {1, 18};
  const uint32_t writable[] = {0x098000ff, 0xc000007f};
  for (unsigned reg = 0; reg < 2; ++reg)
    for (unsigned seed = 0; seed < 2; ++seed)
      for (unsigned pattern = 0; pattern < 5; ++pattern) {
        uint64_t hash = goc_test::capture_hash_seed;
        for (unsigned offset = 0; offset < 32; ++offset)
          for (unsigned width = 1; width <= 32; ++width) {
            std::array<uint32_t, 64> values{};
            values[ids[reg]] =
                insert(seed ? writable[reg] : 0, sources[pattern], offset, width, writable[reg]);
            uint32_t output = 0;
            ASSERT_EQ(read(true, exact, 0, descriptor(ids[reg], offset, width), &output, values),
                      GOC_SUCCESS);
            hash = goc_test::capture_hash_word(hash, output);
          }
        EXPECT_EQ(hash, goc_test::hwreg_read_hashes[reg][seed][pattern]);
      }
}

TEST(Hwreg, GetregSnapshotsAndOptionalSideEffects) {
  std::array<uint32_t, 64> values{};
  for (unsigned id : {10, 11, 12, 15, 16})
    values[id] = id * 0x9876543U;
  for (uint64_t flags : std::initializer_list<uint64_t>{0, exact, GOC_SEMANTICS_EXACT_EMPIRICAL})
    for (unsigned report = 0; report < 32; ++report)
      for (unsigned id : {10, 11, 12, 15, 16}) {
        std::array<uint32_t, 5> state = {0x12345678, 0x98765432, 0x55555555, 0xaaaaaaaa,
                                         0xdeadbeef};
        auto before = state;
        uint32_t output = 0xfeedface;
        bool unsupported = flags && report && id == 12;
        EXPECT_EQ(read(true, flags, 0, descriptor(id, 0, 32), &output, values, &state, report),
                  unsupported ? GOC_ERROR_UNSUPPORTED_GLOBAL_STATE : GOC_SUCCESS);
        EXPECT_EQ(output, unsupported ? 0xfeedfaceU : values[id]);
        EXPECT_EQ(state, before);
      }
  // A wave without snapshot visibility supplies zero read views.
  values = {};
  for (unsigned id : {10, 11, 12, 15, 16}) {
    uint32_t output = UINT32_MAX;
    ASSERT_EQ(read(true, exact, 0, descriptor(id, 0, 32), &output, values), GOC_SUCCESS);
    EXPECT_EQ(output, 0U);
  }
  std::array<uint32_t, 5> state{};
  EXPECT_EQ(read(true, exact, 1ULL << 63, 12, nullptr, values, &state),
            GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
}

TEST(Hwreg, GetregClockHalvesAliasesAndErrors) {
  for (bool rdna4 : {false, true}) {
    std::array<uint32_t, 64> values{};
    for (uint64_t clock : {0xffffffffULL, 0x100000000ULL, 0x123456789abcdef0ULL}) {
      values[29] = uint32_t(clock);
      values[30] = uint32_t(clock >> 32);
      uint32_t output = 0;
      ASSERT_EQ(read(rdna4, exact, 0, descriptor(29, 0, 32), &output, values), GOC_SUCCESS);
      EXPECT_EQ(output, uint32_t(clock));
      if (rdna4) {
        ASSERT_EQ(read(true, exact, 0, descriptor(30, 0, 32), &output, values), GOC_SUCCESS);
        EXPECT_EQ(output, clock >> 32);
      }
    }
    values[1] = 0x12345678;
    ASSERT_EQ(read(rdna4, exact, 0, descriptor(1, 4, 8), &values[1], values), GOC_SUCCESS);
    EXPECT_EQ(values[1], 0x67U);
    for (unsigned bit = 0; bit < 64; ++bit)
      EXPECT_EQ(read(rdna4, exact, 1ULL << bit, 1, &values[1], values), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(read(rdna4, 1ULL << 63, 0, 1, &values[1], values), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(read(rdna4, 2 * GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 1,
                   &values[1], values),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(read(rdna4, exact, 0, 0, nullptr, values), GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(values[1], 0x67U);
  }
}

TEST(Hwreg, Rdna3WritesEveryIdFieldAndPrivilege) {
  // Build the reference masks from the manual's fields, independently of the
  // implementation's hexadecimal constants. MODE.EXCP_EN[8] is reserved.
  const uint32_t mode_writable = ((1U << 20) - 1) | (1U << 21) | (1U << 23) | (1U << 27);
  const uint32_t status_writable = 31U | (1U << 8) | (1U << 13) | (1U << 18) | (1U << 20) |
                                   (1U << 21) | (1U << 25) | (1U << 27) | (1U << 29);
  const uint32_t trapsts_writable = 511U | (511U << 10) | (1U << 20);
  for (auto fn : {goc_s_setreg_b32, goc_s_setreg_imm32_b32})
    for (uint64_t flags : std::initializer_list<uint64_t>{0, exact})
      for (bool privileged : {false, true})
        for (unsigned id = 0; id < 64; ++id)
          for (unsigned offset = 0; offset < 32; ++offset)
            for (unsigned width = 1; width <= 32; ++width) {
              std::array<uint32_t, 4> state = {0x12345678, 0x89abcdc0U | (privileged ? 32U : 0U),
                                               0x55555555, 0xaaaaaaaa};
              auto expected = state;
              uint64_t scratch = 0x123456789abcde00ULL;
              uint32_t source = (id * 0x1234567U + offset * 0x7654321U) ^ (width * 0xdeadbeefU);
              bool known = true, unsupported = false;
              switch (id) {
              case 1:
                expected[0] = insert(state[0], source, offset, width, mode_writable);
                break;
              case 2:
                expected[1] =
                    insert(state[1], source, offset, width, privileged ? status_writable : 0);
                break;
              case 3:
                expected[2] = insert(state[2], source, offset, width, trapsts_writable);
                break;
              case 14:
                unsupported = flags;
                break;
              case 20:
              case 21:
                unsupported = flags && privileged;
                break;
              case 5:
              case 6:
              case 7:
              case 8:
              case 9:
              case 13:
              case 15:
              case 16:
              case 17:
              case 18:
              case 19:
              case 23:
              case 24:
              case 25:
              case 26:
              case 27:
              case 28:
              case 29:
                break;
              default:
                known = false;
                break;
              }
              ASSERT_EQ(fn(flags, 0, descriptor(id, offset, width), source, state[1], &state[0],
                           &state[1], &state[2], &state[3], &scratch),
                        unsupported ? GOC_ERROR_UNSUPPORTED_GLOBAL_STATE
                        : known     ? GOC_SUCCESS
                                    : GOC_ERROR_INVALID_FLAGS)
                  << id << '/' << offset << '/' << width;
              EXPECT_EQ(state, expected);
              EXPECT_EQ(scratch, 0x123456789abcde00ULL);
            }
}

TEST(Hwreg, Rdna3NullOutputsAndUnimplementedState) {
  const unsigned known[] = {1,  2,  3,  5,  6,  7,  8,  9,  13, 14, 15, 16,
                            17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29};
  for (auto fn : {goc_s_setreg_b32, goc_s_setreg_imm32_b32}) {
    for (unsigned id : known)
      EXPECT_EQ(fn(exact, 0, descriptor(id, 0, 32), UINT32_MAX, 32, nullptr, nullptr, nullptr,
                   nullptr, nullptr),
                GOC_SUCCESS);
    for (uint64_t flags : std::initializer_list<uint64_t>{0, exact, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (unsigned id : {14, 20, 21})
        for (uint32_t privilege : {0U, 32U})
          for (unsigned pointers = 0; pointers < 8; ++pointers) {
            uint32_t status = privilege, flush = 0xdeadbeef;
            uint64_t scratch = 0x123456789abcde00ULL;
            bool requested = id == 14 ? (pointers & 1) : privilege && (pointers & 6);
            EXPECT_EQ(fn(flags, 0, descriptor(id, 0, 32), 0, privilege, nullptr,
                         pointers & 2 ? &status : nullptr, nullptr, pointers & 1 ? &flush : nullptr,
                         pointers & 4 ? &scratch : nullptr),
                      flags && requested ? GOC_ERROR_UNSUPPORTED_GLOBAL_STATE : GOC_SUCCESS);
            EXPECT_EQ(status, privilege);
            EXPECT_EQ(flush, 0xdeadbeefU);
            EXPECT_EQ(scratch, 0x123456789abcde00ULL);
          }
    uint32_t mode = 0xdeadbeef;
    for (unsigned bit = 0; bit < 64; ++bit)
      EXPECT_EQ(fn(exact, 1ULL << bit, 1, 0, 0, &mode, nullptr, nullptr, nullptr, nullptr),
                GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(1ULL << 63, 0, 1, 0, 0, &mode, nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_INVALID_FLAGS);
    EXPECT_EQ(fn(2 * GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT, 0, 1, 0, 0, &mode,
                 nullptr, nullptr, nullptr, nullptr),
              GOC_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(mode, 0xdeadbeefU);
    uint32_t flush = 0x12345678;
    EXPECT_EQ(fn(exact, UINT64_MAX, 14, 0, 0, nullptr, nullptr, nullptr, &flush, nullptr),
              GOC_ERROR_UNSUPPORTED_GLOBAL_STATE);
    EXPECT_EQ(flush, 0x12345678U);
  }
}

TEST(Hwreg, ArchitectureSpecificWritableFieldsAndExceptionClearing) {
  for (uint32_t source : {0U, UINT32_MAX}) {
    uint32_t mode3 = ~source, mode4 = ~source;
    ASSERT_EQ(goc_s_setreg_b32(exact, 0, descriptor(1, 0, 32), source, 0, &mode3, nullptr, nullptr,
                               nullptr, nullptr),
              GOC_SUCCESS);
    ASSERT_EQ(goc_s_setreg_b32_rdna4(exact, 0, descriptor(1, 0, 32), source, 0, &mode4, nullptr,
                                     nullptr, nullptr, nullptr, nullptr),
              GOC_SUCCESS);
    EXPECT_EQ(mode3, (~source & ~0x08afffffU) | (source & 0x08afffffU));
    EXPECT_EQ(mode4, (~source & ~0x098000ffU) | (source & 0x098000ffU));
    EXPECT_EQ((mode3 >> 20) & 1, (~source >> 20) & 1); // RDNA3 reserved EXCP_EN bit.
    EXPECT_NE(mode3, mode4);
  }
  uint32_t trapsts = UINT32_MAX;
  ASSERT_EQ(goc_s_setreg_b32(exact, 0, descriptor(3, 0, 9), 0, 0, nullptr, nullptr, &trapsts,
                             nullptr, nullptr),
            GOC_SUCCESS);
  EXPECT_EQ(trapsts, 0xfffffe00U);
  uint32_t status = 32;
  ASSERT_EQ(goc_s_setreg_b32(exact, 0, descriptor(2, 0, 32), UINT32_MAX, status, nullptr, &status,
                             nullptr, nullptr, nullptr),
            GOC_SUCCESS);
  EXPECT_EQ(status, 32U | 0x2a34211fU);
  // Guest privilege is not writable through STATUS, even from the trap handler.
  ASSERT_EQ(goc_s_setreg_b32(exact, 0, descriptor(2, 0, 32), 0, status, nullptr, &status, nullptr,
                             nullptr, nullptr),
            GOC_SUCCESS);
  EXPECT_EQ(status, 32U);
}
