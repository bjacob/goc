// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_scalar_bits_hardware.h"
#include "rdna4_scalar_bits_reference.h"
#include "rdna4_scalar_integer_reference.h"

#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

TEST(ScalarBits, HardwareResultsAndScc) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (unsigned seed = 0; seed < 2; ++seed)
        for (uint32_t exec_mask : {0U, UINT32_MAX, 0xaaaaaaaaU})
          for (unsigned op = 0; op < 30; ++op) {
            uint64_t hash = goc_test::capture_hash_seed;
            for (unsigned i = 0; i < 4096; ++i) {
              uint32_t w[4];
              goc_test::scalar_integer_inputs(i, w);
              uint64_t a = (uint64_t(w[1]) << 32) | w[0], b = (uint64_t(w[3]) << 32) | w[2],
                       d64 = 0;
              uint32_t d = 0, cc = seed;
              ASSERT_EQ(goc_test::scalar_bits_call(op, cpu | semantics | GOC_SEMANTICS_STRICT,
                                                   exec_mask, 0, &d, &d64, a, b, &cc),
                        GOC_SUCCESS);
              bool wide = op < 26 && (op & 1);
              for (uint32_t word : {wide ? uint32_t(d64) : d, wide ? uint32_t(d64 >> 32) : 0u, cc})
                hash = goc_test::capture_hash_word(hash, word);
            }
            ASSERT_EQ(hash, goc_test::scalar_bits_hardware[seed][op])
                << cpu << "/" << semantics << "/" << seed << "/" << exec_mask << "/" << op;
          }
}

TEST(ScalarBits, ExecAndOverlappingScalarOutputs) {
  for (unsigned op = 0; op < 30; ++op)
    for (uint32_t exec_mask : rdna4_exec_masks()) {
      uint32_t d = 0, cc = 7;
      uint64_t wide = 0;
      const uint64_t a = 0x87654321abcdef01ULL, b = 0x1234567800000041ULL;
      ASSERT_EQ(goc_test::scalar_bits_call(op, 0, exec_mask, 0, &d, &wide, a, b, &cc), GOC_SUCCESS);
      if (op < 26 && (op & 1)) {
        for (unsigned word = 0; word < 2; ++word) {
          uint64_t storage[] = {123, a, 456}, expected = wide;
          auto bytes = reinterpret_cast<unsigned char *>(&storage[1]);
          auto status = reinterpret_cast<uint32_t *>(bytes + word * sizeof(uint32_t));
          if (op != 19)
            std::memcpy(reinterpret_cast<unsigned char *>(&expected) + word * sizeof(uint32_t), &cc,
                        sizeof(cc));
          ASSERT_EQ(goc_test::scalar_bits_call(op, 0, exec_mask, 0, nullptr, storage + 1,
                                               storage[1], b, status),
                    GOC_SUCCESS);
          EXPECT_EQ(storage[1], expected);
          EXPECT_EQ(storage[0], 123u);
          EXPECT_EQ(storage[2], 456u);
        }
      } else {
        uint32_t storage[] = {123, uint32_t(a), 456};
        ASSERT_EQ(goc_test::scalar_bits_call(op, 0, exec_mask, 0, storage + 1, nullptr, storage[1],
                                             b, storage + 1),
                  GOC_SUCCESS);
        EXPECT_EQ(storage[1], op == 18 ? d : cc);
        EXPECT_EQ(storage[0], 123u);
        EXPECT_EQ(storage[2], 456u);
      }
    }
}

TEST(ScalarBits, EveryShiftCountAndSign) {
  for (uint64_t a : std::initializer_list<uint64_t>{0ULL, 1ULL, 0x8000000080000000ULL, UINT64_MAX})
    for (uint32_t count = 0; count < 256; ++count) {
      uint32_t d, cc;
      uint64_t d64;
      for (unsigned op = 20; op < 26; ++op) {
        unsigned width = op & 1 ? 64 : 32, shift = count % width;
        uint64_t input = width == 64 ? a : uint32_t(a), result = 0;
        for (unsigned bit = 0; bit < width; ++bit) {
          int source = op < 22 ? int(bit) - int(shift) : int(bit) + int(shift);
          bool set = source < 0             ? false
                     : source >= int(width) ? op >= 24 && (input >> (width - 1))
                                            : ((input >> source) & 1);
          result |= uint64_t(set) << bit;
        }
        ASSERT_EQ(goc_test::scalar_bits_call(op, 0, 0, 0, &d, &d64, a, count, &cc), GOC_SUCCESS);
        EXPECT_EQ(width == 64 ? d64 : d, result);
        EXPECT_EQ(cc, unsigned(result != 0));
      }
    }
}

TEST(ScalarBits, ErrorsDoNotWriteAndHostFpStatePreserved) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    std::fesetround(rounding);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
    int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
    for (unsigned op = 0; op < 30; ++op) {
      uint32_t d = 123, cc = 456;
      uint64_t wide = 789;
      for (unsigned bit = 0; bit < 32; ++bit)
        EXPECT_EQ(goc_test::scalar_bits_call(op, 0, 0, 1u << bit, &d, &wide, 0, 0, &cc),
                  GOC_ERROR_INVALID_FLAGS);
      for (unsigned bit = 32; bit < 64; ++bit)
        EXPECT_EQ(goc_test::scalar_bits_call(op, 0, 0, 1ULL << bit, nullptr, nullptr, UINT64_MAX,
                                             63, nullptr),
                  GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(d, 123u);
      EXPECT_EQ(cc, 456u);
      EXPECT_EQ(wide, 789u);
      EXPECT_EQ(goc_test::scalar_bits_call(op, 0, 0, 0, &d, &wide, UINT64_MAX, 63, &cc),
                GOC_SUCCESS);
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
    }
  }
}
