// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_scalar_field_hardware.h"
#include "rdna4_scalar_field_reference.h"
#include "rdna4_scalar_integer_reference.h"

#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

TEST(ScalarField, HardwareResultsAndScc) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu)
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (unsigned seed = 0; seed < 2; ++seed)
        for (uint32_t mask : {UINT32_C(0), UINT32_MAX, UINT32_C(0xaaaaaaaa)})
          for (unsigned op = 0; op < 20; ++op) {
            uint64_t hash = UINT64_C(14695981039346656037);
            for (unsigned i = 0; i < 16384; ++i) {
              uint32_t w[4];
              goc_test::scalar_field_inputs(i, w);
              uint64_t a = (uint64_t(w[1]) << 32) | w[0], b = (uint64_t(w[3]) << 32) | w[2],
                       d64 = a;
              uint32_t d = uint32_t(a), cc = seed;
              ASSERT_EQ(goc_test::scalar_field_call(op, cpu | semantics | GOC_SEMANTICS_STRICT,
                                                    mask, 0, &d, &d64, a, b, &cc),
                        GOC_SUCCESS);
              bool wide = (op == 2 || op == 3 || op == 5 || op == 17 || op == 19);
              for (uint32_t word : {wide ? uint32_t(d64) : d, wide ? uint32_t(d64 >> 32) : 0u, cc})
                hash = (hash ^ word) * UINT64_C(1099511628211);
            }
            ASSERT_EQ(hash, goc_test::scalar_field_hardware[seed][op])
                << cpu << "/" << semantics << "/" << seed << "/" << mask << "/" << op;
          }
}

TEST(ScalarField, ExecAndAliasing) {
  const uint64_t a = UINT64_C(0x87654321abcdef01);
  for (unsigned op = 0; op < 20; ++op)
    for (uint32_t mask : rdna4_exec_masks()) {
      uint32_t d = uint32_t(a), cc = 7;
      uint64_t wide = a;
      ASSERT_EQ(goc_test::scalar_field_call(op, 0, mask, 0, &d, &wide, a, 0x3f0011, &cc),
                GOC_SUCCESS);
      bool wide_output = op == 2 || op == 3 || op == 5 || op == 17 || op == 19;
      bool writes_scc = op < 4 || (op >= 6 && op < 10);
      if (wide_output) {
        for (unsigned word = 0; word < 2; ++word) {
          uint64_t storage[] = {123, a, 456}, expected = wide;
          auto bytes = reinterpret_cast<unsigned char *>(&storage[1]);
          auto status = reinterpret_cast<uint32_t *>(bytes + 4 * word);
          if (writes_scc)
            std::memcpy(reinterpret_cast<unsigned char *>(&expected) + 4 * word, &cc, 4);
          ASSERT_EQ(goc_test::scalar_field_call(op, 0, mask, 0, nullptr, storage + 1, storage[1],
                                                0x3f0011, status),
                    GOC_SUCCESS);
          EXPECT_EQ(storage[1], expected);
          EXPECT_EQ(storage[0], 123u);
          EXPECT_EQ(storage[2], 456u);
        }
      } else {
        uint32_t storage[] = {123, uint32_t(a), 456};
        ASSERT_EQ(goc_test::scalar_field_call(op, 0, mask, 0, storage + 1, nullptr, storage[1],
                                              0x3f0011, storage + 1),
                  GOC_SUCCESS);
        // 64-bit inputs remain by value even when the output is 32 bits.
        uint32_t expected = uint32_t(a), status = 7;
        uint64_t unused = 0;
        ASSERT_EQ(goc_test::scalar_field_call(op, 0, mask, 0, &expected, &unused, uint32_t(a),
                                              0x3f0011, &status),
                  GOC_SUCCESS);
        EXPECT_EQ(storage[1], writes_scc ? status : expected);
        EXPECT_EQ(storage[0], 123u);
        EXPECT_EQ(storage[2], 456u);
      }
    }
}

TEST(ScalarField, EveryFieldWidthAndOffset) {
  for (unsigned width : {32u, 64u})
    for (uint64_t a : {UINT64_C(0), UINT64_C(1), UINT64_C(0x87654321abcdef01), UINT64_MAX})
      for (unsigned count = 0; count < 128; ++count)
        for (unsigned offset = 0; offset < 64; ++offset)
          for (unsigned sign = 0; sign < 2; ++sign) {
            uint64_t expected = 0;
            unsigned off = offset % width, available = width - off,
                     n = count < available ? count : available;
            for (unsigned bit = 0; bit < width; ++bit) {
              bool set =
                  bit < n ? ((a >> (off + bit)) & 1) : sign && n && ((a >> (off + n - 1)) & 1);
              expected |= uint64_t(set) << bit;
            }
            uint32_t d, cc;
            uint64_t d64;
            unsigned op = (width == 64 ? 2 : 0) + sign;
            ASSERT_EQ(
                goc_test::scalar_field_call(op, 0, 0, 0, &d, &d64, a, (count << 16) | offset, &cc),
                GOC_SUCCESS);
            EXPECT_EQ(width == 64 ? d64 : d, expected);
            EXPECT_EQ(cc, unsigned(expected != 0));
          }
}

TEST(ScalarField, CountSentinelsAndSingleBits) {
  for (unsigned width : {32u, 64u}) {
    for (unsigned op = 6; op < 16; op += 2) {
      unsigned index = op + (width == 64);
      for (unsigned bit = 0; bit <= width; ++bit) {
        uint64_t a = bit == width ? 0 : UINT64_C(1) << bit;
        uint32_t d, cc = 7;
        uint64_t unused;
        ASSERT_EQ(goc_test::scalar_field_call(index, 0, 0, 0, &d, &unused, a, 0, &cc), GOC_SUCCESS);
        uint32_t expected;
        if (op < 10)
          expected = op == 6 ? width - unsigned(a != 0) : unsigned(a != 0);
        else if (!a)
          expected = UINT32_MAX;
        else if (op == 10)
          expected = bit;
        else if (op == 14 && bit == width - 1)
          expected = 1;
        else
          expected = width - 1 - bit;
        EXPECT_EQ(d, expected) << op << "/" << width << "/" << bit;
        EXPECT_EQ(cc, op < 10 ? unsigned(expected != 0) : 7u);
      }
    }
  }
}

TEST(ScalarField, ErrorsDoNotWriteAndHostFpStatePreserved) {
  fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    std::fesetround(rounding);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
    int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
    for (unsigned op = 0; op < 20; ++op) {
      uint32_t d = 123, cc = 456;
      uint64_t wide = 789;
      for (unsigned bit = 0; bit < 32; ++bit)
        EXPECT_EQ(goc_test::scalar_field_call(op, 0, 0, 1u << bit, &d, &wide, 0, 0, &cc),
                  GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(d, 123u);
      EXPECT_EQ(cc, 456u);
      EXPECT_EQ(wide, 789u);
      EXPECT_EQ(goc_test::scalar_field_call(op, 0, 0, 0, &d, &wide, UINT64_MAX, 63, &cc),
                GOC_SUCCESS);
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
    }
  }
  EXPECT_EQ(std::fesetenv(&saved), 0);
}
