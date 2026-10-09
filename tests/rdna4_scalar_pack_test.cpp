// SPDX-License-Identifier: MIT

#include "capture_hash.h"
#include "fp_environment.h"
#include "goc/goc.h"
#include "rdna4_exec_masks.h"
#include "rdna4_scalar_integer_reference.h"
#include "rdna4_scalar_pack_hardware.h"
#include "rdna4_scalar_pack_reference.h"

#include <cfenv>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

TEST(ScalarPack, HardwareAndScc) {
  for (unsigned seed = 0; seed < 2; ++seed)
    for (unsigned op = 0; op < 11; ++op)
      for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
        for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
          uint64_t hash = goc_test::capture_hash_seed;
          for (unsigned i = 0; i < 4096; ++i) {
            uint32_t w[4], d = 0, cc = seed;
            uint64_t d64 = 0;
            goc_test::scalar_integer_inputs(i, w);
            uint64_t a = (uint64_t(w[1]) << 32) | w[0], b = (uint64_t(w[3]) << 32) | w[2];
            ASSERT_EQ(goc_test::scalar_pack_call(op, cpu | semantics | GOC_SEMANTICS_STRICT,
                                                 UINT32_MAX, 0, &d, &d64, a, b, &cc,
                                                 seed | 0xfffffffeu),
                      GOC_SUCCESS);
            bool wide = op == 4 || op == 6 || op == 8 || op == 10;
            for (uint32_t word : {wide ? uint32_t(d64) : d, wide ? uint32_t(d64 >> 32) : 0u, cc})
              hash = goc_test::capture_hash_word(hash, word);
          }
          ASSERT_EQ(hash, goc_test::scalar_pack_hardware[seed][op])
              << seed << "/" << op << "/" << semantics << "/" << cpu;
        }
}

TEST(ScalarPack, ExecAndOutputOverlap) {
  for (unsigned op = 0; op < 11; ++op)
    for (unsigned seed = 0; seed < 2; ++seed)
      for (uint32_t mask : rdna4_exec_masks()) {
        uint64_t a = UINT64_C(0x87654321abcdef01), b = UINT64_C(0x1234567800000041), wide = 0;
        uint32_t d = 0, cc = seed;
        ASSERT_EQ(goc_test::scalar_pack_call(op, 0, mask, 0, &d, &wide, a, b, &cc, seed),
                  GOC_SUCCESS);
        if (op == 4 || op == 6 || op == 8 || op == 10) {
          for (unsigned word = 0; word < 2; ++word) {
            uint64_t storage[] = {123, a, 456}, expected = wide;
            auto status = reinterpret_cast<uint32_t *>(
                reinterpret_cast<unsigned char *>(storage + 1) + word * 4);
            if (op >= 7)
              std::memcpy(reinterpret_cast<unsigned char *>(&expected) + word * 4, &cc, 4);
            ASSERT_EQ(goc_test::scalar_pack_call(op, 0, mask, 0, nullptr, storage + 1, storage[1],
                                                 b, status, seed),
                      GOC_SUCCESS);
            EXPECT_EQ(storage[1], expected);
            EXPECT_EQ(storage[0], 123u);
            EXPECT_EQ(storage[2], 456u);
          }
        } else {
          uint32_t storage[] = {123, uint32_t(a), 456};
          ASSERT_EQ(goc_test::scalar_pack_call(op, 0, mask, 0, storage + 1, nullptr, storage[1], b,
                                               storage + 1, seed),
                    GOC_SUCCESS);
          EXPECT_EQ(storage[1], op >= 7 ? cc : d);
          EXPECT_EQ(storage[0], 123u);
          EXPECT_EQ(storage[2], 456u);
        }
      }
}

TEST(ScalarPack, BasisBitsAndEveryQuadPresenceMask) {
  for (unsigned bit = 0; bit < 32; ++bit) {
    uint64_t d;
    ASSERT_EQ(goc_rdna4_s_bitreplicate_b64_b32(0, 0, 0, &d, 1u << bit), GOC_SUCCESS);
    EXPECT_EQ(d, UINT64_C(3) << (2 * bit));
    uint32_t result;
    for (unsigned op = 0; op < 4; ++op) {
      uint32_t cc;
      uint64_t unused;
      ASSERT_EQ(goc_test::scalar_pack_call(op, 0, 0, 0, &result, &unused, 1u << bit, 0, &cc, 0),
                GOC_SUCCESS);
      EXPECT_EQ(result, bit / 16 == unsigned(bool(op & 2)) ? 1u << (bit % 16) : 0u);
      ASSERT_EQ(goc_test::scalar_pack_call(op, 0, 0, 0, &result, &unused, 0, 1u << bit, &cc, 0),
                GOC_SUCCESS);
      EXPECT_EQ(result, bit / 16 == unsigned(bool(op & 1)) ? 1u << (16 + bit % 16) : 0u);
    }
  }
  for (unsigned presence = 0; presence < 65536; ++presence) {
    uint64_t a = 0, expanded = 0;
    for (unsigned group = 0; group < 16; ++group)
      if ((presence >> group) & 1) {
        a |= UINT64_C(1) << (4 * group + (group % 4));
        expanded |= UINT64_C(15) << (4 * group);
      }
    uint64_t d64;
    uint32_t d, cc;
    ASSERT_EQ(goc_rdna4_s_quadmask_b64(0, 0, 0, &d64, a, &cc), GOC_SUCCESS);
    EXPECT_EQ(d64, presence);
    EXPECT_EQ(cc, unsigned(presence != 0));
    ASSERT_EQ(goc_rdna4_s_wqm_b64(0, 0, 0, &d64, a, &cc), GOC_SUCCESS);
    EXPECT_EQ(d64, expanded);
    ASSERT_EQ(goc_rdna4_s_quadmask_b32(0, 0, 0, &d, uint32_t(a), &cc), GOC_SUCCESS);
    EXPECT_EQ(d, presence & 255);
    EXPECT_EQ(cc, unsigned((presence & 255) != 0));
    ASSERT_EQ(goc_rdna4_s_wqm_b32(0, 0, 0, &d, uint32_t(a), &cc), GOC_SUCCESS);
    EXPECT_EQ(d, uint32_t(expanded));
  }
}

TEST(ScalarPack, ErrorsAndHostFpState) {
  goc_test::ScopedFpEnvironment restore;
  ASSERT_TRUE(restore.saved());
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(rounding), 0);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
    int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
    for (unsigned op = 0; op < 11; ++op) {
      uint32_t d = 123, cc = 456;
      uint64_t d64 = 789;
      for (unsigned bit = 0; bit < 32; ++bit)
        EXPECT_EQ(goc_test::scalar_pack_call(op, 0, 0, 1u << bit, &d, &d64, 0, 0, &cc, 0),
                  GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(goc_test::scalar_pack_call(op, UINT64_C(1) << 63, 0, 0, &d, &d64, 0, 0, &cc, 0),
                GOC_ERROR_INVALID_FLAGS);
      for (unsigned bit = 32; bit < 64; ++bit)
        EXPECT_EQ(goc_test::scalar_pack_call(op, 0, 0, UINT64_C(1) << bit, nullptr, nullptr,
                                             UINT64_MAX, 0, nullptr, 0),
                  GOC_ERROR_INVALID_FLAGS);
      EXPECT_EQ(d, 123u);
      EXPECT_EQ(d64, 789u);
      EXPECT_EQ(cc, 456u);
      EXPECT_EQ(goc_test::scalar_pack_call(op,
                                           GOC_FP_FLUSH_INPUT_DENORMALS |
                                               GOC_FP_FLUSH_OUTPUT_DENORMALS | GOC_FP16_OVFL,
                                           0, 0, &d, &d64, UINT64_MAX, 0, &cc, 0),
                GOC_SUCCESS);
      EXPECT_EQ(std::fegetround(), rounding);
      EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
    }
  }
}
