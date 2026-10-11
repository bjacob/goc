// SPDX-License-Identifier: MIT

#include "goc/goc.h"

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <random>
#include <stdint.h>

namespace {

const uint64_t exact = GOC_SEMANTICS_EXACT_EMPIRICAL | GOC_SEMANTICS_STRICT;

int run(unsigned arch, bool backward, uint64_t flags, uint64_t exec_mask, uint64_t mode,
        uint32_t *const *d, const uint32_t *const *addr, const uint32_t *const *data,
        uint16_t offset) {
  if (arch == 0)
    return (backward ? goc_ds_bpermute_b32 : goc_ds_permute_b32)(flags, uint32_t(exec_mask), mode,
                                                                 d, addr, data, offset);
  auto fn = arch == 3
                ? (backward ? goc_ds_bpermute_b32_wave64 : goc_ds_permute_b32_wave64)
                : (backward ? goc_ds_bpermute_b32_rdna4_wave64 : goc_ds_permute_b32_rdna4_wave64);
  return fn(flags, exec_mask, mode, d, addr, data, offset);
}

} // namespace

TEST(DsPermute, MasksOffsetsAliasesAndIndependentRoutingReference) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    SCOPED_TRACE(cpu);
    std::mt19937 random(72251);
    for (unsigned arch : {0U, 3U, 4U})
      for (bool backward : {false, true})
        for (unsigned sample = 0; sample < 150; ++sample)
          for (unsigned alias = 0; alias < 3; ++alias) {
            const unsigned lanes = arch == 0 ? 32 : 64, group = arch == 4 ? 64 : 32;
            const uint32_t low = random();
            const uint32_t high = random();
            uint64_t exec_mask = uint64_t(low) | (uint64_t(high) << 32);
            if (sample < 64)
              exec_mask = 1ULL << sample;
            if (sample == 64)
              exec_mask = UINT64_MAX;
            if (sample == 65)
              exec_mask = 0;
            uint16_t offset = sample < 64 ? sample : random();
            uint32_t words[3][66], before[3][66], expected[66];
            for (auto &reg : words)
              for (auto &word : reg)
                word = random();
            if (sample == 64) // Maximal collisions and discarded low address bits.
              for (unsigned lane = 0; lane < lanes; ++lane)
                words[0][lane + 1] = lane % 4;
            std::memcpy(before, words, sizeof(words));
            std::copy_n(words[alias], 66, expected);
            // Destination-centric reference: search all potential sources for scatter.
            for (unsigned dst = 0; dst < lanes; ++dst) {
              if (!((exec_mask >> dst) & 1))
                continue;
              uint32_t value = 0;
              for (unsigned src = 0; src < lanes; ++src) {
                if (!((exec_mask >> src) & 1) || src / group != dst / group)
                  continue;
                unsigned address_lane = backward ? dst : src;
                uint32_t address = before[0][address_lane + 1] + uint32_t(offset);
                unsigned wanted = (address / 4) % group;
                if (wanted == (backward ? src : dst) % group)
                  value = before[1][src + 1];
              }
              expected[dst + 1] = value;
            }
            const uint32_t *addr[] = {words[0] + 1}, *data[] = {words[1] + 1};
            uint32_t *d[] = {words[alias] + 1};
            ASSERT_EQ(run(arch, backward, (exact | cpu), exec_mask, 0, d, addr, data, offset),
                      GOC_SUCCESS);
            for (unsigned reg = 0; reg < 3; ++reg)
              for (unsigned lane = 0; lane < 66; ++lane)
                ASSERT_EQ(words[reg][lane], reg == alias ? expected[lane] : before[reg][lane])
                    << arch << '/' << backward << '/' << sample << '/' << alias << '/' << lane;
          }
  }
}

TEST(DsPermute, Wave64ArchitecturesDifferAcrossHalfBoundary) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    SCOPED_TRACE(cpu);
    uint32_t address[64], input[64], output[64];
    for (unsigned lane = 0; lane < 64; ++lane) {
      address[lane] = (lane ^ 32) * 4;
      input[lane] = lane + 100;
    }
    const uint32_t *addr = address, *data = input;
    uint32_t *d = output;
    for (bool backward : {false, true})
      for (unsigned arch : {3U, 4U}) {
        ASSERT_EQ(run(arch, backward, (exact | cpu), UINT64_MAX, 0, &d, &addr, &data, 0),
                  GOC_SUCCESS);
        for (unsigned lane = 0; lane < 64; ++lane)
          EXPECT_EQ(output[lane], input[arch == 3 ? lane : lane ^ 32]);
      }
  }
}

TEST(DsPermute, ValidationPreservesOutputsAndEmptyExecAllowsNull) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    SCOPED_TRACE(cpu);
    uint32_t words[64];
    std::fill_n(words, 64, 0xdeadbeefU);
    uint32_t *d = words;
    for (unsigned arch : {0U, 3U, 4U})
      for (bool backward : {false, true}) {
        for (unsigned bit = 0; bit < 64; ++bit)
          EXPECT_EQ(
              run(arch, backward, (exact | cpu), UINT64_MAX, 1ULL << bit, &d, nullptr, nullptr, 0),
              GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(run(arch, backward, 1ULL << 63, UINT64_MAX, 0, &d, nullptr, nullptr, 0),
                  GOC_ERROR_INVALID_FLAGS);
        EXPECT_EQ(run(arch, backward, (exact | cpu), 0, 0, nullptr, nullptr, nullptr, 65535),
                  GOC_SUCCESS);
        for (uint32_t word : words)
          EXPECT_EQ(word, 0xdeadbeefU);
      }
  }
}

TEST(DsPermute, ForceInactiveGathersStoredValuesInBothWaveSizes) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    SCOPED_TRACE(cpu);
    for (unsigned lanes : {32U, 64U})
      for (bool alias : {false, true})
        for (unsigned active = 0; active < lanes; ++active) {
          uint32_t input[64], addresses[64], output[64];
          for (unsigned lane = 0; lane < 64; ++lane) {
            input[lane] = 0x80000000U + lane;
            addresses[lane] = (lane + 1) * 4;
            output[lane] = 0xdeadbeefU;
          }
          uint32_t *d = alias ? input : output;
          const uint32_t *data = input, *addr = addresses;
          uint64_t exec_mask = 1ULL << active;
          ASSERT_EQ(lanes == 32 ? goc_ds_bpermute_fi_b32((exact | cpu), uint32_t(exec_mask), 0, &d,
                                                         &addr, &data, 0)
                                : goc_ds_bpermute_fi_b32_wave64((exact | cpu), exec_mask, 0, &d,
                                                                &addr, &data, 0),
                    GOC_SUCCESS);
          for (unsigned lane = 0; lane < 64; ++lane)
            EXPECT_EQ(d[lane], lane == active ? 0x80000000U + (lane + 1) % lanes
                               : alias        ? 0x80000000U + lane
                                              : 0xdeadbeefU);
        }
    EXPECT_EQ(goc_ds_bpermute_fi_b32((exact | cpu), 0, 0, nullptr, nullptr, nullptr, 0),
              GOC_SUCCESS);
    EXPECT_EQ(goc_ds_bpermute_fi_b32_wave64((exact | cpu), 0, 0, nullptr, nullptr, nullptr, 0),
              GOC_SUCCESS);
  }
}

TEST(DsPermute, ByteAddressOverflowAndScatterCollisions) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    SCOPED_TRACE(cpu);
    uint32_t addresses[64], input[64], output[64];
    std::fill_n(addresses, 64, UINT32_MAX);
    for (unsigned lane = 0; lane < 64; ++lane)
      input[lane] = lane + 1;
    const uint32_t *addr = addresses, *data = input;
    uint32_t *d = output;
    for (unsigned arch : {0U, 3U, 4U})
      for (bool backward : {false, true}) {
        unsigned lanes = arch == 0 ? 32 : 64;
        ASSERT_EQ(run(arch, backward, (exact | cpu), UINT64_MAX, 0, &d, &addr, &data, 5),
                  GOC_SUCCESS);
        for (unsigned lane = 0; lane < lanes; ++lane) {
          unsigned group_base = arch == 3 ? lane / 32 * 32 : 0;
          unsigned group_size = arch == 4 ? 64 : 32;
          uint32_t expected = backward                 ? input[group_base + 1]
                              : lane == group_base + 1 ? input[group_base + group_size - 1]
                                                       : 0;
          EXPECT_EQ(output[lane], expected);
        }
      }
  }
}

TEST(DsPermute, EveryGatherPairInactiveSourcesAndAddressAliases) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    SCOPED_TRACE(cpu);
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (unsigned arch : {0U, 3U, 4U})
        for (unsigned dst = 0; dst < (arch == 0 ? 32U : 64U); ++dst)
          for (unsigned src = 0; src < (arch == 4 ? 64U : 32U); ++src)
            for (bool force_inactive : {false, true}) {
              if (force_inactive && arch == 3)
                continue; // FI exists only on RDNA4.
              const unsigned selected = src + (arch == 3 ? (dst / 32) * 32 : 0);
              uint32_t addresses[66], input[64];
              std::fill_n(addresses, 66, 0xdeadbeefU);
              for (unsigned lane = 0; lane < 64; ++lane)
                input[lane] = 0x7f800001U + lane;
              // Offset addition wraps before ignoring low two and high address bits.
              addresses[dst + 1] = uint32_t(src * 4 + 3 - 65535U);
              uint32_t *d = addresses + 1;
              const uint32_t *addr = d, *data = input;
              uint64_t exec_mask = 1ULL << dst;
              uint64_t flags = cpu | semantics | GOC_SEMANTICS_STRICT;
              int status;
              if (!force_inactive)
                status = run(arch, true, flags, exec_mask, 0, &d, &addr, &data, 65535);
              else if (arch == 0)
                status =
                    goc_ds_bpermute_fi_b32(flags, uint32_t(exec_mask), 0, &d, &addr, &data, 65535);
              else
                status =
                    goc_ds_bpermute_fi_b32_wave64(flags, exec_mask, 0, &d, &addr, &data, 65535);
              ASSERT_EQ(status, GOC_SUCCESS);
              for (unsigned word = 0; word < 66; ++word)
                ASSERT_EQ(addresses[word],
                          word == dst + 1
                              ? (force_inactive || selected == dst ? input[selected] : 0U)
                              : 0xdeadbeefU)
                    << arch << '/' << dst << '/' << src << '/' << force_inactive;
            }
  }
}

TEST(DsPermute, ForceInactiveRejectsFlagsBeforeReadingOperands) {
  for (uint64_t cpu = 0; cpu <= goc_init_cpu_flags(); ++cpu) {
    SCOPED_TRACE(cpu);
    uint32_t output[64];
    std::fill_n(output, 64, 0xdeadbeefU);
    uint32_t *d = output;
    for (uint64_t semantics : {GOC_SEMANTICS_LOOSE, GOC_SEMANTICS_EXACT_EMPIRICAL})
      for (unsigned bit = 0; bit < 64; ++bit)
        for (uint64_t exec_mask : {0ULL, ~0ULL}) {
          EXPECT_EQ(goc_ds_bpermute_fi_b32(semantics, uint32_t(exec_mask), 1ULL << bit, &d, nullptr,
                                           nullptr, 0),
                    GOC_ERROR_INVALID_FLAGS);
          EXPECT_EQ(goc_ds_bpermute_fi_b32_wave64(semantics, exec_mask, 1ULL << bit, &d, nullptr,
                                                  nullptr, 0),
                    GOC_ERROR_INVALID_FLAGS);
        }
    for (uint32_t word : output)
      EXPECT_EQ(word, 0xdeadbeefU);
  }
}
