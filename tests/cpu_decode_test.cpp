// SPDX-License-Identifier: MIT

#include "goc/goc.h"
#include "internal.h"

#include <gtest/gtest.h>
#include <initializer_list>
#include <stdint.h>

TEST(Cpu, FeatureAndOsGating) {
  goc::CpuState s;
  EXPECT_EQ(goc::decode_cpu(s), GOC_CPU_BASELINE);
  s.leaf1_ecx = s.leaf7_ebx = s.leaf7_ecx = s.leaf71_eax = s.ext1_ecx = UINT32_MAX;
  EXPECT_EQ(goc::decode_cpu(s), GOC_CPU_BASELINE);
  s.xcr0 = 0x6;
  EXPECT_EQ(goc::decode_cpu(s), GOC_CPU_X86_64_V3);
  s.xcr0 = 0xe6;
  EXPECT_EQ(goc::decode_cpu(s), GOC_CPU_ZEN4);
  s.leaf71_eax = 0;
  EXPECT_EQ(goc::decode_cpu(s), GOC_CPU_X86_64_V4);
  s.leaf7_ebx &= ~(1u << 16);
  EXPECT_EQ(goc::decode_cpu(s), GOC_CPU_X86_64_V3);
  s.leaf1_ecx &= ~(1u << 27);
  EXPECT_EQ(goc::decode_cpu(s), GOC_CPU_BASELINE);
}

TEST(Cpu, EveryRequiredFeatureIsChecked) {
  goc::CpuState full{UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, 0xe6};
  for (int bit : {0, 9, 12, 13, 19, 20, 22, 23, 26, 27, 28, 29}) {
    auto s = full;
    s.leaf1_ecx &= ~(1u << bit);
    EXPECT_EQ(goc::decode_cpu(s), GOC_CPU_BASELINE);
  }
  for (int bit : {1, 6, 8, 9, 10, 11, 12, 14}) {
    auto s = full;
    s.leaf7_ecx &= ~(1u << bit);
    EXPECT_EQ(goc::decode_cpu(s), GOC_CPU_X86_64_V4);
  }
}
