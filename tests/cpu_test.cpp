// SPDX-License-Identifier: MIT

#include "goc/goc.h"

#include <gtest/gtest.h>

TEST(Cpu, DetectionReturnsOnlyKnownCpuBits) { EXPECT_LE(goc_init_cpu_flags(), GOC_CPU_ZEN4); }
