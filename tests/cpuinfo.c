// SPDX-License-Identifier: MIT

#include "goc.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
  const uint64_t flags = goc_init_cpu_flags();
  const uint64_t cpu = flags & GOC_CPU_MASK;
  const char *name = "unknown";
  if (cpu == GOC_CPU_BASELINE)
    name = "baseline (scalar)";
  else if (cpu == GOC_CPU_X86_64_V3)
    name = "x86-64-v3";
  else if (cpu == GOC_CPU_X86_64_V4)
    name = "x86-64-v4";
  else if (cpu == GOC_CPU_ZEN4)
    name = "Zen4 feature set";

  printf("0x%016" PRIx64 " (%s)\n", flags, name);
  return 0;
}
