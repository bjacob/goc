// SPDX-License-Identifier: MIT

// Intentionally include only the header under test, twice: test self-containment
// and include guards in both C99 and C++17.
#include "detail/goc_common.h"

// Repeated inclusion must be harmless.
#include "detail/goc_common.h"

int main(void) { return (goc_init_cpu_flags() & ~GOC_CPU_MASK) != 0; }
