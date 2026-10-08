// SPDX-License-Identifier: MIT

#include "goc/goc.h"

#include <stdint.h>

extern "C" {

[[gnu::visibility("default")]] uint64_t consumer_cpu_flags(void) { return goc_init_cpu_flags(); }

} // extern "C"
