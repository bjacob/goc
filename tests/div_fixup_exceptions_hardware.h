// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

namespace goc_test {

// RX 9070 / gfx1201, MODE 0xf0, both FP16_OVFL settings (identical flags).
// Each digest covers 4,096 edge triples and 4,096 pseudorandom triples.
// Rows: FP16/32/64. Columns: OMOD, CLAMP, then NEG/ABS variants.
static const uint64_t fixup_exception_hashes[3][16] = {
    {0xc1b45eb5a5b03a46ULL, 0x67d867c1c2566f66ULL, 0xa2db9263565c137eULL, 0x18b5eeec698713c6ULL,
     0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL,
     0xc1b45eb5a5b03a46ULL, 0x67d867c1c2566f66ULL, 0xa2db9263565c137eULL, 0x18b5eeec698713c6ULL,
     0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL},
    {0x15bf1dc9de57fb38ULL, 0x2132563bd5e21078ULL, 0x33b05cd682584078ULL, 0x017b78a4cfd2ee28ULL,
     0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL,
     0x15bf1dc9de57fb38ULL, 0x2132563bd5e21078ULL, 0x33b05cd682584078ULL, 0x017b78a4cfd2ee28ULL,
     0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL},
    {0x660dde9651bc8a9bULL, 0xed23e8a15efbc7abULL, 0x3d20333a97e80dfbULL, 0xd55a51f1ddda2e8bULL,
     0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL,
     0x660dde9651bc8a9bULL, 0xed23e8a15efbc7abULL, 0x3d20333a97e80dfbULL, 0xd55a51f1ddda2e8bULL,
     0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL, 0xb9d103fd6854a325ULL},
};

} // namespace goc_test
