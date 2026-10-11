// SPDX-License-Identifier: MIT

// Offline C++20 generator. Only unchanged rocjitsu models and compiler half
// conversions are used; no GoC implementation or reference helpers are linked.
#include "util/amdgpu_rcp.h"
#include "util/amdgpu_rsq.h"
#include "util/amdgpu_sqrt.h"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <initializer_list>

int main() {
  std::puts("// SPDX-License-Identifier: MIT\n\n#pragma once\n\n#include <stdint.h>\n\n"
            "namespace goc_test {\n\n"
            "// RCP, RSQ, SQRT: cumulative word-wise hashes every 4096 half encodings.\n"
            "// Unmodified inputs, denormals preserved, quiet NaNs, no saturation.\n"
            "// Generated from rocjitsu; see CORPUS_DIAGNOSTICS.md.\n"
            "static const uint64_t half_sfu_checkpoints[3][16] = {");
  for (auto fn : {util::amdgpu_rcp_f32, util::amdgpu_rsq_f32, util::amdgpu_sqrt_f32}) {
    uint64_t hash = 14695981039346656037ULL;
    std::puts("  {");
    for (unsigned input = 0; input < 65536; ++input) {
      float value = float(std::bit_cast<_Float16>(uint16_t(input)));
      uint16_t output = std::bit_cast<uint16_t>(_Float16(fn(value, true)));
      hash = (hash ^ output) * 1099511628211ULL;
      if ((input + 1) % 4096 == 0)
        std::printf("    0x%016llxULL,\n", static_cast<unsigned long long>(hash));
    }
    std::puts("  },");
  }
  std::puts("};\n\n} // namespace goc_test");
}
