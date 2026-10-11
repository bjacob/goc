// SPDX-License-Identifier: MIT

// Offline fixture generator. Compile as C++20 with the rocjitsu util include
// directory and GoC tests/ on the include path; no GoC implementation is linked.
#include "unary_exact_hardware.h"
#include "util/amdgpu_exp.h"
#include "util/amdgpu_log.h"
#include "util/amdgpu_rcp.h"
#include "util/amdgpu_rsq.h"
#include "util/amdgpu_sqrt.h"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <initializer_list>

int main() {
  const auto functions = {util::amdgpu_rcp_f32, util::amdgpu_sqrt_f32, util::amdgpu_rsq_f32,
                          util::amdgpu_exp_f32, util::amdgpu_log_f32};
  uint64_t checkpoints[5][32] = {};
  unsigned counts[5] = {}, op = 0;
  for (auto fn : functions) {
    const unsigned size = 1U << (op == 0 ? 23 : op == 3 ? 25 : 24);
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned index = 0; index < size; ++index) {
      uint32_t input = op == 3   ? (0x3f000000U + (index & 0xffffff)) | ((index >> 24) << 31)
                       : op == 4 ? 0x3f000000U + index
                                 : 0x3f800000U + index;
      uint32_t output = std::bit_cast<uint32_t>(fn(std::bit_cast<float>(input), true));
      hash = (hash ^ output) * 1099511628211ULL;
      if ((index + 1) % (1U << 20) == 0)
        checkpoints[op][counts[op]++] = hash;
    }
    // The independently generated prefix stream must end at the retained
    // hardware digest. Never emit a fixture derived from an unvalidated model.
    if (hash != goc_test::unary_normalized_hashes[op]) {
      std::fprintf(stderr, "rocjitsu/hardware digest mismatch for operation %u\n", op);
      return 1;
    }
    ++op;
  }
  std::puts("// SPDX-License-Identifier: MIT\n\n#pragma once\n\n#include <stdint.h>\n\n"
            "namespace goc_test {\n\n"
            "// Cumulative word-wise FNV after each 2^20 inputs in the dense domains.\n"
            "// Generated directly from rocjitsu; see CORPUS_DIAGNOSTICS.md for provenance.\n"
            "static const uint64_t unary_normalized_checkpoints[5][32] = {");
  for (op = 0; op < 5; ++op) {
    std::puts("  {");
    for (unsigned i = 0; i < counts[op]; ++i)
      std::printf("    0x%016llxULL,\n", static_cast<unsigned long long>(checkpoints[op][i]));
    std::puts("  },");
  }
  std::puts("};\n\n} // namespace goc_test");
}
