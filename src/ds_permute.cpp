// SPDX-License-Identifier: MIT

// Register routing follows rocjitsu's DS handlers. RDNA3 Wave64 routes within
// each half-wave; RDNA4 Wave64 routes across the entire wave.

#include "goc/goc.h"
#include "internal.h"

#include <stdint.h>

namespace {

template <bool Backward, unsigned Lanes, unsigned Group, bool ReadInactive = false>
int permute(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
            const uint32_t *const *addr, const uint32_t *const *data, uint16_t offset) {
  if (instruction_flags >> 32)
    return GOC_ERROR_INVALID_FLAGS;
  if (int error = goc::validate(flags, uint32_t(instruction_flags), true))
    return error;
  if (!exec_mask)
    return GOC_SUCCESS;
  uint32_t result[Lanes] = {};
  for (unsigned lane = 0; lane < Lanes; ++lane) {
    uint32_t address = addr[0][lane] + offset;
    unsigned selected = (lane & ~(Group - 1)) | ((address >> 2) & (Group - 1));
    if constexpr (Backward) {
      result[lane] = (ReadInactive || ((exec_mask >> selected) & 1)) ? data[0][selected] : 0;
    } else {
      // Visit every lane in order. The highest active source wins collisions,
      // matching rocjitsu and the ISA pseudocode's deterministic choice.
      if ((exec_mask >> lane) & 1)
        result[selected] = data[0][lane];
    }
  }
  // Delay stores until every source/address has been consumed, including aliases.
  for (unsigned lane = 0; lane < Lanes; ++lane)
    if ((exec_mask >> lane) & 1)
      d[0][lane] = result[lane];
  return GOC_SUCCESS;
}

} // namespace

int goc_ds_permute_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                       uint32_t *const *d, const uint32_t *const *addr, const uint32_t *const *data,
                       uint16_t offset) {
  return permute<false, 32, 32>(flags, exec_mask, instruction_flags, d, addr, data, offset);
}

int goc_ds_permute_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *addr,
                              const uint32_t *const *data, uint16_t offset) {
  return permute<false, 64, 32>(flags, exec_mask, instruction_flags, d, addr, data, offset);
}

int goc_ds_permute_b32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *addr,
                                    const uint32_t *const *data, uint16_t offset) {
  return permute<false, 64, 64>(flags, exec_mask, instruction_flags, d, addr, data, offset);
}

int goc_ds_bpermute_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                        uint32_t *const *d, const uint32_t *const *addr,
                        const uint32_t *const *data, uint16_t offset) {
  return permute<true, 32, 32>(flags, exec_mask, instruction_flags, d, addr, data, offset);
}

int goc_ds_bpermute_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *addr,
                               const uint32_t *const *data, uint16_t offset) {
  return permute<true, 64, 32>(flags, exec_mask, instruction_flags, d, addr, data, offset);
}

int goc_ds_bpermute_b32_rdna4_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *addr,
                                     const uint32_t *const *data, uint16_t offset) {
  return permute<true, 64, 64>(flags, exec_mask, instruction_flags, d, addr, data, offset);
}

int goc_ds_bpermute_fi_b32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *addr,
                           const uint32_t *const *data, uint16_t offset) {
  return permute<true, 32, 32, true>(flags, exec_mask, instruction_flags, d, addr, data, offset);
}

int goc_ds_bpermute_fi_b32_wave64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *addr,
                                  const uint32_t *const *data, uint16_t offset) {
  return permute<true, 64, 64, true>(flags, exec_mask, instruction_flags, d, addr, data, offset);
}
