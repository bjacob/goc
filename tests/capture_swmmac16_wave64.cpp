// SPDX-License-Identifier: MIT
// Standalone HIP capture: hipcc --offload-arch=gfx1201 -mwavefrontsize64.
#include "swmmac16_hardware.h"

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <type_traits>

#define HIP(call)                                                                                  \
  do {                                                                                             \
    auto e = (call);                                                                               \
    if (e != hipSuccess) {                                                                         \
      std::fprintf(stderr, "%s\n", hipGetErrorString(e));                                          \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)
using Int4 = int __attribute__((ext_vector_type(4)));

template <unsigned Format, unsigned Mode>
__global__ void capture(const uint32_t *v, uint32_t *output) {
  unsigned lane = threadIdx.x;
  uint64_t a = uint64_t(v[lane]) | (uint64_t(v[64 + lane]) << 32);
  Int4 b;
  for (unsigned reg = 0; reg < 4; ++reg)
    b[reg] = v[(4 + reg) * 64 + lane];
  unsigned index = v[20 * 64 + lane];
  std::conditional_t<(Format >= 2), uint64_t, Int4> d;
  if constexpr (Format >= 2)
    d = uint64_t(v[12 * 64 + lane]) | (uint64_t(v[13 * 64 + lane]) << 32);
  else
    for (unsigned reg = 0; reg < 4; ++reg)
      d[reg] = v[(12 + reg) * 64 + lane];
  if constexpr (Format == 0)
    asm volatile(
        "v_swmmac_f32_16x16x32_f16 %0, %1, %2, %3 index_key:%4 neg_lo:[%5,%6,0] neg_hi:[%7,%8,0]"
        : "+v"(d)
        : "v"(a), "v"(b), "v"(index), "n"(Mode >> 4), "n"(bool(Mode & 1)), "n"(bool(Mode & 2)),
          "n"(bool(Mode & 4)), "n"(bool(Mode & 8)));
  else if constexpr (Format == 1)
    asm volatile(
        "v_swmmac_f32_16x16x32_bf16 %0, %1, %2, %3 index_key:%4 neg_lo:[%5,%6,0] neg_hi:[%7,%8,0]"
        : "+v"(d)
        : "v"(a), "v"(b), "v"(index), "n"(Mode >> 4), "n"(bool(Mode & 1)), "n"(bool(Mode & 2)),
          "n"(bool(Mode & 4)), "n"(bool(Mode & 8)));
  else if constexpr (Format == 2)
    asm volatile(
        "v_swmmac_f16_16x16x32_f16 %0, %1, %2, %3 index_key:%4 neg_lo:[%5,%6,0] neg_hi:[%7,%8,0]"
        : "+v"(d)
        : "v"(a), "v"(b), "v"(index), "n"(Mode >> 4), "n"(bool(Mode & 1)), "n"(bool(Mode & 2)),
          "n"(bool(Mode & 4)), "n"(bool(Mode & 8)));
  else if constexpr (Format == 3)
    asm volatile(
        "v_swmmac_bf16_16x16x32_bf16 %0, %1, %2, %3 index_key:%4 neg_lo:[%5,%6,0] neg_hi:[%7,%8,0]"
        : "+v"(d)
        : "v"(a), "v"(b), "v"(index), "n"(Mode >> 4), "n"(bool(Mode & 1)), "n"(bool(Mode & 2)),
          "n"(bool(Mode & 4)), "n"(bool(Mode & 8)));
  if constexpr (Format >= 2) {
    output[lane] = uint32_t(d);
    output[64 + lane] = uint32_t(d >> 32);
  } else
    for (unsigned reg = 0; reg < 4; ++reg)
      output[reg * 64 + lane] = d[reg];
}

template <unsigned Format, unsigned Mode> void run(const uint32_t *input, uint32_t *device_output) {
  hipLaunchKernelGGL((capture<Format, Mode>), dim3(1), dim3(64), 0, 0, input, device_output);
  HIP(hipGetLastError());
  HIP(hipDeviceSynchronize());
  uint32_t output[Format >= 2 ? 128 : 256];
  HIP(hipMemcpy(output, device_output, sizeof output, hipMemcpyDeviceToHost));
  uint64_t hash = 14695981039346656037ULL;
  for (auto word : output)
    hash = (hash ^ word) * 1099511628211ULL;
  std::printf("%u %u %016llx\n", Format, Mode, (unsigned long long)hash);
  if constexpr (Mode < 63)
    run<Format, Mode + 1>(input, device_output);
}

int main() {
  uint32_t input[21][64], *device_input, *device_output;
  HIP(hipMalloc(&device_input, sizeof input));
  HIP(hipMalloc(&device_output, 256 * sizeof(uint32_t)));
  goc_test::swmmac16_capture_inputs(0, input);
  HIP(hipMemcpy(device_input, input, sizeof input, hipMemcpyHostToDevice));
  run<0, 0>(device_input, device_output);
  goc_test::swmmac16_capture_inputs(1, input);
  HIP(hipMemcpy(device_input, input, sizeof input, hipMemcpyHostToDevice));
  run<1, 0>(device_input, device_output);
  goc_test::swmmac16_capture_inputs(2, input);
  HIP(hipMemcpy(device_input, input, sizeof input, hipMemcpyHostToDevice));
  run<2, 0>(device_input, device_output);
  goc_test::swmmac16_capture_inputs(3, input);
  HIP(hipMemcpy(device_input, input, sizeof input, hipMemcpyHostToDevice));
  run<3, 0>(device_input, device_output);
  HIP(hipFree(device_input));
  HIP(hipFree(device_output));
}
