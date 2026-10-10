// SPDX-License-Identifier: MIT
// Standalone HIP capture: hipcc --offload-arch=gfx1201 -mwavefrontsize64.
#include "swmmac8_hardware.h"

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <stdint.h>

#define HIP(call)                                                                                  \
  do {                                                                                             \
    auto e = (call);                                                                               \
    if (e != hipSuccess) {                                                                         \
      std::fprintf(stderr, "%s\n", hipGetErrorString(e));                                          \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)
using Int4 = int __attribute__((ext_vector_type(4)));

template <unsigned Format, unsigned Key>
__global__ void capture(const uint32_t *input, uint32_t *output) {
  unsigned lane = threadIdx.x, sample = blockIdx.x;
  const uint32_t *v = input + sample * 15 * 64;
  uint32_t a = v[lane], index = v[14 * 64 + lane];
  uint64_t b = uint64_t(v[2 * 64 + lane]) | (uint64_t(v[3 * 64 + lane]) << 32);
  Int4 d;
  for (unsigned reg = 0; reg < 4; ++reg)
    d[reg] = v[(6 + reg) * 64 + lane];
  if constexpr (Format == 0)
    asm volatile("v_swmmac_f32_16x16x32_fp8_fp8 %0, %1, %2, %3 index_key:%4"
                 : "+v"(d)
                 : "v"(a), "v"(b), "v"(index), "n"(Key));
  else if constexpr (Format == 1)
    asm volatile("v_swmmac_f32_16x16x32_fp8_bf8 %0, %1, %2, %3 index_key:%4"
                 : "+v"(d)
                 : "v"(a), "v"(b), "v"(index), "n"(Key));
  else if constexpr (Format == 2)
    asm volatile("v_swmmac_f32_16x16x32_bf8_fp8 %0, %1, %2, %3 index_key:%4"
                 : "+v"(d)
                 : "v"(a), "v"(b), "v"(index), "n"(Key));
  else if constexpr (Format == 3)
    asm volatile("v_swmmac_f32_16x16x32_bf8_bf8 %0, %1, %2, %3 index_key:%4"
                 : "+v"(d)
                 : "v"(a), "v"(b), "v"(index), "n"(Key));
  for (unsigned reg = 0; reg < 4; ++reg)
    output[sample * 256 + reg * 64 + lane] = d[reg];
}

template <unsigned Format, unsigned Key> void run(const uint32_t *input, uint32_t *device_output) {
  hipLaunchKernelGGL((capture<Format, Key>), dim3(16), dim3(64), 0, 0, input, device_output);
  HIP(hipGetLastError());
  HIP(hipDeviceSynchronize());
  uint32_t output[16][256];
  HIP(hipMemcpy(output, device_output, sizeof output, hipMemcpyDeviceToHost));
  for (unsigned sample = 0; sample < 16; ++sample) {
    uint64_t hash = 14695981039346656037ULL;
    for (auto word : output[sample])
      hash = (hash ^ word) * 1099511628211ULL;
    std::printf("%u %u %u %016llx\n", Format, sample, Key, (unsigned long long)hash);
  }
}

int main() {
  uint32_t input[16][15][64];
  uint32_t *device_input, *device_output;
  HIP(hipMalloc(&device_input, sizeof input));
  HIP(hipMalloc(&device_output, 16 * 256 * sizeof(uint32_t)));
  for (unsigned sample = 0; sample < 16; ++sample)
    goc_test::swmmac8_capture_inputs(0, sample, input[sample]);
  HIP(hipMemcpy(device_input, input, sizeof input, hipMemcpyHostToDevice));
  run<0, 0>(device_input, device_output);
  run<0, 1>(device_input, device_output);
  run<0, 2>(device_input, device_output);
  run<0, 3>(device_input, device_output);
  for (unsigned sample = 0; sample < 16; ++sample)
    goc_test::swmmac8_capture_inputs(1, sample, input[sample]);
  HIP(hipMemcpy(device_input, input, sizeof input, hipMemcpyHostToDevice));
  run<1, 0>(device_input, device_output);
  run<1, 1>(device_input, device_output);
  run<1, 2>(device_input, device_output);
  run<1, 3>(device_input, device_output);
  for (unsigned sample = 0; sample < 16; ++sample)
    goc_test::swmmac8_capture_inputs(2, sample, input[sample]);
  HIP(hipMemcpy(device_input, input, sizeof input, hipMemcpyHostToDevice));
  run<2, 0>(device_input, device_output);
  run<2, 1>(device_input, device_output);
  run<2, 2>(device_input, device_output);
  run<2, 3>(device_input, device_output);
  for (unsigned sample = 0; sample < 16; ++sample)
    goc_test::swmmac8_capture_inputs(3, sample, input[sample]);
  HIP(hipMemcpy(device_input, input, sizeof input, hipMemcpyHostToDevice));
  run<3, 0>(device_input, device_output);
  run<3, 1>(device_input, device_output);
  run<3, 2>(device_input, device_output);
  run<3, 3>(device_input, device_output);
  HIP(hipFree(device_input));
  HIP(hipFree(device_output));
}
