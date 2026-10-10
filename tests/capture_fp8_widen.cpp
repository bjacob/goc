// SPDX-License-Identifier: MIT
// Standalone HIP capture generator: compile for gfx1201 with hipcc.
// Prints the input code, two single results, then each packed result pair.

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <stdint.h>
#define HIP(x)                                                                                     \
  do {                                                                                             \
    auto e = (x);                                                                                  \
    if (e != hipSuccess) {                                                                         \
      fprintf(stderr, "%s\n", hipGetErrorString(e));                                               \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

__global__ void probe(uint32_t *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned a = i | (((i + 1) & 255) << 8), x, y;
  uint64_t p, q;
  asm volatile("v_cvt_f32_fp8 %0, %1 byte_sel:0" : "=v"(x) : "v"(a));
  asm volatile("v_cvt_f32_bf8 %0, %1 byte_sel:0" : "=v"(y) : "v"(a));
  asm volatile("v_cvt_pk_f32_fp8 %0, %1" : "=v"(p) : "v"(a));
  asm volatile("v_cvt_pk_f32_bf8 %0, %1" : "=v"(q) : "v"(a));
  out[i] = x;
  out[256 + i] = y;
  out[512 + i] = uint32_t(p);
  out[768 + i] = uint32_t(p >> 32);
  out[1024 + i] = uint32_t(q);
  out[1280 + i] = uint32_t(q >> 32);
}

int main() {
  uint32_t *d, out[1536];
  HIP(hipMalloc(&d, sizeof out));
  hipLaunchKernelGGL(probe, dim3(8), dim3(32), 0, 0, d);
  HIP(hipGetLastError());
  HIP(hipDeviceSynchronize());
  HIP(hipMemcpy(out, d, sizeof out, hipMemcpyDeviceToHost));
  for (unsigned i = 0; i < 256; ++i)
    printf("%02x %08x %08x %08x %08x %08x %08x\n", i, out[i], out[256 + i], out[512 + i],
           out[768 + i], out[1024 + i], out[1280 + i]);
  HIP(hipFree(d));
}
