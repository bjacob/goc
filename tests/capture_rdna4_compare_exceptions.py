# SPDX-License-Identifier: MIT

"""Generate the gfx1201 HIP probe for rdna4_compare_exceptions_hardware.h.

Usage: python3 capture_rdna4_compare_exceptions.py OUTPUT_DIR [--signaling]
Compile generated source with hipcc --offload-arch=gfx1201 -O2, then run it from
OUTPUT_DIR. The binary output is uint32_t[4][3][112][144]: denormal mode,
EXEC={all,0,1}, instruction (listed in ops.txt), and 12x12 operand pair.
--signaling sets CLAMP on vector comparisons; scalar comparisons are unchanged.
Normal ctest uses the recorded tables and requires no GPU.
"""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output_dir", type=Path)
parser.add_argument("--signaling", action="store_true")
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

predicates = "lt eq le gt lg ge o u nge nlg ngt nle neq nlt".split()
ops = [(kind, bits, pred)
       for kind in ("s_cmp", "v_cmp", "v_cmpx")
       for bits in ((16, 32) if kind == "s_cmp" else (16, 32, 64))
       for pred in predicates]
# Same raw encodings and order as compare_exception_inputs in the fixture.
values = [
    [0, 0x8000, 1, 0x3ff, 0x400, 0x3c00, 0x7c00, 0xfc00,
     0x7c01, 0x7e00, 0xfe01, 0x8001],
    [0, 0x80000000, 1, 0x7fffff, 0x800000, 0x3f800000, 0x7f800000,
     0xff800000, 0x7f800001, 0x7fc00000, 0xffc00001, 0x80000001],
    [0, 0x8000000000000000, 1, 0xfffffffffffff, 0x10000000000000,
     0x3ff0000000000000, 0x7ff0000000000000, 0xfff0000000000000,
     0x7ff0000000000001, 0x7ff8000000000000, 0xfff8000000000001,
     0x8000000000000001],
]
source = r'''#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#define HIP(x) do { auto error = (x); if (error != hipSuccess) { \
  fprintf(stderr, "%s\n", hipGetErrorString(error)); exit(1); } } while (0)

__global__ void probe(uint32_t *out, uint32_t mode, uint32_t mask) {
  const uint64_t values[3][12] = {
'''
source += "\n".join("    {" + ", ".join(hex(v) + "ULL" for v in row) + "},"
                    for row in values)
source += "\n  };\n  unsigned i = blockIdx.x;\n  uint32_t status;\n"
for index, (kind, bits, pred) in enumerate(ops):
    fmt = (16, 32, 64).index(bits)
    mnemonic = f"{kind}_{pred}_f{bits}"
    if kind == "s_cmp":
        instruction = mnemonic + " s0, s2"
        moves = []
    else:
        destination = "" if kind == "v_cmpx" else "vcc_lo, "
        operands = "v[0:1], v[2:3]" if bits == 64 else "v0, v2"
        instruction = (mnemonic + ("_e64" if args.signaling else "") + " " +
                       destination + operands + (" clamp" if args.signaling else ""))
        moves = [f"v_mov_b32 v{j}, s{j}" for j in range(4)]
    # Keep setup and EXCP sampling in one asm block. Hardware register 18 is
    # EXCP_FLAG_USER; no traps are enabled. Restore MODE and EXEC after sampling.
    assembly = ([f"s_mov_b32 s{j}, %{j + 1}" for j in range(4)] + [
        "s_mov_b32 s4, exec_lo",
        "s_getreg_b32 s5, hwreg(HW_REG_MODE)",
        "s_setreg_b32 hwreg(HW_REG_MODE), %5",
        "s_setreg_imm32_b32 hwreg(18), 0",
    ] + moves + [
        "s_and_b32 exec_lo, s4, %6", "s_nop 2", instruction,
        "s_waitcnt_depctr 0", "s_nop 7",
        "s_getreg_b32 %0, hwreg(18)", "s_mov_b32 exec_lo, s4",
        "s_setreg_b32 hwreg(HW_REG_MODE), s5",
    ])
    source += f"  {{\n    uint64_t a = values[{fmt}][i / 12], b = values[{fmt}][i % 12];\n"
    source += "    asm volatile(\n"
    source += "\n".join("      " + json.dumps(line + "\n") for line in assembly)
    source += r'''
      : "=&s"(status)
      : "s"(uint32_t(a)), "s"(uint32_t(a >> 32)), "s"(uint32_t(b)),
        "s"(uint32_t(b >> 32)), "s"(mode), "s"(mask)
      : "s0", "s1", "s2", "s3", "s4", "s5", "v0", "v1", "v2", "v3", "scc", "vcc");
'''
    source += f"    if (threadIdx.x == 0) out[{index} * 144 + i] = status;\n  }}\n"
source += r'''}
int main() {
  uint32_t *device;
  const size_t count = 112 * 144;
  HIP(hipMalloc(&device, count * 4));
  auto host = new uint32_t[count];
  FILE *file = fopen("OUTPUT_FILE", "wb");
  if (!file) return 2;
  for (unsigned denorm = 0; denorm < 4; ++denorm)
    for (uint32_t mask : {0xffffffffu, 0u, 1u}) {
      hipLaunchKernelGGL(probe, dim3(144), dim3(32), 0, 0, device,
                         (denorm << 4) | (denorm << 6), mask);
      HIP(hipGetLastError());
      HIP(hipDeviceSynchronize());
      HIP(hipMemcpy(host, device, count * 4, hipMemcpyDeviceToHost));
      if (fwrite(host, 4, count, file) != count) return 3;
    }
  if (fclose(file)) return 4;
  HIP(hipFree(device));
  delete[] host;
}
'''
stem = "goc-compare-excp" + ("-signaling" if args.signaling else "")
source = source.replace("OUTPUT_FILE", stem + ".bin")
(args.output_dir / (stem + ".cpp")).write_text(source)
(args.output_dir / (stem + "-ops.txt")).write_text(
    "\n".join(f"{kind}_{pred}_f{bits}" for kind, bits, pred in ops) + "\n")
