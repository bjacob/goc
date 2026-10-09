#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a gfx1201 HIP exception probe and its deterministic input corpus.

Usage: capture_rdna4_arithmetic_exceptions.py OUTPUT_DIR FAMILY WIDTH
Families: div_fixup (16/32/64), fma/pk_fma/pk_fmac (16), sin/cos (16/32).
Compile probe.cpp with hipcc for gfx1201 and run from OUTPUT_DIR.

capture.bin contains little-endian uint32 records:
[2 saturation settings][16 modifiers][case count][result_lo,result_hi,flags].
Case count is 65,536 for sin/cos, otherwise 8,192. MODE preserves denormals and
selects nearest-even. All 32 lanes use identical inputs. The probe restores MODE
and clears EXCP_FLAG_USER before each instruction.

Ordinary modifier index: bits 0:1 OMOD, bit 2 CLAMP, bit 3 -abs(A)/abs(B)/-C.
Packed FMA: bit 0 neg_lo=[1,0,1], bit 1 neg_hi=[0,1,1], bit 2 CLAMP,
bit 3 swaps low/high source selections. Packed FMAC repeats its unmodified form.
"""

import argparse
import json
import struct
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("family", choices=("div_fixup", "div_fmas", "fma", "pk_fma", "pk_fmac", "sin", "cos"))
parser.add_argument("width", type=int, choices=(16, 32, 64))
parser.add_argument("--boundaries", action="store_true", help="FP32 SIN/COS boundary corpus")
args = parser.parse_args()
out, family, width = args.output, args.family, args.width
if (family in ("fma", "pk_fma", "pk_fmac") and width != 16) or (family in ("sin", "cos") and width == 64):
    parser.error("unsupported family/width combination")
out.mkdir(parents=True, exist_ok=True)
n = 65536 if family in ("sin", "cos") else 8192
frac, bias = {16: (10, 15), 32: (23, 127), 64: (52, 1023)}[width]
sign = 1 << (width - 1)
inf = (2 * bias + 1) << frac
one = bias << frac
values = [
    0, sign, 1, sign | 1, (1 << frac) - 1, 1 << frac, one, sign | one,
    one + (1 << frac), inf - 1, inf - 1 - (1 << frac), inf, sign | inf,
    inf | 1, sign | inf | (1 << (frac - 1)) | 3, 2 << frac,
]
inputs = []
for i in range(n):
    if i < 4096:
        row = [values[(i >> 8) & 15], values[(i >> 4) & 15], values[i & 15]]
    else:
        state = i * 0x9e3779b97f4a7c15 & ((1 << 64) - 1)
        row = []
        for j in range(3):
            state ^= state >> 12
            state ^= (state << 25) & ((1 << 64) - 1)
            state ^= state >> 27
            row.append((state * 0x2545f4914f6cdd1d) & ((1 << width) - 1))
    if family.startswith("pk_"):
        if i < 4096:
            row = [x | (values[(((i >> (8 - j * 4)) & 15) + 5) % 16] << 16)
                   for j, x in enumerate(row)]
        else:
            row = [x | (((x * 0x9e37 + 0x1234) & 65535) << 16) for x in row]
    if family in ("sin", "cos"):
        row = [i if width == 16 else (i << 16) | ((i * 0x9e37) & 65535), 0, 0]
    if args.boundaries:
        if width != 32 or family not in ("sin", "cos"):
            parser.error("--boundaries requires FP32 SIN/COS")
        anchors = [0, 0x00145f30, 0x00800000, 0x3e800000, 0x3f000000,
                   0x3f800000, 0x4a000000, 0x7f800000,
                   0x80000000, 0x80145f30, 0x80800000, 0xbe800000, 0xbf000000,
                   0xbf800000, 0x7fc00000, 0xff800000]
        row = [(anchors[i // 4096] + (i % 4096) - 2048) & 0xffffffff, 0, 0]
    inputs += row
(out / "inputs.bin").write_bytes(struct.pack("<" + "Q" * len(inputs), *inputs))
source = r'''#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#define HIP(x) do { auto e=(x); if(e!=hipSuccess){fprintf(stderr,"%s\n",hipGetErrorString(e));exit(1);} }while(0)
__global__ void probe(const uint64_t *input,uint32_t *out,uint32_t fp_mode) {
  unsigned i=blockIdx.x;
  uint64_t a=input[3*i],b=input[3*i+1],c=input[3*i+2];
  uint32_t status,lo,hi;
'''
for m in range(16):
    operands = ["v[0:1]", "v[2:3]", "v[4:5]"] if width == 64 else ["v0", "v2", "v4"]
    if m & 8:
        operands = ["-|" + operands[0] + "|", "|" + operands[1] + "|", "-" + operands[2]]
    if family in ("sin", "cos"):
        operands = operands[:1]
    dest = "v[6:7]" if width == 64 else "v6"
    instruction = f"v_{family}_f{width} {dest}, " + ", ".join(operands)
    if family.startswith("pk_"):
        instruction = f"v_{family}_f16 v6, v0, v2" + (", v4" if family == "pk_fma" else "")
        if family == "pk_fma":
            instruction += " op_sel:[1,1,1] op_sel_hi:[0,0,0]" if m & 8 else ""
            instruction += " neg_lo:[1,0,1]" if m & 1 else ""
            instruction += " neg_hi:[0,1,1]" if m & 2 else ""
            instruction += " clamp" if m & 4 else ""
    else:
        instruction += " clamp" if m & 4 else ""
        instruction += ["", " mul:2", " mul:4", " div:2"][m & 3]
    assembly = [
        "s_getreg_b32 s7, hwreg(HW_REG_MODE)",
        "s_setreg_b32 hwreg(HW_REG_MODE), %9",
        "s_setreg_imm32_b32 hwreg(18), 0",
    ]
    assembly += [f"v_mov_b32 v{j}, %{j + 3}" for j in range(6)]
    if family == "div_fmas":
        assembly += ["s_bitcmp1_b32 %9, 23", "s_cselect_b32 vcc_lo, -1, 0", "s_mov_b32 vcc_hi, 0"]
    assembly += [
        "v_mov_b32 v6, v4" if family == "pk_fmac" else "v_mov_b32 v6, 0",
        "v_mov_b32 v7, 0", "s_nop 2", instruction, "s_waitcnt_depctr 0", "s_nop 7",
        "s_getreg_b32 %0, hwreg(18)", "v_mov_b32 %1, v6", "v_mov_b32 %2, v7",
        "s_setreg_b32 hwreg(HW_REG_MODE), s7",
    ]
    source += "asm volatile(\n" + "\n".join(json.dumps(line + "\n") for line in assembly)
    source += r'''
: "=&s"(status), "=&v"(lo), "=&v"(hi)
: "v"(uint32_t(a)),"v"(uint32_t(a>>32)),"v"(uint32_t(b)),"v"(uint32_t(b>>32)),"v"(uint32_t(c)),"v"(uint32_t(c>>32)),"s"(fp_mode)
: "vcc","s7","v0","v1","v2","v3","v4","v5","v6","v7");
'''
    source += f"""if(threadIdx.x==0){{
  out[({m}*{n}+i)*3]=lo;
  out[({m}*{n}+i)*3+1]=hi;
  out[({m}*{n}+i)*3+2]=status;
}}
"""
source += r'''}
int main() {
  constexpr size_t n=INPUT_COUNT, size=n*16*3;
  uint64_t input[n*3];
  FILE *f=fopen("inputs.bin","rb");
  if(!f || fread(input,8,n*3,f)!=n*3) return 2;
  fclose(f);
  uint64_t *a;
  uint32_t *out;
  auto host=new uint32_t[size];
  HIP(hipMalloc(&a,sizeof(input)));
  HIP(hipMalloc(&out,size*4));
  HIP(hipMemcpy(a,input,sizeof(input),hipMemcpyHostToDevice));
  f=fopen("capture.bin","wb");
  if(!f) return 2;
  for(unsigned sat=0;sat<2;++sat) {
    hipLaunchKernelGGL(probe,dim3(n),dim3(32),0,0,a,out,0xf0u|(sat<<23));
    HIP(hipGetLastError());
    HIP(hipDeviceSynchronize());
    HIP(hipMemcpy(host,out,size*4,hipMemcpyDeviceToHost));
    if(fwrite(host,4,size,f)!=size) return 3;
  }
  if(fclose(f)) return 4;
  HIP(hipFree(a));
  HIP(hipFree(out));
  delete[]host;
}
'''
(out / "probe.cpp").write_text(source.replace("INPUT_COUNT", str(n)))
