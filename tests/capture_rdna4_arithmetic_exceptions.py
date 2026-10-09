#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a gfx1201 HIP exception probe and its deterministic input corpus.

Usage: capture_rdna4_arithmetic_exceptions.py OUTPUT_DIR FAMILY WIDTH
Families: div_fixup (16/32/64), div_fmas (32/64), sin/cos (16/32).
All other families use WIDTH=16: fma/fmac/fmamk/fmaak, pk_fma/pk_fmac,
fma_mixlo/fma_mixhi, dot2_f32_f16/dot2_f32_bf16.
Compile probe.cpp with hipcc for gfx1201 and run from OUTPUT_DIR.

capture.bin contains little-endian uint32 records:
[2 saturation settings][16 modifiers][case count][result_lo,result_hi,flags].
Case count is 65,536 for sin/cos, otherwise 8,192. MODE preserves denormals and
selects nearest-even. All 32 lanes use identical inputs. The probe restores MODE
and clears EXCP_FLAG_USER before each instruction.

Ordinary modifier index: bits 0:1 OMOD, bit 2 CLAMP, bit 3 -abs(A)/abs(B)/-C.
For DIV_FMAS the outer setting selects condition=false/true (FP16_OVFL is irrelevant).
Packed FMA: bit 0 neg_lo=[1,0,1], bit 1 neg_hi=[0,1,1], bit 2 CLAMP,
bit 3 swaps low/high source selections. Packed FMAC repeats its unmodified form.
FMAC uses ordinary modifiers without negating C. Literal FMA repeats its
unmodified form with literal 0x3c01. Mixed FMA: bits 0:2 select FP16 sources,
bit 3 enables CLAMP; --mix-high selects high halves and -A/abs(B)/-abs(C).
DOT2 uses packed modifiers, but neg_hi affects only B and op_sel affects A/B.
"""

import argparse
import json
import struct
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument(
    "family", choices=("div_fixup", "div_fmas", "dot2_f32_f16", "dot2_f32_bf16",
                       "fma_mixlo", "fma_mixhi", "fma", "fmac", "fmamk", "fmaak",
                       "pk_fma", "pk_fmac", "sin", "cos"))
parser.add_argument("width", type=int, choices=(16, 32, 64))
parser.add_argument("--boundaries", action="store_true", help="FP32 SIN/COS boundary corpus")
parser.add_argument("--rounding-boundaries", action="store_true", help="FMA underflow/overflow corpus")
parser.add_argument("--mix-high", action="store_true", help="mixed FMA high halves and source modifiers")
args = parser.parse_args()
out, family, width = args.output, args.family, args.width
allowed_widths = {"div_fixup": (16, 32, 64), "div_fmas": (32, 64),
                  "sin": (16, 32), "cos": (16, 32)}
if width not in allowed_widths.get(family, (16,)):
    parser.error("unsupported family/width combination")
if args.boundaries and (family not in ("sin", "cos") or width != 32):
    parser.error("--boundaries requires FP32 SIN/COS")
if args.rounding_boundaries and family not in ("fma", "div_fmas"):
    parser.error("--rounding-boundaries requires FMA or DIV_FMAS")
if args.mix_high and not family.startswith("fma_mix"):
    parser.error("--mix-high requires mixed FMA")
out.mkdir(parents=True, exist_ok=True)
n = 65536 if family in ("sin", "cos") else 8192
input_width = 32 if family.startswith("fma_mix") else width
frac, bias = {16: (10, 15), 32: (23, 127), 64: (52, 1023)}[input_width]
if family == "dot2_f32_bf16":
    frac, bias = 7, 127
sign = 1 << (input_width - 1)
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
            row.append((state * 0x2545f4914f6cdd1d) & ((1 << input_width) - 1))
    if family.startswith("pk_"):
        if i < 4096:
            row = [x | (values[(((i >> (8 - j * 4)) & 15) + 5) % 16] << 16)
                   for j, x in enumerate(row)]
        else:
            row = [x | (((x * 0x9e37 + 0x1234) & 65535) << 16) for x in row]
    if family.startswith("dot2_"):
        if i < 4096:
            row[:2] = [x | (values[(((i >> (8 - j * 4)) & 15) + 5) % 16] << 16)
                       for j, x in enumerate(row[:2])]
            fp32 = [0, 0x80000000, 1, 0x80000001, 0x7fffff, 0x800000,
                    0x3f800000, 0xbf800000, 0x40000000, 0x7f7fffff, 0x7effffff,
                    0x7f800000, 0xff800000, 0x7f800001, 0xffc00003, 0x1000000]
            row[2] = fp32[i & 15]
        else:
            row[:2] = [x | (((x * 0x9e37 + 0x1234) & 65535) << 16) for x in row[:2]]
            row[2] = (state * 0x2545f4914f6cdd1d) & 0xffffffff
    if family in ("sin", "cos"):
        row = [i if width == 16 else (i << 16) | ((i * 0x9e37) & 65535), 0, 0]
    if args.boundaries:
        anchors = [0, 0x00145f30, 0x00800000, 0x3e800000, 0x3f000000,
                   0x3f800000, 0x4a000000, 0x7f800000,
                   0x80000000, 0x80145f30, 0x80800000, 0xbe800000, 0xbf000000,
                   0xbf800000, 0x7fc00000, 0xff800000]
        row = [(anchors[i // 4096] + (i % 4096) - 2048) & 0xffffffff, 0, 0]
    if args.rounding_boundaries:
        index = i & 4095
        a = ((1 << frac) - 64 + (index >> 5)) if i < 4096 else inf - 1 - (index >> 5)
        b = one - 2 + ((index >> 3) & 3)
        if i < 4096:
            addends = [0, 1, 2, 3, sign, sign | 1, sign | 2, sign | 3]
        else:
            quarter_ulp = (2 * bias - frac - 2) << frac
            addends = [0, 1, one, sign | one, quarter_ulp, sign | quarter_ulp,
                       quarter_ulp + (1 << frac), sign | (quarter_ulp + (1 << frac))]
        row = [a, b, addends[index & 7]]
    if family == "fmamk":
        row[1] = 0x3c01
    elif family == "fmaak":
        row[2] = 0x3c01
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
    if family in ("fmac", "fmamk", "fmaak"):
        if family == "fmac":
            instruction = "v_fmac_f16 v6, " + ("-|v0|, |v2|" if m & 8 else "v0, v2")
            instruction += " clamp" if m & 4 else ""
            instruction += ["", " mul:2", " mul:4", " div:2"][m & 3]
        elif family == "fmamk":
            instruction = "v_fmamk_f16 v6, v0, 0x3c01, v4"
        else:
            instruction = "v_fmaak_f16 v6, v0, v2, 0x3c01"
    elif family.startswith("dot2_"):
        instruction = f"v_{family} v6, v0, v2, v4"
        instruction += " op_sel:[1,1,0] op_sel_hi:[0,0,0]" if m & 8 else ""
        instruction += " neg_lo:[1,0,1]" if m & 1 else ""
        instruction += " neg_hi:[0,1,0]" if m & 2 else ""
        instruction += " clamp" if m & 4 else ""
    elif family.startswith("fma_mix"):
        sources = "-v0, |v2|, -|v4|" if args.mix_high else "v0, v2, v4"
        instruction = f"v_{family}_f16 v6, {sources}"
        instruction += " op_sel:[1,1,1]" if args.mix_high else ""
        instruction += f" op_sel_hi:[{m & 1},{(m >> 1) & 1},{(m >> 2) & 1}]"
        instruction += " clamp" if m & 8 else ""
    elif family.startswith("pk_"):
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
        "v_mov_b32 v6, v4" if family in ("pk_fmac", "fmac") else "v_mov_b32 v6, 0",
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
