#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a gfx1201 exhaustive FP16 unary result/exception probe.

Usage: capture_half_unary.py OUTPUT_DIR MNEMONIC
Compile OUTPUT_DIR/probe.cpp with hipcc and run in OUTPUT_DIR. capture.bin holds
little-endian uint32 (low-half result, EXCP_FLAG_USER) pairs, ordered by
[FP16_OVFL: off,on][denormal MODE nibble: 15,10,5,0][modifier: 0..31]
[input half: 0..65535]. Modifier bits: NEG=0, ABS=1, OMOD=2:3, CLAMP=4.
Each case uses one identical-input Wave32, guest nearest-even rounding, and
low-half source/destination selection. MODE is saved/restored around the opcode.
"""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("mnemonic", choices=["trunc", "ceil", "rndne", "floor", "sqrt",
                                         "rcp", "rsq", "exp", "log", "fract", "frexp_mant"])
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
source = r'''#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#define HIP(x) do {auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s\n",hipGetErrorString(e));exit(1);}}while(0)
__global__ void probe(uint32_t *out, unsigned modifier, uint32_t fp_mode) {
  unsigned i=blockIdx.x;
  uint32_t a=0x5a5a0000|i, result=0xdeadbeef, status=0, saved;
  switch(modifier) {
'''
for modifier in range(32):
    operand = "abs(%3)" if modifier & 2 else "%3"
    if modifier & 1:
        operand = "-" + operand
    instruction = f"v_{args.mnemonic}_f16_e64 %0, {operand}"
    instruction += " clamp" if modifier & 16 else ""
    instruction += ["", " mul:2", " mul:4", " div:2"][(modifier >> 2) & 3]
    assembly = ("s_getreg_b32 %1, hwreg(HW_REG_MODE)\n"
                "s_setreg_b32 hwreg(HW_REG_MODE), %4\n"
                "s_setreg_imm32_b32 hwreg(18), 0\ns_nop 7\n" + instruction +
                "\ns_waitcnt_depctr 0\ns_nop 7\ns_getreg_b32 %2, hwreg(18)\n"
                "s_setreg_b32 hwreg(HW_REG_MODE), %1\ns_nop 7")
    source += (f"case {modifier}: asm volatile({json.dumps(assembly)}"
               ':"+&v"(result),"=&s"(saved),"=&s"(status):"v"(a),"s"(fp_mode));break;\n')
source += r'''}
  if(threadIdx.x==0){out[2*i]=result&65535;out[2*i+1]=status;}
}
int main(){const unsigned n=65536;uint32_t *device;auto host=new uint32_t[2*n];
 HIP(hipMalloc(&device,8*n));FILE *f=fopen("capture.bin","wb");if(!f)return 2;
 const unsigned denorm[]={15,10,5,0};
 for(unsigned saturate=0;saturate<2;++saturate)for(unsigned policy=0;policy<4;++policy)
 for(unsigned modifier=0;modifier<32;++modifier){
 uint32_t mode=(saturate<<23)|(denorm[policy]<<4);
 hipLaunchKernelGGL(probe,dim3(n),dim3(32),0,0,device,modifier,mode);
 HIP(hipGetLastError());HIP(hipDeviceSynchronize());HIP(hipMemcpy(host,device,8*n,hipMemcpyDeviceToHost));
 if(fwrite(host,8,n,f)!=n)return 3;
 }
 if(fclose(f))return 4;HIP(hipFree(device));delete[]host;
}
'''
(args.output / "probe.cpp").write_text(source)
