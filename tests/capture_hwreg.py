#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a gfx1201 MODE/EXCP_FLAG_USER HWREG field probe.

Compile OUTPUT_DIR/probe.cpp with hipcc and run it in OUTPUT_DIR. capture.bin
contains little-endian (initialized register, updated register, extracted field)
triples in [register: MODE, EXCP_FLAG_USER][seed: 0, ~0][source: 0, ~0,
0xa5a5a5a5, 0x5a5a5a5a, literal ~0][offset: 0..31][size-minus-one: 0..31]
order. Every case uses one Wave32 and restores the selected GPU register.
This tests user-mode field access, not privilege or performance-snapshot effects.
"""
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
s = r'''#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#define HIP(x) do {auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s\n",hipGetErrorString(e));exit(1);}}while(0)
__global__ void probe(uint32_t *out, unsigned reg, uint32_t seed, uint32_t source, bool literal) {
  unsigned index=blockIdx.x; uint32_t before=0,after=0,field=0,saved;
  switch(reg*1024+index) {
'''
for reg, ident in enumerate([1, 18]):
    for offset in range(32):
        for size in range(1, 33):
            descriptor = f"hwreg({ident},{offset},{size})"
            full = f"hwreg({ident})"
            prologue = f"s_getreg_b32 %0, {full}\ns_setreg_b32 {full}, %4\ns_nop 7\ns_getreg_b32 %1, {full}\n"
            epilogue = f"\ns_nop 7\ns_getreg_b32 %2, {full}\ns_getreg_b32 %3, {descriptor}\ns_setreg_b32 {full}, %0\ns_nop 7"
            s += f"case {reg*1024+offset*32+size-1}:\n"
            for literal in [True, False]:
                instruction = f"s_setreg_imm32_b32 {descriptor}, 0xffffffff" if literal else f"s_setreg_b32 {descriptor}, %5"
                assembly = prologue + instruction + epilogue
                s += ('if(literal)' if literal else 'else') + ' asm volatile(' + json.dumps(assembly) + ':"=&s"(saved),"=&s"(before),"=&s"(after),"=&s"(field):"s"(seed),"s"(source));\n'
            s += "break;\n"
s += r'''}
  if(threadIdx.x==0){out[3*index]=before;out[3*index+1]=after;out[3*index+2]=field;}
}
int main(){uint32_t *device; uint32_t host[3*1024];
 HIP(hipMalloc(&device,sizeof(host)));FILE *f=fopen("capture.bin","wb");if(!f)return 2;
 const uint32_t sources[]={0,0xffffffff,0xa5a5a5a5,0x5a5a5a5a,0xffffffff};
 for(unsigned reg=0;reg<2;++reg)for(unsigned seed=0;seed<2;++seed)for(unsigned pattern=0;pattern<5;++pattern){
 hipLaunchKernelGGL(probe,dim3(1024),dim3(32),0,0,device,reg,seed?0xffffffff:0,sources[pattern],pattern==4);
 HIP(hipGetLastError());HIP(hipDeviceSynchronize());HIP(hipMemcpy(host,device,sizeof(host),hipMemcpyDeviceToHost));
 if(fwrite(host,sizeof(host),1,f)!=1)return 3;}
 if(fclose(f))return 4;HIP(hipFree(device));}
'''
(out / 'probe.cpp').write_text(s)
