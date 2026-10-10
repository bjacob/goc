# SPDX-License-Identifier: MIT
"""Generate a gfx1201 Wave32 MOV_B16 modifier probe.

Usage: python3 capture_half_move.py OUTPUT_DIR
Compile probe.cpp with hipcc --offload-arch=gfx1201 -O2 and run in OUTPUT_DIR.
move.bin contains uint32_t[32][4][32]: ABS/NEG/OMOD/CLAMP, half selectors,
and lane. Raw VOP3 encoding is used because LLVM rejects modifier syntax.
"""
from pathlib import Path
import sys
out=Path(sys.argv[1]);out.mkdir(parents=True, exist_ok=True)
s='''#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#define HIP(x) do {auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s\\n",hipGetErrorString(e));exit(1);}}while(0)
__global__ void probe(uint32_t *out) {
unsigned lane=threadIdx.x;
const unsigned values[]={0,0x8000,1,0x8001,0x3ff,0x83ff,0x400,0x8400,0x3c00,0xbc00,0x3800,0xb800,0x4000,0xc000,0x7bff,0xfbff,0x7c00,0xfc00,0x7c01,0xfc01,0x7e01,0xfe01,0x3555,0xb555,0x3bff,0xbbff,0x3c01,0xbc01,0x7bfe,0xfbfe,0x2aaa,0xaaaa};
unsigned a=values[lane]|(values[(lane+7)%32]<<16),d=0xabcd1234;
switch(blockIdx.x){
'''
for mode in range(32):
 for halves in range(4):
  w0=0xd59c0000|((mode&1)<<8)|(((mode>>4)&1)<<15)|((halves&1)<<11)|(((halves>>1)&1)<<14)
  w1=0x02010101|(((mode>>1)&1)<<29)|(((mode>>2)&3)<<27)
  s+=f'case {mode*4+halves}: asm volatile("v_mov_b32 v0, %0\\n v_mov_b32 v1, %1\\n .long {w0}, {w1}\\n v_mov_b32 %0, v0" : "+v"(d) : "v"(a) : "v0","v1");break;\n'
s+='''}out[blockIdx.x*32+lane]=d;
}
int main(){constexpr size_t n=128*32;uint32_t *p;HIP(hipMalloc(&p,n*4));auto h=new uint32_t[n];hipLaunchKernelGGL(probe,dim3(128),dim3(32),0,0,p);HIP(hipGetLastError());HIP(hipDeviceSynchronize());HIP(hipMemcpy(h,p,n*4,hipMemcpyDeviceToHost));FILE*f=fopen("move.bin","wb");if(!f)return 2;fwrite(h,4,n,f);fclose(f);HIP(hipFree(p));delete[]h;}
'''
(out/'probe.cpp').write_text(s)
