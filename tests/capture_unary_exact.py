#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a gfx1201 HIP raw unary-result probe.

Arguments: OUTPUT_DIR [rcp|sqrt|rsq], default rcp.
Compile probe.cpp with hipcc. Run `probe normalized` for all mantissas
at exponent 127 (also 128 for sqrt/rsq), or `probe` for [4 denormal policies][32 modifiers][8192 inputs].
Writes little-endian uint32 results to capture.bin. Modifiers: NEG bit0, ABS
bit1, OMOD bits2:3, CLAMP bit4. Guest rounding is nearest-even.
"""
import json
from pathlib import Path
import sys
op = sys.argv[2] if len(sys.argv) > 2 else "rcp"
if op not in ("rcp", "sqrt", "rsq"):
    raise SystemExit("expected rcp, sqrt or rsq")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
s = r'''#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#define HIP(x) do {auto e=(x); if(e!=hipSuccess){fprintf(stderr,"%s\n",hipGetErrorString(e));exit(1);}}while(0)
__device__ uint32_t input_word(unsigned i) {
  const uint32_t edge[]={0,0x80000000,1,0x80000001,0x7fffff,0x807fffff,0x800000,0x80800000,
    0x3f800000,0xbf800000,0x40000000,0xc0000000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,
    0x7f800001,0xff800001,0x7fc12345,0xffc12345,0x7e800000,0xfe800000,0x7e7fffff,0x7e800001,
    0x00800001,0x00ffffff,0x01000000,0x7e000000,0x7dffffff,0x7e000001,0x3f7fffff,0x3f800001};
  if(i<32)return edge[i];
  if(i<4096)return (i-32)*0x9e3779b9U;
  return ((i-4096)&1 ? 0x80000000 : 0) | (((i-4096)>>1)%254+1)<<23 | ((i*0x7395a831U)&0x7fffff);
}
__global__ void probe(uint32_t *out,unsigned n,bool normalized,unsigned modifier,uint32_t fp_mode) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  uint32_t a=normalized?0x3f800000+i:input_word(i),result;
  switch(modifier){
'''
for m in range(32):
    operand='abs(%2)' if m&2 else '%2'
    if m&1: operand='-'+operand
    inst='v_'+op+'_f32 %0, '+operand+(' clamp' if m&16 else '')+['',' mul:2',' mul:4',' div:2'][(m>>2)&3]
    assembly='s_getreg_b32 %1, hwreg(HW_REG_MODE)\ns_setreg_b32 hwreg(HW_REG_MODE), %3\n'+inst+'\ns_setreg_b32 hwreg(HW_REG_MODE), %1\n'
    s+=f'case {m}:{{uint32_t saved; asm volatile({json.dumps(assembly)} : "=&v"(result),"=&s"(saved):"v"(a),"s"(fp_mode));break;}}\n'
s+=r'''}out[i]=result;}
int main(int argc,char **argv){bool normalized=argc==2&&!strcmp(argv[1],"normalized");
 if(argc>2||(argc==2&&!normalized))return 2;
 unsigned n=normalized?1U<<23:8192;uint32_t *device;auto host=new uint32_t[n];
 HIP(hipMalloc(&device,n*4));FILE *f=fopen("capture.bin","wb");if(!f)return 2;
 for(unsigned policy=0;policy<(normalized?1:4);++policy)for(unsigned modifier=0;modifier<(normalized?1:32);++modifier){
 unsigned nibble=15-policy*5;
 hipLaunchKernelGGL(probe,dim3((n+255)/256),dim3(256),0,0,device,n,normalized,modifier,nibble<<4);
 HIP(hipGetLastError());HIP(hipDeviceSynchronize());HIP(hipMemcpy(host,device,n*4,hipMemcpyDeviceToHost));
 if(fwrite(host,4,n,f)!=n)return 3;}
 if(fclose(f))return 4;HIP(hipFree(device));delete[]host;}
'''
(out/'probe.cpp').write_text(s.replace('1U<<23:8192', '1U<<'+str(23 if op == 'rcp' else 24)+':8192'))
