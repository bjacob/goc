#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a gfx1201 HIP exception probe and its deterministic input corpus.

Usage: capture_rdna4_arithmetic_exceptions.py OUTPUT_DIR {div_fixup,fma} {16,32,64}
FMA currently uses width 16 only. Compile probe.cpp with hipcc for gfx1201 and
run from OUTPUT_DIR. capture.bin contains little-endian uint32 records
[2 saturation settings][16 modifier combinations][8192 cases][result_lo,result_hi,flags].
MODE preserves denormals and selects nearest-even. All 32 lanes use identical
inputs. The probe restores MODE; it clears EXCP_FLAG_USER before each instruction.
"""

from pathlib import Path
import json,struct,sys
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
family=sys.argv[2];width=int(sys.argv[3]);n=8192
frac,bias={16:(10,15),32:(23,127),64:(52,1023)}[width]
sign=1<<(width-1);inf=(2*bias+1)<<frac;one=bias<<frac
vals=[0,sign,1,sign|1,(1<<frac)-1,1<<frac,one,sign|one,one+(1<<frac),inf-1,inf-1-(1<<frac),inf,sign|inf,inf|1,sign|inf|(1<<(frac-1))|3,2<<frac]
inputs=[]
for i in range(n):
 if i<4096:row=[vals[(i>>8)&15],vals[(i>>4)&15],vals[i&15]]
 else:
  state=i*0x9e3779b97f4a7c15 & ((1<<64)-1);row=[]
  for j in range(3):
   state^=state>>12;state^=(state<<25)&((1<<64)-1);state^=state>>27
   row.append((state*0x2545f4914f6cdd1d)&((1<<width)-1))
 inputs+=row
(out/'inputs.bin').write_bytes(struct.pack('<'+'Q'*len(inputs),*inputs))
s=r'''#include <hip/hip_runtime.h>
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
 operands=['v[0:1]','v[2:3]','v[4:5]'] if width==64 else ['v0','v2','v4']
 if m&8:operands=['-|'+operands[0]+'|','|'+operands[1]+'|','-'+operands[2]]
 inst=f'v_{family}_f{width} '+('v[6:7]' if width==64 else 'v6')+', '+', '.join(operands)
 inst+=(' clamp' if m&4 else '')+['',' mul:2',' mul:4',' div:2'][m&3]
 asm=['s_getreg_b32 s7, hwreg(HW_REG_MODE)','s_setreg_b32 hwreg(HW_REG_MODE), %9','s_setreg_imm32_b32 hwreg(18), 0']
 asm += [f'v_mov_b32 v{j}, %{j+3}' for j in range(6)]
 asm += ['v_mov_b32 v6, 0','v_mov_b32 v7, 0','s_nop 2',inst,'s_waitcnt_depctr 0','s_nop 7','s_getreg_b32 %0, hwreg(18)','v_mov_b32 %1, v6','v_mov_b32 %2, v7','s_setreg_b32 hwreg(HW_REG_MODE), s7']
 s+='asm volatile(\n'+'\n'.join(json.dumps(x+'\n') for x in asm)+r'''
: "=&s"(status), "=&v"(lo), "=&v"(hi)
: "v"(uint32_t(a)),"v"(uint32_t(a>>32)),"v"(uint32_t(b)),"v"(uint32_t(b>>32)),"v"(uint32_t(c)),"v"(uint32_t(c>>32)),"s"(fp_mode)
: "s7","v0","v1","v2","v3","v4","v5","v6","v7");
'''
 s+=f'if(threadIdx.x==0){{out[({m}*{n}+i)*3]=lo;out[({m}*{n}+i)*3+1]=hi;out[({m}*{n}+i)*3+2]=status;}}\n'
s+='''}
int main(){constexpr size_t n=8192;uint64_t input[n*3];FILE*f=fopen("inputs.bin","rb");if(!f||fread(input,8,n*3,f)!=n*3)return 2;fclose(f);
uint64_t *a;uint32_t *out;constexpr size_t size=n*16*3;auto host=new uint32_t[size];HIP(hipMalloc(&a,sizeof(input)));HIP(hipMalloc(&out,size*4));HIP(hipMemcpy(a,input,sizeof(input),hipMemcpyHostToDevice));
f=fopen("capture.bin","wb");if(!f)return 2;for(unsigned sat=0;sat<2;++sat){hipLaunchKernelGGL(probe,dim3(n),dim3(32),0,0,a,out,0xf0u|(sat<<23));HIP(hipGetLastError());HIP(hipDeviceSynchronize());HIP(hipMemcpy(host,out,size*4,hipMemcpyDeviceToHost));if(fwrite(host,4,size,f)!=size)return 3;}if(fclose(f))return 4;HIP(hipFree(a));HIP(hipFree(out));delete[]host;}
'''
(out/'probe.cpp').write_text(s)
