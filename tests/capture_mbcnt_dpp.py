# SPDX-License-Identifier: MIT
"""Generate gfx1201 Wave64 MBCNT DPP probes into OUTPUT_DIR.

Compile with hipcc --offload-arch=gfx1201 -mwavefrontsize64 -O2.
mbcnt.bin stores [2 operations][7 descriptors][4 EXEC masks][64 lanes].
"""
from pathlib import Path
import sys

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
modes = [0x539770100000000, 0x539770300000000, 0x1ff010400000000,
         0x1ff010c00000000, 0x1ff010600000000, 0x1ff010e00000000, 0xa3670c00000000]
masks = [0, 1 << 63, 0xaaaaaaaa55555555, 0xffffffffffffffff]
s = '''#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#define HIP(x) do {auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s\\n",hipGetErrorString(e));exit(1);}}while(0)
__global__ void probe(uint32_t *out) {
unsigned lane=threadIdx.x,a=(0x85930271u*lane)^0xffffffffu,b=0xffffffe0u+lane,d=0xdead0000u+lane;
switch(blockIdx.x) {
'''
for op, mnemonic in enumerate(['v_mbcnt_lo_u32_b32', 'v_mbcnt_hi_u32_b32']):
    for m, mode in enumerate(modes):
        fi = (mode >> 33) & 1
        if mode & (1 << 32):
            descriptor = f'dpp8:[7,6,5,4,3,2,1,0] fi:{fi}'
        else:
            control = (mode >> 40) & 511
            route = 'row_shl:1' if control == 0x101 else f'row_xmask:{control & 15}'
            descriptor = f'{route} row_mask:{(mode >> 49) & 15} bank_mask:{(mode >> 53) & 15} bound_ctrl:{(mode >> 35) & 1} fi:{fi}'
        for k, exec_mask in enumerate(masks):
            code = ['s_mov_b64 s[64:65], exec', f's_mov_b32 exec_lo, {exec_mask & 0xffffffff}',
                    f's_mov_b32 exec_hi, {exec_mask >> 32}',
                    f'{mnemonic} %0, %1, %2 {descriptor}', 's_mov_b64 exec, s[64:65]']
            s += f'case {(op*7+m)*4+k}: asm volatile("' + '\\n'.join(code) + '" : "+v"(d) : "v"(a), "v"(b) : "s64","s65","exec");break;\n'
s += '''}out[blockIdx.x*64+lane]=d;
}
int main(){constexpr size_t n=56*64;uint32_t *p;HIP(hipMalloc(&p,n*4));auto h=new uint32_t[n];hipLaunchKernelGGL(probe,dim3(56),dim3(64),0,0,p);HIP(hipGetLastError());HIP(hipDeviceSynchronize());HIP(hipMemcpy(h,p,n*4,hipMemcpyDeviceToHost));FILE*f=fopen("mbcnt.bin","wb");if(!f)return 2;fwrite(h,4,n,f);fclose(f);HIP(hipFree(p));delete[]h;}
'''
(out / 'probe.cpp').write_text(s)
