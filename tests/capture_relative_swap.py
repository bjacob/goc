# SPDX-License-Identifier: MIT
"""Generate a gfx1201 V_SWAPREL_B32 probe: OUTPUT_DIR [32|64].

Compile with hipcc --offload-arch=gfx1201 -O2; add -mwavefrontsize64 for Wave64.
swap.bin stores [6 M0 values][4 EXEC masks][lanes][8 VGPRs].
"""
from pathlib import Path
import sys

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
lanes = int(sys.argv[2]) if len(sys.argv) > 2 else 32
assert lanes in (32, 64)
offsets = [0, 2 | (4 << 16), 1 << 16, 256, 256 << 16, 0xfc00fc00 | 2 | (4 << 16)]
masks = [0, 1, 0xaaaaaaaaaaaaaaaa, 0xffffffffffffffff]
s = '''#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#define HIP(x) do {auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s\\n",hipGetErrorString(e));exit(1);}}while(0)
__global__ void probe(uint32_t *out) {
unsigned result[8], lane=threadIdx.x;
switch(blockIdx.x) {
'''
for offset_index, m0 in enumerate(offsets):
    for mask_index, mask in enumerate(masks):
        code = ['s_mov_b32 s64, m0', 's_mov_b64 s[66:67], exec']
        code += [f'v_add_nc_u32 v{i}, {1000*i}, %8' for i in range(8)]
        code += [f's_mov_b32 m0, {m0}', f's_mov_b32 exec_lo, {mask & 0xffffffff}']
        if lanes == 64:
            code += [f's_mov_b32 exec_hi, {mask >> 32}']
        code += ['v_swaprel_b32 v0, v1', 's_mov_b64 exec, s[66:67]', 's_mov_b32 m0, s64']
        code += [f'v_mov_b32 %{i}, v{i}' for i in range(8)]
        outputs = ','.join(f'"=v"(result[{i}])' for i in range(8))
        clobbers = ','.join(f'"v{i}"' for i in range(8)) + ',"s64","s66","s67","m0","exec"'
        s += f'case {offset_index*4+mask_index}: asm volatile("' + '\\n'.join(code) + f'" : {outputs} : "v"(lane) : {clobbers});break;\n'
s += f'''}}
for(unsigned i=0;i<8;++i)out[(blockIdx.x*{lanes}+lane)*8+i]=result[i];
}}
int main(){{constexpr size_t n=24*{lanes}*8;uint32_t *p;HIP(hipMalloc(&p,n*4));auto h=new uint32_t[n];hipLaunchKernelGGL(probe,dim3(24),dim3({lanes}),0,0,p);HIP(hipGetLastError());HIP(hipDeviceSynchronize());HIP(hipMemcpy(h,p,n*4,hipMemcpyDeviceToHost));FILE*f=fopen("swap.bin","wb");if(!f)return 2;fwrite(h,4,n,f);fclose(f);HIP(hipFree(p));delete[]h;}}
'''
(out / 'probe.cpp').write_text(s)
