# SPDX-License-Identifier: MIT
"""Generate a gfx1201 probe of scalar relative moves within ordinary SGPRs.

Compile probe.cpp with hipcc --offload-arch=gfx1201 -O2, then run it.
relative.bin stores [5 instructions][5 offsets][18 words]: S64..S81.
The five instructions are RELS32, RELS64, RELD32, RELD64, RELSD_2.
"""
from pathlib import Path
import sys

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
s = '''#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#define HIP(x) do {auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s\\n",hipGetErrorString(e));exit(1);}}while(0)
__global__ void probe(uint32_t *out) {
unsigned result[18];
switch(blockIdx.x) {
'''
ops = ['s_movrels_b32 s80, s64', 's_movrels_b64 s[80:81], s[64:65]',
       's_movreld_b32 s64, s80', 's_movreld_b64 s[64:65], s[80:81]',
       's_movrelsd_2_b32 s64, s64']
for op, instruction in enumerate(ops):
    for case in range(5):
        offset = case * 2 if case < 4 else 256
        if op == 4:
            offset |= (6 - case * 2 if case < 4 else 6) << 16
        lines = ['s_mov_b32 s82, m0', 's_mov_b64 s[84:85], s[0:1]',
                 's_mov_b32 s0, 0x11223344', 's_mov_b32 s1, 0x55667788'] + [f's_mov_b32 s{64+i}, {0x12340000+i}' for i in range(18)]
        lines += [f's_mov_b32 m0, {offset}', instruction, 's_nop 0']
        lines += [f'v_mov_b32 %{i}, s{64+i}' for i in range(18)]
        lines += ['s_mov_b32 m0, s82', 's_mov_b64 s[0:1], s[84:85]']
        code = '\\n'.join(lines)
        outputs = ','.join(f'"=v"(result[{i}])' for i in range(18))
        clobbers = ','.join(f'"s{i}"' for i in list(range(64, 83)) + [0, 1, 84, 85]) + ',"m0"'
        s += f'case {op*5+case}: asm volatile("{code}" : {outputs} : : {clobbers}); break;\n'
s += '''}
if(threadIdx.x==0)for(unsigned i=0;i<18;++i)out[blockIdx.x*18+i]=result[i];
}
int main(){constexpr size_t n=25*18;uint32_t *p;HIP(hipMalloc(&p,n*4));auto h=new uint32_t[n];hipLaunchKernelGGL(probe,dim3(25),dim3(32),0,0,p);HIP(hipGetLastError());HIP(hipDeviceSynchronize());HIP(hipMemcpy(h,p,n*4,hipMemcpyDeviceToHost));FILE*f=fopen("relative.bin","wb");if(!f)return 2;fwrite(h,4,n,f);fclose(f);HIP(hipFree(p));delete[]h;}
'''
(out / 'probe.cpp').write_text(s)
