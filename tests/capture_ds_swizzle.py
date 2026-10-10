# SPDX-License-Identifier: MIT
"""Generate an RX 9070/gfx1201 DS_SWIZZLE probe.

Usage: python3 capture_ds_swizzle.py OUTPUT_DIR [--wave64]
Compile probe.cpp with hipcc --offload-arch=gfx1201 -O2; add -mwavefrontsize64
for --wave64. Run it in OUTPUT_DIR. Writes uint32_t
results[8][len(offsets.json)][wave_size] to swizzle.bin. Source lane i contains
i+1; inactive destinations retain 0xdead0000+i. Normal tests require no GPU.
Covers every rotate/FFT mask, every quad selector, and bit-mode samples.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output_dir', type=Path)
parser.add_argument('--wave64', action='store_true')
args = parser.parse_args()
out = args.output_dir
out.mkdir(parents=True, exist_ok=True)
lanes = 64 if args.wave64 else 32
mask_type = f'uint{lanes}_t'
masks = ([0xffffffffffffffff, 0, 0xaaaaaaaaaaaaaaaa, 0x5555555555555555,
          1, 1 << 63, 0xffffffff, 0xffffffff00000000] if args.wave64 else
         [0xffffffff, 0, 0xaaaaaaaa, 0x55555555, 1, 0x80000000, 0xffff, 0xffff0000])
saved, exec_reg = ('s[0:1]', 'exec') if args.wave64 else ('s0', 'exec_lo')
clobbers = '"s0", "s1", "scc", "memory"' if args.wave64 else '"s0", "scc", "memory"'
offsets = list(range(0xc000, 0xc800)) + list(range(0xe000, 0xe020))
offsets += list(range(0x8000, 0x8100))
offsets += [31 | (xor << 10) for xor in range(32)]
(out / 'offsets.json').write_text(json.dumps(offsets))
source = r'''#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#define HIP(x) do {auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s\n",hipGetErrorString(e));exit(1);}}while(0)
'''
source += f'''__global__ void probe(uint32_t *out, {mask_type} mask) {{
  unsigned lane=threadIdx.x, a=lane+1, d=0xdead0000u+lane;
  switch(blockIdx.x) {{
'''
for i, offset in enumerate(offsets):
    asm = (f's_mov_b{lanes} {saved}, {exec_reg}\\n '
           f's_and_b{lanes} {exec_reg}, {saved}, %1\\n '
           f'ds_swizzle_b32 %0, %2 offset:{offset}\\n '
           f's_wait_dscnt 0\\n s_mov_b{lanes} {exec_reg}, {saved}')
    source += f'case {i}: asm volatile("{asm}" : "+v"(d) : "s"(mask), "v"(a) : {clobbers}); break;\n'
source += f'''}} out[blockIdx.x*{lanes}+lane]=d;
}}
int main() {{
 constexpr size_t n={len(offsets)}*{lanes};
 uint32_t *p; HIP(hipMalloc(&p,n*4)); auto h=new uint32_t[n];
 FILE *f=fopen("swizzle.bin","wb"); if(!f)return 2;
 const {mask_type} masks[]={{{','.join(hex(m)+('ULL' if args.wave64 else 'U') for m in masks)}}};
 for(auto mask:masks) {{
  hipLaunchKernelGGL(probe,dim3({len(offsets)}),dim3({lanes}),0,0,p,mask);
  HIP(hipGetLastError()); HIP(hipDeviceSynchronize()); HIP(hipMemcpy(h,p,n*4,hipMemcpyDeviceToHost));
  if(fwrite(h,4,n,f)!=n)return 3;
 }}
 if(fclose(f))return 4; HIP(hipFree(p)); delete[]h;
}}
'''
(out / 'probe.cpp').write_text(source)
