import os,sys,pathlib,json
import argparse
parser=argparse.ArgumentParser(description="Compile-only FA4 AOT probe; does not establish runtime support")
parser.add_argument("arch", choices=["80", "86", "89", "90a", "100a", "120"])
parser.add_argument("output", type=pathlib.Path)
args=parser.parse_args()
arch=args.arch
os.environ['FLASH_ATTENTION_ARCH']='sm_'+arch
os.environ['CUTE_DSL_ARCH']='sm_'+arch
os.environ['FLASH_ATTENTION_NUM_SMS']='132'
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
import flash_attn.cute.interface as i
out=args.output;out.mkdir(parents=True,exist_ok=True)
with FakeTensorMode():
 q=torch.empty((256,16,64),device='cuda',dtype=torch.float16)
 cu=torch.empty((3,),device='cuda',dtype=torch.int32)
 i.flash_attn_varlen_func(q,q,q,cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=129,max_seqlen_k=129)
for n,f in enumerate(i._flash_attn_fwd.compile_cache.cache.values()):
 f.export_to_c(str(out/f'probe{n}.o'),f'probe{n}')
print(json.dumps({'arch':arch,'exports':len(i._flash_attn_fwd.compile_cache.cache),'executed':False}),flush=True)
