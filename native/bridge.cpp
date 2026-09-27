#include <tvm/ffi/function.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <cmath>
#include <string>
extern "C" {
int __tvm_ffi_fa4_bert_fp16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
int __tvm_ffi_fa4_modern_local_fp16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
int __tvm_ffi_fa4_qwen_fp16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
}
thread_local std::string last_error;
extern "C" const char* fa4_probe_error(){return last_error.c_str();}
struct StreamScope {
 int device;void* previous;
 StreamScope(int d,void* stream):device(d),previous(nullptr){if(TVMFFIEnvSetStream(kDLCUDA,d,stream,&previous))throw std::runtime_error("Unable to set TVM stream");}
 ~StreamScope(){TVMFFIEnvSetStream(kDLCUDA,device,previous,nullptr);}
};
// Experimental fixed specializations: 0 BERT, 1 ModernBERT local64, 2 Qwen GQA4, 3 ModernBERT global.
// Pointers remain owned by the caller; no device allocation or synchronization here.
extern "C" int fa4_probe_forward(int mode,int device,void* stream,void* q,void* k,void* v,void* o,void* offsets,int64_t total,int64_t batch,const int64_t* strides){
 try{
  if(mode<0||mode>3||total<=0||batch<=0)throw std::runtime_error("unsupported probe geometry");
  static auto bert=tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_bert_fp16,nullptr);
  static auto modern=tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_modern_local_fp16,nullptr);
  static auto qwen=tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_qwen_fp16,nullptr);
  // Global d64 MHA export accepts a runtime head count (12 or 16).
  auto& fn=mode==0?bert:mode==1?modern:mode==2?qwen:bert;
  int64_t h=mode==0?16:mode==2?32:12,hk=mode==2?8:h,d=mode==2?128:64;
  int64_t qs[]={total,h,d},ks[]={total,hk,d},cs[]={batch+1},st[12],cst[]={1};
  for(int j=0;j<12;j++)st[j]=strides[j];
  auto tensor=[&](void* p,int n,int64_t* sh,int64_t* stride,DLDataType dt){return DLTensor{p,{kDLCUDA,device},n,dt,sh,stride,0};};
  DLTensor qt=tensor(q,3,qs,st,{kDLFloat,16,1}),kt=tensor(k,3,ks,st+3,{kDLFloat,16,1}),vt=tensor(v,3,ks,st+6,{kDLFloat,16,1}),ot=tensor(o,3,qs,st+9,{kDLFloat,16,1}),ct=tensor(offsets,1,cs,cst,{kDLInt,32,1});
  tvm::ffi::Array<tvm::ffi::Any> aux{nullptr,nullptr};tvm::ffi::Any bound=mode==1?tvm::ffi::Any(int64_t(64)):tvm::ffi::Any(nullptr);
  StreamScope scope(device,stream);
  fn(&qt,&kt,&vt,&ot,nullptr,1./std::sqrt(double(d)),&ct,&ct,nullptr,nullptr,nullptr,bound,bound,nullptr,nullptr,aux,nullptr,nullptr);
  last_error.clear();return 0;
 }catch(const std::exception& e){last_error=e.what();return -1;}
}
