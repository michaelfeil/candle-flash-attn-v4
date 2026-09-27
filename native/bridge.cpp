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
int __tvm_ffi_fa4_bert_bf16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
int __tvm_ffi_fa4_modern_local_bf16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
int __tvm_ffi_fa4_qwen_bf16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);

}
thread_local std::string last_error;
extern "C" const char* candle_fa4_error_v4(){return last_error.c_str();}
struct StreamScope {
 int device;void* previous;
 StreamScope(int d,void* stream):device(d),previous(nullptr){if(TVMFFIEnvSetStream(kDLCUDA,d,stream,&previous))throw std::runtime_error("Unable to set TVM stream");}
 ~StreamScope(){TVMFFIEnvSetStream(kDLCUDA,device,previous,nullptr);}
};
// Export families: 0 global d64 MHA, 1 windowed d64 MHA, 2 causal d128 GQA4.
// Pointers remain owned by the caller; no device allocation or synchronization here.
extern "C" int candle_fa4_forward_v4(int mode,int dtype,int device,void* stream,void* q,void* k,void* v,void* o,void* offsets,void* offsets_k,int64_t total,int64_t total_k,int64_t batch,int64_t h,int64_t hk,double scale,int left,int right,const int64_t* strides){
 try{
  if(dtype<0||dtype>1||mode<0||mode>2||total<=0||total_k<=0||batch<=0)throw std::runtime_error("unsupported probe geometry");
  static tvm::ffi::Function functions[2][3] = {
    {tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_bert_fp16,nullptr),
     tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_modern_local_fp16,nullptr),
     tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_qwen_fp16,nullptr)},
    {tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_bert_bf16,nullptr),
     tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_modern_local_bf16,nullptr),
     tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_qwen_bf16,nullptr)}
  };
  if(h<=0||hk<=0||(!std::isfinite(scale)||scale<=0)||(mode!=2&&h!=hk)||(mode==2&&(h%hk||h/hk!=4))
     ||(mode==1&&(left<0||right<0)))throw std::runtime_error("unsupported attention parameters");
  auto& fn=functions[dtype][mode];
  DLDataType data_type{static_cast<uint8_t>(dtype==0?kDLFloat:kDLBfloat),16,1};
  int64_t d=mode==2?128:64;
  int64_t qs[]={total,h,d},ks[]={total_k,hk,d},cs[]={batch+1},st[12],cst[]={1};
  for(int j=0;j<12;j++)st[j]=strides[j];
  auto tensor=[&](void* p,int n,int64_t* sh,int64_t* stride,DLDataType dt){return DLTensor{p,{kDLCUDA,device},n,dt,sh,stride,0};};
  DLTensor qt=tensor(q,3,qs,st,data_type),kt=tensor(k,3,ks,st+3,data_type),vt=tensor(v,3,ks,st+6,data_type),ot=tensor(o,3,qs,st+9,data_type),ct=tensor(offsets,1,cs,cst,{kDLInt,32,1}),ckt=tensor(offsets_k,1,cs,cst,{kDLInt,32,1});
  tvm::ffi::Array<tvm::ffi::Any> aux{nullptr,nullptr};tvm::ffi::Any lb=mode==1?tvm::ffi::Any(int64_t(left)):tvm::ffi::Any(nullptr);
  tvm::ffi::Any rb=mode==1?tvm::ffi::Any(int64_t(right)):tvm::ffi::Any(nullptr);
  StreamScope scope(device,stream);
  fn(&qt,&kt,&vt,&ot,nullptr,scale,&ct,&ckt,nullptr,nullptr,nullptr,lb,rb,nullptr,nullptr,aux,nullptr,nullptr);
  last_error.clear();return 0;
 }catch(const std::exception& e){last_error=e.what();return -1;}
}
