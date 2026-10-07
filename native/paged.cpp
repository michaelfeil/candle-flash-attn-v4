// Included by bridge.cpp so all exports share stream and first-launch handling.
extern "C" {
int __tvm_ffi_fa4_qwen_paged_gqa2_fp16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
int __tvm_ffi_fa4_qwen_paged_gqa4_fp16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
int __tvm_ffi_fa4_qwen_paged_gqa2_bf16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
int __tvm_ffi_fa4_qwen_paged_gqa4_bf16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
}
extern "C" int candle_fa4_paged_v1(int mode,int dtype,int device,void* stream,
 void* q,void* k,void* v,void* o,void* offsets,void* used,void* table,
 int64_t total,int64_t pages,int64_t batch,int64_t columns,int64_t h,int64_t hk,
 double scale,const int64_t* strides) {
 try {
  if(mode<0||mode>1||dtype<0||dtype>1||total<=0||pages<=0||batch<=0||columns<=0)
   throw std::runtime_error("invalid paged FA4 geometry");
  static tvm::ffi::Function functions[2][2] = {
   {tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_qwen_paged_gqa2_fp16,nullptr),tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_qwen_paged_gqa4_fp16,nullptr)},
   {tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_qwen_paged_gqa2_bf16,nullptr),tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_qwen_paged_gqa4_bf16,nullptr)}
  };
  DLDataType dt{static_cast<uint8_t>(dtype==0?kDLFloat:kDLBfloat),16,1}, i32{kDLInt,32,1};
  int64_t qs[]={total,h,128},ks[]={pages,64,hk,128},cs[]={batch+1},us[]={batch},ts[]={batch,columns},one[]={1},tst[]={columns,1};
  int64_t st[14];for(int j=0;j<14;j++)st[j]=strides[j];
  auto tensor=[&](void*p,int n,int64_t*sh,int64_t*stride,DLDataType type){return DLTensor{p,{kDLCUDA,device},n,type,sh,stride,0};};
  DLTensor qt=tensor(q,3,qs,st,dt),kt=tensor(k,4,ks,st+3,dt),vt=tensor(v,4,ks,st+7,dt),ot=tensor(o,3,qs,st+11,dt);
  DLTensor ct=tensor(offsets,1,cs,one,i32),ut=tensor(used,1,us,one,i32),pt=tensor(table,2,ts,tst,i32);
  tvm::ffi::Array<tvm::ffi::Any> aux{nullptr,nullptr};
  StreamScope scope(device,stream);
  invoke_export(12+dtype*2+mode,[&]{functions[dtype][mode](&qt,&kt,&vt,&ot,nullptr,scale,&ct,nullptr,nullptr,&ut,&pt,nullptr,nullptr,nullptr,nullptr,aux,nullptr,nullptr);});
  last_error.clear();return 0;
 }catch(const std::exception&e){last_error=e.what();return -1;}
}
