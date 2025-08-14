// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -o - %s   \
// RUN: | FileCheck --check-prefix=LLVM %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -o - %s   \
// RUN: | FileCheck --check-prefix=CIR %s

#include "__clang_cuda_builtin_vars.h"

// LLVM: define{{.*}} void @_Z6kernelPi(ptr %0)
// CIR-LABEL: @_Z6kernelPi
__attribute__((global))
void kernel(int *out) {
  int i = 0;

  // out[i++] = threadIdx.x;
  // CIR-DISABLED:  cir.func linkonce_odr @_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv()
  // CIR-DISABLED:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.tid.x"
  // LLVM-DISABLED: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.tid.x()

  // out[i++] = threadIdx.y;
  // CIR-DISABLED:  cir.func linkonce_odr @_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_yEv()
  // CIR-DISABLED:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.tid.y"
  // LLVM-DISABLED: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.tid.y()

  // out[i++] = threadIdx.z;
  // CIR-DISABLED:  cir.func linkonce_odr @_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_zEv()
  // CIR-DISABLED:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.tid.z"
  // LLVM-DISABLED: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.tid.z()


  // out[i++] = blockIdx.x;
  // CIR-DISABLED:  cir.func linkonce_odr @_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv()
  // CIR-DISABLED:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.ctaid.x"
  // LLVM-DISABLED: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

  // out[i++] = blockIdx.y;
  // CIR-DISABLED:  cir.func linkonce_odr @_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_yEv()
  // CIR-DISABLED:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.ctaid.y"
  // LLVM-DISABLED: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()

  // out[i++] = blockIdx.z;
  // CIR-DISABLED:  cir.func linkonce_odr @_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_zEv()
  // CIR-DISABLED:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.ctaid.z"
  // LLVM-DISABLED: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()


  // out[i++] = blockDim.x;
  // CIR-DISABLED:  cir.func linkonce_odr @_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv()
  // CIR-DISABLED:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.ntid.x"
  // LLVM-DISABLED: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

  // out[i++] = blockDim.y;
  // CIR-DISABLED:  cir.func linkonce_odr @_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_yEv()
  // CIR-DISABLED:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.ntid.y"
  // LLVM-DISABLED: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ntid.y()

  // out[i++] = blockDim.z;
  // CIR-DISABLED:  cir.func linkonce_odr @_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_zEv()
  // CIR-DISABLED:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.ntid.z"
  // LLVM-DISABLED: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.ntid.z()


  // out[i++] = gridDim.x;
  // CIR-DISABLED:  cir.func linkonce_odr @_ZN24__cuda_builtin_gridDim_t17__fetch_builtin_xEv()
  // CIR-DISABLED:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.nctaid.x"
  // LLVM-DISABLED: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()

  // out[i++] = gridDim.y;
  // CIR-DISABLED:  cir.func linkonce_odr @_ZN24__cuda_builtin_gridDim_t17__fetch_builtin_yEv()
  // CIR-DISABLED:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.nctaid.y"
  // LLVM-DISABLED: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()

  // out[i++] = gridDim.z;
  // CIR-DISABLED:  cir.func linkonce_odr @_ZN24__cuda_builtin_gridDim_t17__fetch_builtin_zEv()
  // CIR-DISABLED:  cir.llvm.intrinsic "nvvm.read.ptx.sreg.nctaid.z"
  // LLVM-DISABLED: call{{.*}} i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()


  // out[i++] = warpSize;
  // CIR-DISABLED: [[REGISTER:%.*]] = cir.const #cir.int<32>
  // CIR-DISABLED: cir.store{{.*}} [[REGISTER]]
  // LLVM-DISABLED: store i32 32,


  // CIR-DISABLED: cir.return loc
  // LLVM-DISABLED: ret void
}
