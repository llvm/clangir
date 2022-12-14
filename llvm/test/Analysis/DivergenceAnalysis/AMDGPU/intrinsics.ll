; RUN: opt -mtriple amdgcn-- -passes='print<divergence>' -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -mtriple amdgcn-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; CHECK: DIVERGENT: %swizzle = call i32 @llvm.amdgcn.ds.swizzle(i32 %src, i32 100) #0
define amdgpu_kernel void @ds_swizzle(ptr addrspace(1) %out, i32 %src) #0 {
  %swizzle = call i32 @llvm.amdgcn.ds.swizzle(i32 %src, i32 100) #0
  store i32 %swizzle, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: DIVERGENT: %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false) #0
define amdgpu_kernel void @v_permlane16_b32(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #0 {
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false) #0
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false) #0
define amdgpu_kernel void @v_permlanex16_b32(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #0 {
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false) #0
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 1, i32 1, i32 1, i1 false) #0
define amdgpu_kernel void @update_dpp(ptr addrspace(1) %out, i32 %in1, i32 %in2) #0 {
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 1, i32 1, i32 1, i1 false) #0
  store i32 %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %in, i32 1, i32 1, i32 1, i1 true) #0
define amdgpu_kernel void @mov_dpp(ptr addrspace(1) %out, i32 %in) #0 {
  %tmp0 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %in, i32 1, i32 1, i32 1, i1 true) #0
  store i32 %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call i32 @llvm.amdgcn.mov.dpp8.i32(i32 %in, i32 1) #0
define amdgpu_kernel void @mov_dpp8(ptr addrspace(1) %out, i32 %in) #0 {
  %tmp0 = call i32 @llvm.amdgcn.mov.dpp8.i32(i32 %in, i32 1) #0
  store i32 %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call i32 @llvm.amdgcn.writelane(i32 0, i32 1, i32 2)
define amdgpu_kernel void @writelane(ptr addrspace(1) %out) #0 {
  %tmp0 = call i32 @llvm.amdgcn.writelane(i32 0, i32 1, i32 2)
  store i32 %tmp0, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.ds.swizzle(i32, i32) #1
declare i32 @llvm.amdgcn.permlane16(i32, i32, i32, i32, i1, i1) #1
declare i32 @llvm.amdgcn.permlanex16(i32, i32, i32, i32, i1, i1) #1
declare i32 @llvm.amdgcn.mov.dpp.i32(i32, i32, i32, i32, i1) #1
declare i32 @llvm.amdgcn.mov.dpp8.i32(i32, i32) #1
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32, i32, i32, i1) #1
declare i32 @llvm.amdgcn.writelane(i32, i32, i32) #1

attributes #0 = { nounwind convergent }
attributes #1 = { nounwind readnone convergent }
