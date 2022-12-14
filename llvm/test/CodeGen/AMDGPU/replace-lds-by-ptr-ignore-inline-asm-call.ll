; RUN: opt -S -mtriple=amdgcn--  -amdgpu-replace-lds-use-with-pointer -amdgpu-enable-lds-replace-with-pointer=true < %s | FileCheck %s

; DESCRIPTION:
;
; We do not know what to do with inline asm call, we ignore it, hence pointer replacement for
; @used_only_within_func does not take place.
;

; CHECK: @used_only_within_func = addrspace(3) global [4 x i32] undef, align 4
@used_only_within_func = addrspace(3) global [4 x i32] undef, align 4

; CHECK-NOT: @used_only_within_func.ptr

define void @f0(i32 %x) {
; CHECK-LABEL: entry:
; CHECK:   store i32 %x, ptr inttoptr (i64 add (i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @used_only_within_func to ptr) to i64), i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @used_only_within_func to ptr) to i64)) to ptr), align 4
; CHECK:   ret void
entry:
  store i32 %x, ptr inttoptr (i64 add (i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @used_only_within_func to ptr) to i64), i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @used_only_within_func to ptr) to i64)) to ptr), align 4
  ret void
}

define amdgpu_kernel void @k0() {
; CHECK-LABEL: entry:
; CHECK:   call i32 asm "s_mov_b32 $0, 0", "=s"()
; CHECK:   ret void
entry:
  call i32 asm "s_mov_b32 $0, 0", "=s"()
  ret void
}
