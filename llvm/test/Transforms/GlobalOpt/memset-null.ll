; RUN: opt -passes=globalopt -S < %s | FileCheck %s
; PR10047

%0 = type { i32, ptr, ptr }
%struct.A = type { [100 x i32] }

; CHECK: @a
@a = global %struct.A zeroinitializer, align 4
@llvm.global_ctors = appending global [2 x %0] [%0 { i32 65535, ptr @_GLOBAL__I_a, ptr null }, %0 { i32 65535, ptr @_GLOBAL__I_b, ptr null }]

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) nounwind

; CHECK-NOT: GLOBAL__I_a
define internal void @_GLOBAL__I_a() nounwind {
entry:
  tail call void @llvm.memset.p0.i64(ptr align 4 @a, i8 0, i64 400, i1 false) nounwind
  ret void
}

%struct.X = type { i8 }
@y = global ptr null, align 8
@x = global %struct.X zeroinitializer, align 1

define internal void @_GLOBAL__I_b() nounwind {
entry:
  %tmp.i.i.i = load ptr, ptr @y, align 8
  tail call void @llvm.memset.p0.i64(ptr %tmp.i.i.i, i8 0, i64 10, i1 false) nounwind
  ret void
}
