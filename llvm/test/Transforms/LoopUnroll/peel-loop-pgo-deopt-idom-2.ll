; REQUIRES: asserts
; RUN: opt < %s -S -debug-only=loop-unroll -passes=loop-unroll -unroll-runtime 2>&1 | FileCheck %s
; RUN: opt < %s -S -debug-only=loop-unroll -passes='require<profile-summary>,function(require<opt-remark-emit>,loop-unroll)' 2>&1 | FileCheck %s

; Regression test for setting the correct idom for exit blocks.

; CHECK: Loop Unroll: F[basic]
; CHECK: PEELING loop %for.body with iteration count 2!

define i32 @basic(ptr %p, i32 %k, i1 %c1, i1 %c2) #0 !prof !3 {
entry:
  br label %for.body

for.body:
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %latch ]
  %p.addr.04 = phi ptr [ %p, %entry ], [ %incdec.ptr, %latch ]
  %incdec.ptr = getelementptr inbounds i32, ptr %p.addr.04, i32 1
  store i32 %i.05, ptr %p.addr.04, align 4
  %inc = add nsw i32 %i.05, 1
  %cmp = icmp slt i32 %inc, %k
  br i1 %c1, label %left, label %right

left:
  br label %latch

right:
  br i1 %c1, label %latch, label %side_exit, !prof !2

latch:
  br i1 %cmp, label %for.body, label %for.end, !prof !1

for.end:
  ret i32 %inc

side_exit:
  %rval = call i32(...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 %inc) ]
  ret i32 %rval
}

declare i32 @llvm.experimental.deoptimize.i32(...)

attributes #0 = { nounwind }

!1 = !{!"branch_weights", i32 1, i32 1}
!2 = !{!"branch_weights", i32 1, i32 0}
!3 = !{!"function_entry_count", i64 1}
