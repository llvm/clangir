; RUN: opt -S -passes=inline %s | FileCheck %s
; RUN: opt -S -passes='cgscc(inline)' %s | FileCheck %s
; RUN: opt -S -passes='module-inline' %s | FileCheck %s

declare void @foo()
declare void @bar()

define void @callee(ptr %arg) {
  %cmp = icmp eq ptr %arg, null
  br i1 %cmp, label %expensive, label %done

; This block is designed to be too expensive to inline.  We can only inline
; callee if this block is known to be dead.
expensive:
  call void @foo()
  call void @foo()
  call void @foo()
  call void @foo()
  call void @foo()
  call void @foo()
  call void @foo()
  call void @foo()
  call void @foo()
  call void @foo()
  ret void

done:
  call void @bar()
  ret void
}

; Positive test - arg is known non null
define void @caller(ptr nonnull %arg) {
; CHECK-LABEL: @caller
; CHECK: call void @bar()
  call void @callee(ptr nonnull %arg)
  ret void
}

; Negative test - arg is not known to be non null
define void @caller2(ptr %arg) {
; CHECK-LABEL: @caller2
; CHECK: call void @callee(
  call void @callee(ptr %arg)
  ret void
}

