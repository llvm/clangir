; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; When a function does a dynamic stack allocation, the function's stack size
; is reported as UINT64_MAX.

; CHECK:       .section .llvm_stackmaps
; CHECK-NEXT:  __LLVM_StackMaps:
; Header
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 0
; Num Functions
; CHECK-NEXT:   .long 1
; Num LargeConstants
; CHECK-NEXT:   .long 0
; Num Callsites
; CHECK-NEXT:   .long 1

; Functions and stack size
; CHECK-NEXT:   .quad f
; CHECK-NEXT:   .quad -1

define void @f(i32 %nelems) {
entry:
  %mem = alloca i32, i32 %nelems
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 0, i32 0, ptr %mem)
  ret void
}

declare void @llvm.experimental.stackmap(i64, i32, ...)
