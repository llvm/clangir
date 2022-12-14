; RUN: llc -mtriple=aarch64-linux-gnu %s -o - | FileCheck %s

define i8 @byval_match(ptr byval(i8) align 1, ptr byval(i8) %ptr) {
; CHECK-LABEL: byval_match:
; CHECK: ldrb w0, [sp, #8]
  %res = load i8, ptr %ptr
  ret i8 %res
}

define void @caller_match(ptr %p0, ptr %p1) {
; CHECK-LABEL: caller_match:
; CHECK: ldrb [[P1:w[0-9]+]], [x1]
; CHECK: strb [[P1]], [sp, #8]
; CHECK: ldrb [[P0:w[0-9]+]], [x0]
; CHECK: strb [[P0]], [sp]
; CHECK: bl byval_match
  call i8 @byval_match(ptr byval(i8) align 1 %p0, ptr byval(i8) %p1)
  ret void
}

define i8 @byval_large(ptr byval([3 x i64]) align 8, ptr byval(i8) %ptr) {
; CHECK-LABEL: byval_large:
; CHECK: ldrb w0, [sp, #24]
  %res = load i8, ptr %ptr
  ret i8 %res
}

define void @caller_large(ptr %p0, ptr %p1) {
; CHECK-LABEL: caller_large:
; CHECK: ldr [[P0HI:x[0-9]+]], [x0, #16]
; CHECK: ldr [[P0LO:q[0-9]+]], [x0]
; CHECK: str [[P0HI]], [sp, #16]
; CHECK: str [[P0LO]], [sp]
; CHECK: bl byval_large
  call i8 @byval_large(ptr byval([3 x i64]) align 8 %p0, ptr byval(i8) %p1)
  ret void
}
