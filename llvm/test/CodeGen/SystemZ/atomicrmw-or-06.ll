; Test 64-bit atomic ORs, z196 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Check OR of a variable.
define i64 @f1(i64 %dummy, ptr %src, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: laog %r2, %r4, 0(%r3)
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 %b seq_cst
  ret i64 %res
}

; Check OR of 1, which needs a temporary.
define i64 @f2(i64 %dummy, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: lghi [[TMP:%r[0-5]]], 1
; CHECK: laog %r2, [[TMP]], 0(%r3)
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 1 seq_cst
  ret i64 %res
}

; Check the high end of the LAOG range.
define i64 @f3(i64 %dummy, ptr %src, i64 %b) {
; CHECK-LABEL: f3:
; CHECK: laog %r2, %r4, 524280(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 65535
  %res = atomicrmw or ptr %ptr, i64 %b seq_cst
  ret i64 %res
}

; Check the next doubleword up, which needs separate address logic.
define i64 @f4(i64 %dummy, ptr %src, i64 %b) {
; CHECK-LABEL: f4:
; CHECK: agfi %r3, 524288
; CHECK: laog %r2, %r4, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 65536
  %res = atomicrmw or ptr %ptr, i64 %b seq_cst
  ret i64 %res
}

; Check the low end of the LAOG range.
define i64 @f5(i64 %dummy, ptr %src, i64 %b) {
; CHECK-LABEL: f5:
; CHECK: laog %r2, %r4, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 -65536
  %res = atomicrmw or ptr %ptr, i64 %b seq_cst
  ret i64 %res
}

; Check the next doubleword down, which needs separate address logic.
define i64 @f6(i64 %dummy, ptr %src, i64 %b) {
; CHECK-LABEL: f6:
; CHECK: agfi %r3, -524296
; CHECK: laog %r2, %r4, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 -65537
  %res = atomicrmw or ptr %ptr, i64 %b seq_cst
  ret i64 %res
}
