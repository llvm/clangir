; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s
;
; Test that DAGCombiner does not change the addressing as the displacements
; are known to be out of range. Only one addition is needed.

define void @fun(ptr %Src, ptr %Dst) {
; CHECK-LABEL: fun:
; CHECK:       # %bb.0:
; CHECK-NEXT:    aghi %r2, 4096
; CHECK-NEXT:    vl %v0, 0(%r2), 3
; CHECK-NEXT:    vst %v0, 0(%r3), 3
; CHECK-NEXT:    vl %v0, 16(%r2), 3
; CHECK-NEXT:    vst %v0, 0(%r3), 3
; CHECK-NEXT:    br %r14

  %splitgep = getelementptr i8, ptr %Src, i64 4096
  %V0 = load <2 x i64>, ptr %splitgep, align 8
  store volatile <2 x i64> %V0, ptr %Dst, align 8

  %1 = getelementptr i8, ptr %splitgep, i64 16
  %V1 = load <2 x i64>, ptr %1, align 8
  store volatile <2 x i64> %V1, ptr %Dst, align 8

  ret void
}
