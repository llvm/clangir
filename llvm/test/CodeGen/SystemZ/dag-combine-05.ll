; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
;
; Test that DAGCombiner does not produce an addcarry node if the carry
; producer is not legal. This can happen e.g. with an uaddo with a type
; that is not legal.
;
; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 < %s | FileCheck %s

define void @fun(i16 %arg0, ptr %src, ptr %dst) {
; CHECK-LABEL: fun:
; CHECK:       # %bb.0: # %bb
; CHECK-NEXT:    llhr %r0, %r2
; CHECK-NEXT:    llh %r2, 0(%r3)
; CHECK-NEXT:    chi %r0, 9616
; CHECK-NEXT:    lhi %r1, 0
; CHECK-NEXT:    lochil %r1, 1
; CHECK-NEXT:    afi %r2, 65535
; CHECK-NEXT:    llhr %r3, %r2
; CHECK-NEXT:    lhi %r0, 0
; CHECK-NEXT:    cr %r3, %r2
; CHECK-NEXT:    lochilh %r0, 1
; CHECK-NEXT:    ar %r0, %r1
; CHECK-NEXT:    st %r0, 0(%r4)
; CHECK-NEXT:    br %r14
bb:
  %tmp = icmp ult i16 %arg0, 9616
  %tmp1 = zext i1 %tmp to i32
  %tmp2 = load i16, ptr %src
  %0 = call { i16, i1 } @llvm.uadd.with.overflow.i16(i16 %tmp2, i16 -1)
  %math = extractvalue { i16, i1 } %0, 0
  %ov = extractvalue { i16, i1 } %0, 1
  %tmp5 = zext i1 %ov to i32
  %tmp6 = add nuw nsw i32 %tmp5, %tmp1
  store i32 %tmp6, ptr %dst
  ret void
}

declare { i16, i1 } @llvm.uadd.with.overflow.i16(i16, i16) #1
