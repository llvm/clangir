; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=wasm32-unknown-unknown | FileCheck %s

; Regression test for PR41149.

define void @mod() {
; CHECK-LABEL: mod:
; CHECK:         .functype mod () -> ()
; CHECK-NEXT:    .local i32
; CHECK-NEXT:  # %bb.0:
; CHECK-NEXT:    local.get 0
; CHECK-NEXT:    local.get 0
; CHECK-NEXT:    i32.load8_u 0
; CHECK-NEXT:    local.tee 0
; CHECK-NEXT:    local.get 0
; CHECK-NEXT:    i32.extend8_s
; CHECK-NEXT:    i32.const 7
; CHECK-NEXT:    i32.shr_s
; CHECK-NEXT:    local.tee 0
; CHECK-NEXT:    i32.xor
; CHECK-NEXT:    local.get 0
; CHECK-NEXT:    i32.sub
; CHECK-NEXT:    i32.store8 0
; CHECK-NEXT:    # fallthrough-return
  %tmp = load <4 x i8>, ptr undef
  %tmp2 = icmp slt <4 x i8> %tmp, zeroinitializer
  %tmp3 = sub <4 x i8> zeroinitializer, %tmp
  %tmp4 = select <4 x i1> %tmp2, <4 x i8> %tmp3, <4 x i8> %tmp
  store <4 x i8> %tmp4, ptr undef
  ret void
}
