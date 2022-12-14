; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc %s -o - -fast-isel=true -O1 -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios8.0.0"

; The machine verifier was asserting on this test because the AND instruction was
; sunk below the test which killed %tmp340.
; The kill flags on the test had to be cleared because the AND was going to read
; registers in a BB after the test instruction.

define i32 @test(ptr %ptr) {
; CHECK-LABEL: test:
; CHECK:       ; %bb.0: ; %bb
; CHECK-NEXT:    mov x8, x0
; CHECK-NEXT:    mov w9, wzr
; CHECK-NEXT:  LBB0_1: ; %.thread
; CHECK-NEXT:    ; =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    lsr w11, w9, #1
; CHECK-NEXT:    sub w10, w9, #1
; CHECK-NEXT:    mov w9, w11
; CHECK-NEXT:    tbnz w10, #0, LBB0_1
; CHECK-NEXT:  ; %bb.2: ; %bb343
; CHECK-NEXT:    and w9, w10, #0x1
; CHECK-NEXT:    mov w0, #-1
; CHECK-NEXT:    str w9, [x8]
; CHECK-NEXT:    ret
bb:
  br label %.thread

.thread:                                          ; preds = %.thread, %bb
  %loc = phi i32 [ %next_iter, %.thread ], [ 0, %bb ]
  %next_iter = lshr i32 %loc, 1
  %tmp340 = sub i32 %loc, 1
  %tmp341 = and i32 %tmp340, 1
  %tmp342 = icmp eq i32 %tmp341, 0
  br i1 %tmp342, label %bb343, label %.thread

bb343:                                            ; preds = %.thread
  store i32 %tmp341, ptr %ptr, align 4
  ret i32 -1
}
