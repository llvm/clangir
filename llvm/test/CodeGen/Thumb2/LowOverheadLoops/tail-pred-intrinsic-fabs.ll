; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=thumbv8.1m.main-none-none-eabi -mattr=+mve.fp -verify-machineinstrs -tail-predication=enabled -o - %s | FileCheck %s

define arm_aapcs_vfpcc void @fabs(ptr noalias nocapture readonly %pSrcA, ptr noalias nocapture %pDst, i32 %blockSize) {
; CHECK-LABEL: fabs:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r7, lr}
; CHECK-NEXT:    push {r7, lr}
; CHECK-NEXT:    cmp r2, #0
; CHECK-NEXT:    it eq
; CHECK-NEXT:    popeq {r7, pc}
; CHECK-NEXT:  .LBB0_1: @ %vector.ph
; CHECK-NEXT:    dlstp.32 lr, r2
; CHECK-NEXT:  .LBB0_2: @ %vector.body
; CHECK-NEXT:    @ =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    vldrw.u32 q0, [r0], #16
; CHECK-NEXT:    vabs.f32 q0, q0
; CHECK-NEXT:    vstrw.32 q0, [r1], #16
; CHECK-NEXT:    letp lr, .LBB0_2
; CHECK-NEXT:  @ %bb.3: @ %while.end
; CHECK-NEXT:    pop {r7, pc}
entry:
  %cmp3 = icmp eq i32 %blockSize, 0
  br i1 %cmp3, label %while.end, label %vector.ph

vector.ph:                                        ; preds = %entry
  %n.rnd.up = add i32 %blockSize, 3
  %n.vec = and i32 %n.rnd.up, -4
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %next.gep = getelementptr float, ptr %pDst, i32 %index
  %next.gep13 = getelementptr float, ptr %pSrcA, i32 %index
  %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %index, i32 %blockSize)
  %wide.masked.load = call <4 x float> @llvm.masked.load.v4f32.p0(ptr %next.gep13, i32 4, <4 x i1> %active.lane.mask, <4 x float> undef)
  %0 = call fast <4 x float> @llvm.fabs.v4f32(<4 x float> %wide.masked.load)
  call void @llvm.masked.store.v4f32.p0(<4 x float> %0, ptr %next.gep, i32 4, <4 x i1> %active.lane.mask)
  %index.next = add i32 %index, 4
  %1 = icmp eq i32 %index.next, %n.vec
  br i1 %1, label %while.end, label %vector.body

while.end:                                        ; preds = %vector.body, %entry
  ret void
}

declare <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32, i32)

declare <4 x float> @llvm.masked.load.v4f32.p0(ptr, i32 immarg, <4 x i1>, <4 x float>)

declare <4 x float> @llvm.fabs.v4f32(<4 x float>)

declare void @llvm.masked.store.v4f32.p0(<4 x float>, ptr, i32 immarg, <4 x i1>)
