; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=thumbv8.1m.main -mattr=+mve.fp -verify-machineinstrs -o - %s | FileCheck %s

define arm_aapcs_vfpcc <4 x i32> @test_vldrwq_gather_base_wb_s32(ptr %addr) {
; CHECK-LABEL: test_vldrwq_gather_base_wb_s32:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vldrw.u32 q1, [r0]
; CHECK-NEXT:    vldrw.u32 q0, [q1, #80]!
; CHECK-NEXT:    vstrw.32 q1, [r0]
; CHECK-NEXT:    bx lr
entry:
  %0 = load <4 x i32>, ptr %addr, align 8
  %1 = tail call { <4 x i32>, <4 x i32> } @llvm.arm.mve.vldr.gather.base.wb.v4i32.v4i32(<4 x i32> %0, i32 80)
  %2 = extractvalue { <4 x i32>, <4 x i32> } %1, 1
  store <4 x i32> %2, ptr %addr, align 8
  %3 = extractvalue { <4 x i32>, <4 x i32> } %1, 0
  ret <4 x i32> %3
}

declare { <4 x i32>, <4 x i32> } @llvm.arm.mve.vldr.gather.base.wb.v4i32.v4i32(<4 x i32>, i32)

define arm_aapcs_vfpcc <4 x float> @test_vldrwq_gather_base_wb_f32(ptr %addr) {
; CHECK-LABEL: test_vldrwq_gather_base_wb_f32:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vldrw.u32 q1, [r0]
; CHECK-NEXT:    vldrw.u32 q0, [q1, #64]!
; CHECK-NEXT:    vstrw.32 q1, [r0]
; CHECK-NEXT:    bx lr
entry:
  %0 = load <4 x i32>, ptr %addr, align 8
  %1 = tail call { <4 x float>, <4 x i32> } @llvm.arm.mve.vldr.gather.base.wb.v4f32.v4i32(<4 x i32> %0, i32 64)
  %2 = extractvalue { <4 x float>, <4 x i32> } %1, 1
  store <4 x i32> %2, ptr %addr, align 8
  %3 = extractvalue { <4 x float>, <4 x i32> } %1, 0
  ret <4 x float> %3
}

declare { <4 x float>, <4 x i32> } @llvm.arm.mve.vldr.gather.base.wb.v4f32.v4i32(<4 x i32>, i32)

define arm_aapcs_vfpcc <2 x i64> @test_vldrdq_gather_base_wb_z_u64(ptr %addr, i16 zeroext %p) {
; CHECK-LABEL: test_vldrdq_gather_base_wb_z_u64:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmsr p0, r1
; CHECK-NEXT:    vldrw.u32 q1, [r0]
; CHECK-NEXT:    vpst
; CHECK-NEXT:    vldrdt.u64 q0, [q1, #656]!
; CHECK-NEXT:    vstrw.32 q1, [r0]
; CHECK-NEXT:    bx lr
entry:
  %0 = load <2 x i64>, ptr %addr, align 8
  %1 = zext i16 %p to i32
  %2 = tail call <2 x i1> @llvm.arm.mve.pred.i2v.v2i1(i32 %1)
  %3 = tail call { <2 x i64>, <2 x i64> } @llvm.arm.mve.vldr.gather.base.wb.predicated.v2i64.v2i64.v2i1(<2 x i64> %0, i32 656, <2 x i1> %2)
  %4 = extractvalue { <2 x i64>, <2 x i64> } %3, 1
  store <2 x i64> %4, ptr %addr, align 8
  %5 = extractvalue { <2 x i64>, <2 x i64> } %3, 0
  ret <2 x i64> %5
}

declare <4 x i1> @llvm.arm.mve.pred.i2v.v4i1(i32)
declare <2 x i1> @llvm.arm.mve.pred.i2v.v2i1(i32)

declare { <2 x i64>, <2 x i64> } @llvm.arm.mve.vldr.gather.base.wb.predicated.v2i64.v2i64.v2i1(<2 x i64>, i32, <2 x i1>)
