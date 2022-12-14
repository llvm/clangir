; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=ve | FileCheck %s
; RUN: llc < %s -mtriple=ve -relocation-model=pic \
; RUN:     | FileCheck %s --check-prefix=PIC

;;; Check stack frame allocation of a function which calls other functions
;;; under following conditions and combinations of them:
;;;   - access variable or not
;;;   - no stack object, a stack object using BP, or a stack object not using BP
;;;   - isPositionIndependent or not

@data = external global i8, align 1

; Function Attrs: nounwind
define ptr @test_frame0(ptr %0, ptr %1) {
; CHECK-LABEL: test_frame0:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s9, (, %s11)
; CHECK-NEXT:    st %s10, 8(, %s11)
; CHECK-NEXT:    or %s9, 0, %s11
; CHECK-NEXT:    lea %s11, -240(, %s11)
; CHECK-NEXT:    brge.l.t %s11, %s8, .LBB0_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    ld %s61, 24(, %s14)
; CHECK-NEXT:    or %s62, 0, %s0
; CHECK-NEXT:    lea %s63, 315
; CHECK-NEXT:    shm.l %s63, (%s61)
; CHECK-NEXT:    shm.l %s8, 8(%s61)
; CHECK-NEXT:    shm.l %s11, 16(%s61)
; CHECK-NEXT:    monc
; CHECK-NEXT:    or %s0, 0, %s62
; CHECK-NEXT:  .LBB0_2:
; CHECK-NEXT:    lea %s2, fun@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, fun@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
; CHECK-NEXT:    ld %s10, 8(, %s11)
; CHECK-NEXT:    ld %s9, (, %s11)
; CHECK-NEXT:    b.l.t (, %s10)
;
; PIC-LABEL: test_frame0:
; PIC:       # %bb.0:
; PIC-NEXT:    st %s9, (, %s11)
; PIC-NEXT:    st %s10, 8(, %s11)
; PIC-NEXT:    st %s15, 24(, %s11)
; PIC-NEXT:    st %s16, 32(, %s11)
; PIC-NEXT:    or %s9, 0, %s11
; PIC-NEXT:    lea %s11, -240(, %s11)
; PIC-NEXT:    brge.l.t %s11, %s8, .LBB0_2
; PIC-NEXT:  # %bb.1:
; PIC-NEXT:    ld %s61, 24(, %s14)
; PIC-NEXT:    or %s62, 0, %s0
; PIC-NEXT:    lea %s63, 315
; PIC-NEXT:    shm.l %s63, (%s61)
; PIC-NEXT:    shm.l %s8, 8(%s61)
; PIC-NEXT:    shm.l %s11, 16(%s61)
; PIC-NEXT:    monc
; PIC-NEXT:    or %s0, 0, %s62
; PIC-NEXT:  .LBB0_2:
; PIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; PIC-NEXT:    and %s15, %s15, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; PIC-NEXT:    lea %s12, fun@plt_lo(-24)
; PIC-NEXT:    and %s12, %s12, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s12, fun@plt_hi(%s16, %s12)
; PIC-NEXT:    bsic %s10, (, %s12)
; PIC-NEXT:    or %s11, 0, %s9
; PIC-NEXT:    ld %s16, 32(, %s11)
; PIC-NEXT:    ld %s15, 24(, %s11)
; PIC-NEXT:    ld %s10, 8(, %s11)
; PIC-NEXT:    ld %s9, (, %s11)
; PIC-NEXT:    b.l.t (, %s10)
  %3 = tail call ptr @fun(ptr %0, ptr %1)
  ret ptr %3
}

declare ptr @fun(ptr, ptr)

; Function Attrs: nounwind
define ptr @test_frame32(ptr %0) {
; CHECK-LABEL: test_frame32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s9, (, %s11)
; CHECK-NEXT:    st %s10, 8(, %s11)
; CHECK-NEXT:    or %s9, 0, %s11
; CHECK-NEXT:    lea %s11, -272(, %s11)
; CHECK-NEXT:    brge.l.t %s11, %s8, .LBB1_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    ld %s61, 24(, %s14)
; CHECK-NEXT:    or %s62, 0, %s0
; CHECK-NEXT:    lea %s63, 315
; CHECK-NEXT:    shm.l %s63, (%s61)
; CHECK-NEXT:    shm.l %s8, 8(%s61)
; CHECK-NEXT:    shm.l %s11, 16(%s61)
; CHECK-NEXT:    monc
; CHECK-NEXT:    or %s0, 0, %s62
; CHECK-NEXT:  .LBB1_2:
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    lea %s0, fun@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, fun@hi(, %s0)
; CHECK-NEXT:    lea %s0, 240(, %s11)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
; CHECK-NEXT:    ld %s10, 8(, %s11)
; CHECK-NEXT:    ld %s9, (, %s11)
; CHECK-NEXT:    b.l.t (, %s10)
;
; PIC-LABEL: test_frame32:
; PIC:       # %bb.0:
; PIC-NEXT:    st %s9, (, %s11)
; PIC-NEXT:    st %s10, 8(, %s11)
; PIC-NEXT:    st %s15, 24(, %s11)
; PIC-NEXT:    st %s16, 32(, %s11)
; PIC-NEXT:    or %s9, 0, %s11
; PIC-NEXT:    lea %s11, -272(, %s11)
; PIC-NEXT:    brge.l.t %s11, %s8, .LBB1_2
; PIC-NEXT:  # %bb.1:
; PIC-NEXT:    ld %s61, 24(, %s14)
; PIC-NEXT:    or %s62, 0, %s0
; PIC-NEXT:    lea %s63, 315
; PIC-NEXT:    shm.l %s63, (%s61)
; PIC-NEXT:    shm.l %s8, 8(%s61)
; PIC-NEXT:    shm.l %s11, 16(%s61)
; PIC-NEXT:    monc
; PIC-NEXT:    or %s0, 0, %s62
; PIC-NEXT:  .LBB1_2:
; PIC-NEXT:    or %s1, 0, %s0
; PIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; PIC-NEXT:    and %s15, %s15, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; PIC-NEXT:    lea %s12, fun@plt_lo(-24)
; PIC-NEXT:    and %s12, %s12, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s12, fun@plt_hi(%s16, %s12)
; PIC-NEXT:    lea %s0, 240(, %s11)
; PIC-NEXT:    bsic %s10, (, %s12)
; PIC-NEXT:    or %s11, 0, %s9
; PIC-NEXT:    ld %s16, 32(, %s11)
; PIC-NEXT:    ld %s15, 24(, %s11)
; PIC-NEXT:    ld %s10, 8(, %s11)
; PIC-NEXT:    ld %s9, (, %s11)
; PIC-NEXT:    b.l.t (, %s10)
  %2 = alloca [32 x i8], align 1
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %2)
  %3 = call ptr @fun(ptr nonnull %2, ptr %0)
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %2)
  ret ptr %3
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

; Function Attrs: nounwind
define ptr @test_align32(i32 signext %0, ptr nocapture readnone %1) {
; CHECK-LABEL: test_align32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s9, (, %s11)
; CHECK-NEXT:    st %s10, 8(, %s11)
; CHECK-NEXT:    st %s17, 40(, %s11)
; CHECK-NEXT:    or %s9, 0, %s11
; CHECK-NEXT:    lea %s11, -288(, %s11)
; CHECK-NEXT:    and %s11, %s11, (59)1
; CHECK-NEXT:    or %s17, 0, %s11
; CHECK-NEXT:    brge.l.t %s11, %s8, .LBB2_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    ld %s61, 24(, %s14)
; CHECK-NEXT:    or %s62, 0, %s0
; CHECK-NEXT:    lea %s63, 315
; CHECK-NEXT:    shm.l %s63, (%s61)
; CHECK-NEXT:    shm.l %s8, 8(%s61)
; CHECK-NEXT:    shm.l %s11, 16(%s61)
; CHECK-NEXT:    monc
; CHECK-NEXT:    or %s0, 0, %s62
; CHECK-NEXT:  .LBB2_2:
; CHECK-NEXT:    lea %s0, 15(, %s0)
; CHECK-NEXT:    and %s0, -16, %s0
; CHECK-NEXT:    lea %s1, __ve_grow_stack_align@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, __ve_grow_stack_align@hi(, %s1)
; CHECK-NEXT:    or %s1, -32, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    lea %s0, 240(, %s11)
; CHECK-NEXT:    lea %s0, 31(, %s0)
; CHECK-NEXT:    and %s1, -32, %s0
; CHECK-NEXT:    lea %s0, fun@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, fun@hi(, %s0)
; CHECK-NEXT:    lea %s0, 256(, %s17)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
; CHECK-NEXT:    ld %s17, 40(, %s11)
; CHECK-NEXT:    ld %s10, 8(, %s11)
; CHECK-NEXT:    ld %s9, (, %s11)
; CHECK-NEXT:    b.l.t (, %s10)
;
; PIC-LABEL: test_align32:
; PIC:       # %bb.0:
; PIC-NEXT:    st %s9, (, %s11)
; PIC-NEXT:    st %s10, 8(, %s11)
; PIC-NEXT:    st %s15, 24(, %s11)
; PIC-NEXT:    st %s16, 32(, %s11)
; PIC-NEXT:    st %s17, 40(, %s11)
; PIC-NEXT:    or %s9, 0, %s11
; PIC-NEXT:    lea %s11, -288(, %s11)
; PIC-NEXT:    and %s11, %s11, (59)1
; PIC-NEXT:    or %s17, 0, %s11
; PIC-NEXT:    brge.l.t %s11, %s8, .LBB2_2
; PIC-NEXT:  # %bb.1:
; PIC-NEXT:    ld %s61, 24(, %s14)
; PIC-NEXT:    or %s62, 0, %s0
; PIC-NEXT:    lea %s63, 315
; PIC-NEXT:    shm.l %s63, (%s61)
; PIC-NEXT:    shm.l %s8, 8(%s61)
; PIC-NEXT:    shm.l %s11, 16(%s61)
; PIC-NEXT:    monc
; PIC-NEXT:    or %s0, 0, %s62
; PIC-NEXT:  .LBB2_2:
; PIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; PIC-NEXT:    and %s15, %s15, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; PIC-NEXT:    lea %s0, 15(, %s0)
; PIC-NEXT:    and %s0, -16, %s0
; PIC-NEXT:    lea %s12, __ve_grow_stack_align@plt_lo(-24)
; PIC-NEXT:    and %s12, %s12, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s12, __ve_grow_stack_align@plt_hi(%s16, %s12)
; PIC-NEXT:    or %s1, -32, (0)1
; PIC-NEXT:    bsic %s10, (, %s12)
; PIC-NEXT:    lea %s0, 240(, %s11)
; PIC-NEXT:    lea %s0, 31(, %s0)
; PIC-NEXT:    and %s1, -32, %s0
; PIC-NEXT:    lea %s12, fun@plt_lo(-24)
; PIC-NEXT:    and %s12, %s12, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s12, fun@plt_hi(%s16, %s12)
; PIC-NEXT:    lea %s0, 256(, %s17)
; PIC-NEXT:    bsic %s10, (, %s12)
; PIC-NEXT:    or %s11, 0, %s9
; PIC-NEXT:    ld %s17, 40(, %s11)
; PIC-NEXT:    ld %s16, 32(, %s11)
; PIC-NEXT:    ld %s15, 24(, %s11)
; PIC-NEXT:    ld %s10, 8(, %s11)
; PIC-NEXT:    ld %s9, (, %s11)
; PIC-NEXT:    b.l.t (, %s10)
  %3 = alloca [32 x i8], align 32
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %3)
  %4 = sext i32 %0 to i64
  %5 = alloca i8, i64 %4, align 32
  %6 = call ptr @fun(ptr nonnull %3, ptr nonnull %5)
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %3)
  ret ptr %6
}

; Function Attrs: nounwind
define ptr @test_frame0_var(ptr %0, ptr %1) {
; CHECK-LABEL: test_frame0_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s9, (, %s11)
; CHECK-NEXT:    st %s10, 8(, %s11)
; CHECK-NEXT:    or %s9, 0, %s11
; CHECK-NEXT:    lea %s11, -240(, %s11)
; CHECK-NEXT:    brge.l.t %s11, %s8, .LBB3_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    ld %s61, 24(, %s14)
; CHECK-NEXT:    or %s62, 0, %s0
; CHECK-NEXT:    lea %s63, 315
; CHECK-NEXT:    shm.l %s63, (%s61)
; CHECK-NEXT:    shm.l %s8, 8(%s61)
; CHECK-NEXT:    shm.l %s11, 16(%s61)
; CHECK-NEXT:    monc
; CHECK-NEXT:    or %s0, 0, %s62
; CHECK-NEXT:  .LBB3_2:
; CHECK-NEXT:    lea %s2, data@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, data@hi(, %s2)
; CHECK-NEXT:    ld1b.zx %s2, (, %s2)
; CHECK-NEXT:    st1b %s2, (, %s0)
; CHECK-NEXT:    lea %s2, fun@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, fun@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
; CHECK-NEXT:    ld %s10, 8(, %s11)
; CHECK-NEXT:    ld %s9, (, %s11)
; CHECK-NEXT:    b.l.t (, %s10)
;
; PIC-LABEL: test_frame0_var:
; PIC:       # %bb.0:
; PIC-NEXT:    st %s9, (, %s11)
; PIC-NEXT:    st %s10, 8(, %s11)
; PIC-NEXT:    st %s15, 24(, %s11)
; PIC-NEXT:    st %s16, 32(, %s11)
; PIC-NEXT:    or %s9, 0, %s11
; PIC-NEXT:    lea %s11, -240(, %s11)
; PIC-NEXT:    brge.l.t %s11, %s8, .LBB3_2
; PIC-NEXT:  # %bb.1:
; PIC-NEXT:    ld %s61, 24(, %s14)
; PIC-NEXT:    or %s62, 0, %s0
; PIC-NEXT:    lea %s63, 315
; PIC-NEXT:    shm.l %s63, (%s61)
; PIC-NEXT:    shm.l %s8, 8(%s61)
; PIC-NEXT:    shm.l %s11, 16(%s61)
; PIC-NEXT:    monc
; PIC-NEXT:    or %s0, 0, %s62
; PIC-NEXT:  .LBB3_2:
; PIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; PIC-NEXT:    and %s15, %s15, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; PIC-NEXT:    lea %s2, data@got_lo
; PIC-NEXT:    and %s2, %s2, (32)0
; PIC-NEXT:    lea.sl %s2, data@got_hi(, %s2)
; PIC-NEXT:    ld %s2, (%s2, %s15)
; PIC-NEXT:    ld1b.zx %s2, (, %s2)
; PIC-NEXT:    st1b %s2, (, %s0)
; PIC-NEXT:    lea %s12, fun@plt_lo(-24)
; PIC-NEXT:    and %s12, %s12, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s12, fun@plt_hi(%s16, %s12)
; PIC-NEXT:    bsic %s10, (, %s12)
; PIC-NEXT:    or %s11, 0, %s9
; PIC-NEXT:    ld %s16, 32(, %s11)
; PIC-NEXT:    ld %s15, 24(, %s11)
; PIC-NEXT:    ld %s10, 8(, %s11)
; PIC-NEXT:    ld %s9, (, %s11)
; PIC-NEXT:    b.l.t (, %s10)
  %3 = load i8, ptr @data, align 1
  store i8 %3, ptr %0, align 1
  %4 = tail call ptr @fun(ptr nonnull %0, ptr %1)
  ret ptr %4
}

; Function Attrs: nounwind
define ptr @test_frame32_var(ptr %0) {
; CHECK-LABEL: test_frame32_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s9, (, %s11)
; CHECK-NEXT:    st %s10, 8(, %s11)
; CHECK-NEXT:    or %s9, 0, %s11
; CHECK-NEXT:    lea %s11, -272(, %s11)
; CHECK-NEXT:    brge.l.t %s11, %s8, .LBB4_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    ld %s61, 24(, %s14)
; CHECK-NEXT:    or %s62, 0, %s0
; CHECK-NEXT:    lea %s63, 315
; CHECK-NEXT:    shm.l %s63, (%s61)
; CHECK-NEXT:    shm.l %s8, 8(%s61)
; CHECK-NEXT:    shm.l %s11, 16(%s61)
; CHECK-NEXT:    monc
; CHECK-NEXT:    or %s0, 0, %s62
; CHECK-NEXT:  .LBB4_2:
; CHECK-NEXT:    lea %s1, data@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, data@hi(, %s1)
; CHECK-NEXT:    ld1b.zx %s2, (, %s1)
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    st1b %s2, 240(, %s11)
; CHECK-NEXT:    lea %s0, fun@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, fun@hi(, %s0)
; CHECK-NEXT:    lea %s0, 240(, %s11)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
; CHECK-NEXT:    ld %s10, 8(, %s11)
; CHECK-NEXT:    ld %s9, (, %s11)
; CHECK-NEXT:    b.l.t (, %s10)
;
; PIC-LABEL: test_frame32_var:
; PIC:       # %bb.0:
; PIC-NEXT:    st %s9, (, %s11)
; PIC-NEXT:    st %s10, 8(, %s11)
; PIC-NEXT:    st %s15, 24(, %s11)
; PIC-NEXT:    st %s16, 32(, %s11)
; PIC-NEXT:    or %s9, 0, %s11
; PIC-NEXT:    lea %s11, -272(, %s11)
; PIC-NEXT:    brge.l.t %s11, %s8, .LBB4_2
; PIC-NEXT:  # %bb.1:
; PIC-NEXT:    ld %s61, 24(, %s14)
; PIC-NEXT:    or %s62, 0, %s0
; PIC-NEXT:    lea %s63, 315
; PIC-NEXT:    shm.l %s63, (%s61)
; PIC-NEXT:    shm.l %s8, 8(%s61)
; PIC-NEXT:    shm.l %s11, 16(%s61)
; PIC-NEXT:    monc
; PIC-NEXT:    or %s0, 0, %s62
; PIC-NEXT:  .LBB4_2:
; PIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; PIC-NEXT:    and %s15, %s15, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; PIC-NEXT:    lea %s1, data@got_lo
; PIC-NEXT:    and %s1, %s1, (32)0
; PIC-NEXT:    lea.sl %s1, data@got_hi(, %s1)
; PIC-NEXT:    ld %s1, (%s1, %s15)
; PIC-NEXT:    ld1b.zx %s2, (, %s1)
; PIC-NEXT:    or %s1, 0, %s0
; PIC-NEXT:    st1b %s2, 240(, %s11)
; PIC-NEXT:    lea %s12, fun@plt_lo(-24)
; PIC-NEXT:    and %s12, %s12, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s12, fun@plt_hi(%s16, %s12)
; PIC-NEXT:    lea %s0, 240(, %s11)
; PIC-NEXT:    bsic %s10, (, %s12)
; PIC-NEXT:    or %s11, 0, %s9
; PIC-NEXT:    ld %s16, 32(, %s11)
; PIC-NEXT:    ld %s15, 24(, %s11)
; PIC-NEXT:    ld %s10, 8(, %s11)
; PIC-NEXT:    ld %s9, (, %s11)
; PIC-NEXT:    b.l.t (, %s10)
  %2 = alloca [32 x i8], align 1
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %2)
  %3 = load i8, ptr @data, align 1
  store i8 %3, ptr %2, align 1
  %4 = call ptr @fun(ptr nonnull %2, ptr %0)
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %2)
  ret ptr %4
}

; Function Attrs: nounwind
define ptr @test_align32_var(i32 signext %0, ptr nocapture readnone %1) {
; CHECK-LABEL: test_align32_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s9, (, %s11)
; CHECK-NEXT:    st %s10, 8(, %s11)
; CHECK-NEXT:    st %s17, 40(, %s11)
; CHECK-NEXT:    or %s9, 0, %s11
; CHECK-NEXT:    lea %s11, -288(, %s11)
; CHECK-NEXT:    and %s11, %s11, (59)1
; CHECK-NEXT:    or %s17, 0, %s11
; CHECK-NEXT:    brge.l.t %s11, %s8, .LBB5_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    ld %s61, 24(, %s14)
; CHECK-NEXT:    or %s62, 0, %s0
; CHECK-NEXT:    lea %s63, 315
; CHECK-NEXT:    shm.l %s63, (%s61)
; CHECK-NEXT:    shm.l %s8, 8(%s61)
; CHECK-NEXT:    shm.l %s11, 16(%s61)
; CHECK-NEXT:    monc
; CHECK-NEXT:    or %s0, 0, %s62
; CHECK-NEXT:  .LBB5_2:
; CHECK-NEXT:    lea %s0, 15(, %s0)
; CHECK-NEXT:    and %s0, -16, %s0
; CHECK-NEXT:    lea %s1, __ve_grow_stack_align@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, __ve_grow_stack_align@hi(, %s1)
; CHECK-NEXT:    or %s1, -32, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    lea %s0, 240(, %s11)
; CHECK-NEXT:    lea %s1, data@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, data@hi(, %s1)
; CHECK-NEXT:    ld1b.zx %s2, (, %s1)
; CHECK-NEXT:    lea %s0, 31(, %s0)
; CHECK-NEXT:    and %s1, -32, %s0
; CHECK-NEXT:    st1b %s2, (, %s1)
; CHECK-NEXT:    lea %s0, fun@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, fun@hi(, %s0)
; CHECK-NEXT:    lea %s0, 256(, %s17)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
; CHECK-NEXT:    ld %s17, 40(, %s11)
; CHECK-NEXT:    ld %s10, 8(, %s11)
; CHECK-NEXT:    ld %s9, (, %s11)
; CHECK-NEXT:    b.l.t (, %s10)
;
; PIC-LABEL: test_align32_var:
; PIC:       # %bb.0:
; PIC-NEXT:    st %s9, (, %s11)
; PIC-NEXT:    st %s10, 8(, %s11)
; PIC-NEXT:    st %s15, 24(, %s11)
; PIC-NEXT:    st %s16, 32(, %s11)
; PIC-NEXT:    st %s17, 40(, %s11)
; PIC-NEXT:    or %s9, 0, %s11
; PIC-NEXT:    lea %s11, -288(, %s11)
; PIC-NEXT:    and %s11, %s11, (59)1
; PIC-NEXT:    or %s17, 0, %s11
; PIC-NEXT:    brge.l.t %s11, %s8, .LBB5_2
; PIC-NEXT:  # %bb.1:
; PIC-NEXT:    ld %s61, 24(, %s14)
; PIC-NEXT:    or %s62, 0, %s0
; PIC-NEXT:    lea %s63, 315
; PIC-NEXT:    shm.l %s63, (%s61)
; PIC-NEXT:    shm.l %s8, 8(%s61)
; PIC-NEXT:    shm.l %s11, 16(%s61)
; PIC-NEXT:    monc
; PIC-NEXT:    or %s0, 0, %s62
; PIC-NEXT:  .LBB5_2:
; PIC-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; PIC-NEXT:    and %s15, %s15, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; PIC-NEXT:    lea %s0, 15(, %s0)
; PIC-NEXT:    and %s0, -16, %s0
; PIC-NEXT:    lea %s12, __ve_grow_stack_align@plt_lo(-24)
; PIC-NEXT:    and %s12, %s12, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s12, __ve_grow_stack_align@plt_hi(%s16, %s12)
; PIC-NEXT:    or %s1, -32, (0)1
; PIC-NEXT:    bsic %s10, (, %s12)
; PIC-NEXT:    lea %s0, data@got_lo
; PIC-NEXT:    and %s0, %s0, (32)0
; PIC-NEXT:    lea.sl %s0, data@got_hi(, %s0)
; PIC-NEXT:    ld %s0, (%s0, %s15)
; PIC-NEXT:    lea %s1, 240(, %s11)
; PIC-NEXT:    ld1b.zx %s0, (, %s0)
; PIC-NEXT:    lea %s1, 31(, %s1)
; PIC-NEXT:    and %s1, -32, %s1
; PIC-NEXT:    st1b %s0, (, %s1)
; PIC-NEXT:    lea %s12, fun@plt_lo(-24)
; PIC-NEXT:    and %s12, %s12, (32)0
; PIC-NEXT:    sic %s16
; PIC-NEXT:    lea.sl %s12, fun@plt_hi(%s16, %s12)
; PIC-NEXT:    lea %s0, 256(, %s17)
; PIC-NEXT:    bsic %s10, (, %s12)
; PIC-NEXT:    or %s11, 0, %s9
; PIC-NEXT:    ld %s17, 40(, %s11)
; PIC-NEXT:    ld %s16, 32(, %s11)
; PIC-NEXT:    ld %s15, 24(, %s11)
; PIC-NEXT:    ld %s10, 8(, %s11)
; PIC-NEXT:    ld %s9, (, %s11)
; PIC-NEXT:    b.l.t (, %s10)
  %3 = alloca [32 x i8], align 32
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %3)
  %4 = sext i32 %0 to i64
  %5 = alloca i8, i64 %4, align 32
  %6 = load i8, ptr @data, align 1
  store i8 %6, ptr %5, align 32
  %7 = call ptr @fun(ptr nonnull %3, ptr nonnull %5)
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %3)
  ret ptr %7
}
