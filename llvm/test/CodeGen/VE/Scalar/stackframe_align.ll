; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=ve | FileCheck %s
; RUN: llc < %s -mtriple=ve --frame-pointer=all \
; RUN:     | FileCheck %s --check-prefix=CHECKFP

;;; Check stack frame allocation with static and dynamic stack object with
;;; alignments as a test of getFrameIndexReference().

;; Allocated buffer places from 9 to 15 bytes in 16 bytes local vars area.

; Function Attrs: nounwind
define ptr @test_frame7(ptr %0) {
; CHECK-LABEL: test_frame7:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.l %s11, -16, %s11
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
; CHECK-NEXT:    ld1b.zx %s1, (, %s0)
; CHECK-NEXT:    lea %s0, 9(, %s11)
; CHECK-NEXT:    st1b %s1, 9(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
;
; CHECKFP-LABEL: test_frame7:
; CHECKFP:       # %bb.0:
; CHECKFP-NEXT:    st %s9, (, %s11)
; CHECKFP-NEXT:    st %s10, 8(, %s11)
; CHECKFP-NEXT:    or %s9, 0, %s11
; CHECKFP-NEXT:    lea %s11, -192(, %s11)
; CHECKFP-NEXT:    brge.l.t %s11, %s8, .LBB0_2
; CHECKFP-NEXT:  # %bb.1:
; CHECKFP-NEXT:    ld %s61, 24(, %s14)
; CHECKFP-NEXT:    or %s62, 0, %s0
; CHECKFP-NEXT:    lea %s63, 315
; CHECKFP-NEXT:    shm.l %s63, (%s61)
; CHECKFP-NEXT:    shm.l %s8, 8(%s61)
; CHECKFP-NEXT:    shm.l %s11, 16(%s61)
; CHECKFP-NEXT:    monc
; CHECKFP-NEXT:    or %s0, 0, %s62
; CHECKFP-NEXT:  .LBB0_2:
; CHECKFP-NEXT:    ld1b.zx %s1, (, %s0)
; CHECKFP-NEXT:    lea %s0, -7(, %s9)
; CHECKFP-NEXT:    st1b %s1, -7(, %s9)
; CHECKFP-NEXT:    or %s11, 0, %s9
; CHECKFP-NEXT:    ld %s10, 8(, %s11)
; CHECKFP-NEXT:    ld %s9, (, %s11)
; CHECKFP-NEXT:    b.l.t (, %s10)
  %2 = alloca [7 x i8], align 1
  %3 = load i8, ptr %0, align 1
  store i8 %3, ptr %2, align 1
  ret ptr %2
}

;; Allocated buffer is aligned by 8, so it places from 8 to 14 bytes in 16
;; bytes local vars area.

; Function Attrs: nounwind
define ptr @test_frame7_align8(ptr %0) {
; CHECK-LABEL: test_frame7_align8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.l %s11, -16, %s11
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
; CHECK-NEXT:    ld1b.zx %s1, (, %s0)
; CHECK-NEXT:    lea %s0, 8(, %s11)
; CHECK-NEXT:    st1b %s1, 8(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
;
; CHECKFP-LABEL: test_frame7_align8:
; CHECKFP:       # %bb.0:
; CHECKFP-NEXT:    st %s9, (, %s11)
; CHECKFP-NEXT:    st %s10, 8(, %s11)
; CHECKFP-NEXT:    or %s9, 0, %s11
; CHECKFP-NEXT:    lea %s11, -192(, %s11)
; CHECKFP-NEXT:    brge.l.t %s11, %s8, .LBB1_2
; CHECKFP-NEXT:  # %bb.1:
; CHECKFP-NEXT:    ld %s61, 24(, %s14)
; CHECKFP-NEXT:    or %s62, 0, %s0
; CHECKFP-NEXT:    lea %s63, 315
; CHECKFP-NEXT:    shm.l %s63, (%s61)
; CHECKFP-NEXT:    shm.l %s8, 8(%s61)
; CHECKFP-NEXT:    shm.l %s11, 16(%s61)
; CHECKFP-NEXT:    monc
; CHECKFP-NEXT:    or %s0, 0, %s62
; CHECKFP-NEXT:  .LBB1_2:
; CHECKFP-NEXT:    ld1b.zx %s1, (, %s0)
; CHECKFP-NEXT:    lea %s0, -8(, %s9)
; CHECKFP-NEXT:    st1b %s1, -8(, %s9)
; CHECKFP-NEXT:    or %s11, 0, %s9
; CHECKFP-NEXT:    ld %s10, 8(, %s11)
; CHECKFP-NEXT:    ld %s9, (, %s11)
; CHECKFP-NEXT:    b.l.t (, %s10)
  %2 = alloca [7 x i8], align 8
  %3 = load i8, ptr %0, align 1
  store i8 %3, ptr %2, align 1
  ret ptr %2
}

;; Allocated buffer is aligned by 16, so it places from 0 to 15 bytes in 16
;; bytes local vars area.

; Function Attrs: nounwind
define ptr @test_frame16_align16(ptr %0) {
; CHECK-LABEL: test_frame16_align16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.l %s11, -16, %s11
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
; CHECK-NEXT:    ld1b.zx %s1, (, %s0)
; CHECK-NEXT:    lea %s0, (, %s11)
; CHECK-NEXT:    st1b %s1, (, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
;
; CHECKFP-LABEL: test_frame16_align16:
; CHECKFP:       # %bb.0:
; CHECKFP-NEXT:    st %s9, (, %s11)
; CHECKFP-NEXT:    st %s10, 8(, %s11)
; CHECKFP-NEXT:    or %s9, 0, %s11
; CHECKFP-NEXT:    lea %s11, -192(, %s11)
; CHECKFP-NEXT:    brge.l.t %s11, %s8, .LBB2_2
; CHECKFP-NEXT:  # %bb.1:
; CHECKFP-NEXT:    ld %s61, 24(, %s14)
; CHECKFP-NEXT:    or %s62, 0, %s0
; CHECKFP-NEXT:    lea %s63, 315
; CHECKFP-NEXT:    shm.l %s63, (%s61)
; CHECKFP-NEXT:    shm.l %s8, 8(%s61)
; CHECKFP-NEXT:    shm.l %s11, 16(%s61)
; CHECKFP-NEXT:    monc
; CHECKFP-NEXT:    or %s0, 0, %s62
; CHECKFP-NEXT:  .LBB2_2:
; CHECKFP-NEXT:    ld1b.zx %s1, (, %s0)
; CHECKFP-NEXT:    lea %s0, -16(, %s9)
; CHECKFP-NEXT:    st1b %s1, -16(, %s9)
; CHECKFP-NEXT:    or %s11, 0, %s9
; CHECKFP-NEXT:    ld %s10, 8(, %s11)
; CHECKFP-NEXT:    ld %s9, (, %s11)
; CHECKFP-NEXT:    b.l.t (, %s10)
  %2 = alloca [16 x i8], align 16
  %3 = load i8, ptr %0, align 1
  store i8 %3, ptr %2, align 1
  ret ptr %2
}

;; Allocated buffer is aligned by 32, so it places from 0 to 15 bytes in 48
;; bytes local vars area.  Or it places from 192 (aligned to 32 bytes) to
;; 207 bytes in 224 + alpha allocated local vars area.

; Function Attrs: nounwind
define ptr @test_frame16_align32(ptr %0) {
; CHECK-LABEL: test_frame16_align32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s9, (, %s11)
; CHECK-NEXT:    st %s10, 8(, %s11)
; CHECK-NEXT:    or %s9, 0, %s11
; CHECK-NEXT:    lea %s11, -224(, %s11)
; CHECK-NEXT:    and %s11, %s11, (59)1
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
; CHECK-NEXT:    ld1b.zx %s1, (, %s0)
; CHECK-NEXT:    lea %s0, 192(, %s11)
; CHECK-NEXT:    st1b %s1, 192(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
; CHECK-NEXT:    ld %s10, 8(, %s11)
; CHECK-NEXT:    ld %s9, (, %s11)
; CHECK-NEXT:    b.l.t (, %s10)
;
; CHECKFP-LABEL: test_frame16_align32:
; CHECKFP:       # %bb.0:
; CHECKFP-NEXT:    st %s9, (, %s11)
; CHECKFP-NEXT:    st %s10, 8(, %s11)
; CHECKFP-NEXT:    or %s9, 0, %s11
; CHECKFP-NEXT:    lea %s11, -224(, %s11)
; CHECKFP-NEXT:    and %s11, %s11, (59)1
; CHECKFP-NEXT:    brge.l.t %s11, %s8, .LBB3_2
; CHECKFP-NEXT:  # %bb.1:
; CHECKFP-NEXT:    ld %s61, 24(, %s14)
; CHECKFP-NEXT:    or %s62, 0, %s0
; CHECKFP-NEXT:    lea %s63, 315
; CHECKFP-NEXT:    shm.l %s63, (%s61)
; CHECKFP-NEXT:    shm.l %s8, 8(%s61)
; CHECKFP-NEXT:    shm.l %s11, 16(%s61)
; CHECKFP-NEXT:    monc
; CHECKFP-NEXT:    or %s0, 0, %s62
; CHECKFP-NEXT:  .LBB3_2:
; CHECKFP-NEXT:    ld1b.zx %s1, (, %s0)
; CHECKFP-NEXT:    lea %s0, 192(, %s11)
; CHECKFP-NEXT:    st1b %s1, 192(, %s11)
; CHECKFP-NEXT:    or %s11, 0, %s9
; CHECKFP-NEXT:    ld %s10, 8(, %s11)
; CHECKFP-NEXT:    ld %s9, (, %s11)
; CHECKFP-NEXT:    b.l.t (, %s10)
  %2 = alloca [16 x i8], align 32
  %3 = load i8, ptr %0, align 1
  store i8 %3, ptr %2, align 1
  ret ptr %2
}

;; Allocated buffer is aligned by 32, so it places from 0 to 31 bytes in 48
;; + alpha bytes local vars area, or it places from 192 (32 bytes aligned 176)
;; to 223 in 224 + alpha bytes local vars area..

; Function Attrs: nounwind
define ptr @test_frame32_align32(ptr %0) {
; CHECK-LABEL: test_frame32_align32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s9, (, %s11)
; CHECK-NEXT:    st %s10, 8(, %s11)
; CHECK-NEXT:    or %s9, 0, %s11
; CHECK-NEXT:    lea %s11, -224(, %s11)
; CHECK-NEXT:    and %s11, %s11, (59)1
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
; CHECK-NEXT:    ld1b.zx %s1, (, %s0)
; CHECK-NEXT:    lea %s0, 192(, %s11)
; CHECK-NEXT:    st1b %s1, 192(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
; CHECK-NEXT:    ld %s10, 8(, %s11)
; CHECK-NEXT:    ld %s9, (, %s11)
; CHECK-NEXT:    b.l.t (, %s10)
;
; CHECKFP-LABEL: test_frame32_align32:
; CHECKFP:       # %bb.0:
; CHECKFP-NEXT:    st %s9, (, %s11)
; CHECKFP-NEXT:    st %s10, 8(, %s11)
; CHECKFP-NEXT:    or %s9, 0, %s11
; CHECKFP-NEXT:    lea %s11, -224(, %s11)
; CHECKFP-NEXT:    and %s11, %s11, (59)1
; CHECKFP-NEXT:    brge.l.t %s11, %s8, .LBB4_2
; CHECKFP-NEXT:  # %bb.1:
; CHECKFP-NEXT:    ld %s61, 24(, %s14)
; CHECKFP-NEXT:    or %s62, 0, %s0
; CHECKFP-NEXT:    lea %s63, 315
; CHECKFP-NEXT:    shm.l %s63, (%s61)
; CHECKFP-NEXT:    shm.l %s8, 8(%s61)
; CHECKFP-NEXT:    shm.l %s11, 16(%s61)
; CHECKFP-NEXT:    monc
; CHECKFP-NEXT:    or %s0, 0, %s62
; CHECKFP-NEXT:  .LBB4_2:
; CHECKFP-NEXT:    ld1b.zx %s1, (, %s0)
; CHECKFP-NEXT:    lea %s0, 192(, %s11)
; CHECKFP-NEXT:    st1b %s1, 192(, %s11)
; CHECKFP-NEXT:    or %s11, 0, %s9
; CHECKFP-NEXT:    ld %s10, 8(, %s11)
; CHECKFP-NEXT:    ld %s9, (, %s11)
; CHECKFP-NEXT:    b.l.t (, %s10)
  %2 = alloca [32 x i8], align 32
  %3 = load i8, ptr %0, align 1
  store i8 %3, ptr %2, align 1
  ret ptr %2
}

;; Dynamically allocated buffer is aligned by 16, so it places from 0 to 31
;; bytes in allocated area, or it places from 240 (32 bytes aligned 176+64)
;; to 271 in allocated area (actually it places not newly allocated area
;; but in somewhere between newly allocated area and allocated area at the
;; prologue since VE ABI requires the reserved area at the top of stack).

;; FIXME: (size+15)/16*16 is not enough.

; Function Attrs: nounwind
define ptr @test_frame_dynalign16(ptr %0, i64 %1) {
; CHECK-LABEL: test_frame_dynalign16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s9, (, %s11)
; CHECK-NEXT:    st %s10, 8(, %s11)
; CHECK-NEXT:    or %s9, 0, %s11
; CHECK-NEXT:    lea %s11, -240(, %s11)
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
; CHECK-NEXT:    or %s2, 0, %s0
; CHECK-NEXT:    lea %s0, 15(, %s1)
; CHECK-NEXT:    and %s0, -16, %s0
; CHECK-NEXT:    lea %s1, __ve_grow_stack@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, __ve_grow_stack@hi(, %s1)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    lea %s0, 240(, %s11)
; CHECK-NEXT:    ld1b.zx %s1, (, %s2)
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
; CHECK-NEXT:    ld %s10, 8(, %s11)
; CHECK-NEXT:    ld %s9, (, %s11)
; CHECK-NEXT:    b.l.t (, %s10)
;
; CHECKFP-LABEL: test_frame_dynalign16:
; CHECKFP:       # %bb.0:
; CHECKFP-NEXT:    st %s9, (, %s11)
; CHECKFP-NEXT:    st %s10, 8(, %s11)
; CHECKFP-NEXT:    or %s9, 0, %s11
; CHECKFP-NEXT:    lea %s11, -240(, %s11)
; CHECKFP-NEXT:    brge.l.t %s11, %s8, .LBB5_2
; CHECKFP-NEXT:  # %bb.1:
; CHECKFP-NEXT:    ld %s61, 24(, %s14)
; CHECKFP-NEXT:    or %s62, 0, %s0
; CHECKFP-NEXT:    lea %s63, 315
; CHECKFP-NEXT:    shm.l %s63, (%s61)
; CHECKFP-NEXT:    shm.l %s8, 8(%s61)
; CHECKFP-NEXT:    shm.l %s11, 16(%s61)
; CHECKFP-NEXT:    monc
; CHECKFP-NEXT:    or %s0, 0, %s62
; CHECKFP-NEXT:  .LBB5_2:
; CHECKFP-NEXT:    or %s2, 0, %s0
; CHECKFP-NEXT:    lea %s0, 15(, %s1)
; CHECKFP-NEXT:    and %s0, -16, %s0
; CHECKFP-NEXT:    lea %s1, __ve_grow_stack@lo
; CHECKFP-NEXT:    and %s1, %s1, (32)0
; CHECKFP-NEXT:    lea.sl %s12, __ve_grow_stack@hi(, %s1)
; CHECKFP-NEXT:    bsic %s10, (, %s12)
; CHECKFP-NEXT:    lea %s0, 240(, %s11)
; CHECKFP-NEXT:    ld1b.zx %s1, (, %s2)
; CHECKFP-NEXT:    st1b %s1, (, %s0)
; CHECKFP-NEXT:    or %s11, 0, %s9
; CHECKFP-NEXT:    ld %s10, 8(, %s11)
; CHECKFP-NEXT:    ld %s9, (, %s11)
; CHECKFP-NEXT:    b.l.t (, %s10)
  %3 = alloca i8, i64 %1, align 16
  %4 = load i8, ptr %0, align 1
  store i8 %4, ptr %3, align 1
  ret ptr %3
}

;; This test allocates static buffer with 16 bytes align and dynamic buffer
;; with 32 bytes align.  In LLVM, stack frame is always aligned to 32 bytes
;; (bigger one).  So, LLVM allocates 176 (RSA) + 64 (call site) + 32 (32 bytes
;; aligned 16 bytes data) + 16 (pad to align) if FP is not eliminated.
;; Statically allocated buffer is aligned to 16, so it places from 16 to 31
;; bytes from BP in 32 + alpha bytes local vars area, or it places from 272
;; to 287 bytes from BP in 288 + alpha bytes local vars area.
;; Dynamically allocated buffer is aligned to 32, so it places from aligned
;; address between 240 and 271 from SP.

; Function Attrs: nounwind
define ptr @test_frame16_align16_dynalign32(ptr %0, i64 %n) {
; CHECK-LABEL: test_frame16_align16_dynalign32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s9, (, %s11)
; CHECK-NEXT:    st %s10, 8(, %s11)
; CHECK-NEXT:    st %s17, 40(, %s11)
; CHECK-NEXT:    or %s9, 0, %s11
; CHECK-NEXT:    lea %s11, -288(, %s11)
; CHECK-NEXT:    and %s11, %s11, (59)1
; CHECK-NEXT:    or %s17, 0, %s11
; CHECK-NEXT:    brge.l.t %s11, %s8, .LBB6_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    ld %s61, 24(, %s14)
; CHECK-NEXT:    or %s62, 0, %s0
; CHECK-NEXT:    lea %s63, 315
; CHECK-NEXT:    shm.l %s63, (%s61)
; CHECK-NEXT:    shm.l %s8, 8(%s61)
; CHECK-NEXT:    shm.l %s11, 16(%s61)
; CHECK-NEXT:    monc
; CHECK-NEXT:    or %s0, 0, %s62
; CHECK-NEXT:  .LBB6_2:
; CHECK-NEXT:    ld1b.zx %s0, (, %s0)
; CHECK-NEXT:    st1b %s0, 272(, %s17)
; CHECK-NEXT:    lea %s0, 15(, %s1)
; CHECK-NEXT:    and %s0, -16, %s0
; CHECK-NEXT:    lea %s1, __ve_grow_stack_align@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, __ve_grow_stack_align@hi(, %s1)
; CHECK-NEXT:    or %s1, -32, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    lea %s0, 240(, %s11)
; CHECK-NEXT:    ld1b.zx %s1, 272(, %s17)
; CHECK-NEXT:    lea %s0, 31(, %s0)
; CHECK-NEXT:    and %s0, -32, %s0
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
; CHECK-NEXT:    ld %s17, 40(, %s11)
; CHECK-NEXT:    ld %s10, 8(, %s11)
; CHECK-NEXT:    ld %s9, (, %s11)
; CHECK-NEXT:    b.l.t (, %s10)
;
; CHECKFP-LABEL: test_frame16_align16_dynalign32:
; CHECKFP:       # %bb.0:
; CHECKFP-NEXT:    st %s9, (, %s11)
; CHECKFP-NEXT:    st %s10, 8(, %s11)
; CHECKFP-NEXT:    st %s17, 40(, %s11)
; CHECKFP-NEXT:    or %s9, 0, %s11
; CHECKFP-NEXT:    lea %s11, -288(, %s11)
; CHECKFP-NEXT:    and %s11, %s11, (59)1
; CHECKFP-NEXT:    or %s17, 0, %s11
; CHECKFP-NEXT:    brge.l.t %s11, %s8, .LBB6_2
; CHECKFP-NEXT:  # %bb.1:
; CHECKFP-NEXT:    ld %s61, 24(, %s14)
; CHECKFP-NEXT:    or %s62, 0, %s0
; CHECKFP-NEXT:    lea %s63, 315
; CHECKFP-NEXT:    shm.l %s63, (%s61)
; CHECKFP-NEXT:    shm.l %s8, 8(%s61)
; CHECKFP-NEXT:    shm.l %s11, 16(%s61)
; CHECKFP-NEXT:    monc
; CHECKFP-NEXT:    or %s0, 0, %s62
; CHECKFP-NEXT:  .LBB6_2:
; CHECKFP-NEXT:    ld1b.zx %s0, (, %s0)
; CHECKFP-NEXT:    st1b %s0, 272(, %s17)
; CHECKFP-NEXT:    lea %s0, 15(, %s1)
; CHECKFP-NEXT:    and %s0, -16, %s0
; CHECKFP-NEXT:    lea %s1, __ve_grow_stack_align@lo
; CHECKFP-NEXT:    and %s1, %s1, (32)0
; CHECKFP-NEXT:    lea.sl %s12, __ve_grow_stack_align@hi(, %s1)
; CHECKFP-NEXT:    or %s1, -32, (0)1
; CHECKFP-NEXT:    bsic %s10, (, %s12)
; CHECKFP-NEXT:    lea %s0, 240(, %s11)
; CHECKFP-NEXT:    ld1b.zx %s1, 272(, %s17)
; CHECKFP-NEXT:    lea %s0, 31(, %s0)
; CHECKFP-NEXT:    and %s0, -32, %s0
; CHECKFP-NEXT:    st1b %s1, (, %s0)
; CHECKFP-NEXT:    or %s11, 0, %s9
; CHECKFP-NEXT:    ld %s17, 40(, %s11)
; CHECKFP-NEXT:    ld %s10, 8(, %s11)
; CHECKFP-NEXT:    ld %s9, (, %s11)
; CHECKFP-NEXT:    b.l.t (, %s10)
  %2 = alloca [16 x i8], align 16
  %3 = load i8, ptr %0, align 1
  store i8 %3, ptr %2, align 1
  %4 = alloca i8, i64 %n, align 32
  %5 = load i8, ptr %2, align 1
  store i8 %5, ptr %4, align 1
  ret ptr %4
}

