; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -passes=inline -S < %s | FileCheck %s
; RUN: opt -passes='cgscc(inline)' -S < %s | FileCheck %s
; RUN: opt -passes='module-inline' -S < %s | FileCheck %s

define dso_local i32 @main() {
; CHECK-LABEL: @main(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    [[I_I:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[I1_I:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[I:%.*]] = alloca i32, align 4
; CHECK-NEXT:    store i32 0, ptr [[I]], align 4
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 4, ptr [[I_I]])
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 4, ptr [[I1_I]])
; CHECK-NEXT:    store i32 0, ptr [[I1_I]], align 4
; CHECK-NEXT:    [[I2_I:%.*]] = load i32, ptr [[I1_I]], align 4
; CHECK-NEXT:    callbr void asm sideeffect "", "r,!i,!i,~{dirflag},~{fpsr},~{flags}"(i32 [[I2_I]])
; CHECK-NEXT:    to label [[BB3_I:%.*]] [label [[BB5_I:%.*]], label %bb4.i]
; CHECK:       bb3.i:
; CHECK-NEXT:    store i32 0, ptr [[I_I]], align 4
; CHECK-NEXT:    br label [[T32_EXIT:%.*]]
; CHECK:       bb4.i:
; CHECK-NEXT:    store i32 1, ptr [[I_I]], align 4
; CHECK-NEXT:    br label [[T32_EXIT]]
; CHECK:       bb5.i:
; CHECK-NEXT:    store i32 2, ptr [[I_I]], align 4
; CHECK-NEXT:    br label [[T32_EXIT]]
; CHECK:       t32.exit:
; CHECK-NEXT:    [[I7_I:%.*]] = load i32, ptr [[I_I]], align 4
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr [[I_I]])
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr [[I1_I]])
; CHECK-NEXT:    ret i32 [[I7_I]]
;
bb:
  %i = alloca i32, align 4
  store i32 0, ptr %i, align 4
  %i1 = call i32 @t32(i32 0)
  ret i32 %i1
}

define internal i32 @t32(i32 %arg) {
bb:
  %i = alloca i32, align 4
  %i1 = alloca i32, align 4
  store i32 %arg, ptr %i1, align 4
  %i2 = load i32, ptr %i1, align 4
  callbr void asm sideeffect "", "r,!i,!i,~{dirflag},~{fpsr},~{flags}"(i32 %i2)
  to label %bb3 [label %bb5, label %bb4]

bb3:                                              ; preds = %bb
  store i32 0, ptr %i, align 4
  br label %bb6

bb4:                                              ; preds = %bb
  store i32 1, ptr %i, align 4
  br label %bb6

bb5:                                              ; preds = %bb
  store i32 2, ptr %i, align 4
  br label %bb6

bb6:                                              ; preds = %bb5, %bb4, %bb3
  %i7 = load i32, ptr %i, align 4
  ret i32 %i7
}
