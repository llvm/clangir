; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -passes=simplifycfg,instcombine -simplifycfg-require-and-preserve-domtree=1 -S < %s | FileCheck %s

define zeroext i1 @switch_ob_one_two_cases(i32 %arg) {
; CHECK-LABEL: @switch_ob_one_two_cases(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    [[TMP0:%.*]] = and i32 [[ARG:%.*]], -3
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq i32 [[TMP0]], 0
; CHECK-NEXT:    ret i1 [[TMP1]]
;
bb:
  switch i32 %arg, label %bb1 [
  i32 0, label %bb2
  i32 2, label %bb2
  ]

bb1:
  br label %bb2

bb2:
  %i = phi i1 [ false, %bb1 ], [ true, %bb ], [ true, %bb ]
  ret i1 %i
}

define zeroext i1 @switch_ob_one_two_cases2(i32 %arg) {
; CHECK-LABEL: @switch_ob_one_two_cases2(
; CHECK-NEXT:    [[I:%.*]] = icmp eq i32 [[ARG:%.*]], 7
; CHECK-NEXT:    [[I1:%.*]] = icmp eq i32 [[ARG]], 11
; CHECK-NEXT:    [[I2:%.*]] = or i1 [[I]], [[I1]]
; CHECK-NEXT:    ret i1 [[I2]]
;
  %i = icmp eq i32 %arg, 7
  %i1 = icmp eq i32 %arg, 11
  %i2 = or i1 %i, %i1
  ret i1 %i2
}
