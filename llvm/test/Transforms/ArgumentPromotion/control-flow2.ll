; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --function-signature --scrub-attributes
; RUN: opt < %s -passes=argpromotion -S | FileCheck %s

target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

define internal i32 @callee(i1 %C, ptr %P) {
; CHECK-LABEL: define {{[^@]+}}@callee
; CHECK-SAME: (i1 [[C:%.*]], i32 [[P_0_VAL:%.*]]) {
; CHECK-NEXT:    br i1 [[C]], label [[T:%.*]], label [[F:%.*]]
; CHECK:       T:
; CHECK-NEXT:    ret i32 17
; CHECK:       F:
; CHECK-NEXT:    ret i32 [[P_0_VAL]]
;
  br i1 %C, label %T, label %F

T:              ; preds = %0
  ret i32 17

F:              ; preds = %0
  %X = load i32, ptr %P               ; <i32> [#uses=1]
  ret i32 %X
}

define i32 @foo() {
; CHECK-LABEL: define {{[^@]+}}@foo() {
; CHECK-NEXT:    [[A:%.*]] = alloca i32, align 4
; CHECK-NEXT:    store i32 17, ptr [[A]], align 4
; CHECK-NEXT:    [[A_VAL:%.*]] = load i32, ptr [[A]], align 4
; CHECK-NEXT:    [[X:%.*]] = call i32 @callee(i1 false, i32 [[A_VAL]])
; CHECK-NEXT:    ret i32 [[X]]
;
  %A = alloca i32         ; <ptr> [#uses=2]
  store i32 17, ptr %A
  %X = call i32 @callee( i1 false, ptr %A )              ; <i32> [#uses=1]
  ret i32 %X
}

