; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -S -passes=jump-threading < %s | FileCheck %s

declare i8 @mcguffin()

; Check there's no phi here.
define i32 @test(i1 %foo, i8 %b) {
; CHECK-LABEL: @test(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.*]] = call i8 @mcguffin()
; CHECK-NEXT:    br i1 [[FOO:%.*]], label [[RT:%.*]], label [[JT:%.*]]
; CHECK:       jt:
; CHECK-NEXT:    [[CMP_A:%.*]] = icmp eq i8 [[B:%.*]], [[A]]
; CHECK-NEXT:    br i1 [[CMP_A]], label [[RT]], label [[RF:%.*]]
; CHECK:       rt:
; CHECK-NEXT:    ret i32 7
; CHECK:       rf:
; CHECK-NEXT:    ret i32 8
;
entry:
  %a = call i8 @mcguffin()
  br i1 %foo, label %bb1, label %bb2
bb1:
  br label %jt
bb2:
  br label %jt
jt:
  %x = phi i8 [%a, %bb1], [%b, %bb2]
  %cmp.a = icmp eq i8 %x, %a
  br i1 %cmp.a, label %rt, label %rf
rt:
  ret i32 7
rf:
  ret i32 8
}
