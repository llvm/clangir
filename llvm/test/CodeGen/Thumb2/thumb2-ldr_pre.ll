; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

define ptr @test1(ptr %X, ptr %dest) {
; CHECK-LABEL: test1:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    ldr r2, [r0, #16]!
; CHECK-NEXT:    str r2, [r1]
; CHECK-NEXT:    bx lr
        %Y = getelementptr i32, ptr %X, i32 4               ; <ptr> [#uses=2]
        %A = load i32, ptr %Y               ; <i32> [#uses=1]
        store i32 %A, ptr %dest
        ret ptr %Y
}


define i32 @test2(i32 %a, i32 %b) {
; CHECK-LABEL: test2:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    ldr r2, [r0, #-64]!
; CHECK-NEXT:    subs r0, r0, r1
; CHECK-NEXT:    add r0, r2
; CHECK-NEXT:    bx lr
        %tmp1 = sub i32 %a, 64          ; <i32> [#uses=2]
        %tmp2 = inttoptr i32 %tmp1 to ptr              ; <ptr> [#uses=1]
        %tmp3 = load i32, ptr %tmp2         ; <i32> [#uses=1]
        %tmp4 = sub i32 %tmp1, %b               ; <i32> [#uses=1]
        %tmp5 = add i32 %tmp4, %tmp3            ; <i32> [#uses=1]
        ret i32 %tmp5
}


define ptr @test3(ptr %X, ptr %dest) {
; CHECK-LABEL: test3:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    ldrsb r2, [r0, #4]!
; CHECK-NEXT:    str r2, [r1]
; CHECK-NEXT:    bx lr
        %tmp1 = getelementptr i8, ptr %X, i32 4
        %tmp2 = load i8, ptr %tmp1
        %tmp3 = sext i8 %tmp2 to i32
        store i32 %tmp3, ptr %dest
        ret ptr %tmp1
}
