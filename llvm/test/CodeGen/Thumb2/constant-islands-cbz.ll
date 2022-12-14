; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=thumbv8m.base-none-eabi %s -o - | FileCheck %s --check-prefix=CHECK-T1
; RUN: llc < %s -mtriple=thumbv8m.main-none-eabi %s -o - | FileCheck %s --check-prefix=CHECK-T2

define ptr @test(ptr returned %this, i32 %event_size, ptr %event_pointer) {
; CHECK-T1-LABEL: test:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    .save {r4, lr}
; CHECK-T1-NEXT:    push {r4, lr}
; CHECK-T1-NEXT:    mov r4, r0
; CHECK-T1-NEXT:    movs r0, #0
; CHECK-T1-NEXT:    str r0, [r4, #4]
; CHECK-T1-NEXT:    str r0, [r4, #8]
; CHECK-T1-NEXT:    str r0, [r4, #12]
; CHECK-T1-NEXT:    str r0, [r4, #16]
; CHECK-T1-NEXT:    mov r0, r4
; CHECK-T1-NEXT:    cbz r2, .LBB0_2
; CHECK-T1-NEXT:  @ %bb.1: @ %if.else
; CHECK-T1-NEXT:    bl equeue_create_inplace
; CHECK-T1-NEXT:    mov r0, r4
; CHECK-T1-NEXT:    pop {r4, pc}
; CHECK-T1-NEXT:  .LBB0_2: @ %if.then
; CHECK-T1-NEXT:    bl equeue_create
; CHECK-T1-NEXT:    mov r0, r4
; CHECK-T1-NEXT:    pop {r4, pc}
;
; CHECK-T2-LABEL: test:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    .save {r4, lr}
; CHECK-T2-NEXT:    push {r4, lr}
; CHECK-T2-NEXT:    mov r4, r0
; CHECK-T2-NEXT:    movs r0, #0
; CHECK-T2-NEXT:    strd r0, r0, [r4, #4]
; CHECK-T2-NEXT:    strd r0, r0, [r4, #12]
; CHECK-T2-NEXT:    mov r0, r4
; CHECK-T2-NEXT:    cbz r2, .LBB0_2
; CHECK-T2-NEXT:  @ %bb.1: @ %if.else
; CHECK-T2-NEXT:    bl equeue_create_inplace
; CHECK-T2-NEXT:    mov r0, r4
; CHECK-T2-NEXT:    pop {r4, pc}
; CHECK-T2-NEXT:  .LBB0_2: @ %if.then
; CHECK-T2-NEXT:    bl equeue_create
; CHECK-T2-NEXT:    mov r0, r4
; CHECK-T2-NEXT:    pop {r4, pc}
entry:
  %_update = getelementptr inbounds i32, ptr %this, i32 1
  tail call void @llvm.memset.p0.i32(ptr nonnull align 4 %_update, i8 0, i32 16, i1 false) #4
  %tobool = icmp eq ptr %event_pointer, null
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call4 = tail call i32 @equeue_create(ptr %this, i32 %event_size) #5
  br label %if.end

if.else:                                          ; preds = %entry
  %call6 = tail call i32 @equeue_create_inplace(ptr %this, i32 %event_size, ptr nonnull %event_pointer) #5
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret ptr %this
}

declare dso_local i32 @equeue_create(ptr, i32) local_unnamed_addr #1
declare dso_local i32 @equeue_create_inplace(ptr, i32, ptr) local_unnamed_addr #1
declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1 immarg) #2
