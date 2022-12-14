; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple armv6t2 %s -o - | FileCheck %s
; RUN: llc -mtriple thumbv6t2 %s -o - | FileCheck %s --check-prefix=CHECK-T2
; RUN: llc -mtriple armv7 %s -o - | FileCheck %s
; RUN: llc -mtriple thumbv7 %s -o - | FileCheck %s --check-prefix=CHECK-T2
; RUN: llc -mtriple thumbv7m %s -o - | FileCheck %s --check-prefix=CHECK-T2
; RUN: llc -mtriple thumbv8m.main %s -o - | FileCheck %s --check-prefix=CHECK-T2

@a = common dso_local local_unnamed_addr global i16 0, align 2

define dso_local arm_aapcscc ptr @pr36577() {
; CHECK-LABEL: pr36577:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    movw r0, :lower16:a
; CHECK-NEXT:    mvn r1, #7
; CHECK-NEXT:    movt r0, :upper16:a
; CHECK-NEXT:    ldrh r0, [r0]
; CHECK-NEXT:    mvn r0, r0, lsr #7
; CHECK-NEXT:    orr r0, r1, r0, lsl #2
; CHECK-NEXT:    bx lr
;
; CHECK-T2-LABEL: pr36577:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    movw r0, :lower16:a
; CHECK-T2-NEXT:    mvn r1, #7
; CHECK-T2-NEXT:    movt r0, :upper16:a
; CHECK-T2-NEXT:    ldrh r0, [r0]
; CHECK-T2-NEXT:    mvn.w r0, r0, lsr #7
; CHECK-T2-NEXT:    orr.w r0, r1, r0, lsl #2
; CHECK-T2-NEXT:    bx lr
entry:
  %0 = load i16, ptr @a, align 2
  %1 = lshr i16 %0, 7
  %2 = and i16 %1, 1
  %3 = zext i16 %2 to i32
  %4 = xor i32 %3, -1
  %add.ptr = getelementptr inbounds ptr, ptr null, i32 %4
  ret ptr %add.ptr
}

