; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>' -verify-memoryssa -disable-output < %s 2>&1 | FileCheck %s
;
; Invariant loads should be considered live on entry, because, once the
; location is known to be dereferenceable, the value can never change.

@g = external global i32

declare void @clobberAllTheThings()

; CHECK-LABEL: define i32 @foo
define i32 @foo() {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: call void @clobberAllTheThings()
  call void @clobberAllTheThings()
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %1 = load i32
  %1 = load i32, ptr @g, align 4, !invariant.load !0
  ret i32 %1
}

; CHECK-LABEL: define i32 @bar
define i32 @bar(ptr %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: call void @clobberAllTheThings()
  call void @clobberAllTheThings()

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: %1 = load atomic i32
  %1 = load atomic i32, ptr %a acquire, align 4, !invariant.load !0

; CHECK: MemoryUse(2)
; CHECK-NEXT: %2 = load i32
  %2 = load i32, ptr %a, align 4
  ret i32 %2
}

!0 = !{}
