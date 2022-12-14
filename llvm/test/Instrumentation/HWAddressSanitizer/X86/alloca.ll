; Test alloca instrumentation.
;
; RUN: opt < %s -passes=hwasan -S | FileCheck %s --check-prefixes=CHECK,NO-UAR-TAGS
; RUN: opt < %s -passes=hwasan -hwasan-uar-retag-to-zero=0 -S | FileCheck %s --check-prefixes=CHECK,UAR-TAGS

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @use32(ptr)

define void @test_alloca() sanitize_hwaddress {
; CHECK-LABEL: @test_alloca(
; CHECK: %[[FP:[^ ]*]] = call ptr @llvm.frameaddress.p0(i32 0)
; CHECK: %[[A:[^ ]*]] = ptrtoint ptr %[[FP]] to i64
; CHECK: %[[B:[^ ]*]] = lshr i64 %[[A]], 20
; CHECK: %[[A_XOR_B:[^ ]*]] = xor i64 %[[A]], %[[B]]
; CHECK: %[[BASE_TAG:[^ ]*]] = and i64 %[[A_XOR_B]], 63

; CHECK: %[[X:[^ ]*]] = alloca { i32, [12 x i8] }, align 16
; CHECK: %[[X_TAG:[^ ]*]] = xor i64 %[[BASE_TAG]], 0
; CHECK: %[[X1:[^ ]*]] = ptrtoint ptr %[[X]] to i64
; CHECK: %[[C:[^ ]*]] = shl i64 %[[X_TAG]], 57
; CHECK: %[[D:[^ ]*]] = or i64 %[[X1]], %[[C]]
; CHECK: %[[X_HWASAN:[^ ]*]] = inttoptr i64 %[[D]] to ptr

; CHECK: %[[X_TAG2:[^ ]*]] = trunc i64 %[[X_TAG]] to i8
; CHECK: call void @__hwasan_tag_memory(ptr %[[X]], i8 %[[X_TAG2]], i64 16)

; CHECK: call void @use32(ptr nonnull %[[X_HWASAN]])

; UAR-TAGS: %[[BASE_TAG_COMPL:[^ ]*]] = xor i64 %[[BASE_TAG]], 63
; UAR-TAGS: %[[X_TAG_UAR:[^ ]*]] = trunc i64 %[[BASE_TAG_COMPL]] to i8
; NO-UAR-TAGS: call void @__hwasan_tag_memory(ptr %[[X]], i8 0, i64 16)
; UAR-TAGS: call void @__hwasan_tag_memory(ptr %[[X]], i8 %[[X_TAG_UAR]], i64 16)
; CHECK: ret void


entry:
  %x = alloca i32, align 4
  call void @use32(ptr nonnull %x)
  ret void
}
