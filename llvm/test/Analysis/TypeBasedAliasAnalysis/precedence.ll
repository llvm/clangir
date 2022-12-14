; RUN: opt -aa-pipeline=tbaa -passes=gvn,instcombine -S < %s | FileCheck %s --check-prefix=TBAA
; RUN: opt -aa-pipeline=basic-aa,tbaa -passes=gvn,instcombine -S < %s | FileCheck %s --check-prefix=BASICAA

; According to the TBAA metadata the load and store don't alias. However,
; according to the actual code, they do. Disabling basicaa shows the raw TBAA
; results.

target datalayout = "e-p:64:64:64"

; Test for simple MustAlias aliasing.

; TBAA:    @trouble
; TBAA:      ret i32 0
; BASICAA: @trouble
; BASICAA:   ret i32 1075000115
define i32 @trouble(ptr %x) nounwind {
entry:
  store i32 0, ptr %x, !tbaa !0
  store float 0x4002666660000000, ptr %x, !tbaa !3
  %tmp3 = load i32, ptr %x, !tbaa !0
  ret i32 %tmp3
}

; Test for PartialAlias aliasing. GVN doesn't yet eliminate the load
; in the BasicAA case.

; TBAA:    @offset
; TBAA:      ret i64 0
; BASICAA: @offset
; BASICAA:   ret i64 %tmp3
define i64 @offset(ptr %x) nounwind {
entry:
  store i64 0, ptr %x, !tbaa !4
  %0 = getelementptr i8, ptr %x, i64 1
  store i8 1, ptr %0, !tbaa !5
  %tmp3 = load i64, ptr %x, !tbaa !4
  ret i64 %tmp3
}

!0 = !{!2, !2, i64 0}
!1 = !{!"simple"}
!2 = !{!"int", !1}
!3 = !{!6, !6, i64 0}
!4 = !{!7, !7, i64 0}
!5 = !{!8, !8, i64 0}
!6 = !{!"float", !1}
!7 = !{!"long", !1}
!8 = !{!"small", !1}
