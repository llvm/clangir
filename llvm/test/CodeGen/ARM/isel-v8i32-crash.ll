; RUN: llc < %s -mtriple=armv7-linux-gnu | FileCheck %s

; Check we don't crash when trying to combine:
;   (d1 = <float 8.000000e+00, float 8.000000e+00, ...>) (power of 2)
;   vmul.f32        d0, d1, d0
;   vcvt.s32.f32    d0, d0
; into:
;   vcvt.s32.f32    d0, d0, #3
; when we have a vector length of 8, due to use of v8i32 types.

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; CHECK: func:
; CHECK: vcvt.s32.f32  q[[R:[0-9]]], q[[R]], #3
define void @func(ptr nocapture %pb, ptr nocapture readonly %pf) #0 {
entry:
  %0 = load <8 x float>, ptr %pf, align 4
  %1 = fmul <8 x float> %0, <float 8.000000e+00, float 8.000000e+00, float 8.000000e+00, float 8.000000e+00, float 8.000000e+00, float 8.000000e+00, float 8.000000e+00, float 8.000000e+00>
  %2 = fptosi <8 x float> %1 to <8 x i16>
  store <8 x i16> %2, ptr %pb, align 2
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
