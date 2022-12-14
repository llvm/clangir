; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -S -passes=instcombine < %s | FileCheck %s

define float @test1(float %x) nounwind readnone ssp {
; CHECK-LABEL: @test1(
; CHECK-NEXT:    [[SQRTF:%.*]] = call float @sqrtf(float [[X:%.*]]) #[[ATTR4:[0-9]+]]
; CHECK-NEXT:    ret float [[SQRTF]]
;
  %conv = fpext float %x to double
  %call = tail call double @sqrt(double %conv) readnone nounwind
  %conv1 = fptrunc double %call to float
  ret float %conv1
}

; PR8096

define float @test2(float %x) nounwind readnone ssp {
; CHECK-LABEL: @test2(
; CHECK-NEXT:    [[SQRTF:%.*]] = call float @sqrtf(float [[X:%.*]]) #[[ATTR4]]
; CHECK-NEXT:    ret float [[SQRTF]]
;
  %conv = fpext float %x to double
  %call = tail call double @sqrt(double %conv) nounwind
  %conv1 = fptrunc double %call to float
  ret float %conv1
}

; rdar://9763193
; Can't fold (fptrunc (sqrt (fpext x))) -> (sqrtf x) since there is another
; use of sqrt result.

define float @test3(ptr %v) nounwind uwtable ssp {
; CHECK-LABEL: @test3(
; CHECK-NEXT:    [[CALL34:%.*]] = call double @sqrt(double 0x7FF8000000000000) #[[ATTR4]]
; CHECK-NEXT:    [[CALL36:%.*]] = call i32 @foo(double [[CALL34]]) #[[ATTR5:[0-9]+]]
; CHECK-NEXT:    [[CONV38:%.*]] = fptrunc double [[CALL34]] to float
; CHECK-NEXT:    ret float [[CONV38]]
;
  %arrayidx13 = getelementptr inbounds float, ptr %v, i64 2
  %tmp14 = load float, ptr %arrayidx13
  %mul18 = fmul float %tmp14, %tmp14
  %add19 = fadd float undef, %mul18
  %conv = fpext float %add19 to double
  %call34 = call double @sqrt(double %conv) readnone
  %call36 = call i32 (double) @foo(double %call34) nounwind
  %conv38 = fptrunc double %call34 to float
  ret float %conv38
}

; PR43347 - https://bugs.llvm.org/show_bug.cgi?id=43347

define void @0(float %f) {
; CHECK-LABEL: @0(
; CHECK-NEXT:    [[SQRTF:%.*]] = call float @sqrtf(float [[F:%.*]]) #[[ATTR2:[0-9]+]]
; CHECK-NEXT:    ret void
;
  %d = fpext float %f to double
  %r = call double @sqrt(double %d)
  ret void
}

define float @sqrt_call_nnan_f32(float %x) {
; CHECK-LABEL: @sqrt_call_nnan_f32(
; CHECK-NEXT:    [[SQRT:%.*]] = call nnan float @sqrtf(float [[X:%.*]])
; CHECK-NEXT:    ret float [[SQRT]]
;
  %sqrt = call nnan float @sqrtf(float %x)
  ret float %sqrt
}

define double @sqrt_call_nnan_f64(double %x) {
; CHECK-LABEL: @sqrt_call_nnan_f64(
; CHECK-NEXT:    [[SQRT:%.*]] = tail call nnan ninf double @sqrt(double [[X:%.*]])
; CHECK-NEXT:    ret double [[SQRT]]
;
  %sqrt = tail call nnan ninf double @sqrt(double %x)
  ret double %sqrt
}

define float @sqrt_call_fabs_f32(float %x) {
; CHECK-LABEL: @sqrt_call_fabs_f32(
; CHECK-NEXT:    [[A:%.*]] = call float @llvm.fabs.f32(float [[X:%.*]])
; CHECK-NEXT:    [[SQRT:%.*]] = tail call float @sqrtf(float [[A]])
; CHECK-NEXT:    ret float [[SQRT]]
;
  %a = call float @llvm.fabs.f32(float %x)
  %sqrt = tail call float @sqrtf(float %a)
  ret float %sqrt
}

declare i32 @foo(double)
declare double @sqrt(double) readnone
declare float @sqrtf(float)
declare float @llvm.fabs.f32(float)
