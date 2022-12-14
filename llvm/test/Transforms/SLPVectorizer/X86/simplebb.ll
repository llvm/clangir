; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -passes=slp-vectorizer,dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; Simple 3-pair chain with loads and stores
define void @test1(ptr %a, ptr %b, ptr %c) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:    [[TMP2:%.*]] = load <2 x double>, ptr [[A:%.*]], align 8
; CHECK-NEXT:    [[TMP4:%.*]] = load <2 x double>, ptr [[B:%.*]], align 8
; CHECK-NEXT:    [[TMP5:%.*]] = fmul <2 x double> [[TMP2]], [[TMP4]]
; CHECK-NEXT:    store <2 x double> [[TMP5]], ptr [[C:%.*]], align 8
; CHECK-NEXT:    ret void
;
  %i0 = load double, ptr %a, align 8
  %i1 = load double, ptr %b, align 8
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds double, ptr %a, i64 1
  %i3 = load double, ptr %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, ptr %b, i64 1
  %i4 = load double, ptr %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  store double %mul, ptr %c, align 8
  %arrayidx5 = getelementptr inbounds double, ptr %c, i64 1
  store double %mul5, ptr %arrayidx5, align 8
  ret void
}

; Simple 3-pair chain with loads and stores, obfuscated with bitcasts
define void @test2(ptr %a, ptr %b, ptr %e) {
; CHECK-LABEL: @test2(
; CHECK-NEXT:    [[TMP2:%.*]] = load <2 x double>, ptr [[A:%.*]], align 8
; CHECK-NEXT:    [[TMP4:%.*]] = load <2 x double>, ptr [[B:%.*]], align 8
; CHECK-NEXT:    [[TMP5:%.*]] = fmul <2 x double> [[TMP2]], [[TMP4]]
; CHECK-NEXT:    store <2 x double> [[TMP5]], ptr [[E:%.*]], align 8
; CHECK-NEXT:    ret void
;
  %i0 = load double, ptr %a, align 8
  %i1 = load double, ptr %b, align 8
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds double, ptr %a, i64 1
  %i3 = load double, ptr %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, ptr %b, i64 1
  %i4 = load double, ptr %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  store double %mul, ptr %e, align 8
  %carrayidx5 = getelementptr inbounds i8, ptr %e, i64 8
  store double %mul5, ptr %carrayidx5, align 8
  ret void
}

; Don't vectorize volatile loads.
define void @test_volatile_load(ptr %a, ptr %b, ptr %c) {
; CHECK-LABEL: @test_volatile_load(
; CHECK-NEXT:    [[I0:%.*]] = load volatile double, ptr [[A:%.*]], align 8
; CHECK-NEXT:    [[I1:%.*]] = load volatile double, ptr [[B:%.*]], align 8
; CHECK-NEXT:    [[ARRAYIDX3:%.*]] = getelementptr inbounds double, ptr [[A]], i64 1
; CHECK-NEXT:    [[I3:%.*]] = load double, ptr [[ARRAYIDX3]], align 8
; CHECK-NEXT:    [[ARRAYIDX4:%.*]] = getelementptr inbounds double, ptr [[B]], i64 1
; CHECK-NEXT:    [[I4:%.*]] = load double, ptr [[ARRAYIDX4]], align 8
; CHECK-NEXT:    [[TMP1:%.*]] = insertelement <2 x double> poison, double [[I0]], i32 0
; CHECK-NEXT:    [[TMP2:%.*]] = insertelement <2 x double> [[TMP1]], double [[I3]], i32 1
; CHECK-NEXT:    [[TMP3:%.*]] = insertelement <2 x double> poison, double [[I1]], i32 0
; CHECK-NEXT:    [[TMP4:%.*]] = insertelement <2 x double> [[TMP3]], double [[I4]], i32 1
; CHECK-NEXT:    [[TMP5:%.*]] = fmul <2 x double> [[TMP2]], [[TMP4]]
; CHECK-NEXT:    store <2 x double> [[TMP5]], ptr [[C:%.*]], align 8
; CHECK-NEXT:    ret void
;
  %i0 = load volatile double, ptr %a, align 8
  %i1 = load volatile double, ptr %b, align 8
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds double, ptr %a, i64 1
  %i3 = load double, ptr %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, ptr %b, i64 1
  %i4 = load double, ptr %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  store double %mul, ptr %c, align 8
  %arrayidx5 = getelementptr inbounds double, ptr %c, i64 1
  store double %mul5, ptr %arrayidx5, align 8
  ret void
}

; Don't vectorize volatile stores.
define void @test_volatile_store(ptr %a, ptr %b, ptr %c) {
; CHECK-LABEL: @test_volatile_store(
; CHECK-NEXT:    [[I0:%.*]] = load double, ptr [[A:%.*]], align 8
; CHECK-NEXT:    [[I1:%.*]] = load double, ptr [[B:%.*]], align 8
; CHECK-NEXT:    [[MUL:%.*]] = fmul double [[I0]], [[I1]]
; CHECK-NEXT:    [[ARRAYIDX3:%.*]] = getelementptr inbounds double, ptr [[A]], i64 1
; CHECK-NEXT:    [[I3:%.*]] = load double, ptr [[ARRAYIDX3]], align 8
; CHECK-NEXT:    [[ARRAYIDX4:%.*]] = getelementptr inbounds double, ptr [[B]], i64 1
; CHECK-NEXT:    [[I4:%.*]] = load double, ptr [[ARRAYIDX4]], align 8
; CHECK-NEXT:    [[MUL5:%.*]] = fmul double [[I3]], [[I4]]
; CHECK-NEXT:    store volatile double [[MUL]], ptr [[C:%.*]], align 8
; CHECK-NEXT:    [[ARRAYIDX5:%.*]] = getelementptr inbounds double, ptr [[C]], i64 1
; CHECK-NEXT:    store volatile double [[MUL5]], ptr [[ARRAYIDX5]], align 8
; CHECK-NEXT:    ret void
;
  %i0 = load double, ptr %a, align 8
  %i1 = load double, ptr %b, align 8
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds double, ptr %a, i64 1
  %i3 = load double, ptr %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, ptr %b, i64 1
  %i4 = load double, ptr %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  store volatile double %mul, ptr %c, align 8
  %arrayidx5 = getelementptr inbounds double, ptr %c, i64 1
  store volatile double %mul5, ptr %arrayidx5, align 8
  ret void
}

