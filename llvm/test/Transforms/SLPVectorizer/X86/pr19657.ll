; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -passes=slp-vectorizer -S -mcpu=corei7-avx | FileCheck %s --check-prefixes=ANY,AVX
; RUN: opt < %s -passes=slp-vectorizer -slp-max-reg-size=128 -S -mcpu=corei7-avx | FileCheck %s --check-prefixes=ANY,MAX128

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @store_chains(ptr %x) {
; AVX-LABEL: @store_chains(
; AVX-NEXT:    [[TMP2:%.*]] = load <4 x double>, ptr [[X:%.*]], align 8
; AVX-NEXT:    [[TMP3:%.*]] = fadd <4 x double> [[TMP2]], [[TMP2]]
; AVX-NEXT:    [[TMP4:%.*]] = fadd <4 x double> [[TMP3]], [[TMP2]]
; AVX-NEXT:    store <4 x double> [[TMP4]], ptr [[X]], align 8
; AVX-NEXT:    ret void
;
; MAX128-LABEL: @store_chains(
; MAX128-NEXT:    [[TMP2:%.*]] = load <2 x double>, ptr [[X:%.*]], align 8
; MAX128-NEXT:    [[TMP3:%.*]] = fadd <2 x double> [[TMP2]], [[TMP2]]
; MAX128-NEXT:    [[TMP4:%.*]] = fadd <2 x double> [[TMP3]], [[TMP2]]
; MAX128-NEXT:    store <2 x double> [[TMP4]], ptr [[X]], align 8
; MAX128-NEXT:    [[TMP6:%.*]] = getelementptr inbounds double, ptr [[X]], i64 2
; MAX128-NEXT:    [[TMP8:%.*]] = load <2 x double>, ptr [[TMP6]], align 8
; MAX128-NEXT:    [[TMP9:%.*]] = fadd <2 x double> [[TMP8]], [[TMP8]]
; MAX128-NEXT:    [[TMP10:%.*]] = fadd <2 x double> [[TMP9]], [[TMP8]]
; MAX128-NEXT:    store <2 x double> [[TMP10]], ptr [[TMP6]], align 8
; MAX128-NEXT:    ret void
;
  %1 = load double, ptr %x, align 8
  %2 = fadd double %1, %1
  %3 = fadd double %2, %1
  store double %3, ptr %x, align 8
  %4 = getelementptr inbounds double, ptr %x, i64 1
  %5 = load double, ptr %4, align 8
  %6 = fadd double %5, %5
  %7 = fadd double %6, %5
  store double %7, ptr %4, align 8
  %8 = getelementptr inbounds double, ptr %x, i64 2
  %9 = load double, ptr %8, align 8
  %10 = fadd double %9, %9
  %11 = fadd double %10, %9
  store double %11, ptr %8, align 8
  %12 = getelementptr inbounds double, ptr %x, i64 3
  %13 = load double, ptr %12, align 8
  %14 = fadd double %13, %13
  %15 = fadd double %14, %13
  store double %15, ptr %12, align 8
  ret void
}

define void @store_chains_prefer_width_attr(ptr %x) #0 {
; ANY-LABEL: @store_chains_prefer_width_attr(
; ANY-NEXT:    [[TMP2:%.*]] = load <2 x double>, ptr [[X:%.*]], align 8
; ANY-NEXT:    [[TMP3:%.*]] = fadd <2 x double> [[TMP2]], [[TMP2]]
; ANY-NEXT:    [[TMP4:%.*]] = fadd <2 x double> [[TMP3]], [[TMP2]]
; ANY-NEXT:    store <2 x double> [[TMP4]], ptr [[X]], align 8
; ANY-NEXT:    [[TMP6:%.*]] = getelementptr inbounds double, ptr [[X]], i64 2
; ANY-NEXT:    [[TMP8:%.*]] = load <2 x double>, ptr [[TMP6]], align 8
; ANY-NEXT:    [[TMP9:%.*]] = fadd <2 x double> [[TMP8]], [[TMP8]]
; ANY-NEXT:    [[TMP10:%.*]] = fadd <2 x double> [[TMP9]], [[TMP8]]
; ANY-NEXT:    store <2 x double> [[TMP10]], ptr [[TMP6]], align 8
; ANY-NEXT:    ret void
;
  %1 = load double, ptr %x, align 8
  %2 = fadd double %1, %1
  %3 = fadd double %2, %1
  store double %3, ptr %x, align 8
  %4 = getelementptr inbounds double, ptr %x, i64 1
  %5 = load double, ptr %4, align 8
  %6 = fadd double %5, %5
  %7 = fadd double %6, %5
  store double %7, ptr %4, align 8
  %8 = getelementptr inbounds double, ptr %x, i64 2
  %9 = load double, ptr %8, align 8
  %10 = fadd double %9, %9
  %11 = fadd double %10, %9
  store double %11, ptr %8, align 8
  %12 = getelementptr inbounds double, ptr %x, i64 3
  %13 = load double, ptr %12, align 8
  %14 = fadd double %13, %13
  %15 = fadd double %14, %13
  store double %15, ptr %12, align 8
  ret void
}

attributes #0 = { "prefer-vector-width"="128" }
