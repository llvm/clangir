; RUN: llc --mtriple=wasm32-unknown-unknown -asm-verbose=false < %s | FileCheck %s

@i32_global = local_unnamed_addr addrspace(1) global i32 undef
@i64_global = local_unnamed_addr addrspace(1) global i64 undef
@f32_global = local_unnamed_addr addrspace(1) global float undef
@f64_global = local_unnamed_addr addrspace(1) global double undef

define void @set_i32_global(i32 %v) {
; CHECK-LABEL: set_i32_global:
; CHECK-NEXT: functype       set_i32_global (i32) -> ()
; CHECK-NEXT: local.get 0
; CHECK-NEXT: global.set i32_global
; CHECK-NEXT: end_function
  store i32 %v, ptr addrspace(1) @i32_global
  ret void
}

define void @set_i64_global(i64 %v) {
; CHECK-LABEL: set_i64_global:
; CHECK-NEXT: functype       set_i64_global (i64) -> ()
; CHECK-NEXT: local.get 0
; CHECK-NEXT: global.set i64_global
; CHECK-NEXT: end_function
  store i64 %v, ptr addrspace(1) @i64_global
  ret void
}

define void @set_f32_global(float %v) {
; CHECK-LABEL: set_f32_global:
; CHECK-NEXT: functype       set_f32_global (f32) -> ()
; CHECK-NEXT: local.get 0
; CHECK-NEXT: global.set f32_global
; CHECK-NEXT: end_function
  store float %v, ptr addrspace(1) @f32_global
  ret void
}

define void @set_f64_global(double %v) {
; CHECK-LABEL: set_f64_global:
; CHECK-NEXT: functype       set_f64_global (f64) -> ()
; CHECK-NEXT: local.get 0
; CHECK-NEXT: global.set f64_global
; CHECK-NEXT: end_function
  store double %v, ptr addrspace(1) @f64_global
  ret void
}

; CHECK: .globaltype i32_global, i32
; CHECK: .globl i32_global
; CHECK-LABEL: i32_global:

; CHECK: .globaltype i64_global, i64
; CHECK: .globl i64_global
; CHECK-LABEL: i64_global:

; CHECK: .globaltype f32_global, f32
; CHECK: .globl f32_global
; CHECK-LABEL: f32_global:

; CHECK: .globaltype f64_global, f64
; CHECK: .globl f64_global
; CHECK-LABEL: f64_global:
