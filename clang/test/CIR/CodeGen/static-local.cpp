// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIRGEN

int fnA();

void foo() {
  static int val = fnA();
}


//      CIRGEN: cir.func private @_Z3fnAv() -> !s32i attributes
// CIRGEN-NEXT: cir.global "private" internal dsolocal @_ZZ3foovE3val = ctor : !s32i {
// CIRGEN-NEXT:   %0 = cir.get_global @_ZZ3foovE3val : !cir.ptr<!s32i>
// CIRGEN-NEXT:   %1 = cir.call @_Z3fnAv() : () -> !s32i
// CIRGEN-NEXT:   cir.store %1, %0 : !s32i, !cir.ptr<!s32i>
// CIRGEN-NEXT: } {alignment = 4 : i64, ast = #cir.var.decl.ast, static_local}
// CIRGEN-NEXT: cir.func @_Z3foov() attributes
// CIRGEN-NEXT:   %0 = cir.get_global @_ZZ3foovE3val : !cir.ptr<!s32i> {static_local}
// CIRGEN-NEXT:   cir.return
// CIRGEN-NEXT: }

//      CIR: cir.func private @_Z3fnAv() -> !s32i
// CIR-NEXT: cir.global "private" internal dsolocal @_ZZ3foovE3val = #cir.int<0> : !s32i {alignment = 4 : i64, ast = #cir.var.decl.ast, static_local}
// CIR-NEXT: cir.global "private" internal dsolocal @_ZGVZ3foovE3val = #cir.int<0> : !s64i {alignment = 8 : i64}
// CIR-NEXT: cir.func private @__cxa_guard_acquire(!cir.ptr<!s64i>) -> !s32i
// CIR-NEXT: cir.func private @__cxa_guard_release(!cir.ptr<!s64i>)
// CIR-NEXT: cir.func @_Z3foov() extra(#fn_attr) {
// CIR-NEXT:   %0 = cir.get_global @_ZZ3foovE3val : !cir.ptr<!s32i> {static_local}
// CIR-NEXT:   %1 = cir.get_global @_ZGVZ3foovE3val : !cir.ptr<!s64i>
// CIR-NEXT:   %2 = cir.cast(bitcast, %1 : !cir.ptr<!s64i>), !cir.ptr<!s8i>
// CIR-NEXT:   %3 = cir.load atomic(acquire) %2 : !cir.ptr<!s8i>, !s8i
// CIR-NEXT:   %4 = cir.const #cir.int<0> : !s8i
// CIR-NEXT:   %5 = cir.cmp(eq, %3, %4) : !s8i, !cir.bool
// CIR-NEXT:   cir.brcond %5 ^bb1, ^bb3
// CIR-NEXT: ^bb1:  // pred: ^bb0
// CIR-NEXT:   %6 = cir.call @__cxa_guard_acquire(%1) : (!cir.ptr<!s64i>) -> !s32i
// CIR-NEXT:   %7 = cir.const #cir.int<0> : !s32i
// CIR-NEXT:   %8 = cir.cmp(ne, %6, %7) : !s32i, !cir.bool
// CIR-NEXT:   cir.brcond %8 ^bb2, ^bb3
// CIR-NEXT: ^bb2:  // pred: ^bb1
// CIR-NEXT:   %9 = cir.get_global @_ZZ3foovE3val : !cir.ptr<!s32i>
// CIR-NEXT:   %10 = cir.call @_Z3fnAv() : () -> !s32i
// CIR-NEXT:   cir.store %10, %9 : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   cir.call @__cxa_guard_release(%1) : (!cir.ptr<!s64i>) -> ()
// CIR-NEXT:   cir.br ^bb3
// CIR-NEXT: ^bb3:  // 3 preds: ^bb0, ^bb1, ^bb2
// CIR-NEXT:   cir.return
// CIR-NEXT: }

//      CHECK: @_ZZ3foovE3val = internal global i32 0, align 4
//      CHECK: @_ZGVZ3foovE3val = internal global i64 0, align 8
//      CHECK: declare i32 @_Z3fnAv()
//      CHECK: declare i32 @__cxa_guard_acquire(ptr)
//      CHECK: declare void @__cxa_guard_release(ptr)

//      CHECK: define dso_local void @_Z3foov()
// CHECK-NEXT:   %1 = load atomic i8, ptr @_ZGVZ3foovE3val acquire, align 1
// CHECK-NEXT:   %2 = icmp eq i8 %1, 0
// CHECK-NEXT:   br i1 %2, label %3, label %8
//  CHECK-DAG: 3:
// CHECK-NEXT:   %4 = call i32 @__cxa_guard_acquire(ptr @_ZGVZ3foovE3val)
// CHECK-NEXT:   %5 = icmp ne i32 %4, 0
// CHECK-NEXT:   br i1 %5, label %6, label %8
//  CHECK-DAG: 6:
// CHECK-NEXT:   %7 = call i32 @_Z3fnAv()
// CHECK-NEXT:   store i32 %7, ptr @_ZZ3foovE3val, align 4
// CHECK-NEXT:   call void @__cxa_guard_release(ptr @_ZGVZ3foovE3val)
// CHECK-NEXT:   br label %8
//  CHECK-DAG: 8:
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
