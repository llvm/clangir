// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir -clangir-disable-passes %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIRGEN
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=LLVM

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

//      CIR: cir.func private @__cxa_guard_release(!cir.ptr<!s64i>)
//      CIR: cir.func private @__cxa_guard_acquire(!cir.ptr<!s64i>) -> !s32i
//      CIR: cir.func private @_Z3fnAv() -> !s32i
//      CIR: cir.global "private" internal dsolocal @_ZZ3foovE3val = #cir.int<0> : !s32i {alignment = 4 : i64, ast = #cir.var.decl.ast, static_local}
//      CIR: cir.global "private" internal dsolocal @_ZGVZ3foovE3val = #cir.int<0> : !s64i {alignment = 8 : i64}
//      CIR: cir.func @_Z3foov() extra(#fn_attr) {
// CIR-NEXT:   %0 = cir.get_global @_ZZ3foovE3val : !cir.ptr<!s32i> {static_local}
// CIR-NEXT:   %1 = cir.get_global @_ZGVZ3foovE3val : !cir.ptr<!s64i>
// CIR-NEXT:   %2 = cir.cast(bitcast, %1 : !cir.ptr<!s64i>), !cir.ptr<!s8i>
// CIR-NEXT:   %3 = cir.load atomic(acquire) %2 : !cir.ptr<!s8i>, !s8i
// CIR-NEXT:   %4 = cir.const #cir.int<0> : !s8i
// CIR-NEXT:   %5 = cir.cmp(eq, %3, %4) : !s8i, !cir.bool
// CIR-NEXT:   cir.if %5 {
// CIR-NEXT:     %6 = cir.call @__cxa_guard_acquire(%1) : (!cir.ptr<!s64i>) -> !s32i
// CIR-NEXT:     %7 = cir.const #cir.int<0> : !s32i
// CIR-NEXT:     %8 = cir.cmp(ne, %6, %7) : !s32i, !cir.bool
// CIR-NEXT:     cir.if %8 {
// CIR-NEXT:       %9 = cir.get_global @_ZZ3foovE3val : !cir.ptr<!s32i>
// CIR-NEXT:       %10 = cir.call @_Z3fnAv() : () -> !s32i
// CIR-NEXT:       cir.store %10, %9 : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:       cir.call @__cxa_guard_release(%1) : (!cir.ptr<!s64i>) -> ()
// CIR-NEXT:     }
// CIR-NEXT:   }
// CIR-NEXT:   cir.return
// CIR-NEXT: }

//      LLVM: @_ZZ3foovE3val = internal global i32 0, align 4
//      LLVM: @_ZGVZ3foovE3val = internal global i64 0, align 8
//      LLVM: declare void @__cxa_guard_release(ptr)
//      LLVM: declare i32 @__cxa_guard_acquire(ptr)
//      LLVM: declare i32 @_Z3fnAv()

//      LLVM: define dso_local void @_Z3foov()
// LLVM-NEXT:   %1 = load atomic i8, ptr @_ZGVZ3foovE3val acquire, align 1
// LLVM-NEXT:   %2 = icmp eq i8 %1, 0
// LLVM-NEXT:   br i1 %2, label %3, label %9
//  LLVM-DAG: 3:
// LLVM-NEXT:   %4 = call i32 @__cxa_guard_acquire(ptr @_ZGVZ3foovE3val)
// LLVM-NEXT:   %5 = icmp ne i32 %4, 0
// LLVM-NEXT:   br i1 %5, label %6, label %8
//  LLVM-DAG: 6:
// LLVM-NEXT:   %7 = call i32 @_Z3fnAv()
// LLVM-NEXT:   store i32 %7, ptr @_ZZ3foovE3val, align 4
// LLVM-NEXT:   call void @__cxa_guard_release(ptr @_ZGVZ3foovE3val)
// LLVM-NEXT:   br label %8
//  LLVM-DAG: 8:
// LLVM-NEXT:   br label %9
//  LLVM-DAG: 9:
// LLVM-NEXT:   ret void
// LLVM-NEXT: }
