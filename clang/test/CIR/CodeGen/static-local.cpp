// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -clangir-disable-passes %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIRGEN
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

int fnA();

void foo() {
  static int val = fnA();
}


//      CIRGEN: cir.func private @_Z3fnAv() -> !s32i
//      CIRGEN: cir.global "private" internal dso_local static_local @_ZZ3foovE3val = ctor : !s32i {
// CIRGEN-NEXT:   %0 = cir.get_global @_ZZ3foovE3val : !cir.ptr<!s32i>
// CIRGEN-NEXT:   %1 = cir.call @_Z3fnAv() : () -> !s32i
// CIRGEN-NEXT:   cir.store align(4) %1, %0 : !s32i, !cir.ptr<!s32i>
// CIRGEN-NEXT: } {alignment = 4 : i64, ast = #cir.var.decl.ast}
// CIRGEN-NEXT: cir.func
//      CIRGEN:   %0 = cir.get_global static_local @_ZZ3foovE3val : !cir.ptr<!s32i>
// CIRGEN-NEXT:   cir.return
// CIRGEN-NEXT: }

//      CIR: cir.func private @__cxa_guard_release(!cir.ptr<!s64i>)
//      CIR: cir.func private @__cxa_guard_acquire(!cir.ptr<!s64i>) -> !s32i
//      CIR: cir.func private @_Z3fnAv() -> !s32i
//      CIR: cir.global "private" internal dso_local static_local @_ZZ3foovE3val = #cir.int<0> : !s32i {alignment = 4 : i64, ast = #cir.var.decl.ast}
//      CIR: cir.global "private" internal dso_local @_ZGVZ3foovE3val = #cir.int<0> : !s64i {alignment = 8 : i64}
//      CIR: cir.func {{.*}} @_Z3foov() extra(#fn_attr) {
// CIR-NEXT:   %0 = cir.get_global static_local @_ZZ3foovE3val : !cir.ptr<!s32i>
// CIR-NEXT:   %1 = cir.get_global @_ZGVZ3foovE3val : !cir.ptr<!s64i>
// CIR-NEXT:   %2 = cir.cast bitcast %1 : !cir.ptr<!s64i> -> !cir.ptr<!s8i>
// CIR-NEXT:   %3 = cir.load align(8) syncscope(system) atomic(acquire) %2 : !cir.ptr<!s8i>, !s8i
// CIR-NEXT:   %4 = cir.const #cir.int<1> : !s8i
// CIR-NEXT:   %5 = cir.binop(and, %3, %4) : !s8i
// CIR-NEXT:   %6 = cir.const #cir.int<0> : !s8i
// CIR-NEXT:   %7 = cir.cmp(eq, %5, %6) : !s8i, !cir.bool
// CIR-NEXT:   cir.if %7 {
// CIR-NEXT:     %8 = cir.call @__cxa_guard_acquire(%1) : (!cir.ptr<!s64i>) -> !s32i
// CIR-NEXT:     %9 = cir.const #cir.int<0> : !s32i
// CIR-NEXT:     %10 = cir.cmp(ne, %8, %9) : !s32i, !cir.bool
// CIR-NEXT:     cir.if %10 {
// CIR-NEXT:       %11 = cir.get_global @_ZZ3foovE3val : !cir.ptr<!s32i>
// CIR-NEXT:       %12 = cir.call @_Z3fnAv() : () -> !s32i
// CIR-NEXT:       cir.store align(4) %12, %11 : !s32i, !cir.ptr<!s32i>
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
// LLVM-NEXT:   %1 = load atomic i8, ptr @_ZGVZ3foovE3val acquire, align 8
// LLVM-NEXT:   %2 = and i8 %1, 1
// LLVM-NEXT:   %3 = icmp eq i8 %2, 0
// LLVM-NEXT:   br i1 %3, label %4, label %10

//  LLVM-DAG: 4:
// LLVM-NEXT:   %5 = call i32 @__cxa_guard_acquire(ptr @_ZGVZ3foovE3val)
// LLVM-NEXT:   %6 = icmp ne i32 %5, 0
// LLVM-NEXT:   br i1 %6, label %7, label %9

//  LLVM-DAG: 7:
// LLVM-NEXT:   %8 = call i32 @_Z3fnAv()
// LLVM-NEXT:   store i32 %8, ptr @_ZZ3foovE3val, align 4
// LLVM-NEXT:   call void @__cxa_guard_release(ptr @_ZGVZ3foovE3val)
// LLVM-NEXT:   br label %9

//  LLVM-DAG: 9:
// LLVM-NEXT:   br label %10

//  LLVM-DAG: 10:
// LLVM-NEXT:   ret void
// LLVM-NEXT: }

//      OGCG: @_ZZ3foovE3val = internal global i32 0, align 4
//      OGCG: @_ZGVZ3foovE3val = internal global i64 0, align 8

//      OGCG: define dso_local void @_Z3foov()
//      OGCG: entry:
// OGCG-NEXT:   %[[GUARD_LOAD:.*]] = load atomic i8, ptr @_ZGVZ3foovE3val acquire, align 8
// OGCG-NEXT:   %[[GUARD_UNINIT:.*]] = icmp eq i8 %[[GUARD_LOAD]], 0
// OGCG-NEXT:   br i1 %[[GUARD_UNINIT]], label %[[INIT_CHECK:.*]], label %[[INIT_END:.*]],

// OGCG-DAG: [[INIT_CHECK]]:
// OGCG-NEXT:   %[[GUARD_ACQ:.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVZ3foovE3val)
// OGCG-NEXT:   %[[TOBOOL:.*]] = icmp ne i32 %[[GUARD_ACQ]], 0
// OGCG-NEXT:   br i1 %[[TOBOOL]], label %[[INIT:.*]], label %[[INIT_END2:.*]]

// OGCG-DAG: [[INIT]]:
// OGCG-NEXT:   %[[CALL:.*]] = call noundef i32 @_Z3fnAv()
// OGCG-NEXT:   store i32 %[[CALL]], ptr @_ZZ3foovE3val, align 4
// OGCG-NEXT:   call void @__cxa_guard_release(ptr @_ZGVZ3foovE3val)
// OGCG-NEXT:   br label %[[INIT_END3:.*]]

// OGCG-DAG: [[INIT_END3]]:
// OGCG-NEXT:   ret void
// OGCG-NEXT: }

//      OGCG: declare i32 @__cxa_guard_acquire(ptr)
//      OGCG: declare noundef i32 @_Z3fnAv()
//      OGCG: declare void @__cxa_guard_release(ptr)
