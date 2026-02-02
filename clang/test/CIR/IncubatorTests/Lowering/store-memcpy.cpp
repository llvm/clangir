// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void foo() {
  char s1[] = "Hello";
}

// CIR-DAG: cir.global "private"{{.*}}constant{{.*}}@__const._Z3foov.s1 = #cir.const_array<"Hello\00" : !cir.array<!s8i x 6>>
// CIR: cir.func{{.*}}@_Z3foov
// CIR:   %[[S1:.*]] = cir.alloca !cir.array<!s8i x 6>, !cir.ptr<!cir.array<!s8i x 6>>, ["s1", init]
// CIR:   %[[CONST:.*]] = cir.get_global @__const._Z3foov.s1 : !cir.ptr<!cir.array<!s8i x 6>>
// CIR:   cir.copy %[[CONST]] to %[[S1]] : !cir.ptr<!cir.array<!s8i x 6>>
// CIR:   cir.return
// CIR: }

// LLVM: @__const._Z3foov.s1 = private constant [6 x i8] c"Hello\00"
// LLVM: @_Z3foov()
// LLVM:   %[[S1:.*]] = alloca [6 x i8], i64 1, align 1
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr %[[S1]], ptr @__const._Z3foov.s1, i64 6, i1 false)
// LLVM:   ret void

// OGCG: @__const._Z3foov.s1 = private unnamed_addr constant [6 x i8] c"Hello\00"
// OGCG: @_Z3foov()
// OGCG:   %[[S1:.*]] = alloca [6 x i8], align 1
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[S1]], ptr align 1 @__const._Z3foov.s1, i64 6, i1 false)
// OGCG:   ret void
