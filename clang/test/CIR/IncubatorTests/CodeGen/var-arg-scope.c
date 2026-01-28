// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

void f1(__builtin_va_list c) {
  { __builtin_va_arg(c, void *); }
}

// CIR: cir.func {{.*}} @f1(%arg0: !rec___va_list)
// CIR: [[VAR_LIST:%.*]] = cir.alloca !rec___va_list, !cir.ptr<!rec___va_list>, ["c", init] {alignment = 8 : i64}
// CIR: cir.store %arg0, [[VAR_LIST]] : !rec___va_list, !cir.ptr<!rec___va_list>
// CIR: cir.scope {
// CIR-NEXT: [[TMP:%.*]] = cir.va_arg [[VAR_LIST]] : (!cir.ptr<!rec___va_list>) -> !cir.ptr<!void>
// CIR-NEXT: }
// CIR-NEXT: cir.return

// LLVM: %struct.__va_list = type { ptr, ptr, ptr, i32, i32 }
// LLVM: define dso_local void @f1(%struct.__va_list %0)
// LLVM: [[VARLIST:%.*]] = alloca %struct.__va_list, i64 1, align 8
// LLVM: br label %[[SCOPE_FRONT:.*]]
// LLVM: [[SCOPE_FRONT]]:
// LLVM: [[TMP:%.*]] = va_arg ptr [[VARLIST]], ptr
// LLVM: br label %[[OUT_SCOPE:.*]]
// LLVM: [[OUT_SCOPE]]:
// LLVM-NEXT:  ret void
