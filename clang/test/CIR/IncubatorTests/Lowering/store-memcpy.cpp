// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t2.cir 2>&1 | FileCheck -check-prefix=AFTER %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

void foo() {
  char s1[] = "Hello";
}
// AFTER: @_Z3foov
// AFTER:    %[[S1:.*]] = cir.alloca !cir.array<!s8i x 6>, !cir.ptr<!cir.array<!s8i x 6>>, ["s1", init]
// AFTER:    %[[HELLO:.*]] = cir.const #cir.const_array<"Hello\00" : !cir.array<!s8i x 6>>
// AFTER:    cir.store{{.*}} %[[HELLO]], %[[S1]] : !cir.array<!s8i x 6>, !cir.ptr<!cir.array<!s8i x 6>>
// AFTER:    cir.return
// AFTER:  }

// LLVM: @_Z3foov()
// LLVM:   %[[S1:.*]] = alloca [6 x i8], i64 1, align 1
// LLVM:   store [6 x i8] c"Hello\00", ptr %[[S1]]
// LLVM:   ret void
