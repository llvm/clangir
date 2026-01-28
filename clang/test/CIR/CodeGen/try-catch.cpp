// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir-flat %s -o %t.flat.cir
// RUN: FileCheck --input-file=%t.flat.cir %s -check-prefix=CIR_FLAT
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void empty_try_block_with_catch_all() {
  try {} catch (...) {}
}

// CIR: cir.func{{.*}} @_Z30empty_try_block_with_catch_allv()
// CIR:   cir.return

// CIR_FLAT: cir.func{{.*}} @_Z30empty_try_block_with_catch_allv()
// CIR_FLAT:   cir.return

// LLVM: define{{.*}} void @_Z30empty_try_block_with_catch_allv()
// LLVM:  ret void

// OGCG: define{{.*}} void @_Z30empty_try_block_with_catch_allv()
// OGCG:   ret void

void empty_try_block_with_catch_with_int_exception() {
  try {} catch (int e) {}
}

// CIR: cir.func{{.*}} @_Z45empty_try_block_with_catch_with_int_exceptionv()
// CIR:   cir.return

// CIR_FLAT: cir.func{{.*}} @_Z45empty_try_block_with_catch_with_int_exceptionv()
// CIR_FLAT:   cir.return

// LLVM: define{{.*}} void @_Z45empty_try_block_with_catch_with_int_exceptionv()
// LLVM:  ret void

// OGCG: define{{.*}} void @_Z45empty_try_block_with_catch_with_int_exceptionv()
// OGCG:   ret void

void try_catch_with_empty_catch_all() {
  int a = 1;
  try {
    return;
    ++a;
  } catch (...) {
  }
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store{{.*}} %[[CONST_1]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i
// CIR: cir.scope {
// CIR:   cir.try {
// CIR:     cir.return
// CIR:   ^bb1:  // no predecessors
// CIR:     %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:     %[[RESULT:.*]] = cir.unary(inc, %[[TMP_A]]) nsw : !s32i, !s32i
// CIR:     cir.store{{.*}} %[[RESULT]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:     cir.yield
// CIR:   }
// CIR: }

// CIR_FLAT: cir.func{{.*}} @_Z30try_catch_with_empty_catch_allv()
// CIR_FLAT:   %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR_FLAT:   %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR_FLAT:   cir.store{{.*}} %[[CONST_1]], %[[A_ADDR]]
// CIR_FLAT:   cir.br ^bb[[#SCOPE_ENTRY:]]
// CIR_FLAT: ^bb[[#SCOPE_ENTRY]]:
// CIR_FLAT:   cir.br ^bb[[#TRY_ENTRY:]]
// CIR_FLAT: ^bb[[#TRY_ENTRY]]:
// CIR_FLAT:   cir.return

// LLVM:   %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:   store i32 1, ptr %[[A_ADDR]], align 4
// LLVM:   br label %[[BB_2:.*]]
// LLVM: [[BB_2]]:
// LLVM:   br label %[[BB_3:.*]]
// LLVM: [[BB_3]]:
// LLVM:   ret void
// LLVM: [[BB_4:.*]]:
// LLVM:   %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:   %[[RESULT:.*]] = add nsw i32 %[[TMP_A]], 1
// LLVM:   store i32 %[[RESULT]], ptr %[[A_ADDR]], align 4
// LLVM:   br label %[[BB_7:.*]]
// LLVM: [[BB_7]]:
// LLVM:   br label %[[BB_8:.*]]
// LLVM: [[BB_8]]:
// LLVM:   ret void

// OGCG: %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG: store i32 1, ptr %[[A_ADDR]], align 4
// OGCG: ret void

void try_catch_with_empty_catch_all_2() {
  int a = 1;
  try {
    ++a;
    return;
  } catch (...) {
  }
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store{{.*}} %[[CONST_1]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.scope {
// CIR:   cir.try {
// CIR:     %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:     %[[RESULT:.*]] = cir.unary(inc, %[[TMP_A]]) nsw : !s32i, !s32i
// CIR:     cir.store{{.*}} %[[RESULT]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:     cir.return
// CIR:   }
// CIR: }

// CIR_FLAT: cir.func{{.*}} @_Z32try_catch_with_empty_catch_all_2v()
// CIR_FLAT:   %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR_FLAT:   %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR_FLAT:   cir.store{{.*}} %[[CONST_1]], %[[A_ADDR]]
// CIR_FLAT:   cir.br ^bb[[#SCOPE_ENTRY:]]
// CIR_FLAT: ^bb[[#SCOPE_ENTRY]]:
// CIR_FLAT:   cir.br ^bb[[#TRY_ENTRY:]]
// CIR_FLAT: ^bb[[#TRY_ENTRY]]:
// CIR_FLAT:   %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]]
// CIR_FLAT:   %[[RESULT:.*]] = cir.unary(inc, %[[TMP_A]]) nsw
// CIR_FLAT:   cir.store{{.*}} %[[RESULT]], %[[A_ADDR]]
// CIR_FLAT:   cir.return

// LLVM:   %[[A_ADDR]] = alloca i32, i64 1, align 4
// LLVM:   store i32 1, ptr %[[A_ADDR]], align 4
// LLVM:   br label %[[BB_2:.*]]
// LLVM: [[BB_2]]:
// LLVM:   br label %[[BB_3:.*]]
// LLVM: [[BB_3]]:
// LLVM:   %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:   %[[RESULT:.*]] = add nsw i32 %[[TMP_A:.*]], 1
// LLVM:   store i32 %[[RESULT]], ptr %[[A_ADDR]], align 4
// LLVM:   ret void
// LLVM: [[BB_6:.*]]:
// LLVM:   br label %[[BB_7:.*]]
// LLVM: [[BB_7]]:
// LLVM:   ret void

// OGCG: %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG: store i32 1, ptr %[[A_ADDR]], align 4
// OGCG: %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG: %[[RESULT:.*]] = add nsw i32 %[[TMP_A]], 1
// OGCG: store i32 %[[RESULT]], ptr %[[A_ADDR]], align 4
// OGCG: ret void

void try_catch_with_alloca() {
  try {
    int a;
    int b;
    int c = a + b;
  } catch (...) {
  }
}

// CIR: cir.scope {
// CIR:   %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a"]
// CIR:   %[[B_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b"]
// CIR:   %[[C_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["c", init]
// CIR:   cir.try {
// CIR:     %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:     %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:     %[[RESULT:.*]] = cir.binop(add, %[[TMP_A]], %[[TMP_B]]) nsw : !s32i
// CIR:     cir.store{{.*}} %[[RESULT]], %[[C_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:     cir.yield
// CIR:   }
// CIR: }

// CIR_FLAT: cir.func{{.*}} @_Z21try_catch_with_allocav()
// CIR_FLAT:   %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a"]
// CIR_FLAT:   %[[B_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b"]
// CIR_FLAT:   %[[C_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["c", init]
// CIR_FLAT:   cir.br ^bb[[#SCOPE_ENTRY:]]
// CIR_FLAT: ^bb[[#SCOPE_ENTRY]]:
// CIR_FLAT:   cir.br ^bb[[#TRY_ENTRY:]]
// CIR_FLAT: ^bb[[#TRY_ENTRY]]:
// CIR_FLAT:   %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]]
// CIR_FLAT:   %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]]
// CIR_FLAT:   %[[RESULT:.*]] = cir.binop(add, %[[TMP_A]], %[[TMP_B]]) nsw
// CIR_FLAT:   cir.store{{.*}} %[[RESULT]], %[[C_ADDR]]
// CIR_FLAT:   cir.br ^bb[[#TRY_CONT:]]
// CIR_FLAT: ^bb[[#TRY_CONT]]:
// CIR_FLAT:   cir.br ^bb[[#SCOPE_EXIT:]]
// CIR_FLAT: ^bb[[#SCOPE_EXIT]]:
// CIR_FLAT:   cir.return

// LLVM:  %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:  %[[B_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:  %[[C_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:  br label %[[LABEL_1:.*]]
// LLVM: [[LABEL_1]]:
// LLVM:  br label %[[LABEL_2:.*]]
// LLVM: [[LABEL_2]]:
// LLVM:  %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:  %[[TMP_B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// LLVM:  %[[RESULT:.*]] = add nsw i32 %[[TMP_A]], %[[TMP_B]]
// LLVM:  store i32 %[[RESULT]], ptr %[[C_ADDR]], align 4
// LLVM:  br label %[[LABEL_3:.*]]
// LLVM: [[LABEL_3]]:
// LLVM:  br label %[[LABEL_4:.*]]
// LLVM: [[LABEL_4]]:
// LLVM:  ret void

// OGCG: %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[B_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[C_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG: %[[TMP_B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// OGCG: %[[RESULT:.*]] = add nsw i32 %[[TMP_A]], %[[TMP_B]]
// OGCG: store i32 %[[RESULT]], ptr %[[C_ADDR]], align 4

void function_with_noexcept() noexcept;

void calling_noexcept_function_inside_try_block() {
  try {
    function_with_noexcept();
  } catch (...) {
  }
}

// CIR: cir.scope {
// CIR:   cir.try {
// CIR:     cir.call @_Z22function_with_noexceptv() nothrow : () -> ()
// CIR:     cir.yield
// CIR:   }
// CIR: }

// CIR_FLAT: cir.func{{.*}} @_Z42calling_noexcept_function_inside_try_blockv()
// CIR_FLAT:   cir.br ^bb[[#SCOPE_ENTRY:]]
// CIR_FLAT: ^bb[[#SCOPE_ENTRY]]:
// CIR_FLAT:   cir.br ^bb[[#TRY_ENTRY:]]
// CIR_FLAT: ^bb[[#TRY_ENTRY]]:
// CIR_FLAT:   cir.call @_Z22function_with_noexceptv() nothrow
// CIR_FLAT:   cir.br ^bb[[#TRY_CONT:]]
// CIR_FLAT: ^bb[[#TRY_CONT]]:
// CIR_FLAT:   cir.br ^bb[[#SCOPE_EXIT:]]
// CIR_FLAT: ^bb[[#SCOPE_EXIT]]:
// CIR_FLAT:   cir.return

// LLVM:   br label %[[LABEL_1:.*]]
// LLVM: [[LABEL_1]]:
// LLVM:   br label %[[LABEL_2:.*]]
// LLVM: [[LABEL_2]]:
// LLVM:   call void @_Z22function_with_noexceptv()
// LLVM:   br label %[[LABEL_3:.*]]
// LLVM: [[LABEL_3]]:
// LLVM:   br label %[[LABEL_4:.*]]
// LLVM: [[LABEL_4]]:
// LLVM:   ret void

// OGCG: call void @_Z22function_with_noexceptv()
// OGCG: ret void
