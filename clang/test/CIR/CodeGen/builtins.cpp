// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -emit-cir %s -o %t.cir  
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:  -emit-llvm -fno-clangir-call-conv-lowering -o - %s \
// RUN:  | opt -S -passes=instcombine,mem2reg,simplifycfg -o %t.ll 
// RUN: FileCheck  --check-prefix=LLVM --input-file=%t.ll %s

// This test file is a collection of test cases for all target-independent
// builtins that are related to memory operations.

int *test_addressof(int s) {
  return __builtin_addressof(s);
  
  // CIR-LABEL: test_addressof
  // CIR: cir.store %arg0, [[ADDR:%.*]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.store [[ADDR]], [[SAVE:%.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR: [[RES:%.*]] = cir.load [[SAVE]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: cir.return [[RES]] : !cir.ptr<!s32i>

  // LLVM: {{.*}}test_addressof{{.*}}(i32{{.*}}[[S:%.*]])
  // LLVM: store i32 [[S]], ptr [[ADDR:%.*]], align 4
  // LLVM: store ptr [[ADDR]], ptr [[SAVE:%.*]], align 8
  // LLVM: [[RES:%.*]] = load ptr, ptr [[SAVE]], align 8
  // LLVM: ret ptr [[RES]]
}

namespace std { template<typename T> T *addressof(T &); }
int *test_std_addressof(int s) {
  return std::addressof(s);
  
  // CIR-LABEL: test_std_addressof
  // CIR: cir.store %arg0, [[ADDR:%.*]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.store [[ADDR]], [[SAVE:%.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR: [[RES:%.*]] = cir.load [[SAVE]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: cir.return [[RES]] : !cir.ptr<!s32i>

  // LLVM: {{.*}}test_std_addressof{{.*}}(i32{{.*}}[[S:%.*]])
  // LLVM: store i32 [[S]], ptr [[ADDR:%.*]], align 4
  // LLVM: store ptr [[ADDR]], ptr [[SAVE:%.*]], align 8
  // LLVM: [[RES:%.*]] = load ptr, ptr [[SAVE]], align 8
  // LLVM: ret ptr [[RES]]
}

namespace std { template<typename T> T *__addressof(T &); }
int *test_std_addressof2(int s) {
  return std::__addressof(s);
  
  // CIR-LABEL: test_std_addressof2
  // CIR: cir.store %arg0, [[ADDR:%.*]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.store [[ADDR]], [[SAVE:%.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR: [[RES:%.*]] = cir.load [[SAVE]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: cir.return [[RES]] : !cir.ptr<!s32i>

  // LLVM: {{.*}}test_std_addressof2{{.*}}(i32{{.*}}[[S:%.*]])
  // LLVM: store i32 [[S]], ptr [[ADDR:%.*]], align 4
  // LLVM: store ptr [[ADDR]], ptr [[SAVE:%.*]], align 8
  // LLVM: [[RES:%.*]] = load ptr, ptr [[SAVE]], align 8
  // LLVM: ret ptr [[RES]]
}
