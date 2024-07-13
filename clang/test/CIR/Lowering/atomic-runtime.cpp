// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: cir-opt %t.cir -cir-to-llvm -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

// Test __atomic_* built-ins that have a memory order parameter with a runtime
// value.  This requires generating a switch statement, so the amount of
// generated code is surprisingly large.
//
// This is just a quick smoke test.  Only atomic_load_n is tested.

int runtime_load(int *ptr, int order) {
  return __atomic_load_n(ptr, order);
}

// CHECK:  %[[T8:.*]] = llvm.load %[[T1:.*]] {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
// CHECK:  %[[T9:.*]] = llvm.load %[[T3:.*]] {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK:  llvm.switch %[[T9]] : i32, ^[[BB1:.*]] [
// CHECK:    1: ^[[BB2:.*]],
// CHECK:    2: ^[[BB2]],
// CHECK:    5: ^[[BB3:.*]]
// CHECK:  ]
// CHECK: [[BB1]]:
// CHECK:  %[[T10:.*]] = llvm.load %[[T8]] atomic monotonic {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK:  llvm.store %[[T10]], %[[T7:.*]] {alignment = 4 : i64} : i32, !llvm.ptr
// CHECK:  llvm.br ^[[BB4:.*]]
// CHECK: [[BB2]]:
// CHECK:  %[[T11:.*]] = llvm.load %[[T8]] atomic acquire {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK:  llvm.store %[[T11]], %[[T7]] {alignment = 4 : i64} : i32, !llvm.ptr
// CHECK:  llvm.br ^[[BB4]]
// CHECK: [[BB3]]:
// CHECK:  %[[T12:.*]] = llvm.load %[[T8]] atomic seq_cst {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK:  llvm.store %[[T12]], %[[T7]] {alignment = 4 : i64} : i32, !llvm.ptr
// CHECK:  llvm.br ^[[BB4]]
// CHECK: [[BB4]]:
// CHECK:  %[[T13:.*]] = llvm.load %[[T7]] {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK:  llvm.store %[[T13]], %[[T5:.*]] {alignment = 4 : i64} : i32, !llvm.ptr
// CHECK:  %[[T14:.*]] = llvm.load %[[T5]] {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK:  llvm.return %[[T14]] : i32
