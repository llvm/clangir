// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s


int main() {
  int a[10];
  int b[4][7];
  int *aa = a;
  int *p = &a[0];
  auto *ap = &a;
  *ap[3] = 7;
  auto *ap10 = (int (*)[10]) p;

  auto v15 = b[1][5];
  int *bpd = b[0];
  auto pb36 = &b[3][6];
  auto pb2 = &b[2];
  auto pb2a = b[2];

  return a[3];

  // CHECK: %[[ALLOCA:.+]] = memref.alloca() {alignment = 8 : i64} : memref<!named_tuple.named_tuple<"s", [i32, f64, i8, tensor<5xf32>]>>
  // CHECK: %[[C_7:.+]] = arith.constant 7 : i32/home/rkeryell/Xilinx/Projects/LLVM/worktrees/clangir/build/bin/clang -cc1 -internal-isystem /home/rkeryell/Xilinx/Projects/LLVM/worktrees/clangir/build/lib/clang/20/include -nostdsysteminc -triple x86_64-unknown-linux-gnu -fno-clangir-direct-lowering -emit-mlir /home/rkeryell/Xilinx/Projects/LLVM/worktrees/clangir/clang/test/CIR/Lowering/ThroughMLIR/cast.cpp -o 
  // CHECK: %[[I8_EQUIV_A:.+]] = named_tuple.cast %[[ALLOCA]] : memref<!named_tuple.named_tuple<"s", [i32, f64, i8, tensor<5xf32>]>> to memref<40xi8>
  // CHECK: %[[OFFSET_A:.+]] = arith.constant 0 : index
  // CHECK: %[[VIEW_A:.+]] = memref.view %[[I8_EQUIV_A]][%[[OFFSET_A]]][] : memref<40xi8> to memref<i32>
  // CHECK: memref.store %[[C_7]], %[[VIEW_A]][] : memref<i32>
 }
