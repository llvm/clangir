// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fopenmp-enable-irbuilder -fopenmp -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// CHECK: cir.func
void omp_master_1() {
// CHECK: omp.master {
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: }
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: }
#pragma omp master
{
}
}

// CHECK: cir.func
void omp_master_2() {
// CHECK: %[[YVarDecl:.+]] = {{.*}} ["y", init]
// CHECK: omp.master {
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[XVarDecl:.+]] = {{.*}} ["x", init]
// CHECK-NEXT: %[[C1:.+]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store %[[C1]], %[[XVarDecl]]
// CHECK-NEXT: %[[XVal:.+]] = cir.load %[[XVarDecl]]
// CHECK-NEXT: %[[COne:.+]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: %[[BinOpVal:.+]] = cir.binop(add, %[[XVal]], %[[COne]])
// CHECK-NEXT: cir.store %[[BinOpVal]], %[[YVarDecl]]
// CHECK-NEXT: }
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: }
  int y = 0;
#pragma omp master
{
  int x = 1;
  y = x + 1;
}
}

