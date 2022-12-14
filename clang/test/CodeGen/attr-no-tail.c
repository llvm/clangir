// RUN: %clang_cc1 -triple x86_64-apple-macosx10.7.0 %s -emit-llvm -o - | FileCheck %s

// CHECK: %{{[a-z0-9]+}} = notail call i32 @callee0(i32 noundef %
// CHECK: %{{[a-z0-9]+}} = notail call i32 @callee1(i32 noundef %

// Check that indirect calls do not have the notail marker.
// CHECK: store ptr @callee1, ptr [[ALLOCA1:%[A-Za-z0-9]+]], align 8
// CHECK: [[INDIRFUNC:%[0-9]+]] = load ptr, ptr [[ALLOCA1]], align 8
// CHECK: %{{[a-z0-9]+}} = call i32 [[INDIRFUNC]](i32 noundef %{{[0-9]+}}

// CHECK: %{{[a-z0-9]+}} = call i32 @callee2(i32 noundef %

int callee0(int a) __attribute__((not_tail_called)) {
  return a + 1;
}

int callee1(int) __attribute__((not_tail_called));

int callee2(int);

typedef int (*FuncTy)(int);

int foo0(int a) {
  if (a > 1)
    return callee0(a);
  if (a == 1)
    return callee1(a);
  if (a < 0) {
    FuncTy F = callee1;
    return (*F)(a);
  }
  return callee2(a);
}
