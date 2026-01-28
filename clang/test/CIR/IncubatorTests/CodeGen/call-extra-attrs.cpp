// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

__attribute__((nothrow))
int s0(int a, int b) {
  int x = a + b;
  return x;
}

__attribute__((noinline))
int s1(int a, int b) {
  return s0(a,b);
}

int s2(int a, int b) {
  return s1(a, b);
}

// CIR: cir.func{{.*}} no_inline optnone {{.*}} @_Z2s0ii(%{{.*}}, %{{.*}}) -> {{.*}} extra(#cir<extra({nothrow = #cir.nothrow, side_effect = 0 : i32})>)
// CIR: cir.func{{.*}} no_inline optnone {{.*}} @_Z2s1ii(%{{.*}}, %{{.*}}) -> {{.*}} extra(#cir<extra({nothrow = #cir.nothrow, side_effect = 0 : i32})>)
// CIR: cir.call @_Z2s0ii(%{{.*}}, %{{.*}}) nothrow : ({{.*}}, {{.*}}) -> {{.*}}
// CIR: cir.func {{.*}} optnone {{.*}} @_Z2s2ii(%{{.*}}, %{{.*}}) -> {{.*}} extra(#cir<extra({nothrow = #cir.nothrow, side_effect = 0 : i32})>)
// CIR-NOT: cir.call @_Z2s1ii{{.*}}nothrow
// CIR: cir.call @_Z2s1ii(%{{.*}}, %{{.*}}) : ({{.*}}, {{.*}}) -> {{.*}}

// LLVM: define dso_local i32 @_Z2s0ii(i32 %0, i32 %1) #[[#ATTR1:]]
// LLVM: define dso_local i32 @_Z2s1ii(i32 %0, i32 %1) #[[#ATTR1:]]
// LLVM: define dso_local i32 @_Z2s2ii(i32 %0, i32 %1) #[[#ATTR1:]]

// LLVM: attributes #[[#ATTR1]] = {{.*}} noinline nounwind optnone
