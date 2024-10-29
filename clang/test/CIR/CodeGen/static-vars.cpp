// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -fclangir -emit-cir %s -o %t1.cir
// RUN: FileCheck --check-prefix=CIR-AARCH64 --input-file=%t1.cir %s 
// RUN: %clang_cc1 -triple aarch64-none-linux-android24 -fclangir -emit-llvm %s -o %t1.ll
// RUN: FileCheck --check-prefix=LLVM-AARCH64 --input-file=%t1.ll %s 

void func1(void) {
  // Should lower default-initialized static vars.
  static int i;
  // CHECK-DAG: cir.global "private" internal dsolocal @_ZZ5func1vE1i = #cir.int<0> : !s32i

  // Should lower constant-initialized static vars.
  static int j = 1;
  // CHECK-DAG: cir.global "private" internal dsolocal @_ZZ5func1vE1j = #cir.int<1> : !s32i

  // Should properly shadow static vars in nested scopes.
  {
    static int j = 2;
    // CHECK-DAG: cir.global "private" internal dsolocal @_ZZ5func1vE1j_0 = #cir.int<2> : !s32i
  }
  {
    static int j = 3;
    // CHECK-DAG: cir.global "private" internal dsolocal @_ZZ5func1vE1j_1 = #cir.int<3> : !s32i
  }

  // Should lower basic static vars arithmetics.
  j++;
  // CHECK-DAG: %[[#V2:]] = cir.get_global @_ZZ5func1vE1j : !cir.ptr<!s32i>
  // CHECK-DAG: %[[#V3:]] = cir.load %[[#V2]] : !cir.ptr<!s32i>, !s32i
  // CHECK-DAG: %[[#V4:]] = cir.unary(inc, %[[#V3]]) : !s32i, !s32i
  // CHECK-DAG: cir.store %[[#V4]], %[[#V2]] : !s32i, !cir.ptr<!s32i>
}

// Should shadow static vars on different functions.
void func2(void) {
  static char i;
  // CHECK-DAG: cir.global "private" internal dsolocal @_ZZ5func2vE1i = #cir.int<0> : !s8i
  static float j;
  // CHECK-DAG: cir.global "private" internal dsolocal @_ZZ5func2vE1j = #cir.fp<0.000000e+00> : !cir.float
}

class b {
public:
  // CIR-AARCH64-DAG: cir.global linkonce_odr comdat @_ZZN1b4testEvE1c = #cir.int<0> : !s32i

  // LLVM-AARCH64-DAG: $_ZZN1b4testEvE1c = comdat any
  // LLVM-AARCH64-DAG: @_ZZN1b4testEvE1c = linkonce_odr global i32 0, comdat, align 4
  void test() { static int c; }
  // CIR-AARCH64-LABEL: @_ZN1b4testEv
  // CIR-AARCH64: {{%.*}} = cir.get_global @_ZZN1b4testEvE1c : !cir.ptr<!s32i>
};

void foo(void) { b().test(); }
