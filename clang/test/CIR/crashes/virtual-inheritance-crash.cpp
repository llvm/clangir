// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG
//
// Test that CIR correctly handles virtual inheritance with thunks.
//
// Virtual inheritance requires:
// - VTT (Virtual Table Table) for construction
// - Virtual base pointer adjustments in thunks
// - vtable offset lookups for dynamic adjustment
//
// This test verifies that CIR can compile virtual inheritance hierarchies
// without crashing during thunk generation.

struct Base {
    virtual ~Base() {}
    int b;
};

struct A : virtual Base {
    int a;
};

struct B : virtual Base {
    int b;
};

struct C : A, B {
    int c;
};

C* make_c() {
    return new C();
}

// LLVM-DAG: @_ZTV1C = {{.*}} constant
// LLVM-DAG: @_ZTT1C = {{.*}} constant
// LLVM-DAG: define {{.*}} @_Z6make_cv()

// OGCG-DAG: @_ZTV1C = {{.*}} constant
// OGCG-DAG: @_ZTT1C = {{.*}} constant
// OGCG-DAG: define {{.*}} @_Z6make_cv()
