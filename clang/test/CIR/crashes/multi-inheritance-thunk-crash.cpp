// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG
//
// Test that CIR correctly generates thunks for multiple inheritance.
//
// Multiple inheritance requires generating thunks to adjust the 'this' pointer
// when calling virtual functions through a base class pointer that is not the
// primary base. The thunk adjusts 'this' and then calls the actual implementation.

struct A {
    virtual ~A() {}
    virtual int foo() { return 1; }
    int a;
};

struct B {
    virtual ~B() {}
    virtual int bar() { return 2; }
    int b;
};

struct C : A, B {
    int foo() override { return 3; }
    int bar() override { return 4; }
};

C* make_c() {
    return new C();
}

// CIR: cir.func comdat linkonce_odr @_ZThn16_N1C3barEv
// CIR:   cir.ptr_stride
// CIR:   cir.call @_ZN1C3barEv
// CIR:   cir.return

// LLVM: define {{.*}} @_Z6make_cv()
// LLVM: define {{.*}} @_ZThn16_N1C3barEv(ptr %0)
// LLVM:   %[[#ADJUSTED:]] = getelementptr i8, ptr %{{.*}}, i64 -16
// LLVM:   %[[#]] = call i32 @_ZN1C3barEv(ptr %[[#ADJUSTED]])
// LLVM:   ret i32

// OGCG: define {{.*}} @_Z6make_cv()
// OGCG: define {{.*}} @_ZThn16_N1C3barEv(ptr{{.*}} %this)
// OGCG:   getelementptr inbounds i8, ptr %{{.*}}, i64 -16
// OGCG:   tail call {{.*}} @_ZN1C3barEv(
// OGCG:   ret i32
