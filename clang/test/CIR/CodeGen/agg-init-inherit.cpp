// RUN: %clang_cc1 -std=c++17 -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct A1 {
  A1();
};

class B : public A1 {};

void f1() {
  B v{};
}

// CHECK: cir.func dso_local @_Z2f1v()
// CHECK:     %0 = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["v", init]
// CHECK:     %1 = cir.base_class_addr %0 : !cir.ptr<!rec_B> nonnull [0] -> !cir.ptr<!rec_A1>
// CHECK:     cir.call @_ZN2A1C2Ev(%1) : (!cir.ptr<!rec_A1>) -> ()
// CHECK:     cir.return

struct A2 {
    A2();
};
class C : public A1, public A2 {};

void f2() {
  C v{};
}

// CHECK: cir.func dso_local @_Z2f2v()
// CHECK:     %0 = cir.alloca !rec_C, !cir.ptr<!rec_C>, ["v", init]
// CHECK:     %1 = cir.base_class_addr %0 : !cir.ptr<!rec_C> nonnull [0] -> !cir.ptr<!rec_A1>
// CHECK:     cir.call @_ZN2A1C2Ev(%1) : (!cir.ptr<!rec_A1>) -> ()
// CHECK:     %2 = cir.base_class_addr %0 : !cir.ptr<!rec_C> nonnull [0] -> !cir.ptr<!rec_A2>
// CHECK:     cir.call @_ZN2A2C2Ev(%2) : (!cir.ptr<!rec_A2>) -> ()
// CHECK:     cir.return

struct A3 {
    A3();
    ~A3();
};
class D : public A3 {};

void f3() {
  D v{};
}

// CHECK: cir.func dso_local @_Z2f3v()
// CHECK:     %0 = cir.alloca !rec_D, !cir.ptr<!rec_D>, ["v", init]
// CHECK:     %1 = cir.base_class_addr %0 : !cir.ptr<!rec_D> nonnull [0] -> !cir.ptr<!rec_A3>
// CHECK:     cir.call @_ZN2A3C2Ev(%1) : (!cir.ptr<!rec_A3>) -> ()
// CHECK:     cir.call @_ZN1DD2Ev(%0) : (!cir.ptr<!rec_D>) -> ()
// CHECK:     cir.return
