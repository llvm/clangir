// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef __typeof(sizeof(int)) size_t;

namespace test1 {
  struct A { void operator delete(void*,size_t); int x; };
  void a(A *x) {
    delete x;
  }
  // CHECK: cir.func dso_local @_ZN5test11aEPNS_1AE

  // CHECK: %[[CONST:.*]] = cir.const #cir.int<4> : !u64i
  // CHECK: cir.call @_ZN5test11AdlEPvm({{.*}}, %[[CONST]])
}

namespace test2 {
  struct A {
    ~A() {}
  };
  struct B {
    A *a;
    ~B();
  };
    // CHECK-LABEL: cir.func{{.*}} @_ZN5test21BD2Ev
    // CHECK:         cir.call @_ZN5test21AD2Ev
    // CHECK:         cir.call @_ZdlPvm
    // CHECK:         cir.return
    B::~B() { delete a; }
}

namespace test3 {
  struct X {
    virtual ~X();
  };

// Calling delete with a virtual destructor.
// CHECK-LABEL:   cir.func dso_local @_ZN5test37destroyEPNS_1XE
// CHECK:           %[[ARG_VAR:.*]] = cir.alloca !cir.ptr<!rec_test33A3AX>
// CHECK:           %[[ARG:.*]] = cir.load{{.*}} %[[ARG_VAR]] : !cir.ptr<!cir.ptr<!rec_test33A3AX>>, !cir.ptr<!rec_test33A3AX>
// CHECK:           %[[ARG_PTR:.*]] = cir.cast(bitcast, %[[ARG]]
// CHECK:           %[[VTABLE:.*]] = cir.load{{.*}} %[[ARG_PTR]]
// CHECK:           %[[DTOR_PTR:.*]] = cir.vtable.address_point( %[[VTABLE]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_test33A3AX>)>>>, address_point = <index = 0, offset = 1>)
// CHECK:           %[[DTOR_FUN:.*]] = cir.load{{.*}} %[[DTOR_PTR]]
// CHECK:           cir.call %[[DTOR_FUN]](%[[ARG]])
// CHECK:           cir.return
  void destroy(X *x) {
    delete x;
  }
}
