// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct S {
  S();
  ~S();
};

S create();

// CHECK: cir.func dso_local @_Z1fv()
// CHECK:   %[[TMP:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["ref.tmp0"]
// CHECK:   %[[REF:.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["s", init, const]
// CHECK:   cir.scope {
// CHECK:     %[[VAL:.*]] = cir.call @_Z6createv() : () -> !rec_S
// CHECK:     cir.store align({{.*}}) %[[VAL]], %[[TMP]]
// CHECK:   }
// CHECK:   cir.store align({{.*}}) %[[TMP]], %[[REF]]
// CHECK:   cir.call @_ZN1SD1Ev(%[[TMP]]) : (!cir.ptr<!rec_S>) -> ()
// CHECK:   cir.return
void f() {
  auto&& s = create();
}
