// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

class A {
public:
  void Foo() {}
};

// Statement expression result must be returned by value.
// The local var a should be copied into a temporary and this temporary should
// be used to call the Foo method.
void test1() {
  A a;
  ({a;}).Foo();
}
// CHECK: @_Z5test1v
// CHECK: %0 = cir.alloca !ty_22A22, cir.ptr <!ty_22A22>, ["a"]
// CHECK: cir.scope {
// CHECK:   %1 = cir.alloca !ty_22A22, cir.ptr <!ty_22A22>, ["ref.tmp0"]
// CHECK:   cir.scope {
// CHECK:     cir.call @_ZN1AC1ERKS_(%1, %0) : (!cir.ptr<!ty_22A22>, !cir.ptr<!ty_22A22>) -> ()
// CHECK:   }
// CHECK:   cir.call @_ZN1A3FooEv(%1) : (!cir.ptr<!ty_22A22>) -> ()
// CHECK: }
