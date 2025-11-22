// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Test basic ext_vector_type with bool elements
typedef bool bool4 __attribute__((ext_vector_type(4)));

// CHECK-LABEL: cir.func {{.*}}@_Z10test_basicv
void test_basic() {
  // CHECK: %[[ALLOCA:.*]] = cir.alloca !u8i, !cir.ptr<!u8i>, ["v"
  bool4 v = {true, false, true, false};
  // CHECK: %[[CONST:.*]] = cir.const #cir.int<5> : !u8i
  // CHECK: cir.store {{.*}} %[[CONST]], %[[ALLOCA]]
}

// CHECK-LABEL: cir.func {{.*}}@_Z14test_subscriptv
void test_subscript() {
  bool4 v = {true, false, true, false};
  // CHECK: %[[LOAD:.*]] = cir.load{{.*}}!u8i
  // CHECK: %[[IDX:.*]] = cir.const #cir.int<2>
  // CHECK: %[[SHIFT:.*]] = cir.shift(right, %[[LOAD]]{{.*}}, %[[IDX]]
  // CHECK: cir.binop(and,{{.*}}){{.*}}!u8i
  // CHECK: cir.cmp(ne,{{.*}}){{.*}}!cir.bool
  bool b = v[2];
}

// CHECK-LABEL: cir.func {{.*}}@_Z8test_ops
void test_ops(bool4 a, bool4 b) {
  // CHECK: cir.binop(and,{{.*}}){{.*}}!u8i
  bool4 c = a & b;
}
