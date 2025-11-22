// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Test basic ext_vector_type with bool elements
typedef bool bool4 __attribute__((ext_vector_type(4)));
typedef bool bool2 __attribute__((ext_vector_type(2)));
typedef bool bool16 __attribute__((ext_vector_type(16)));

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

// NOTE: The following operations are not yet fully implemented for
// ExtVectorBoolType and require special handling:
// - Element assignment (v[2] = true): Requires bit manipulation to set/clear individual bits
// - Unary logical NOT (!v): May require special handling beyond bitwise NOT

// Test bitwise operations
// CHECK-LABEL: cir.func {{.*}}@_Z16test_bitwise_opsv
void test_bitwise_ops() {
  bool4 a = {true, false, true, false};
  bool4 b = {false, true, true, false};

  // Bitwise AND
  // CHECK: cir.binop(and,{{.*}}){{.*}}!u8i
  bool4 c = a & b;

  // Bitwise OR
  // CHECK: cir.binop(or,{{.*}}){{.*}}!u8i
  bool4 d = a | b;

  // Bitwise XOR
  // CHECK: cir.binop(xor,{{.*}}){{.*}}!u8i
  bool4 e = a ^ b;
}

// Test different vector sizes
// CHECK-LABEL: cir.func {{.*}}@_Z17test_vector_sizesv
void test_vector_sizes() {
  // bool2 uses u8i (padded to 8 bits minimum)
  // CHECK: cir.alloca !u8i, !cir.ptr<!u8i>, ["v2"
  bool2 v2 = {true, false};
  // CHECK-DAG: cir.const #cir.int<1> : !u8i
  // CHECK-DAG: cir.store{{.*}}!u8i, !cir.ptr<!u8i>

  // bool16 uses u16i
  // CHECK-DAG: cir.alloca !u16i, !cir.ptr<!u16i>, ["v16"
  bool16 v16 = {true, false, true, false, true, false, true, false,
                false, true, false, true, false, true, false, true};
  // CHECK-DAG: cir.const #cir.int<43605> : !u16i
  // CHECK-DAG: cir.store{{.*}}!u16i, !cir.ptr<!u16i>
}

// Test function parameters and returns
// CHECK-LABEL: cir.func {{.*}}@_Z12invert_bool4
// CHECK-SAME: %arg0: !u8i
// CHECK-SAME: -> !u8i
bool4 invert_bool4(bool4 v) {
  // Bitwise NOT
  // CHECK: %[[LOAD:.*]] = cir.load{{.*}}!u8i
  // CHECK: cir.unary(not, %[[LOAD]]){{.*}}!u8i
  return ~v;
}

// Test all bits set and all bits clear
// CHECK-LABEL: cir.func {{.*}}@_Z15test_edge_casesv
void test_edge_cases() {
  // All false (0)
  // CHECK-DAG: cir.alloca !u8i, !cir.ptr<!u8i>, ["all_false"
  bool4 all_false = {false, false, false, false};
  // CHECK-DAG: cir.const #cir.int<0> : !u8i
  // CHECK-DAG: cir.store{{.*}}!u8i, !cir.ptr<!u8i>

  // All true (15 = 0b1111 for 4 bits)
  // CHECK-DAG: cir.alloca !u8i, !cir.ptr<!u8i>, ["all_true"
  bool4 all_true = {true, true, true, true};
  // CHECK-DAG: cir.const #cir.int<15> : !u8i
  // CHECK-DAG: cir.store{{.*}}!u8i, !cir.ptr<!u8i>
}

// Test subscript with variable index
// CHECK-LABEL: cir.func {{.*}}@_Z23test_variable_subscript
void test_variable_subscript(int idx) {
  bool4 v = {true, false, true, false};
  // CHECK: cir.load{{.*}}!u8i
  // CHECK: cir.load{{.*}}!s32i
  // CHECK: cir.shift(right,{{.*}}){{.*}}!u8i
  // CHECK: cir.binop(and,{{.*}}){{.*}}!u8i
  bool b = v[idx];
}

// Test initialization with all same value
// CHECK-LABEL: cir.func {{.*}}@_Z18test_same_init_valv
void test_same_init_val() {
  // Initialize all elements to true using splat
  // CHECK: cir.alloca !u8i, !cir.ptr<!u8i>, ["v"
  bool4 v = {true, true, true, true};
  // CHECK: cir.const #cir.int<15> : !u8i
  // CHECK: cir.store{{.*}}!u8i, !cir.ptr<!u8i>
}

// Test multiple operations in sequence
// CHECK-LABEL: cir.func {{.*}}@_Z17test_multiple_opsv
void test_multiple_ops() {
  bool4 a = {true, false, true, false};
  bool4 b = {false, true, true, false};

  // CHECK: cir.binop(and,{{.*}}){{.*}}!u8i
  bool4 c = a & b;
  // CHECK: cir.binop(or,{{.*}}){{.*}}!u8i
  bool4 d = c | b;
  // CHECK: cir.unary(not,{{.*}}){{.*}}!u8i
  bool4 e = ~d;
}

// Test reading specific elements
// CHECK-LABEL: cir.func {{.*}}@_Z18test_read_elementsv
void test_read_elements() {
  bool4 v = {true, false, true, false};

  // Read element 0
  // CHECK: cir.load{{.*}}!u8i
  // CHECK: cir.const #cir.int<0>
  // CHECK: cir.shift(right,{{.*}}){{.*}}!u8i
  // CHECK: cir.binop(and,{{.*}}){{.*}}!u8i
  bool e0 = v[0];

  // Read element 3
  // CHECK: cir.load{{.*}}!u8i
  // CHECK: cir.const #cir.int<3>
  // CHECK: cir.shift(right,{{.*}}){{.*}}!u8i
  // CHECK: cir.binop(and,{{.*}}){{.*}}!u8i
  bool e3 = v[3];
}
