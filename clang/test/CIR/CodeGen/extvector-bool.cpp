// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG

// Test basic ext_vector_type with bool elements
typedef bool bool4 __attribute__((ext_vector_type(4)));
typedef bool bool2 __attribute__((ext_vector_type(2)));
typedef bool bool16 __attribute__((ext_vector_type(16)));

// CIR-LABEL: cir.func {{.*}}@_Z10test_basicv
void test_basic() {
  // CIR: %[[ALLOCA:.*]] = cir.alloca !u8i, !cir.ptr<!u8i>, ["v"
  bool4 v = {true, false, true, false};
  // CIR: %[[CONST:.*]] = cir.const #cir.int<5> : !u8i
  // CIR: cir.store {{.*}} %[[CONST]], %[[ALLOCA]]

  // LLVM-LABEL: define {{.*}}@_Z10test_basicv
  // LLVM: alloca i8
  // LLVM: store i8 5

  // OGCG-LABEL: define {{.*}}@_Z10test_basicv
  // OGCG: alloca i8
  // OGCG: store i8
}

// CIR-LABEL: cir.func {{.*}}@_Z14test_subscriptv
void test_subscript() {
  bool4 v = {true, false, true, false};
  // CIR: %[[LOAD:.*]] = cir.load{{.*}}!u8i
  // CIR: %[[IDX:.*]] = cir.const #cir.int<2>
  // CIR: %[[SHIFT:.*]] = cir.shift(right, %[[LOAD]]{{.*}}, %[[IDX]]
  // CIR: cir.binop(and,{{.*}}){{.*}}!u8i
  // CIR: cir.cmp(ne,{{.*}}){{.*}}!cir.bool
  bool b = v[2];

  // LLVM-LABEL: define {{.*}}@_Z14test_subscriptv
  // LLVM: lshr i8
  // LLVM: and i8
  // LLVM: icmp ne i8

  // OGCG-LABEL: define {{.*}}@_Z14test_subscriptv
  // OGCG: extractelement
  // OGCG: zext i1
}

// CIR-LABEL: cir.func {{.*}}@_Z8test_ops
void test_ops(bool4 a, bool4 b) {
  // CIR: cir.binop(and,{{.*}}){{.*}}!u8i
  bool4 c = a & b;

  // LLVM-LABEL: define {{.*}}@_Z8test_opsDv4_bS_
  // LLVM: and i8

  // OGCG-LABEL: define {{.*}}@_Z8test_opsDv4_bS_
  // OGCG: shufflevector
  // OGCG: and <4 x i1>
}

// Test bitwise operations
// CIR-LABEL: cir.func {{.*}}@_Z16test_bitwise_opsv
void test_bitwise_ops() {
  bool4 a = {true, false, true, false};
  bool4 b = {false, true, true, false};

  // Bitwise AND
  // CIR: cir.binop(and,{{.*}}){{.*}}!u8i
  bool4 c = a & b;

  // Bitwise OR
  // CIR: cir.binop(or,{{.*}}){{.*}}!u8i
  bool4 d = a | b;

  // Bitwise XOR
  // CIR: cir.binop(xor,{{.*}}){{.*}}!u8i
  bool4 e = a ^ b;

  // LLVM-LABEL: define {{.*}}@_Z16test_bitwise_opsv
  // LLVM: and i8
  // LLVM: or i8
  // LLVM: xor i8

  // OGCG-LABEL: define {{.*}}@_Z16test_bitwise_opsv
  // OGCG: and <4 x i1>
  // OGCG: or <4 x i1>
  // OGCG: xor <4 x i1>
}

// Test different vector sizes
// CIR-LABEL: cir.func {{.*}}@_Z17test_vector_sizesv
void test_vector_sizes() {
  // bool2 uses u8i (padded to 8 bits minimum)
  // CIR: cir.alloca !u8i, !cir.ptr<!u8i>, ["v2"
  bool2 v2 = {true, false};
  // CIR-DAG: cir.const #cir.int<1> : !u8i
  // CIR-DAG: cir.store{{.*}}!u8i, !cir.ptr<!u8i>

  // bool16 uses u16i
  // CIR-DAG: cir.alloca !u16i, !cir.ptr<!u16i>, ["v16"
  bool16 v16 = {true, false, true, false, true, false, true, false,
                false, true, false, true, false, true, false, true};
  // CIR-DAG: cir.const #cir.int<43605> : !u16i
  // CIR-DAG: cir.store{{.*}}!u16i, !cir.ptr<!u16i>

  // LLVM-LABEL: define {{.*}}@_Z17test_vector_sizesv
  // LLVM-DAG: alloca i8
  // LLVM-DAG: store i8 1
  // LLVM-DAG: alloca i16
  // LLVM-DAG: store i16

  // OGCG-LABEL: define {{.*}}@_Z17test_vector_sizesv
  // OGCG-DAG: alloca i8
  // OGCG-DAG: store i8{{.*}}, ptr %
  // OGCG-DAG: alloca i16
  // OGCG-DAG: store i16
}

// Test function parameters and returns
// CIR-LABEL: cir.func {{.*}}@_Z12invert_bool4
// CIR-SAME: %arg0: !u8i
// CIR-SAME: -> !u8i
bool4 invert_bool4(bool4 v) {
  // Bitwise NOT
  // CIR: %[[LOAD:.*]] = cir.load{{.*}}!u8i
  // CIR: cir.unary(not, %[[LOAD]]){{.*}}!u8i
  return ~v;

  // LLVM-LABEL: define {{.*}}@_Z12invert_bool4Dv4_b
  // LLVM: xor i8

  // OGCG-LABEL: define {{.*}}@_Z12invert_bool4Dv4_b
  // OGCG: xor <4 x i1>
}

// Test all bits set and all bits clear
// CIR-LABEL: cir.func {{.*}}@_Z15test_edge_casesv
void test_edge_cases() {
  // All false (0)
  // CIR-DAG: cir.alloca !u8i, !cir.ptr<!u8i>, ["all_false"
  bool4 all_false = {false, false, false, false};
  // CIR-DAG: cir.const #cir.int<0> : !u8i
  // CIR-DAG: cir.store{{.*}}!u8i, !cir.ptr<!u8i>

  // All true (15 = 0b1111 for 4 bits)
  // CIR-DAG: cir.alloca !u8i, !cir.ptr<!u8i>, ["all_true"
  bool4 all_true = {true, true, true, true};
  // CIR-DAG: cir.const #cir.int<15> : !u8i
  // CIR-DAG: cir.store{{.*}}!u8i, !cir.ptr<!u8i>
}

// Test subscript with variable index
// CIR-LABEL: cir.func {{.*}}@_Z23test_variable_subscript
void test_variable_subscript(int idx) {
  bool4 v = {true, false, true, false};
  // CIR: cir.load{{.*}}!u8i
  // CIR: cir.load{{.*}}!s32i
  // CIR: cir.shift(right,{{.*}}){{.*}}!u8i
  // CIR: cir.binop(and,{{.*}}){{.*}}!u8i
  bool b = v[idx];
}

// Test initialization with all same value
// CIR-LABEL: cir.func {{.*}}@_Z18test_same_init_valv
void test_same_init_val() {
  // Initialize all elements to true using splat
  // CIR: cir.alloca !u8i, !cir.ptr<!u8i>, ["v"
  bool4 v = {true, true, true, true};
  // CIR: cir.const #cir.int<15> : !u8i
  // CIR: cir.store{{.*}}!u8i, !cir.ptr<!u8i>
}

// Test multiple operations in sequence
// CIR-LABEL: cir.func {{.*}}@_Z17test_multiple_opsv
void test_multiple_ops() {
  bool4 a = {true, false, true, false};
  bool4 b = {false, true, true, false};

  // CIR: cir.binop(and,{{.*}}){{.*}}!u8i
  bool4 c = a & b;
  // CIR: cir.binop(or,{{.*}}){{.*}}!u8i
  bool4 d = c | b;
  // CIR: cir.unary(not,{{.*}}){{.*}}!u8i
  bool4 e = ~d;
}

// Test reading specific elements
// CIR-LABEL: cir.func {{.*}}@_Z18test_read_elementsv
void test_read_elements() {
  bool4 v = {true, false, true, false};

  // Read element 0
  // CIR: cir.load{{.*}}!u8i
  // CIR: cir.const #cir.int<0>
  // CIR: cir.shift(right,{{.*}}){{.*}}!u8i
  // CIR: cir.binop(and,{{.*}}){{.*}}!u8i
  bool e0 = v[0];

  // Read element 3
  // CIR: cir.load{{.*}}!u8i
  // CIR: cir.const #cir.int<3>
  // CIR: cir.shift(right,{{.*}}){{.*}}!u8i
  // CIR: cir.binop(and,{{.*}}){{.*}}!u8i
  bool e3 = v[3];
}
