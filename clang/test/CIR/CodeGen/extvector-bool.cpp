// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll -DDISABLE_ELEMENT_ASSIGN_TEST
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG
//
// NOTE: Element assignment test (test_element_assign) is excluded from OGCG
// testing due to a bug in classic CodeGen (clang/lib/CodeGen/CGExpr.cpp:2585-2587).
// Classic CodeGen calls VecTy->getScalarType() on an IntegerType before bitcasting
// to VectorType for ExtVectorBoolType, causing assertion failure. CIR correctly
// performs the bitcast first. This is a justifiable divergence fixing a bug.
// Bug verified: classic CodeGen crashes with assertion when compiling element assignment.
//
// The OGCG tests below verify that CIR's LLVM lowering for comparisons and logical
// NOT matches classic CodeGen's output, demonstrating consistency where classic
// CodeGen works correctly.

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

#ifndef DISABLE_ELEMENT_ASSIGN_TEST
// Test element assignment (v[2] = true)
// NOTE: This test is disabled for classic CodeGen due to a bug in CGExpr.cpp:2585-2587
// where VecTy->getScalarType() is called on an integer type before bitcasting to vector.
// CIR-LABEL: cir.func {{.*}}@_Z{{.*}}test_element_assignv
void test_element_assign() {
  bool4 v = {true, false, true, false};
  // CIR: cir.load{{.*}}!u8i
  // CIR: cir.cast bitcast{{.*}}!u8i -> !cir.vector<!cir.int<u, 1> x 8>
  // CIR: cir.cast bool_to_int{{.*}}!cir.bool -> !cir.int<u, 1>
  // CIR: cir.vec.insert
  // CIR: cir.cast bitcast{{.*}}!cir.vector<!cir.int<u, 1> x 8> -> !u8i
  // CIR: cir.store{{.*}}!u8i, !cir.ptr<!u8i>
  v[2] = true;

  // LLVM-LABEL: define {{.*}}@_Z{{.*}}test_element_assignv
  // LLVM: %[[VEC_LOAD:.*]] = load i8
  // LLVM: %[[VEC_BITCAST:.*]] = bitcast i8 %[[VEC_LOAD]] to <8 x i1>
  // LLVM: %[[VEC_INSERT:.*]] = insertelement <8 x i1> %[[VEC_BITCAST]], i1 true, i32 2
  // LLVM: %[[VEC_BITCAST_BACK:.*]] = bitcast <8 x i1> %[[VEC_INSERT]] to i8
  // LLVM: store i8 %[[VEC_BITCAST_BACK]]
}
#endif

// Test comparison operations (a == b, a != b)
// CIR-LABEL: cir.func {{.*}}@_Z{{.*}}test_comparisonv
void test_comparison() {
  bool4 a = {true, false, true, false};
  bool4 b = {false, true, true, false};

  // Test equality
  // CIR: cir.load{{.*}}!u8i
  // CIR: cir.load{{.*}}!u8i
  // CIR: cir.cast bitcast{{.*}}!u8i -> !cir.vector<!cir.int<u, 1> x 8>
  // CIR: cir.cast bitcast{{.*}}!u8i -> !cir.vector<!cir.int<u, 1> x 8>
  // CIR: cir.vec.shuffle{{.*}}!cir.vector<!cir.int<u, 1> x 8>{{.*}}!cir.vector<!cir.int<u, 1> x 4>
  // CIR: cir.vec.shuffle{{.*}}!cir.vector<!cir.int<u, 1> x 8>{{.*}}!cir.vector<!cir.int<u, 1> x 4>
  // CIR: cir.vec.cmp(eq,{{.*}}!cir.vector<!cir.int<u, 1> x 4>
  // CIR: cir.vec.shuffle{{.*}}!cir.vector<!cir.int<u, 1> x 4>{{.*}}!cir.vector<!cir.int<u, 1> x 8>
  // CIR: cir.cast bitcast{{.*}}!cir.vector<!cir.int<u, 1> x 8> -> !u8i
  bool4 c = a == b;

  // Test inequality
  // CIR: cir.load{{.*}}!u8i
  // CIR: cir.load{{.*}}!u8i
  // CIR: cir.cast bitcast{{.*}}!u8i -> !cir.vector<!cir.int<u, 1> x 8>
  // CIR: cir.cast bitcast{{.*}}!u8i -> !cir.vector<!cir.int<u, 1> x 8>
  // CIR: cir.vec.shuffle{{.*}}!cir.vector<!cir.int<u, 1> x 8>{{.*}}!cir.vector<!cir.int<u, 1> x 4>
  // CIR: cir.vec.shuffle{{.*}}!cir.vector<!cir.int<u, 1> x 8>{{.*}}!cir.vector<!cir.int<u, 1> x 4>
  // CIR: cir.vec.cmp(ne,{{.*}}!cir.vector<!cir.int<u, 1> x 4>
  // CIR: cir.vec.shuffle{{.*}}!cir.vector<!cir.int<u, 1> x 4>{{.*}}!cir.vector<!cir.int<u, 1> x 8>
  // CIR: cir.cast bitcast{{.*}}!cir.vector<!cir.int<u, 1> x 8> -> !u8i
  bool4 d = a != b;

  // LLVM-LABEL: define {{.*}}@_Z{{.*}}test_comparisonv
  // LLVM: %[[A_LOAD:.*]] = load i8
  // LLVM: %[[B_LOAD:.*]] = load i8
  // LLVM: %[[A_BITCAST:.*]] = bitcast i8 %[[A_LOAD]] to <8 x i1>
  // LLVM: %[[B_BITCAST:.*]] = bitcast i8 %[[B_LOAD]] to <8 x i1>
  // LLVM: %[[A_EXTRACT:.*]] = shufflevector <8 x i1> %[[A_BITCAST]], <8 x i1> %[[A_BITCAST]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: %[[B_EXTRACT:.*]] = shufflevector <8 x i1> %[[B_BITCAST]], <8 x i1> %[[B_BITCAST]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: %[[VEC_CMP:.*]] = icmp eq <4 x i1> %[[A_EXTRACT]], %[[B_EXTRACT]]
  // LLVM: %[[RESULT_PAD:.*]] = shufflevector <4 x i1> %[[VEC_CMP]], <4 x i1> %[[VEC_CMP]]
  // LLVM: %[[RESULT_BITCAST:.*]] = bitcast <8 x i1> %[[RESULT_PAD]] to i8
  // LLVM: store i8 %[[RESULT_BITCAST]]
  // LLVM: %[[A_LOAD2:.*]] = load i8
  // LLVM: %[[B_LOAD2:.*]] = load i8
  // LLVM: %[[A_BITCAST2:.*]] = bitcast i8 %[[A_LOAD2]] to <8 x i1>
  // LLVM: %[[B_BITCAST2:.*]] = bitcast i8 %[[B_LOAD2]] to <8 x i1>
  // LLVM: %[[A_EXTRACT2:.*]] = shufflevector <8 x i1> %[[A_BITCAST2]], <8 x i1> %[[A_BITCAST2]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: %[[B_EXTRACT2:.*]] = shufflevector <8 x i1> %[[B_BITCAST2]], <8 x i1> %[[B_BITCAST2]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: %[[VEC_CMP2:.*]] = icmp ne <4 x i1> %[[A_EXTRACT2]], %[[B_EXTRACT2]]
  // LLVM: %[[RESULT_PAD2:.*]] = shufflevector <4 x i1> %[[VEC_CMP2]], <4 x i1> %[[VEC_CMP2]]
  // LLVM: %[[RESULT_BITCAST2:.*]] = bitcast <8 x i1> %[[RESULT_PAD2]] to i8
  // LLVM: store i8 %[[RESULT_BITCAST2]]

  // OGCG-LABEL: define {{.*}}@_Z{{.*}}test_comparisonv
  // OGCG: bitcast i8 {{.*}} to <8 x i1>
  // OGCG: shufflevector <8 x i1> {{.*}}, <8 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: bitcast i8 {{.*}} to <8 x i1>
  // OGCG: shufflevector <8 x i1> {{.*}}, <8 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: icmp eq <4 x i1>
  // OGCG: shufflevector <4 x i1> {{.*}}, <4 x i1> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  // OGCG: bitcast <8 x i1> {{.*}} to i8
  // OGCG: icmp ne <4 x i1>
}

// Test logical NOT (!v)
// CIR-LABEL: cir.func {{.*}}@_Z{{.*}}test_logical_notv
void test_logical_not() {
  bool4 v = {true, false, true, false};

  // CIR: cir.load{{.*}}!u8i
  // CIR: cir.cast bitcast{{.*}}!u8i -> !cir.vector<!cir.int<u, 1> x 8>
  // CIR: cir.vec.shuffle{{.*}}!cir.vector<!cir.int<u, 1> x 8>{{.*}}!cir.vector<!cir.int<u, 1> x 4>
  // CIR: cir.const #cir.zero : !cir.vector<!cir.int<u, 1> x 4>
  // CIR: cir.vec.cmp(eq,{{.*}}!cir.vector<!cir.int<u, 1> x 4>
  // CIR: cir.vec.shuffle{{.*}}!cir.vector<!cir.int<u, 1> x 4>{{.*}}!cir.vector<!cir.int<u, 1> x 8>
  // CIR: cir.cast bitcast{{.*}}!cir.vector<!cir.int<u, 1> x 8> -> !u8i
  bool4 n = !v;

  // LLVM-LABEL: define {{.*}}@_Z{{.*}}test_logical_notv
  // LLVM: %[[VEC_LOAD:.*]] = load i8
  // LLVM: %[[VEC_BITCAST:.*]] = bitcast i8 %[[VEC_LOAD]] to <8 x i1>
  // LLVM: %[[VEC_EXTRACT:.*]] = shufflevector <8 x i1> %[[VEC_BITCAST]], <8 x i1> %[[VEC_BITCAST]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: icmp eq <4 x i1> %[[VEC_EXTRACT]], zeroinitializer
  // LLVM: shufflevector <4 x i1>
  // LLVM: bitcast <8 x i1> %{{.*}} to i8
  // LLVM: store i8 %{{.*}}

  // OGCG-LABEL: define {{.*}}@_Z{{.*}}test_logical_notv
  // OGCG: bitcast i8 {{.*}} to <8 x i1>
  // OGCG: shufflevector <8 x i1> {{.*}}, <8 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: icmp eq <4 x i1> {{.*}}, zeroinitializer
  // OGCG: shufflevector <4 x i1> {{.*}}, <4 x i1> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  // OGCG: bitcast <8 x i1> {{.*}} to i8
}
