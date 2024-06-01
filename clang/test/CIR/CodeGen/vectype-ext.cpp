// RUN: %clang_cc1 -fclangir -emit-cir -triple x86_64-unknown-linux-gnu %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -fclangir -S -emit-llvm -triple x86_64-unknown-linux-gnu %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

typedef int vi4 __attribute__((ext_vector_type(4)));
typedef int vi2 __attribute__((ext_vector_type(2)));
typedef double vd2 __attribute__((ext_vector_type(2)));
typedef long vl2 __attribute__((ext_vector_type(2)));
typedef unsigned short vus2 __attribute__((ext_vector_type(2)));

typedef int vi7 __attribute__((ext_vector_type(7)));

// CIR: cir.func {{@.*vector_int_test.*}}
// LLVM: define void {{@.*vector_int_test.*}}
void vector_int_test(int x) {

  // Vector constant. Not yet implemented. Expected results will change from
  // cir.vec.create to cir.const.
  vi4 a = { 1, 2, 3, 4 };
  // CIR: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} : !s32i, !s32i, !s32i, !s32i) : !cir.vector<!s32i x 4>

  // Non-const vector initialization.
  vi4 b = { x, 5, 6, x + 1 };
  // CIR: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} : !s32i, !s32i, !s32i, !s32i) : !cir.vector<!s32i x 4>

  // Incomplete vector initialization.
  vi4 bb = { x, x + 1 };
  // CIR: %[[#zero:]] = cir.const #cir.int<0> : !s32i
  // CIR: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}}, %[[#zero]], %[[#zero]] : !s32i, !s32i, !s32i, !s32i) : !cir.vector<!s32i x 4>

  // Scalar to vector conversion, a.k.a. vector splat.  Only valid as an
  // operand of a binary operator, not as a regular conversion.
  bb = a + 7;
  // CIR: %[[#seven:]] = cir.const #cir.int<7> : !s32i
  // CIR: %{{[0-9]+}} = cir.vec.splat %[[#seven]] : !s32i, !cir.vector<!s32i x 4>

  // Vector to vector conversion
  vd2 bbb = { };
  bb = (vi4)bbb;
  // CIR: %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.vector<!cir.double x 2>), !cir.vector<!s32i x 4>

  // Extract element
  int c = a[x];
  // CIR: %{{[0-9]+}} = cir.vec.extract %{{[0-9]+}}[%{{[0-9]+}} : !s32i] : !cir.vector<!s32i x 4>

  // Insert element
  a[x] = x;
  // CIR: %[[#LOADEDVI:]] = cir.load %[[#STORAGEVI:]] : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR: %[[#UPDATEDVI:]] = cir.vec.insert %{{[0-9]+}}, %[[#LOADEDVI]][%{{[0-9]+}} : !s32i] : !cir.vector<!s32i x 4>
  // CIR: cir.store %[[#UPDATEDVI]], %[[#STORAGEVI]] : !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>

  // Compound assignment
  a[x] += a[0];
  // CIR: %[[#RHSCA:]] = cir.vec.extract %{{[0-9]+}}[%{{[0-9]+}} : !s32i] : !cir.vector<!s32i x 4>
  // CIR: %[[#LHSCA:]] = cir.vec.extract %{{[0-9]+}}[%{{[0-9]+}} : !s32i] : !cir.vector<!s32i x 4>
  // CIR: %[[#SUMCA:]] = cir.binop(add, %[[#LHSCA]], %[[#RHSCA]]) : !s32i
  // CIR: cir.vec.insert %[[#SUMCA]], %{{[0-9]+}}[%{{[0-9]+}} : !s32i] : !cir.vector<!s32i x 4>

  // Binary arithmetic operations
  vi4 d = a + b;
  // CIR: %{{[0-9]+}} = cir.binop(add, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 e = a - b;
  // CIR: %{{[0-9]+}} = cir.binop(sub, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 f = a * b;
  // CIR: %{{[0-9]+}} = cir.binop(mul, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 g = a / b;
  // CIR: %{{[0-9]+}} = cir.binop(div, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 h = a % b;
  // CIR: %{{[0-9]+}} = cir.binop(rem, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 i = a & b;
  // CIR: %{{[0-9]+}} = cir.binop(and, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 j = a | b;
  // CIR: %{{[0-9]+}} = cir.binop(or, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 k = a ^ b;
  // CIR: %{{[0-9]+}} = cir.binop(xor, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>

  // Unary arithmetic operations
  vi4 l = +a;
  // CIR: %{{[0-9]+}} = cir.unary(plus, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  vi4 m = -a;
  // CIR: %{{[0-9]+}} = cir.unary(minus, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  vi4 n = ~a;
  // CIR: %{{[0-9]+}} = cir.unary(not, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>

  // TODO: Ternary conditional operator

  // Comparisons
  vi4 o = a == b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(eq, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  vi4 p = a != b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(ne, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  vi4 q = a < b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(lt, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  vi4 r = a > b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(gt, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  vi4 s = a <= b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(le, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  vi4 t = a >= b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(ge, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>

  // __builtin_shufflevector
  vi4 u = __builtin_shufflevector(a, b, 7, 5, 3, 1);
  // CIR: %{{[0-9]+}} = cir.vec.shuffle(%{{[0-9]+}}, %{{[0-9]+}} : !cir.vector<!s32i x 4>) [#cir.int<7> : !s64i, #cir.int<5> : !s64i, #cir.int<3> : !s64i, #cir.int<1> : !s64i] : !cir.vector<!s32i x 4>
  vi4 v = __builtin_shufflevector(a, b);
  // CIR: %{{[0-9]+}} = cir.vec.shuffle.dynamic %{{[0-9]+}} : !cir.vector<!s32i x 4>, %{{[0-9]+}} : !cir.vector<!s32i x 4>
}

// CIR: cir.func {{@.*vector_double_test.*}}
// LLVM: define void {{@.*vector_double_test.*}}
void vector_double_test(int x, double y) {
  // Vector constant. Not yet implemented. Expected results will change from
  // cir.vec.create to cir.const.
  vd2 a = { 1.5, 2.5 };
  // CIR: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}} : !cir.double, !cir.double) : !cir.vector<!cir.double x 2>

  // Non-const vector initialization.
  vd2 b = { y, y + 1.0 };
  // CIR: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}} : !cir.double, !cir.double) : !cir.vector<!cir.double x 2>

  // Incomplete vector initialization
  vd2 bb = { y };
  // CIR: [[#dzero:]] = cir.const #cir.fp<0.000000e+00> : !cir.double
  // CIR: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %[[#dzero]] : !cir.double, !cir.double) : !cir.vector<!cir.double x 2>

  // Scalar to vector conversion, a.k.a. vector splat.  Only valid as an
  // operand of a binary operator, not as a regular conversion.
  bb = a + 2.5;
  // CIR: %[[#twohalf:]] = cir.const #cir.fp<2.500000e+00> : !cir.double
  // CIR: %{{[0-9]+}} = cir.vec.splat %[[#twohalf]] : !cir.double, !cir.vector<!cir.double x 2>

  // Extract element
  double c = a[x];
  // CIR: %{{[0-9]+}} = cir.vec.extract %{{[0-9]+}}[%{{[0-9]+}} : !s32i] : !cir.vector<!cir.double x 2>

  // Insert element
  a[x] = y;
  // CIR: %[[#LOADEDVF:]] = cir.load %[[#STORAGEVF:]] : !cir.ptr<!cir.vector<!cir.double x 2>>, !cir.vector<!cir.double x 2>
  // CIR: %[[#UPDATEDVF:]] = cir.vec.insert %{{[0-9]+}}, %[[#LOADEDVF]][%{{[0-9]+}} : !s32i] : !cir.vector<!cir.double x 2>
  // CIR: cir.store %[[#UPDATEDVF]], %[[#STORAGEVF]] : !cir.vector<!cir.double x 2>, !cir.ptr<!cir.vector<!cir.double x 2>>

  // Binary arithmetic operations
  vd2 d = a + b;
  // CIR: %{{[0-9]+}} = cir.binop(add, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>
  vd2 e = a - b;
  // CIR: %{{[0-9]+}} = cir.binop(sub, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>
  vd2 f = a * b;
  // CIR: %{{[0-9]+}} = cir.binop(mul, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>
  vd2 g = a / b;
  // CIR: %{{[0-9]+}} = cir.binop(div, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>

  // Unary arithmetic operations
  vd2 l = +a;
  // CIR: %{{[0-9]+}} = cir.unary(plus, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!cir.double x 2>
  vd2 m = -a;
  // CIR: %{{[0-9]+}} = cir.unary(minus, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!cir.double x 2>

  // Comparisons
  vl2 o = a == b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(eq, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>
  vl2 p = a != b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(ne, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>
  vl2 q = a < b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(lt, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>
  vl2 r = a > b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(gt, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>
  vl2 s = a <= b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(le, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>
  vl2 t = a >= b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(ge, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>

  // __builtin_convertvector
  vus2 w = __builtin_convertvector(a, vus2);
  // CIR: %{{[0-9]+}} = cir.cast(float_to_int, %{{[0-9]+}} : !cir.vector<!cir.double x 2>), !cir.vector<!u16i x 2>
}

// CIR: cir.func {{@.*vector_swizzle.*}}
// LLVM: define void {{@.*vector_swizzle.*}}
void vector_swizzle() {

  vi4 a = { 1, 2, 3, 4 };

  vi2 b = a.wz;
  // CIR:      %[[#LOAD1:]] = cir.load %{{[0-9]+}} : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#SHUFFLE1:]] = cir.vec.shuffle(%[[#LOAD1]], %[[#LOAD1]] : !cir.vector<!s32i x 4>) [#cir.int<3> : !s32i, #cir.int<2> : !s32i] : !cir.vector<!s32i x 2>
  // CIR-NEXT: cir.store %[[#SHUFFLE1]], %{{[0-9]+}} : !cir.vector<!s32i x 2>, !cir.ptr<!cir.vector<!s32i x 2>>

  // LLVM:      %[[#LOAD1:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#SHUFFLE1:]] = shufflevector <4 x i32> %[[#LOAD1]], <4 x i32> %[[#LOAD1]], <2 x i32> <i32 3, i32 2>
  // LLVM-NEXT: store <2 x i32> %[[#SHUFFLE1]], ptr %{{[0-9]+}}, align 8

  a.wz = a.xy;
  // CIR-NEXT: %[[#LOAD2:]] = cir.load %{{[0-9]+}} : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#SHUFFLE2:]] = cir.vec.shuffle(%[[#LOAD2]], %[[#LOAD2]] : !cir.vector<!s32i x 4>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<!s32i x 2>
  // CIR-NEXT: %[[#LOAD3:]] = cir.load %{{[0-9]+}} : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#SHUFFLE3:]] = cir.vec.shuffle(%[[#SHUFFLE2]], %[[#SHUFFLE2]] : !cir.vector<!s32i x 2>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#SHUFFLE4:]] = cir.vec.shuffle(%[[#LOAD3]], %[[#SHUFFLE3]] : !cir.vector<!s32i x 4>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<5> : !s32i, #cir.int<4> : !s32i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: cir.store %[[#SHUFFLE4]], %{{[0-9]+}} : !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>

  // LLVM-NEXT: %[[#LOAD2:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#SHUFFLE2:]] = shufflevector <4 x i32> %[[#LOAD2]], <4 x i32> %[[#LOAD2]], <2 x i32> <i32 0, i32 1>
  // LLVM-NEXT: %[[#LOAD3:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#SHUFFLE3:]] = shufflevector <2 x i32> %[[#SHUFFLE2]], <2 x i32> %[[#SHUFFLE2]], <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  // LLVM-NEXT: %[[#SHUFFLE4:]] = shufflevector <4 x i32> %[[#LOAD3]], <4 x i32> %[[#SHUFFLE3]], <4 x i32> <i32 0, i32 1, i32 5, i32 4>
  // LLVM-NEXT: store <4 x i32> %[[#SHUFFLE4]], ptr %{{[0-9]+}}, align 16

  a.xy = b;
  // CIR-NEXT: %[[#LOAD4RHS:]] = cir.load %{{[0-9]+}} : !cir.ptr<!cir.vector<!s32i x 2>>, !cir.vector<!s32i x 2>
  // CIR-NEXT: %[[#LOAD5LHS:]] = cir.load %{{[0-9]+}} : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#SHUFFLE5:]] = cir.vec.shuffle(%[[#LOAD4RHS]], %[[#LOAD4RHS]] : !cir.vector<!s32i x 2>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#SHUFFLE6:]] = cir.vec.shuffle(%[[#LOAD5LHS]], %[[#SHUFFLE5]] : !cir.vector<!s32i x 4>) [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: cir.store %[[#SHUFFLE6]], %{{[0-9]+}} : !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>

  // LLVM-NEXT: %[[#LOAD4RHS:]] = load <2 x i32>, ptr %{{[0-9]+}}, align 8
  // LLVM-NEXT: %[[#LOAD5LHS:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#SHUFFLE5:]] = shufflevector <2 x i32> %[[#LOAD4RHS]], <2 x i32> %[[#LOAD4RHS]], <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  // LLVM-NEXT: %[[#SHUFFLE6:]] = shufflevector <4 x i32> %[[#LOAD5LHS]], <4 x i32> %[[#SHUFFLE5]], <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  // LLVM-NEXT: store <4 x i32> %[[#SHUFFLE6]], ptr %{{[0-9]+}}, align 16

  b = a.yw;
  // CIR-NEXT: %[[#LOAD6:]] = cir.load %{{[0-9]+}} : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#SHUFFLE7:]] = cir.vec.shuffle(%[[#LOAD6]], %[[#LOAD6]] : !cir.vector<!s32i x 4>) [#cir.int<1> : !s32i, #cir.int<3> : !s32i] : !cir.vector<!s32i x 2>
  // CIR-NEXT: cir.store %[[#SHUFFLE7]], %{{[0-9]+}} : !cir.vector<!s32i x 2>, !cir.ptr<!cir.vector<!s32i x 2>>

  // LLVM-NEXT: %[[#LOAD6:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#SHUFFLE7:]] = shufflevector <4 x i32> %[[#LOAD6]], <4 x i32> %[[#LOAD6]], <2 x i32> <i32 1, i32 3>
  // LLVM-NEXT: store <2 x i32> %[[#SHUFFLE7]], ptr %{{[0-9]+}}, align 8

  a.s0 = 1;
  // CIR-NEXT: cir.const #cir.int<1>
  // CIR-NEXT: %[[#LOAD7:]] = cir.load %{{[0-9]+}} : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#INSERT_INDEX:]] = cir.const #cir.int<0> : !s64i
  // CIR-NEXT: %[[#INSERT1:]] = cir.vec.insert %{{[0-9]+}}, %[[#LOAD7]][%[[#INSERT_INDEX]] : !s64i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: cir.store %[[#INSERT1]], %{{[0-9]+}} : !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>

  // LLVM-NEXT: %[[#LOAD7:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#INSERT1:]] = insertelement <4 x i32> %[[#LOAD7]], i32 1, i64 0
  // LLVM-NEXT: store <4 x i32> %[[#INSERT1]], ptr %{{[0-9]+}}, align 16

  int one_elem_load = a.s2;
  // CIR-NEXT: %[[#LOAD8:]] = cir.load %{{[0-9]+}} : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#EXTRACT_INDEX:]] = cir.const #cir.int<2> : !s64i
  // CIR-NEXT: %[[#EXTRACT1:]] = cir.vec.extract %[[#LOAD8]][%[[#EXTRACT_INDEX]] : !s64i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: cir.store %[[#EXTRACT1]], %{{[0-9]+}} : !s32i, !cir.ptr<!s32i>

  // LLVM-NEXT: %[[#LOAD8:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#EXTRACT1:]] = extractelement <4 x i32> %[[#LOAD8]], i64 2
  // LLVM-NEXT: store i32 %[[#EXTRACT1]], ptr %{{[0-9]+}}, align 4

}

// CIR: cir.func {{@.*vector_extend.*}}
// LLVM: define void {{@.*vector_extend.*}}
void vector_extend() {
  vi4 a;
  // CIR: %[[#PVECA:]] = cir.alloca !cir.vector<!s32i x 4>
  // LLVM: %[[#PVECA:]] = alloca <4 x i32>

  vi2 b = {1, 2};
  // CIR-NEXT: %[[#PVECB:]] = cir.alloca !cir.vector<!s32i x 2>
  // LLVM-NEXT: %[[#PVECB:]] = alloca <2 x i32>

  vi7 c = {};
  // CIR-NEXT: %[[#PVECC:]] = cir.alloca !cir.vector<!s32i x 7>
  // LLVM-NEXT: %[[#PVECC:]] = alloca <7 x i32>

  a.lo = b;
  // CIR: %[[#VECB:]] = cir.load %[[#PVECB]] : !cir.ptr<!cir.vector<!s32i x 2>>, !cir.vector<!s32i x 2>
  // CIR-NEXT: %[[#VECA:]] = cir.load %[[#PVECA]] : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#EXTVECB:]] = cir.vec.shuffle(%[[#VECB]], %[[#VECB]] : !cir.vector<!s32i x 2>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#RESULT:]] = cir.vec.shuffle(%[[#VECA]], %[[#EXTVECB]] : !cir.vector<!s32i x 4>) [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: cir.store %[[#RESULT]], %[[#PVECA]] : !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>

  // LLVM: %[[#VECB:]] = load <2 x i32>, ptr %[[#PVECB]], align 8
  // LLVM-NEXT: %[[#VECA:]] = load <4 x i32>, ptr %[[#PVECA]], align 16
  // LLVM-NEXT: %[[#EXTVECB:]] = shufflevector <2 x i32> %[[#VECB]], <2 x i32> %[[#VECB]], <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  // LLVM-NEXT: %[[#RESULT:]] = shufflevector <4 x i32> %[[#VECA]], <4 x i32> %[[#EXTVECB]], <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  // LLVM-NEXT: store <4 x i32> %[[#RESULT]], ptr %[[#PVECA]], align 16

  // OpenCL C Specification 6.3.7. Vector Components
  // The suffixes .lo (or .even) and .hi (or .odd) for a 3-component vector type
  // operate as if the 3-component vector type is a 4-component vector type with
  // the value in the w component undefined.
  a = c.hi;

  // CIR-NEXT: %[[#VECC:]] = cir.load %[[#PVECC]] : !cir.ptr<!cir.vector<!s32i x 7>>, !cir.vector<!s32i x 7>
  // CIR-NEXT: %[[#HIPART:]] = cir.vec.shuffle(%[[#VECC]], %[[#VECC]] : !cir.vector<!s32i x 7>) [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: cir.store %[[#HIPART]], %[[#PVECA]] : !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>

  // LLVM-NEXT: %[[#VECC:]] = load <7 x i32>, ptr %[[#PVECC]], align 32
  // LLVM-NEXT: %[[#HIPART:]] = shufflevector <7 x i32> %[[#VECC]], <7 x i32> %[[#VECC]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM-NEXT: store <4 x i32> %[[#HIPART]], ptr %[[#PVECA]], align 16

  // c.hi is c[4, 5, 6, 7], in which 7 should be ignored in CIRGen for store
  c.hi = a;

  // CIR-NEXT: %[[#VECA:]] = cir.load %[[#PVECA]] : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#VECC:]] = cir.load %[[#PVECC]] : !cir.ptr<!cir.vector<!s32i x 7>>, !cir.vector<!s32i x 7>
  // CIR-NEXT: %[[#EXTVECA:]] = cir.vec.shuffle(%[[#VECA]], %[[#VECA]] : !cir.vector<!s32i x 4>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<!s32i x 7>
  // CIR-NEXT: %[[#RESULT:]] = cir.vec.shuffle(%[[#VECC]], %[[#EXTVECA]] : !cir.vector<!s32i x 7>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<7> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i] : !cir.vector<!s32i x 7>
  // CIR-NEXT: cir.store %[[#RESULT]], %[[#PVECC]] : !cir.vector<!s32i x 7>, !cir.ptr<!cir.vector<!s32i x 7>>

  // LLVM-NEXT: %[[#VECA:]] = load <4 x i32>, ptr %[[#PVECA]], align 16
  // LLVM-NEXT: %[[#VECC:]] = load <7 x i32>, ptr %[[#PVECC]], align 32
  // LLVM-NEXT: %[[#EXTVECA:]] = shufflevector <4 x i32> %[[#VECA]], <4 x i32> %[[#VECA]], <7 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison>
  // LLVM-NEXT: %[[#RESULT:]] = shufflevector <7 x i32> %[[#VECC]], <7 x i32> %[[#EXTVECA]], <7 x i32> <i32 0, i32 1, i32 2, i32 3, i32 7, i32 8, i32 9>
  // LLVM-NEXT: store <7 x i32> %[[#RESULT]], ptr %[[#PVECC]], align 32
}

// TODO(cir): Enable concat test when OpenCL lands
