// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

typedef int vi4 __attribute__((vector_size(16)));
typedef unsigned int uvi4 __attribute__((vector_size(16)));
typedef double vd2 __attribute__((vector_size(16)));
typedef long long vll2 __attribute__((vector_size(16)));
typedef unsigned short vus2 __attribute__((vector_size(4)));

vi4 vec_a;
// CHECK: cir.global external @[[VEC_A:.*]] = #cir.zero : !cir.vector<!s32i x 4>

vd2 b;
// CHECK: cir.global external @[[VEC_B:.*]] = #cir.zero : !cir.vector<!cir.double x 2>

vll2 c;
// CHECK: cir.global external @[[VEC_C:.*]] = #cir.zero : !cir.vector<!s64i x 2>

vi4 d = { 1, 2, 3, 4 };

// CHECK: cir.global external @[[VEC_D:.*]] = #cir.const_vector<[#cir.int<1> : !s32i, #cir.int<2> :
// CHECK-SAME: !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.vector<!s32i x 4>

void vector_int_test(int x, unsigned short usx) {

  // Vector constant.
  vi4 a = { 1, 2, 3, 4 };
  // CHECK: %{{[0-9]+}} = cir.const #cir.const_vector<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.vector<!s32i x 4>

  // Non-const vector initialization.
  vi4 b = { x, 5, 6, x + 1 };
  // CHECK: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} : !s32i, !s32i, !s32i, !s32i) : !cir.vector<!s32i x 4>

  // Incomplete vector initialization.
  vi4 bb = { x, x + 1 };
  // CHECK: %[[#zero:]] = cir.const #cir.int<0> : !s32i
  // CHECK: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}}, %[[#zero]], %[[#zero]] : !s32i, !s32i, !s32i, !s32i) : !cir.vector<!s32i x 4>

  // Scalar to vector conversion, a.k.a. vector splat.  Only valid as an
  // operand of a binary operator, not as a regular conversion.
  bb = a + 7;
  // CHECK: %[[#seven:]] = cir.const #cir.int<7> : !s32i
  // CHECK: %{{[0-9]+}} = cir.vec.splat %[[#seven]] : !s32i, !cir.vector<!s32i x 4>

  // Vector to vector conversion
  vd2 bbb = { };
  bb = (vi4)bbb;
  // CHECK: %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.vector<!cir.double x 2>), !cir.vector<!s32i x 4>

  // Extract element
  int c = a[x];
  // CHECK: %{{[0-9]+}} = cir.vec.extract %{{[0-9]+}}[%{{[0-9]+}} : !s32i] : !cir.vector<!s32i x 4>

  // Insert element
  a[x] = x;
  // CHECK: %[[#LOADEDVI:]] = cir.load{{.*}} %[[#STORAGEVI:]] : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CHECK: %[[#UPDATEDVI:]] = cir.vec.insert %{{[0-9]+}}, %[[#LOADEDVI]][%{{[0-9]+}} : !s32i] : !cir.vector<!s32i x 4>
  // CHECK: cir.store{{.*}} %[[#UPDATEDVI]], %[[#STORAGEVI]] : !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>

  // Compound assignment
  a[x] += a[0];
  // CHECK: %[[#LOADCA1:]] = cir.load{{.*}} %{{[0-9]+}} : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CHECK: %[[#RHSCA:]] = cir.vec.extract %[[#LOADCA1]][%{{[0-9]+}} : !s32i] : !cir.vector<!s32i x 4>
  // CHECK: %[[#LOADCAIDX2:]] = cir.load{{.*}} %{{[0-9]+}} : !cir.ptr<!s32i>, !s32i
  // CHECK: %[[#LOADCAVEC3:]] = cir.load{{.*}} %{{[0-9]+}} : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CHECK: %[[#LHSCA:]] = cir.vec.extract %[[#LOADCAVEC3]][%[[#LOADCAIDX2]] : !s32i] : !cir.vector<!s32i x 4>
  // CHECK: %[[#SUMCA:]] = cir.binop(add, %[[#LHSCA]], %[[#RHSCA]]) nsw : !s32i
  // CHECK: %[[#LOADCAVEC4:]] = cir.load{{.*}} %{{[0-9]+}} : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CHECK: %[[#RESULTCAVEC:]] = cir.vec.insert %[[#SUMCA]], %[[#LOADCAVEC4]][%[[#LOADCAIDX2]] : !s32i] : !cir.vector<!s32i x 4>
  // CHECK: cir.store{{.*}} %[[#RESULTCAVEC]], %{{[0-9]+}} : !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>

  // Binary arithmetic operations
  vi4 d = a + b;
  // CHECK: %{{[0-9]+}} = cir.binop(add, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 e = a - b;
  // CHECK: %{{[0-9]+}} = cir.binop(sub, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 f = a * b;
  // CHECK: %{{[0-9]+}} = cir.binop(mul, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 g = a / b;
  // CHECK: %{{[0-9]+}} = cir.binop(div, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 h = a % b;
  // CHECK: %{{[0-9]+}} = cir.binop(rem, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 i = a & b;
  // CHECK: %{{[0-9]+}} = cir.binop(and, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 j = a | b;
  // CHECK: %{{[0-9]+}} = cir.binop(or, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 k = a ^ b;
  // CHECK: %{{[0-9]+}} = cir.binop(xor, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>

  // Unary arithmetic operations
  vi4 l = +a;
  // CHECK: %{{[0-9]+}} = cir.unary(plus, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  vi4 m = -a;
  // CHECK: %{{[0-9]+}} = cir.unary(minus, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  vi4 n = ~a;
  // CHECK: %{{[0-9]+}} = cir.unary(not, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>

  // Ternary conditional operator
  vi4 tc = a ? b : d;
  // CHECK: %{{[0-9]+}} = cir.vec.ternary(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>

  // Comparisons
  vi4 o = a == b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(eq, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  vi4 p = a != b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(ne, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  vi4 q = a < b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(lt, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  vi4 r = a > b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(gt, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  vi4 s = a <= b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(le, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  vi4 t = a >= b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(ge, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>

  // __builtin_shufflevector
  vi4 u = __builtin_shufflevector(a, b, 7, 5, 3, 1);
  // CHECK: %{{[0-9]+}} = cir.vec.shuffle(%{{[0-9]+}}, %{{[0-9]+}} : !cir.vector<!s32i x 4>) [#cir.int<7> : !s64i, #cir.int<5> : !s64i, #cir.int<3> : !s64i, #cir.int<1> : !s64i] : !cir.vector<!s32i x 4>
  vi4 v = __builtin_shufflevector(a, b);
  // CHECK: %{{[0-9]+}} = cir.vec.shuffle.dynamic %{{[0-9]+}} : !cir.vector<!s32i x 4>, %{{[0-9]+}} : !cir.vector<!s32i x 4>

  // Shifts
  vi4 w = a << b;
  // CHECK: %{{[0-9]+}} = cir.shift(left, {{%.*}} : !cir.vector<!s32i x 4>,
  // CHECK-SAME: {{%.*}} : !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>
  vi4 y = a >> b;
  // CHECK: %{{[0-9]+}} = cir.shift(right, {{%.*}} : !cir.vector<!s32i x 4>,
  // CHECK-SAME: {{%.*}} : !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  vus2 z = { usx, usx };
  // CHECK: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}} : !u16i, !u16i) : !cir.vector<!u16i x 2>
  vus2 zamt = { 3, 4 };
  // CHECK: %{{[0-9]+}} = cir.const #cir.const_vector<[#cir.int<3> : !u16i, #cir.int<4> : !u16i]> : !cir.vector<!u16i x 2>
  vus2 zzz = z >> zamt;
  // CHECK: %{{[0-9]+}} = cir.shift(right, {{%.*}} : !cir.vector<!u16i x 2>,
  // CHECK-SAME: {{%.*}} : !cir.vector<!u16i x 2>) -> !cir.vector<!u16i x 2>

  // Vector to scalar conversion
  unsigned int zi = (unsigned int)z;
  // CHECK: %{{[0-9]+}} = cir.cast(bitcast, {{%.*}} : !cir.vector<!u16i x 2>), !u32i

  // Scalar to vector conversion
  vus2 zz = (vus2)zi;
  // CHECK: %{{[0-9]+}} = cir.cast(bitcast, {{%.*}} : !u32i), !cir.vector<!u16i x 2>

  // Vector to vector conversion
  vll2 aaa = (vll2)a;
  // CHECK: %{{[0-9]+}} = cir.cast(bitcast, {{%.*}} : !cir.vector<!s32i x 4>), !cir.vector<!s64i x 2>
}

void vector_double_test(int x, double y) {
  // Vector constant.
  vd2 a = { 1.5, 2.5 };
  // CHECK: %{{[0-9]+}} = cir.const #cir.const_vector<[#cir.fp<1.500000e+00> : !cir.double, #cir.fp<2.500000e+00> : !cir.double]> : !cir.vector<!cir.double x 2>

  // Non-const vector initialization.
  vd2 b = { y, y + 1.0 };
  // CHECK: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}} : !cir.double, !cir.double) : !cir.vector<!cir.double x 2>

  // Incomplete vector initialization
  vd2 bb = { y };
  // CHECK: [[#dzero:]] = cir.const #cir.fp<0.000000e+00> : !cir.double
  // CHECK: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %[[#dzero]] : !cir.double, !cir.double) : !cir.vector<!cir.double x 2>

  // Scalar to vector conversion, a.k.a. vector splat.  Only valid as an
  // operand of a binary operator, not as a regular conversion.
  bb = a + 2.5;
  // CHECK: %[[#twohalf:]] = cir.const #cir.fp<2.500000e+00> : !cir.double
  // CHECK: %{{[0-9]+}} = cir.vec.splat %[[#twohalf]] : !cir.double, !cir.vector<!cir.double x 2>

  // Extract element
  double c = a[x];
  // CHECK: %{{[0-9]+}} = cir.vec.extract %{{[0-9]+}}[%{{[0-9]+}} : !s32i] : !cir.vector<!cir.double x 2>

  // Insert element
  a[x] = y;
  // CHECK: %[[#LOADEDVF:]] = cir.load{{.*}} %[[#STORAGEVF:]] : !cir.ptr<!cir.vector<!cir.double x 2>>, !cir.vector<!cir.double x 2>
  // CHECK: %[[#UPDATEDVF:]] = cir.vec.insert %{{[0-9]+}}, %[[#LOADEDVF]][%{{[0-9]+}} : !s32i] : !cir.vector<!cir.double x 2>
  // CHECK: cir.store{{.*}} %[[#UPDATEDVF]], %[[#STORAGEVF]] : !cir.vector<!cir.double x 2>, !cir.ptr<!cir.vector<!cir.double x 2>>

  // Binary arithmetic operations
  vd2 d = a + b;
  // CHECK: %{{[0-9]+}} = cir.binop(add, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>
  vd2 e = a - b;
  // CHECK: %{{[0-9]+}} = cir.binop(sub, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>
  vd2 f = a * b;
  // CHECK: %{{[0-9]+}} = cir.binop(mul, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>
  vd2 g = a / b;
  // CHECK: %{{[0-9]+}} = cir.binop(div, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>

  // Unary arithmetic operations
  vd2 l = +a;
  // CHECK: %{{[0-9]+}} = cir.unary(plus, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!cir.double x 2>
  vd2 m = -a;
  // CHECK: %{{[0-9]+}} = cir.unary(minus, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!cir.double x 2>

  // Comparisons
  vll2 o = a == b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(eq, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>
  vll2 p = a != b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(ne, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>
  vll2 q = a < b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(lt, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>
  vll2 r = a > b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(gt, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>
  vll2 s = a <= b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(le, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>
  vll2 t = a >= b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(ge, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>

  // __builtin_convertvector
  vus2 w = __builtin_convertvector(a, vus2);
  // CHECK: %{{[0-9]+}} = cir.cast(float_to_int, %{{[0-9]+}} : !cir.vector<!cir.double x 2>), !cir.vector<!u16i x 2>
}

void vector_integers_shifts_test() {
  vi4 a = {1, 2, 3, 4};
  uvi4 b = {5u, 6u, 7u, 8u};

  vi4 shl = a << b;
  // CHECK:  %{{[0-9]+}} = cir.shift(left, %{{[0-9]+}} : !cir.vector<!s32i x 4>, %{{[0-9]+}} : !cir.vector<!u32i x 4>) -> !cir.vector<!s32i x 4>
  uvi4 shr = b >> a;
  // CHECK: %{{[0-9]+}} = cir.shift(right, %{{[0-9]+}} : !cir.vector<!u32i x 4>, %{{[0-9]+}} : !cir.vector<!s32i x 4>) -> !cir.vector<!u32i x 4>
}
