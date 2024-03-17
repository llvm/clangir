// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#ifdef __cplusplus
#define BOOL bool
#define VOID
#else
#define BOOL _Bool
#define VOID void
#endif

double _Complex float_complex_basic(VOID) {
  double _Complex x = 2.0 + 3.0i;
  return x;
}

//      CHECK: cir.func @{{.*}}float_complex_basic{{.*}}() -> !cir.complex<!cir.double>
//      CHECK:   %[[#REAL_FP:]] = cir.const(#cir.fp<2.000000e+00> : !cir.double) : !cir.double
// CHECK-NEXT:   %[[#IMAG:]] = cir.const(#cir.complex<#cir.fp<0.000000e+00> : !cir.double, #cir.fp<3.000000e+00> : !cir.double> : !cir.complex<!cir.double>) : !cir.complex<!cir.double>
// CHECK-NEXT:   %[[#REAL:]] = cir.cast(float_to_complex, %[[#REAL_FP]] : !cir.double), !cir.complex<!cir.double>
// CHECK-NEXT:   %{{.+}} = cir.binop(add, %[[#REAL]], %[[#IMAG]]) : !cir.complex<!cir.double>
//      CHECK: }

int _Complex int_complex_basic(VOID) {
  int _Complex x = 2 + 3i;
  return x;
}

//      CHECK: cir.func @{{.*}}int_complex_basic{{.*}}() -> !cir.complex<!s32i>
//      CHECK:   %[[#REAL_INT:]] = cir.const(#cir.int<2> : !s32i) : !s32i
// CHECK-NEXT:   %[[#REAL:]] = cir.cast(int_to_complex, %[[#REAL_INT]] : !s32i), !cir.complex<!s32i>
// CHECK-NEXT:   %[[#IMAG:]] = cir.const(#cir.complex<#cir.int<0> : !s32i, #cir.int<3> : !s32i> : !cir.complex<!s32i>) : !cir.complex<!s32i>
// CHECK-NEXT:   %{{.+}} = cir.binop(add, %[[#REAL]], %[[#IMAG]]) : !cir.complex<!s32i>
//      CHECK: }

int _Complex integral_to_complex(int x) {
  return x;
}

// CHECK: cir.func @{{.*}}integral_to_complex{{.*}}(%{{.+}}: !s32i loc({{.+}})) -> !cir.complex<!s32i>
// CHECK:   %{{.+}} = cir.cast(int_to_complex, %{{.+}} : !s32i), !cir.complex<!s32i>
// CHECK: }

float _Complex float_to_complex(float x) {
  return x;
}

// CHECK: cir.func @{{.*}}float_to_complex{{.*}}(%{{.+}}: !cir.float loc({{.+}})) -> !cir.complex<!cir.float>
// CHECK:   %{{.+}} = cir.cast(float_to_complex, %{{.+}} : !cir.float), !cir.complex<!cir.float>
// CHECK: }

double _Complex complex_cast(float _Complex x) {
  return x;
}

// CHECK: cir.func @{{.*}}complex_cast{{.*}}(%{{.+}}: !cir.complex<!cir.float> loc({{.+}})) -> !cir.complex<!cir.double>
// CHECK:   %{{.+}} = cir.cast(complex, %{{.+}} : !cir.complex<!cir.float>), !cir.complex<!cir.double>
// CHECK: }

float complex_to_element(float _Complex x) {
  return (float)x;
}

// CHECK: cir.func @{{.*}}complex_to_element{{.*}}(%{{.+}}: !cir.complex<!cir.float> loc({{.+}})) -> !cir.float
// CHECK:   %{{.+}} = cir.complex.real(%{{.+}} : !cir.complex<!cir.float>) : !cir.float
// CHECK: }

BOOL complex_to_bool(float _Complex x) {
  return x;
}

//      CHECK: cir.func @{{.*}}complex_to_bool{{.*}}(%{{.+}}: !cir.complex<!cir.float> loc({{.+}})) -> !cir.bool
//      CHECK:   %[[#ZERO:]] = cir.const(#cir.complex<#cir.fp<0.000000e+00> : !cir.float, #cir.fp<0.000000e+00> : !cir.float> : !cir.complex<!cir.float>) : !cir.complex<!cir.float>
// CHECK-NEXT:   %{{.+}} = cir.cmp(eq, %{{.+}}, %[[#ZERO]]) : !cir.complex<!cir.float>, !cir.bool
//      CHECK: }

float _Complex unary_op(float _Complex x) {
  return -x;
}

// CHECK: cir.func @{{.*}}unary_op{{.*}}(%{{.+}}: !cir.complex<!cir.float> loc({{.+}})) -> !cir.complex<!cir.float>
// CHECK:   %{{.+}} = cir.unary(minus, %{{.+}}) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CHECK: }

float _Complex bin_op(float _Complex x, float _Complex y) {
  return x + y;
}

// CHECK: cir.func @{{.*}}bin_op{{.*}}(%{{.+}}: !cir.complex<!cir.float> loc({{.+}}), %{{.+}}: !cir.complex<!cir.float> loc({{.+}})) -> !cir.complex<!cir.float>
// CHECK:   %{{.+}} = cir.binop(add, %{{.+}}, %{{.+}}) : !cir.complex<!cir.float>
// CHECK: }

float _Complex bin_op_with_real(float x, float _Complex y) {
  return x + y;
}

//      CHECK: cir.func @{{.*}}bin_op_with_real{{.*}}(%{{.+}}: !cir.float loc({{.+}}), %{{.+}}: !cir.complex<!cir.float> loc({{.+}})) -> !cir.complex<!cir.float>
//      CHECK:   %[[#OP:]] = cir.cast(float_to_complex, %{{.+}} : !cir.float), !cir.complex<!cir.float>
// CHECK-NEXT:   %{{.+}} = cir.binop(add, %[[#OP]], %{{.+}}) : !cir.complex<!cir.float>
//      CHECK: }

double _Complex global;

// CHECK: cir.global external @{{.*}}global{{.*}} = #cir.complex<#cir.fp<0.000000e+00> : !cir.double, #cir.fp<0.000000e+00> : !cir.double> : !cir.complex<!cir.double>
