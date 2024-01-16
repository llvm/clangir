// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: cir-opt %t.cir -cir-to-llvm -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

typedef int vi4 __attribute__((vector_size(16)));
typedef double vd2 __attribute__((vector_size(16)));

void vector_int_test(int x) {

  // Vector constant. Not yet implemented. Expected results will change when
  // fully implemented.
  vi4 a = { 1, 2, 3, 4 };
  // CHECK: %[[#T30:]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[#T31:]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: %[[#T32:]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK: %[[#T33:]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: %[[#T34:]] = llvm.mlir.undef : vector<4xi32>
  // CHECK: %[[#T35:]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[#T36:]] = llvm.insertelement %[[#T30]], %[[#T34]][%[[#T35]] : i64] : vector<4xi32>
  // CHECK: %[[#T37:]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[#T38:]] = llvm.insertelement %[[#T31]], %[[#T36]][%[[#T37]] : i64] : vector<4xi32>
  // CHECK: %[[#T39:]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[#T40:]] = llvm.insertelement %[[#T32]], %[[#T38]][%[[#T39]] : i64] : vector<4xi32>
  // CHECK: %[[#T41:]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: %[[#T42:]] = llvm.insertelement %[[#T33]], %[[#T40]][%[[#T41]] : i64] : vector<4xi32>
  // CHECK: llvm.store %[[#T42]], %[[#T3:]] : vector<4xi32>, !llvm.ptr

  // Non-const vector initialization.
  vi4 b = { x, 5, 6, x + 1 };
  // CHECK: %[[#T43:]] = llvm.load %[[#T1:]] : !llvm.ptr -> i32
  // CHECK: %[[#T44:]] = llvm.mlir.constant(5 : i32) : i32
  // CHECK: %[[#T45:]] = llvm.mlir.constant(6 : i32) : i32
  // CHECK: %[[#T46:]] = llvm.load %[[#T1]] : !llvm.ptr -> i32
  // CHECK: %[[#T47:]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[#T48:]] = llvm.add %[[#T46]], %[[#T47]]  : i32
  // CHECK: %[[#T49:]] = llvm.mlir.undef : vector<4xi32>
  // CHECK: %[[#T50:]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[#T51:]] = llvm.insertelement %[[#T43]], %[[#T49]][%[[#T50]] : i64] : vector<4xi32>
  // CHECK: %[[#T52:]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[#T53:]] = llvm.insertelement %[[#T44]], %[[#T51]][%[[#T52]] : i64] : vector<4xi32>
  // CHECK: %[[#T54:]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[#T55:]] = llvm.insertelement %[[#T45]], %[[#T53]][%[[#T54]] : i64] : vector<4xi32>
  // CHECK: %[[#T56:]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: %[[#T57:]] = llvm.insertelement %[[#T48]], %[[#T55]][%[[#T56]] : i64] : vector<4xi32>
  // CHECK: llvm.store %[[#T57]], %[[#T5:]] : vector<4xi32>, !llvm.ptr

  // Extract element.
  int c = a[x];
  // CHECK: %[[#T58:]] = llvm.load %[[#T3]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T59:]] = llvm.load %[[#T1]] : !llvm.ptr -> i32
  // CHECK: %[[#T60:]] = llvm.extractelement %[[#T58]][%[[#T59]] : i32] : vector<4xi32>
  // CHECK: llvm.store %[[#T60]], %[[#T7:]] : i32, !llvm.ptr

  // Insert element.
  a[x] = x;
  // CHECK: %[[#T61:]] = llvm.load %[[#T1]] : !llvm.ptr -> i32
  // CHECK: %[[#T62:]] = llvm.load %[[#T1]] : !llvm.ptr -> i32
  // CHECK: %[[#T63:]] = llvm.load %[[#T3]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T64:]] = llvm.insertelement %[[#T61]], %[[#T63]][%[[#T62]] : i32] : vector<4xi32>
  // CHECK: llvm.store %[[#T64]], %[[#T3]] : vector<4xi32>, !llvm.ptr

  // Binary arithmetic operators.
  vi4 d = a + b;
  // CHECK: %[[#T65:]] = llvm.load %[[#T3]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T66:]] = llvm.load %[[#T5]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T67:]] = llvm.add %[[#T65]], %[[#T66]]  : vector<4xi32>
  // CHECK: llvm.store %[[#T67]], %[[#T9:]] : vector<4xi32>, !llvm.ptr
  vi4 e = a - b;
  // CHECK: %[[#T68:]] = llvm.load %[[#T3]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T69:]] = llvm.load %[[#T5]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T70:]] = llvm.sub %[[#T68]], %[[#T69]]  : vector<4xi32>
  // CHECK: llvm.store %[[#T70]], %[[#T11:]] : vector<4xi32>, !llvm.ptr
  vi4 f = a * b;
  // CHECK: %[[#T71:]] = llvm.load %[[#T3]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T72:]] = llvm.load %[[#T5]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T73:]] = llvm.mul %[[#T71]], %[[#T72]]  : vector<4xi32>
  // CHECK: llvm.store %[[#T73]], %[[#T13:]] : vector<4xi32>, !llvm.ptr
  vi4 g = a / b;
  // CHECK: %[[#T74:]] = llvm.load %[[#T3]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T75:]] = llvm.load %[[#T5]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T76:]] = llvm.sdiv %[[#T74]], %[[#T75]]  : vector<4xi32>
  // CHECK: llvm.store %[[#T76]], %[[#T15:]] : vector<4xi32>, !llvm.ptr
  vi4 h = a % b;
  // CHECK: %[[#T77:]] = llvm.load %[[#T3]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T78:]] = llvm.load %[[#T5]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T79:]] = llvm.srem %[[#T77]], %[[#T78]]  : vector<4xi32>
  // CHECK: llvm.store %[[#T79]], %[[#T17:]] : vector<4xi32>, !llvm.ptr
  vi4 i = a & b;
  // CHECK: %[[#T80:]] = llvm.load %[[#T3]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T81:]] = llvm.load %[[#T5]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T82:]] = llvm.and %[[#T80]], %[[#T81]]  : vector<4xi32>
  // CHECK: llvm.store %[[#T82]], %[[#T19:]] : vector<4xi32>, !llvm.ptr
  vi4 j = a | b;
  // CHECK: %[[#T83:]] = llvm.load %[[#T3]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T84:]] = llvm.load %[[#T5]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T85:]] = llvm.or %[[#T83]], %[[#T84]]  : vector<4xi32>
  // CHECK: llvm.store %[[#T85]], %[[#T21:]] : vector<4xi32>, !llvm.ptr
  vi4 k = a ^ b;
  // CHECK: %[[#T86:]] = llvm.load %[[#T3]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T87:]] = llvm.load %[[#T5]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T88:]] = llvm.xor %[[#T86]], %[[#T87]]  : vector<4xi32>
  // CHECK: llvm.store %[[#T88]], %[[#T23:]] : vector<4xi32>, !llvm.ptr

  // Unary arithmetic operators.
  vi4 l = +a;
  // CHECK: %[[#T89:]] = llvm.load %[[#T3]] : !llvm.ptr -> vector<4xi32>
  // CHECK: llvm.store %[[#T89]], %[[#T25:]] : vector<4xi32>, !llvm.ptr
  vi4 m = -a;
  // CHECK: %[[#T90:]] = llvm.load %[[#T3]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T91:]] = llvm.mlir.zero : vector<4xi32>
  // CHECK: %[[#T92:]] = llvm.sub %[[#T91]], %[[#T90]]  : vector<4xi32>
  // CHECK: llvm.store %[[#T92]], %[[#T27:]] : vector<4xi32>, !llvm.ptr
  vi4 n = ~a;
  // CHECK: %[[#T93:]] = llvm.load %[[#T3]] : !llvm.ptr -> vector<4xi32>
  // CHECK: %[[#T94:]] = llvm.mlir.constant(-1 : i32) : i32
  // CHECK: %[[#T95:]] = llvm.mlir.undef : vector<4xi32>
  // CHECK: %[[#T96:]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[#T97:]] = llvm.insertelement %[[#T94]], %[[#T95]][%[[#T96]] : i64] : vector<4xi32>
  // CHECK: %[[#T98:]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[#T99:]] = llvm.insertelement %[[#T94]], %[[#T97]][%[[#T98]] : i64] : vector<4xi32>
  // CHECK: %[[#T100:]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[#T101:]] = llvm.insertelement %[[#T94]], %[[#T99]][%[[#T100]] : i64] : vector<4xi32>
  // CHECK: %[[#T102:]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: %[[#T103:]] = llvm.insertelement %[[#T94]], %[[#T101]][%[[#T102]] : i64] : vector<4xi32>
  // CHECK: %[[#T104:]] = llvm.xor %[[#T103]], %[[#T93]]  : vector<4xi32>
  // CHECK: llvm.store %[[#T104]], %[[#T29:]] : vector<4xi32>, !llvm.ptr
}

void vector_double_test(int x, double y) {

  // Vector constant. Not yet implemented. Expected results will change when
  // fully implemented.
  vd2 a = { 1.5, 2.5 };
  // CHECK: %[[#T22:]] = llvm.mlir.constant(1.500000e+00 : f64) : f64
  // CHECK: %[[#T23:]] = llvm.mlir.constant(2.500000e+00 : f64) : f64
  // CHECK: %[[#T24:]] = llvm.mlir.undef : vector<2xf64>
  // CHECK: %[[#T25:]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[#T26:]] = llvm.insertelement %[[#T22]], %[[#T24]][%[[#T25]] : i64] : vector<2xf64>
  // CHECK: %[[#T27:]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[#T28:]] = llvm.insertelement %[[#T23]], %[[#T26]][%[[#T27]] : i64] : vector<2xf64>
  // CHECK: llvm.store %[[#T28]], %[[#T5:]] : vector<2xf64>, !llvm.ptr

  // Non-const vector initialization.
  vd2 b = { y, y + 1.0 };
  // CHECK: %[[#T29:]] = llvm.load %[[#T3:]] : !llvm.ptr -> f64
  // CHECK: %[[#T30:]] = llvm.load %[[#T3]] : !llvm.ptr -> f64
  // CHECK: %[[#T31:]] = llvm.mlir.constant(1.000000e+00 : f64) : f64
  // CHECK: %[[#T32:]] = llvm.fadd %[[#T30]], %[[#T31]]  : f64
  // CHECK: %[[#T33:]] = llvm.mlir.undef : vector<2xf64>
  // CHECK: %[[#T34:]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[#T35:]] = llvm.insertelement %[[#T29]], %[[#T33]][%[[#T34]] : i64] : vector<2xf64>
  // CHECK: %[[#T36:]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[#T37:]] = llvm.insertelement %[[#T32]], %[[#T35]][%[[#T36]] : i64] : vector<2xf64>
  // CHECK: llvm.store %[[#T37]], %[[#T7:]] : vector<2xf64>, !llvm.ptr

  // Extract element.
  double c = a[x];
  // CHECK: 38 = llvm.load %[[#T5]] : !llvm.ptr -> vector<2xf64>
  // CHECK: %[[#T39:]] = llvm.load %[[#T1]] : !llvm.ptr -> i32
  // CHECK: %[[#T40:]] = llvm.extractelement %[[#T38]][%[[#T39]] : i32] : vector<2xf64>
  // CHECK: llvm.store %[[#T40]], %[[#T9:]] : f64, !llvm.ptr

  // Insert element.
  a[x] = y;
  // CHECK: %[[#T41:]] = llvm.load %[[#T3]] : !llvm.ptr -> f64
  // CHECK: %[[#T42:]] = llvm.load %[[#T1:]] : !llvm.ptr -> i32
  // CHECK: %[[#T43:]] = llvm.load %[[#T5]] : !llvm.ptr -> vector<2xf64>
  // CHECK: %[[#T44:]] = llvm.insertelement %[[#T41]], %[[#T43]][%[[#T42]] : i32] : vector<2xf64>
  // CHECK: llvm.store %[[#T44]], %[[#T5]] : vector<2xf64>, !llvm.ptr

  // Binary arithmetic operators.
  vd2 d = a + b;
  // CHECK: %[[#T45:]] = llvm.load %[[#T5]] : !llvm.ptr -> vector<2xf64>
  // CHECK: %[[#T46:]] = llvm.load %[[#T7]] : !llvm.ptr -> vector<2xf64>
  // CHECK: %[[#T47:]] = llvm.fadd %[[#T45]], %[[#T46]]  : vector<2xf64>
  // CHECK: llvm.store %[[#T47]], %[[#T11:]] : vector<2xf64>, !llvm.ptr
  vd2 e = a - b;
  // CHECK: %[[#T48:]] = llvm.load %[[#T5]] : !llvm.ptr -> vector<2xf64>
  // CHECK: %[[#T49:]] = llvm.load %[[#T7]] : !llvm.ptr -> vector<2xf64>
  // CHECK: %[[#T50:]] = llvm.fsub %[[#T48]], %[[#T49]]  : vector<2xf64>
  // CHECK: llvm.store %[[#T50]], %[[#T13:]] : vector<2xf64>, !llvm.ptr
  vd2 f = a * b;
  // CHECK: %[[#T51:]] = llvm.load %[[#T5]] : !llvm.ptr -> vector<2xf64>
  // CHECK: %[[#T52:]] = llvm.load %[[#T7]] : !llvm.ptr -> vector<2xf64>
  // CHECK: %[[#T53:]] = llvm.fmul %[[#T51]], %[[#T52]]  : vector<2xf64>
  // CHECK: llvm.store %[[#T53]], %[[#T15:]] : vector<2xf64>, !llvm.ptr
  vd2 g = a / b;
  // CHECK: %[[#T54:]] = llvm.load %[[#T5]] : !llvm.ptr -> vector<2xf64>
  // CHECK: %[[#T55:]] = llvm.load %[[#T7]] : !llvm.ptr -> vector<2xf64>
  // CHECK: %[[#T56:]] = llvm.fdiv %[[#T54]], %[[#T55]]  : vector<2xf64>
  // CHECK: llvm.store %[[#T56]], %[[#T17:]] : vector<2xf64>, !llvm.ptr

  // Unary arithmetic operators.
  vd2 l = +a;
  // CHECK: %[[#T57:]] = llvm.load %[[#T5]] : !llvm.ptr -> vector<2xf64>
  // CHECK: llvm.store %[[#T57]], %[[#T19:]] : vector<2xf64>, !llvm.ptr
  vd2 m = -a;
  // CHECK: %[[#T58:]] = llvm.load %[[#T5]] : !llvm.ptr -> vector<2xf64>
  // CHECK: %[[#T59:]] = llvm.fneg %[[#T58]]  : vector<2xf64>
  // CHECK: llvm.store %[[#T59]], %[[#T21:]] : vector<2xf64>, !llvm.ptr
}
