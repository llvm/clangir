// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CHECK %s
// RUN: cir-opt %t.cir -cir-to-llvm -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir --check-prefix=CHECK-MLIR %s

long double i = 0;
long double t2(long double i2) {
    return i2 + i ; 
}

// CHECK: cir.global external @i = #cir.fp<0.000000e+00> : !cir.long_double<!cir.f128> {alignment = 16 : i64} loc({{.*}})
// CHECK-LABEL:   cir.func @t2(%arg0: !cir.long_double<!cir.f128> loc({{.*}})) -> !cir.long_double<!cir.f128>
// CHECK-NEXT:    %[[#I2:]] = cir.alloca !cir.long_double<!cir.f128>, !cir.ptr<!cir.long_double<!cir.f128>>, ["i2", init] {alignment = 16 : i64}
// CHECK-NEXT:    %[[#RETVAL:]] = cir.alloca !cir.long_double<!cir.f128>, !cir.ptr<!cir.long_double<!cir.f128>>, ["__retval"] {alignment = 16 : i64}
// CHECK-NEXT:    cir.store %arg0, %[[#I2]] : !cir.long_double<!cir.f128>, !cir.ptr<!cir.long_double<!cir.f128>>
// CHECK-NEXT:    %[[#I2_LOAD:]] = cir.load %[[#I2]] : !cir.ptr<!cir.long_double<!cir.f128>>, !cir.long_double<!cir.f128>
// CHECK-NEXT:    %[[#I:]] = cir.get_global @i : !cir.ptr<!cir.long_double<!cir.f128>>
// CHECK-NEXT:    %[[#I_LOAD:]] = cir.load %[[#I]] : !cir.ptr<!cir.long_double<!cir.f128>>, !cir.long_double<!cir.f128>
// CHECK-NEXT:    %[[#ADD:]] = cir.binop(add, %[[#I2_LOAD]], %[[#I_LOAD]]) : !cir.long_double<!cir.f128>
// CHECK-NEXT:    cir.store %[[#ADD]], %[[#RETVAL]] : !cir.long_double<!cir.f128>, !cir.ptr<!cir.long_double<!cir.f128>>
// CHECK-NEXT:    %[[#RETVAL_LOAD:]] = cir.load %[[#RETVAL]] : !cir.ptr<!cir.long_double<!cir.f128>>, !cir.long_double<!cir.f128>
// CHECK-NEXT:    cir.return %[[#RETVAL_LOAD]] : !cir.long_double<!cir.f128>

// CHECK-MLIR:          llvm.mlir.global external @i(0.000000e+00 : f128) {addr_space = 0 : i32, alignment = 16 : i64} : f128
// CHECK-MLIR-LABEL:    llvm.func @t2(%arg0: f128) -> f128 attributes {cir.extra_attrs = #fn_attr, global_visibility = #cir<visibility default>}
// CHECK-MLIR-NEXT:     %[[#CONST1:]] = llvm.mlir.constant(1 : index) : i64
// CHECK-MLIR-NEXT:     %[[#OP1:]] = llvm.alloca %[[#CONST1]] x f128 {alignment = 16 : i64} : (i64) -> !llvm.ptr
// CHECK-MLIR-NEXT:     %[[#CONST2:]] = llvm.mlir.constant(1 : index) : i64
// CHECK-MLIR-NEXT:     %[[#RETVAL:]] = llvm.alloca %[[#CONST2]] x f128 {alignment = 16 : i64} : (i64) -> !llvm.ptr
// CHECK-MLIR-NEXT:     llvm.store %arg0, %[[#OP1]] {alignment = 16 : i64} : f128, !llvm.ptr
// CHECK-MLIR-NEXT:     %[[#OP1_LOAD:]] = llvm.load %[[#OP1]] {alignment = 16 : i64} : !llvm.ptr -> f128
// CHECK-MLIR-NEXT:     %[[#OP2_ADDR:]] = llvm.mlir.addressof @i : !llvm.ptr
// CHECK-MLIR-NEXT:     %[[#OP2_LOAD:]] = llvm.load %[[#OP2_ADDR]] {alignment = 16 : i64} : !llvm.ptr -> f128
// CHECK-MLIR-NEXT:     %[[#OP_RES:]] = llvm.fadd %[[#OP1_LOAD]], %[[#OP2_LOAD]]  : f128
// CHECK-MLIR-NEXT:     llvm.store %[[#OP_RES:]], %[[#RETVAL]] {alignment = 16 : i64} : f128, !llvm.ptr
// CHECK-MLIR-NEXT:     %[[#RETVAL_LOAD:]] = llvm.load %[[#RETVAL]] {alignment = 16 : i64} : !llvm.ptr -> f128
// CHECK-MLIR-NEXT:     llvm.return %[[#RETVAL_LOAD]] : f128