// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-canonicalize -o %t.cir %s 2>&1 | FileCheck --check-prefix=CIR-BEFORE %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-canonicalize -o %t.cir %s 2>&1 | FileCheck --check-prefix=CIR-AFTER %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll --check-prefixes=LLVM %s

struct CX {
  double real;
  double imag;
};

void complex_lvalue_bitcast() {
  struct CX a;
  (double _Complex &)a = {};
}

// CIR-BEFORE: %{{.*}} = cir.cast(bitcast, %{{.*}} : !cir.ptr<!rec_CX>), !cir.ptr<!cir.complex<!cir.double>>

// CIR-AFTER: %{{.*}} = cir.cast(bitcast, %{{.*}} : !cir.ptr<!rec_CX>), !cir.ptr<!cir.complex<!cir.double>>

// LLVM: %[[A_ADDR:.*]] = alloca %struct.CX, i64 1, align 8
// LLVM: store { double, double } zeroinitializer, ptr %[[A_ADDR]], align 8
