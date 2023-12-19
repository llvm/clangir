// RUN: %clang_cc1 -fclangir-enable -emit-cir -mmlir --mlir-print-ir-after-all %s -o - 2>&1 | FileCheck %s -check-prefix=CIR

// CIR: IR Dump After LibOpt (cir-lib-opt)