// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s
int foo() { int a; return __builtin_arm_ldrex(&a); }
// CHECK: cir.func no_proto  @foo() -> !s32i
// CHECK:  [[RET_VAL:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:  [[A:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a"] 
// CHECK:  [[INTRN_RET:%.*]] = "cir.intrinsic_call"([[A]]) <{name = "llvm.aarch64.ldxr"}> : (!cir.ptr<!s32i>) -> !s64i
// CHECK:  [[RET_CAST:%.*]] = cir.cast(bitcast, [[RET_VAL]] : !cir.ptr<!s32i>), !cir.ptr<!s64i>
// CHECK:  cir.store [[INTRN_RET]], [[RET_CAST]] : !s64i, !cir.ptr<!s64i>
// CHECK:  [[RET:%.*]] = cir.load [[RET_VAL]] : !cir.ptr<!s32i>, !s32i
// CHECK:  cir.return [[RET]] : !s32i 
