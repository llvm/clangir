// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s
int foo() { int a; return __builtin_arm_ldrex(&a); }
void bar() { int a; int* p = (int*)__builtin_arm_ldrex(&a); }
// CHECK: cir.func no_proto  @foo() -> !s32i
// CHECK:  [[RET_VAL:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:  [[A:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a"] 
// CHECK:  [[INTRN_RET:%.*]] = cir.llvm.intrinsic "llvm.aarch64.ldxr" [[A]] : (!cir.ptr<!s32i>) -> !s64i
// CHECK:  [[RET_CAST:%.*]] = cir.cast(integral, [[INTRN_RET]] : !s64i), !s32i
// CHECK:  cir.store [[RET_CAST]], [[RET_VAL]] : !s32i, !cir.ptr<!s32i>
// CHECK:  [[RET:%.*]] = cir.load [[RET_VAL]] : !cir.ptr<!s32i>, !s32i
// CHECK:  cir.return [[RET]] : !s32i 

// CHECK: cir.func no_proto  @bar()
// CHECK: [[A:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a"]
// CHECK: [[PTR_P:%.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p", init]
// CHECK: [[INTRN_RET:%.*]] = cir.llvm.intrinsic "llvm.aarch64.ldxr" [[A]] : (!cir.ptr<!s32i>) -> !s64i
// CHECK: [[INT_CAST:%.*]] = cir.cast(integral, [[INTRN_RET]] : !s64i), !s32i
// CHECK: [[INT_U64:%.*]] = cir.cast(integral, [[INT_CAST]] : !s32i), !u64i
// CHECK: [[PTR_CAST:%.*]] = cir.cast(int_to_ptr, [[INT_U64]] : !u64i), !cir.ptr<!s32i>
// CHECK: cir.store [[PTR_CAST]], [[PTR_P]]  : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK: cir.return
