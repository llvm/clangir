// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir-flat -mmlir --mlir-print-ir-after=cir-call-conv-lowering %s -o %t.cir

int printf(const char *str, ...);

// CHECK: cir.func {{.*@bar}}
// CHECK:   %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CHECK:   %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CHECK:   cir.store %arg0, %0 : !s32i, !cir.ptr<!s32i>
// CHECK:   cir.store %arg1, %1 : !s32i, !cir.ptr<!s32i>
// CHECK:   %3 = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 7>>
// CHECK:   %4 = cir.cast(array_to_ptrdecay, %3 : !cir.ptr<!cir.array<!s8i x 7>>), !cir.ptr<!s8i>
// CHECK:   %5 = cir.load %0 : !cir.ptr<!s32i>, !s32i
// CHECK:   %6 = cir.load %1 : !cir.ptr<!s32i>, !s32i
// CHECK:   %7 = cir.call @printf(%4, %5, %6) : (!cir.ptr<!s8i>, !s32i, !s32i) -> !s32i
void bar(int a, int b) {
  printf("%d %d\n", a, b);
}
