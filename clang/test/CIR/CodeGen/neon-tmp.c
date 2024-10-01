// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:            -ffreestanding -emit-cir -target-feature +neon %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:            -ffreestanding -emit-llvm -target-feature +neon %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// REQUIRES: aarch64-registered-target || arm-registered-target
#include <arm_neon.h>

uint8x8_t test_vmovn_u16(uint16x8_t a) {
  return vmovn_u16(a);
}

// CIR-LABEL: vmovn_u16
// CIR: [[TMP:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!u16i x 8>), !cir.vector<!s8i x 16>
// CIR: [[SRC:%.*]] = cir.cast(bitcast, [[TMP]] : !cir.vector<!s8i x 16>), !cir.vector<!u16i x 8>
// CIR: {{.*}} = cir.cast(integral, [[SRC]] : !cir.vector<!u16i x 8>), !cir.vector<!u8i x 8>

// LLVM: {{.*}}test_vmovn_u16(<8 x i16>{{.*}}[[A:%.*]])
// LLVM: store <8 x i16> [[A]], ptr [[A_ADDR:%.*]], align 16
// LLVM: [[TMP0:%.*]] = load <8 x i16>, ptr [[A_ADDR]], align 16
// LLVM: store <8 x i16> [[TMP0]], ptr [[P0:%.*]], align 16
// LLVM: [[INTRN_ARG:%.*]] = load <8 x i16>, ptr [[P0]], align 16
// LLVM: {{%.*}} = bitcast <8 x i16> [[INTRN_ARG]] to <16 x i8>
// LLVM: {{%.*}} = trunc <8 x i16> [[INTRN_ARG]] to <8 x i8>
