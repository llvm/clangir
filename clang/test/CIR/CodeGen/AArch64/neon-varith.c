// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:            -emit-cir -target-feature +neon %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:            -emit-llvm -target-feature +neon %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// This test file contains tests of aarch64 NEON vector arithmetic intrinsics.

// REQUIRES: aarch64-registered-target || arm-registered-target
#include <arm_neon.h>

uint8x8_t test_vqadd_u8(uint8x8_t a, uint8x8_t b) {
  return vqadd_u8(a,b);
}

// CIR-LABEL: vqadd_u8
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.uqadd" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!u8i x 8>, !cir.vector<!u8i x 8>) -> !cir.vector<!u8i x 8>
// CIR: cir.return

// LLVM: {{.*}}test_vqadd_u8(<8 x i8>{{.*}} [[A:%.*]], <8 x i8>{{.*}} [[B:%.*]])
// LLVM: store <8 x i8> [[A]], ptr [[A_ADDR:%.*]], align 8
// LLVM: store <8 x i8> [[B]], ptr [[B_ADDR:%.*]], align 8
// LLVM: [[TMP_A:%.*]] = load <8 x i8>, ptr [[A_ADDR]], align 8
// LLVM: [[TMP_B:%.*]] = load <8 x i8>, ptr [[B_ADDR]], align 8
// LLVM: store <8 x i8> [[TMP_A]], ptr [[P0_ADDR:%.*]], align 8
// LLVM: store <8 x i8> [[TMP_B]], ptr [[P1_ADDR:%.*]], align 8
// LLVM: [[INTRN_A:%.*]] = load <8 x i8>, ptr [[P0_ADDR]], align 8
// LLVM: [[INTRN_B:%.*]] = load <8 x i8>, ptr [[P1_ADDR]], align 8
// LLVM: {{%.*}} = call <8 x i8> @llvm.aarch64.neon.uqadd.v8i8(<8 x i8> [[INTRN_A]], <8 x i8> [[INTRN_B]])
// LLVM: ret <8 x i8>

int8x8_t test_vqadd_s8(int8x8_t a, int8x8_t b) {
  return vqadd_s8(a,b);
}

// CIR-LABEL: vqadd_s8
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqadd" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s8i x 8>, !cir.vector<!s8i x 8>) -> !cir.vector<!s8i x 8>
// CIR: cir.return

// LLVM: {{.*}}test_vqadd_s8(<8 x i8>{{.*}} [[A:%.*]], <8 x i8>{{.*}} [[B:%.*]])
// LLVM: store <8 x i8> [[A]], ptr [[A_ADDR:%.*]], align 8
// LLVM: store <8 x i8> [[B]], ptr [[B_ADDR:%.*]], align 8
// LLVM: [[TMP_A:%.*]] = load <8 x i8>, ptr [[A_ADDR]], align 8
// LLVM: [[TMP_B:%.*]] = load <8 x i8>, ptr [[B_ADDR]], align 8
// LLVM: store <8 x i8> [[TMP_A]], ptr [[P0_ADDR:%.*]], align 8
// LLVM: store <8 x i8> [[TMP_B]], ptr [[P1_ADDR:%.*]], align 8
// LLVM: [[INTRN_A:%.*]] = load <8 x i8>, ptr [[P0_ADDR]], align 8
// LLVM: [[INTRN_B:%.*]] = load <8 x i8>, ptr [[P1_ADDR]], align 8
// LLVM: {{%.*}} = call <8 x i8> @llvm.aarch64.neon.sqadd.v8i8(<8 x i8> [[INTRN_A]], <8 x i8> [[INTRN_B]])
// LLVM: ret <8 x i8>

uint16x4_t test_vqadd_u16(uint16x4_t a, uint16x4_t b) {
  return vqadd_u16(a,b);
}

// CIR-LABEL: vqadd_u16
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.uqadd" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!u16i x 4>, !cir.vector<!u16i x 4>) -> !cir.vector<!u16i x 4>
// CIR: cir.return

// LLVM: {{.*}}test_vqadd_u16(<4 x i16>{{.*}} [[A:%.*]], <4 x i16>{{.*}} [[B:%.*]])
// LLVM: store <4 x i16> [[A]], ptr [[A_ADDR:%.*]], align 8
// LLVM: store <4 x i16> [[B]], ptr [[B_ADDR:%.*]], align 8
// LLVM: [[TMP_A:%.*]] = load <4 x i16>, ptr [[A_ADDR]], align 8
// LLVM: [[TMP_B:%.*]] = load <4 x i16>, ptr [[B_ADDR]], align 8
// LLVM: store <4 x i16> [[TMP_A]], ptr [[P0_ADDR:%.*]], align 8
// LLVM: store <4 x i16> [[TMP_B]], ptr [[P1_ADDR:%.*]], align 8
// LLVM: [[INTRN_A:%.*]] = load <4 x i16>, ptr [[P0_ADDR]], align 8
// LLVM: [[INTRN_B:%.*]] = load <4 x i16>, ptr [[P1_ADDR]], align 8
// LLVM: {{%.*}} = call <4 x i16> @llvm.aarch64.neon.uqadd.v4i16(<4 x i16> [[INTRN_A]], <4 x i16> [[INTRN_B]])
// LLVM: ret <4 x i16>

int16x4_t test_vqadd_s16(int16x4_t a, int16x4_t b) {
  return vqadd_s16(a,b);
}

// CIR-LABEL: vqadd_u16
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqadd" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!s16i x 4>
// CIR: cir.return

// LLVM: {{.*}}test_vqadd_s16(<4 x i16>{{.*}} [[A:%.*]], <4 x i16>{{.*}} [[B:%.*]])
// LLVM: store <4 x i16> [[A]], ptr [[A_ADDR:%.*]], align 8
// LLVM: store <4 x i16> [[B]], ptr [[B_ADDR:%.*]], align 8
// LLVM: [[TMP_A:%.*]] = load <4 x i16>, ptr [[A_ADDR]], align 8
// LLVM: [[TMP_B:%.*]] = load <4 x i16>, ptr [[B_ADDR]], align 8
// LLVM: store <4 x i16> [[TMP_A]], ptr [[P0_ADDR:%.*]], align 8
// LLVM: store <4 x i16> [[TMP_B]], ptr [[P1_ADDR:%.*]], align 8
// LLVM: [[INTRN_A:%.*]] = load <4 x i16>, ptr [[P0_ADDR]], align 8
// LLVM: [[INTRN_B:%.*]] = load <4 x i16>, ptr [[P1_ADDR]], align 8
// LLVM: {{%.*}} = call <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16> [[INTRN_A]], <4 x i16> [[INTRN_B]])
// LLVM: ret <4 x i16>

uint32x2_t test_vqadd_u32(uint32x2_t a, uint32x2_t b) {
  return vqadd_u32(a,b);
}

// CIR-LABEL: vqadd_u32
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.uqadd" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!u32i x 2>, !cir.vector<!u32i x 2>) -> !cir.vector<!u32i x 2>
// CIR: cir.return

// LLVM: {{.*}}test_vqadd_u32(<2 x i32>{{.*}} [[A:%.*]], <2 x i32>{{.*}} [[B:%.*]])
// LLVM: store <2 x i32> [[A]], ptr [[A_ADDR:%.*]], align 8
// LLVM: store <2 x i32> [[B]], ptr [[B_ADDR:%.*]], align 8
// LLVM: [[TMP_A:%.*]] = load <2 x i32>, ptr [[A_ADDR]], align 8
// LLVM: [[TMP_B:%.*]] = load <2 x i32>, ptr [[B_ADDR]], align 8
// LLVM: store <2 x i32> [[TMP_A]], ptr [[P0_ADDR:%.*]], align 8
// LLVM: store <2 x i32> [[TMP_B]], ptr [[P1_ADDR:%.*]], align 8
// LLVM: [[INTRN_A:%.*]] = load <2 x i32>, ptr [[P0_ADDR]], align 8
// LLVM: [[INTRN_B:%.*]] = load <2 x i32>, ptr [[P1_ADDR]], align 8
// LLVM: {{%.*}} = call <2 x i32> @llvm.aarch64.neon.uqadd.v2i32(<2 x i32> [[INTRN_A]], <2 x i32> [[INTRN_B]])
// LLVM: ret <2 x i32>

int32x2_t test_vqadd_s32(int32x2_t a, int32x2_t b) {
  return vqadd_s32(a,b);
}

// CIR-LABEL: vqadd_s32
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqadd" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>
// CIR: cir.return

// LLVM: {{.*}}test_vqadd_s32(<2 x i32>{{.*}} [[A:%.*]], <2 x i32>{{.*}} [[B:%.*]])
// LLVM: store <2 x i32> [[A]], ptr [[A_ADDR:%.*]], align 8
// LLVM: store <2 x i32> [[B]], ptr [[B_ADDR:%.*]], align 8
// LLVM: [[TMP_A:%.*]] = load <2 x i32>, ptr [[A_ADDR]], align 8
// LLVM: [[TMP_B:%.*]] = load <2 x i32>, ptr [[B_ADDR]], align 8
// LLVM: store <2 x i32> [[TMP_A]], ptr [[P0_ADDR:%.*]], align 8
// LLVM: store <2 x i32> [[TMP_B]], ptr [[P1_ADDR:%.*]], align 8
// LLVM: [[INTRN_A:%.*]] = load <2 x i32>, ptr [[P0_ADDR]], align 8
// LLVM: [[INTRN_B:%.*]] = load <2 x i32>, ptr [[P1_ADDR]], align 8
// LLVM: {{%.*}} = call <2 x i32> @llvm.aarch64.neon.sqadd.v2i32(<2 x i32> [[INTRN_A]], <2 x i32> [[INTRN_B]])
// LLVM: ret <2 x i32>

uint64x1_t test_vqadd_u64(uint64x1_t a, uint64x1_t b) {
  return vqadd_u64(a,b);
}

// CIR-LABEL: vqadd_u64
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.uqadd" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!u64i x 1>, !cir.vector<!u64i x 1>) -> !cir.vector<!u64i x 1>
// CIR: cir.return

// LLVM: {{.*}}test_vqadd_u64(<1 x i64>{{.*}} [[A:%.*]], <1 x i64>{{.*}} [[B:%.*]])
// LLVM: store <1 x i64> [[A]], ptr [[A_ADDR:%.*]], align 8
// LLVM: store <1 x i64> [[B]], ptr [[B_ADDR:%.*]], align 8
// LLVM: [[TMP_A:%.*]] = load <1 x i64>, ptr [[A_ADDR]], align 8
// LLVM: [[TMP_B:%.*]] = load <1 x i64>, ptr [[B_ADDR]], align 8
// LLVM: store <1 x i64> [[TMP_A]], ptr [[P0_ADDR:%.*]], align 8
// LLVM: store <1 x i64> [[TMP_B]], ptr [[P1_ADDR:%.*]], align 8
// LLVM: [[INTRN_A:%.*]] = load <1 x i64>, ptr [[P0_ADDR]], align 8
// LLVM: [[INTRN_B:%.*]] = load <1 x i64>, ptr [[P1_ADDR]], align 8
// LLVM: {{%.*}} = call <1 x i64> @llvm.aarch64.neon.uqadd.v1i64(<1 x i64> [[INTRN_A]], <1 x i64> [[INTRN_B]])
// LLVM: ret <1 x i64>

int64x1_t test_vqadd_s64(int64x1_t a, int64x1_t b) {
  return vqadd_s64(a,b);
}

// CIR-LABEL: vqadd_s64
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqadd" {{%.*}}, {{%.*}} : 
// CIR-SAME: (!cir.vector<!s64i x 1>, !cir.vector<!s64i x 1>) -> !cir.vector<!s64i x 1>
// CIR: cir.return

// LLVM: {{.*}}test_vqadd_s64(<1 x i64>{{.*}} [[A:%.*]], <1 x i64>{{.*}} [[B:%.*]])
// LLVM: store <1 x i64> [[A]], ptr [[A_ADDR:%.*]], align 8
// LLVM: store <1 x i64> [[B]], ptr [[B_ADDR:%.*]], align 8
// LLVM: [[TMP_A:%.*]] = load <1 x i64>, ptr [[A_ADDR]], align 8
// LLVM: [[TMP_B:%.*]] = load <1 x i64>, ptr [[B_ADDR]], align 8
// LLVM: store <1 x i64> [[TMP_A]], ptr [[P0_ADDR:%.*]], align 8
// LLVM: store <1 x i64> [[TMP_B]], ptr [[P1_ADDR:%.*]], align 8
// LLVM: [[INTRN_A:%.*]] = load <1 x i64>, ptr [[P0_ADDR]], align 8
// LLVM: [[INTRN_B:%.*]] = load <1 x i64>, ptr [[P1_ADDR]], align 8
// LLVM: {{%.*}} = call <1 x i64> @llvm.aarch64.neon.sqadd.v1i64(<1 x i64> [[INTRN_A]], <1 x i64> [[INTRN_B]])
// LLVM: ret <1 x i64>

uint8x8_t test_vqrshrun_n_s16(int16x8_t a) {
  return vqrshrun_n_s16(a, 3);
}

// CIR-LABEL: test_vqrshrun_n_s16
// CIR: [[INTRN_ARG1:%.*]] = cir.const #cir.int<3> : !s32i
// CIR: [[INTRN_ARG0:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s16i x 8> 
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqrshrun" [[INTRN_ARG0]], [[INTRN_ARG1]] :
// CIR-SAME: (!cir.vector<!s16i x 8>, !s32i) -> !cir.vector<!u8i x 8>

// LLVM: {{.*}}test_vqrshrun_n_s16(<8 x i16>{{.*}} [[A:%.*]])
// LLVM: store <8 x i16> [[A]], ptr [[A_ADDR:%.*]], align 16
// LLVM: [[A_VAL:%.*]] = load <8 x i16>, ptr [[A_ADDR]], align 16
// LLVM: store <8 x i16> [[A_VAL]], ptr [[S0:%.*]], align 16
// LLVM: [[S0_VAL:%.*]] = load <8 x i16>, ptr [[S0]], align 16
// LLVM: [[S0_VAL_CAST:%.*]] = bitcast <8 x i16> [[S0_VAL]] to <16 x i8>
// LLVM: [[INTRN_ARG:%.*]] = bitcast <16 x i8> [[S0_VAL_CAST]] to <8 x i16>
// LLVM: {{%.*}} = call <8 x i8> @llvm.aarch64.neon.sqrshrun.v8i8(<8 x i16> [[INTRN_ARG]], i32 3)
// LLVM: ret <8 x i8> {{%.*}}

uint16x4_t test_vqrshrun_n_s32(int32x4_t a) {
  return vqrshrun_n_s32(a, 7);
}

// CIR-LABEL: test_vqrshrun_n_s32
// CIR: [[INTRN_ARG1:%.*]] = cir.const #cir.int<7> : !s32i
// CIR: [[INTRN_ARG0:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s32i x 4> 
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqrshrun" [[INTRN_ARG0]], [[INTRN_ARG1]] :
// CIR-SAME: (!cir.vector<!s32i x 4>, !s32i) -> !cir.vector<!u16i x 4>

// LLVM: {{.*}}test_vqrshrun_n_s32(<4 x i32>{{.*}} [[A:%.*]])
// LLVM: store <4 x i32> [[A]], ptr [[A_ADDR:%.*]], align 16
// LLVM: [[A_VAL:%.*]] = load <4 x i32>, ptr [[A_ADDR]], align 16
// LLVM: store <4 x i32> [[A_VAL]], ptr [[S0:%.*]], align 16
// LLVM: [[S0_VAL:%.*]] = load <4 x i32>, ptr [[S0]], align 16
// LLVM: [[S0_VAL_CAST:%.*]] = bitcast <4 x i32> [[S0_VAL]] to <16 x i8>
// LLVM: [[INTRN_ARG:%.*]] = bitcast <16 x i8> [[S0_VAL_CAST]] to <4 x i32>
// LLVM: {{%.*}} = call <4 x i16> @llvm.aarch64.neon.sqrshrun.v4i16(<4 x i32> [[INTRN_ARG]], i32 7)
// LLVM: ret <4 x i16> {{%.*}}

uint32x2_t test_vqrshrun_n_s64(int64x2_t a) {
  return vqrshrun_n_s64(a, 15);
}

// CIR-LABEL: test_vqrshrun_n_s64
// CIR: [[INTRN_ARG1:%.*]] = cir.const #cir.int<15> : !s32i
// CIR: [[INTRN_ARG0:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s64i x 2> 
// CIR: {{%.*}} = cir.llvm.intrinsic "llvm.aarch64.neon.sqrshrun" [[INTRN_ARG0]], [[INTRN_ARG1]] :
// CIR-SAME: (!cir.vector<!s64i x 2>, !s32i) -> !cir.vector<!u32i x 2>

// LLVM: {{.*}}test_vqrshrun_n_s64(<2 x i64>{{.*}} [[A:%.*]])
// LLVM: store <2 x i64> [[A]], ptr [[A_ADDR:%.*]], align 16
// LLVM: [[A_VAL:%.*]] = load <2 x i64>, ptr [[A_ADDR]], align 16
// LLVM: store <2 x i64> [[A_VAL]], ptr [[S0:%.*]], align 16
// LLVM: [[S0_VAL:%.*]] = load <2 x i64>, ptr [[S0]], align 16
// LLVM: [[S0_VAL_CAST:%.*]] = bitcast <2 x i64> [[S0_VAL]] to <16 x i8>
// LLVM: [[INTRN_ARG:%.*]] = bitcast <16 x i8> [[S0_VAL_CAST]] to <2 x i64>
// LLVM: {{%.*}} = call <2 x i32> @llvm.aarch64.neon.sqrshrun.v2i32(<2 x i64> [[INTRN_ARG]], i32 15)
// LLVM: ret <2 x i32> {{%.*}}
