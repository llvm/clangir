// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:            -emit-cir -target-feature +neon %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:            -emit-llvm -target-feature +neon %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// This test file contains test cases to those of 
// clang/test/CodeGen/aarch64-neon-vget.c 
// The difference is that this file only tests uses vget intrinsics, as we feel
// it would be proper to have a separate test file testing vset intrinsics 
// with the file name aarch64-neon-vset.c

// REQUIRES: aarch64-registered-target || arm-registered-target
#include <arm_neon.h>

uint8_t test_vget_lane_u8(uint8x8_t a) {
  return vget_lane_u8(a, 7);
}

// CIR-LABEL: test_vget_lane_u8
// CIR: [[IDX:%.*]]  = cir.const #cir.int<7> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u8i x 8>

// LLVM: define dso_local i8 @test_vget_lane_u8(<8 x i8> [[ARG:%.*]])
// LLVM: [[ARG_SAVE:%.*]] = alloca <8 x i8>, i64 1, align 8
// LLVM: store <8 x i8> [[ARG]], ptr [[ARG_SAVE]], align 8
// LLVM: [[TMP:%.*]] = load <8 x i8>, ptr [[ARG_SAVE:%.*]], align 8
// LLVM: store <8 x i8> [[TMP]], ptr [[S0:%.*]], align 8
// LLVM: [[INTRN_ARG:%.*]] = load <8 x i8>, ptr [[S0]], align 8
// LLVM: {{%.*}} = extractelement <8 x i8> [[INTRN_ARG]], i32 7
// LLVM: ret i8 {{%.*}}

uint8_t test_vgetq_lane_u8(uint8x16_t a) {
  return vgetq_lane_u8(a, 15);
}

// CIR-LABEL: test_vgetq_lane_u8
// CIR: [[IDX:%.*]]  = cir.const #cir.int<15> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u8i x 16>

// LLVM: define dso_local i8 @test_vgetq_lane_u8(<16 x i8> [[ARG:%.*]])
// LLVM: [[ARG_SAVE:%.*]] = alloca <16 x i8>, i64 1, align 16
// LLVM: store <16 x i8> [[ARG]], ptr [[ARG_SAVE]], align 16
// LLVM: [[TMP:%.*]] = load <16 x i8>, ptr [[ARG_SAVE:%.*]], align 16
// LLVM: store <16 x i8> [[TMP]], ptr [[S0:%.*]], align 16
// LLVM: [[INTRN_ARG:%.*]] = load <16 x i8>, ptr [[S0]], align 16
// LLVM: {{%.*}} = extractelement <16 x i8> [[INTRN_ARG]], i32 15
// LLVM: ret i8 {{%.*}}

uint16_t test_vget_lane_u16(uint16x4_t a) {
  return vget_lane_u16(a, 3);
}

// CIR-LABEL: test_vget_lane_u16
// CIR: [[IDX:%.*]]  = cir.const #cir.int<3> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u16i x 4>

// LLVM: define dso_local i16 @test_vget_lane_u16(<4 x i16> [[ARG:%.*]])
// LLVM: [[ARG_SAVE:%.*]] = alloca <4 x i16>, i64 1, align 8
// LLVM: store <4 x i16> [[ARG]], ptr [[ARG_SAVE]], align 8
// LLVM: [[TMP:%.*]] = load <4 x i16>, ptr [[ARG_SAVE:%.*]], align 8
// LLVM: store <4 x i16> [[TMP]], ptr [[S0:%.*]], align 8
// LLVM: [[INTRN_ARG:%.*]] = load <4 x i16>, ptr [[S0]], align 8
// LLVM: {{%.*}} = extractelement <4 x i16> [[INTRN_ARG]], i32 3
// LLVM: ret i16 {{%.*}}

uint16_t test_vgetq_lane_u16(uint16x8_t a) {
  return vgetq_lane_u16(a, 7);
}

// CIR-LABEL: test_vgetq_lane_u16
// CIR: [[IDX:%.*]]  = cir.const #cir.int<7> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u16i x 8>

// LLVM: define dso_local i16 @test_vgetq_lane_u16(<8 x i16> [[ARG:%.*]])
// LLVM: [[ARG_SAVE:%.*]] = alloca <8 x i16>, i64 1, align 16
// LLVM: store <8 x i16> [[ARG]], ptr [[ARG_SAVE]], align 16
// LLVM: [[TMP:%.*]] = load <8 x i16>, ptr [[ARG_SAVE:%.*]], align 16
// LLVM: store <8 x i16> [[TMP]], ptr [[S0:%.*]], align 16
// LLVM: [[INTRN_ARG:%.*]] = load <8 x i16>, ptr [[S0]], align 16
// LLVM: {{%.*}} = extractelement <8 x i16> [[INTRN_ARG]], i32 7
// LLVM: ret i16 {{%.*}}

uint32_t test_vget_lane_u32(uint32x2_t a) {
  return vget_lane_u32(a, 1);
}

// CIR-LABEL: test_vget_lane_u32
// CIR: [[IDX:%.*]]  = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u32i x 2>

// LLVM: define dso_local i32 @test_vget_lane_u32(<2 x i32> [[ARG:%.*]])
// LLVM: [[ARG_SAVE:%.*]] = alloca <2 x i32>, i64 1, align 8
// LLVM: store <2 x i32> [[ARG]], ptr [[ARG_SAVE]], align 8
// LLVM: [[TMP:%.*]] = load <2 x i32>, ptr [[ARG_SAVE:%.*]], align 8
// LLVM: store <2 x i32> [[TMP]], ptr [[S0:%.*]], align 8
// LLVM: [[INTRN_ARG:%.*]] = load <2 x i32>, ptr [[S0]], align 8
// LLVM: {{%.*}} = extractelement <2 x i32> [[INTRN_ARG]], i32 1
// LLVM: ret i32 {{%.*}}

uint32_t test_vgetq_lane_u32(uint32x4_t a) {
  return vgetq_lane_u32(a, 3);
}

// CIR-LABEL: test_vgetq_lane_u32
// CIR: [[IDX:%.*]]  = cir.const #cir.int<3> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u32i x 4>

// LLVM: define dso_local i32 @test_vgetq_lane_u32(<4 x i32> [[ARG:%.*]])
// LLVM: [[ARG_SAVE:%.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: store <4 x i32> [[ARG]], ptr [[ARG_SAVE]], align 16
// LLVM: [[TMP:%.*]] = load <4 x i32>, ptr [[ARG_SAVE:%.*]], align 16
// LLVM: store <4 x i32> [[TMP]], ptr [[S0:%.*]], align 16
// LLVM: [[INTRN_ARG:%.*]] = load <4 x i32>, ptr [[S0]], align 16
// LLVM: {{%.*}} = extractelement <4 x i32> [[INTRN_ARG]], i32 3
// LLVM: ret i32 {{%.*}}

uint64_t test_vget_lane_u64(uint64x1_t a) {
  return vget_lane_u64(a, 0);
}

// CIR-LABEL: test_vget_lane_u64
// CIR: [[IDX:%.*]]  = cir.const #cir.int<0> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u64i x 1>

// LLVM: define dso_local i64 @test_vget_lane_u64(<1 x i64> [[ARG:%.*]])
// LLVM: [[ARG_SAVE:%.*]] = alloca <1 x i64>, i64 1, align 8
// LLVM: store <1 x i64> [[ARG]], ptr [[ARG_SAVE]], align 8
// LLVM: [[TMP:%.*]] = load <1 x i64>, ptr [[ARG_SAVE:%.*]], align 8
// LLVM: store <1 x i64> [[TMP]], ptr [[S0:%.*]], align 8
// LLVM: [[INTRN_ARG:%.*]] = load <1 x i64>, ptr [[S0]], align 8
// LLVM: {{%.*}} = extractelement <1 x i64> [[INTRN_ARG]], i32 0
// LLVM: ret i64 {{%.*}}

uint64_t test_vgetq_lane_u64(uint64x2_t a) {
  return vgetq_lane_u64(a, 1);
}

// CIR-LABEL: test_vgetq_lane_u64
// CIR: [[IDX:%.*]]  = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!u64i x 2>

// LLVM: define dso_local i64 @test_vgetq_lane_u64(<2 x i64> [[ARG:%.*]])
// LLVM: [[ARG_SAVE:%.*]] = alloca <2 x i64>, i64 1, align 16
// LLVM: store <2 x i64> [[ARG]], ptr [[ARG_SAVE]], align 16
// LLVM: [[TMP:%.*]] = load <2 x i64>, ptr [[ARG_SAVE:%.*]], align 16
// LLVM: store <2 x i64> [[TMP]], ptr [[S0:%.*]], align 16
// LLVM: [[INTRN_ARG:%.*]] = load <2 x i64>, ptr [[S0]], align 16
// LLVM: {{%.*}} = extractelement <2 x i64> [[INTRN_ARG]], i32 1
// LLVM: ret i64 {{%.*}}

float32_t test_vget_lane_f32(float32x2_t a) {
  return vget_lane_f32(a, 1);
}

// CIR-LABEL: test_vget_lane_f32
// CIR: [[IDX:%.*]]  = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!cir.float x 2>

// LLVM: define dso_local float @test_vget_lane_f32(<2 x float> [[ARG:%.*]])
// LLVM: [[ARG_SAVE:%.*]] = alloca <2 x float>, i64 1, align 8
// LLVM: store <2 x float> [[ARG]], ptr [[ARG_SAVE]], align 8
// LLVM: [[TMP:%.*]] = load <2 x float>, ptr [[ARG_SAVE:%.*]], align 8
// LLVM: store <2 x float> [[TMP]], ptr [[S0:%.*]], align 8
// LLVM: [[INTRN_ARG:%.*]] = load <2 x float>, ptr [[S0]], align 8
// LLVM: {{%.*}} = extractelement <2 x float> [[INTRN_ARG]], i32 1
// LLVM: ret float {{%.*}}

float64_t test_vget_lane_f64(float64x1_t a) {
  return vget_lane_f64(a, 0);
}

// CIR-LABEL: test_vget_lane_f64
// CIR: [[IDX:%.*]]  = cir.const #cir.int<0> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!cir.double x 1>

// LLVM: define dso_local double @test_vget_lane_f64(<1 x double> [[ARG:%.*]])
// LLVM: [[ARG_SAVE:%.*]] = alloca <1 x double>, i64 1, align 8
// LLVM: store <1 x double> [[ARG]], ptr [[ARG_SAVE]], align 8
// LLVM: [[TMP:%.*]] = load <1 x double>, ptr [[ARG_SAVE:%.*]], align 8
// LLVM: store <1 x double> [[TMP]], ptr [[S0:%.*]], align 8
// LLVM: [[INTRN_ARG:%.*]] = load <1 x double>, ptr [[S0]], align 8
// LLVM: {{%.*}} = extractelement <1 x double> [[INTRN_ARG]], i32 0
// LLVM: ret double {{%.*}}

float32_t test_vgetq_lane_f32(float32x4_t a) {
  return vgetq_lane_f32(a, 3);
}

// CIR-LABEL: test_vgetq_lane_f32
// CIR: [[IDX:%.*]]  = cir.const #cir.int<3> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!cir.float x 4>

// LLVM: define dso_local float @test_vgetq_lane_f32(<4 x float> [[ARG:%.*]])
// LLVM: [[ARG_SAVE:%.*]] = alloca <4 x float>, i64 1, align 16
// LLVM: store <4 x float> [[ARG]], ptr [[ARG_SAVE]], align 16
// LLVM: [[TMP:%.*]] = load <4 x float>, ptr [[ARG_SAVE:%.*]], align 16
// LLVM: store <4 x float> [[TMP]], ptr [[S0:%.*]], align 16
// LLVM: [[INTRN_ARG:%.*]] = load <4 x float>, ptr [[S0]], align 16
// LLVM: {{%.*}} = extractelement <4 x float> [[INTRN_ARG]], i32 3
// LLVM: ret float {{%.*}}

float64_t test_vgetq_lane_f64(float64x2_t a) {
  return vgetq_lane_f64(a, 1);
}

// CIR-LABEL: test_vgetq_lane_f64
// CIR: [[IDX:%.*]]  = cir.const #cir.int<1> : !s32i
// CIR: {{%.*}} = cir.vec.extract {{%.*}}[[[IDX]] : !s32i] : !cir.vector<!cir.double x 2>

// LLVM: define dso_local double @test_vgetq_lane_f64(<2 x double> [[ARG:%.*]])
// LLVM: [[ARG_SAVE:%.*]] = alloca <2 x double>, i64 1, align 16
// LLVM: store <2 x double> [[ARG]], ptr [[ARG_SAVE]], align 16
// LLVM: [[TMP:%.*]] = load <2 x double>, ptr [[ARG_SAVE:%.*]], align 16
// LLVM: store <2 x double> [[TMP]], ptr [[S0:%.*]], align 16
// LLVM: [[INTRN_ARG:%.*]] = load <2 x double>, ptr [[S0]], align 16
// LLVM: {{%.*}} = extractelement <2 x double> [[INTRN_ARG]], i32 1
// LLVM: ret double {{%.*}}
