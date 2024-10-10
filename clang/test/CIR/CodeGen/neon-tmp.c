// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:    -fclangir -disable-O0-optnone \
// RUN:  -flax-vector-conversions=none -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:  -fclangir   -disable-O0-optnone \
// RUN:  -flax-vector-conversions=none -emit-llvm -o - %s \
// RUN: | opt -S -passes=mem2reg,simplifycfg -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// REQUIRES: aarch64-registered-target || arm-registered-target

// This test mimics clang/test/CodeGen/aarch64-neon-intrinsics.c, which eventually
// CIR shall be able to support fully. Since this is going to take some time to converge,
// the unsupported/NYI code is commented out, so that we can incrementally improve this.
// The NYI filecheck used contains the LLVM output from OG codegen that should guide the
// correct result when implementing this into the CIR pipeline.

#include <arm_neon.h>

uint8x8_t test_vld1_dup_u8(uint8_t const * ptr) {
  return vld1_dup_u8(ptr);
}

// CIR-LABEL: vld1_dup_u8
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!u8i>, !u8i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !u8i, !cir.vector<!u8i x 8> poison

// LLVM: {{.*}}test_vld1_dup_u8(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i8, ptr [[PTR]], align 1
// LLVM: [[VEC:%.*]] = insertelement <8 x i8> poison, i8 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <8 x i8> [[VEC]], <8 x i8> poison, <8 x i32> zeroinitializer

int8x8_t test_vld1_dup_s8(int8_t const * ptr) {
  return vld1_dup_s8(ptr);
}

// CIR-LABEL: test_vld1_dup_s8
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!s8i>, !s8i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !s8i, !cir.vector<!s8i x 8> poison

// LLVM: {{.*}}test_vld1_dup_s8(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i8, ptr [[PTR]], align 1
// LLVM: [[VEC:%.*]] = insertelement <8 x i8> poison, i8 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <8 x i8> [[VEC]], <8 x i8> poison, <8 x i32> zeroinitializer

uint16x4_t test_vld1_dup_u16(uint16_t const * ptr) {
  return vld1_dup_u16(ptr);
}

// CIR-LABEL: test_vld1_dup_u16
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!u16i>, !u16i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !u16i, !cir.vector<!u16i x 4> poison

// LLVM: {{.*}}test_vld1_dup_u16(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i16, ptr [[PTR]], align 2
// LLVM: [[VEC:%.*]] = insertelement <4 x i16> poison, i16 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <4 x i16> [[VEC]], <4 x i16> poison, <4 x i32> zeroinitializer

int16x4_t test_vld1_dup_s16(int16_t const * ptr) {
  return vld1_dup_s16(ptr);
}

// CIR-LABEL: test_vld1_dup_s16
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!s16i>, !s16i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !s16i, !cir.vector<!s16i x 4> poison

// LLVM: {{.*}}test_vld1_dup_s16(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i16, ptr [[PTR]], align 2
// LLVM: [[VEC:%.*]] = insertelement <4 x i16> poison, i16 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <4 x i16> [[VEC]], <4 x i16> poison, <4 x i32> zeroinitializer

int32x2_t test_vld1_dup_s32(int32_t const * ptr) {
  return vld1_dup_s32(ptr);
}

// CIR-LABEL: test_vld1_dup_s32
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!s32i>, !s32i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !s32i, !cir.vector<!s32i x 2> poison

// LLVM: {{.*}}test_vld1_dup_s32(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i32, ptr [[PTR]], align 4
// LLVM: [[VEC:%.*]] = insertelement <2 x i32> poison, i32 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <2 x i32> [[VEC]], <2 x i32> poison, <2 x i32> zeroinitializer

int64x1_t test_vld1_dup_s64(int64_t const * ptr) {
  return vld1_dup_s64(ptr);
}

// CIR-LABEL: test_vld1_dup_s64
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!s64i>, !s64i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !s64i, !cir.vector<!s64i x 1> poison

// LLVM: {{.*}}test_vld1_dup_s64(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i64, ptr [[PTR]], align 8
// LLVM: [[VEC:%.*]] = insertelement <1 x i64> poison, i64 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <1 x i64> [[VEC]], <1 x i64> poison, <1 x i32> zeroinitializer

float32x2_t test_vld1_dup_f32(float32_t const * ptr) {
  return vld1_dup_f32(ptr);
}

// CIR-LABEL: test_vld1_dup_f32
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!cir.float>, !cir.float
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !cir.float, !cir.vector<!cir.float x 2> poison

// LLVM: {{.*}}test_vld1_dup_f32(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load float, ptr [[PTR]], align 4
// LLVM: [[VEC:%.*]] = insertelement <2 x float> poison, float [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <2 x float> [[VEC]], <2 x float> poison, <2 x i32> zeroinitializer

float64x1_t test_vld1_dup_f64(float64_t const * ptr) {
  return vld1_dup_f64(ptr);
}

// CIR-LABEL: test_vld1_dup_f64
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!cir.double>, !cir.double
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !cir.double, !cir.vector<!cir.double x 1> poison

// LLVM: {{.*}}test_vld1_dup_f64(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load double, ptr [[PTR]], align 8
// LLVM: [[VEC:%.*]] = insertelement <1 x double> poison, double [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <1 x double> [[VEC]], <1 x double> poison, <1 x i32> zeroinitializer

uint8x16_t test_vld1q_dup_u8(uint8_t const * ptr) {
  return vld1q_dup_u8(ptr);
}

// CIR-LABEL: test_vld1q_dup_u8
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!u8i>, !u8i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !u8i, !cir.vector<!u8i x 16> poison

// LLVM: {{.*}}test_vld1q_dup_u8(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i8, ptr [[PTR]], align 1
// LLVM: [[VEC:%.*]] = insertelement <16 x i8> poison, i8 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <16 x i8> [[VEC]], <16 x i8> poison, <16 x i32> zeroinitializer

int8x16_t test_vld1q_dup_s8(int8_t const * ptr) {
  return vld1q_dup_s8(ptr);
}

// CIR-LABEL: test_vld1q_dup_s8
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!s8i>, !s8i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !s8i, !cir.vector<!s8i x 16> poison

// LLVM: {{.*}}test_vld1q_dup_s8(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i8, ptr [[PTR]], align 1
// LLVM: [[VEC:%.*]] = insertelement <16 x i8> poison, i8 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <16 x i8> [[VEC]], <16 x i8> poison, <16 x i32> zeroinitializer

uint16x8_t test_vld1q_dup_u16(uint16_t const * ptr) {
  return vld1q_dup_u16(ptr);
}

// CIR-LABEL: test_vld1q_dup_u16
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!u16i>, !u16i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !u16i, !cir.vector<!u16i x 8> poison

// LLVM: {{.*}}test_vld1q_dup_u16(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i16, ptr [[PTR]], align 2
// LLVM: [[VEC:%.*]] = insertelement <8 x i16> poison, i16 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <8 x i16> [[VEC]], <8 x i16> poison, <8 x i32> zeroinitializer

int16x8_t test_vld1q_dup_s16(int16_t const * ptr) {
  return vld1q_dup_s16(ptr);
}

// CIR-LABEL: test_vld1q_dup_s16
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!s16i>, !s16i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !s16i, !cir.vector<!s16i x 8> poison

// LLVM: {{.*}}test_vld1q_dup_s16(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i16, ptr [[PTR]], align 2
// LLVM: [[VEC:%.*]] = insertelement <8 x i16> poison, i16 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <8 x i16> [[VEC]], <8 x i16> poison, <8 x i32> zeroinitializer

int32x4_t test_vld1q_dup_s32(int32_t const * ptr) {
  return vld1q_dup_s32(ptr);
}

// CIR-LABEL: test_vld1q_dup_s32
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!s32i>, !s32i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !s32i, !cir.vector<!s32i x 4> poison

// LLVM: {{.*}}test_vld1q_dup_s32(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i32, ptr [[PTR]], align 4
// LLVM: [[VEC:%.*]] = insertelement <4 x i32> poison, i32 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <4 x i32> [[VEC]], <4 x i32> poison, <4 x i32> zeroinitializer

int64x2_t test_vld1q_dup_s64(int64_t const * ptr) {
  return vld1q_dup_s64(ptr);
}

// CIR-LABEL: test_vld1q_dup_s64
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!s64i>, !s64i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !s64i, !cir.vector<!s64i x 2> poison

// LLVM: {{.*}}test_vld1q_dup_s64(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i64, ptr [[PTR]], align 8
// LLVM: [[VEC:%.*]] = insertelement <2 x i64> poison, i64 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <2 x i64> [[VEC]], <2 x i64> poison, <2 x i32> zeroinitializer

float32x4_t test_vld1q_dup_f32(float32_t const * ptr) {
  return vld1q_dup_f32(ptr);
}

// CIR-LABEL: test_vld1q_dup_f32
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!cir.float>, !cir.float
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !cir.float, !cir.vector<!cir.float x 4> poison

// LLVM: {{.*}}test_vld1q_dup_f32(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load float, ptr [[PTR]], align 4
// LLVM: [[VEC:%.*]] = insertelement <4 x float> poison, float [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <4 x float> [[VEC]], <4 x float> poison, <4 x i32> zeroinitializer

float64x2_t test_vld1q_dup_f64(float64_t const * ptr) {
  return vld1q_dup_f64(ptr);
}

// CIR-LABEL: test_vld1q_dup_f64
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!cir.double>, !cir.double
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !cir.double, !cir.vector<!cir.double x 2> poison

// LLVM: {{.*}}test_vld1q_dup_f64(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load double, ptr [[PTR]], align 8
// LLVM: [[VEC:%.*]] = insertelement <2 x double> poison, double [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <2 x double> [[VEC]], <2 x double> poison, <2 x i32> zeroinitializer
