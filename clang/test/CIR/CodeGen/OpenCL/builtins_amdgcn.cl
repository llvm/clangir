// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu tahiti -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu tahiti -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu tahiti -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

//===----------------------------------------------------------------------===//
// Test AMDGPU builtins
//===----------------------------------------------------------------------===//

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef unsigned long ulong;
typedef unsigned int uint;

// CIR-LABEL: @test_wave_reduce_add_u32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.add" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @test_wave_reduce_add_u32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_add_u32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_add_u32(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_add_u32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_add_u64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.add" {{.*}} : (!u64i, !s32i) -> !u64i
// LLVM: define{{.*}} void @test_wave_reduce_add_u64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.add.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_add_u64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.add.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_add_u64(global long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_add_u64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_sub_u32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.sub" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @test_wave_reduce_sub_u32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.sub.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_sub_u32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.sub.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_sub_u32(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_sub_u32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_sub_u64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.sub" {{.*}} : (!u64i, !s32i) -> !u64i
// LLVM: define{{.*}} void @test_wave_reduce_sub_u64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.sub.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_sub_u64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.sub.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_sub_u64(global long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_sub_u64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_min_i32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.min" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @test_wave_reduce_min_i32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.min.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_min_i32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.min.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_min_i32(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_min_i32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_min_u32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.umin" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @test_wave_reduce_min_u32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.umin.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_min_u32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.umin.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_min_u32(global uint* out, uint in) {
  *out = __builtin_amdgcn_wave_reduce_min_u32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_min_i64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.min" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @test_wave_reduce_min_i64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.min.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_min_i64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.min.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_min_i64(global long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_min_i64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_min_u64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.umin" {{.*}} : (!u64i, !s32i) -> !u64i
// LLVM: define{{.*}} void @test_wave_reduce_min_u64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.umin.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_min_u64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.umin.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_min_u64(global ulong* out, ulong in) {
  *out = __builtin_amdgcn_wave_reduce_min_u64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_max_i32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.max" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @test_wave_reduce_max_i32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.max.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_max_i32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.max.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_max_i32(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_max_i32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_max_u32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.umax" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @test_wave_reduce_max_u32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.umax.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_max_u32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.umax.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_max_u32(global uint* out, uint in) {
  *out = __builtin_amdgcn_wave_reduce_max_u32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_max_i64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.max" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @test_wave_reduce_max_i64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.max.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_max_i64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.max.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_max_i64(global long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_max_i64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_max_u64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.umax" {{.*}} : (!u64i, !s32i) -> !u64i
// LLVM: define{{.*}} void @test_wave_reduce_max_u64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.umax.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_max_u64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.umax.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_max_u64(global ulong* out, ulong in) {
  *out = __builtin_amdgcn_wave_reduce_max_u64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_and_b32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.and" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @test_wave_reduce_and_b32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.and.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_and_b32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.and.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_and_b32(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_and_b32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_and_b64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.and" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @test_wave_reduce_and_b64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.and.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_and_b64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.and.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_and_b64(global long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_and_b64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_or_b32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.or" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @test_wave_reduce_or_b32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.or.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_or_b32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.or.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_or_b32(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_or_b32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_or_b64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.or" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @test_wave_reduce_or_b64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.or.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_or_b64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.or.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_or_b64(global long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_or_b64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_xor_b32
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.xor" {{.*}} : (!s32i, !s32i) -> !s32i
// LLVM: define{{.*}} void @test_wave_reduce_xor_b32(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.xor.i32(i32 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_xor_b32(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.xor.i32(i32 %{{.*}}, i32 0)
void test_wave_reduce_xor_b32(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_xor_b32(in, 0);
}

// CIR-LABEL: @test_wave_reduce_xor_b64
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.xor" {{.*}} : (!s64i, !s32i) -> !s64i
// LLVM: define{{.*}} void @test_wave_reduce_xor_b64(
// LLVM: call i64 @llvm.amdgcn.wave.reduce.xor.i64(i64 %{{.*}}, i32 0)
// OGCG: define{{.*}} void @test_wave_reduce_xor_b64(
// OGCG: call i64 @llvm.amdgcn.wave.reduce.xor.i64(i64 %{{.*}}, i32 0)
void test_wave_reduce_xor_b64(global long* out, long in) {
  *out = __builtin_amdgcn_wave_reduce_xor_b64(in, 0);
}

// CIR-LABEL: @test_wave_reduce_add_u32_iterative
// CIR: cir.const #cir.int<1> : !s32i
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.add" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @test_wave_reduce_add_u32_iterative(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 1)
// OGCG: define{{.*}} void @test_wave_reduce_add_u32_iterative(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 1)
void test_wave_reduce_add_u32_iterative(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_add_u32(in, 1);
}

// CIR: cir.const #cir.int<2> : !s32i
// CIR: cir.llvm.intrinsic "amdgcn.wave.reduce.add" {{.*}} : (!u32i, !s32i) -> !u32i
// LLVM: define{{.*}} void @test_wave_reduce_add_u32_dpp(
// LLVM: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 2)
// OGCG: define{{.*}} void @test_wave_reduce_add_u32_dpp(
// OGCG: call i32 @llvm.amdgcn.wave.reduce.add.i32(i32 %{{.*}}, i32 2)
void test_wave_reduce_add_u32_dpp(global int* out, int in) {
  *out = __builtin_amdgcn_wave_reduce_add_u32(in, 2);
}
