// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s


#include <immintrin.h>

void test_mm_mask_storeu_epi64(void *__P, __mmask8 __U, __m128i __A) {
  // CIR-LABEL: _mm_mask_storeu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s64i x 2>, !cir.ptr<!cir.vector<!s64i x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>)

  // LLVM-LABEL: @test_mm_mask_storeu_epi64
  // LLVM: @llvm.masked.store.v2i64.p0(<2 x i64> %{{.*}}, ptr %{{.*}}, i32 1, <2 x i1> %{{.*}})
  return _mm_mask_storeu_epi64(__P, __U, __A); 
}

void test_mm_mask_storeu_epi32(void *__P, __mmask8 __U, __m128i __A) {
  // CIR-LABEL: _mm_mask_storeu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>)

  // LLVM-LABEL: @test_mm_mask_storeu_epi32
  // LLVM: @llvm.masked.store.v4i32.p0(<4 x i32> %{{.*}}, ptr %{{.*}}, i32 1, <4 x i1> %{{.*}})
  return _mm_mask_storeu_epi32(__P, __U, __A); 
}

void test_mm_mask_storeu_pd(void *__P, __mmask8 __U, __m128d __A) {
  // CIR-LABEL: _mm_mask_storeu_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.double x 2>, !cir.ptr<!cir.vector<!cir.double x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>)

  // LLVM-LABEL: @test_mm_mask_storeu_pd
  // LLVM: @llvm.masked.store.v2f64.p0(<2 x double> %{{.*}}, ptr %{{.*}}, i32 1, <2 x i1> %{{.*}})
  return _mm_mask_storeu_pd(__P, __U, __A); 
}

void test_mm_mask_storeu_ps(void *__P, __mmask8 __U, __m128 __A) {
  // CIR-LABEL: _mm_mask_storeu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.float x 4>, !cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>)

  // LLVM-LABEL: @test_mm_mask_storeu_ps
  // LLVM: @llvm.masked.store.v4f32.p0(<4 x float> %{{.*}}, ptr %{{.*}}, i32 1, <4 x i1> %{{.*}})
  return _mm_mask_storeu_ps(__P, __U, __A); 
}

void test_mm256_mask_storeu_epi32(void *__P, __mmask8 __U, __m256i __A) {
  // CIR-LABEL: _mm256_mask_storeu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s32i x 8>, !cir.ptr<!cir.vector<!s32i x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>)

  // LLVM-LABEL: @test_mm256_mask_storeu_epi32
  // LLVM: @llvm.masked.store.v8i32.p0(<8 x i32> %{{.*}}, ptr %{{.*}}, i32 1, <8 x i1> %{{.*}})
  return _mm256_mask_storeu_epi32(__P, __U, __A); 
}

void test_mm256_mask_storeu_epi64(void *__P, __mmask8 __U, __m256i __A) {
  // CIR-LABEL: _mm256_mask_storeu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s64i x 4>, !cir.ptr<!cir.vector<!s64i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>)

  // LLVM-LABEL: @test_mm256_mask_storeu_epi64
  // LLVM: @llvm.masked.store.v4i64.p0(<4 x i64> %{{.*}}, ptr %{{.*}}, i32 1, <4 x i1> %{{.*}})
  return _mm256_mask_storeu_epi64(__P, __U, __A); 
}

void test_mm256_mask_storeu_ps(void *__P, __mmask8 __U, __m256 __A) {
  // CIR-LABEL: _mm256_mask_storeu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.float x 8>, !cir.ptr<!cir.vector<!cir.float x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: @test_mm256_mask_storeu_ps
  // LLVM: @llvm.masked.store.v8f32.p0(<8 x float> %{{.*}}, ptr %{{.*}}, i32 1, <8 x i1> %{{.*}})
  return _mm256_mask_storeu_ps(__P, __U, __A); 
}
