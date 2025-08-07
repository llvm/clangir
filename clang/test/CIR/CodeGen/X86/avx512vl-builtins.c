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

void test_mm_mask_store_epi64(void *__P, __mmask8 __U, __m128i __A) {
  // CIR-LABEL: _mm_mask_store_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s64i x 2>, !cir.ptr<!cir.vector<!s64i x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>) -> !void

  // LLVM-LABEL: @test_mm_mask_store_epi64
  // LLVM: @llvm.masked.store.v2i64.p0(<2 x i64> %{{.*}}, ptr %{{.*}}, i32 16, <2 x i1> %{{.*}})
  return _mm_mask_store_epi64(__P, __U, __A); 
}

void test_mm_mask_store_ps(void *__P, __mmask8 __U, __m128 __A) {
  // CIR-LABEL: _mm_mask_store_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.float x 4>, !cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>) -> !void

  // LLVM-LABEL: @test_mm_mask_store_ps
  // LLVM: @llvm.masked.store.v4f32.p0(<4 x float> %{{.*}}, ptr %{{.*}}, i32 16, <4 x i1> %{{.*}})
  return _mm_mask_store_ps(__P, __U, __A); 
}

void test_mm_mask_store_pd(void *__P, __mmask8 __U, __m128d __A) {
  // CIR-LABEL: _mm_mask_store_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.double x 2>, !cir.ptr<!cir.vector<!cir.double x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>) -> !void

  // LLVM-LABEL: @test_mm_mask_store_pd
  // LLVM: @llvm.masked.store.v2f64.p0(<2 x double> %{{.*}}, ptr %{{.*}}, i32 16, <2 x i1> %{{.*}})
  return _mm_mask_store_pd(__P, __U, __A); 
}

void test_mm256_mask_store_epi32(void *__P, __mmask8 __U, __m256i __A) {
  // CIR-LABEL: _mm256_mask_store_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s32i x 8>, !cir.ptr<!cir.vector<!s32i x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: @test_mm256_mask_store_epi32
  // LLVM: @llvm.masked.store.v8i32.p0(<8 x i32> %{{.*}}, ptr %{{.*}}, i32 32, <8 x i1> %{{.*}})
  return _mm256_mask_store_epi32(__P, __U, __A); 
}

void test_mm256_mask_store_epi64(void *__P, __mmask8 __U, __m256i __A) {
  // CIR-LABEL: _mm256_mask_store_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s64i x 4>, !cir.ptr<!cir.vector<!s64i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>) -> !void

  // LLVM-LABEL: @test_mm256_mask_store_epi64
  // LLVM: @llvm.masked.store.v4i64.p0(<4 x i64> %{{.*}}, ptr %{{.*}}, i32 32, <4 x i1> %{{.*}})
  return _mm256_mask_store_epi64(__P, __U, __A); 
}

void test_mm256_mask_store_ps(void *__P, __mmask8 __U, __m256 __A) {
  // CIR-LABEL: _mm256_mask_store_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.float x 8>, !cir.ptr<!cir.vector<!cir.float x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: @test_mm256_mask_store_ps
  // LLVM: @llvm.masked.store.v8f32.p0(<8 x float> %{{.*}}, ptr %{{.*}}, i32 32, <8 x i1> %{{.*}})
  return _mm256_mask_store_ps(__P, __U, __A); 
}

void test_mm256_mask_store_pd(void *__P, __mmask8 __U, __m256d __A) {
  // CIR-LABEL: _mm256_mask_store_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.double x 4>, !cir.ptr<!cir.vector<!cir.double x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>) -> !void

  // LLVM-LABEL: @test_mm256_mask_store_pd
  // LLVM: @llvm.masked.store.v4f64.p0(<4 x double> %{{.*}}, ptr %{{.*}}, i32 32, <4 x i1> %{{.*}})
  return _mm256_mask_store_pd(__P, __U, __A); 
}
  
__m128 test_mm_mask_loadu_ps(__m128 __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_loadu_ps
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>

  // LLVM-LABEL: @test_mm_mask_loadu_ps
  // LLVM: @llvm.masked.load.v4f32.p0(ptr %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_mask_loadu_ps(__W, __U, __P); 
}

__m128 test_mm_maskz_loadu_ps(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_loadu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>

  // LLVM-LABEL: @test_mm_maskz_loadu_ps
  // LLVM: @llvm.masked.load.v4f32.p0(ptr %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_maskz_loadu_ps(__U, __P); 
}

__m256 test_mm256_mask_loadu_ps(__m256 __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_mask_loadu_ps
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.float x 8>) -> !cir.vector<!cir.float x 8>

  // LLVM-LABEL: @test_mm256_mask_loadu_ps
  // LLVM: @llvm.masked.load.v8f32.p0(ptr %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_mask_loadu_ps(__W, __U, __P); 
}

__m256 test_mm256_maskz_loadu_ps(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_loadu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.float x 8>) -> !cir.vector<!cir.float x 8>

  // LLVM-LABEL: @test_mm256_maskz_loadu_ps
  // LLVM: @llvm.masked.load.v8f32.p0(ptr %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_maskz_loadu_ps(__U, __P); 
}

__m256d test_mm256_mask_loadu_pd(__m256d __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_mask_loadu_pd
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.double x 4>) -> !cir.vector<!cir.double x 4>

  // LLVM-LABEL: @test_mm256_mask_loadu_pd
  // LLVM: @llvm.masked.load.v4f64.p0(ptr %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_mask_loadu_pd(__W, __U, __P); 
}

__m128i test_mm_mask_loadu_epi32(__m128i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_loadu_epi32
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM-LABEL: @test_mm_mask_loadu_epi32
  // LLVM: @llvm.masked.load.v4i32.p0(ptr %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_mask_loadu_epi32(__W, __U, __P); 
}

__m256i test_mm256_mask_loadu_epi32(__m256i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_mask_loadu_epi32
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s32i x 8>) -> !cir.vector<!s32i x 8>

  // LLVM-LABEL: @test_mm256_mask_loadu_epi32
  // LLVM: @llvm.masked.load.v8i32.p0(ptr %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_mask_loadu_epi32(__W, __U, __P); 
}

__m128i test_mm_mask_loadu_epi64(__m128i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_loadu_epi64
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM-LABEL: @test_mm_mask_loadu_epi64
  // LLVM: @llvm.masked.load.v2i64.p0(ptr %{{.*}}, i32 1, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_mask_loadu_epi64(__W, __U, __P); 
}

__m256i test_mm256_mask_loadu_epi64(__m256i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_mask_loadu_epi64
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s64i x 4>) -> !cir.vector<!s64i x 4>

  // LLVM-LABEL: @test_mm256_mask_loadu_epi64
  // LLVM: @llvm.masked.load.v4i64.p0(ptr %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_mask_loadu_epi64(__W, __U, __P); 
}

__m256i test_mm256_maskz_loadu_epi64(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_loadu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s64i x 4>) -> !cir.vector<!s64i x 4>

  // LLVM-LABEL: @test_mm256_maskz_loadu_epi64
  // LLVM: @llvm.masked.load.v4i64.p0(ptr %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_maskz_loadu_epi64(__U, __P); 
}

__m128 test_mm_mask_load_ps(__m128 __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_load_ps
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>

  // LLVM-LABEL: @test_mm_mask_load_ps
  // LLVM: @llvm.masked.load.v4f32.p0(ptr %{{.*}}, i32 16, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_mask_load_ps(__W, __U, __P); 
}

__m128 test_mm_maskz_load_ps(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_load_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>

  // LLVM-LABEL: @test_mm_maskz_load_ps
  // LLVM: @llvm.masked.load.v4f32.p0(ptr %{{.*}}, i32 16, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_maskz_load_ps(__U, __P); 
}

__m256 test_mm256_mask_load_ps(__m256 __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_mask_load_ps
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.float x 8>) -> !cir.vector<!cir.float x 8>

  // LLVM-LABEL: @test_mm256_mask_load_ps
  // LLVM: @llvm.masked.load.v8f32.p0(ptr %{{.*}}, i32 32, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_mask_load_ps(__W, __U, __P); 
}

__m256 test_mm256_maskz_load_ps(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_load_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.float x 8>) -> !cir.vector<!cir.float x 8>

  // LLVM-LABEL: @test_mm256_maskz_load_ps
  // LLVM: @llvm.masked.load.v8f32.p0(ptr %{{.*}}, i32 32, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_maskz_load_ps(__U, __P); 
}

__m128d test_mm_mask_load_pd(__m128d __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_load_pd
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>

  // LLVM-LABEL: @test_mm_mask_load_pd
  // LLVM: @llvm.masked.load.v2f64.p0(ptr %{{.*}}, i32 16, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_mask_load_pd(__W, __U, __P); 
}

__m128d test_mm_maskz_load_pd(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_load_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>

  // LLVM-LABEL: @test_mm_maskz_load_pd
  // LLVM: @llvm.masked.load.v2f64.p0(ptr %{{.*}}, i32 16, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_maskz_load_pd(__U, __P); 
}

__m128d test_mm_maskz_loadu_pd(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_loadu_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>

  // LLVM-LABEL: @test_mm_maskz_loadu_pd
  // LLVM: @llvm.masked.load.v2f64.p0(ptr %{{.*}}, i32 1, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_maskz_loadu_pd(__U, __P); 
}

__m256d test_mm256_mask_load_pd(__m256d __W, __mmask8 __U, void const *__P) {
  //CIR-LABEL: _mm256_mask_load_pd
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.double x 4>) -> !cir.vector<!cir.double x 4>

  // LLVM-LABEL: @test_mm256_mask_load_pd
  // LLVM: @llvm.masked.load.v4f64.p0(ptr %{{.*}}, i32 32, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_mask_load_pd(__W, __U, __P); 
}

__m256d test_mm256_maskz_load_pd(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_load_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.double x 4>) -> !cir.vector<!cir.double x 4>

  // LLVM-LABEL: @test_mm256_maskz_load_pd
  // LLVM: @llvm.masked.load.v4f64.p0(ptr %{{.*}}, i32 32, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_maskz_load_pd(__U, __P); 
}

__m256d test_mm256_maskz_loadu_pd(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_loadu_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.double x 4>) -> !cir.vector<!cir.double x 4>

  // LLVM-LABEL: @test_mm256_maskz_loadu_pd
  // LLVM: @llvm.masked.load.v4f64.p0(ptr %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_maskz_loadu_pd(__U, __P); 
}

__m128i test_mm_mask_load_epi32(__m128i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_load_epi32
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM-LABEL: @test_mm_mask_load_epi32
  // LLVM: @llvm.masked.load.v4i32.p0(ptr %{{.*}}, i32 16, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_mask_load_epi32(__W, __U, __P); 
}

__m128i test_mm_maskz_load_epi32(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_load_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM-LABEL: @test_mm_maskz_load_epi32
  // LLVM: @llvm.masked.load.v4i32.p0(ptr %{{.*}}, i32 16, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maskz_load_epi32(__U, __P); 
}

__m128i test_mm_maskz_loadu_epi32(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_loadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM-LABEL: @test_mm_maskz_loadu_epi32
  // LLVM: @llvm.masked.load.v4i32.p0(ptr %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maskz_loadu_epi32(__U, __P); 
}

__m256i test_mm256_maskz_load_epi32(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_load_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s32i x 8>) -> !cir.vector<!s32i x 8>

  // LLVM-LABEL: @test_mm256_maskz_load_epi32
  // LLVM: @llvm.masked.load.v8i32.p0(ptr %{{.*}}, i32 32, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_maskz_load_epi32(__U, __P); 
}

__m256i test_mm256_maskz_loadu_epi32(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_loadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s32i x 8>) -> !cir.vector<!s32i x 8>

  // LLVM-LABEL: @test_mm256_maskz_loadu_epi32
  // LLVM: @llvm.masked.load.v8i32.p0(ptr %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_maskz_loadu_epi32(__U, __P); 
}

__m128i test_mm_mask_load_epi64(__m128i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_load_epi64
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM-LABEL: @test_mm_mask_load_epi64
  // LLVM: @llvm.masked.load.v2i64.p0(ptr %{{.*}}, i32 16, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_mask_load_epi64(__W, __U, __P); 
}

__m128i test_mm_maskz_loadu_epi64(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_loadu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM-LABEL: @test_mm_maskz_loadu_epi64
  // LLVM: @llvm.masked.load.v2i64.p0(ptr %{{.*}}, i32 1, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_maskz_loadu_epi64(__U, __P); 
}

__m128i test_mm_maskz_load_epi64(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_load_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM-LABEL: @test_mm_maskz_load_epi64
  // LLVM: @llvm.masked.load.v2i64.p0(ptr %{{.*}}, i32 16, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_maskz_load_epi64(__U, __P); 
}

__m256i test_mm256_mask_load_epi64(__m256i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_mask_load_epi64
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s64i x 4>) -> !cir.vector<!s64i x 4>

  // LLVM-LABEL: @test_mm256_mask_load_epi64
  // LLVM: @llvm.masked.load.v4i64.p0(ptr %{{.*}}, i32 32, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_mask_load_epi64(__W, __U, __P); 
}

__m256i test_mm256_maskz_load_epi64(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_load_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s64i x 4>) -> !cir.vector<!s64i x 4>

  // LLVM-LABEL: @test_mm256_maskz_load_epi64
  // LLVM: @llvm.masked.load.v4i64.p0(ptr %{{.*}}, i32 32, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_maskz_load_epi64(__U, __P); 
}

__m128d test_mm_mask_expandloadu_pd(__m128d __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_expandloadu_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 2>>, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>

  // LLVM-LABEL: @test_mm_mask_expandloadu_pd
  // LLVM: @llvm.masked.expandload.v2f64(ptr %{{.*}}, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_mask_expandloadu_pd(__W,__U,__P); 
}

__m128d test_mm_maskz_expandloadu_pd(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_expandloadu_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %4, %8, %5 : (!cir.ptr<!cir.vector<!cir.double x 2>>, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>

  // LLVM-LABEL: @test_mm_maskz_expandloadu_pd
  // LLVM: @llvm.masked.expandload.v2f64(ptr %{{.*}}, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_maskz_expandloadu_pd(__U,__P); 
}

__m256d test_mm256_mask_expandloadu_pd(__m256d __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_mask_expandloadu_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 4>>, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.double x 4>) -> !cir.vector<!cir.double x 4>

  // LLVM-LABEL: @test_mm256_mask_expandloadu_pd
  // LLVM: @llvm.masked.expandload.v4f64(ptr %{{.*}}, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_mask_expandloadu_pd(__W,__U,__P); 
}

__m256d test_mm256_maskz_expandloadu_pd(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_expandloadu_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 4>>, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.double x 4>) -> !cir.vector<!cir.double x 4>

  // LLVM-LABEL: @test_mm256_maskz_expandloadu_pd
  // LLVM: @llvm.masked.expandload.v4f64(ptr %{{.*}}, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_maskz_expandloadu_pd(__U,__P); 
}

__m128 test_mm_mask_expandloadu_ps(__m128 __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_expandloadu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 4>>, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>

  // LLVM-LABEL: @test_mm_mask_expandloadu_ps
  // LLVM: @llvm.masked.expandload.v4f32(ptr %{{.*}}, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_mask_expandloadu_ps(__W,__U,__P); 
}

__m128 test_mm_maskz_expandloadu_ps(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_expandloadu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 4>>, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>

  // LLVM-LABEL: @test_mm_maskz_expandloadu_ps
  // LLVM: @llvm.masked.expandload.v4f32(ptr %{{.*}}, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_maskz_expandloadu_ps(__U,__P); 
}

__m256 test_mm256_mask_expandloadu_ps(__m256 __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_mask_expandloadu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 8>>, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.float x 8>) -> !cir.vector<!cir.float x 8>

  // LLVM-LABEL: @test_mm256_mask_expandloadu_ps
  // LLVM: @llvm.masked.expandload.v8f32(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_mask_expandloadu_ps(__W,__U,__P); 
}

__m256 test_mm256_maskz_expandloadu_ps(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_expandloadu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 8>>, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.float x 8>) -> !cir.vector<!cir.float x 8>

  // LLVM-LABEL: @test_mm256_maskz_expandloadu_ps
  // LLVM: @llvm.masked.expandload.v8f32(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_maskz_expandloadu_ps(__U,__P); 
}

__m128i test_mm_mask_expandloadu_epi64(__m128i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_expandloadu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 2>>, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM-LABEL: @test_mm_mask_expandloadu_epi64
  // LLVM: @llvm.masked.expandload.v2i64(ptr %{{.*}}, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_mask_expandloadu_epi64(__W,__U,__P); 
}

__m128i test_mm_maskz_expandloadu_epi64(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_expandloadu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 2>>, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM-LABEL: @test_mm_maskz_expandloadu_epi64
  // LLVM: @llvm.masked.expandload.v2i64(ptr %{{.*}}, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_maskz_expandloadu_epi64(__U,__P); 
}

__m128i test_mm_mask_expandloadu_epi32(__m128i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_mask_expandloadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM-LABEL: @test_mm_mask_expandloadu_epi32
  // LLVM: @llvm.masked.expandload.v4i32(ptr %{{.*}}, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_mask_expandloadu_epi32(__W,__U,__P); 
}

__m128i test_mm_maskz_expandloadu_epi32(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm_maskz_expandloadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM-LABEL: @test_mm_maskz_expandloadu_epi32
  // LLVM: @llvm.masked.expandload.v4i32(ptr %{{.*}}, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maskz_expandloadu_epi32(__U,__P); 
}

__m256i test_mm256_mask_expandloadu_epi32(__m256i __W, __mmask8 __U,   void const *__P) {
  // CIR-LABEL: _mm256_mask_expandloadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 8>>, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s32i x 8>) -> !cir.vector<!s32i x 8>

  // LLVM-LABEL: @test_mm256_mask_expandloadu_epi32
  // LLVM: @llvm.masked.expandload.v8i32(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_mask_expandloadu_epi32(__W,__U,__P); 
}

__m256i test_mm256_maskz_expandloadu_epi32(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm256_maskz_expandloadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 8>>, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s32i x 8>) -> !cir.vector<!s32i x 8>

  // LLVM-LABEL: @test_mm256_maskz_expandloadu_epi32
  // LLVM: @llvm.masked.expandload.v8i32(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_maskz_expandloadu_epi32(__U,__P);
}
