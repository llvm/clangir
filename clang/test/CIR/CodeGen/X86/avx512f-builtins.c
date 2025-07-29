// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

#include <immintrin.h>

void test_mm512_mask_storeu_epi64(void *__P, __mmask8 __U, __m512i __A) {
  // CIR-LABEL: _mm512_mask_storeu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s64i x 8>, !cir.ptr<!s64i>, !u32i, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: test_mm512_mask_storeu_epi64
  // LLVM: @llvm.masked.store.v8i64.p0(<8 x i64> %{{.*}}, ptr %{{.*}}, i32 1, <8 x i1> %{{.*}})
  return _mm512_mask_storeu_epi64(__P, __U, __A); 
}

void test_mm512_mask_storeu_epi32(void *__P, __mmask16 __U, __m512i __A) {
  // CIR-LABEL: _mm512_mask_storeu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s32i x 16>, !cir.ptr<!s32i>, !u32i, !cir.vector<!cir.int<s, 1> x 16>) -> !void

  // LLVM-LABEL: test_mm512_mask_storeu_epi32
  // LLVM: @llvm.masked.store.v16i32.p0(<16 x i32> %{{.*}}, ptr %{{.*}}, i32 1, <16 x i1> %{{.*}})
  return _mm512_mask_storeu_epi32(__P, __U, __A); 
}

void test_mm_mask_store_ss(float * __P, __mmask8 __U, __m128 __A){
  // CIR-LABEL: _mm_mask_store_ss
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.float x 4>, !cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>) -> !void

  // LLVM-LABEL: test_mm_mask_store_ss
  // LLVM: call void @llvm.masked.store.v4f32.p0(<4 x float> %{{.*}}, ptr %{{.*}}, i32 1, <4 x i1> %{{.*}})

  _mm_mask_store_ss(__P, __U, __A);
}

void test_mm_mask_store_sd(double * __P, __mmask8 __U, __m128d __A){
  // CIR-LABEL: _mm_mask_store_sd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.double x 2>, !cir.ptr<!cir.vector<!cir.double x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>) -> !void

  // LLVM-LABEL: test_mm_mask_store_sd
  // LLVM: call void @llvm.masked.store.v2f64.p0(<2 x double> %{{.*}}, ptr %{{.*}}, i32 1, <2 x i1> %{{.*}})
  _mm_mask_store_sd(__P, __U, __A);
}

void test_mm512_mask_store_pd(void *p, __m512d a, __mmask8 m){
  // CIR-LABEL: _mm512_mask_store_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.double x 8>, !cir.ptr<!cir.vector<!cir.double x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: test_mm512_mask_store_pd
  // LLVM: @llvm.masked.store.v8f64.p0(<8 x double> %{{.*}}, ptr %{{.*}}, i32 64, <8 x i1> %{{.*}})
  _mm512_mask_store_pd(p, m, a);
}

void test_mm512_mask_store_epi32(void *__P, __mmask16 __U, __m512i __A) {
  // CIR-LABEL: _mm512_mask_store_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s32i x 16>, !cir.ptr<!cir.vector<!s32i x 16>>, !u32i, !cir.vector<!cir.int<s, 1> x 16>) -> !void

  // LLVM-LABEL: test_mm512_mask_store_epi32
  // LLVM: @llvm.masked.store.v16i32.p0(<16 x i32> %{{.*}}, ptr %{{.*}}, i32 64, <16 x i1> %{{.*}})
  return _mm512_mask_store_epi32(__P, __U, __A); 
}

void test_mm512_mask_store_epi64(void *__P, __mmask8 __U, __m512i __A) {
  // CIR-LABEL: _mm512_mask_store_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s64i x 8>, !cir.ptr<!cir.vector<!s64i x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: test_mm512_mask_store_epi64
  // LLVM: @llvm.masked.store.v8i64.p0(<8 x i64> %{{.*}}, ptr %{{.*}}, i32 64, <8 x i1> %{{.*}})
  return _mm512_mask_store_epi64(__P, __U, __A); 
}

void test_mm512_mask_store_ps(void *p, __m512 a, __mmask16 m){
  // CIR-LABEL: _mm512_mask_store_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.float x 16>, !cir.ptr<!cir.vector<!cir.float x 16>>, !u32i, !cir.vector<!cir.int<s, 1> x 16>) -> !void

  // LLVM-LABEL: test_mm512_mask_store_ps
  // LLVM: @llvm.masked.store.v16f32.p0(<16 x float> %{{.*}}, ptr %{{.*}}, i32 64, <16 x i1> %{{.*}})
  _mm512_mask_store_ps(p, m, a);
}

__m512 test_mm512_mask_loadu_ps (__m512 __W, __mmask16 __U, void *__P)
{
  // CIR-LABEL: _mm512_mask_loadu_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.float>, !u32i, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!cir.float x 16>) -> !cir.vector<!cir.float x 16>

  // LLVM-LABEL: test_mm512_mask_loadu_ps
  // LLVM: @llvm.masked.load.v16f32.p0(ptr %{{.*}}, i32 1, <16 x i1> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_mask_loadu_ps (__W,__U, __P);
}

__m512 test_mm512_maskz_load_ps(__mmask16 __U, void *__P)
{

  // CIR-LABEL: _mm512_maskz_load_ps
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 16>>, !u32i, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!cir.float x 16>) -> !cir.vector<!cir.float x 16>

  // LLVM-LABEL: test_mm512_maskz_load_ps
  // LLVM: @llvm.masked.load.v16f32.p0(ptr %{{.*}}, i32 64, <16 x i1> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_maskz_load_ps(__U, __P);
}

__m512d test_mm512_mask_loadu_pd (__m512d __W, __mmask8 __U, void *__P)
{
  // CIR-LABEL: _mm512_mask_loadu_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.double>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.double x 8>) -> !cir.vector<!cir.double x 8>

  // LLVM-LABEL: test_mm512_mask_loadu_pd
  // LLVM: @llvm.masked.load.v8f64.p0(ptr %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x double> %{{.*}})
  return _mm512_mask_loadu_pd (__W,__U, __P);
}

__m512d test_mm512_maskz_load_pd(__mmask8 __U, void *__P)
{
  // CIR-LABEL: _mm512_maskz_load_pd
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.double x 8>) -> !cir.vector<!cir.double x 8>

  // LLVM-LABEL: test_mm512_maskz_load_pd
  // LLVM: @llvm.masked.load.v8f64.p0(ptr %{{.*}}, i32 64, <8 x i1> %{{.*}}, <8 x double> %{{.*}})
  return _mm512_maskz_load_pd(__U, __P);
}

__m512i test_mm512_mask_loadu_epi32 (__m512i __W, __mmask16 __U, void *__P)
{
  // CIR-LABEL: _mm512_mask_loadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!s32i>, !u32i, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!s32i x 16>) -> !cir.vector<!s32i x 16>

  // LLVM-LABEL: test_mm512_mask_loadu_epi32
  // LLVM: @llvm.masked.load.v16i32.p0(ptr %{{.*}}, i32 1, <16 x i1> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_mask_loadu_epi32 (__W,__U, __P);
}

__m512i test_mm512_maskz_loadu_epi32 (__mmask16 __U, void *__P)
{
  // CIR-LABEL: _mm512_maskz_loadu_epi32
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!s32i>, !u32i, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!s32i x 16>) -> !cir.vector<!s32i x 16>

  // LLVM-LABEL: test_mm512_maskz_loadu_epi32
  // LLVM: @llvm.masked.load.v16i32.p0(ptr %{{.*}}, i32 1, <16 x i1> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_maskz_loadu_epi32 (__U, __P);
}

__m512i test_mm512_mask_loadu_epi64 (__m512i __W, __mmask8 __U, void *__P)
{
  // CIR-LABEL: _mm512_mask_loadu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s64i x 8>) -> !cir.vector<!s64i x 8>

  // LLVM-LABEL: test_mm512_mask_loadu_epi64 
  // LLVM: @llvm.masked.load.v8i64.p0(ptr %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_mask_loadu_epi64 (__W,__U, __P);
}

__m512i test_mm512_maskz_loadu_epi64 (__mmask16 __U, void *__P)
{
  // CIR-LABEL: _mm512_maskz_loadu_epi64
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!s64i>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s64i x 8>) -> !cir.vector<!s64i x 8>

  // LLVM-LABEL: test_mm512_maskz_loadu_epi64
  // LLVM: @llvm.masked.load.v8i64.p0(ptr %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_maskz_loadu_epi64 (__U, __P);
}

__m128 test_mm_mask_load_ss(__m128 __A, __mmask8 __U, const float* __W)
{
  // CIR-LABEL: _mm_mask_load_ss
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>
  
  // LLVM-LABEL: test_mm_mask_load_ss
  // LLVM: call {{.*}}<4 x float> @llvm.masked.load.v4f32.p0(ptr %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_mask_load_ss(__A, __U, __W);
}

__m128 test_mm_maskz_load_ss (__mmask8 __U, const float * __W)
{
  // CIR-LABEL: _mm_maskz_load_ss
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>, !cir.vector<!cir.float x 4>) -> !cir.vector<!cir.float x 4>

  // LLVM-LABEL: test_mm_maskz_load_ss
  // LLVM: call {{.*}}<4 x float> @llvm.masked.load.v4f32.p0(ptr %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_maskz_load_ss (__U, __W);
}

__m128d test_mm_mask_load_sd (__m128d __A, __mmask8 __U, const double * __W)
{
  // CIR-LABEL: _mm_mask_load_sd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>

  // LLVM-LABEL: test_mm_mask_load_sd
  // LLVM: call {{.*}}<2 x double> @llvm.masked.load.v2f64.p0(ptr %{{.*}}, i32 1, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_mask_load_sd (__A, __U, __W);
}

__m128d test_mm_maskz_load_sd (__mmask8 __U, const double * __W)
{
  // CIR-LABEL: _mm_maskz_load_sd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 2>>, !u32i, !cir.vector<!cir.int<s, 1> x 2>, !cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>

  // LLVM-LABEL: test_mm_maskz_load_sd
  // LLVM: call {{.*}}<2 x double> @llvm.masked.load.v2f64.p0(ptr %{{.*}}, i32 1, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_maskz_load_sd (__U, __W);
}

__m512 test_mm512_mask_load_ps (__m512 __W, __mmask16 __U, void *__P)
{
  // CIR-LABEL: _mm512_mask_load_ps
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.float x 16>>, !u32i, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!cir.float x 16>) -> !cir.vector<!cir.float x 16>

  // LLVM-LABEL: test_mm512_mask_load_ps
  // LLVM: @llvm.masked.load.v16f32.p0(ptr %{{.*}}, i32 64, <16 x i1> %{{.*}}, <16 x float> %{{.*}})
  return _mm512_mask_load_ps (__W,__U, __P);
}

__m512d test_mm512_mask_load_pd (__m512d __W, __mmask8 __U, void *__P)
{
  // CIR-LABEL: _mm512_mask_load_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.double x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.double x 8>) -> !cir.vector<!cir.double x 8>

  // LLVM-LABEL: test_mm512_mask_load_pd
  // LLVM: @llvm.masked.load.v8f64.p0(ptr %{{.*}}, i32 64, <8 x i1> %{{.*}}, <8 x double> %{{.*}})
  return _mm512_mask_load_pd (__W,__U, __P);
}

__m512i test_mm512_mask_load_epi32(__m512i __W, __mmask16 __U, void const *__P) {
  // CIR-LABEL: _mm512_mask_load_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 16>>, !u32i, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!s32i x 16>) -> !cir.vector<!s32i x 16>

  // LLVM-LABEL: test_mm512_mask_load_epi32
  // LLVM: @llvm.masked.load.v16i32.p0(ptr %{{.*}}, i32 64, <16 x i1> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_mask_load_epi32(__W, __U, __P); 
}

__m512i test_mm512_mask_load_epi64(__m512i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm512_mask_load_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s64i x 8>) -> !cir.vector<!s64i x 8>

  // LLVM-LABEL: test_mm512_mask_load_epi64
  // LLVM: @llvm.masked.load.v8i64.p0(ptr %{{.*}}, i32 64, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_mask_load_epi64(__W, __U, __P); 
}

__m512i test_mm512_maskz_load_epi64(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm512_maskz_load_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s64i x 8>) -> !cir.vector<!s64i x 8>

  // LLVM-LABEL: test_mm512_maskz_load_epi64
  // LLVM: @llvm.masked.load.v8i64.p0(ptr %{{.*}}, i32 64, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_maskz_load_epi64(__U, __P); 
}

__m512i test_mm512_mask_expandloadu_epi64(__m512i __W, __mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm512_mask_expandloadu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 8>>, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s64i x 8>) -> !cir.vector<!s64i x 8>

  // LLVM-LABEL: test_mm512_mask_expandloadu_epi64
  // LLVM: @llvm.masked.expandload.v8i64(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_mask_expandloadu_epi64(__W, __U, __P); 
}

__m512i test_mm512_maskz_expandloadu_epi64(__mmask8 __U, void const *__P) {
  // CIR-LABEL: _mm512_maskz_expandloadu_epi64
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s64i x 8>>, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s64i x 8>) -> !cir.vector<!s64i x 8>

  // LLVM-LABEL: test_mm512_maskz_expandloadu_epi64
  // LLVM: @llvm.masked.expandload.v8i64(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_maskz_expandloadu_epi64(__U, __P); 
}

__m512i test_mm512_mask_expandloadu_epi32(__m512i __W, __mmask16 __U, void const *__P) {
  // CIR-LABEL: _mm512_mask_expandloadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 16>>, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!s32i x 16>) -> !cir.vector<!s32i x 16>

  // LLVM-LABEL: test_mm512_mask_expandloadu_epi32
  // LLVM: @llvm.masked.expandload.v16i32(ptr %{{.*}}, <16 x i1> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_mask_expandloadu_epi32(__W, __U, __P); 
}

__m512i test_mm512_maskz_expandloadu_epi32(__mmask16 __U, void const *__P) {
  // CIR-LABEL: _mm512_maskz_expandloadu_epi32
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s32i x 16>>, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!s32i x 16>) -> !cir.vector<!s32i x 16>

  // LLVM-LABEL: test_mm512_maskz_expandloadu_epi32
  // LLVM: @llvm.masked.expandload.v16i32(ptr %{{.*}}, <16 x i1> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_maskz_expandloadu_epi32(__U, __P); 
}
