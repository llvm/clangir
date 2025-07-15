// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -target-feature +avx512vl -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -target-feature +avx512vl -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.1-512 -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.1-512 -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s


#include <immintrin.h>

void test_mm_mask_storeu_epi16(void *__P, __mmask8 __U, __m128i __A) {
  // CIR-LABEL: _mm_mask_storeu_epi16
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s16i x 8>, !cir.ptr<!cir.vector<!s16i x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>)

  // LLVM-LABEL: @test_mm_mask_storeu_epi16
  // LLVM: @llvm.masked.store.v8i16.p0(<8 x i16> %{{.*}}, ptr %{{.*}}, i32 1, <8 x i1> %{{.*}})
  return _mm_mask_storeu_epi16(__P, __U, __A); 
}

void test_mm_mask_storeu_epi8(void *__P, __mmask16 __U, __m128i __A) {
  // CIR-LABEL: _mm_mask_storeu_epi8
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<{{!s8i|!u8i}} x 16>, !cir.ptr<!cir.vector<{{!s8i|!u8i}} x 16>>, !u32i, !cir.vector<!cir.int<s, 1> x 16>)

  // LLVM-LABEL: @test_mm_mask_storeu_epi8
  // LLVM: @llvm.masked.store.v16i8.p0(<16 x i8> %{{.*}}, ptr %{{.*}}, i32 1, <16 x i1> %{{.*}})
  return _mm_mask_storeu_epi8(__P, __U, __A); 
}

void test_mm256_mask_storeu_epi8(void *__P, __mmask32 __U, __m256i __A) {
  // CIR-LABEL: _mm256_mask_storeu_epi8
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<{{!s8i|!u8i}} x 32>, !cir.ptr<!cir.vector<{{!s8i|!u8i}} x 32>>, !u32i, !cir.vector<!cir.int<s, 1> x 32>) -> !void

  // LLVM-LABEL: @test_mm256_mask_storeu_epi8
  // LLVM: @llvm.masked.store.v32i8.p0(<32 x i8> %{{.*}}, ptr %{{.*}}, i32 1, <32 x i1> %{{.*}})
  return _mm256_mask_storeu_epi8(__P, __U, __A); 
}

void test_mm256_mask_storeu_pd(void *__P, __mmask8 __U, __m256d __A) {
  // CIR-LABEL: _mm256_mask_storeu_pd
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.double x 4>, !cir.ptr<!cir.vector<!cir.double x 4>>, !u32i, !cir.vector<!cir.int<s, 1> x 4>) -> !void

  // LLVM-LABEL: @test_mm256_mask_storeu_pd
  // LLVM: @llvm.masked.store.v4f64.p0(<4 x double> %{{.*}}, ptr %{{.*}}, i32 1, <4 x i1> %{{.*}})
  return _mm256_mask_storeu_pd(__P, __U, __A); 
}

__m128i test_mm_movm_epi8(__mmask16 __A) {
  // CIR-LABEL: _mm_movm_epi8
  // CIR: %{{.*}} = cir.cast(bitcast, %{{.*}} : !u16i), !cir.vector<!cir.int<s, 1> x 16>
  // CIR: %{{.*}} = cir.cast(integral, %{{.*}} : !cir.vector<!cir.int<s, 1> x 16>), !cir.vector<{{!s8i|!u8i}} x 16>

  // LLVM-LABEL: @test_mm_movm_epi8
  // LLVM: %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: %{{.*}} = sext <16 x i1> %{{.*}} to <16 x i8>
  return _mm_movm_epi8(__A); 
}

__m256i test_mm256_movm_epi8(__mmask32 __A) {
  // CIR-LABEL: _mm256_movm_epi8
  // CIR: %{{.*}} = cir.cast(bitcast, %{{.*}} : !u32i), !cir.vector<!cir.int<s, 1> x 32>
  // CIR: %{{.*}} = cir.cast(integral, %{{.*}} : !cir.vector<!cir.int<s, 1> x 32>), !cir.vector<{{!s8i|!u8i}} x 32>

  // LLVM-LABEL: @test_mm256_movm_epi8
  // LLVM: %{{.*}} = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: %{{.*}} = sext <32 x i1> %{{.*}} to <32 x i8>
  return _mm256_movm_epi8(__A); 
}

__m512i test_mm512_movm_epi8(__mmask64 __A) {
  // CIR-LABEL: _mm512_movm_epi8
  // CIR: %{{.*}} = cir.cast(bitcast, %{{.*}} : !u64i), !cir.vector<!cir.int<s, 1> x 64>
  // CIR: %{{.*}} = cir.cast(integral, %{{.*}} : !cir.vector<!cir.int<s, 1> x 64>), !cir.vector<{{!s8i|!u8i}} x 64>

  // LLVM-LABEL: @test_mm512_movm_epi8
  // LLVM:  %{{.*}} = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM:  %{{.*}} = sext <64 x i1> %{{.*}} to <64 x i8>
  return _mm512_movm_epi8(__A); 
}

__m128i test_mm_movm_epi16(__mmask8 __A) {
  // CIR-LABEL: _mm_movm_epi16
  // CIR: %{{.*}} = cir.cast(bitcast, %{{.*}} : !u8i), !cir.vector<!cir.int<s, 1> x 8>
  // CIR: %{{.*}} = cir.cast(integral, %{{.*}} : !cir.vector<!cir.int<s, 1> x 8>), !cir.vector<!s16i x 8>

  // LLVM-LABEL: @test_mm_movm_epi16
  // LLVM: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %{{.*}} = sext <8 x i1> %{{.*}} to <8 x i16>
  return _mm_movm_epi16(__A); 
}

__m256i test_mm256_movm_epi16(__mmask16 __A) {
  // CIR-LABEL: _mm256_movm_epi16
  // CIR: %{{.*}} = cir.cast(bitcast, %{{.*}} : !u16i), !cir.vector<!cir.int<s, 1> x 16>
  // CIR: %{{.*}} = cir.cast(integral, %{{.*}} : !cir.vector<!cir.int<s, 1> x 16>), !cir.vector<!s16i x 16>

  // LLVM-LABEL: @test_mm256_movm_epi16
  // LLVM: %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: %{{.*}} = sext <16 x i1> %{{.*}} to <16 x i16>
  return _mm256_movm_epi16(__A); 
}
