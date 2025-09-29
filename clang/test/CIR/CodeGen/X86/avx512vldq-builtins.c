// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512dq -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG

#include <immintrin.h>

__m128i test_mm_movm_epi32(__mmask8 __A) {
  // CIR-LABEL: _mm_movm_epi32
  // CIR: %{{.*}} = cir.cast(bitcast, %{{.*}} : !u8i), !cir.vector<!cir.int<s, 1> x 8>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.int<s, 1> x 8>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<!cir.int<s, 1> x 4>
  // CIR: %{{.*}} = cir.cast(integral, %{{.*}} : !cir.vector<!cir.int<s, 1> x 4>), !cir.vector<!s32i x 4>

  // LLVM-LABEL: @test_mm_movm_epi32
  // LLVM: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: %{{.*}} = sext <4 x i1> %{{.*}} to <4 x i32>
  return _mm_movm_epi32(__A); 
}

__m256i test_mm256_movm_epi32(__mmask8 __A) {
  // CIR-LABEL: _mm256_movm_epi32
  // CIR: %{{.*}} = cir.cast(bitcast, %{{.*}} : !u8i), !cir.vector<!cir.int<s, 1> x 8>
  // CIR: %{{.*}} = cir.cast(integral, %{{.*}} : !cir.vector<!cir.int<s, 1> x 8>), !cir.vector<!s32i x 8>

  // LLVM-LABEL: @test_mm256_movm_epi32
  // LLVM: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %{{.*}} = sext <8 x i1> %{{.*}} to <8 x i32>
  return _mm256_movm_epi32(__A); 
}

__m512i test_mm512_movm_epi32(__mmask16 __A) {
  // CIR-LABEL: _mm512_movm_epi32
  // CIR: %{{.*}} = cir.cast(bitcast, %{{.*}} : !u16i), !cir.vector<!cir.int<s, 1> x 16>
  // CIR: %{{.*}} = cir.cast(integral, %{{.*}} : !cir.vector<!cir.int<s, 1> x 16>), !cir.vector<!s32i x 16>

  // LLVM-LABEL: @test_mm512_movm_epi32
  // LLVM: %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: %{{.*}} = sext <16 x i1> %{{.*}} to <16 x i32>
  return _mm512_movm_epi32(__A); 
}

__m128i test_mm_movm_epi64(__mmask8 __A) {
  // CIR-LABEL: _mm_movm_epi64
  // CIR: %{{.*}} = cir.cast(bitcast, %{{.*}} : !u8i), !cir.vector<!cir.int<s, 1> x 8>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.int<s, 1> x 8>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<!cir.int<s, 1> x 2>
  // CIR: %{{.*}} = cir.cast(integral, %{{.*}} : !cir.vector<!cir.int<s, 1> x 2>), !cir.vector<!s64i x 2>

  // LLVM-LABEL: @test_mm_movm_epi64
  // LLVM: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // LLVM: %{{.*}} = sext <2 x i1> %{{.*}} to <2 x i64>
  return _mm_movm_epi64(__A); 
}

__m256i test_mm256_movm_epi64(__mmask8 __A) {
  // CIR-LABEL: _mm256_movm_epi64
  // CIR: %{{.*}} = cir.cast(bitcast, %{{.*}} : !u8i), !cir.vector<!cir.int<s, 1> x 8>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.int<s, 1> x 8>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<!cir.int<s, 1> x 4>
  // CIR: %{{.*}} = cir.cast(integral, %{{.*}} : !cir.vector<!cir.int<s, 1> x 4>), !cir.vector<!s64i x 4>

  // LLVM-LABEL: @test_mm256_movm_epi64
  // LLVM: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: %{{.*}} = sext <4 x i1> %{{.*}} to <4 x i64>
  return _mm256_movm_epi64(__A); 
}

__m256d test_mm256_insertf64x2(__m256d __A, __m128d __B) {
  // CIR-LABEL: test_mm256_insertf64x2
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.double x 4>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i] : !cir.vector<!cir.double x 4>

  // LLVM-LABEL: @test_mm256_insertf64x2
  // LLVM: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  return _mm256_insertf64x2(__A, __B, 1); 
}

__m256i test_mm256_inserti64x2(__m256i __A, __m128i __B) {
  // CIR-LABEL: test_mm256_inserti64x2
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s64i x 4>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i] : !cir.vector<!s64i x 4>

  // LLVM-LABEL: @test_mm256_inserti64x2
  // LLVM: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  return _mm256_inserti64x2(__A, __B, 1); 
}

__mmask8 test_mm256_movepi32_mask(__m256i __A) {
  // LLVM-LABEL: @test_mm256_movepi32_mask
  // LLVM: [[CMP:%.*]] = icmp slt <8 x i32> %{{.*}}, zeroinitializer

  // OGCG-LABEL: @test_mm256_movepi32_mask
  // OGCG: [[CMP:%.*]] = icmp slt <8 x i32> %{{.*}}, zeroinitializer
  return _mm256_movepi32_mask(__A); 
}

__mmask8 test_mm_movepi64_mask(__m128i __A) {
  // CIR-LABEL: _mm_movepi64_mask
  // CIR: %{{.*}} = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<!s64i x 2>, !cir.vector<!cir.int<u, 1> x 2>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.int<u, 1> x 2>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<!cir.int<u, 1> x 8>
  // CIR: %{{.*}} = cir.cast(bitcast, %{{.*}} : !cir.vector<!cir.int<u, 1> x 8>), !u8i

  // LLVM-LABEL: @test_mm_movepi64_mask
  // LLVM: [[CMP:%.*]] = icmp slt <2 x i64> %{{.*}}, zeroinitializer
  // LLVM: [[SHUF:%.*]] = shufflevector <2 x i1> [[CMP]], <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>

  // OGCG-LABEL: @test_mm_movepi64_mask
  // OGCG: [[CMP:%.*]] = icmp slt <2 x i64> %{{.*}}, zeroinitializer
  // OGCG: [[SHUF:%.*]] = shufflevector <2 x i1> [[CMP]], <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  return _mm_movepi64_mask(__A); 
}

__mmask8 test_mm256_movepi64_mask(__m256i __A) {
  // CIR-LABEL: _mm256_movepi64_mask
  // CIR: %{{.*}} = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<!s64i x 4>, !cir.vector<!cir.int<u, 1> x 4>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.int<u, 1> x 4>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<!cir.int<u, 1> x 8>

  // LLVM-LABEL: @test_mm256_movepi64_mask
  // LLVM: [[CMP:%.*]] = icmp slt <4 x i64> %{{.*}}, zeroinitializer
  // LLVM: [[SHUF:%.*]] = shufflevector <4 x i1> [[CMP]], <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>

  // OGCG-LABEL: @test_mm256_movepi64_mask
  // OGCG: [[CMP:%.*]] = icmp slt <4 x i64> %{{.*}}, zeroinitializer
  // OGCG: [[SHUF:%.*]] = shufflevector <4 x i1> [[CMP]], <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm256_movepi64_mask(__A); 
}
