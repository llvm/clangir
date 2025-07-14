// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

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
