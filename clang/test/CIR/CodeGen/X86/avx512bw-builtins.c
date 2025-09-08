// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw  -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -fno-signed-char  -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux  -target-feature +avx512bw -fno-signed-char  -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM-UNSIGNED-CHAR --input-file=%t.ll %s

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefix=OGCG
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -fno-signed-char -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefix=OGCG

#include <immintrin.h>

void test_mm512_mask_storeu_epi16(void *__P, __mmask32 __U, __m512i __A) {
  // CIR-LABEL: _mm512_mask_storeu_epi16
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s16i x 32>, !cir.ptr<!cir.vector<!s16i x 32>>, !u32i, !cir.vector<!cir.int<s, 1> x 32>) -> !void

  // LLVM-LABEL: @test_mm512_mask_storeu_epi16
  // LLVM: @llvm.masked.store.v32i16.p0(<32 x i16> %{{.*}}, ptr %{{.*}}, i32 1, <32 x i1> %{{.*}})
  return _mm512_mask_storeu_epi16(__P, __U, __A);
}

void test_mm512_mask_storeu_epi8(void *__P, __mmask64 __U, __m512i __A) {
  // CIR-LABEL: _mm512_mask_storeu_epi8
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<{{!s8i|!u8i}} x 64>, !cir.ptr<!cir.vector<{{!s8i|!u8i}} x 64>>, !u32i, !cir.vector<!cir.int<s, 1> x 64>) -> !void

  // LLVM-LABEL: @test_mm512_mask_storeu_epi8
  // LLVM: @llvm.masked.store.v64i8.p0(<64 x i8> %{{.*}}, ptr %{{.*}}, i32 1, <64 x i1> %{{.*}})
  return _mm512_mask_storeu_epi8(__P, __U, __A); 
}

__m512i test_mm512_movm_epi16(__mmask32 __A) {
  // CIR-LABEL: _mm512_movm_epi16
  // CIR: %{{.*}} = cir.cast(bitcast, %{{.*}} : !u32i), !cir.vector<!cir.int<s, 1> x 32>
  // CIR: %{{.*}} = cir.cast(integral, %{{.*}} : !cir.vector<!cir.int<s, 1> x 32>), !cir.vector<!s16i x 32>
  // LLVM-LABEL: @test_mm512_movm_epi16
  // LLVM:  %{{.*}} = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM:  %{{.*}} = sext <32 x i1> %{{.*}} to <32 x i16>
  return _mm512_movm_epi16(__A); 
}

__m512i test_mm512_mask_loadu_epi8(__m512i __W, __mmask64 __U, void const *__P) {
  // CIR-LABEL: _mm512_mask_loadu_epi8
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<{{!s8i|!u8i}} x 64>>, !u32i, !cir.vector<!cir.int<s, 1> x 64>, !cir.vector<{{!s8i|!u8i}} x 64>) -> !cir.vector<{{!s8i|!u8i}} x 64>

  // LLVM-LABEL: @test_mm512_mask_loadu_epi8
  // LLVM: @llvm.masked.load.v64i8.p0(ptr %{{.*}}, i32 1, <64 x i1> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_mask_loadu_epi8(__W, __U, __P); 
}

__m512i test_mm512_mask_loadu_epi16(__m512i __W, __mmask32 __U, void const *__P) {
  // CIR-LABEL: _mm512_mask_loadu_epi16
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s16i x 32>>, !u32i, !cir.vector<!cir.int<s, 1> x 32>, !cir.vector<!s16i x 32>) -> !cir.vector<!s16i x 32>

  // LLVM-LABEL: @test_mm512_mask_loadu_epi16
  // LLVM: @llvm.masked.load.v32i16.p0(ptr %{{.*}}, i32 1, <32 x i1> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_mask_loadu_epi16(__W, __U, __P); 
}

__m512i test_mm512_maskz_loadu_epi16(__mmask32 __U, void const *__P) {
  // CIR-LABEL: _mm512_maskz_loadu_epi16
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s16i x 32>>, !u32i, !cir.vector<!cir.int<s, 1> x 32>, !cir.vector<!s16i x 32>) -> !cir.vector<!s16i x 32>

  // LLVM-LABEL: @test_mm512_maskz_loadu_epi16
  // LLVM: @llvm.masked.load.v32i16.p0(ptr %{{.*}}, i32 1, <32 x i1> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_maskz_loadu_epi16(__U, __P); 
}

__m512i test_mm512_maskz_loadu_epi8(__mmask64 __U, void const *__P) {
  // CIR-LABEL: _mm512_maskz_loadu_epi8
  // CIR: cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<{{!s8i|!u8i}} x 64>>, !u32i, !cir.vector<!cir.int<s, 1> x 64>, !cir.vector<{{!s8i|!u8i}} x 64>) -> !cir.vector<{{!s8i|!u8i}} x 64>

  // LLVM-LABEL: @test_mm512_maskz_loadu_epi8
  // LLVM: @llvm.masked.load.v64i8.p0(ptr %{{.*}}, i32 1, <64 x i1> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_maskz_loadu_epi8(__U, __P); 
}

__mmask64 test_mm512_movepi8_mask(__m512i __A) {
  // CIR-LABEL: @_mm512_movepi8_mask
  // CIR: %{{.*}} = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<{{!s8i|!u8i}} x 64>, !cir.vector<!cir.int<u, 1> x 64>

  // LLVM-LABEL: @test_mm512_movepi8_mask
  // LLVM: [[CMP:%.*]] = icmp slt <64 x i8> %{{.*}}, zeroinitializer

  // In the unsigned case below, the canonicalizer proves the comparison is
  // always false (no i8 unsigned value can be < 0) and folds it away.
  // LLVM-UNSIGNED-CHAR: store i64 0, ptr %{{.*}}, align 8
  
  // OGCG-LABEL: @test_mm512_movepi8_mask
  // OGCG: [[CMP:%.*]] = icmp slt <64 x i8> %{{.*}}, zeroinitializer
  return _mm512_movepi8_mask(__A); 
}

__mmask32 test_mm512_movepi16_mask(__m512i __A) {
  // CIR-LABEL: @_mm512_movepi16_mask
  // CIR: %{{.*}} = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<!s16i x 32>, !cir.vector<!cir.int<u, 1> x 32>

  // LLVM-LABEL: @test_mm512_movepi16_mask
  // LLVM: [[CMP:%.*]] = icmp slt <32 x i16> %{{.*}}, zeroinitializer

  // OGCG-LABEL: @test_mm512_movepi16_mask
  // OGCG: [[CMP:%.*]] = icmp slt <32 x i16> %{{.*}}, zeroinitializer
  return _mm512_movepi16_mask(__A); 
}
