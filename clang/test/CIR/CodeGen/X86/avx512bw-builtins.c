// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw  -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -fno-signed-char  -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux  -target-feature +avx512bw -fno-signed-char  -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

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
