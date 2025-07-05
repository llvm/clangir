// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

#include <immintrin.h>

void test_mm512_mask_storeu_epi64(void *__P, __mmask8 __U, __m512i __A) {
  // CIR-LABEL: _mm512_mask_storeu_epi64
  // LLVM-LABEL: test_mm512_mask_storeu_epi64
  // TBA
  // LLVM: @llvm.masked.store.v8i64.p0(<8 x i64> %{{.*}}, ptr %{{.*}}, i32 1, <8 x i1> %{{.*}})
  return _mm512_mask_storeu_epi64(__P, __U, __A); 
}

void test_mm512_mask_storeu_epi32(void *__P, __mmask16 __U, __m512i __A) {
  
  // CIR-LABEL: _mm512_mask_storeu_epi32
  // LLVM-LABEL: test_mm512_mask_storeu_epi32
  // TBA
  // LLVM: @llvm.masked.store.v16i32.p0(<16 x i32> %{{.*}}, ptr %{{.*}}, i32 1, <16 x i1> %{{.*}})
  return _mm512_mask_storeu_epi32(__P, __U, __A); 
}
