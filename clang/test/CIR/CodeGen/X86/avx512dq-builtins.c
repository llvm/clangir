// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -fclangir -emit-cir -o %t.cir -Wall -Werror 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

#include <immintrin.h>

__m512i test_mm512_movm_epi64(__mmask8 __A) {
  // CIR-LABEL: _mm512_movm_epi64
  // CIR: %{{.*}} = cir.cast(bitcast, %{{.*}} : !u8i), !cir.vector<!cir.int<s, 1> x 8>
  // CIR: %{{.*}} = cir.cast(integral, %{{.*}} : !cir.vector<!cir.int<s, 1> x 8>), !cir.vector<!s64i x 8>
  // LLVM-LABEL: @test_mm512_movm_epi64
  // LLVM: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %{{.*}} = sext <8 x i1> %{{.*}} to <8 x i64>
  return _mm512_movm_epi64(__A); 
}
