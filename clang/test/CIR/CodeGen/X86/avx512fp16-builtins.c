// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512fp16 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512fp16 -fclangir -emit-llvm -o %t.ll  -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s


#include <immintrin.h>

void test_mm_mask_store_sh(void *__P, __mmask8 __U, __m128h __A) {
  // CIR-LABEL: _mm_mask_store_sh
  // CIR: cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.f16 x 8>, !cir.ptr<!cir.vector<!cir.f16 x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: @test_mm_mask_store_sh
  // LLVM: call void @llvm.masked.store.v8f16.p0(<8 x half> %{{.*}}, ptr %{{.*}}, i32 1, <8 x i1> %{{.*}})
  _mm_mask_store_sh(__P, __U, __A);
}
