// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2-256 -fclangir -emit-cir -o %t.cir -Wno-invalid-feature-combination -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2-256 -fclangir -emit-llvm -o %t.ll -Wno-invalid-feature-combination -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

#include <immintrin.h>

void test_mm_mask_store_sbh(void *__P, __mmask8 __U, __m128bh __A) {
  // CIR-LABEL: _mm_mask_store_sbh
  // CIR: cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.bf16 x 8>, !cir.ptr<!cir.vector<!cir.bf16 x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: @test_mm_mask_store_sbh
  // LLVM: call void @llvm.masked.store.v8bf16.p0(<8 x bfloat> %{{.*}}, ptr %{{.*}}, i32 1, <8 x i1> %{{.*}})
  _mm_mask_store_sbh(__P, __U, __A);
}

__m128bh test_mm_load_sbh(void const *A) {
  // CIR-LABEL: _mm_load_sbh
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.bf16 x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.bf16 x 8>) -> !cir.vector<!cir.bf16 x 8> 

  // LLVM-LABEL: @test_mm_load_sbh
  // NOTE: OG represents the mask using a bitcast from splat (i8 1), see IR-differences #1767
  // LLVM: %{{.*}} = call <8 x bfloat> @llvm.masked.load.v8bf16.p0(ptr %{{.*}}, i32 1,  <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, <8 x bfloat> %{{.*}})
  return _mm_load_sbh(A);
}

__m128bh test_mm_mask_load_sbh(__m128bh __A, __mmask8 __U, const void *__W) {
  // CIR-LABEL: _mm_mask_load_sbh
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.bf16 x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.bf16 x 8>) -> !cir.vector<!cir.bf16 x 8>

  // LLVM-LABEL: @test_mm_mask_load_sbh
  // LLVM: %{{.*}} = call <8 x bfloat> @llvm.masked.load.v8bf16.p0(ptr %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}})
  return _mm_mask_load_sbh(__A, __U, __W);
}
