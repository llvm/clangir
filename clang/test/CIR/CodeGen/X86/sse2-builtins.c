// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK --input-file=%t.cir %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK --input-file=%t.ll %s

// This test mimics clang/test/CodeGen/X86/sse2-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

void test_mm_clflush(void* A) {
  // CIR-LABEL: test_mm_clflush
  // LLVM-LABEL: test_mm_clflush
  _mm_clflush(A);
  // CIR-CHECK: {{%.*}} = cir.llvm.intrinsic "x86.sse2.clflush" {{%.*}} : (!cir.ptr<!void>) -> !void
  // LLVM-CHECK: call void @llvm.x86.sse2.clflush(ptr {{%.*}})
}

// Lowering to pextrw requires optimization.
int test_mm_extract_epi16(__m128i A) {
    
  // CIR-CHECK-LABEL: test_mm_extract_epi16
  // CIR-CHECK %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : {{!u32i|!u64i}}] : !cir.vector<!s16i x 8>
  // CIR-CHECK %{{.*}} = cir.cast(integral, %{{.*}} : !u16i), !s32i

  // LLVM-CHECK-LABEL: test_mm_extract_epi16
  // LLVM-CHECK: extractelement <8 x i16> %{{.*}}, {{i32|i64}} 1
  // LLVM-CHECK: zext i16 %{{.*}} to i32
  return _mm_extract_epi16(A, 1);
}

void test_mm_lfence(void) {
  // CIR-CHECK-LABEL: test_mm_lfence
  // LLVM-CHECK-LABEL: test_mm_lfence
  _mm_lfence();
  // CIR-CHECK: {{%.*}} = cir.llvm.intrinsic "x86.sse2.lfence" : () -> !void
  // LLVM-CHECK: call void @llvm.x86.sse2.lfence()
}

void test_mm_mfence(void) {
  // CIR-CHECK-LABEL: test_mm_mfence
  // LLVM-CHECK-LABEL: test_mm_mfence
  _mm_mfence();
  // CIR-CHECK: {{%.*}} = cir.llvm.intrinsic "x86.sse2.mfence" : () -> !void
  // LLVM-CHECK: call void @llvm.x86.sse2.mfence()
}
