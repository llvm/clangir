// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK,CIR-X64 --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK,CIR-X64 --input-file=%t.cir %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK,LLVM-X64 --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK,LLVM-X64 --input-file=%t.ll %s

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

__m128d test_mm_undefined_pd(void) {
  // CIR-X64-LABEL: _mm_undefined_pd
  // CIR-X64: %{{.*}} = cir.const #cir.zero : !cir.vector<!cir.double x 2>
  // CIR-X64: cir.return %{{.*}} : !cir.vector<!cir.double x 2>

  // LLVM-X64-LABEL: test_mm_undefined_pd
  // LLVM-X64: store <2 x double> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM-X64: %{{.*}} = load <2 x double>, ptr %[[A]], align 16
  // LLVM-X64: ret <2 x double> %{{.*}}
  return _mm_undefined_pd();
}

__m128i test_mm_undefined_si128(void) {
  // CIR-LABEL: _mm_undefined_si128
  // CIR-CHECK: %[[A:.*]] = cir.const #cir.zero : !cir.vector<!cir.double x 2>
  // CIR-CHECK: %{{.*}} = cir.cast(bitcast, %[[A]] : !cir.vector<!cir.double x 2>), !cir.vector<!s64i x 2>
  // CIR-CHECK: cir.return %{{.*}} : !cir.vector<!s64i x 2>

  // LLVM-CHECK-LABEL: test_mm_undefined_si128
  // LLVM-CHECK: store <2 x i64> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM-CHECK: %{{.*}} = load <2 x i64>, ptr %[[A]], align 16
  // LLVM-CHECK: ret <2 x i64> %{{.*}}
  return _mm_undefined_si128();
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
