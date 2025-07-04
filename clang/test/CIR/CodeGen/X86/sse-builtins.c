// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// This test mimics clang/test/CodeGen/X86/sse-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>


void test_mm_prefetch(char const* p) {
  // CIR-LABEL: test_mm_prefetch
  // LLVM-LABEL: test_mm_prefetch
  _mm_prefetch(p, 0);
  // CIR: cir.prefetch(%{{.*}} : !cir.ptr<!void>) locality(0) read
  // LLVM: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 0, i32 1)
}

void test_mm_sfence(void) {
  // CIR-LABEL: test_mm_sfence
  // LLVM-LABEL: test_mm_sfence
  _mm_sfence();
  // CIR: {{%.*}} = cir.llvm.intrinsic "x86.sse.sfence" : () -> !void
  // LLVM: call void @llvm.x86.sse.sfence()
}
