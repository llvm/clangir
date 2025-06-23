// RUN: %clang_cc1 -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// This test mimics clang/test/CodeGen/builtins-x86.c, which eventually
// CIR shall be able to support fully.

void test_mm_clflush(const void* tmp_vCp) {
  // CIR-LABEL: test_mm_clflush
  // LLVM-LABEL: test_mm_clflush
  _mm_clflush(tmp_vCp);
  // CIR: {{%.*}} = cir.llvm.intrinsic "x86.sse2.clflush" {{%.*}} : (!cir.ptr<!void>) -> !void
  // LLVM: call void @llvm.x86.sse2.clflush(ptr {{%.*}})
}

void test_mm_lfence() {
  // CIR-LABEL: test_mm_lfence
  // LLVM-LABEL: test_mm_lfence
  _mm_lfence();
  // CIR: {{%.*}} = cir.llvm.intrinsic "x86.sse2.lfence" : () -> !void
  // LLVM: call void @llvm.x86.sse2.lfence()
}

void test_mm_pause() {
  // CIR-LABEL: test_mm_pause
  // LLVM-LABEL: test_mm_pause
  _mm_pause();
  // CIR: {{%.*}} = cir.llvm.intrinsic "x86.sse2.pause" : () -> !void
  // LLVM: call void @llvm.x86.sse2.pause()
}

void test_mm_mfence() {
  // CIR-LABEL: test_mm_mfence
  // LLVM-LABEL: test_mm_mfence
  _mm_mfence();
  // CIR: {{%.*}} = cir.llvm.intrinsic "x86.sse2.mfence" : () -> !void
  // LLVM: call void @llvm.x86.sse2.mfence()
}

void test_mm_sfence() {
  // CIR-LABEL: test_mm_sfence
  // LLVM-LABEL: test_mm_sfence
  _mm_sfence();
  // CIR: {{%.*}} = cir.llvm.intrinsic "x86.sse.sfence" : () -> !void
  // LLVM: call void @llvm.x86.sse.sfence()
}
