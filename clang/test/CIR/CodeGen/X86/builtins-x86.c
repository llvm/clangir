// Global variables of intergal types
// RUN: %clang_cc1 -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
# include <emmintrin.h> 
void test_mm_clflush(const void* tmp_vCp) {
  // CIR-LABEL: test_mm_clflush
  // LLVM-LABEL: test_mm_clflush
  _mm_clflush(tmp_vCp);
  // CIR: {{%.*}} = cir.llvm.intrinsic "x86.sse2.clflush" {{%.*}} : (!cir.ptr<!void>) -> !void
  // LLVM: call void @llvm.x86.sse2.clflush(ptr {{%.*}})
}

void test_mm_prefetch(char const* p) {
 // CIR-LABEL: test_mm_prefetch
  // LLVM-LABEL: test_mm_prefetch
  _mm_prefetch(p, 0);
  // CIR: {{%.}} = cir.llvm.intrinsic "llvm.prefetch.p0" {{%.}}, {{%.}} : (!cir.ptr<!char>, !cir.i32) -> !void
  // LLVM: call void @llvm.prefetch.p0(ptr {{.}}, i32 0, i32 0, i32 1)
}