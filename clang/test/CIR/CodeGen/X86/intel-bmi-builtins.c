// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Intel and AMD use different names for the same BMI intrinsics:
// Intel uses single underscores (e.g. _tzcnt_u16),
// AMD uses double underscores (e.g. __tzcnt_u16).
// Unlike the traditinal tests in clang/test/CodeGen/X86/bmi-builtins.c
// which combines both, we split them into separate files to avoid symbol 
// conflicts and keep tests isolated.

#include <immintrin.h>

unsigned short test_tzcnt_u16(unsigned short __X) {
  // CIR-LABEL: _tzcnt_u16
  // LLVM-LABEL: _tzcnt_u16
  return _tzcnt_u16(__X);
  // CIR: {{%.*}} = cir.llvm.intrinsic "cttz" {{%.*}} : (!u16i, !cir.bool) -> !u16i
  // LLVM: i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
}

unsigned int test_tzcnt_u32(unsigned int __X) {
  // CIR-LABEL: _tzcnt_u32
  // LLVM-LABEL: _tzcnt_u32
  return _tzcnt_u32(__X);
  // CIR: {{%.*}} = cir.llvm.intrinsic "cttz" {{%.*}} : (!u32i, !cir.bool) -> !u32i
  // LLVM: i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
}

int test_mm_tzcnt_32(unsigned int __X) {
  // CIR-LABEL: _mm_tzcnt_32
  // LLVM-LABEL: _mm_tzcnt_32
  return _mm_tzcnt_32(__X);
  // CIR: {{%.*}} = cir.llvm.intrinsic "cttz" {{%.*}} : (!u32i, !cir.bool) -> !u32i
  // LLVM: i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
}

#ifdef __x86_64__
unsigned long long test_tzcnt_u64(unsigned long long __X) {
  // CIR-LABEL: _tzcnt_u64
  // LLVM-LABEL: _tzcnt_u64
  return _tzcnt_u64(__X);
  // CIR: {{%.*}} = cir.llvm.intrinsic "cttz" {{%.*}} : (!u64i, !cir.bool) -> !u64i
  // LLVM: i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
}

long long test_mm_tzcnt_64(unsigned long long __X) {
  // CIR-LABEL: _mm_tzcnt_64
  // LLVM-LABEL: _mm_tzcnt_64
  return _mm_tzcnt_64(__X);
  // CIR: {{%.*}} = cir.llvm.intrinsic "cttz" {{%.*}} : (!u64i, !cir.bool) -> !u64i
  // LLVM: i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
}
#endif