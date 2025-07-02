// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK,CIR-X64 --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK,CIR-X64 --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK,LLVM-X64 --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK,LLVM-X64 --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK,CIR-X64 --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK,CIR-X64 --input-file=%t.cir %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK,LLVM-X64 --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK,LLVM-X64 --input-file=%t.ll %s

#include <immintrin.h>

int test_mm256_extract_epi8(__m256i A) {
  // CIR-CHECK-LABEL: test_mm256_extract_epi8
  // CIR-CHECK %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : {{!u32i|!u64i}}] : !cir.vector<!s8i x 32>
  // CIR-CHECK %{{.*}} = cir.cast(integral, %{{.*}} : !u8i), !s32i

  // LLVM-CHECK-LABEL: test_mm256_extract_epi8
  // LLVM-CHECK: extractelement <32 x i8> %{{.*}}, {{i32|i64}} 31
  // LLVM-CHECK: zext i8 %{{.*}} to i32
  return _mm256_extract_epi8(A, 31);
}

int test_mm256_extract_epi16(__m256i A) {
  // CIR-CHECK-LABEL: test_mm256_extract_epi16
  // CIR-CHECK %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : {{!u32i|!u64i}}] : !cir.vector<!s16i x 16>
  // CIR-CHECK %{{.*}} = cir.cast(integral, %{{.*}} : !u16i), !s32i

  // LLVM-CHECK-LABEL: test_mm256_extract_epi16
  // LLVM-CHECK: extractelement <16 x i16> %{{.*}}, {{i32|i64}} 15
  // LLVM-CHECK: zext i16 %{{.*}} to i32
  return _mm256_extract_epi16(A, 15);
}

int test_mm256_extract_epi32(__m256i A) {
  // CIR-CHECK-LABEL: test_mm256_extract_epi32
  // CIR-CHECK %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : {{!u32i|!u64i}}] : !cir.vector<!s32i x 8>

  // LLVM-CHECK-LABEL: test_mm256_extract_epi32
  // LLVM-CHECK: extractelement <8 x i32> %{{.*}}, {{i32|i64}} 7
  return _mm256_extract_epi32(A, 7);
}

#if __x86_64__
long long test_mm256_extract_epi64(__m256i A) {
  // CIR-X64-LABEL: test_mm256_extract_epi64
  // LLVM-X64-LABEL: test_mm256_extract_epi64
  return _mm256_extract_epi64(A, 3);
}
#endif
