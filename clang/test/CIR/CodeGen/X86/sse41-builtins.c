// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR-CHECK --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR-CHECK --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefix=LLVM-CHECK --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefix=LLVM-CHECK --input-file=%t.ll %s


#include <immintrin.h>

int test_mm_extract_epi8(__m128i x) {
  // CIR-CHECK-LABEL: test_mm_extract_epi8
  // CIR-CHECK %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : {{!u32i|!u64i}}] : !cir.vector<!s8i x 16>
  // CIR-CHECK %{{.*}} = cir.cast(integral, %{{.*}} : !u8i), !s32i

  // LLVM-CHECK-LABEL: test_mm_extract_epi8
  // LLVM-CHECK: extractelement <16 x i8> %{{.*}}, {{i32|i64}} 1
  // LLVM-CHECK: zext i8 %{{.*}} to i32
  return _mm_extract_epi8(x, 1);
}

int test_mm_extract_epi32(__m128i x) {
  // CIR-CHECK-LABEL: test_mm_extract_epi32
  // CIR-CHECK %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : {{!u32i|!u64i}}] : !cir.vector<!s32i x 4>

  // LLVM-CHECK-LABEL: test_mm_extract_epi32
  // LLVM-CHECK: extractelement <4 x i32> %{{.*}}, {{i32|i64}} 1
  return _mm_extract_epi32(x, 1);
}

long long test_mm_extract_epi64(__m128i x) {
  // CIR-CHECK-LABEL: test_mm_extract_epi64
  // CIR-CHECK %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : {{!u32i|!u64i}}] : !cir.vector<!s64i x 2>

  // LLVM-CHECK-LABEL: test_mm_extract_epi64
  // LLVM-CHECK: extractelement <2 x i64> %{{.*}}, {{i32|i64}} 1
  return _mm_extract_epi64(x, 1);
}

int test_mm_extract_ps(__m128 x) {
  // CIR-CHECK-LABEL: test_mm_extract_ps
  // CIR-CHECK %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : {{!u32i|!u64i}}] : !cir.vector<!cir.float x 4>

  // LLVM-CHECK-LABEL: test_mm_extract_ps
  // LLVM-CHECK: extractelement <4 x float> %{{.*}}, {{i32|i64}} 1
  return _mm_extract_ps(x, 1);
}

__m128i test_mm_insert_epi8(__m128i x, char b) {

  // CIR-CHECK-LABEL: test_mm_insert_epi8
  // CIR-CHECK-LABEL: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[{{%.*}} : {{!u32i|!u64i}}] : !cir.vector<{{!s8i|!u8i}} x 16>

  // LLVM-CHECK-LABEL: test_mm_insert_epi8 
  // LLVM-CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, {{i32|i64}} 1
  return _mm_insert_epi8(x, b, 1);
}

__m128i test_mm_insert_epi32(__m128i x, int b) {

  // CIR-CHECK-LABEL: test_mm_insert_epi32
  // CIR-CHECK-LABEL: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[{{%.*}} : {{!u32i|!u64i}}] : !cir.vector<!s32i x 4>

  // LLVM-CHECK-LABEL: test_mm_insert_epi32
  // LLVM-CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, {{i32|i64}} 1
  return _mm_insert_epi32(x, b, 1);
}

#ifdef __x86_64__
__m128i test_mm_insert_epi64(__m128i x, long long b) {

  // CIR-X64-LABEL: test_mm_insert_epi64
  // CIR-X64: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[{{%.*}} : {{!u32i|!u64i}}] : !cir.vector<!s64i x 2>

  // LLVM-X64-LABEL: test_mm_insert_epi64
  // LLVM-X64: insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, {{i32|i64}} 1
  return _mm_insert_epi64(x, b, 1);
}
#endif
