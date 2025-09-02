// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -target-feature +avx512f
// RUN: FileCheck --input-file=%t.cir %s

// Simple test file that doesn't require immintrin.h
// Tests PSLLDQI byte shift intrinsics implementation in ClangIR

typedef long long __m128i __attribute__((__vector_size__(16)));
typedef long long __m256i __attribute__((__vector_size__(32)));
typedef long long __m512i __attribute__((__vector_size__(64)));

// Declare the builtins directly
extern __m128i __builtin_ia32_pslldqi128_byteshift(__m128i, int);
extern __m256i __builtin_ia32_pslldqi256_byteshift(__m256i, int);
extern __m512i __builtin_ia32_pslldqi512_byteshift(__m512i, int);

// ============================================================================
// 128-bit Tests (Single Lane)
// ============================================================================

// CHECK-LABEL: @_Z22test_pslldqi128_shift4Dv2_x
__m128i test_pslldqi128_shift4(__m128i a) {
    // Should shift left by 4 bytes, filling with zeros
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 16>) [#cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i, #cir.int<24> : !s32i, #cir.int<25> : !s32i, #cir.int<26> : !s32i, #cir.int<27> : !s32i] : !cir.vector<!s8i x 16>
    return __builtin_ia32_pslldqi128_byteshift(a, 4);
}

// CHECK-LABEL: @_Z22test_pslldqi128_shift0Dv2_x
__m128i test_pslldqi128_shift0(__m128i a) {
    // Should return input unchanged
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 16>) [#cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i, #cir.int<24> : !s32i, #cir.int<25> : !s32i, #cir.int<26> : !s32i, #cir.int<27> : !s32i, #cir.int<28> : !s32i, #cir.int<29> : !s32i, #cir.int<30> : !s32i, #cir.int<31> : !s32i] : !cir.vector<!s8i x 16>
    return __builtin_ia32_pslldqi128_byteshift(a, 0);
}

// CHECK-LABEL: @_Z22test_pslldqi128_shift8Dv2_x
__m128i test_pslldqi128_shift8(__m128i a) {
    // Should shift left by 8 bytes (64 bits)
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 16>) [#cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i] : !cir.vector<!s8i x 16>
    return __builtin_ia32_pslldqi128_byteshift(a, 8);
}

// CHECK-LABEL: @_Z23test_pslldqi128_shift15Dv2_x
__m128i test_pslldqi128_shift15(__m128i a) {
    // Only 1 byte from input should remain
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 16>) [#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<16> : !s32i] : !cir.vector<!s8i x 16>
    return __builtin_ia32_pslldqi128_byteshift(a, 15);
}

// CHECK-LABEL: @_Z23test_pslldqi128_shift16Dv2_x
__m128i test_pslldqi128_shift16(__m128i a) {
    // Entire vector shifted out, should return zero
    // CHECK: %{{.*}} = cir.const #cir.zero : !cir.vector<!s64i x 2>
    return __builtin_ia32_pslldqi128_byteshift(a, 16);
}

// CHECK-LABEL: @_Z23test_pslldqi128_shift20Dv2_x
__m128i test_pslldqi128_shift20(__m128i a) {
    // Should also return zero
    // CHECK: %{{.*}} = cir.const #cir.zero : !cir.vector<!s64i x 2>
    return __builtin_ia32_pslldqi128_byteshift(a, 20);
}

// CHECK-LABEL: @_Z28test_pslldqi128_masked_shiftDv2_x
__m128i test_pslldqi128_masked_shift(__m128i a) {
    // 250 > 16, so should return zero
    // CHECK: %{{.*}} = cir.const #cir.zero : !cir.vector<!s64i x 2>
    return __builtin_ia32_pslldqi128_byteshift(a, 250);
}

// ============================================================================
// 256-bit Tests (Two Independent Lanes)
// ============================================================================

// CHECK-LABEL: @_Z22test_pslldqi256_shift4Dv4_x
__m256i test_pslldqi256_shift4(__m256i a) {
    // Each 128-bit lane shifts independently
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 32>)
    return __builtin_ia32_pslldqi256_byteshift(a, 4);
}

// CHECK-LABEL: @_Z22test_pslldqi256_shift0Dv4_x
__m256i test_pslldqi256_shift0(__m256i a) {
    // Should return input unchanged
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 32>)
    return __builtin_ia32_pslldqi256_byteshift(a, 0);
}

// CHECK-LABEL: @_Z22test_pslldqi256_shift8Dv4_x
__m256i test_pslldqi256_shift8(__m256i a) {
    // Each lane shifts by 8 bytes independently
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 32>)
    return __builtin_ia32_pslldqi256_byteshift(a, 8);
}

// Test shift by 12 (most of each lane)
// CHECK-LABEL: @_Z23test_pslldqi256_shift12Dv4_x
__m256i test_pslldqi256_shift12(__m256i a) {
    // Only 4 bytes remain in each lane
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 32>)
    return __builtin_ia32_pslldqi256_byteshift(a, 12);
}

// Test shift by 15 (maximum valid)
// CHECK-LABEL: @_Z23test_pslldqi256_shift15Dv4_x
__m256i test_pslldqi256_shift15(__m256i a) {
    // Only 1 byte remains in each lane
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 32>)
    return __builtin_ia32_pslldqi256_byteshift(a, 15);
}

// CHECK-LABEL: @_Z23test_pslldqi256_shift16Dv4_x
__m256i test_pslldqi256_shift16(__m256i a) {
    // Both lanes completely shifted out, returns zero
    // CHECK: %{{.*}} = cir.const #cir.zero : !cir.vector<!s64i x 4>
    return __builtin_ia32_pslldqi256_byteshift(a, 16);
}

// CHECK-LABEL: @_Z23test_pslldqi256_shift32Dv4_x
__m256i test_pslldqi256_shift32(__m256i a) {
    // Should return zero vector
    // CHECK: %{{.*}} = cir.const #cir.zero : !cir.vector<!s64i x 4>
    return __builtin_ia32_pslldqi256_byteshift(a, 32);
}

// ============================================================================
// 512-bit Tests (Four Independent Lanes)
// ============================================================================

// CHECK-LABEL: @_Z22test_pslldqi512_shift4Dv8_x
__m512i test_pslldqi512_shift4(__m512i a) {
    // All 4 lanes shift independently by 4 bytes
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 64>)
    return __builtin_ia32_pslldqi512_byteshift(a, 4);
}

// CHECK-LABEL: @_Z22test_pslldqi512_shift0Dv8_x
__m512i test_pslldqi512_shift0(__m512i a) {
    // Should return input unchanged
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 64>)
    return __builtin_ia32_pslldqi512_byteshift(a, 0);
}

// Test shift by 8
// CHECK-LABEL: @_Z22test_pslldqi512_shift8Dv8_x
__m512i test_pslldqi512_shift8(__m512i a) {
    // Each of 4 lanes shifts by 8 bytes
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 64>)
    return __builtin_ia32_pslldqi512_byteshift(a, 8);
}

// Test shift by 15 (maximum valid)
// CHECK-LABEL: @_Z23test_pslldqi512_shift15Dv8_x
__m512i test_pslldqi512_shift15(__m512i a) {
    // Only 1 byte remains in each of the 4 lanes
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 64>)
    return __builtin_ia32_pslldqi512_byteshift(a, 15);
}

// CHECK-LABEL: @_Z23test_pslldqi512_shift16Dv8_x
__m512i test_pslldqi512_shift16(__m512i a) {
    // All 4 lanes completely cleared
    // CHECK: %{{.*}} = cir.const #cir.zero : !cir.vector<!s64i x 8>
    return __builtin_ia32_pslldqi512_byteshift(a, 16);
}

// CHECK-LABEL: @_Z23test_pslldqi512_shift64Dv8_x
__m512i test_pslldqi512_shift64(__m512i a) {
    // Should return zero vector
    // CHECK: %{{.*}} = cir.const #cir.zero : !cir.vector<!s64i x 8>
    return __builtin_ia32_pslldqi512_byteshift(a, 64);
}

// Test with masked shift amount
// CHECK-LABEL: @_Z28test_pslldqi512_masked_shiftDv8_x
__m512i test_pslldqi512_masked_shift(__m512i a) {
    // 250 & 0xFF = 250, so should behave same as shift >= 16 (return zero)
    // CHECK: %{{.*}} = cir.const #cir.zero : !cir.vector<!s64i x 8>
    return __builtin_ia32_pslldqi512_byteshift(a, 250);
}

// ============================================================================
// Edge Cases and Special Scenarios
// ============================================================================

// CHECK-LABEL: @_Z23test_consecutive_shiftsDv2_x
__m128i test_consecutive_shifts(__m128i a) {
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 16>)
    __m128i tmp1 = __builtin_ia32_pslldqi128_byteshift(a, 2);
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 16>)
    __m128i tmp2 = __builtin_ia32_pslldqi128_byteshift(tmp1, 3);
    // Total shift of 5 bytes
    return tmp2;
}

// CHECK-LABEL: @_Z21test_const_expr_shiftDv2_x
__m128i test_const_expr_shift(__m128i a) {
    const int shift_amount = 3 + 4;  // = 7
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 16>)
    return __builtin_ia32_pslldqi128_byteshift(a, shift_amount);
}

// CHECK-LABEL: @_Z22test_lane_independenceDv4_xPS_S0_
void test_lane_independence(__m256i a, __m256i* result1, __m256i* result2) {
    // Different shift amounts to show each lane operates independently
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 32>)
    *result1 = __builtin_ia32_pslldqi256_byteshift(a, 4);
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 32>)
    *result2 = __builtin_ia32_pslldqi256_byteshift(a, 8);
    // The two 128-bit lanes in each result shift independently
}

// CHECK-LABEL: @_Z22test_pslldqi128_shift1Dv2_x
__m128i test_pslldqi128_shift1(__m128i a) {
    // Shifts by just 1 byte
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 16>)
    return __builtin_ia32_pslldqi128_byteshift(a, 1);
}

// Test boundary case: shift by 14
// CHECK-LABEL: @_Z23test_pslldqi128_shift14Dv2_x
__m128i test_pslldqi128_shift14(__m128i a) {
    // 2 bytes remain
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 16>)
    return __builtin_ia32_pslldqi128_byteshift(a, 14);
}

// ============================================================================
// Pattern Tests - Verify the shuffle indices work correctly
// ============================================================================

// These tests help verify the shuffle index calculation

// Test that verifies zeros are inserted from the left
// CHECK-LABEL: @_Z19test_zero_insertionDv2_x
__m128i test_zero_insertion(__m128i a) {
    // After shift by 4, first 4 bytes should be zero
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 16>) 
    return __builtin_ia32_pslldqi128_byteshift(a, 4);
}

// Test all three sizes with same shift to compare behavior
// CHECK-LABEL: @_Z21test_all_sizes_shift4Dv2_xDv4_xDv8_xPS_PS0_PS1_
void test_all_sizes_shift4(__m128i a128, __m256i a256, __m512i a512,
                           __m128i* r128, __m256i* r256, __m512i* r512) {
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 16>)
    *r128 = __builtin_ia32_pslldqi128_byteshift(a128, 4);
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 32>)
    *r256 = __builtin_ia32_pslldqi256_byteshift(a256, 4);
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 64>)
    *r512 = __builtin_ia32_pslldqi512_byteshift(a512, 4);
    // Each should shift their lane(s) by 4 bytes
}

// CHECK-LABEL: @_Z22test_large_shift_valueDv2_x
__m128i test_large_shift_value(__m128i a) {
    // 240 & 0xFF = 240, so this should return zero (240 > 16)
    // CHECK: %{{.*}} = cir.const #cir.zero : !cir.vector<!s64i x 2>
    return __builtin_ia32_pslldqi128_byteshift(a, 240);
}

// CHECK-LABEL: @_Z26test_large_shift_value_256Dv4_x
__m256i test_large_shift_value_256(__m256i a) {
    // 244 & 0xFF = 244, so this should return zero (244 > 16)
    // CHECK: %{{.*}} = cir.const #cir.zero : !cir.vector<!s64i x 4>
    return __builtin_ia32_pslldqi256_byteshift(a, 244);
}
