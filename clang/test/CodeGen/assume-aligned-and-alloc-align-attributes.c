// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

void *my_aligned_alloc(int size, int alignment) __attribute__((assume_aligned(32), alloc_align(2)));

// CHECK-LABEL: @t0_immediate0(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CALL:%.*]] = call align 32 ptr @my_aligned_alloc(i32 noundef 320, i32 noundef 16)
// CHECK-NEXT:    ret ptr [[CALL]]
//
void *t0_immediate0(void) {
  return my_aligned_alloc(320, 16);
};

// CHECK-LABEL: @t1_immediate1(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CALL:%.*]] = call align 32 ptr @my_aligned_alloc(i32 noundef 320, i32 noundef 32)
// CHECK-NEXT:    ret ptr [[CALL]]
//
void *t1_immediate1(void) {
  return my_aligned_alloc(320, 32);
};

// CHECK-LABEL: @t2_immediate2(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CALL:%.*]] = call align 64 ptr @my_aligned_alloc(i32 noundef 320, i32 noundef 64)
// CHECK-NEXT:    ret ptr [[CALL]]
//
void *t2_immediate2(void) {
  return my_aligned_alloc(320, 64);
};

// CHECK-LABEL: @t3_variable(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[ALIGNMENT_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store i32 [[ALIGNMENT:%.*]], ptr [[ALIGNMENT_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[ALIGNMENT_ADDR]], align 4
// CHECK-NEXT:    [[CALL:%.*]] = call align 32 ptr @my_aligned_alloc(i32 noundef 320, i32 noundef [[TMP0]])
// CHECK-NEXT:    [[CASTED_ALIGN:%.*]] = zext i32 [[TMP0]] to i64
// CHECK-NEXT:    call void @llvm.assume(i1 true) [ "align"(ptr [[CALL]], i64 [[CASTED_ALIGN]]) ]
// CHECK-NEXT:    ret ptr [[CALL]]
//
void *t3_variable(int alignment) {
  return my_aligned_alloc(320, alignment);
};
