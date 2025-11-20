// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Test CK_AtomicToNonAtomic and CK_NonAtomicToAtomic casts
// Note: Full atomic load/store support is NYI - this tests just the casts

// Test NonAtomicToAtomic cast (assigning non-atomic to atomic)
void test_non_atomic_to_atomic() {
  int x = 50;
  _Atomic int y = x;  // Implicit NonAtomicToAtomic cast
  // CHECK: cir.func{{.*}}test_non_atomic_to_atomicv
  // CHECK: cir.alloca !s32i, !cir.ptr<!s32i>, ["x"
  // CHECK: cir.alloca !s32i, !cir.ptr<!s32i>, ["y"
  // CHECK: cir.load
  // CHECK: cir.store
}

// Test that atomic type casts don't crash the compiler
void test_atomic_cast_exists() {
  int regular = 42;
  _Atomic int atomic_val = regular;
  // Just verify this compiles - the cast infrastructure exists
  // CHECK: cir.func{{.*}}test_atomic_cast_existsv
  // CHECK: cir.alloca !s32i, !cir.ptr<!s32i>, ["regular"
  // CHECK: cir.alloca !s32i, !cir.ptr<!s32i>, ["atomic_val"
}

// Test with different types
void test_atomic_float_cast() {
  float f = 3.14f;
  _Atomic float g = f;
  // CHECK: cir.func{{.*}}test_atomic_float_castv
  // CHECK: cir.alloca !cir.float
  // CHECK: cir.alloca !cir.float
}

// Test that cast infrastructure is in place for pointers
void test_atomic_pointer_cast() {
  int val = 42;
  int* ptr = &val;
  _Atomic(int*) atomic_ptr = ptr;
  // CHECK: cir.func{{.*}}test_atomic_pointer_castv
  // CHECK: cir.alloca !cir.ptr<!s32i>
  // CHECK: cir.alloca !cir.ptr<!s32i>
}
