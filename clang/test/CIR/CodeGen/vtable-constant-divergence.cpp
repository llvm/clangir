// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-codegen.ll
// RUN: FileCheck --check-prefix=CIR --input-file=%t-cir.ll %s
// RUN: FileCheck --check-prefix=CODEGEN --input-file=%t-codegen.ll %s

// XFAIL: *

// This test documents a divergence between CIR and CodeGen:
// CIR emits vtables as 'global' (mutable) instead of 'constant' (immutable).
// This is a bug that needs to be fixed.
//
// Expected (CodeGen):
//   @_ZTV4Base = linkonce_odr unnamed_addr constant { [3 x ptr] } ...
//
// Actual (CIR):
//   @_ZTV4Base = linkonce_odr global { [3 x ptr] } ...
//
// The vtable should be marked as 'constant' because:
// 1. Vtables are never modified at runtime
// 2. They should be placed in .rodata (read-only data section)
// 3. Writable vtables are a security vulnerability
// 4. CodeGen has always emitted them as constant

class Base {
public:
  virtual void foo() {}
};

void test() {
  Base b;
  b.foo();
}

// Both should emit vtable as constant
// CIR: @_ZTV4Base = linkonce_odr unnamed_addr constant
// CODEGEN: @_ZTV4Base = linkonce_odr unnamed_addr constant
