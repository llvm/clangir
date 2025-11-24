// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ogcg.ll %s

// Test APValue emission for member function pointers with CIR, LLVM lowering,
// and comparison to original CodeGen.

struct S {
  void foo();
  virtual void bar();
};

// Test 1: Non-virtual member function pointer
// CIR: cir.global external @pmf1 = #cir.method<@_ZN1S3fooEv>
// LLVM: @pmf1 = global { i64, i64 } { i64 ptrtoint (ptr @_ZN1S3fooEv to i64), i64 0 }
// OGCG: @pmf1 = global { i64, i64 } { i64 ptrtoint (ptr @_ZN1S3fooEv to i64), i64 0 }
extern void (S::*pmf1)();
void (S::*pmf1)() = &S::foo;

// Test 2: Virtual member function pointer
// CIR: cir.global external @pmf2 = #cir.method<vtable_offset = {{[0-9]+}}>
// LLVM: @pmf2 = global { i64, i64 } { i64 {{[0-9]+}}, i64 0 }
// OGCG: @pmf2 = global { i64, i64 } { i64 {{[0-9]+}}, i64 0 }
extern void (S::*pmf2)();
void (S::*pmf2)() = &S::bar;

// Test 3: Null member function pointer
// CIR: cir.global external @pmf3 = #cir.method<null>
// LLVM: @pmf3 = global { i64, i64 } zeroinitializer
// OGCG: @pmf3 = global { i64, i64 } zeroinitializer
extern void (S::*pmf3)();
void (S::*pmf3)() = nullptr;
