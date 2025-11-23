// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --input-file=%t.ogcg.ll %s -check-prefix=LLVM

// Test base-to-derived member data pointer conversion

class Base {
public:
  int baseField1;
  int baseField2;
};

class Derived : public Base {
public:
  int derivedField;
};

// Test 1: Base-to-derived conversion with zero offset (baseField1 at field index 0)
// CIR: cir.global external @ptrZeroOffset = #cir.data_member<0>
// LLVM: @ptrZeroOffset = global i64 0
int Derived::*ptrZeroOffset = static_cast<int Derived::*>(&Base::baseField1);

// Test 2: Base-to-derived conversion with non-zero offset (baseField2 at field index 1)
// CIR: cir.global external @ptrNonZeroOffset = #cir.data_member<1>
// LLVM: @ptrNonZeroOffset = global i64 4
int Derived::*ptrNonZeroOffset = static_cast<int Derived::*>(&Base::baseField2);

// Test 3: Reinterpret cast (should preserve original value)
// CIR: cir.global external @ptrReinterpret = #cir.data_member<0>
// LLVM: @ptrReinterpret = global i64 0
int Derived::*ptrReinterpret = reinterpret_cast<int Derived::*>(&Base::baseField1);
