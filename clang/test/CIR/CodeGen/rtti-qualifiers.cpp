// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG

namespace std {
  class type_info {
  public:
    virtual ~type_info();
    const char* name() const { return __name; }
  protected:
    const char *__name;
  };
}

// Test RTTI with qualified types
// This tests the fix for the bug where RTTI descriptors were being created
// with top-level qualifiers, violating the Itanium ABI assertion:
// "RTTI info cannot have top-level qualifiers"

struct Simple {
  int x;
};

// Test 1: typeid with const-qualified type
void test_const_type(const std::type_info*& out) {
  // CIR-LABEL: cir.func {{.*}}@_Z15test_const_type
  out = &typeid(const int);
  // CIR: cir.get_global @_ZTIi
  // Note: Should use unqualified type's RTTI (_ZTIi, not _ZTIKi)

  // LLVM-LABEL: define {{.*}}@_Z15test_const_type
  // LLVM: store ptr @_ZTIi

  // OGCG-LABEL: define {{.*}}@_Z15test_const_type
  // OGCG: store ptr @_ZTIi
}

// Test 2: typeid with volatile-qualified type
void test_volatile_type(const std::type_info*& out) {
  // CIR-LABEL: cir.func {{.*}}@_Z18test_volatile_type
  out = &typeid(volatile int);
  // CIR: cir.get_global @_ZTIi
  // Note: Should use unqualified type's RTTI

  // LLVM-LABEL: define {{.*}}@_Z18test_volatile_type
  // LLVM: store ptr @_ZTIi

  // OGCG-LABEL: define {{.*}}@_Z18test_volatile_type
  // OGCG: store ptr @_ZTIi
}

// Test 3: typeid with const-volatile-qualified type
void test_const_volatile_type(const std::type_info*& out) {
  // CIR-LABEL: cir.func {{.*}}@_Z24test_const_volatile_type
  out = &typeid(const volatile int);
  // CIR: cir.get_global @_ZTIi
  // Note: Should use unqualified type's RTTI

  // LLVM-LABEL: define {{.*}}@_Z24test_const_volatile_type
  // LLVM: store ptr @_ZTIi

  // OGCG-LABEL: define {{.*}}@_Z24test_const_volatile_type
  // OGCG: store ptr @_ZTIi
}

// Test 4: typeid with const pointer (the pointer itself is const)
void test_const_pointer(const std::type_info*& out) {
  // CIR-LABEL: cir.func {{.*}}@_Z18test_const_pointer
  int* const ptr = nullptr;
  out = &typeid(ptr);
  // CIR: cir.get_global @_ZTIPi
  // Note: typeid of a const pointer should use unqualified pointer type RTTI

  // LLVM-LABEL: define {{.*}}@_Z18test_const_pointer
  // LLVM: store ptr @_ZTIPi

  // OGCG-LABEL: define {{.*}}@_Z18test_const_pointer
  // OGCG: store ptr @_ZTIPi
}

// Test 5: typeid with pointer to const (pointee is const)
void test_pointer_to_const(const std::type_info*& out) {
  // CIR-LABEL: cir.func {{.*}}@_Z21test_pointer_to_const
  const int* ptr = nullptr;
  out = &typeid(ptr);
  // CIR: cir.get_global @_ZTIPKi
  // Note: Pointer to const int has different RTTI (qualifiers on pointee matter)

  // LLVM-LABEL: define {{.*}}@_Z21test_pointer_to_const
  // LLVM: store ptr @_ZTIPKi

  // OGCG-LABEL: define {{.*}}@_Z21test_pointer_to_const
  // OGCG: store ptr @_ZTIPKi
}

// Test 6: typeid with qualified struct type
void test_const_struct(const std::type_info*& out) {
  // CIR-LABEL: cir.func {{.*}}@_Z17test_const_struct
  out = &typeid(const Simple);
  // CIR: cir.get_global @_ZTI6Simple
  // Note: Should use unqualified struct's RTTI

  // LLVM-LABEL: define {{.*}}@_Z17test_const_struct
  // LLVM: store ptr @_ZTI6Simple

  // OGCG-LABEL: define {{.*}}@_Z17test_const_struct
  // OGCG: store ptr @_ZTI6Simple
}

// Test 7: typeid with qualified expression
void test_const_expr(const std::type_info*& out) {
  // CIR-LABEL: cir.func {{.*}}@_Z15test_const_expr
  const int x = 42;
  out = &typeid(x);
  // CIR: cir.get_global @_ZTIi
  // Note: Should use unqualified type's RTTI

  // LLVM-LABEL: define {{.*}}@_Z15test_const_expr
  // LLVM: store ptr @_ZTIi

  // OGCG-LABEL: define {{.*}}@_Z15test_const_expr
  // OGCG: store ptr @_ZTIi
}

// Test 8: dynamic_cast with qualified types
struct PolyBase {
  virtual ~PolyBase() = default;
};

struct PolyDerived : PolyBase {
  int value;
};

void test_dynamic_cast_qualified(const PolyBase* b, const PolyDerived*& out) {
  // CIR-LABEL: cir.func {{.*}}@_Z27test_dynamic_cast_qualified
  out = dynamic_cast<const PolyDerived*>(b);
  // CIR: cir.call @__dynamic_cast
  // Note: Should use unqualified RTTI descriptors for Base and Derived

  // LLVM-LABEL: define {{.*}}@_Z27test_dynamic_cast_qualified
  // LLVM: call ptr @__dynamic_cast

  // OGCG-LABEL: define {{.*}}@_Z27test_dynamic_cast_qualified
  // OGCG: call ptr @__dynamic_cast
}
