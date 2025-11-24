// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG

namespace std {
  class type_info {
  public:
    virtual ~type_info();
  };
}

// Basic polymorphic class hierarchy
struct Base {
  virtual ~Base() {}
  int x;
};

struct Derived : Base {
  int y;
};

extern void use_typeinfo(const std::type_info*);

// Test 1: Basic polymorphic typeid - pointer
// CIR-LABEL: cir.func dso_local @_Z22test_polymorphic_basicP4Base
void test_polymorphic_basic(Base* ptr) {
  // CIR: cir.vtable.get_vptr
  // CIR: cir.load{{.*}}!cir.vptr
  // CIR: cir.cast bitcast
  // CIR: cir.ptr_stride
  // CIR: cir.load

  // LLVM-LABEL: @_Z22test_polymorphic_basicP4Base
  // LLVM: load ptr, ptr %
  // LLVM: getelementptr ptr, ptr %{{.*}}, i64 -1
  // LLVM: load ptr, ptr %

  // OGCG-LABEL: @_Z22test_polymorphic_basicP4Base
  // OGCG: load ptr, ptr %
  // OGCG: getelementptr inbounds ptr, ptr %{{.*}}, i64 -1
  // OGCG: load ptr, ptr %
  use_typeinfo(&typeid(*ptr));
}

// Test 2: Polymorphic typeid - reference (no null check)
// CIR-LABEL: cir.func dso_local @_Z14test_referenceR4Base
void test_reference(Base& ref) {
  // CIR-NOT: cir.cmp(eq
  // CIR: cir.vtable.get_vptr
  // CIR: cir.ptr_stride

  // LLVM-LABEL: @_Z14test_referenceR4Base
  // LLVM: load ptr, ptr %
  // LLVM: getelementptr ptr, ptr %{{.*}}, i64 -1

  // OGCG-LABEL: @_Z14test_referenceR4Base
  // OGCG: load ptr, ptr %
  // OGCG: getelementptr inbounds ptr, ptr %{{.*}}, i64 -1
  use_typeinfo(&typeid(ref));
}

// Test 3: Derived class pointer
// CIR-LABEL: cir.func dso_local @_Z20test_derived_pointerP7Derived
void test_derived_pointer(Derived* ptr) {
  // CIR: cir.vtable.get_vptr
  // CIR: cir.ptr_stride

  // LLVM-LABEL: @_Z20test_derived_pointerP7Derived
  // LLVM: getelementptr ptr, ptr %{{.*}}, i64 -1

  // OGCG-LABEL: @_Z20test_derived_pointerP7Derived
  // OGCG: getelementptr inbounds ptr, ptr %{{.*}}, i64 -1
  use_typeinfo(&typeid(*ptr));
}

// Test 4: Const qualified pointer
// CIR-LABEL: cir.func dso_local @_Z14test_const_ptrPK4Base
void test_const_ptr(const Base* ptr) {
  // CIR: cir.vtable.get_vptr
  // CIR: cir.ptr_stride

  // LLVM-LABEL: @_Z14test_const_ptrPK4Base
  // LLVM: getelementptr ptr, ptr %{{.*}}, i64 -1

  // OGCG-LABEL: @_Z14test_const_ptrPK4Base
  // OGCG: getelementptr inbounds ptr, ptr %{{.*}}, i64 -1
  use_typeinfo(&typeid(*ptr));
}
