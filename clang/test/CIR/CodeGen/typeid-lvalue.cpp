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
    bool operator==(const type_info& __arg) const {
     return __name == __arg.__name;
    }

    bool operator!=(const type_info& __arg) const {
      return !operator==(__arg);
    }

    bool before(const type_info& __arg) const {
      return __name < __arg.__name;
    }

    unsigned long hash_code() const {
      return reinterpret_cast<unsigned long long>(__name);
    }
  protected:
    const char *__name;
  };
}

// Test 1: Non-polymorphic type - simple struct
struct Simple {
  int x;
};

void test_simple_type(const std::type_info*& out) {
  // CIR-LABEL: cir.func {{.*}}@_Z16test_simple_type
  out = &typeid(Simple);
  // CIR: cir.get_global @_ZTI6Simple

  // LLVM-LABEL: define {{.*}}@_Z16test_simple_type
  // LLVM: store ptr @_ZTI6Simple

  // OGCG-LABEL: define {{.*}}@_Z16test_simple_type
  // OGCG: store ptr @_ZTI6Simple
}

// Test 2: Non-polymorphic type - expression operand
void test_expression_operand(const std::type_info*& out) {
  // CIR-LABEL: cir.func {{.*}}@_Z23test_expression_operand
  Simple s;
  out = &typeid(s);
  // CIR: cir.get_global @_ZTI6Simple

  // LLVM-LABEL: define {{.*}}@_Z23test_expression_operand
  // LLVM: store ptr @_ZTI6Simple

  // OGCG-LABEL: define {{.*}}@_Z23test_expression_operand
  // OGCG: store ptr @_ZTI6Simple
}

// Test 3: Polymorphic base class
struct Base {
  virtual ~Base() = default;
};

struct Derived : Base {
  int y;
};

// Test with non-polymorphic lookup (type operand)
void test_polymorphic_type(const std::type_info*& out) {
  // CIR-LABEL: cir.func {{.*}}@_Z21test_polymorphic_type
  out = &typeid(Base);
  // CIR: cir.get_global @_ZTI4Base

  // LLVM-LABEL: define {{.*}}@_Z21test_polymorphic_type
  // LLVM: store ptr @_ZTI4Base

  // OGCG-LABEL: define {{.*}}@_Z21test_polymorphic_type
  // OGCG: store ptr @_ZTI4Base
}

// Test 4: Built-in type
void test_builtin_type(const std::type_info*& out) {
  // CIR-LABEL: cir.func {{.*}}@_Z17test_builtin_type
  out = &typeid(int);
  // CIR: cir.get_global @_ZTIi

  // LLVM-LABEL: define {{.*}}@_Z17test_builtin_type
  // LLVM: store ptr @_ZTIi

  // OGCG-LABEL: define {{.*}}@_Z17test_builtin_type
  // OGCG: store ptr @_ZTIi
}

// Test 5: Passing typeid as function argument
void consume_type_info(const std::type_info& ti) {
  // CIR-LABEL: cir.func {{.*}}@_Z17consume_type_info

  // LLVM-LABEL: define {{.*}}@_Z17consume_type_info

  // OGCG-LABEL: define {{.*}}@_Z17consume_type_info
}

void test_function_argument() {
  // CIR-LABEL: cir.func {{.*}}@_Z22test_function_argumentv
  consume_type_info(typeid(int));
  // CIR: cir.get_global @_ZTIi
  // CIR: cir.call @_Z17consume_type_info

  // LLVM-LABEL: define {{.*}}@_Z22test_function_argumentv
  // LLVM: call {{.*}}@_Z17consume_type_infoRKSt9type_info(ptr @_ZTIi)

  // OGCG-LABEL: define {{.*}}@_Z22test_function_argumentv
  // OGCG: call {{.*}}@_Z17consume_type_infoRKSt9type_info(ptr {{.*}}@_ZTIi)
}
