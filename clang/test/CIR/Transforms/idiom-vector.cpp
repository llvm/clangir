// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-idiom-recognizer -emit-cir -I%S/../Inputs -mmlir --mlir-print-ir-after-all %s -o - 2>&1 | FileCheck %s -check-prefix=PASS_ENABLED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -I%S/../Inputs -fclangir-idiom-recognizer="remarks=found-calls" -clangir-verify-diagnostics %s -o %t.cir

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -fclangir-idiom-recognizer -emit-cir -I%S/../Inputs -mmlir --mlir-print-ir-before=cir-idiom-recognizer %s -o - 2>&1 | FileCheck %s -check-prefix=BEFORE-IDIOM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -fclangir-idiom-recognizer -emit-cir -I%S/../Inputs -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o - 2>&1 | FileCheck %s -check-prefix=AFTER-IDIOM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -fclangir-idiom-recognizer -emit-cir -I%S/../Inputs -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o - 2>&1 | FileCheck %s -check-prefix=AFTER-LOWERING-PREPARE

// PASS_ENABLED:  IR Dump After IdiomRecognizer (cir-idiom-recognizer)

namespace std {
template <typename T> class vector {
public:
  vector() {}   // expected-remark {{found call to std::vector_cxx_ctor()}}
  ~vector() {}; // expected-remark{{found call to std::vector_cxx_dtor()}}
};
}; // namespace std

void vector_test() {
  std::vector<int> v; // expected-remark {{found call to std::vector_cxx_ctor()}}

  // BEFORE-IDIOM: cir.call @_ZNSt6vectorIiEC1Ev(
  // BEFORE-IDIOM: cir.call @_ZNSt6vectorIiED1Ev(
  // AFTER-IDIOM: cir.std.vector_cxx_ctor(
  // AFTER-IDIOM: cir.std.vector_cxx_dtor(
  // AFTER-LOWERING-PREPARE: cir.call @_ZNSt6vectorIiEC1Ev(
  // AFTER-LOWERING-PREPARE: cir.call @_ZNSt6vectorIiED1Ev(
}
