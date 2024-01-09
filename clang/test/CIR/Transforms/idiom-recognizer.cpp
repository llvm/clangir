// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -fclangir-idiom-recognizer -emit-cir -I%S/../Inputs -mmlir --mlir-print-ir-after-all %s -o - 2>&1 | FileCheck %s -check-prefix=PASS_ENABLED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir-enable -emit-cir -I%S/../Inputs -fclangir-idiom-recognizer="remarks=found-calls" -clangir-verify-diagnostics %s -o %t.cir

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir-enable -fclangir-idiom-recognizer -emit-cir -I%S/../Inputs -mmlir --mlir-print-ir-before=cir-idiom-recognizer %s -o - 2>&1 | FileCheck %s -check-prefix=BEFORE-IDIOM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir-enable -fclangir-idiom-recognizer -emit-cir -I%S/../Inputs -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o - 2>&1 | FileCheck %s -check-prefix=AFTER-IDIOM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir-enable -fclangir-idiom-recognizer -emit-cir -I%S/../Inputs -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o - 2>&1 | FileCheck %s -check-prefix=AFTER-LOWERING-PREPARE

// PASS_ENABLED:  IR Dump After IdiomRecognizer (cir-idiom-recognizer)

#include "std-cxx.h"

int test_find(unsigned char n = 3)
{
    unsigned num_found = 0;
    std::array<unsigned char, 9> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto f = std::find(v.begin(), v.end(), n); // expected-remark {{found call to std::find()}}
                                               // expected-remark@-1 {{found call to begin() iterator}}
                                               // expected-remark@-2 {{found call to end() iterator}}

    // BEFORE-IDIOM: {{.*}} cir.call @_ZNSt5arrayIhLj9EE5beginEv(
    // AFTER-IDIOM: {{.*}} cir.iterator_begin(@_ZNSt5arrayIhLj9EE5beginEv,
    // AFTER-LOWERING-PREPARE: {{.*}} cir.call @_ZNSt5arrayIhLj9EE5beginEv(

    // BEFORE-IDIOM: {{.*}} cir.call @_ZNSt5arrayIhLj9EE3endEv(
    // AFTER-IDIOM: {{.*}} cir.iterator_end(@_ZNSt5arrayIhLj9EE3endEv,
    // AFTER-LOWERING-PREPARE: {{.*}} cir.call @_ZNSt5arrayIhLj9EE3endEv(

    // BEFORE-IDIOM: {{.*}} cir.call @_ZSt4findIPhhET_S1_S1_RKT0_(
    // AFTER-IDIOM: {{.*}} cir.std.find(@_ZSt4findIPhhET_S1_S1_RKT0_,
    // AFTER-LOWERING-PREPARE: {{.*}} cir.call @_ZSt4findIPhhET_S1_S1_RKT0_(

    if (f != v.end()) // expected-remark {{found call to end() iterator}}
        num_found++;
    return num_found;
}

namespace yolo {
template<typename T, unsigned N> struct array {
  T arr[N];
  typedef T value_type;
  typedef value_type* iterator;
  constexpr iterator begin() { return iterator(arr); }
};
}

int iter_test()
{
  yolo::array<unsigned char, 3> v = {1, 2, 3};
  (void)v.begin(); // no remark should be produced.
}