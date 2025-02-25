// RUN: %clang -std=c++20 -target x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#include <string>

void foo(const char *path) {
  std::string foo = path;
  for (auto ch : foo)
    putchar(ch);
}

// CHECK:      cir.for : cond {
// CHECK-NEXT:   %[[V11:.*]] = cir.call @_ZN9__gnu_cxxeqIPcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEbRKNS_17__normal_iteratorIT_T0_EESD_QrqXeqcldtfp_4baseEcldtfp0_4baseERSt14convertible_toIbEE({{.*}}, {{.*}}) : (!cir.ptr<!ty___gnu_cxx3A3A__normal_iterator3Cchar_2A2C_std3A3A__cxx113A3Abasic_string3Cchar3E3E>, !cir.ptr<!ty___gnu_cxx3A3A__normal_iterator3Cchar_2A2C_std3A3A__cxx113A3Abasic_string3Cchar3E3E>) -> !cir.bool
// CHECK-NEXT:   %[[V12:.*]] = cir.unary(not, {{.*}}) : !cir.bool, !cir.bool
// CHECK-NEXT:   cir.condition(%[[V12]])
// CHECK-NEXT: } body {
// CHECK-NEXT:   %[[V11:.*]] = cir.call @_ZNK9__gnu_cxx17__normal_iteratorIPcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEdeEv({{.*}}) : (!cir.ptr<!ty___gnu_cxx3A3A__normal_iterator3Cchar_2A2C_std3A3A__cxx113A3Abasic_string3Cchar3E3E>) -> !cir.ptr<!s8i>
// CHECK-NEXT:   %[[V12:.*]] = cir.load %[[V11]] : !cir.ptr<!s8i>, !s8i
// CHECK-NEXT:   cir.store %[[V12]], {{.*}} : !s8i, !cir.ptr<!s8i>
// CHECK-NEXT:   %[[V13:.*]] = cir.load {{.*}} : !cir.ptr<!s8i>, !s8i
// CHECK-NEXT:   %[[V14:.*]] = cir.cast(integral, %[[V13]] : !s8i), !s32i
// CHECK-NEXT:   cir.try synthetic cleanup {
// CHECK-NEXT:     %[[V16:.*]] = cir.call exception @putchar(%[[V14]]) : (!s32i) -> !s32i cleanup {
// CHECK-NEXT:       cir.call @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev({{.*}}) : (!cir.ptr<!ty_std3A3A__cxx113A3Abasic_string3Cchar2C_std3A3Achar_traits3Cchar3E2C_std3A3Aallocator3Cchar3E3E>) -> ()
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     cir.store %[[V16]], {{.*}} : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:     cir.yield
// CHECK-NEXT:   } catch [#cir.unwind {
// CHECK-NEXT:     cir.resume
// CHECK-NEXT:   }]
// CHECK-NEXT:   %[[V15:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: }
