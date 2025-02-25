// RUN: %clang -std=c++20 -target x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#include <iostream>
#include <string>

void foo(const char *path) {
  std::string foo = path;
  std::cout << foo;
}

// CHECK:      cir.func @_Z3fooPKc(%arg0: !cir.ptr<!s8i>
// CHECK-NEXT:   %[[V0:.*]] = cir.alloca !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>, ["path", init] {alignment = 8 : i64}
// CHECK-NEXT:   %[[V1:.*]] = cir.alloca !ty_std3A3A__cxx113A3Abasic_string3Cchar2C_std3A3Achar_traits3Cchar3E2C_std3A3Aallocator3Cchar3E3E, !cir.ptr<!ty_std3A3A__cxx113A3Abasic_string3Cchar2C_std3A3Achar_traits3Cchar3E2C_std3A3Aallocator3Cchar3E3E>, ["foo", init] {alignment = 8 : i64}
// CHECK-NEXT:   %[[V2:.*]] = cir.alloca !cir.ptr<!ty_std3A3Abasic_ostream3Cchar2C_std3A3Achar_traits3Cchar3E3E>, !cir.ptr<!cir.ptr<!ty_std3A3Abasic_ostream3Cchar2C_std3A3Achar_traits3Cchar3E3E>>, ["tmp.try.call.res"] {alignment = 8 : i64}
// CHECK-NEXT:   cir.store %arg0, %[[V0]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     %[[V5:.*]] = cir.alloca !ty_std3A3Aallocator3Cchar3E, !cir.ptr<!ty_std3A3Aallocator3Cchar3E>, ["ref.tmp0"] {alignment = 1 : i64}
// CHECK-NEXT:     %[[V6:.*]] = cir.load %[[V0]] : !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!s8i>
// CHECK-NEXT:     cir.call @_ZNSaIcEC1Ev(%[[V5]]) : (!cir.ptr<!ty_std3A3Aallocator3Cchar3E>) -> ()
// CHECK-NEXT:     cir.try synthetic cleanup {
// CHECK-NEXT:       cir.call exception @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_(%[[V1]], %[[V6]], %[[V5]]) : (!cir.ptr<!ty_std3A3A__cxx113A3Abasic_string3Cchar2C_std3A3Achar_traits3Cchar3E2C_std3A3Aallocator3Cchar3E3E>, !cir.ptr<!s8i>, !cir.ptr<!ty_std3A3Aallocator3Cchar3E>) -> () cleanup {
// CHECK-NEXT:         cir.call @_ZNSaIcED1Ev(%[[V5]]) : (!cir.ptr<!ty_std3A3Aallocator3Cchar3E>) -> ()
// CHECK-NEXT:         cir.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     } catch [#cir.unwind {
// CHECK-NEXT:       cir.resume
// CHECK-NEXT:     }]
// CHECK-NEXT:     cir.call @_ZNSaIcED1Ev(%[[V5]]) : (!cir.ptr<!ty_std3A3Aallocator3Cchar3E>) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[V3:.*]] = cir.get_global @_ZSt4cout : !cir.ptr<!ty_std3A3Abasic_ostream3Cchar2C_std3A3Achar_traits3Cchar3E3E>
// CHECK-NEXT:   cir.try synthetic cleanup {
// CHECK-NEXT:     %[[V5:.*]] = cir.call exception @_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKNSt7__cxx1112basic_stringIS4_S5_T1_EE(%[[V3]], %[[V1]]) : (!cir.ptr<!ty_std3A3Abasic_ostream3Cchar2C_std3A3Achar_traits3Cchar3E3E>, !cir.ptr<!ty_std3A3A__cxx113A3Abasic_string3Cchar2C_std3A3Achar_traits3Cchar3E2C_std3A3Aallocator3Cchar3E3E>) -> !cir.ptr<!ty_std3A3Abasic_ostream3Cchar2C_std3A3Achar_traits3Cchar3E3E> cleanup {
// CHECK-NEXT:       cir.call @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev(%[[V1]]) : (!cir.ptr<!ty_std3A3A__cxx113A3Abasic_string3Cchar2C_std3A3Achar_traits3Cchar3E2C_std3A3Aallocator3Cchar3E3E>) -> ()
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     cir.store %[[V5]], %[[V2]] : !cir.ptr<!ty_std3A3Abasic_ostream3Cchar2C_std3A3Achar_traits3Cchar3E3E>, !cir.ptr<!cir.ptr<!ty_std3A3Abasic_ostream3Cchar2C_std3A3Achar_traits3Cchar3E3E>>
// CHECK-NEXT:     cir.yield
// CHECK-NEXT:   } catch [#cir.unwind {
// CHECK-NEXT:     cir.resume
// CHECK-NEXT:   }]
// CHECK-NEXT:   %[[V4:.*]] = cir.load %[[V2]] : !cir.ptr<!cir.ptr<!ty_std3A3Abasic_ostream3Cchar2C_std3A3Achar_traits3Cchar3E3E>>, !cir.ptr<!ty_std3A3Abasic_ostream3Cchar2C_std3A3Achar_traits3Cchar3E3E>
// CHECK-NEXT:   cir.call @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev(%[[V1]]) : (!cir.ptr<!ty_std3A3A__cxx113A3Abasic_string3Cchar2C_std3A3Achar_traits3Cchar3E2C_std3A3Aallocator3Cchar3E3E>) -> ()
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
