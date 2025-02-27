// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -I%S/../Inputs -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#include "std-cxx.h"

// CHECK-LABEL: @_Z3fooPKc

void foo(const char* path) {
  std::string str = path;
  str = path;
  str = path;
}

// CHECK: cir.try synthetic cleanup {
// CHECK:   cir.call exception @_ZNSbIcEC1EPKcRKNS_9AllocatorE({{.*}}, {{.*}}, {{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>, !cir.ptr<!s8i>, !cir.ptr<!ty_std3A3Abasic_string3Cchar3E3A3AAllocator>) -> () cleanup {
// CHECK:     cir.call @_ZNSbIcED1Ev({{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>) -> ()
// CHECK:     cir.yield
// CHECK:   }
// CHECK:   cir.yield
// CHECK: } catch [#cir.unwind {
// CHECK:   cir.resume
// CHECK: }]
// CHECK: cir.try synthetic cleanup {
// CHECK:   {{.*}} = cir.call exception @_ZNSbIcEaSERKS_({{.*}}, {{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>, !cir.ptr<!ty_std3A3Abasic_string3Cchar3E>) -> !cir.ptr<!ty_std3A3Abasic_string3Cchar3E> cleanup {
// CHECK:     cir.call @_ZNSbIcED1Ev({{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>) -> ()
// CHECK:     cir.yield
// CHECK:   }
// CHECK:   cir.store {{.*}}, {{.*}} : !cir.ptr<!ty_std3A3Abasic_string3Cchar3E>, !cir.ptr<!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>>
// CHECK:   cir.yield
// CHECK: } catch [#cir.unwind {
// CHECK:   cir.resume
// CHECK: }]
// CHECK: {{.*}} = cir.load {{.*}} : !cir.ptr<!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>>, !cir.ptr<!ty_std3A3Abasic_string3Cchar3E>
// CHECK: cir.call @_ZNSbIcED1Ev({{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>) -> ()
// CHECK: cir.try synthetic cleanup {
// CHECK:   cir.call exception @_ZNSbIcEC1EPKcRKNS_9AllocatorE({{.*}}, {{.*}}, {{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>, !cir.ptr<!s8i>, !cir.ptr<!ty_std3A3Abasic_string3Cchar3E3A3AAllocator>) -> ()
// CHECK:   cir.yield
// CHECK: } catch [#cir.unwind {
// CHECK:   cir.resume
// CHECK: }]
// CHECK: cir.try synthetic cleanup {
// CHECK:   {{.*}} = cir.call exception @_ZNSbIcEaSERKS_({{.*}}, {{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>, !cir.ptr<!ty_std3A3Abasic_string3Cchar3E>) -> !cir.ptr<!ty_std3A3Abasic_string3Cchar3E> cleanup {
// CHECK:     cir.call @_ZNSbIcED1Ev({{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>) -> ()
// CHECK:     cir.call @_ZNSbIcED1Ev({{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>) -> ()
// CHECK:     cir.yield
// CHECK:   }
// CHECK:   cir.store {{.*}}, {{.*}} : !cir.ptr<!ty_std3A3Abasic_string3Cchar3E>, !cir.ptr<!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>>
// CHECK:   cir.yield
// CHECK: } catch [#cir.unwind {
// CHECK:   cir.resume
// CHECK: }]
