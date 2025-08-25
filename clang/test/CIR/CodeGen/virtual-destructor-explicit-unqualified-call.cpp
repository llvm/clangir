// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s


class A {
  virtual ~A();
  A B(A);
};
A A::B(A) { 
  // CIR-LABEL:   cir.func dso_local @_ZN1A1BES_(
  // CIR-SAME:      %[[THIS_ARG:.*]]: !cir.ptr<!rec_A>
  // CIR-NEXT:           %[[THIS_VAR:.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>
  // CIR:                %[[THIS:.*]] = cir.load %[[THIS_VAR]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A> 
  // CIR-NEXT:           %[[VPTR_PTR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_A> -> !cir.ptr<!cir.vptr> 
  // CIR-NEXT:           %[[VPTR:.*]] = cir.load align(8) %[[VPTR_PTR]] : !cir.ptr<!cir.vptr>, !cir.vptr 
  // CIR-NEXT:           %[[DTOR_PTR:.*]] = cir.vtable.get_virtual_fn_addr %[[VPTR]][0] : !cir.vptr -> !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_A>)>>> 
  // CIR-NEXT:           %[[DTOR:.*]] = cir.load align(8) %[[DTOR_PTR]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_A>)>>>, !cir.ptr<!cir.func<(!cir.ptr<!rec_A>)>> 
  // CIR-NEXT:           cir.call %[[DTOR]](%[[THIS]]) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_A>)>>, !cir.ptr<!rec_A>) -> () 
  // CIR-NEXT:           cir.trap 
  // CIR-NEXT:         } 

  this->~A(); 
}
