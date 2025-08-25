// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s


// CIR-LABEL:   cir.func dso_local @_ZN1A1BES_(
// CIR-SAME:      %[[ARG0:.*]]: !cir.ptr<!rec_A>
// CIR-SAME:      %[[ARG1:.*]]: !rec_A
// CIR-NEXT:           %[[VAL_0:.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["this", init] {alignment = 8 : i64} 
// CIR-NEXT:           %[[VAL_1:.*]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["", init] {alignment = 8 : i64} 
// CIR-NEXT:           %[[VAL_2:.*]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["__retval"] {alignment = 8 : i64} 
// CIR-NEXT:           cir.store %[[ARG0]], %[[VAL_0]] : !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>> 
// CIR-NEXT:           cir.store %[[ARG1]], %[[VAL_1]] : !rec_A, !cir.ptr<!rec_A> 
// CIR-NEXT:           %[[VAL_3:.*]] = cir.load %[[VAL_0]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A> 
// CIR-NEXT:           %[[VAL_4:.*]] = cir.vtable.get_vptr %[[VAL_3]] : !cir.ptr<!rec_A> -> !cir.ptr<!cir.vptr> 
// CIR-NEXT:           %[[VAL_5:.*]] = cir.load align(8) %[[VAL_4]] : !cir.ptr<!cir.vptr>, !cir.vptr 
// CIR-NEXT:           %[[VAL_6:.*]] = cir.vtable.get_virtual_fn_addr %[[VAL_5]][0] : !cir.vptr -> !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_A>)>>> 
// CIR-NEXT:           %[[VAL_7:.*]] = cir.load align(8) %[[VAL_6]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_A>)>>>, !cir.ptr<!cir.func<(!cir.ptr<!rec_A>)>> 
// CIR-NEXT:           cir.call %[[VAL_7]](%[[VAL_3]]) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_A>)>>, !cir.ptr<!rec_A>) -> () 
// CIR-NEXT:           cir.trap 
// CIR-NEXT:         } 


class A {
  virtual ~A();
  A B(A);
};
A A::B(A) { this->~A(); }
