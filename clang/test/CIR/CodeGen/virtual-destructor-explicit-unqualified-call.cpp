// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s


class A {
  virtual ~A();
  A B(A);
};
A A::B(A) { this->~A(); }



//CIR: cir.func dso_local @_ZN1A1BES_(%arg0: !cir.ptr<!rec_A> 
//CIR-NEXT: %0 = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["this", init] {alignment = 8 : i64} loc(#loc3)
//CIR-NEXT: %1 = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["", init] {alignment = 8 : i64} loc(#loc4)
//CIR-NEXT: %2 = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["__retval"] {alignment = 8 : i64} loc(#loc2)
//CIR-NEXT: cir.store %arg0, %0 : !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>> loc(#loc5)
//CIR-NEXT: cir.store %arg1, %1 : !rec_A, !cir.ptr<!rec_A> loc(#loc5)
//CIR-NEXT: %3 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A> loc(#loc3)
//CIR-NEXT: %4 = cir.vtable.get_vptr %3 : !cir.ptr<!rec_A> -> !cir.ptr<!cir.vptr> loc(#loc6)
//CIR-NEXT: %5 = cir.load align(8) %4 : !cir.ptr<!cir.vptr>, !cir.vptr loc(#loc6)
//CIR-NEXT: %6 = cir.vtable.get_virtual_fn_addr %5[0] : !cir.vptr -> !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_A>)>>> loc(#loc6)
//CIR-NEXT: %7 = cir.load align(8) %6 : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_A>)>>>, !cir.ptr<!cir.func<(!cir.ptr<!rec_A>)>> loc(#loc6)
//CIR-NEXT: cir.call %7(%3) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_A>)>>, !cir.ptr<!rec_A>) -> () extra(#fn_attr1) loc(#loc7)
//CIR-NEXT: cir.trap loc(#loc2)
//CIR-NEXT:} 
