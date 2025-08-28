// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o %t.cir
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o %t.cir
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.cir %s

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


  // LLVM-LABEL: define dso_local %class.A @_ZN1A1BES_(
  // LLVM:         %[[VAL_0:.*]] = alloca ptr, i64 1, align 8
  // LLVM:         %[[VAL_6:.*]] = load ptr, ptr %[[VAL_0]], align 8
  // LLVM-NEXT:    %[[VAL_7:.*]] = load ptr, ptr %[[VAL_6]], align 8
  // LLVM-NEXT:    %[[VAL_8:.*]] = getelementptr inbounds ptr, ptr %[[VAL_7]], i32 0
  // LLVM-NEXT:    %[[VAL_9:.*]] = load ptr, ptr %[[VAL_8]], align 8
  // LLVM-NEXT:    call void %[[VAL_9]](ptr %[[VAL_6]])
  // LLVM-NEXT:    call void @llvm.trap()
  

  // OGCG-LABEL: define dso_local void @_ZN1A1BES_(
  // OGCG:         %[[VAL_0:.*]] = alloca ptr, align 8
  // OGCG-NEXT:    %[[VAL_1:.*]] = alloca ptr, align 8
  // OGCG:         %[[VAL_6:.*]] = load ptr, ptr %[[VAL_1]], align 8
  // OGCG-NEXT:    %[[VAL_7:.*]] = load ptr, ptr %[[VAL_6]], align 8
  // OGCG-NEXT:    %[[VAL_8:.*]] = getelementptr inbounds ptr, ptr %[[VAL_7]], i64 0
  // OGCG-NEXT:    %[[VAL_9:.*]] = load ptr, ptr %[[VAL_8]], align 8
  // OGCG-NEXT:    call void %[[VAL_9]](ptr noundef nonnull align 8 dereferenceable(8) %[[VAL_6]]) #2
  // OGCG-NEXT:    call void @llvm.trap()


  this->~A(); 
}
