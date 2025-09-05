// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct Delegating {
  Delegating();
  Delegating(int);
};

// Check that the constructor being delegated to is called with the correct
// arguments.
Delegating::Delegating() : Delegating(0) {}

// CIR-LABEL: cir.func dso_local @_ZN10DelegatingC2Ev(%arg0: !cir.ptr<!rec_Delegating> {{.*}}) {{.*}} {
// CIR-NEXT:    %0 = cir.alloca !cir.ptr<!rec_Delegating>, !cir.ptr<!cir.ptr<!rec_Delegating>>, ["this", init] {alignment = 8 : i64}
// CIR-NEXT:    cir.store{{.*}} %arg0, %0 : !cir.ptr<!rec_Delegating>, !cir.ptr<!cir.ptr<!rec_Delegating>>
// CIR-NEXT:    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_Delegating>>, !cir.ptr<!rec_Delegating>
// CIR-NEXT:    %2 = cir.const #cir.int<0> : !s32i
// CIR-NEXT:    cir.call @_ZN10DelegatingC2Ei(%1, %2) : (!cir.ptr<!rec_Delegating>, !s32i) -> ()
// CIR-NEXT:    cir.return
// CIR-NEXT:  }

struct DelegatingWithZeroing {
  int i;
  DelegatingWithZeroing() = default;
  DelegatingWithZeroing(int);
};

// Check that the delegating constructor performs zero-initialization here.
// FIXME: we should either emit the trivial default constructor or remove the
// call to it in a lowering pass.
DelegatingWithZeroing::DelegatingWithZeroing(int) : DelegatingWithZeroing() {}

// CIR-LABEL: cir.func dso_local @_ZN21DelegatingWithZeroingC2Ei(%arg0: !cir.ptr<!rec_DelegatingWithZeroing> {{.*}}, %arg1: !s32i {{.*}}) {{.*}} {
// CIR-NEXT:    %0 = cir.alloca !cir.ptr<!rec_DelegatingWithZeroing>, !cir.ptr<!cir.ptr<!rec_DelegatingWithZeroing>>, ["this", init] {alignment = 8 : i64}
// CIR-NEXT:    %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["", init] {alignment = 4 : i64}
// CIR-NEXT:    cir.store{{.*}} %arg0, %0 : !cir.ptr<!rec_DelegatingWithZeroing>, !cir.ptr<!cir.ptr<!rec_DelegatingWithZeroing>>
// CIR-NEXT:    cir.store{{.*}} %arg1, %1 : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:    %2 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_DelegatingWithZeroing>>, !cir.ptr<!rec_DelegatingWithZeroing>
// CIR-NEXT:    %3 = cir.const #cir.zero : !rec_DelegatingWithZeroing
// CIR-NEXT:    cir.store{{.*}} %3, %2 : !rec_DelegatingWithZeroing, !cir.ptr<!rec_DelegatingWithZeroing>
// CIR-NEXT:    cir.call @_ZN21DelegatingWithZeroingC2Ev(%2) : (!cir.ptr<!rec_DelegatingWithZeroing>) -> () extra(#fn_attr{{[0-9]*}})
// CIR-NEXT:    cir.return
// CIR-NEXT:  }

// Check that we call the destructor whenever a cleanup is needed.
// FIXME: enable and check this when exceptions are fully supported.
#if 0
void canThrow();
struct HasNonTrivialDestructor {
  HasNonTrivialDestructor();
  HasNonTrivialDestructor(int);
  ~HasNonTrivialDestructor();
};

HasNonTrivialDestructor::HasNonTrivialDestructor(int)
    : HasNonTrivialDestructor() {
  canThrow();
}
#endif

// From clang/test/CodeGenCXX/cxx0x-delegating-ctors.cpp, check that virtual
// inheritance and delegating constructors interact correctly.
void other();

class Base {
public:
  Base() { squawk(); }

  virtual void squawk();
};

class Derived : public virtual Base {
public:
  Derived();
  Derived(const void *inVoid);

  virtual void squawk();
};

Derived::Derived() : Derived(nullptr) { other(); }
Derived::Derived(const void *inVoid) { squawk(); }

// Note: OGCG emits the constructors in a different order.
// OGCG: define {{.*}} void @_ZN7DerivedC2Ev(ptr {{.*}} %[[THIS_ARG:.*]], ptr {{.*}} %[[VTT_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   %[[VTT_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   store ptr %[[VTT_ARG]], ptr %[[VTT_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   %[[VTT:.*]] = load ptr, ptr %[[VTT_ADDR]]
// OGCG:   call void @_ZN7DerivedC2EPKv(ptr {{.*}} %[[THIS]], ptr {{.*}} %[[VTT]], ptr {{.*}} null)
// OGCG:   call void @_Z5otherv()
// OGCG:   ret void

// CIR:      cir.func {{.*}} @_ZN7DerivedC2EPKv(
// CIR-SAME:       %[[THIS_ARG:.*]]: !cir.ptr<!rec_Derived>
// CIR-SAME:       %[[VTT_ARG:.*]]: !cir.ptr<!cir.ptr<!void>>
// CIR-SAME:       %[[INVOID_ARG:.*]]: !cir.ptr<!void>
// CIR:        %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CIR:        %[[VTT_ADDR:.*]] = cir.alloca {{.*}} ["vtt", init]
// CIR:        %[[INVOID_ADDR:.*]] = cir.alloca {{.*}} ["inVoid", init]
// CIR:        cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:        cir.store %[[VTT_ARG]], %[[VTT_ADDR]]
// CIR:        cir.store %[[INVOID_ARG]], %[[INVOID_ADDR]]
// CIR:        %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR:        %[[VTT:.*]] = cir.load{{.*}} %[[VTT_ADDR]]
// CIR:        %[[VPTR_GLOBAL_ADDR:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 0 -> !cir.ptr<!cir.ptr<!void>>
// CIR:        %[[VPTR_PTR:.*]] = cir.cast(bitcast, %[[VPTR_GLOBAL_ADDR]] : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!cir.vptr>
// CIR:        %[[VPTR:.*]] = cir.load{{.*}} %[[VPTR_PTR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:        %[[VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_Derived> -> !cir.ptr<!cir.vptr>
// CIR:        cir.store{{.*}} %[[VPTR]], %[[VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:        %[[VPTR_BASE_ADDR:.*]] = cir.vtt.address_point %[[VTT]] : !cir.ptr<!cir.ptr<!void>>, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR:        %[[VPTR_BASE_PTR:.*]] = cir.cast(bitcast, %[[VPTR_BASE_ADDR]] : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!cir.vptr>
// CIR:        %[[VPTR_BASE:.*]] = cir.load{{.*}} %[[VPTR_BASE_PTR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:        %[[VPTR_DERIVED_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_Derived> -> !cir.ptr<!cir.vptr>
// CIR:        %[[VPTR_DERIVED:.*]] = cir.load{{.*}} %[[VPTR_DERIVED_ADDR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:        %[[VPTR_DERIVED_AS_I8PTR:.*]] = cir.cast(bitcast, %[[VPTR_DERIVED]] : !cir.vptr), !cir.ptr<!u8i>
// CIR:        %[[BASE_LOC_OFFSET:.*]] = cir.const #cir.int<-32> : !s64i
// CIR:        %[[BASE_OFFSET_PTR:.*]] = cir.ptr_stride(%[[VPTR_DERIVED_AS_I8PTR]] : !cir.ptr<!u8i>, %[[BASE_LOC_OFFSET]] : !s64i), !cir.ptr<!u8i>
// CIR:        %[[BASE_OFFSET_I64PTR:.*]] = cir.cast(bitcast, %[[BASE_OFFSET_PTR]] : !cir.ptr<!u8i>), !cir.ptr<!s64i>
// CIR:        %[[BASE_OFFSET:.*]] = cir.load{{.*}} %[[BASE_OFFSET_I64PTR]] : !cir.ptr<!s64i>, !s64i
// CIR:        %[[THIS_AS_I8PTR:.*]] = cir.cast(bitcast, %[[THIS]] : !cir.ptr<!rec_Derived>), !cir.ptr<!u8i>
// CIR:        %[[BASE_PTR:.*]] = cir.ptr_stride(%[[THIS_AS_I8PTR]] : !cir.ptr<!u8i>, %[[BASE_OFFSET]] : !s64i), !cir.ptr<!u8i>
// CIR:        %[[BASE_AS_I8PTR:.*]] = cir.cast(bitcast, %[[BASE_PTR]] : !cir.ptr<!u8i>), !cir.ptr<!rec_Derived>
// CIR:        %[[BASE_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[BASE_AS_I8PTR]] : !cir.ptr<!rec_Derived> -> !cir.ptr<!cir.vptr>
// CIR:        cir.store{{.*}} %[[VPTR_BASE]], %[[BASE_VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:        %[[VPTR_BASE_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_Derived> -> !cir.ptr<!cir.vptr>
// CIR:        %[[VPTR_BASE:.*]] = cir.load{{.*}} %[[VPTR_BASE_ADDR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:        %[[SQUAWK_FN_ADDR:.*]] = cir.vtable.get_virtual_fn_addr %[[VPTR_BASE]][0] : !cir.vptr -> !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_Derived>)>>>
// CIR:        %[[SQUAWK:.*]] = cir.load{{.*}} %[[SQUAWK_FN_ADDR]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_Derived>)>>>, !cir.ptr<!cir.func<(!cir.ptr<!rec_Derived>)>>
// CIR:        cir.call %[[SQUAWK]](%[[THIS]]) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_Derived>)>>, !cir.ptr<!rec_Derived>) -> ()
// CIR:        cir.return

// LLVM: define {{.*}} void @_ZN7DerivedC2EPKv(ptr %[[THIS_ARG:.*]], ptr %[[VTT_ARG:.*]], ptr %[[INVOID_ARG:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   %[[VTT_ADDR:.*]] = alloca ptr
// LLVM:   %[[INVOID_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   store ptr %[[VTT_ARG]], ptr %[[VTT_ADDR]]
// LLVM:   store ptr %[[INVOID_ARG]], ptr %[[INVOID_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   %[[VTT:.*]] = load ptr, ptr %[[VTT_ADDR]]
// LLVM:   %[[VPTR:.*]] = load ptr, ptr %[[VTT]]
// LLVM:   store ptr %[[VPTR]], ptr %[[THIS]]
// LLVM:   %[[VTT_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i32 1
// LLVM:   %[[VPTR_BASE:.*]] = load ptr, ptr %[[VTT_ADDR]]
// LLVM:   %[[VPTR:.*]] = load ptr, ptr %[[THIS]]
// LLVM:   %[[BASE_OFFSET_ADDR:.*]] = getelementptr i8, ptr %[[VPTR]], i64 -32
// LLVM:   %[[BASE_OFFSET:.*]] = load i64, ptr %[[BASE_OFFSET_ADDR]]
// LLVM:   %[[BASE_PTR:.*]] = getelementptr i8, ptr %[[THIS]], i64 %[[BASE_OFFSET]]
// LLVM:   store ptr %[[VPTR_BASE]], ptr %[[BASE_PTR]]
// LLVM:   %[[VPTR:.*]] = load ptr, ptr %[[THIS]]
// LLVM:   %[[SQUAWK_FN_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VPTR]], i32 0
// LLVM:   %[[SQUAWK:.*]] = load ptr, ptr %[[SQUAWK_FN_ADDR]]
// LLVM:   call void %[[SQUAWK]](ptr %[[THIS]])
// LLVM:   ret void

// OGCG: define {{.*}} void @_ZN7DerivedC2EPKv(ptr {{.*}} %[[THIS_ARG:.*]], ptr {{.*}} %[[VTT_ARG:.*]], ptr {{.*}} %[[INVOID_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   %[[VTT_ADDR:.*]] = alloca ptr
// OGCG:   %[[INVOID_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   store ptr %[[VTT_ARG]], ptr %[[VTT_ADDR]]
// OGCG:   store ptr %[[INVOID_ARG]], ptr %[[INVOID_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   %[[VTT:.*]] = load ptr, ptr %[[VTT_ADDR]]
// OGCG:   %[[VPTR:.*]] = load ptr, ptr %[[VTT]]
// OGCG:   store ptr %[[VPTR]], ptr %[[THIS]]
// OGCG:   %[[VTT_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTT]], i64 1
// OGCG:   %[[VPTR_BASE:.*]] = load ptr, ptr %[[VTT_ADDR]]
// OGCG:   %[[VPTR:.*]] = load ptr, ptr %[[THIS]]
// OGCG:   %[[BASE_OFFSET_ADDR:.*]] = getelementptr i8, ptr %[[VPTR]], i64 -32
// OGCG:   %[[BASE_OFFSET:.*]] = load i64, ptr %[[BASE_OFFSET_ADDR]]
// OGCG:   %[[BASE_PTR:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i64 %[[BASE_OFFSET]]
// OGCG:   store ptr %[[VPTR_BASE]], ptr %[[BASE_PTR]]
// OGCG:   %[[VPTR:.*]] = load ptr, ptr %[[THIS]]
// OGCG:   %[[SQUAWK_FN_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VPTR]], i64 0
// OGCG:   %[[SQUAWK:.*]] = load ptr, ptr %[[SQUAWK_FN_ADDR]]
// OGCG:   call void %[[SQUAWK]](ptr {{.*}} %[[THIS]])
// OGCG:   ret void

// CIR: cir.func {{.*}} @_ZN7DerivedC2Ev(%[[THIS_ARG:.*]]: !cir.ptr<!rec_Derived> {{.*}}, %[[VTT_ARG:.*]]: !cir.ptr<!cir.ptr<!void>> {{.*}})
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CIR:   %[[VTT_ADDR:.*]] = cir.alloca {{.*}} ["vtt", init]
// CIR:   cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:   cir.store %[[VTT_ARG]], %[[VTT_ADDR]]
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR:   %[[VTT:.*]] = cir.load {{.*}} %[[VTT_ADDR]]
// CIR:   %[[NULLPTR:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR:   cir.call @_ZN7DerivedC2EPKv(%[[THIS]], %[[VTT]], %[[NULLPTR]]) : (!cir.ptr<!rec_Derived>, !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>) -> ()
// CIR:   cir.call @_Z5otherv() : () -> ()
// CIR:   cir.return

// LLVM: define {{.*}} void @_ZN7DerivedC2Ev(ptr %[[THIS_ARG:.*]], ptr %[[VTT_ARG:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   %[[VTT_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   store ptr %[[VTT_ARG]], ptr %[[VTT_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   %[[VTT:.*]] = load ptr, ptr %[[VTT_ADDR]]
// LLVM:   call void @_ZN7DerivedC2EPKv(ptr %[[THIS]], ptr %[[VTT]], ptr null)
// LLVM:   call void @_Z5otherv()
// LLVM:   ret void

// See above for the OGCG _ZN7DerivedC2Ev constructor.

// TODO: add the Derived C1 constructors and Base

// CIR: cir.func {{.*}} @_ZN4BaseC2Ev(%[[THIS_ARG:.*]]: !cir.ptr<!rec_Base> {{.*}})
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CIR:   cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR:   %[[VTT_ADDR_POINT:.*]] = cir.vtable.address_point(@_ZTV4Base, address_point = <index = 0, offset = 2>) : !cir.vptr
// CIR:   %[[VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_Base> -> !cir.ptr<!cir.vptr>
// CIR:   cir.store{{.*}} %[[VTT_ADDR_POINT]], %[[VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:   %[[VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_Base> -> !cir.ptr<!cir.vptr>
// CIR:   %[[VPTR:.*]] = cir.load{{.*}} %[[VPTR_ADDR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:   %[[VIRTUAL_FN_ADDR:.*]] = cir.vtable.get_virtual_fn_addr %[[VPTR]][0] : !cir.vptr -> !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_Base>)>>>
// CIR:   %[[VIRTUAL_FN:.*]] = cir.load{{.*}} %[[VIRTUAL_FN_ADDR]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_Base>)>>>, !cir.ptr<!cir.func<(!cir.ptr<!rec_Base>)>>
// CIR:   cir.call %[[VIRTUAL_FN]](%[[THIS]]) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_Base>)>>, !cir.ptr<!rec_Base>) -> ()
// CIR:   cir.return

// LLVM: define {{.*}} void @_ZN4BaseC2Ev(ptr %[[THIS_ARG:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV4Base, i64 16), ptr %[[THIS]]
// LLVM:   %[[VPTR:.*]] = load ptr, ptr %[[THIS]]
// LLVM:   %[[SQUAWK_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VPTR]], i32 0
// LLVM:   %[[SQUAWK:.*]] = load ptr, ptr %[[SQUAWK_ADDR]]
// LLVM:   call void %[[SQUAWK]](ptr %[[THIS]])
// LLVM:   ret void

// The base constructor is emitted last for OGCG.
// The _ZN7DerivedC1Ev constructor is emitted earlier for OGCG.

// OGCG: define {{.*}} void @_ZN7DerivedC1Ev(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   call void @_ZN7DerivedC1EPKv(ptr {{.*}} %[[THIS]], ptr {{.*}} null)
// OGCG:   call void @_Z5otherv()
// OGCG:   ret void

// CIR: cir.func {{.*}} @_ZN7DerivedC1EPKv(%[[THIS_ARG:.*]]: !cir.ptr<!rec_Derived> {{.*}}, %[[INVOID_ARG:.*]]: !cir.ptr<!void> {{.*}})
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CIR:   %[[INVOID_ADDR:.*]] = cir.alloca {{.*}} ["inVoid", init]
// CIR:   cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:   cir.store %[[INVOID_ARG]], %[[INVOID_ADDR]]
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR:   %[[BASE:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR:   cir.call @_ZN4BaseC2Ev(%[[BASE]])
// CIR:   %[[VPTR_GLOBAL:.*]] = cir.vtable.address_point(@_ZTV7Derived, address_point = <index = 0, offset = 4>) : !cir.vptr
// CIR:   %[[VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_Derived> -> !cir.ptr<!cir.vptr>
// CIR:   cir.store{{.*}} %[[VPTR_GLOBAL]], %[[VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:   %[[VPTR_GLOBAL:.*]] = cir.vtable.address_point(@_ZTV7Derived, address_point = <index = 0, offset = 4>) : !cir.vptr
// CIR:   %[[VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_Derived> -> !cir.ptr<!cir.vptr>
// CIR:   cir.store{{.*}} %[[VPTR_GLOBAL]], %[[VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:   %[[VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_Derived> -> !cir.ptr<!cir.vptr>
// CIR:   %[[VPTR:.*]] = cir.load{{.*}} %[[VPTR_ADDR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:   %[[SQUAWK_ADDR:.*]] = cir.vtable.get_virtual_fn_addr %[[VPTR]][0] : !cir.vptr -> !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_Derived>)>>>
// CIR:   %[[SQUAWK:.*]] = cir.load{{.*}} %[[SQUAWK_ADDR]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_Derived>)>>>, !cir.ptr<!cir.func<(!cir.ptr<!rec_Derived>)>>
// CIR:   cir.call %[[SQUAWK]](%[[THIS]])
// CIR:   cir.return

// LLVM: define {{.*}} void @_ZN7DerivedC1EPKv(ptr %[[THIS_ARG:.*]], ptr %[[INVOID_ARG:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   %[[INVOID_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   store ptr %[[INVOID_ARG]], ptr %[[INVOID_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   call void @_ZN4BaseC2Ev(ptr %[[THIS]])
// LLVM:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV7Derived, i64 32), ptr %[[THIS]]
// LLVM:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV7Derived, i64 32), ptr %[[THIS]]
// LLVM:   %[[VPTR:.*]] = load ptr, ptr %[[THIS]]
// LLVM:   %[[SQUAWK_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VPTR]], i32 0
// LLVM:   %[[SQUAWK:.*]] = load ptr, ptr %[[SQUAWK_ADDR]]
// LLVM:   call void %[[SQUAWK]](ptr %[[THIS]])
// LLVM:   ret void

// OGCG: define {{.*}} void @_ZN7DerivedC1EPKv(ptr {{.*}} %[[THIS_ARG:.*]], ptr {{.*}} %[[INVOID_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   %[[INVOID_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   store ptr %[[INVOID_ARG]], ptr %[[INVOID_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   call void @_ZN4BaseC2Ev(ptr {{.*}} %[[THIS]])
// OGCG:   store ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV7Derived, i32 0, i32 0, i32 4), ptr %[[THIS]]
// OGCG:   store ptr getelementptr inbounds inrange(-32, 8) ({ [5 x ptr] }, ptr @_ZTV7Derived, i32 0, i32 0, i32 4), ptr %[[THIS]]
// OGCG:   %[[VPTR:.*]] = load ptr, ptr %[[THIS]]
// OGCG:   %[[SQUAWK_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VPTR]], i64 0
// OGCG:   %[[SQUAWK:.*]] = load ptr, ptr %[[SQUAWK_ADDR]]
// OGCG:   call void %[[SQUAWK]](ptr {{.*}} %[[THIS]])
// OGCG:   ret void

// CIR: cir.func {{.*}} @_ZN7DerivedC1Ev(%[[THIS_ARG:.*]]: !cir.ptr<!rec_Derived> {{.*}})
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CIR:   cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR:   %[[NULLPTR:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR:   cir.call @_ZN7DerivedC1EPKv(%[[THIS]], %[[NULLPTR]]) : (!cir.ptr<!rec_Derived>, !cir.ptr<!void>) -> ()
// CIR:   cir.call @_Z5otherv() : () -> ()
// CIR:   cir.return

// LLVM: define {{.*}} void @_ZN7DerivedC1Ev(ptr %[[THIS_ARG:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   call void @_ZN7DerivedC1EPKv(ptr %[[THIS]], ptr null)
// LLVM:   call void @_Z5otherv()
// LLVM:   ret void

// The _ZN7DerivedC1Ev constructor was emitted earlier for OGCG.

// OGCG: define {{.*}} void @_ZN4BaseC2Ev(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV4Base, i32 0, i32 0, i32 2), ptr %[[THIS]]
// OGCG:   %[[VPTR:.*]] = load ptr, ptr %[[THIS]]
// OGCG:   %[[SQUAWK_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VPTR]], i64 0
// OGCG:   %[[SQUAWK:.*]] = load ptr, ptr %[[SQUAWK_ADDR]]
// OGCG:   call void %[[SQUAWK]](ptr {{.*}} %[[THIS]])
// OGCG:   ret void
