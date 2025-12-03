// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-threadsafe-statics -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-threadsafe-statics -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Test static local variable with non-trivial destructor

class A {
public:
  ~A();
};

void test_static_with_dtor() {
  static A obj;
}

// CIR: cir.global "private" internal dso_local @_ZZ21test_static_with_dtorvE3obj = #cir.zero : !rec_A
// CIR: cir.global "private" internal dso_local @_ZGVZ21test_static_with_dtorvE3obj = #cir.int<0> : !u8i

// CIR-LABEL: cir.func {{.*}} @_Z21test_static_with_dtorv()
// CIR: %[[OBJ:.*]] = cir.get_global @_ZZ21test_static_with_dtorvE3obj : !cir.ptr<!rec_A>
// CIR: %[[GUARD:.*]] = cir.get_global @_ZGVZ21test_static_with_dtorvE3obj : !cir.ptr<!u8i>
// CIR: %[[GUARD_VAL:.*]] = cir.load %[[GUARD]] : !cir.ptr<!u8i>, !u8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
// CIR: %[[NEED_INIT:.*]] = cir.cmp(eq, %[[GUARD_VAL]], %[[ZERO]]) : !u8i, !cir.bool
// CIR: cir.if %[[NEED_INIT]] {
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !u8i
// CIR:   cir.store %[[ONE]], %[[GUARD]] : !u8i, !cir.ptr<!u8i>
// CIR: }
// CIR: cir.return

// LLVM-LABEL: define dso_local void @_Z21test_static_with_dtorv()
// LLVM: %[[GUARD_VAL:.*]] = load i8, ptr @_ZGVZ21test_static_with_dtorvE3obj
// LLVM: %[[NEED_INIT:.*]] = icmp eq i8 %[[GUARD_VAL]], 0
// LLVM: br i1 %[[NEED_INIT]], label %[[INIT_BLOCK:.*]], label %[[END_BLOCK:.*]]
// LLVM: [[INIT_BLOCK]]:
// LLVM: store i8 1, ptr @_ZGVZ21test_static_with_dtorvE3obj
// LLVM: br label %[[END_BLOCK]]
// LLVM: [[END_BLOCK]]:

// Test static local variable with dynamic initialization

class B {
public:
  B(int);
  ~B();
};

void test_static_with_init() {
  static B obj(42);
}

// CIR: cir.global "private" internal dso_local @_ZGVZ21test_static_with_initvE3obj = #cir.int<0> : !u8i

// CIR-LABEL: cir.func {{.*}} @_Z21test_static_with_initv()
// CIR: %[[OBJ:.*]] = cir.get_global @_ZZ21test_static_with_initvE3obj : !cir.ptr<!rec_B>
// CIR: %[[GUARD:.*]] = cir.get_global @_ZGVZ21test_static_with_initvE3obj : !cir.ptr<!u8i>
// CIR: %[[GUARD_VAL:.*]] = cir.load %[[GUARD]] : !cir.ptr<!u8i>, !u8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
// CIR: %[[NEED_INIT:.*]] = cir.cmp(eq, %[[GUARD_VAL]], %[[ZERO]]) : !u8i, !cir.bool
// CIR: cir.if %[[NEED_INIT]] {
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !u8i
// CIR:   %[[OBJ2:.*]] = cir.get_global @_ZZ21test_static_with_initvE3obj : !cir.ptr<!rec_B>
// CIR:   %[[INIT_VAL:.*]] = cir.const #cir.int<42> : !s32i
// CIR:   cir.call @_ZN1BC1Ei(%[[OBJ2]], %[[INIT_VAL]]) : (!cir.ptr<!rec_B>, !s32i) -> ()
// CIR:   cir.call @_ZN1BD1Ev({{.*}}) : (!cir.ptr<!rec_B>) -> ()
// CIR:   cir.store %[[ONE]], %[[GUARD]] : !u8i, !cir.ptr<!u8i>
// CIR: }
// CIR: cir.return

// LLVM-LABEL: define dso_local void @_Z21test_static_with_initv()
// LLVM: %[[GUARD_VAL:.*]] = load i8, ptr @_ZGVZ21test_static_with_initvE3obj
// LLVM: %[[NEED_INIT:.*]] = icmp eq i8 %[[GUARD_VAL]], 0
// LLVM: br i1 %[[NEED_INIT]], label %[[INIT_BLOCK:.*]], label %[[END_BLOCK:.*]]
// LLVM: [[INIT_BLOCK]]:
// LLVM: call {{.*}} @_ZN1BC1Ei
// LLVM: store i8 1, ptr @_ZGVZ21test_static_with_initvE3obj
// LLVM: br label %[[END_BLOCK]]
// LLVM: [[END_BLOCK]]:

// Test static local variable with constant initializer but non-trivial destructor

class C {
public:
  C() = default;
  ~C();
  int x = 0;
};

void test_static_const_init_with_dtor() {
  static C obj;
}

// CIR: cir.global "private" internal dso_local @_ZGVZ32test_static_const_init_with_dtorvE3obj = #cir.int<0> : !u8i

// CIR-LABEL: cir.func {{.*}} @_Z32test_static_const_init_with_dtorv()
// CIR: %[[OBJ:.*]] = cir.get_global @_ZZ32test_static_const_init_with_dtorvE3obj : !cir.ptr<!rec_C>
// CIR: %[[GUARD:.*]] = cir.get_global @_ZGVZ32test_static_const_init_with_dtorvE3obj : !cir.ptr<!u8i>
// CIR: %[[GUARD_VAL:.*]] = cir.load %[[GUARD]] : !cir.ptr<!u8i>, !u8i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
// CIR: %[[NEED_INIT:.*]] = cir.cmp(eq, %[[GUARD_VAL]], %[[ZERO]]) : !u8i, !cir.bool
// CIR: cir.if %[[NEED_INIT]] {
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !u8i
// CIR:   cir.store %[[ONE]], %[[GUARD]] : !u8i, !cir.ptr<!u8i>
// CIR: }
// CIR: cir.return
// Test static variable with reference parameter and non-trivial destructor

class RefHolder {
public:
  RefHolder(const int&);
  ~RefHolder();
};

int external_value = 42;

void test_static_with_ref_param() {
  static RefHolder rh(external_value);
}

// CIR-LABEL: cir.func {{.*}} @_Z26test_static_with_ref_paramv()
// CIR: cir.get_global @_ZGVZ26test_static_with_ref_paramvE2rh : !cir.ptr<!u8i>
// CIR: cir.load
// CIR: cir.cmp(eq,
// CIR: cir.if
// CIR: cir.call @_ZN9RefHolderC1ERKi
// CIR: cir.call @_ZN9RefHolderD1Ev
// CIR: cir.return

// LLVM-LABEL: define dso_local void @_Z26test_static_with_ref_paramv()
// LLVM: load i8, ptr @_ZGVZ26test_static_with_ref_paramvE2rh
// LLVM: icmp eq i8
// LLVM: br i1
// LLVM: call {{.*}} @_ZN9RefHolderC1ERKi
// LLVM: store i8 1, ptr @_ZGVZ26test_static_with_ref_paramvE2rh

