// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

enum class EFMode { Always, Verbose };

class PSEvent {
 public:
  PSEvent(
      EFMode m,
      const char* n);
  ~PSEvent();

 private:
  const char* n;
  EFMode m;
};

void blue() {
  PSEvent p(EFMode::Verbose, __FUNCTION__);
}

class A
{
public:
    A() noexcept {}
    A(const A&) noexcept = default;

    virtual ~A() noexcept;
    virtual const char* quack() const noexcept;
};

class B : public A
{
public:
    virtual ~B() noexcept {}
};

// Class A
// CHECK: ![[ClassA:rec_.*]] = !cir.record<class "A" {!cir.ptr<!cir.ptr<!cir.func<() -> !u32i>>>} #cir.record.decl.ast>

// Class B
// CHECK: ![[ClassB:rec_.*]] = !cir.record<class "B" {![[ClassA]]}>

// CHECK: cir.func @_Z4bluev()
// CHECK:   %0 = cir.alloca !rec_PSEvent, !cir.ptr<!rec_PSEvent>, ["p", init] {alignment = 8 : i64}
// CHECK:   %1 = cir.const #cir.int<1> : !s32i
// CHECK:   %2 = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 5>>
// CHECK:   %3 = cir.cast(array_to_ptrdecay, %2 : !cir.ptr<!cir.array<!s8i x 5>>), !cir.ptr<!s8i>
// CHECK:   cir.call @_ZN7PSEventC1E6EFModePKc(%0, %1, %3) : (!cir.ptr<!rec_PSEvent>, !s32i, !cir.ptr<!s8i>) -> ()
// CHECK:   cir.return
// CHECK: }

// @B::~B() #1 definition call into base @A::~A()
// CHECK:  cir.func linkonce_odr @_ZN1BD2Ev{{.*}}{
// CHECK:    cir.call @_ZN1AD2Ev(

// void foo()
// CHECK: cir.func @_Z3foov()
// CHECK:   cir.scope {
// CHECK:     cir.call @_ZN1BC2Ev(%0) : (!cir.ptr<!rec_B>) -> ()
// CHECK:     cir.call @_ZN1BD2Ev(%0) : (!cir.ptr<!rec_B>) -> ()

// operator delete(void*) declaration
// CHECK:   cir.func private @_ZdlPvm(!cir.ptr<!void>, !u64i)

// B dtor => @B::~B() #2
// Calls dtor #1
// Calls operator delete
//
// CHECK:   cir.func linkonce_odr @_ZN1BD0Ev(%arg0: !cir.ptr<![[ClassB]]>
// CHECK:     %0 = cir.alloca !cir.ptr<![[ClassB]]>, !cir.ptr<!cir.ptr<![[ClassB]]>>, ["this", init] {alignment = 8 : i64}
// CHECK:     cir.store %arg0, %0 : !cir.ptr<![[ClassB]]>, !cir.ptr<!cir.ptr<![[ClassB]]>>
// CHECK:     %1 = cir.load %0 : !cir.ptr<!cir.ptr<![[ClassB]]>>, !cir.ptr<![[ClassB]]>
// CHECK:     cir.call @_ZN1BD2Ev(%1) : (!cir.ptr<![[ClassB]]>) -> ()
// CHECK:     %2 = cir.cast(bitcast, %1 : !cir.ptr<![[ClassB]]>), !cir.ptr<!void>
// CHECK:     cir.call @_ZdlPvm(%2, %3) : (!cir.ptr<!void>, !u64i) -> ()
// CHECK:     cir.return
// CHECK:   }

void foo() { B(); }

class A2 {
public:
  ~A2();
};

struct B2 {
  template <typename> using C = A2;
};

struct E {
  typedef B2::C<int> D;
};

struct F {
  F(long, A2);
};

class G : F {
public:
  A2 h;
  G(long) : F(i(), h) {}
  long i() { k(E::D()); };
  long k(E::D);
};

int j;
void m() { G l(j); }

// CHECK: cir.func private @_ZN1G1kE2A2(!cir.ptr<!rec_G>, !rec_A2) -> !s64i
// CHECK: cir.func linkonce_odr @_ZN1G1iEv(%arg0: !cir.ptr<!rec_G>
// CHECK:   %[[V0:.*]] = cir.alloca !cir.ptr<!rec_G>, !cir.ptr<!cir.ptr<!rec_G>>, ["this", init] {alignment = 8 : i64}
// CHECK:   %[[V1:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["__retval"] {alignment = 8 : i64}
// CHECK:   cir.store %arg0, %[[V0]] : !cir.ptr<!rec_G>, !cir.ptr<!cir.ptr<!rec_G>>
// CHECK:   %[[V2:.*]] = cir.load %[[V0]] : !cir.ptr<!cir.ptr<!rec_G>>, !cir.ptr<!rec_G>
// CHECK:   %[[V3:.*]] = cir.scope {
// CHECK:     %[[V4:.*]] = cir.alloca !rec_A2, !cir.ptr<!rec_A2>, ["agg.tmp0"] {alignment = 1 : i64}
// CHECK:     cir.call @_ZN2A2C2Ev(%[[V4]]) : (!cir.ptr<!rec_A2>) -> ()
// CHECK:     %[[V5:.*]] = cir.load %[[V4]] : !cir.ptr<!rec_A2>, !rec_A2
// CHECK:     %[[V6:.*]] = cir.call @_ZN1G1kE2A2(%[[V2]], %[[V5]]) : (!cir.ptr<!rec_G>, !rec_A2) -> !s64i
// CHECK:     cir.call @_ZN2A2D1Ev(%[[V4]]) : (!cir.ptr<!rec_A2>) -> ()
// CHECK:     cir.yield %[[V6]] : !s64i
// CHECK:   } : !s64i
// CHECK:   cir.trap
// CHECK: }
