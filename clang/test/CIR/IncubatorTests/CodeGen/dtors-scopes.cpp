// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -std=c++20 -fclangir -emit-cir %s -o %t2.cir
// RUN: FileCheck --input-file=%t2.cir %s --check-prefix=DTOR_BODY

extern "C" int printf(char const*, ...);
struct C {
  C()  { printf("++A\n"); }
  ~C()  { printf("--A\n"); }
};
void dtor1() {
  {
    C c;
  }
  printf("Done\n");
}

// CHECK: cir.func {{.*}} @_Z5dtor1v()
// CHECK:   cir.scope {
// CHECK:     [[C:%.*]] = cir.alloca !rec_C, !cir.ptr<!rec_C>, ["c", init] {alignment = 1 : i64}
// CHECK:     cir.call @_ZN1CC2Ev([[C]]) : (!cir.ptr<!rec_C>) -> ()
// CHECK:     cir.call @_ZN1CD2Ev([[C]]){{.*}} : (!cir.ptr<!rec_C>) -> ()
// CHECK:   }

// DTOR_BODY: cir.func {{.*}} @_ZN1CD2Ev{{.*}}{
// DTOR_BODY:   [[STR:%.*]] = cir.get_global @".str{{.*}}"
// DTOR_BODY:   [[PTR:%.*]] = cir.cast array_to_ptrdecay [[STR]]
// DTOR_BODY:   {{.*}} = cir.call @printf([[PTR]])
// DTOR_BODY:   cir.return

// DTOR_BODY: cir.func {{.*}} @_ZN1CD1Ev({{.*}}!cir.ptr<!rec_C>

// DTOR_BODY:   cir.call @_ZN1CD2Ev
// DTOR_BODY:   cir.return
// DTOR_BODY: }
