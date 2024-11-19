// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

class String {
  char *storage{nullptr};
  long size;
  long capacity;

public:
  String() : size{0} {}
  String(int size) : size{size} {}
  String(const char *s) {}
};

void test() {
  String s1{};
  String s2{1};
  String s3{"abcdefghijklmnop"};
}

//      CHECK: cir.func linkonce_odr @_ZN6StringC2Ev
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22class2EString22>
// CHECK-NEXT:   cir.store %arg0, %0
// CHECK-NEXT:   %1 = cir.load %0
// CHECK-NEXT:   %2 = "cir.struct_element_addr"(%1) <{member_name = "storage"}>
// CHECK-NEXT:   %3 = cir.cst(#cir.null : !cir.ptr<i8>) : !cir.ptr<i8>
// CHECK-NEXT:   cir.store %3, %2 : !cir.ptr<i8>, cir.ptr <!cir.ptr<i8>>
// CHECK-NEXT:   %4 = "cir.struct_element_addr"(%1) <{member_name = "size"}> : (!cir.ptr<!ty_22class2EString22>) -> !cir.ptr<i64>
// CHECK-NEXT:   %5 = cir.cst(0 : i32) : i32
// CHECK-NEXT:   %6 = cir.cast(integral, %5 : i32), i64
// CHECK-NEXT:   cir.store %6, %4 : i64, cir.ptr <i64>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
//      CHECK: cir.func linkonce_odr @_ZN6StringC2Ei
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22class2EString22>
// CHECK-NEXT:   %1 = cir.alloca i32, cir.ptr <i32>, ["size", init]
// CHECK-NEXT:   cir.store %arg0, %0
// CHECK-NEXT:   cir.store %arg1, %1
// CHECK-NEXT:   %2 = cir.load %0
// CHECK-NEXT:   %3 = "cir.struct_element_addr"(%2) <{member_name = "storage"}>
// CHECK-NEXT:   %4 = cir.cst(#cir.null : !cir.ptr<i8>)
// CHECK-NEXT:   cir.store %4, %3
// CHECK-NEXT:   %5 = "cir.struct_element_addr"(%2) <{member_name = "size"}> : (!cir.ptr<!ty_22class2EString22>) -> !cir.ptr<i64>
// CHECK-NEXT:   %6 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:   %7 = cir.cast(integral, %6 : i32), i64
// CHECK-NEXT:   cir.store %7, %5 : i64, cir.ptr <i64>
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

//      CHECK: cir.func linkonce_odr @_ZN6StringC2EPKc
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22class2EString22>, cir.ptr <!cir.ptr<!ty_22class2EString22>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca !cir.ptr<i8>, cir.ptr <!cir.ptr<i8>>, ["s", init] {alignment = 8 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_22class2EString22>, cir.ptr <!cir.ptr<!ty_22class2EString22>>
// CHECK-NEXT:   cir.store %arg1, %1 : !cir.ptr<i8>, cir.ptr <!cir.ptr<i8>>
// CHECK-NEXT:   %2 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22class2EString22>>, !cir.ptr<!ty_22class2EString22>
// CHECK-NEXT:   %3 = "cir.struct_element_addr"(%2) <{member_name = "storage"}> : (!cir.ptr<!ty_22class2EString22>) -> !cir.ptr<!cir.ptr<i8>>
// CHECK-NEXT:   %4 = cir.cst(#cir.null : !cir.ptr<i8>) : !cir.ptr<i8>
// CHECK-NEXT:   cir.store %4, %3 : !cir.ptr<i8>, cir.ptr <!cir.ptr<i8>>
// CHECK-NEXT:   cir.return

//      CHECK: cir.func linkonce_odr @_ZN6StringC1EPKc
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22class2EString22>, cir.ptr <!cir.ptr<!ty_22class2EString22>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca !cir.ptr<i8>, cir.ptr <!cir.ptr<i8>>, ["s", init] {alignment = 8 : i64}
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_22class2EString22>, cir.ptr <!cir.ptr<!ty_22class2EString22>>
// CHECK-NEXT:   cir.store %arg1, %1 : !cir.ptr<i8>, cir.ptr <!cir.ptr<i8>>
// CHECK-NEXT:   %2 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22class2EString22>>, !cir.ptr<!ty_22class2EString22>
// CHECK-NEXT:   %3 = cir.load %1 : cir.ptr <!cir.ptr<i8>>, !cir.ptr<i8>
// CHECK-NEXT:   cir.call @_ZN6StringC2EPKc(%2, %3) : (!cir.ptr<!ty_22class2EString22>, !cir.ptr<i8>) -> ()
// CHECK-NEXT:   cir.return

// CHECK: cir.func @_Z4testv() {
// CHECK:   cir.call @_ZN6StringC1Ev(%0) : (!cir.ptr<!ty_22class2EString22>) -> ()
// CHECK:   cir.call @_ZN6StringC1Ei(%1, %3) : (!cir.ptr<!ty_22class2EString22>, i32) -> ()
// CHECK:   cir.call @_ZN6StringC1EPKc(%2, %5) : (!cir.ptr<!ty_22class2EString22>, !cir.ptr<i8>) -> ()
