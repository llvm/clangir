// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fenable-clangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

unsigned id0() {
  unsigned a = 1;
  return ++a;
}

// CHECK: cir.func @_Z3id0v() -> i32 {
// CHECK: %[[#RET:]] = cir.alloca i32, cir.ptr <i32>, ["__retval", uninitialized]
// CHECK: %[[#A:]] = cir.alloca i32, cir.ptr <i32>, ["a", cinit]
// CHECK: %[[#BEFORE_A:]] = cir.load %[[#A]]
// CHECK: %[[#AFTER_A:]] = cir.unary(inc, %[[#BEFORE_A]])
// CHECK: cir.store %[[#AFTER_A]], %[[#A]]
// CHECK: cir.store %[[#AFTER_A]], %[[#RET]]


unsigned id1() {
  unsigned a = 1;
  return --a;
}

// CHECK: cir.func @_Z3id1v() -> i32 {
// CHECK: %[[#RET:]] = cir.alloca i32, cir.ptr <i32>, ["__retval", uninitialized]
// CHECK: %[[#A:]] = cir.alloca i32, cir.ptr <i32>, ["a", cinit]
// CHECK: %[[#BEFORE_A:]] = cir.load %[[#A]]
// CHECK: %[[#AFTER_A:]] = cir.unary(dec, %[[#BEFORE_A]])
// CHECK: cir.store %[[#AFTER_A]], %[[#A]]
// CHECK: cir.store %[[#AFTER_A]], %[[#RET]]

unsigned id2() {
  unsigned a = 1;
  return a++;
}

// CHECK: cir.func @_Z3id2v() -> i32 {
// CHECK: %[[#RET:]] = cir.alloca i32, cir.ptr <i32>, ["__retval", uninitialized]
// CHECK: %[[#A:]] = cir.alloca i32, cir.ptr <i32>, ["a", cinit]
// CHECK: %[[#BEFORE_A:]] = cir.load %[[#A]]
// CHECK: %[[#AFTER_A:]] = cir.unary(inc, %[[#BEFORE_A]])
// CHECK: cir.store %[[#AFTER_A]], %[[#A]]
// CHECK: cir.store %[[#BEFORE_A]], %[[#RET]]

unsigned id3() {
  unsigned a = 1;
  return a--;
}

// CHECK: cir.func @_Z3id3v() -> i32 {
// CHECK: %[[#RET:]] = cir.alloca i32, cir.ptr <i32>, ["__retval", uninitialized]
// CHECK: %[[#A:]] = cir.alloca i32, cir.ptr <i32>, ["a", cinit]
// CHECK: %[[#BEFORE_A:]] = cir.load %[[#A]]
// CHECK: %[[#AFTER_A:]] = cir.unary(dec, %[[#BEFORE_A]])
// CHECK: cir.store %[[#AFTER_A]], %[[#A]]
// CHECK: cir.store %[[#BEFORE_A]], %[[#RET]]
