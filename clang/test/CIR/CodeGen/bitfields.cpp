// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct __long {
  struct __attribute__((__packed__)) {
      unsigned __is_long_ : 1;
      unsigned __cap_ : sizeof(unsigned) * 8 - 1;
  };
  unsigned __size_;
  unsigned *__data_;
};

void m() {
  __long l;
}

// CHECK: !ty_22struct2Eanon22 = !cir.struct<"struct.anon", !u32i, #cir.recdecl.ast>
// CHECK: !ty_22struct2E__long22 = !cir.struct<"struct.__long", !ty_22struct2Eanon22, !u32i, !cir.ptr<!u32i>>


struct S {
  int a : 4;
  int b : 27;
  int c : 17;
  int d : 2;
  int e : 15;
}; // 65 bits in total, i.e. mote than 64

void store_field() {  
  S s;
  s.a = 3;
} 

// CHECK: cir.func @_Z11store_field
// CHECK: %1 = cir.const(#cir.int<3> : !s32i) : !s32i 
// CHECK: %2 = cir.cast(bitcast, %0 : !cir.ptr<!ty_22struct2ES22>), !cir.ptr<!u32i> 
// CHECK: %3 = cir.cast(integral, %1 : !s32i), !u32i 
// CHECK: %4 = cir.load %2 : cir.ptr <!u32i>, !u32i 
// CHECK: %5 = cir.const(#cir.int<15> : !u32i) : !u32i 
// CHECK: %6 = cir.binop(and, %3, %5) : !u32i 
// CHECK: %7 = cir.const(#cir.int<4294967280> : !u32i) : !u32i 
// CHECK: %8 = cir.binop(and, %4, %7) : !u32i 
// CHECK: %9 = cir.binop(or, %8, %6) : !u32i 
// CHECK: cir.store %9, %2 : !u32i, cir.ptr <!u32i> 

void store_neg_field() {  
  S s;
  s.d = -1;
} 

// CHECK: cir.func @_Z15store_neg_field
// CHECK: %1 = cir.const(#cir.int<1> : !s32i) : !s32i 
// CHECK: %2 = cir.unary(minus, %1) : !s32i, !s32i 
// CHECK: %3 = "cir.struct_element_addr"(%0) {member_index = 1 : index, member_name = "d"} : (!cir.ptr<!ty_22struct2ES22>) -> !cir.ptr<!s32i> 
// CHECK: %4 = cir.cast(bitcast, %3 : !cir.ptr<!s32i>), !cir.ptr<!u24i> 
// CHECK: %5 = cir.cast(integral, %2 : !s32i), !u24i 
// CHECK: %6 = cir.load %4 : cir.ptr <!u24i>, !u24i 
// CHECK: %7 = cir.const(#cir.int<3> : !u24i) : !u24i 
// CHECK: %8 = cir.binop(and, %5, %7) : !u24i 
// CHECK: %9 = cir.const(#cir.int<17> : !u24i) : !u24i 
// CHECK: %10 = cir.shift(left, %8 : !u24i, %9 : !u24i) -> !u24i 
// CHECK: %11 = cir.const(#cir.int<16383999> : !u24i) : !u24i 
// CHECK: %12 = cir.binop(and, %6, %11) : !u24i 
// CHECK: %13 = cir.binop(or, %12, %10) : !u24i 
// CHECK: cir.store %13, %4 : !u24i, cir.ptr <!u24i> 

// 16383999 decimal = 1111 1001 1111 1111 1111 1111 binary