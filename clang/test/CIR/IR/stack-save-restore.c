// Test the CIR operations can parse and print correctly (roundtrip)

// RUN: cir-opt %s | cir-opt | FileCheck %s

!u8i = !cir.int<u, 8>

module  {
  cir.func @foo() {
    %0 = cir.alloca !cir.ptr<!u8i>, cir.ptr <!cir.ptr<!u8i>>, ["saved_stack"] {alignment = 8 : i64}
    %1 = cir.stack_save : !cir.ptr<!u8i>
    cir.store %1, %0 : !cir.ptr<!u8i>, cir.ptr <!cir.ptr<!u8i>>
    %2 = cir.load %0 : cir.ptr <!cir.ptr<!u8i>>, !cir.ptr<!u8i>
    cir.stack_restore %2 : !cir.ptr<!u8i>
    cir.return
  }
}

//CHECK: module  {

//CHECK-NEXT:  cir.func @foo() {
//CHECK-NEXT:    %0 = cir.alloca !cir.ptr<!u8i>, cir.ptr <!cir.ptr<!u8i>>, ["saved_stack"] {alignment = 8 : i64}
//CHECK-NEXT:    %1 = cir.stack_save : !cir.ptr<!u8i>
//CHECK-NEXT:    cir.store %1, %0 : !cir.ptr<!u8i>, cir.ptr <!cir.ptr<!u8i>>
//CHECK-NEXT:    %2 = cir.load %0 : cir.ptr <!cir.ptr<!u8i>>, !cir.ptr<!u8i>
//CHECK-NEXT:    cir.stack_restore %2 : !cir.ptr<!u8i>
//CHECK-NEXT:    cir.return
//CHECK-NEXT:  }

//CHECK: }
