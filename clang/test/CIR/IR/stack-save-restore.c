// Test the CIR operations can parse and print correctly (roundtrip)

// RUN: cir-opt %s | cir-opt | FileCheck %s

!u8i = !cir.int<u, 8>

module  {
  cir.func @stack_save() {
    %0 = cir.stack_save : <!u8i>
    cir.return
  }

  cir.func @stack_restore(%arg0: !cir.ptr<!u8i>) {    
    cir.stack_restore %arg0 : !cir.ptr<!u8i>
    cir.return
  }
}

//CHECK: module  {

//CHECK-NEXT: cir.func @stack_save() {    
//CHECK-NEXT:   %0 = cir.stack_save : <!u8i>
//CHECK-NEXT:   cir.return
//CHECK-NEXT: }

//CHECK-NEXT: cir.func @stack_restore(%arg0: !cir.ptr<!u8i>) {    
//CHECK-NEXT:   cir.stack_restore %arg0 : !cir.ptr<!u8i>
//CHECK-NEXT:   cir.return
//CHECK-NEXT: }

//CHECK-NEXT: }
