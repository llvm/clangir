// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Should implicitly zero-initialize global array elements.
struct S {
  int i;
} arr[3] = {{1}};
// CHECK: cir.global external @arr = #cir.const_array<[#cir.const_struct<{#cir.int<1> : !s32i}> : !ty_22struct2ES22, #cir.zero : !ty_22struct2ES22, #cir.zero : !ty_22struct2ES22]> : !cir.array<!ty_22struct2ES22 x 3>
