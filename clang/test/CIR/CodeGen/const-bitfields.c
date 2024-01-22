// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o - 2>&1 | FileCheck %s

struct T {
  int X : 5;
  int Y : 6;
  int Z : 9;
  int W;  
};

struct T GV = { 1, 5, 256, 42 };

int getZ() {
  return GV.Z;
}
// CHECK: !ty_22T22 = !cir.struct<struct "T" {!cir.int<u, 32>} #cir.record.decl.ast>
// CHECK: !ty_anon_struct = !cir.struct<struct  {!cir.int<u, 8>, !cir.int<u, 8>, !cir.int<u, 8>}>
// CHECK: #bfi_Z = #cir.bitfield_info<name = "Z", storage_type = !u32i, size = 9, offset = 11, is_signed = true>
// CHECK: cir.global external @GV = #cir.const_struct<{#cir.int<161> : !u8i, #cir.int<0> : !u8i, #cir.int<8> : !u8i}> : !ty_anon_struct

// CHECK: cir.func {{.*@getZ()}}
// CHECK:   %1 = cir.get_global @GV : cir.ptr <!ty_anon_struct>
// CHECK:   %2 = cir.cast(bitcast, %1 : !cir.ptr<!ty_anon_struct>), !cir.ptr<!ty_22T22>
// CHECK:   %3 = cir.cast(bitcast, %2 : !cir.ptr<!ty_22T22>), !cir.ptr<!u32i>
// CHECK:   %4 = cir.get_bitfield(#bfi_Z, %3 : !cir.ptr<!u32i>) -> !s32i

int getW() {
  return GV.W;
}