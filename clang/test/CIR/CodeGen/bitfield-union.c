// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void main() {
    union demo {
        int x;
        int y : 4;
        int z : 8;
    };
    union demo d;
    d.x = 1;
    d.y = 2;
    d.z = 0;
}

// CHECK: !ty_22demo22 = !cir.struct<union "demo" {!s32i, !u8i, !u8i}>
// CHECK: #bfi_y = #cir.bitfield_info<name = "y", storage_type = !u8i, size = 4, offset = 0, is_signed = true>
// CHECK: #bfi_z = #cir.bitfield_info<name = "z", storage_type = !u8i, size = 8, offset = 0, is_signed = true>

//   cir.func no_proto @main() extra(#fn_attr) {
//     %0 = cir.alloca !ty_22demo22, !cir.ptr<!ty_22demo22>, ["d"] {alignment = 4 : i64}
//     %1 = cir.const #cir.int<1> : !s32i
//     %2 = cir.get_member %0[0] {name = "x"} : !cir.ptr<!ty_22demo22> -> !cir.ptr<!s32i>
//     cir.store %1, %2 : !s32i, !cir.ptr<!s32i>
//     %3 = cir.const #cir.int<2> : !s32i
//     %4 = cir.cast(bitcast, %0 : !cir.ptr<!ty_22demo22>), !cir.ptr<!u8i>
//     %5 = cir.set_bitfield(#bfi_y, %4 : !cir.ptr<!u8i>, %3 : !s32i) -> !s32i
//     %6 = cir.const #cir.int<0> : !s32i loc(#loc10)
//     %7 = cir.cast(bitcast, %0 : !cir.ptr<!ty_22demo22>), !cir.ptr<!u8i>
//     %8 = cir.set_bitfield(#bfi_z, %7 : !cir.ptr<!u8i>, %6 : !s32i) -> !s32i
//     cir.return
//   }
