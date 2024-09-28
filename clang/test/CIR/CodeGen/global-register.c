// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

register long long testTwo __asm__("x20");

void main() {
    testTwo = 1;
}

// cir.RG "private" @llvm.named.register.x20 !u64i
// cir.func no_proto @main() extra(#fn_attr) {
// %0 = cir.const #cir.int<1> : !s32i
// %1 = cir.cast(integral, %0 : !s32i), !s64i
// %2 = cir.get_RG @llvm.named.register.x20 : !cir.ptr<!u64i>
// %3 = cir.cast(bitcast, %2 : !cir.ptr<!u64i>), !cir.ptr<!s64i>
// cir.store %1, %3 : !s64i, !cir.ptr<!s64i>
// cir.return