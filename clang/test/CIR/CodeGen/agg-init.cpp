// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// CHECK: !ty_22yep_22 = !cir.struct<struct "yep_" {!cir.int<u, 32>, !cir.int<u, 32>}>

typedef enum xxy_ {
  xxy_Low = 0,
  xxy_High = 0x3f800000,
  xxy_EnumSize = 0x7fffffff
} xxy;

typedef struct yep_ {
  unsigned int Status;
  xxy HC;
} yop;

void use() { yop{}; }

// CHECK: cir.func @_Z3usev()
// CHECK:   %0 = cir.alloca !ty_22yep_22, !cir.ptr<!ty_22yep_22>, ["agg.tmp.ensured"] {alignment = 4 : i64}
// CHECK:   %1 = cir.get_member %0[0] {name = "Status"} : !cir.ptr<!ty_22yep_22> -> !cir.ptr<!u32i>
// CHECK:   %2 = cir.const(#cir.int<0> : !u32i) : !u32i
// CHECK:   cir.store %2, %1 : !u32i, !cir.ptr<!u32i>
// CHECK:   %3 = cir.get_member %0[1] {name = "HC"} : !cir.ptr<!ty_22yep_22> -> !cir.ptr<!u32i>
// CHECK:   %4 = cir.const(#cir.int<0> : !u32i) : !u32i
// CHECK:   cir.store %4, %3 : !u32i, !cir.ptr<!u32i>
// CHECK:   cir.return
// CHECK: }

typedef unsigned long long Flags;

typedef enum XType {
    A = 0,
    Y = 1000066001,
    X = 1000070000
} XType;

typedef struct Yo {
    XType type;
    const void* __attribute__((__may_alias__)) next;
    Flags createFlags;
} Yo;

void yo() {
  Yo ext = {X};
  Yo ext2 = {Y, &ext};
}

// CHECK: cir.func @_Z2yov()
// CHECK:   %0 = cir.alloca !ty_22Yo22, !cir.ptr<!ty_22Yo22>, ["ext"] {alignment = 8 : i64}
// CHECK:   %1 = cir.alloca !ty_22Yo22, !cir.ptr<!ty_22Yo22>, ["ext2", init] {alignment = 8 : i64}
// CHECK:   %2 = cir.const(#cir.const_struct<{#cir.int<1000070000> : !u32i, #cir.ptr<null> : !cir.ptr<!void>, #cir.int<0> : !u64i}> : !ty_22Yo22) : !ty_22Yo22
// CHECK:   cir.store %2, %0 : !ty_22Yo22, !cir.ptr<!ty_22Yo22>
// CHECK:   %3 = cir.get_member %1[0] {name = "type"} : !cir.ptr<!ty_22Yo22> -> !cir.ptr<!u32i>
// CHECK:   %4 = cir.const(#cir.int<1000066001> : !u32i) : !u32i
// CHECK:   cir.store %4, %3 : !u32i, !cir.ptr<!u32i>
// CHECK:   %5 = cir.get_member %1[1] {name = "next"} : !cir.ptr<!ty_22Yo22> -> !cir.ptr<!cir.ptr<!void>>
// CHECK:   %6 = cir.cast(bitcast, %0 : !cir.ptr<!ty_22Yo22>), !cir.ptr<!void>
// CHECK:   cir.store %6, %5 : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CHECK:   %7 = cir.get_member %1[2] {name = "createFlags"} : !cir.ptr<!ty_22Yo22> -> !cir.ptr<!u64i>
// CHECK:   %8 = cir.const(#cir.int<0> : !u64i) : !u64i
// CHECK:   cir.store %8, %7 : !u64i, !cir.ptr<!u64i>
