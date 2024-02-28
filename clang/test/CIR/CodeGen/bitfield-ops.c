// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o - 2>&1 | FileCheck %s

typedef struct {
  int a : 4;
  int b : 27;
  int c : 17;
  int d : 2;
  int e : 15;
  unsigned f;
} S;

typedef struct {
  int a : 3;  // one bitfield with size < 8
  unsigned b;
} T; 

// CHECK: #bfi_e = #cir.bitfield_info<name = "e", storage_type = !u16i, size = 15, offset = 0, is_signed = true>
// CHECK: !ty_22S22 = !cir.struct<struct "S" {!cir.int<u, 32>, !cir.array<!cir.int<u, 8> x 3>, !cir.int<u, 16>, !cir.int<u, 32>}>
// CHECK: #bfi_d = #cir.bitfield_info<name = "d", storage_type = !cir.array<!u8i x 3>, size = 2, offset = 17, is_signed = true>

// CHECK: cir.func {{.*@store_field}}
// CHECK:   [[TMP0:%.*]] = cir.alloca !ty_22S22, cir.ptr <!ty_22S22>, ["s"]
// CHECK:   [[TMP1:%.*]] = cir.const(#cir.int<3> : !s32i) : !s32i
// CHECK:   [[TMP2:%.*]] = cir.get_member [[TMP0]][2] {name = "e"} : !cir.ptr<!ty_22S22> -> !cir.ptr<!u16i>
// CHECK:   cir.set_bitfield(#bfi_e, [[TMP2]] : !cir.ptr<!u16i>, [[TMP1]] : !s32i)            
void store_field() {
  S s;
  s.e = 3;
}

// CHECK: cir.func {{.*@load_field}}
// CHECK:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!ty_22S22>, cir.ptr <!cir.ptr<!ty_22S22>>, ["s", init]
// CHECK:   [[TMP1:%.*]] = cir.load [[TMP0]] : cir.ptr <!cir.ptr<!ty_22S22>>, !cir.ptr<!ty_22S22>
// CHECK:   [[TMP2:%.*]] = cir.get_member [[TMP1]][1] {name = "d"} : !cir.ptr<!ty_22S22> -> !cir.ptr<!cir.array<!u8i x 3>>
// CHECK:   [[TMP3:%.*]] = cir.get_bitfield(#bfi_d, [[TMP2]] : !cir.ptr<!cir.array<!u8i x 3>>) -> !s32i
int load_field(S* s) {
  return s->d;
}

// CHECK: cir.func {{.*@unOp}}
// CHECK:   [[TMP0:%.*]] = cir.get_member {{.*}}[1] {name = "d"} : !cir.ptr<!ty_22S22> -> !cir.ptr<!cir.array<!u8i x 3>>
// CHECK:   [[TMP1:%.*]] = cir.get_bitfield(#bfi_d, [[TMP0]] : !cir.ptr<!cir.array<!u8i x 3>>) -> !s32i
// CHECK:   [[TMP2:%.*]] = cir.unary(inc, [[TMP1]]) : !s32i, !s32i
// CHECK:   cir.set_bitfield(#bfi_d, [[TMP0]] : !cir.ptr<!cir.array<!u8i x 3>>, [[TMP2]] : !s32i)
void unOp(S* s) {
  s->d++;
}

// CHECK: cir.func {{.*@binOp}}
// CHECK:   [[TMP0:%.*]] = cir.const(#cir.int<42> : !s32i) : !s32i
// CHECK:   [[TMP1:%.*]] = cir.get_member {{.*}}[1] {name = "d"} : !cir.ptr<!ty_22S22> -> !cir.ptr<!cir.array<!u8i x 3>>
// CHECK:   [[TMP2:%.*]] = cir.get_bitfield(#bfi_d, [[TMP1]] : !cir.ptr<!cir.array<!u8i x 3>>) -> !s32i
// CHECK:   [[TMP3:%.*]] = cir.binop(or, [[TMP2]], [[TMP0]]) : !s32i
// CHECK:   cir.set_bitfield(#bfi_d, [[TMP1]] : !cir.ptr<!cir.array<!u8i x 3>>, [[TMP3]] : !s32i)
void binOp(S* s) {
   s->d |= 42;
}


// CHECK: cir.func {{.*@load_non_bitfield}}
// CHECK:   cir.get_member {{%.}}[3] {name = "f"} : !cir.ptr<!ty_22S22> -> !cir.ptr<!u32i>
unsigned load_non_bitfield(S *s) {
  return s->f;
}

// just create a usage of T type 
// CHECK: cir.func {{.*@load_one_bitfield}}
int load_one_bitfield(T* t) {
  return t->a;
}
