// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
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

typedef struct {
  int a : 4;
  int b : 27;
  int c : 17;
  int d : 2;
  int e : 15;
  unsigned f; // type other than int above, not a bitfield
} S;

typedef struct {
  int a : 3;  // one bitfield with size < 8
  unsigned b;
} T;
// CHECK: !ty_T = !cir.struct<struct "T" {!u8i, !u32i} #cir.record.decl.ast>
// CHECK: !ty_anon2E0_ = !cir.struct<struct "anon.0" {!u32i} #cir.record.decl.ast>
// CHECK: !ty_S = !cir.struct<struct "S" {!u32i, !cir.array<!u8i x 3>, !u16i, !u32i}>
// CHECK: !ty___long = !cir.struct<struct "__long" {!ty_anon2E0_, !u32i, !cir.ptr<!u32i>}>

// CHECK: cir.func @_Z11store_field
// CHECK:   [[TMP0:%.*]] = cir.alloca !ty_S, !cir.ptr<!ty_S>
// CHECK:   [[TMP1:%.*]] = cir.const #cir.int<3> : !s32i
// CHECK:   [[TMP2:%.*]] = cir.cast(bitcast, [[TMP0]] : !cir.ptr<!ty_S>), !cir.ptr<!u32i>
// CHECK:   cir.set_bitfield(#bfi_a, [[TMP2]] : !cir.ptr<!u32i>, [[TMP1]] : !s32i)
void store_field() {
  S s;
  s.a = 3;
}

// CHECK: cir.func @_Z10load_field
// CHECK:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!ty_S>, !cir.ptr<!cir.ptr<!ty_S>>, ["s", init, const]
// CHECK:   [[TMP1:%.*]] = cir.load [[TMP0]] : !cir.ptr<!cir.ptr<!ty_S>>, !cir.ptr<!ty_S>
// CHECK:   [[TMP2:%.*]] = cir.get_member [[TMP1]][1] {name = "d"} : !cir.ptr<!ty_S> -> !cir.ptr<!cir.array<!u8i x 3>>
// CHECK:   [[TMP3:%.*]] = cir.get_bitfield(#bfi_d, [[TMP2]] : !cir.ptr<!cir.array<!u8i x 3>>) -> !s32i
int load_field(S& s) {
  return s.d;
}

// CHECK: cir.func @_Z17load_non_bitfield
// CHECK:   cir.get_member {{%.}}[3] {name = "f"} : !cir.ptr<!ty_S> -> !cir.ptr<!u32i>
unsigned load_non_bitfield(S& s) {
  return s.f;
}

// just create a usage of T type
// CHECK: cir.func @_Z17load_one_bitfield
int load_one_bitfield(T& t) {
  return t.a;
}
