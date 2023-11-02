// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=BEFORE

typedef struct {
  int a : 4;
  int b : 27;
  int c : 17;
  int d : 2;
  int e : 15;
  unsigned f; // type other than int above, not a bitfield
} S; 

// BEFORE: cir.func {{.*@store_field}}
// BEFORE:   [[TMP0:%.*]] = cir.alloca !ty_22S22, cir.ptr <!ty_22S22>, ["s"] 
// BEFORE:   [[TMP1:%.*]] = cir.const(#cir.int<3> : !s32i) : !s32i
// BEFORE:   [[TMP2:%.*]] = cir.get_member [[TMP0]][2] {name = "e"} : !cir.ptr<!ty_22S22> -> !cir.ptr<!u16i>
// BEFORE:   [[TMP3:%.*]] = cir.set_bitfield [[TMP1]], [[TMP2]](<name = "e", elt_type = !u16i, storage_size = 16, size = 15, offset = 0, is_signed = true>) : (!s32i, !cir.ptr<!u16i>) -> !s32i
void store_field() {
  S s;
  s.e = 3;
}

// BEFORE: cir.func {{.*@load_field}}
// BEFORE:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!ty_22S22>, cir.ptr <!cir.ptr<!ty_22S22>>, ["s", init] 
// BEFORE:   [[TMP1:%.*]] = cir.load [[TMP0]] : cir.ptr <!cir.ptr<!ty_22S22>>, !cir.ptr<!ty_22S22>
// BEFORE:   [[TMP2:%.*]] = cir.get_member [[TMP1]][1] {name = "d"} : !cir.ptr<!ty_22S22> -> !cir.ptr<!u32i>
// BEFORE:   [[TMP3:%.*]] = cir.get_bitfield [[TMP2]](<name = "d", elt_type = !u32i, storage_size = 32, size = 2, offset = 17, is_signed = true>) : !cir.ptr<!u32i> -> !s32i
int load_field(S* s) {
  return s->d;
}