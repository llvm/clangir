// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef struct {
    int a;
    int b;
} A;

// CHECK: cir.func @foo
// CHECK:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!ty_22A22>, cir.ptr <!cir.ptr<!ty_22A22>>, ["a1", init] 
// CHECK:   [[TMP1:%.*]] = cir.alloca !cir.ptr<!ty_22A22>, cir.ptr <!cir.ptr<!ty_22A22>>, ["a2", init] 
// CHECK:   cir.store %arg0, [[TMP0]] : !cir.ptr<!ty_22A22>, cir.ptr <!cir.ptr<!ty_22A22>> 
// CHECK:   cir.store %arg1, [[TMP1]] : !cir.ptr<!ty_22A22>, cir.ptr <!cir.ptr<!ty_22A22>> 
// CHECK:   [[TMP2:%.*]] = cir.load [[TMP0]] : cir.ptr <!cir.ptr<!ty_22A22>>, !cir.ptr<!ty_22A22> 
// CHECK:   [[TMP3:%.*]] = cir.const(#cir.int<1> : !s32i) : !s32i 
// CHECK:   [[TMP4:%.*]] = cir.ptr_stride([[TMP2]] : !cir.ptr<!ty_22A22>, [[TMP3]] : !s32i), !cir.ptr<!ty_22A22> 
// CHECK:   [[TMP5:%.*]] = cir.load [[TMP1]] : cir.ptr <!cir.ptr<!ty_22A22>>, !cir.ptr<!ty_22A22> 
// CHECK:   [[TMP6:%.*]] = cir.const(#cir.int<1> : !s32i) : !s32i 
// CHECK:   [[TMP7:%.*]] = cir.ptr_stride([[TMP5]] : !cir.ptr<!ty_22A22>, [[TMP6]] : !s32i), !cir.ptr<!ty_22A22> 
// CHECK:   cir.copy [[TMP7]] to [[TMP4]] : !cir.ptr<!ty_22A22> 
void foo(A* a1, A* a2) {
    a1[1] = a2[1];
}
