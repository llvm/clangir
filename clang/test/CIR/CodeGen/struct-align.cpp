// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s

struct Foo {
  short a;
  short b;
  short c;
  short d;
  long e;   // Make the struct 8-byte aligned
};

void test(Foo *ptr) {
  ptr->a = 1;  // align 8
  ptr->b = 2;  // align 2
  ptr->c = 3;  // align 4
  ptr->d = 4;  // align 2
}

// CIR-LABEL: _Z4testP3Foo
// LLVM-LABEL: _Z4testP3Foo

// CIR: [[PTR:%.*]] = cir.load {{%.*}} : !cir.ptr<!cir.ptr<!ty_Foo>>, !cir.ptr<!ty_Foo>
// CIR: [[PTRA:%.*]] = cir.get_member [[PTR]][0] {name = "a"} : !cir.ptr<!ty_Foo> -> !cir.ptr<!s16i>
// CIR: cir.store align(8) {{%.*}}, [[PTRA]] : !s16i, !cir.ptr<!s16i>
// CIR: [[PTR:%.*]] = cir.load {{%.*}} : !cir.ptr<!cir.ptr<!ty_Foo>>, !cir.ptr<!ty_Foo>
// CIR: [[PTRB:%.*]] = cir.get_member [[PTR]][1] {name = "b"} : !cir.ptr<!ty_Foo> -> !cir.ptr<!s16i>
// CIR: cir.store align(2) {{%.*}}, [[PTRB]] : !s16i, !cir.ptr<!s16i>
// CIR: [[PTR:%.*]] = cir.load {{%.*}} : !cir.ptr<!cir.ptr<!ty_Foo>>, !cir.ptr<!ty_Foo>
// CIR: [[PTRC:%.*]] = cir.get_member [[PTR]][2] {name = "c"} : !cir.ptr<!ty_Foo> -> !cir.ptr<!s16i>
// CIR: cir.store align(4) {{%.*}}, [[PTRC]] : !s16i, !cir.ptr<!s16i>
// CIR: [[PTR:%.*]] = cir.load {{%.*}} : !cir.ptr<!cir.ptr<!ty_Foo>>, !cir.ptr<!ty_Foo>
// CIR: [[PTRD:%.*]] = cir.get_member [[PTR]][3] {name = "d"} : !cir.ptr<!ty_Foo> -> !cir.ptr<!s16i>
// CIR: cir.store align(2) {{%.*}}, [[PTRD]] : !s16i, !cir.ptr<!s16i>

// FIXME: miss inbounds and nuw attribute
// LLVM: [[PTR:%.*]] = load ptr, ptr {{%.*}}, align 8
// LLVM: [[PTRA:%.*]] = getelementptr %struct.Foo, ptr [[PTR]], i32 0, i32 0
// LLVM: store i16 1, ptr [[PTRA]], align 8
// LLVM: [[PTR:%.*]] = load ptr, ptr {{%.*}}, align 8
// LLVM: [[PTRB:%.*]] = getelementptr %struct.Foo, ptr [[PTR]], i32 0, i32 1
// LLVM: store i16 2, ptr [[PTRB]], align 2
// LLVM: [[PTR:%.*]] = load ptr, ptr {{%.*}}, align 8
// LLVM: [[PTRC:%.*]] = getelementptr %struct.Foo, ptr [[PTR]], i32 0, i32 2
// LLVM: store i16 3, ptr [[PTRC]], align 4
// LLVM: [[PTR:%.*]] = load ptr, ptr {{%.*}}, align 8
// LLVM: [[PTRD:%.*]] = getelementptr %struct.Foo, ptr [[PTR]], i32 0, i32 3
// LLVM: store i16 4, ptr [[PTRD]], align 2