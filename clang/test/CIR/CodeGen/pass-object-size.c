// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir-enable -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
void b(void *__attribute__((pass_object_size(0))));
void e(void *__attribute__((pass_object_size(2))));
void c() {
  int a;
  int d[a];
    b(d);
    e(d);
}

// CIR: %{{[0-9]+}} = cir.objsize(%{{[0-9]+}} : <!void>, max) -> !u64i
// CIR: %{{[0-9]+}} = cir.objsize(%{{[0-9]+}} : <!void>, min) -> !u64i
// LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{[0-9]+}}, i1 false, i1 true, i1 false),
// LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{[0-9]+}}, i1 true, i1 true, i1 false),
