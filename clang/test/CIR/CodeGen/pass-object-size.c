// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir-enable -emit-llvm %s -o %t.ll
void b(void *__attribute__((pass_object_size(0))));
void c() {
  int a;
  int d[a];
    b(d);
}

// CIR: %{{[0-9]+}} = cir.objsize(%{{[0-9]+}} : <!void>, max) -> !u64i
