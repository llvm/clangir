// RUN: %clang_cc1 -triple s390x-linux  -O2 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple s390x-linux  -O2 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
//
void BI_builtin_setjmp(void *env) {
  // XFAIL: *
  __builtin_setjmp(env);
}

