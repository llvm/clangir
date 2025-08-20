// RUN: %clang_cc1 -triple x86_64-unknown-linux -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

//TODO: Add codegen test for setjmp

void BI_setjmp(void *env) {

  // CIR-LABEL: BI_setjmp
  // CIR: cir.llvm.intrinsic "frameaddress"
  // CIR: cir.llvm.intrinsic "stacksave"
  // CIR: cir.llvm.intrinsic "eh.sjlj.setjmp"


  // LLVM-LABEL: BI_setjmp
  // LLVM: @llvm.frameaddress
  // LLVM: @llvm.stacksave
  // LLVM: @llvm.eh.sjlj.setjmp
  __builtin_setjmp(env);
}

