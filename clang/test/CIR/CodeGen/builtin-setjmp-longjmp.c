// RUN: %clang_cc1 -triple x86_64-unknown-linux -O2 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -O2 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

void BI_builtin_setjmp(void *env) {

  // CIR-LABEL: BI_builtin_setjmp
  // CIR: [[ENV_ALLOCA:%[0-9]+]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>,
  // CIR: cir.store %arg0, [[ENV_ALLOCA]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
  // CIR: [[ENV_LOAD:%[0-9]+]] = cir.load align(8) [[ENV_ALLOCA]]
  // CIR: [[CAST1:%[0-9]+]] = cir.cast(bitcast, [[ENV_LOAD]] : !cir.ptr<!void>), !cir.ptr<!cir.ptr<!void>>
  // CIR: [[CAST2:%[0-9]+]] = cir.cast(bitcast, [[CAST1]] : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!cir.ptr<!u8i>>
  // CIR: [[ZERO:%[0-9]+]] = cir.const #cir.int<0> : !s32i
  // CIR: [[FA:%[0-9]+]] = cir.llvm.intrinsic "frameaddress" [[ZERO]]
  // CIR: cir.store [[FA]], [[CAST2]] : !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>
  // CIR: [[SS:%[0-9]+]] = cir.llvm.intrinsic "stacksave"
  // CIR: [[TWO:%[0-9]+]] = cir.const #cir.int<2> : !s32i
  // CIR: [[GEP:%[0-9]+]] = cir.ptr_stride([[CAST2]] : !cir.ptr<!cir.ptr<!u8i>>, [[TWO]] : !s32i),
  // CIR: cir.store [[SS]], [[GEP]] : !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>
  // CIR: [[SJ:%[0-9]+]] = cir.llvm.intrinsic "eh.sjlj.setjmp" [[CAST2]]


  // LLVM-LABEL: BI_builtin_setjmp
  // LLVM-SAME: (ptr {{.*}}[[ENV:%[0-9]+]])
  // LLVM: [[FA:%[0-9]+]] = {{.*}}@llvm.frameaddress.p0(i32 0) 
  // LLVM: store ptr [[FA]], ptr [[ENV]]
  // LLVM: [[SS:%[0-9]+]] = {{.*}}@llvm.stacksave.p0() 
  // LLVM: [[GEP:%[0-9]+]] = getelementptr{{.*}}i8, ptr [[ENV]], i64 16
  // LLVM: store ptr [[SS]], ptr [[GEP]]
  // LLVM: @llvm.eh.sjlj.setjmp(ptr{{.*}}[[ENV]])
  __builtin_setjmp(env);
}

