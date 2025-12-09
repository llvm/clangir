// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

void allocas(unsigned long n) {
    char *a = (char *)__builtin_alloca(n);
    char *uninitialized_a = (char *)__builtin_alloca_uninitialized(n);
}

// CIR-LABEL: cir.func {{.*}} @allocas
// CIR:         %[[ALLOCA1:.*]] = cir.alloca !u8i, !cir.ptr<!u8i>, {{.*}} ["bi_alloca"]
// CIR:         cir.cast bitcast %[[ALLOCA1]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
// CIR:         %[[ALLOCA2:.*]] = cir.alloca !u8i, !cir.ptr<!u8i>, {{.*}} ["bi_alloca"]
// CIR:         cir.cast bitcast %[[ALLOCA2]] : !cir.ptr<!u8i> -> !cir.ptr<!void>

// LLVM-LABEL: define {{.*}} void @allocas(i64 %{{.*}})
// LLVM:         %[[BI_ALLOCA1:.*]] = alloca i8, i64 %{{.*}}, align 8, addrspace(5)
// LLVM:         addrspacecast ptr addrspace(5) %[[BI_ALLOCA1]] to ptr
// LLVM:         %[[BI_ALLOCA2:.*]] = alloca i8, i64 %{{.*}}, align 8, addrspace(5)
// LLVM:         addrspacecast ptr addrspace(5) %[[BI_ALLOCA2]] to ptr

// OGCG-LABEL: define {{.*}} void @allocas(i64 {{.*}} %n)
// OGCG:         %[[BI_ALLOCA1:.*]] = alloca i8, i64 %{{.*}}, align 8, addrspace(5)
// OGCG:         addrspacecast ptr addrspace(5) %[[BI_ALLOCA1]] to ptr
// OGCG:         %[[BI_ALLOCA2:.*]] = alloca i8, i64 %{{.*}}, align 8, addrspace(5)
// OGCG:         addrspacecast ptr addrspace(5) %[[BI_ALLOCA2]] to ptr
