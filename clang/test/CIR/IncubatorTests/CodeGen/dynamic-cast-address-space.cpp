// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=CIR-BEFORE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=OGCG

struct Base {
  virtual ~Base();
};

struct Derived : Base {};

// Test dynamic_cast to void* with address space attribute.
// The result pointer should preserve the address space of the source pointer.

// CIR-BEFORE: cir.func {{.*}} @_Z30ptr_cast_to_complete_addrspacePU3AS1
// CIR-BEFORE:   %{{.+}} = cir.dyn_cast ptr %{{.+}} : !cir.ptr<!rec_Base, target_address_space(1)> -> !cir.ptr<!void, target_address_space(1)>
// CIR-BEFORE: }

// CIR: cir.func {{.*}} @_Z30ptr_cast_to_complete_addrspacePU3AS1
// CIR:   %[[#SRC:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!rec_Base, target_address_space(1)>>, !cir.ptr<!rec_Base, target_address_space(1)>
// CIR:   %[[#SRC_IS_NOT_NULL:]] = cir.cast ptr_to_bool %[[#SRC]] : !cir.ptr<!rec_Base, target_address_space(1)> -> !cir.bool
// CIR:   %{{.+}} = cir.ternary(%[[#SRC_IS_NOT_NULL]], true {
// CIR:     %[[#SRC_BYTES_PTR:]] = cir.cast bitcast %{{.+}} : !cir.ptr<!rec_Base, target_address_space(1)> -> !cir.ptr<!u8i, target_address_space(1)>
// CIR:     %[[#DST_BYTES_PTR:]] = cir.ptr_stride %[[#SRC_BYTES_PTR]], %{{.+}} : (!cir.ptr<!u8i, target_address_space(1)>, !s64i) -> !cir.ptr<!u8i, target_address_space(1)>
// CIR:     %[[#CASTED_PTR:]] = cir.cast bitcast %[[#DST_BYTES_PTR]] : !cir.ptr<!u8i, target_address_space(1)> -> !cir.ptr<!void, target_address_space(1)>
// CIR:     cir.yield %[[#CASTED_PTR]] : !cir.ptr<!void, target_address_space(1)>
// CIR:   }, false {
// CIR:     %[[#NULL_PTR:]] = cir.const #cir.ptr<null> : !cir.ptr<!void, target_address_space(1)>
// CIR:     cir.yield %[[#NULL_PTR]] : !cir.ptr<!void, target_address_space(1)>
// CIR:   }) : (!cir.bool) -> !cir.ptr<!void, target_address_space(1)>
// CIR: }

// LLVM: define dso_local ptr addrspace(1) @_Z30ptr_cast_to_complete_addrspacePU3AS1
// LLVM-SAME: (ptr addrspace(1) %{{.+}})
// LLVM:   %[[#SRC:]] = load ptr addrspace(1), ptr %{{.+}}, align 8
// LLVM:   %[[#SRC_IS_NOT_NULL:]] = icmp ne ptr addrspace(1) %[[#SRC]], null
// LLVM:   br i1 %[[#SRC_IS_NOT_NULL]], label %[[#TRUE_BLOCK:]], label %[[#FALSE_BLOCK:]]
// LLVM: [[#TRUE_BLOCK]]:
// LLVM:   %[[#VTABLE:]] = load ptr, ptr addrspace(1) %[[#SRC]], align 8
// LLVM:   %[[#OFFSET_PTR:]] = getelementptr i64, ptr %[[#VTABLE]], i64 -2
// LLVM:   %[[#OFFSET:]] = load i64, ptr %[[#OFFSET_PTR]], align 8
// LLVM:   %[[#RESULT:]] = getelementptr i8, ptr addrspace(1) %[[#SRC]], i64 %[[#OFFSET]]
// LLVM:   br label %[[#MERGE:]]
// LLVM: [[#FALSE_BLOCK]]:
// LLVM:   br label %[[#MERGE]]
// LLVM: [[#MERGE]]:
// LLVM:   %[[#PHI:]] = phi ptr addrspace(1) [ null, %[[#FALSE_BLOCK]] ], [ %[[#RESULT]], %[[#TRUE_BLOCK]] ]
// LLVM:   ret ptr addrspace(1)
// LLVM: }

// OGCG: define dso_local noundef ptr addrspace(1) @_Z30ptr_cast_to_complete_addrspacePU3AS1
// OGCG-SAME: (ptr addrspace(1) noundef %{{.+}})
// OGCG:   %[[SRC:[a-z0-9]+]] = load ptr addrspace(1), ptr %{{.+}}, align 8
// OGCG:   icmp eq ptr addrspace(1) %[[SRC]], null
// OGCG: dynamic_cast.notnull:
// OGCG:   %[[VTABLE:[a-z0-9]+]] = load ptr, ptr addrspace(1) %[[SRC]], align 8
// OGCG:   getelementptr inbounds i64, ptr %[[VTABLE]], i64 -2
// OGCG:   %[[OFFSET:[a-z0-9.]+]] = load i64, ptr %{{.+}}, align 8
// OGCG:   %[[RESULT:[0-9]+]] = getelementptr inbounds i8, ptr addrspace(1) %[[SRC]], i64 %[[OFFSET]]
// OGCG: dynamic_cast.end:
// OGCG:   %[[PHI:[0-9]+]] = phi ptr addrspace(1) [ %[[RESULT]], %dynamic_cast.notnull ], [ null, %dynamic_cast.null ]
// OGCG:   ret ptr addrspace(1) %[[PHI]]
// OGCG: }
void __attribute__((address_space(1))) *ptr_cast_to_complete_addrspace(Base __attribute__((address_space(1))) *ptr) {
  return dynamic_cast<void __attribute__((address_space(1))) *>(ptr);
}
