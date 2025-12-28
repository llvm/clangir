// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -std=c++20 -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -std=c++20 -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=OGCG

struct Base {
  virtual ~Base();
};

struct Derived : Base {};

// Check that RTTI globals are emitted in the correct address space (addrspace(1) for AMDGCN).
// CIR-DAG: cir.global {{.*}} lang_address_space(offload_global) @_ZTI4Base : !cir.ptr<!u8i, lang_address_space(offload_global)>
// CIR-DAG: cir.global {{.*}} lang_address_space(offload_global) @_ZTVN10__cxxabiv120__si_class_type_infoE : !cir.ptr<!cir.ptr<!u8i, lang_address_space(offload_global)>>
// CIR-DAG: cir.global {{.*}} lang_address_space(offload_global) @_ZTS7Derived = {{.*}} : !cir.array<!s8i x 8>
// CIR-DAG: cir.global {{.*}} lang_address_space(offload_global) @_ZTI7Derived = #cir.typeinfo<{{{.*}}}> : !rec_{{.*}}

// Check the __dynamic_cast function signature uses globals address space for RTTI pointers.
// CIR: cir.func private @__dynamic_cast(!cir.ptr<!void>, !cir.ptr<!u8i, lang_address_space(offload_global)>, !cir.ptr<!u8i, lang_address_space(offload_global)>, !s64i) -> !cir.ptr<!void>

// LLVM-DAG: @_ZTI4Base = external addrspace(1) constant ptr addrspace(1)
// LLVM-DAG: @_ZTVN10__cxxabiv120__si_class_type_infoE = external addrspace(1) global
// LLVM-DAG: @_ZTS7Derived = {{.*}}addrspace(1) constant [{{.*}} x i8]
// LLVM-DAG: @_ZTI7Derived = {{.*}}addrspace(1) constant { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1) }

// OGCG-DAG: @_ZTI4Base = external addrspace(1) constant ptr addrspace(1)
// OGCG-DAG: @_ZTVN10__cxxabiv120__si_class_type_infoE = external addrspace(1) global [0 x ptr addrspace(1)]
// OGCG-DAG: @_ZTS7Derived = {{.*}} addrspace(1) constant [{{.*}} x i8]
// OGCG-DAG: @_ZTI7Derived = {{.*}} addrspace(1) constant { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1) }

// Test dynamic_cast with __dynamic_cast runtime call.
// The RTTI pointers passed to __dynamic_cast should be in the globals address space.

// CIR-LABEL: cir.func {{.*}} @_Z8ptr_castP4Base
// CIR:   cir.call @__dynamic_cast({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!cir.ptr<!void>, !cir.ptr<!u8i, lang_address_space(offload_global)>, !cir.ptr<!u8i, lang_address_space(offload_global)>, !s64i) -> !cir.ptr<!void>

// LLVM-LABEL: define {{.*}} @_Z8ptr_castP4Base
// LLVM:   call ptr @__dynamic_cast(ptr {{.*}}, ptr addrspace(1) @_ZTI4Base, ptr addrspace(1) @_ZTI7Derived, i64 0)

// OGCG-LABEL: define {{.*}} @_Z8ptr_castP4Base
// OGCG:   call ptr @__dynamic_cast(ptr {{.*}}, ptr addrspace(1) @_ZTI4Base, ptr addrspace(1) @_ZTI7Derived, i64 0)
Derived *ptr_cast(Base *b) {
  return dynamic_cast<Derived *>(b);
}

// Test reference dynamic_cast with __cxa_bad_cast on failure.
// The RTTI pointers passed to __dynamic_cast should be in the globals address space.

// CIR-LABEL: cir.func {{.*}} @_Z8ref_castR4Base
// CIR:   cir.call @__dynamic_cast({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!cir.ptr<!void>, !cir.ptr<!u8i, lang_address_space(offload_global)>, !cir.ptr<!u8i, lang_address_space(offload_global)>, !s64i) -> !cir.ptr<!void>
// CIR:   cir.call @__cxa_bad_cast()
// CIR:   cir.unreachable

// LLVM-LABEL: define {{.*}} @_Z8ref_castR4Base
// LLVM:   call ptr @__dynamic_cast(ptr {{.*}}, ptr addrspace(1) @_ZTI4Base, ptr addrspace(1) @_ZTI7Derived, i64 0)
// LLVM:   call void @__cxa_bad_cast()

// OGCG-LABEL: define {{.*}} @_Z8ref_castR4Base
// OGCG:   call ptr @__dynamic_cast(ptr {{.*}}, ptr addrspace(1) @_ZTI4Base, ptr addrspace(1) @_ZTI7Derived, i64 0)
// OGCG:   call void @__cxa_bad_cast()
Derived &ref_cast(Base &b) {
  return dynamic_cast<Derived &>(b);
}
