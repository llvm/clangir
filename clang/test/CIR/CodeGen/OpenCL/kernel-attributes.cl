// RUN: %clang_cc1 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM

typedef unsigned int uint4 __attribute__((ext_vector_type(4)));

kernel  __attribute__((vec_type_hint(int))) __attribute__((reqd_work_group_size(1,2,4))) void kernel1(int a) {}
kernel __attribute__((vec_type_hint(uint4))) __attribute__((work_group_size_hint(8,16,32))) void kernel2(int a) {}
kernel __attribute__((intel_reqd_sub_group_size(8))) void kernel3(int a) {}

// CIR: #fn_attr[[KERNEL3:[0-9]*]] = {{.+}}ocl.kernel_metadata = #cir.ocl.kernel_metadata<intelReqdSubGroupSize = 8 : i32>{{.+}}
// CIR: #fn_attr[[KERNEL1:[0-9]*]] = {{.+}}ocl.kernel_metadata = #cir.ocl.kernel_metadata<reqdWorkGroupSize = [1 : i32, 2 : i32, 4 : i32], vecTypeHint = !s32i>{{.+}}
// CIR: #fn_attr[[KERNEL2:[0-9]*]] = {{.+}}ocl.kernel_metadata = #cir.ocl.kernel_metadata<workGroupSizeHint = [8 : i32, 16 : i32, 32 : i32], vecTypeHint = !cir.vector<!u32i x 4>>{{.+}}

// CIR: cir.func @kernel1{{.+}} extra(#fn_attr[[KERNEL1]])
// CIR: cir.func @kernel2{{.+}} extra(#fn_attr[[KERNEL2]])
// CIR: cir.func @kernel3{{.+}} extra(#fn_attr[[KERNEL3]])

// LLVM: define{{.*}}@kernel1(i32 {{[^%]*}}%0) {{[^{]+}} !reqd_work_group_size ![[MD2:[0-9]+]] !vec_type_hint ![[MD1:[0-9]+]]
// LLVM: define{{.*}}@kernel2(i32 {{[^%]*}}%0) {{[^{]+}} !vec_type_hint ![[MD3:[0-9]+]] !work_group_size_hint ![[MD4:[0-9]+]]
// LLVM: define{{.*}}@kernel3(i32 {{[^%]*}}%0) {{[^{]+}} !intel_reqd_sub_group_size ![[MD5:[0-9]+]]

// LLVM: [[MD2]] = !{i32 1, i32 2, i32 4}
// LLVM: [[MD1]] = !{i32 undef, i32 1}
// LLVM: [[MD3]] = !{<4 x i32> undef, i32 0}
// LLVM: [[MD4]] = !{i32 8, i32 16, i32 32}
// LLVM: [[MD5]] = !{i32 8}
