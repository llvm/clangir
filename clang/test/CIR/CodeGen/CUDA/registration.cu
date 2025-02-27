#include "../Inputs/cuda.h"

// COM: We don't need to include a real GPU binary.
// COM: Here we include the source file itself to see if the global variable
// COM: containing the "binary" file content is correctly generated.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-cir -target-sdk-version=12.3 \
// RUN:            -fcuda-include-gpubinary %s \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.cir %s

// CIR-HOST: module @"{{.*}}" attributes {
// CIR-HOST:   cir.cu.binary_handle = #cir.cu.binary_handle<{{.*}}registration.cu>,
// CIR-HOST:   cir.global_ctors = [#cir.global_ctor<"__cuda_module_ctor", {{[0-9]+}}>]
// CIR-HOST: }

// COM: Content of const_array should be the same as content of this file.
// CIR-HOST: cir.global {{.*}} @__cuda_fatbin_str = #cir.const_array<"#include{{.+}}">

// COM: The first value is CUDA file head magic number.
// CIR-HOST: cir.global {{.*}} @__cuda_fatbin_wrapper = #cir.const_struct<{
// CIR-HOST:   #cir.int<1180844977> : !s32i,
// CIR-HOST:   #cir.int<1> : !s32i,
// CIR-HOST:   #cir.ptr<null> : !cir.ptr<!void>,
// CIR-HOST:   #cir.ptr<null> : !cir.ptr<!void>
// CIR-HOST: }>

// CIR-HOST: cir.func private @__cudaRegisterFatBinary
// CIR-HOST: cir.func {{.*}} @__cuda_module_ctor() {
// CIR-HOST:   cir.call @__cudaRegisterFatBinary(%1)
// CIR-HOST: }
