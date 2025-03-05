#include "../Inputs/cuda.h"

// RUN: echo "sample fatbin" > %t.fatbin
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-cir -target-sdk-version=12.3 \
// RUN:            -fcuda-include-gpubinary %t.fatbin \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-llvm -target-sdk-version=12.3 \
// RUN:            -fcuda-include-gpubinary %t.fatbin \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-HOST --input-file=%t.ll %s

// COM: OG doesn't emit anything if there is nothing to register.
// COM: Here we still emit the template for test purposes,
// COM: and the behaviour will be fixed later.

// CIR-HOST: module @"{{.*}}" attributes {
// CIR-HOST:   cir.cu.binary_handle = #cir.cu.binary_handle<{{.*}}.fatbin>,
// CIR-HOST:   cir.global_ctors = [#cir.global_ctor<"__cuda_module_ctor", {{[0-9]+}}>]
// CIR-HOST: }

// The content in const array should be the same as echoed above,
// with a trailing line break ('\n', 0x0A).
// CIR-HOST: cir.global "private" constant cir_private @__cuda_fatbin_str =
// CIR-HOST-SAME: #cir.const_array<"sample fatbin\0A">
// CIR-HOST-SAME: {{.*}}section = ".nv_fatbin"

// LLVM-HOST: @__cuda_fatbin_str = private constant [14 x i8] c"sample fatbin\0A", section ".nv_fatbin"

// The first value is CUDA file head magic number.
// CIR-HOST: cir.global "private" internal @__cuda_fatbin_wrapper
// CIR-HOST: = #cir.const_struct<{
// CIR-HOST:   #cir.int<1180844977> : !s32i,
// CIR-HOST:   #cir.int<1> : !s32i,
// CIR-HOST:   #cir.ptr<null> : !cir.ptr<!void>,
// CIR-HOST:   #cir.ptr<null> : !cir.ptr<!void>
// CIR-HOST: }>
// CIR-HOST-SAME: {{.*}}section = ".nvFatBinSegment"

// COM: @__cuda_fatbin_wrapper is constant for OG.
// COM: However, as we don't have a way to put @__cuda_fatbin_str directly
// COM: to its third field in Clang IR, we can't mark this variable as 
// COM: constant: we need to initialize it later, at the beginning
// COM: of @__cuda_module_ctor.

// LLVM-HOST: @__cuda_fatbin_wrapper = internal global {
// LLVM-HOST:   i32 1180844977, i32 1, ptr null, ptr null
// LLVM-HOST: }

// LLVM-HOST: @llvm.global_ctors = {{.*}}ptr @__cuda_module_ctor

// CIR-HOST: cir.func private @__cudaRegisterFatBinary
// CIR-HOST: cir.func {{.*}} @__cuda_module_ctor() {
// CIR-HOST:   %[[#F0:]] = cir.get_global @__cuda_fatbin_wrapper
// CIR-HOST:   %[[#F1:]] = cir.get_global @__cuda_fatbin_str
// CIR-HOST:   %[[#F2:]] = cir.get_member %[[#F0]][2]
// CIR-HOST:   %[[#F3:]] = cir.cast(bitcast, %[[#F2]]
// CIR-HOST:   cir.store %[[#F1]], %[[#F3]]
// CIR-HOST:   cir.call @__cudaRegisterFatBinary
// CIR-HOST: }

// LLVM-HOST: define internal void @__cuda_module_ctor() {
// LLVM-HOST:   store ptr @__cuda_fatbin_str, ptr getelementptr {{.*}}, ptr @__cuda_fatbin_wrapper
// LLVM-HOST:   call ptr @__cudaRegisterFatBinary(ptr @__cuda_fatbin_wrapper)
// LLVM-HOST: }
