// RUN: %clang -target x86_64-unknown-linux-gnu -cl-std=CL3.0 -Xclang -finclude-default-header -Xclang -fclangir -emit-cir %s -o - | FileCheck %s

float4 test(int4 in)
{
  return as_float4(in);  // Bit reinterpretation
}

// CHECK: [[LOAD:%.*]] = cir.load align(16) %{{.*}} : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
// CHECK: %{{.*}} = cir.cast bitcast [[LOAD]] : !cir.vector<!s32i x 4> -> !cir.vector<!cir.float x 4>