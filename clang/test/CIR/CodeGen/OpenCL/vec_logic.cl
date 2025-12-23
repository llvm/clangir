// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -O2 -emit-cir -fclangir -o - %s | FileCheck %s --check-prefix=CIR
// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -O2 -emit-llvm -fclangir -o - %s | FileCheck %s --check-prefix=LLVM
// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=OG-LLVM

kernel void test(char4 in1, char4 in2, local char4 *out)
{
    *out = (in1 == (char4)3 && (in1 == (char4)5 || in2 == (char4)7))
            ? in1 : in2;
}


// CIR: [[OR:%.*]] = cir.binop(or, %{{.*}}, %{{.*}}) : !cir.vector<!cir.bool x 4>
// CIR: [[CAST_BOOL:%.*]] = cir.cast bool_to_int [[OR]] : !cir.vector<!cir.bool x 4> -> !cir.vector<!s8i x 4>
// CIR: [[ZERO_VEC:%.*]] = cir.const #cir.const_vector<[#cir.int<0> : !s8i, #cir.int<0> : !s8i, #cir.int<0> : !s8i, #cir.int<0> : !s8i]> : !cir.vector<!s8i x 4>
// CIR: [[CMP_NE:%.*]] = cir.vec.cmp(ne, [[CAST_BOOL]], [[ZERO_VEC]]) : !cir.vector<!s8i x 4>, !cir.vector<!cir.bool x 4>
// CIR: [[AND:%.*]] = cir.binop(and, %{{.*}}, [[CMP_NE]]) : !cir.vector<!cir.bool x 4>
// CIR: [[CAST_FINAL:%.*]] = cir.cast bool_to_int [[AND]] : !cir.vector<!cir.bool x 4> -> !cir.vector<!s8i x 4>
// CIR: [[LOAD_T:%.*]] = cir.load align(4) %{{.*}} : !cir.ptr<!cir.vector<!s8i x 4>, lang_address_space(offload_private)>, !cir.vector<!s8i x 4>
// CIR: [[LOAD_E:%.*]] = cir.load align(4) %{{.*}} : !cir.ptr<!cir.vector<!s8i x 4>, lang_address_space(offload_private)>, !cir.vector<!s8i x 4>
// CIR: cir.vec.ternary([[CAST_FINAL]], [[LOAD_T]], [[LOAD_E]]) : !cir.vector<!s8i x 4>, !cir.vector<!s8i x 4>

// LLVM: [[CMPA:%.*]] = icmp eq <4 x i8> %{{.*}}, splat (i8 3)
// LLVM: [[CMPB:%.*]] = icmp eq <4 x i8> %{{.*}}, splat (i8 7)
// LLVM: [[AND:%.*]] = and <4 x i1> [[CMPA]], [[CMPB]]
// LLVM: select <4 x i1> [[AND]], <4 x i8> %{{.*}}, <4 x i8> %{{.*}}

// OG-LLVM: [[CMPA:%.*]] = icmp eq <4 x i8> %{{.*}}, splat (i8 3)
// OG-LLVM: [[CMPB:%.*]] = icmp eq <4 x i8> %{{.*}}, splat (i8 7)
// OG-LLVM: [[AND:%.*]] = and <4 x i1> [[CMPA]], [[CMPB]]
// OG-LLVM: select <4 x i1> [[AND]], <4 x i8> %{{.*}}, <4 x i8> %{{.*}}