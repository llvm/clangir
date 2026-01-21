!s32i = !cir.int<s, 32>
!u8i = !cir.int<u, 8>
#fn_attr = #cir<extra({nothrow = #cir.nothrow})>
#loc2 = loc("clang/test/CIR/CodeGen/no-unique-address.cpp":15:3)
!rec_Empty = !cir.record<struct "Empty" padded {!u8i} #cir.record.decl.ast>
!rec_S = !cir.record<struct "S" {!s32i}>
module @"/home/brunolopes/Dev/clangir-auto-top-crashes/incubator/clang/test/CIR/CodeGen/no-unique-address.cpp" attributes {cir.lang = #cir.lang<cxx>, cir.module_asm = [], cir.sob = #cir.signed_overflow_behavior<undefined>, cir.triple = "x86_64-unknown-linux-gnu", cir.type_size_info = #cir.type_size_info<char = 8, int = 32, size_t = 64>, dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>} {
  cir.func private @_ZN5EmptyC1Ev(!cir.ptr<!rec_Empty>) special_member<#cir.cxx_ctor<!rec_Empty, default>> extra(#fn_attr) loc(#loc1)
  cir.func no_inline optnone linkonce_odr @_ZN1SC2Ev(%arg0: !cir.ptr<!rec_S> loc("clang/test/CIR/CodeGen/no-unique-address.cpp":15:3)) special_member<#cir.cxx_ctor<!rec_S, default>> extra(#fn_attr) {
    %0 = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["this", init] {alignment = 8 : i64} loc(#loc2)
    cir.store %arg0, %0 : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>> loc(#loc4)
    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S> loc(#loc2)
    %2 = cir.get_member %1[0] {name = "x"} : !cir.ptr<!rec_S> -> !cir.ptr<!s32i> loc(#loc5)
    %3 = cir.const #cir.int<1> : !s32i loc(#loc6)
    cir.store align(4) %3, %2 : !s32i, !cir.ptr<!s32i> loc(#loc14)
    %4 = cir.cast bitcast %1 : !cir.ptr<!rec_S> -> !cir.ptr<!rec_Empty> loc(#loc2)
    cir.call @_ZN5EmptyC1Ev(%4) : (!cir.ptr<!rec_Empty>) -> () extra(#fn_attr) loc(#loc8)
    cir.return loc(#loc3)
  } loc(#loc13)
  cir.func no_inline optnone linkonce_odr @_ZN1SC1Ev(%arg0: !cir.ptr<!rec_S> loc("clang/test/CIR/CodeGen/no-unique-address.cpp":15:3)) special_member<#cir.cxx_ctor<!rec_S, default>> extra(#fn_attr) {
    %0 = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["this", init] {alignment = 8 : i64} loc(#loc2)
    cir.store %arg0, %0 : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>> loc(#loc4)
    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S> loc(#loc2)
    cir.call @_ZN1SC2Ev(%1) : (!cir.ptr<!rec_S>) -> () loc(#loc3)
    cir.return loc(#loc3)
  } loc(#loc13)
  cir.func no_inline optnone dso_local @_Z4testv() extra(#fn_attr) {
    %0 = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s", init] {alignment = 4 : i64} loc(#loc16)
    cir.call @_ZN1SC1Ev(%0) : (!cir.ptr<!rec_S>) -> () loc(#loc12)
    cir.return loc(#loc10)
  } loc(#loc15)
} loc(#loc)
#loc = loc("/home/brunolopes/Dev/clangir-auto-top-crashes/incubator/clang/test/CIR/CodeGen/no-unique-address.cpp":0:0)
#loc1 = loc("clang/test/CIR/CodeGen/no-unique-address.cpp":10:8)
#loc3 = loc("clang/test/CIR/CodeGen/no-unique-address.cpp":15:20)
#loc4 = loc("clang/test/CIR/CodeGen/no-unique-address.cpp":15:19)
#loc5 = loc("clang/test/CIR/CodeGen/no-unique-address.cpp":13:7)
#loc6 = loc("clang/test/CIR/CodeGen/no-unique-address.cpp":15:11)
#loc7 = loc("clang/test/CIR/CodeGen/no-unique-address.cpp":13:3)
#loc8 = loc("clang/test/CIR/CodeGen/no-unique-address.cpp":15:15)
#loc9 = loc("clang/test/CIR/CodeGen/no-unique-address.cpp":18:1)
#loc10 = loc("clang/test/CIR/CodeGen/no-unique-address.cpp":20:1)
#loc11 = loc("clang/test/CIR/CodeGen/no-unique-address.cpp":19:3)
#loc12 = loc("clang/test/CIR/CodeGen/no-unique-address.cpp":19:5)
#loc13 = loc(fused[#loc2, #loc3])
#loc14 = loc(fused[#loc7, #loc5])
#loc15 = loc(fused[#loc9, #loc10])
#loc16 = loc(fused[#loc11, #loc12])
