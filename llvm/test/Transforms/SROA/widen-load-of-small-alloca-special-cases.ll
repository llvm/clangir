; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -passes='sroa<preserve-cfg>' -data-layout="e-n8:16:32:64" -S %s | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-SCALAR,CHECK-SCALAR-64,CHECK-LE-64
; RUN: opt -passes='sroa<modify-cfg>' -data-layout="e-n8:16:32:64" -S %s | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-SCALAR,CHECK-SCALAR-64,CHECK-LE-64
; RUN: opt -passes='sroa<preserve-cfg>' -data-layout="e-n8:16:32" -S %s | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-SCALAR,CHECK-SCALAR-32,CHECK-LE-32
; RUN: opt -passes='sroa<modify-cfg>' -data-layout="e-n8:16:32" -S %s | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-SCALAR,CHECK-SCALAR-32,CHECK-LE-32
; RUN: opt -passes='sroa<preserve-cfg>' -data-layout="E-n8:16:32:64" -S %s | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-SCALAR,CHECK-SCALAR-64,CHECK-BE-64
; RUN: opt -passes='sroa<modify-cfg>' -data-layout="E-n8:16:32:64" -S %s | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-SCALAR,CHECK-SCALAR-64,CHECK-BE-64
; RUN: opt -passes='sroa<preserve-cfg>' -data-layout="E-n8:16:32" -S %s | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-SCALAR,CHECK-SCALAR-32,CHECK-BE-32
; RUN: opt -passes='sroa<modify-cfg>' -data-layout="E-n8:16:32" -S %s | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-SCALAR,CHECK-SCALAR-32,CHECK-BE-32

;; Special test

define void @load_2byte_chunk_of_8byte_alloca_with_2byte_step(ptr %src, i64 %byteOff, ptr %dst) nounwind {
; CHECK-ALL-LABEL: @load_2byte_chunk_of_8byte_alloca_with_2byte_step(
; CHECK-ALL-NEXT:    [[INTERMEDIATE:%.*]] = alloca [8 x i8], align 64
; CHECK-ALL-NEXT:    [[INIT:%.*]] = load <8 x i8>, ptr [[SRC:%.*]], align 1
; CHECK-ALL-NEXT:    store <8 x i8> [[INIT]], ptr [[INTERMEDIATE]], align 64
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF_ADDR:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE]], i64 [[BYTEOFF:%.*]]
; CHECK-ALL-NEXT:    [[CHUNK:%.*]] = load <2 x i8>, ptr [[INTERMEDIATE_OFF_ADDR]], align 1
; CHECK-ALL-NEXT:    store <2 x i8> [[CHUNK]], ptr [[DST:%.*]], align 2
; CHECK-ALL-NEXT:    ret void
;
  %intermediate = alloca [8 x i8], align 64
  %init = load <8 x i8>, ptr %src, align 1
  store <8 x i8> %init, ptr %intermediate, align 64
  %intermediate.off.addr = getelementptr inbounds i16, ptr %intermediate, i64 %byteOff
  %chunk = load <2 x i8>, ptr %intermediate.off.addr, align 1
  store <2 x i8> %chunk, ptr %dst
  ret void
}

define void @load_volatile_2byte_chunk_of_8byte_alloca_with_2byte_step(ptr %src, i64 %byteOff, ptr %dst) nounwind {
; CHECK-ALL-LABEL: @load_volatile_2byte_chunk_of_8byte_alloca_with_2byte_step(
; CHECK-ALL-NEXT:    [[INTERMEDIATE:%.*]] = alloca [8 x i8], align 64
; CHECK-ALL-NEXT:    [[INIT:%.*]] = load <8 x i8>, ptr [[SRC:%.*]], align 1
; CHECK-ALL-NEXT:    store <8 x i8> [[INIT]], ptr [[INTERMEDIATE]], align 64
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF_ADDR:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE]], i64 [[BYTEOFF:%.*]]
; CHECK-ALL-NEXT:    [[CHUNK:%.*]] = load volatile <2 x i8>, ptr [[INTERMEDIATE_OFF_ADDR]], align 1
; CHECK-ALL-NEXT:    store <2 x i8> [[CHUNK]], ptr [[DST:%.*]], align 2
; CHECK-ALL-NEXT:    ret void
;
  %intermediate = alloca [8 x i8], align 64
  %init = load <8 x i8>, ptr %src, align 1
  store <8 x i8> %init, ptr %intermediate, align 64
  %intermediate.off.addr = getelementptr inbounds i16, ptr %intermediate, i64 %byteOff
  %chunk = load volatile <2 x i8>, ptr %intermediate.off.addr, align 1
  store <2 x i8> %chunk, ptr %dst
  ret void
}

define void @store_2byte_chunk_of_8byte_alloca_with_2byte_step(ptr %src, i64 %byteOff, <2 x i8> %reinit, ptr %dst) nounwind {
; CHECK-ALL-LABEL: @store_2byte_chunk_of_8byte_alloca_with_2byte_step(
; CHECK-ALL-NEXT:    [[INTERMEDIATE:%.*]] = alloca [8 x i8], align 64
; CHECK-ALL-NEXT:    [[INIT:%.*]] = load <8 x i8>, ptr [[SRC:%.*]], align 1
; CHECK-ALL-NEXT:    store <8 x i8> [[INIT]], ptr [[INTERMEDIATE]], align 64
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF_ADDR:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE]], i64 [[BYTEOFF:%.*]]
; CHECK-ALL-NEXT:    store <2 x i8> [[REINIT:%.*]], ptr [[INTERMEDIATE_OFF_ADDR]], align 1
; CHECK-ALL-NEXT:    [[FINAL:%.*]] = load <8 x i8>, ptr [[INTERMEDIATE]], align 1
; CHECK-ALL-NEXT:    store <8 x i8> [[FINAL]], ptr [[DST:%.*]], align 8
; CHECK-ALL-NEXT:    ret void
;
  %intermediate = alloca [8 x i8], align 64
  %init = load <8 x i8>, ptr %src, align 1
  store <8 x i8> %init, ptr %intermediate, align 64
  %intermediate.off.addr = getelementptr inbounds i16, ptr %intermediate, i64 %byteOff
  store <2 x i8> %reinit, ptr %intermediate.off.addr, align 1
  %final = load <8 x i8>, ptr %intermediate, align 1
  store <8 x i8> %final, ptr %dst
  ret void
}

define void @store_volatile_2byte_chunk_of_8byte_alloca_with_2byte_step(ptr %src, i64 %byteOff, <2 x i8> %reinit, ptr %dst) nounwind {
; CHECK-ALL-LABEL: @store_volatile_2byte_chunk_of_8byte_alloca_with_2byte_step(
; CHECK-ALL-NEXT:    [[INTERMEDIATE:%.*]] = alloca [8 x i8], align 64
; CHECK-ALL-NEXT:    [[INIT:%.*]] = load <8 x i8>, ptr [[SRC:%.*]], align 1
; CHECK-ALL-NEXT:    store <8 x i8> [[INIT]], ptr [[INTERMEDIATE]], align 64
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF_ADDR:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE]], i64 [[BYTEOFF:%.*]]
; CHECK-ALL-NEXT:    store volatile <2 x i8> [[REINIT:%.*]], ptr [[INTERMEDIATE_OFF_ADDR]], align 1
; CHECK-ALL-NEXT:    [[FINAL:%.*]] = load <8 x i8>, ptr [[INTERMEDIATE]], align 1
; CHECK-ALL-NEXT:    store <8 x i8> [[FINAL]], ptr [[DST:%.*]], align 8
; CHECK-ALL-NEXT:    ret void
;
  %intermediate = alloca [8 x i8], align 64
  %init = load <8 x i8>, ptr %src, align 1
  store <8 x i8> %init, ptr %intermediate, align 64
  %intermediate.off.addr = getelementptr inbounds i16, ptr %intermediate, i64 %byteOff
  store volatile <2 x i8> %reinit, ptr %intermediate.off.addr, align 1
  %final = load <8 x i8>, ptr %intermediate, align 1
  store <8 x i8> %final, ptr %dst
  ret void
}

define void @load_2byte_chunk_of_8byte_alloca_with_2byte_step_with_constant_offset_beforehand(ptr %src, i64 %byteOff, ptr %dst) nounwind {
; CHECK-ALL-LABEL: @load_2byte_chunk_of_8byte_alloca_with_2byte_step_with_constant_offset_beforehand(
; CHECK-ALL-NEXT:    [[INTERMEDIATE:%.*]] = alloca [8 x i8], align 64
; CHECK-ALL-NEXT:    [[INIT:%.*]] = load <8 x i8>, ptr [[SRC:%.*]], align 1
; CHECK-ALL-NEXT:    store <8 x i8> [[INIT]], ptr [[INTERMEDIATE]], align 64
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF_ADDR_CST:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE]], i64 1
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF_ADDR:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE_OFF_ADDR_CST]], i64 [[BYTEOFF:%.*]]
; CHECK-ALL-NEXT:    [[CHUNK:%.*]] = load <2 x i8>, ptr [[INTERMEDIATE_OFF_ADDR]], align 1
; CHECK-ALL-NEXT:    store <2 x i8> [[CHUNK]], ptr [[DST:%.*]], align 2
; CHECK-ALL-NEXT:    ret void
;
  %intermediate = alloca [8 x i8], align 64
  %init = load <8 x i8>, ptr %src, align 1
  store <8 x i8> %init, ptr %intermediate, align 64
  %intermediate.off.addr.cst = getelementptr inbounds i16, ptr %intermediate, i64 1
  %intermediate.off.addr = getelementptr inbounds i16, ptr %intermediate.off.addr.cst, i64 %byteOff
  %chunk = load <2 x i8>, ptr %intermediate.off.addr, align 1
  store <2 x i8> %chunk, ptr %dst
  ret void
}

define void @load_2byte_chunk_of_8byte_alloca_with_2byte_step_with_constant_offset_afterwards(ptr %src, i64 %byteOff, ptr %dst) nounwind {
; CHECK-ALL-LABEL: @load_2byte_chunk_of_8byte_alloca_with_2byte_step_with_constant_offset_afterwards(
; CHECK-ALL-NEXT:    [[INTERMEDIATE:%.*]] = alloca [8 x i8], align 64
; CHECK-ALL-NEXT:    [[INIT:%.*]] = load <8 x i8>, ptr [[SRC:%.*]], align 1
; CHECK-ALL-NEXT:    store <8 x i8> [[INIT]], ptr [[INTERMEDIATE]], align 64
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF_ADDR_VARIABLE:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE]], i64 [[BYTEOFF:%.*]]
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF_ADDR:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE_OFF_ADDR_VARIABLE]], i64 1
; CHECK-ALL-NEXT:    [[CHUNK:%.*]] = load <2 x i8>, ptr [[INTERMEDIATE_OFF_ADDR]], align 1
; CHECK-ALL-NEXT:    store <2 x i8> [[CHUNK]], ptr [[DST:%.*]], align 2
; CHECK-ALL-NEXT:    ret void
;
  %intermediate = alloca [8 x i8], align 64
  %init = load <8 x i8>, ptr %src, align 1
  store <8 x i8> %init, ptr %intermediate, align 64
  %intermediate.off.addr.variable = getelementptr inbounds i16, ptr %intermediate, i64 %byteOff
  %intermediate.off.addr = getelementptr inbounds i16, ptr %intermediate.off.addr.variable, i64 1
  %chunk = load <2 x i8>, ptr %intermediate.off.addr, align 1
  store <2 x i8> %chunk, ptr %dst
  ret void
}

define void @load_2byte_chunk_of_8byte_alloca_with_2byte_step_with_variable_offset_inbetween_constant_offsets(ptr %src, i64 %byteOff, ptr %dst) nounwind {
; CHECK-ALL-LABEL: @load_2byte_chunk_of_8byte_alloca_with_2byte_step_with_variable_offset_inbetween_constant_offsets(
; CHECK-ALL-NEXT:    [[INTERMEDIATE:%.*]] = alloca [8 x i8], align 64
; CHECK-ALL-NEXT:    [[INIT:%.*]] = load <8 x i8>, ptr [[SRC:%.*]], align 1
; CHECK-ALL-NEXT:    store <8 x i8> [[INIT]], ptr [[INTERMEDIATE]], align 64
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF_ADDR_CST:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE]], i64 1
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF_ADDR_VARIABLE:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE_OFF_ADDR_CST]], i64 [[BYTEOFF:%.*]]
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF_ADDR:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE_OFF_ADDR_VARIABLE]], i64 1
; CHECK-ALL-NEXT:    [[CHUNK:%.*]] = load <2 x i8>, ptr [[INTERMEDIATE_OFF_ADDR]], align 1
; CHECK-ALL-NEXT:    store <2 x i8> [[CHUNK]], ptr [[DST:%.*]], align 2
; CHECK-ALL-NEXT:    ret void
;
  %intermediate = alloca [8 x i8], align 64
  %init = load <8 x i8>, ptr %src, align 1
  store <8 x i8> %init, ptr %intermediate, align 64
  %intermediate.off.addr.cst = getelementptr inbounds i16, ptr %intermediate, i64 1
  %intermediate.off.addr.variable = getelementptr inbounds i16, ptr %intermediate.off.addr.cst, i64 %byteOff
  %intermediate.off.addr = getelementptr inbounds i16, ptr %intermediate.off.addr.variable, i64 1
  %chunk = load <2 x i8>, ptr %intermediate.off.addr, align 1
  store <2 x i8> %chunk, ptr %dst
  ret void
}

define void @load_2byte_chunk_of_8byte_alloca_with_2byte_step_select_of_variable_geps(ptr %src, i64 %byteOff0, i64 %byteOff1, i1 %cond, ptr %dst) nounwind {
; CHECK-ALL-LABEL: @load_2byte_chunk_of_8byte_alloca_with_2byte_step_select_of_variable_geps(
; CHECK-ALL-NEXT:    [[INTERMEDIATE:%.*]] = alloca [8 x i8], align 64
; CHECK-ALL-NEXT:    [[INIT:%.*]] = load <8 x i8>, ptr [[SRC:%.*]], align 1
; CHECK-ALL-NEXT:    store <8 x i8> [[INIT]], ptr [[INTERMEDIATE]], align 64
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF0:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE]], i64 [[BYTEOFF0:%.*]]
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF1:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE]], i64 [[BYTEOFF1:%.*]]
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF_ADDR:%.*]] = select i1 [[COND:%.*]], ptr [[INTERMEDIATE_OFF0]], ptr [[INTERMEDIATE_OFF1]]
; CHECK-ALL-NEXT:    [[CHUNK:%.*]] = load <2 x i8>, ptr [[INTERMEDIATE_OFF_ADDR]], align 1
; CHECK-ALL-NEXT:    store <2 x i8> [[CHUNK]], ptr [[DST:%.*]], align 2
; CHECK-ALL-NEXT:    ret void
;
  %intermediate = alloca [8 x i8], align 64
  %init = load <8 x i8>, ptr %src, align 1
  store <8 x i8> %init, ptr %intermediate, align 64
  %intermediate.off0 = getelementptr inbounds i16, ptr %intermediate, i64 %byteOff0
  %intermediate.off1 = getelementptr inbounds i16, ptr %intermediate, i64 %byteOff1
  %intermediate.off.addr = select i1 %cond, ptr %intermediate.off0, ptr %intermediate.off1
  %chunk = load <2 x i8>, ptr %intermediate.off.addr, align 1
  store <2 x i8> %chunk, ptr %dst
  ret void
}

define void @load_2byte_chunk_of_8byte_alloca_with_2byte_step_select_of_variable_and_const_geps(ptr %src, i64 %byteOff0, i64 %byteOff1, i1 %cond, ptr %dst) nounwind {
; CHECK-ALL-LABEL: @load_2byte_chunk_of_8byte_alloca_with_2byte_step_select_of_variable_and_const_geps(
; CHECK-ALL-NEXT:    [[INTERMEDIATE:%.*]] = alloca [8 x i8], align 64
; CHECK-ALL-NEXT:    [[INIT:%.*]] = load <8 x i8>, ptr [[SRC:%.*]], align 1
; CHECK-ALL-NEXT:    store <8 x i8> [[INIT]], ptr [[INTERMEDIATE]], align 64
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF0:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE]], i64 1
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF1:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE]], i64 [[BYTEOFF1:%.*]]
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF_ADDR:%.*]] = select i1 [[COND:%.*]], ptr [[INTERMEDIATE_OFF0]], ptr [[INTERMEDIATE_OFF1]]
; CHECK-ALL-NEXT:    [[CHUNK:%.*]] = load <2 x i8>, ptr [[INTERMEDIATE_OFF_ADDR]], align 1
; CHECK-ALL-NEXT:    store <2 x i8> [[CHUNK]], ptr [[DST:%.*]], align 2
; CHECK-ALL-NEXT:    ret void
;
  %intermediate = alloca [8 x i8], align 64
  %init = load <8 x i8>, ptr %src, align 1
  store <8 x i8> %init, ptr %intermediate, align 64
  %intermediate.off0 = getelementptr inbounds i16, ptr %intermediate, i64 1
  %intermediate.off1 = getelementptr inbounds i16, ptr %intermediate, i64 %byteOff1
  %intermediate.off.addr = select i1 %cond, ptr %intermediate.off0, ptr %intermediate.off1
  %chunk = load <2 x i8>, ptr %intermediate.off.addr, align 1
  store <2 x i8> %chunk, ptr %dst
  ret void
}

define void @load_2byte_chunk_of_8byte_alloca_with_2byte_step_variable_gep_of_select_of_const_geps(ptr %src, i64 %byteOff, i1 %cond, ptr %dst) nounwind {
; CHECK-ALL-LABEL: @load_2byte_chunk_of_8byte_alloca_with_2byte_step_variable_gep_of_select_of_const_geps(
; CHECK-ALL-NEXT:    [[INTERMEDIATE:%.*]] = alloca [8 x i8], align 64
; CHECK-ALL-NEXT:    [[INIT:%.*]] = load <8 x i8>, ptr [[SRC:%.*]], align 1
; CHECK-ALL-NEXT:    store <8 x i8> [[INIT]], ptr [[INTERMEDIATE]], align 64
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF0:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE]], i64 0
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF1:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE]], i64 2
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF_ADDR:%.*]] = select i1 [[COND:%.*]], ptr [[INTERMEDIATE_OFF0]], ptr [[INTERMEDIATE_OFF1]]
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF_ADDR_VAR:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE_OFF_ADDR]], i64 [[BYTEOFF:%.*]]
; CHECK-ALL-NEXT:    [[CHUNK:%.*]] = load <2 x i8>, ptr [[INTERMEDIATE_OFF_ADDR_VAR]], align 1
; CHECK-ALL-NEXT:    store <2 x i8> [[CHUNK]], ptr [[DST:%.*]], align 2
; CHECK-ALL-NEXT:    ret void
;
  %intermediate = alloca [8 x i8], align 64
  %init = load <8 x i8>, ptr %src, align 1
  store <8 x i8> %init, ptr %intermediate, align 64
  %intermediate.off0 = getelementptr inbounds i16, ptr %intermediate, i64 0
  %intermediate.off1 = getelementptr inbounds i16, ptr %intermediate, i64 2
  %intermediate.off.addr = select i1 %cond, ptr %intermediate.off0, ptr %intermediate.off1
  %intermediate.off.addr.var = getelementptr inbounds i16, ptr %intermediate.off.addr, i64 %byteOff
  %chunk = load <2 x i8>, ptr %intermediate.off.addr.var, align 1
  store <2 x i8> %chunk, ptr %dst
  ret void
}

define void @load_ptr_chunk_of_16byte_alloca(ptr %src, i64 %byteOff, ptr %dst) nounwind {
; CHECK-ALL-LABEL: @load_ptr_chunk_of_16byte_alloca(
; CHECK-ALL-NEXT:    [[INTERMEDIATE:%.*]] = alloca [16 x i8], align 64
; CHECK-ALL-NEXT:    [[INIT:%.*]] = load <16 x i8>, ptr [[SRC:%.*]], align 1
; CHECK-ALL-NEXT:    store <16 x i8> [[INIT]], ptr [[INTERMEDIATE]], align 64
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF_ADDR:%.*]] = getelementptr inbounds i8, ptr [[INTERMEDIATE]], i64 [[BYTEOFF:%.*]]
; CHECK-ALL-NEXT:    [[CHUNK:%.*]] = load <1 x ptr>, ptr [[INTERMEDIATE_OFF_ADDR]], align 1
; CHECK-ALL-NEXT:    store <1 x ptr> [[CHUNK]], ptr [[DST:%.*]], align 8
; CHECK-ALL-NEXT:    ret void
;
  %intermediate = alloca [16 x i8], align 64
  %init = load <16 x i8>, ptr %src, align 1
  store <16 x i8> %init, ptr %intermediate, align 64
  %intermediate.off.addr = getelementptr inbounds i8, ptr %intermediate, i64 %byteOff
  %chunk = load <1 x ptr>, ptr %intermediate.off.addr, align 1
  store <1 x ptr> %chunk, ptr %dst
  ret void
}

define void @load_float_chunk_of_16byte_alloca(ptr %src, i64 %byteOff, ptr %dst) nounwind {
; CHECK-ALL-LABEL: @load_float_chunk_of_16byte_alloca(
; CHECK-ALL-NEXT:    [[INTERMEDIATE:%.*]] = alloca [16 x i8], align 64
; CHECK-ALL-NEXT:    [[INIT:%.*]] = load <16 x i8>, ptr [[SRC:%.*]], align 1
; CHECK-ALL-NEXT:    store <16 x i8> [[INIT]], ptr [[INTERMEDIATE]], align 64
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF_ADDR:%.*]] = getelementptr inbounds i8, ptr [[INTERMEDIATE]], i64 [[BYTEOFF:%.*]]
; CHECK-ALL-NEXT:    [[CHUNK:%.*]] = load <1 x float>, ptr [[INTERMEDIATE_OFF_ADDR]], align 1
; CHECK-ALL-NEXT:    store <1 x float> [[CHUNK]], ptr [[DST:%.*]], align 4
; CHECK-ALL-NEXT:    ret void
;
  %intermediate = alloca [16 x i8], align 64
  %init = load <16 x i8>, ptr %src, align 1
  store <16 x i8> %init, ptr %intermediate, align 64
  %intermediate.off.addr = getelementptr inbounds i8, ptr %intermediate, i64 %byteOff
  %chunk = load <1 x float>, ptr %intermediate.off.addr, align 1
  store <1 x float> %chunk, ptr %dst
  ret void
}

define void @two_loads_of_same_2byte_chunks_of_8byte_alloca_with_2byte_step_variable_gep(ptr %src, i64 %byteOff, ptr %dst0, ptr %dst1) nounwind {
; CHECK-ALL-LABEL: @two_loads_of_same_2byte_chunks_of_8byte_alloca_with_2byte_step_variable_gep(
; CHECK-ALL-NEXT:    [[INTERMEDIATE:%.*]] = alloca [8 x i8], align 64
; CHECK-ALL-NEXT:    [[INIT:%.*]] = load <8 x i8>, ptr [[SRC:%.*]], align 1
; CHECK-ALL-NEXT:    store <8 x i8> [[INIT]], ptr [[INTERMEDIATE]], align 64
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE]], i64 [[BYTEOFF:%.*]]
; CHECK-ALL-NEXT:    [[CHUNK0:%.*]] = load <2 x i8>, ptr [[INTERMEDIATE_OFF]], align 1
; CHECK-ALL-NEXT:    store <2 x i8> [[CHUNK0]], ptr [[DST0:%.*]], align 2
; CHECK-ALL-NEXT:    [[CHUNK1:%.*]] = load <2 x i8>, ptr [[INTERMEDIATE_OFF]], align 1
; CHECK-ALL-NEXT:    store <2 x i8> [[CHUNK1]], ptr [[DST1:%.*]], align 2
; CHECK-ALL-NEXT:    ret void
;
  %intermediate = alloca [8 x i8], align 64
  %init = load <8 x i8>, ptr %src, align 1
  store <8 x i8> %init, ptr %intermediate, align 64
  %intermediate.off = getelementptr inbounds i16, ptr %intermediate, i64 %byteOff
  %chunk0 = load <2 x i8>, ptr %intermediate.off, align 1
  store <2 x i8> %chunk0, ptr %dst0
  %chunk1 = load <2 x i8>, ptr %intermediate.off, align 1
  store <2 x i8> %chunk1, ptr %dst1
  ret void
}

define void @two_loads_of_two_2byte_chunks_of_8byte_alloca_with_2byte_step_variable_geps(ptr %src, i64 %byteOff0, i64 %byteOff1, ptr %dst0, ptr %dst1) nounwind {
; CHECK-ALL-LABEL: @two_loads_of_two_2byte_chunks_of_8byte_alloca_with_2byte_step_variable_geps(
; CHECK-ALL-NEXT:    [[INTERMEDIATE:%.*]] = alloca [8 x i8], align 64
; CHECK-ALL-NEXT:    [[INIT:%.*]] = load <8 x i8>, ptr [[SRC:%.*]], align 1
; CHECK-ALL-NEXT:    store <8 x i8> [[INIT]], ptr [[INTERMEDIATE]], align 64
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF0:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE]], i64 [[BYTEOFF0:%.*]]
; CHECK-ALL-NEXT:    [[CHUNK0:%.*]] = load <2 x i8>, ptr [[INTERMEDIATE_OFF0]], align 1
; CHECK-ALL-NEXT:    store <2 x i8> [[CHUNK0]], ptr [[DST0:%.*]], align 2
; CHECK-ALL-NEXT:    [[INTERMEDIATE_OFF1:%.*]] = getelementptr inbounds i16, ptr [[INTERMEDIATE_OFF0]], i64 [[BYTEOFF1:%.*]]
; CHECK-ALL-NEXT:    [[CHUNK1:%.*]] = load <2 x i8>, ptr [[INTERMEDIATE_OFF1]], align 1
; CHECK-ALL-NEXT:    store <2 x i8> [[CHUNK1]], ptr [[DST1:%.*]], align 2
; CHECK-ALL-NEXT:    ret void
;
  %intermediate = alloca [8 x i8], align 64
  %init = load <8 x i8>, ptr %src, align 1
  store <8 x i8> %init, ptr %intermediate, align 64
  %intermediate.off0 = getelementptr inbounds i16, ptr %intermediate, i64 %byteOff0
  %chunk0 = load <2 x i8>, ptr %intermediate.off0, align 1
  store <2 x i8> %chunk0, ptr %dst0
  %intermediate.off1 = getelementptr inbounds i16, ptr %intermediate.off0, i64 %byteOff1
  %chunk1 = load <2 x i8>, ptr %intermediate.off1, align 1
  store <2 x i8> %chunk1, ptr %dst1
  ret void
}
;; NOTE: These prefixes are unused and the list is autogenerated. Do not add tests below this line:
; CHECK-BE-32: {{.*}}
; CHECK-BE-64: {{.*}}
; CHECK-LE-32: {{.*}}
; CHECK-LE-64: {{.*}}
; CHECK-SCALAR: {{.*}}
; CHECK-SCALAR-32: {{.*}}
; CHECK-SCALAR-64: {{.*}}
