; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -S < %s -passes=gvn,dce | FileCheck %s

; Analyze Load from clobbering Load.

define <vscale x 4 x i32> @load_store_clobber_load(ptr %p)  {
; CHECK-LABEL: @load_store_clobber_load(
; CHECK-NEXT:    [[LOAD1:%.*]] = load <vscale x 4 x i32>, ptr [[P:%.*]], align 16
; CHECK-NEXT:    store <vscale x 4 x i32> zeroinitializer, ptr undef, align 16
; CHECK-NEXT:    [[ADD:%.*]] = add <vscale x 4 x i32> [[LOAD1]], [[LOAD1]]
; CHECK-NEXT:    ret <vscale x 4 x i32> [[ADD]]
;
  %load1 = load <vscale x 4 x i32>, ptr %p
  store <vscale x 4 x i32> zeroinitializer, ptr undef
  %load2 = load <vscale x 4 x i32>, ptr %p ; <- load to be eliminated
  %add = add <vscale x 4 x i32> %load1, %load2
  ret <vscale x 4 x i32> %add
}

define <vscale x 4 x i32> @load_store_clobber_load_mayalias(ptr %p, ptr %p2) {
; CHECK-LABEL: @load_store_clobber_load_mayalias(
; CHECK-NEXT:    [[LOAD1:%.*]] = load <vscale x 4 x i32>, ptr [[P:%.*]], align 16
; CHECK-NEXT:    store <vscale x 4 x i32> zeroinitializer, ptr [[P2:%.*]], align 16
; CHECK-NEXT:    [[LOAD2:%.*]] = load <vscale x 4 x i32>, ptr [[P]], align 16
; CHECK-NEXT:    [[SUB:%.*]] = sub <vscale x 4 x i32> [[LOAD1]], [[LOAD2]]
; CHECK-NEXT:    ret <vscale x 4 x i32> [[SUB]]
;
  %load1 = load <vscale x 4 x i32>, ptr %p
  store <vscale x 4 x i32> zeroinitializer, ptr %p2
  %load2 = load <vscale x 4 x i32>, ptr %p
  %sub = sub <vscale x 4 x i32> %load1, %load2
  ret <vscale x 4 x i32> %sub
}

define <vscale x 4 x i32> @load_store_clobber_load_noalias(ptr noalias %p, ptr noalias %p2) {
; CHECK-LABEL: @load_store_clobber_load_noalias(
; CHECK-NEXT:    [[LOAD1:%.*]] = load <vscale x 4 x i32>, ptr [[P:%.*]], align 16
; CHECK-NEXT:    store <vscale x 4 x i32> zeroinitializer, ptr [[P2:%.*]], align 16
; CHECK-NEXT:    [[ADD:%.*]] = add <vscale x 4 x i32> [[LOAD1]], [[LOAD1]]
; CHECK-NEXT:    ret <vscale x 4 x i32> [[ADD]]
;
  %load1 = load <vscale x 4 x i32>, ptr %p
  store <vscale x 4 x i32> zeroinitializer, ptr %p2
  %load2 = load <vscale x 4 x i32>, ptr %p ; <- load to be eliminated
  %add = add <vscale x 4 x i32> %load1, %load2
  ret <vscale x 4 x i32> %add
}

; BasicAA return MayAlias for %gep1,%gep2, could improve as MustAlias.
define i32 @load_clobber_load_gep1(ptr %p) {
; CHECK-LABEL: @load_clobber_load_gep1(
; CHECK-NEXT:    [[GEP1:%.*]] = getelementptr <vscale x 4 x i32>, ptr [[P:%.*]], i64 0, i64 1
; CHECK-NEXT:    [[LOAD1:%.*]] = load i32, ptr [[GEP1]], align 4
; CHECK-NEXT:    [[ADD:%.*]] = add i32 [[LOAD1]], [[LOAD1]]
; CHECK-NEXT:    ret i32 [[ADD]]
;
  %gep1 = getelementptr <vscale x 4 x i32>, ptr %p, i64 0, i64 1
  %load1 = load i32, ptr %gep1
  %gep2 = getelementptr i32, ptr %p, i64 1
  %load2 = load i32, ptr %gep2 ; <- load could be eliminated
  %add = add i32 %load1, %load2
  ret i32 %add
}

define i32 @load_clobber_load_gep2(ptr %p) {
; CHECK-LABEL: @load_clobber_load_gep2(
; CHECK-NEXT:    [[GEP1:%.*]] = getelementptr <vscale x 4 x i32>, ptr [[P:%.*]], i64 1, i64 0
; CHECK-NEXT:    [[LOAD1:%.*]] = load i32, ptr [[GEP1]], align 4
; CHECK-NEXT:    [[GEP2:%.*]] = getelementptr i32, ptr [[P]], i64 4
; CHECK-NEXT:    [[LOAD2:%.*]] = load i32, ptr [[GEP2]], align 4
; CHECK-NEXT:    [[ADD:%.*]] = add i32 [[LOAD1]], [[LOAD2]]
; CHECK-NEXT:    ret i32 [[ADD]]
;
  %gep1 = getelementptr <vscale x 4 x i32>, ptr %p, i64 1, i64 0
  %load1 = load i32, ptr %gep1
  %gep2 = getelementptr i32, ptr %p, i64 4
  %load2 = load i32, ptr %gep2 ; <- can not determine at compile-time if %load1 and %load2 are same addr
  %add = add i32 %load1, %load2
  ret i32 %add
}

; TODO: BasicAA return MayAlias for %gep1,%gep2, could improve as MustAlias.
define i32 @load_clobber_load_gep3(ptr %p) {
; CHECK-LABEL: @load_clobber_load_gep3(
; CHECK-NEXT:    [[GEP1:%.*]] = getelementptr <vscale x 4 x i32>, ptr [[P:%.*]], i64 1, i64 0
; CHECK-NEXT:    [[LOAD1:%.*]] = load i32, ptr [[GEP1]], align 4
; CHECK-NEXT:    [[GEP2:%.*]] = getelementptr <vscale x 4 x float>, ptr [[P]], i64 1, i64 0
; CHECK-NEXT:    [[LOAD2:%.*]] = load float, ptr [[GEP2]], align 4
; CHECK-NEXT:    [[CAST:%.*]] = bitcast float [[LOAD2]] to i32
; CHECK-NEXT:    [[ADD:%.*]] = add i32 [[LOAD1]], [[CAST]]
; CHECK-NEXT:    ret i32 [[ADD]]
;
  %gep1 = getelementptr <vscale x 4 x i32>, ptr %p, i64 1, i64 0
  %load1 = load i32, ptr %gep1
  %gep2 = getelementptr <vscale x 4 x float>, ptr %p, i64 1, i64 0
  %load2 = load float, ptr %gep2 ; <- load could be eliminated
  %cast = bitcast float %load2 to i32
  %add = add i32 %load1, %cast
  ret i32 %add
}

define <vscale x 4 x i32> @load_clobber_load_fence(ptr %p) {
; CHECK-LABEL: @load_clobber_load_fence(
; CHECK-NEXT:    [[LOAD1:%.*]] = load <vscale x 4 x i32>, ptr [[P:%.*]], align 16
; CHECK-NEXT:    call void asm "", "~{memory}"()
; CHECK-NEXT:    [[LOAD2:%.*]] = load <vscale x 4 x i32>, ptr [[P]], align 16
; CHECK-NEXT:    [[SUB:%.*]] = sub <vscale x 4 x i32> [[LOAD1]], [[LOAD2]]
; CHECK-NEXT:    ret <vscale x 4 x i32> [[SUB]]
;
  %load1 = load <vscale x 4 x i32>, ptr %p
  call void asm "", "~{memory}"()
  %load2 = load <vscale x 4 x i32>, ptr %p
  %sub = sub <vscale x 4 x i32> %load1, %load2
  ret <vscale x 4 x i32> %sub
}

define <vscale x 4 x i32> @load_clobber_load_sideeffect(ptr %p) {
; CHECK-LABEL: @load_clobber_load_sideeffect(
; CHECK-NEXT:    [[LOAD1:%.*]] = load <vscale x 4 x i32>, ptr [[P:%.*]], align 16
; CHECK-NEXT:    call void asm sideeffect "", ""()
; CHECK-NEXT:    [[LOAD2:%.*]] = load <vscale x 4 x i32>, ptr [[P]], align 16
; CHECK-NEXT:    [[ADD:%.*]] = add <vscale x 4 x i32> [[LOAD1]], [[LOAD2]]
; CHECK-NEXT:    ret <vscale x 4 x i32> [[ADD]]
;
  %load1 = load <vscale x 4 x i32>, ptr %p
  call void asm sideeffect "", ""()
  %load2 = load <vscale x 4 x i32>, ptr %p
  %add = add <vscale x 4 x i32> %load1, %load2
  ret <vscale x 4 x i32> %add
}

; Analyze Load from clobbering Store.

define <vscale x 4 x i32> @store_forward_to_load(ptr %p) {
; CHECK-LABEL: @store_forward_to_load(
; CHECK-NEXT:    store <vscale x 4 x i32> zeroinitializer, ptr [[P:%.*]], align 16
; CHECK-NEXT:    ret <vscale x 4 x i32> zeroinitializer
;
  store <vscale x 4 x i32> zeroinitializer, ptr %p
  %load = load <vscale x 4 x i32>, ptr %p
  ret <vscale x 4 x i32> %load
}

define <vscale x 4 x i32> @store_forward_to_load_sideeffect(ptr %p) {
; CHECK-LABEL: @store_forward_to_load_sideeffect(
; CHECK-NEXT:    store <vscale x 4 x i32> zeroinitializer, ptr [[P:%.*]], align 16
; CHECK-NEXT:    call void asm sideeffect "", ""()
; CHECK-NEXT:    [[LOAD:%.*]] = load <vscale x 4 x i32>, ptr [[P]], align 16
; CHECK-NEXT:    ret <vscale x 4 x i32> [[LOAD]]
;
  store <vscale x 4 x i32> zeroinitializer, ptr %p
  call void asm sideeffect "", ""()
  %load = load <vscale x 4 x i32>, ptr %p
  ret <vscale x 4 x i32> %load
}

define i32 @store_clobber_load() {
; CHECK-LABEL: @store_clobber_load(
; CHECK-NEXT:    [[ALLOC:%.*]] = alloca <vscale x 4 x i32>, align 16
; CHECK-NEXT:    store <vscale x 4 x i32> undef, ptr [[ALLOC]], align 16
; CHECK-NEXT:    [[PTR:%.*]] = getelementptr <vscale x 4 x i32>, ptr [[ALLOC]], i32 0, i32 1
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, ptr [[PTR]], align 4
; CHECK-NEXT:    ret i32 [[LOAD]]
;
  %alloc = alloca <vscale x 4 x i32>
  store <vscale x 4 x i32> undef, ptr %alloc
  %ptr = getelementptr <vscale x 4 x i32>, ptr %alloc, i32 0, i32 1
  %load = load i32, ptr %ptr
  ret i32 %load
}

; Analyze Load from clobbering MemInst.

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1)

define i32 @memset_clobber_load(ptr %p) {
; CHECK-LABEL: @memset_clobber_load(
; CHECK-NEXT:    tail call void @llvm.memset.p0.i64(ptr [[P:%.*]], i8 1, i64 200, i1 false)
; CHECK-NEXT:    ret i32 16843009
;
  tail call void @llvm.memset.p0.i64(ptr %p, i8 1, i64 200, i1 false)
  %gep = getelementptr <vscale x 4 x i32>, ptr %p, i64 0, i64 5
  %load = load i32, ptr %gep
  ret i32 %load
}

define i32 @memset_clobber_load_vscaled_base(ptr %p) {
; CHECK-LABEL: @memset_clobber_load_vscaled_base(
; CHECK-NEXT:    tail call void @llvm.memset.p0.i64(ptr [[P:%.*]], i8 1, i64 200, i1 false)
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr <vscale x 4 x i32>, ptr [[P]], i64 1, i64 1
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, ptr [[GEP]], align 4
; CHECK-NEXT:    ret i32 [[LOAD]]
;
  tail call void @llvm.memset.p0.i64(ptr %p, i8 1, i64 200, i1 false)
  %gep = getelementptr <vscale x 4 x i32>, ptr %p, i64 1, i64 1
  %load = load i32, ptr %gep
  ret i32 %load
}

define i32 @memset_clobber_load_nonconst_index(ptr %p, i64 %idx1, i64 %idx2) {
; CHECK-LABEL: @memset_clobber_load_nonconst_index(
; CHECK-NEXT:    tail call void @llvm.memset.p0.i64(ptr [[P:%.*]], i8 1, i64 200, i1 false)
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr <vscale x 4 x i32>, ptr [[P]], i64 [[IDX1:%.*]], i64 [[IDX2:%.*]]
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, ptr [[GEP]], align 4
; CHECK-NEXT:    ret i32 [[LOAD]]
;
  tail call void @llvm.memset.p0.i64(ptr %p, i8 1, i64 200, i1 false)
  %gep = getelementptr <vscale x 4 x i32>, ptr %p, i64 %idx1, i64 %idx2
  %load = load i32, ptr %gep
  ret i32 %load
}


; Load elimination across BBs

define ptr @load_from_alloc_replaced_with_undef() {
; CHECK-LABEL: @load_from_alloc_replaced_with_undef(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.*]] = alloca <vscale x 4 x i32>, align 16
; CHECK-NEXT:    br i1 undef, label [[IF_END:%.*]], label [[IF_THEN:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    store <vscale x 4 x i32> zeroinitializer, ptr [[A]], align 16
; CHECK-NEXT:    br label [[IF_END]]
; CHECK:       if.end:
; CHECK-NEXT:    ret ptr [[A]]
;
entry:
  %a = alloca <vscale x 4 x i32>
  %gep = getelementptr <vscale x 4 x i32>, ptr %a, i64 0, i64 1
  %load = load i32, ptr %gep ; <- load to be eliminated
  %tobool = icmp eq i32 %load, 0 ; <- icmp to be eliminated
  br i1 %tobool, label %if.end, label %if.then

if.then:
  store <vscale x 4 x i32> zeroinitializer, ptr %a
  br label %if.end

if.end:
  ret ptr %a
}

define i32 @redundant_load_elimination_1(ptr %p) {
; CHECK-LABEL: @redundant_load_elimination_1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr <vscale x 4 x i32>, ptr [[P:%.*]], i64 1, i64 1
; CHECK-NEXT:    [[LOAD1:%.*]] = load i32, ptr [[GEP]], align 4
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[LOAD1]], 0
; CHECK-NEXT:    br i1 [[CMP]], label [[IF_THEN:%.*]], label [[IF_END:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    br label [[IF_END]]
; CHECK:       if.end:
; CHECK-NEXT:    ret i32 [[LOAD1]]
;
entry:
  %gep = getelementptr <vscale x 4 x i32>, ptr %p, i64 1, i64 1
  %load1 = load i32, ptr %gep
  %cmp = icmp eq i32 %load1, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %load2 = load i32, ptr %gep ; <- load to be eliminated
  %add = add i32 %load1, %load2
  br label %if.end

if.end:
  %result = phi i32 [ %add, %if.then ], [ %load1, %entry ]
  ret i32 %result
}

; TODO: BasicAA return MayAlias for %gep1,%gep2, could improve as NoAlias.
define void @redundant_load_elimination_2(i1 %c, ptr %p, ptr %q) {
; CHECK-LABEL: @redundant_load_elimination_2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[GEP1:%.*]] = getelementptr <vscale x 4 x i32>, ptr [[P:%.*]], i64 1, i64 1
; CHECK-NEXT:    store i32 0, ptr [[GEP1]], align 4
; CHECK-NEXT:    [[GEP2:%.*]] = getelementptr <vscale x 4 x i32>, ptr [[P]], i64 1, i64 0
; CHECK-NEXT:    store i32 1, ptr [[GEP2]], align 4
; CHECK-NEXT:    br i1 [[C:%.*]], label [[IF_ELSE:%.*]], label [[IF_THEN:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    [[T:%.*]] = load i32, ptr [[GEP1]], align 4
; CHECK-NEXT:    store i32 [[T]], ptr [[Q:%.*]], align 4
; CHECK-NEXT:    ret void
; CHECK:       if.else:
; CHECK-NEXT:    ret void
;
entry:
  %gep1 = getelementptr <vscale x 4 x i32>, ptr %p, i64 1, i64 1
  store i32 0, ptr %gep1
  %gep2 = getelementptr <vscale x 4 x i32>, ptr %p, i64 1, i64 0
  store i32 1, ptr %gep2
  br i1 %c, label %if.else, label %if.then

if.then:
  %t = load i32, ptr %gep1 ; <- load could be eliminated
  store i32 %t, ptr %q
  ret void

if.else:
  ret void
}

define void @redundant_load_elimination_zero_index(i1 %c, ptr %p, ptr %q) {
; CHECK-LABEL: @redundant_load_elimination_zero_index(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[GEP1:%.*]] = getelementptr <vscale x 4 x i32>, ptr [[P:%.*]], i64 0, i64 1
; CHECK-NEXT:    store i32 0, ptr [[GEP1]], align 4
; CHECK-NEXT:    store i32 1, ptr [[P]], align 4
; CHECK-NEXT:    br i1 [[C:%.*]], label [[IF_ELSE:%.*]], label [[IF_THEN:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    store i32 0, ptr [[Q:%.*]], align 4
; CHECK-NEXT:    ret void
; CHECK:       if.else:
; CHECK-NEXT:    ret void
;
entry:
  %gep1 = getelementptr <vscale x 4 x i32>, ptr %p, i64 0, i64 1
  store i32 0, ptr %gep1
  store i32 1, ptr %p
  br i1 %c, label %if.else, label %if.then

if.then:
  %t = load i32, ptr %gep1 ; <- load could be eliminated
  store i32 %t, ptr %q
  ret void

if.else:
  ret void
}

define void @redundant_load_elimination_zero_index_1(i1 %c, ptr %p, ptr %q, i64 %i) {
; CHECK-LABEL: @redundant_load_elimination_zero_index_1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[J:%.*]] = add i64 [[I:%.*]], 1
; CHECK-NEXT:    [[GEP1:%.*]] = getelementptr <vscale x 4 x i32>, ptr [[P:%.*]], i64 0, i64 [[J]]
; CHECK-NEXT:    store i32 0, ptr [[GEP1]], align 4
; CHECK-NEXT:    [[GEP2:%.*]] = getelementptr <vscale x 4 x i32>, ptr [[P]], i64 0, i64 [[I]]
; CHECK-NEXT:    store i32 1, ptr [[GEP2]], align 4
; CHECK-NEXT:    br i1 [[C:%.*]], label [[IF_ELSE:%.*]], label [[IF_THEN:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    store i32 0, ptr [[Q:%.*]], align 4
; CHECK-NEXT:    ret void
; CHECK:       if.else:
; CHECK-NEXT:    ret void
;
entry:
  %j = add i64 %i, 1
  %gep1 = getelementptr <vscale x 4 x i32>, ptr %p, i64 0, i64 %j
  store i32 0, ptr %gep1
  %gep2 = getelementptr <vscale x 4 x i32>, ptr %p, i64 0, i64 %i
  store i32 1, ptr %gep2
  br i1 %c, label %if.else, label %if.then

if.then:
  %t = load i32, ptr %gep1 ; <- load could be eliminated
  store i32 %t, ptr %q
  ret void

if.else:
  ret void
}
; TODO: load in if.then could have been eliminated
define void @missing_load_elimination(i1 %c, ptr %p, ptr %q, <vscale x 4 x i32> %v) {
; CHECK-LABEL: @missing_load_elimination(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    store <vscale x 4 x i32> zeroinitializer, ptr [[P:%.*]], align 16
; CHECK-NEXT:    [[P1:%.*]] = getelementptr <vscale x 4 x i32>, ptr [[P]], i64 1
; CHECK-NEXT:    store <vscale x 4 x i32> [[V:%.*]], ptr [[P1]], align 16
; CHECK-NEXT:    br i1 [[C:%.*]], label [[IF_ELSE:%.*]], label [[IF_THEN:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    [[T:%.*]] = load <vscale x 4 x i32>, ptr [[P]], align 16
; CHECK-NEXT:    store <vscale x 4 x i32> [[T]], ptr [[Q:%.*]], align 16
; CHECK-NEXT:    ret void
; CHECK:       if.else:
; CHECK-NEXT:    ret void
;
entry:
  store <vscale x 4 x i32> zeroinitializer, ptr %p
  %p1 = getelementptr <vscale x 4 x i32>, ptr %p, i64 1
  store <vscale x 4 x i32> %v, ptr %p1
  br i1 %c, label %if.else, label %if.then

if.then:
  %t = load <vscale x 4 x i32>, ptr %p ; load could be eliminated
  store <vscale x 4 x i32> %t, ptr %q
  ret void

if.else:
  ret void
}
