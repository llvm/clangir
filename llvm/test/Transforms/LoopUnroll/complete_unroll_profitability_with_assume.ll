; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -S < %s -passes=loop-unroll -unroll-threshold=42 | FileCheck %s --check-prefix=ANALYZE-FULL

; This test is supposed to check that calls to @llvm.assume builtin are not
; prohibiting the analysis of full unroll profitability in case the cost of the
; unrolled loop (not acounting to any simplifications done by such unrolling) is
; higher than some threshold.
;
; Ensure that we indeed are testing this code path by verifying that the loop is
; not unrolled without such analysis:

; RUN: opt -S < %s -passes=loop-unroll -unroll-threshold=42 -unroll-max-iteration-count-to-analyze=2 \
; RUN:   -unroll-peel-max-count=0  | FileCheck %s --check-prefix=DONT-ANALYZE-FULL

; Function Attrs: nounwind
declare void @llvm.assume(i1) #1

define i32 @foo(ptr %a) {
; ANALYZE-FULL-LABEL: @foo(
; ANALYZE-FULL-NEXT:  entry:
; ANALYZE-FULL-NEXT:    br label [[FOR_BODY:%.*]]
; ANALYZE-FULL:       for.body:
; ANALYZE-FULL-NEXT:    br i1 true, label [[DO_STORE:%.*]], label [[FOR_NEXT:%.*]]
; ANALYZE-FULL:       do_store:
; ANALYZE-FULL-NEXT:    store i32 0, ptr [[A:%.*]], align 4
; ANALYZE-FULL-NEXT:    br label [[FOR_NEXT]]
; ANALYZE-FULL:       for.next:
; ANALYZE-FULL-NEXT:    br i1 true, label [[DO_STORE_1:%.*]], label [[FOR_NEXT_1:%.*]]
; ANALYZE-FULL:       do_store.1:
; ANALYZE-FULL-NEXT:    [[GEP_1:%.*]] = getelementptr i32, ptr [[A]], i32 1
; ANALYZE-FULL-NEXT:    store i32 1, ptr [[GEP_1]], align 4
; ANALYZE-FULL-NEXT:    br label [[FOR_NEXT_1]]
; ANALYZE-FULL:       for.next.1:
; ANALYZE-FULL-NEXT:    br i1 true, label [[DO_STORE_2:%.*]], label [[FOR_NEXT_2:%.*]]
; ANALYZE-FULL:       do_store.2:
; ANALYZE-FULL-NEXT:    [[GEP_2:%.*]] = getelementptr i32, ptr [[A]], i32 2
; ANALYZE-FULL-NEXT:    store i32 2, ptr [[GEP_2]], align 4
; ANALYZE-FULL-NEXT:    br label [[FOR_NEXT_2]]
; ANALYZE-FULL:       for.next.2:
; ANALYZE-FULL-NEXT:    br i1 true, label [[DO_STORE_3:%.*]], label [[FOR_NEXT_3:%.*]]
; ANALYZE-FULL:       do_store.3:
; ANALYZE-FULL-NEXT:    [[GEP_3:%.*]] = getelementptr i32, ptr [[A]], i32 3
; ANALYZE-FULL-NEXT:    store i32 3, ptr [[GEP_3]], align 4
; ANALYZE-FULL-NEXT:    br label [[FOR_NEXT_3]]
; ANALYZE-FULL:       for.next.3:
; ANALYZE-FULL-NEXT:    br i1 false, label [[DO_STORE_4:%.*]], label [[FOR_NEXT_4:%.*]]
; ANALYZE-FULL:       do_store.4:
; ANALYZE-FULL-NEXT:    [[GEP_4:%.*]] = getelementptr i32, ptr [[A]], i32 4
; ANALYZE-FULL-NEXT:    store i32 4, ptr [[GEP_4]], align 4
; ANALYZE-FULL-NEXT:    br label [[FOR_NEXT_4]]
; ANALYZE-FULL:       for.next.4:
; ANALYZE-FULL-NEXT:    br i1 false, label [[DO_STORE_5:%.*]], label [[FOR_NEXT_5:%.*]]
; ANALYZE-FULL:       do_store.5:
; ANALYZE-FULL-NEXT:    [[GEP_5:%.*]] = getelementptr i32, ptr [[A]], i32 5
; ANALYZE-FULL-NEXT:    store i32 5, ptr [[GEP_5]], align 4
; ANALYZE-FULL-NEXT:    br label [[FOR_NEXT_5]]
; ANALYZE-FULL:       for.next.5:
; ANALYZE-FULL-NEXT:    br i1 false, label [[DO_STORE_6:%.*]], label [[FOR_NEXT_6:%.*]]
; ANALYZE-FULL:       do_store.6:
; ANALYZE-FULL-NEXT:    [[GEP_6:%.*]] = getelementptr i32, ptr [[A]], i32 6
; ANALYZE-FULL-NEXT:    store i32 6, ptr [[GEP_6]], align 4
; ANALYZE-FULL-NEXT:    br label [[FOR_NEXT_6]]
; ANALYZE-FULL:       for.next.6:
; ANALYZE-FULL-NEXT:    br i1 false, label [[DO_STORE_7:%.*]], label [[FOR_NEXT_7:%.*]]
; ANALYZE-FULL:       do_store.7:
; ANALYZE-FULL-NEXT:    [[GEP_7:%.*]] = getelementptr i32, ptr [[A]], i32 7
; ANALYZE-FULL-NEXT:    store i32 7, ptr [[GEP_7]], align 4
; ANALYZE-FULL-NEXT:    br label [[FOR_NEXT_7]]
; ANALYZE-FULL:       for.next.7:
; ANALYZE-FULL-NEXT:    br i1 false, label [[DO_STORE_8:%.*]], label [[FOR_NEXT_8:%.*]]
; ANALYZE-FULL:       do_store.8:
; ANALYZE-FULL-NEXT:    [[GEP_8:%.*]] = getelementptr i32, ptr [[A]], i32 8
; ANALYZE-FULL-NEXT:    store i32 8, ptr [[GEP_8]], align 4
; ANALYZE-FULL-NEXT:    br label [[FOR_NEXT_8]]
; ANALYZE-FULL:       for.next.8:
; ANALYZE-FULL-NEXT:    ret i32 9
;
; DONT-ANALYZE-FULL-LABEL: @foo(
; DONT-ANALYZE-FULL-NEXT:  entry:
; DONT-ANALYZE-FULL-NEXT:    br label [[FOR_BODY:%.*]]
; DONT-ANALYZE-FULL:       for.body:
; DONT-ANALYZE-FULL-NEXT:    [[INDVAR:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[INDVAR_NEXT:%.*]], [[FOR_NEXT:%.*]] ]
; DONT-ANALYZE-FULL-NEXT:    [[INDVAR_NEXT]] = add i32 [[INDVAR]], 1
; DONT-ANALYZE-FULL-NEXT:    [[CMP:%.*]] = icmp ule i32 [[INDVAR]], 20
; DONT-ANALYZE-FULL-NEXT:    tail call void @llvm.assume(i1 [[CMP]])
; DONT-ANALYZE-FULL-NEXT:    [[CMP2:%.*]] = icmp ule i32 [[INDVAR]], 3
; DONT-ANALYZE-FULL-NEXT:    br i1 [[CMP2]], label [[DO_STORE:%.*]], label [[FOR_NEXT]]
; DONT-ANALYZE-FULL:       do_store:
; DONT-ANALYZE-FULL-NEXT:    [[GEP:%.*]] = getelementptr i32, ptr [[A:%.*]], i32 [[INDVAR]]
; DONT-ANALYZE-FULL-NEXT:    store i32 [[INDVAR]], ptr [[GEP]], align 4
; DONT-ANALYZE-FULL-NEXT:    br label [[FOR_NEXT]]
; DONT-ANALYZE-FULL:       for.next:
; DONT-ANALYZE-FULL-NEXT:    [[EXITCOND:%.*]] = icmp ne i32 [[INDVAR_NEXT]], 9
; DONT-ANALYZE-FULL-NEXT:    br i1 [[EXITCOND]], label [[FOR_BODY]], label [[LOOPEXIT:%.*]]
; DONT-ANALYZE-FULL:       loopexit:
; DONT-ANALYZE-FULL-NEXT:    [[INDVAR_NEXT_LCSSA:%.*]] = phi i32 [ [[INDVAR_NEXT]], [[FOR_NEXT]] ]
; DONT-ANALYZE-FULL-NEXT:    ret i32 [[INDVAR_NEXT_LCSSA]]
;
entry:
  br label %for.body
for.body:
  %indvar = phi i32 [ 0, %entry ], [ %indvar.next, %for.next ]
  %indvar.next = add i32 %indvar, 1
  %cmp = icmp ule i32 %indvar, 20
  tail call void @llvm.assume(i1 %cmp)
  %cmp2 = icmp ule i32 %indvar, 3
  br i1 %cmp2, label %do_store, label %for.next

do_store:
  %gep = getelementptr i32, ptr %a, i32 %indvar
  store i32 %indvar, ptr %gep
  br label %for.next

for.next:
  %exitcond = icmp ne i32 %indvar.next, 9
  br i1 %exitcond, label %for.body, label %loopexit
loopexit:
  ret i32 %indvar.next
}
