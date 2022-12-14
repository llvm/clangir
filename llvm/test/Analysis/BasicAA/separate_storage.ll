; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -passes='gvn' -basic-aa-separate-storage -S | FileCheck %s

declare void @llvm.assume(i1)

; Test basic queries.

define i8 @simple_no(ptr %p1, ptr %p2) {
; CHECK-LABEL: @simple_no(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    store i8 0, ptr [[P1:%.*]], align 1
; CHECK-NEXT:    store i8 1, ptr [[P2:%.*]], align 1
; CHECK-NEXT:    [[LOADOFSTORE:%.*]] = load i8, ptr [[P1]], align 1
; CHECK-NEXT:    ret i8 [[LOADOFSTORE]]
;
entry:
  store i8 0, ptr %p1
  store i8 1, ptr %p2
  %loadofstore = load i8, ptr %p1
  ret i8 %loadofstore
}

define i8 @simple_yes(ptr %p1, ptr %p2) {
; CHECK-LABEL: @simple_yes(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.assume(i1 true) [ "separate_storage"(ptr [[P1:%.*]], ptr [[P2:%.*]]) ]
; CHECK-NEXT:    store i8 0, ptr [[P1]], align 1
; CHECK-NEXT:    store i8 1, ptr [[P2]], align 1
; CHECK-NEXT:    ret i8 0
;
entry:
  call void @llvm.assume(i1 1) ["separate_storage"(ptr %p1, ptr %p2)]
  store i8 0, ptr %p1
  store i8 1, ptr %p2
  %loadofstore = load i8, ptr %p1
  ret i8 %loadofstore
}

define i8 @ptr_to_ptr_no(ptr %pp) {
; CHECK-LABEL: @ptr_to_ptr_no(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[P_BASE:%.*]] = load ptr, ptr [[PP:%.*]], align 8
; CHECK-NEXT:    store i8 0, ptr [[P_BASE]], align 1
; CHECK-NEXT:    [[P_BASE2:%.*]] = load ptr, ptr [[PP]], align 8
; CHECK-NEXT:    [[LOADOFSTORE:%.*]] = load i8, ptr [[P_BASE2]], align 1
; CHECK-NEXT:    ret i8 [[LOADOFSTORE]]
;
entry:
  %p_base = load ptr, ptr %pp
  store i8 0, ptr %p_base
  %p_base2 = load ptr, ptr %pp
  %loadofstore = load i8, ptr %p_base2
  ret i8 %loadofstore
}

define i8 @ptr_to_ptr_yes(ptr %pp) {
; CHECK-LABEL: @ptr_to_ptr_yes(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[P_BASE:%.*]] = load ptr, ptr [[PP:%.*]], align 8
; CHECK-NEXT:    call void @llvm.assume(i1 true) [ "separate_storage"(ptr [[P_BASE]], ptr [[PP]]) ]
; CHECK-NEXT:    store i8 0, ptr [[P_BASE]], align 1
; CHECK-NEXT:    ret i8 0
;
entry:
  %p_base = load ptr, ptr %pp
  call void @llvm.assume(i1 1) ["separate_storage"(ptr %p_base, ptr %pp)]
  store i8 0, ptr %p_base
  %p_base2 = load ptr, ptr %pp
  %loadofstore = load i8, ptr %p_base2
  ret i8 %loadofstore
}

; The analysis should only kick in if executed (or will be executed) at the
; given program point.

define i8 @flow_sensitive(ptr %p1, ptr %p2, i1 %cond) {
; CHECK-LABEL: @flow_sensitive(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[TRUE_BRANCH:%.*]], label [[FALSE_BRANCH:%.*]]
; CHECK:       true_branch:
; CHECK-NEXT:    store i8 11, ptr [[P1:%.*]], align 1
; CHECK-NEXT:    store i8 22, ptr [[P2:%.*]], align 1
; CHECK-NEXT:    [[LOADOFSTORE_TRUE:%.*]] = load i8, ptr [[P1]], align 1
; CHECK-NEXT:    br label [[ENDIF:%.*]]
; CHECK:       false_branch:
; CHECK-NEXT:    call void @llvm.assume(i1 true) [ "separate_storage"(ptr [[P1]], ptr [[P2]]) ]
; CHECK-NEXT:    store i8 33, ptr [[P1]], align 1
; CHECK-NEXT:    store i8 44, ptr [[P2]], align 1
; CHECK-NEXT:    br label [[ENDIF]]
; CHECK:       endif:
; CHECK-NEXT:    [[LOADOFSTORE:%.*]] = phi i8 [ [[LOADOFSTORE_TRUE]], [[TRUE_BRANCH]] ], [ 33, [[FALSE_BRANCH]] ]
; CHECK-NEXT:    ret i8 [[LOADOFSTORE]]
;
entry:
  br i1 %cond, label %true_branch, label %false_branch

true_branch:
  store i8 11, ptr %p1
  store i8 22, ptr %p2
  %loadofstore_true = load i8, ptr %p1
  br label %endif

false_branch:
  call void @llvm.assume(i1 1) ["separate_storage"(ptr %p1, ptr %p2)]
  store i8 33, ptr %p1
  store i8 44, ptr %p2
  %loadofstore_false = load i8, ptr %p1
  br label %endif

endif:
  %loadofstore = phi i8 [ %loadofstore_true, %true_branch ], [ %loadofstore_false, %false_branch ]
  ret i8 %loadofstore
}

define i8 @flow_sensitive_with_dominator(ptr %p1, ptr %p2, i1 %cond) {
; CHECK-LABEL: @flow_sensitive_with_dominator(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.assume(i1 true) [ "separate_storage"(ptr [[P1:%.*]], ptr [[P2:%.*]]) ]
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[TRUE_BRANCH:%.*]], label [[FALSE_BRANCH:%.*]]
; CHECK:       true_branch:
; CHECK-NEXT:    store i8 11, ptr [[P1]], align 1
; CHECK-NEXT:    store i8 22, ptr [[P2]], align 1
; CHECK-NEXT:    br label [[ENDIF:%.*]]
; CHECK:       false_branch:
; CHECK-NEXT:    store i8 33, ptr [[P1]], align 1
; CHECK-NEXT:    store i8 44, ptr [[P2]], align 1
; CHECK-NEXT:    br label [[ENDIF]]
; CHECK:       endif:
; CHECK-NEXT:    [[LOADOFSTORE:%.*]] = phi i8 [ 11, [[TRUE_BRANCH]] ], [ 33, [[FALSE_BRANCH]] ]
; CHECK-NEXT:    ret i8 [[LOADOFSTORE]]
;
entry:
  call void @llvm.assume(i1 1) ["separate_storage"(ptr %p1, ptr %p2)]
  br i1 %cond, label %true_branch, label %false_branch

true_branch:
  store i8 11, ptr %p1
  store i8 22, ptr %p2
  %loadofstore_true = load i8, ptr %p1
  br label %endif

false_branch:
  store i8 33, ptr %p1
  store i8 44, ptr %p2
  %loadofstore_false = load i8, ptr %p1
  br label %endif

endif:
  %loadofstore = phi i8 [ %loadofstore_true, %true_branch ], [ %loadofstore_false, %false_branch ]
  ret i8 %loadofstore
}

; Hints are relative to entire regions of storage, not particular pointers
; inside them. We should know that the whole ranges are disjoint given hints at
; offsets.

define i8 @offset_agnostic(ptr %p1, ptr %p2) {
; CHECK-LABEL: @offset_agnostic(
; CHECK-NEXT:    [[ACCESS1:%.*]] = getelementptr inbounds i8, ptr [[P1:%.*]], i64 12
; CHECK-NEXT:    [[ACCESS2:%.*]] = getelementptr inbounds i8, ptr [[P2:%.*]], i64 34
; CHECK-NEXT:    [[HINT1:%.*]] = getelementptr inbounds i8, ptr [[P1]], i64 56
; CHECK-NEXT:    [[HINT2:%.*]] = getelementptr inbounds i8, ptr [[P2]], i64 78
; CHECK-NEXT:    call void @llvm.assume(i1 true) [ "separate_storage"(ptr [[HINT1]], ptr [[HINT2]]) ]
; CHECK-NEXT:    store i8 0, ptr [[ACCESS1]], align 1
; CHECK-NEXT:    store i8 1, ptr [[ACCESS2]], align 1
; CHECK-NEXT:    ret i8 0
;
  %access1 = getelementptr inbounds i8, ptr %p1, i64 12
  %access2 = getelementptr inbounds i8, ptr %p2, i64 34

  %hint1 = getelementptr inbounds i8, ptr %p1, i64 56
  %hint2 = getelementptr inbounds i8, ptr %p2, i64 78
  call void @llvm.assume(i1 1) ["separate_storage"(ptr %hint1, ptr %hint2)]

  store i8 0, ptr %access1
  store i8 1, ptr %access2
  %loadofstore = load i8, ptr %access1
  ret i8 %loadofstore
}
