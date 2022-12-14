; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -S -passes=dfa-jump-threading -dfa-max-path-length=6 %s | FileCheck %s

; Make the path
;   <%for.body %case1 %case1.1 %case1.2 %case1.3 %case1.4 %for.inc %for.end>
; too long so that it is not jump-threaded.
define i32 @max_path_length(i32 %num) {
; CHECK-LABEL: @max_path_length(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[COUNT:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[INC:%.*]], [[FOR_INC:%.*]] ]
; CHECK-NEXT:    [[STATE:%.*]] = phi i32 [ 1, [[ENTRY]] ], [ [[STATE_NEXT:%.*]], [[FOR_INC]] ]
; CHECK-NEXT:    switch i32 [[STATE]], label [[FOR_INC_JT1:%.*]] [
; CHECK-NEXT:    i32 1, label [[CASE1:%.*]]
; CHECK-NEXT:    i32 2, label [[CASE2:%.*]]
; CHECK-NEXT:    ]
; CHECK:       for.body.jt2:
; CHECK-NEXT:    [[COUNT_JT2:%.*]] = phi i32 [ [[INC_JT2:%.*]], [[FOR_INC_JT2:%.*]] ]
; CHECK-NEXT:    [[STATE_JT2:%.*]] = phi i32 [ [[STATE_NEXT_JT2:%.*]], [[FOR_INC_JT2]] ]
; CHECK-NEXT:    br label [[CASE2]]
; CHECK:       for.body.jt1:
; CHECK-NEXT:    [[COUNT_JT1:%.*]] = phi i32 [ [[INC_JT1:%.*]], [[FOR_INC_JT1]] ]
; CHECK-NEXT:    [[STATE_JT1:%.*]] = phi i32 [ [[STATE_NEXT_JT1:%.*]], [[FOR_INC_JT1]] ]
; CHECK-NEXT:    br label [[CASE1]]
; CHECK:       case1:
; CHECK-NEXT:    [[COUNT2:%.*]] = phi i32 [ [[COUNT_JT1]], [[FOR_BODY_JT1:%.*]] ], [ [[COUNT]], [[FOR_BODY]] ]
; CHECK-NEXT:    br label [[CASE1_1:%.*]]
; CHECK:       case1.1:
; CHECK-NEXT:    br label [[CASE1_2:%.*]]
; CHECK:       case1.2:
; CHECK-NEXT:    br label [[CASE1_3:%.*]]
; CHECK:       case1.3:
; CHECK-NEXT:    br label [[CASE1_4:%.*]]
; CHECK:       case1.4:
; CHECK-NEXT:    br label [[FOR_INC]]
; CHECK:       case2:
; CHECK-NEXT:    [[COUNT1:%.*]] = phi i32 [ [[COUNT_JT2]], [[FOR_BODY_JT2:%.*]] ], [ [[COUNT]], [[FOR_BODY]] ]
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[COUNT1]], 50
; CHECK-NEXT:    br i1 [[CMP]], label [[FOR_INC_JT1]], label [[SI_UNFOLD_FALSE:%.*]]
; CHECK:       si.unfold.false:
; CHECK-NEXT:    br label [[FOR_INC_JT2]]
; CHECK:       for.inc:
; CHECK-NEXT:    [[STATE_NEXT]] = phi i32 [ 2, [[CASE1_4]] ]
; CHECK-NEXT:    [[INC]] = add nsw i32 [[COUNT2]], 1
; CHECK-NEXT:    [[CMP_EXIT:%.*]] = icmp slt i32 [[INC]], [[NUM:%.*]]
; CHECK-NEXT:    br i1 [[CMP_EXIT]], label [[FOR_BODY]], label [[FOR_END:%.*]]
; CHECK:       for.inc.jt2:
; CHECK-NEXT:    [[STATE_NEXT_JT2]] = phi i32 [ 2, [[SI_UNFOLD_FALSE]] ]
; CHECK-NEXT:    [[INC_JT2]] = add nsw i32 [[COUNT1]], 1
; CHECK-NEXT:    [[CMP_EXIT_JT2:%.*]] = icmp slt i32 [[INC_JT2]], [[NUM]]
; CHECK-NEXT:    br i1 [[CMP_EXIT_JT2]], label [[FOR_BODY_JT2]], label [[FOR_END]]
; CHECK:       for.inc.jt1:
; CHECK-NEXT:    [[COUNT3:%.*]] = phi i32 [ [[COUNT1]], [[CASE2]] ], [ [[COUNT]], [[FOR_BODY]] ]
; CHECK-NEXT:    [[STATE_NEXT_JT1]] = phi i32 [ 1, [[CASE2]] ], [ 1, [[FOR_BODY]] ]
; CHECK-NEXT:    [[INC_JT1]] = add nsw i32 [[COUNT3]], 1
; CHECK-NEXT:    [[CMP_EXIT_JT1:%.*]] = icmp slt i32 [[INC_JT1]], [[NUM]]
; CHECK-NEXT:    br i1 [[CMP_EXIT_JT1]], label [[FOR_BODY_JT1]], label [[FOR_END]]
; CHECK:       for.end:
; CHECK-NEXT:    ret i32 0
;
entry:
  br label %for.body

for.body:
  %count = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %state = phi i32 [ 1, %entry ], [ %state.next, %for.inc ]
  switch i32 %state, label %for.inc [
  i32 1, label %case1
  i32 2, label %case2
  ]

case1:
  br label %case1.1

case1.1:
  br label %case1.2

case1.2:
  br label %case1.3

case1.3:
  br label %case1.4

case1.4:
  br label %for.inc

case2:
  %cmp = icmp eq i32 %count, 50
  %sel = select i1 %cmp, i32 1, i32 2
  br label %for.inc

for.inc:
  %state.next = phi i32 [ %sel, %case2 ], [ 1, %for.body ], [ 2, %case1.4 ]
  %inc = add nsw i32 %count, 1
  %cmp.exit = icmp slt i32 %inc, %num
  br i1 %cmp.exit, label %for.body, label %for.end

for.end:
  ret i32 0
}
