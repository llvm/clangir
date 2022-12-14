; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -passes=tailcallelim -verify-dom-info -S | FileCheck %s
; PR4323

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Several cases where tail call elimination should move the load above the call,
; then eliminate the tail recursion.



@global = external global i32		; <ptr> [#uses=1]
@extern_weak_global = extern_weak global i32		; <ptr> [#uses=1]


; This load can be moved above the call because the function won't write to it
; and the call has no side effects.
define fastcc i32 @raise_load_1(ptr %a_arg, i32 %a_len_arg, i32 %start_arg) nounwind readonly willreturn {
; CHECK-LABEL: @raise_load_1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[TAILRECURSE:%.*]]
; CHECK:       tailrecurse:
; CHECK-NEXT:    [[ACCUMULATOR_TR:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[TMP10:%.*]], [[ELSE:%.*]] ]
; CHECK-NEXT:    [[START_ARG_TR:%.*]] = phi i32 [ [[START_ARG:%.*]], [[ENTRY]] ], [ [[TMP7:%.*]], [[ELSE]] ]
; CHECK-NEXT:    [[TMP2:%.*]] = icmp sge i32 [[START_ARG_TR]], [[A_LEN_ARG:%.*]]
; CHECK-NEXT:    br i1 [[TMP2]], label [[IF:%.*]], label [[ELSE]]
; CHECK:       if:
; CHECK-NEXT:    [[ACCUMULATOR_RET_TR:%.*]] = add i32 0, [[ACCUMULATOR_TR]]
; CHECK-NEXT:    ret i32 [[ACCUMULATOR_RET_TR]]
; CHECK:       else:
; CHECK-NEXT:    [[TMP7]] = add i32 [[START_ARG_TR]], 1
; CHECK-NEXT:    [[TMP9:%.*]] = load i32, ptr [[A_ARG:%.*]], align 4
; CHECK-NEXT:    [[TMP10]] = add i32 [[TMP9]], [[ACCUMULATOR_TR]]
; CHECK-NEXT:    br label [[TAILRECURSE]]
;
entry:
  %tmp2 = icmp sge i32 %start_arg, %a_len_arg		; <i1> [#uses=1]
  br i1 %tmp2, label %if, label %else

if:		; preds = %entry
  ret i32 0

else:		; preds = %entry
  %tmp7 = add i32 %start_arg, 1		; <i32> [#uses=1]
  %tmp8 = call fastcc i32 @raise_load_1(ptr %a_arg, i32 %a_len_arg, i32 %tmp7)		; <i32> [#uses=1]
  %tmp9 = load i32, ptr %a_arg		; <i32> [#uses=1]
  %tmp10 = add i32 %tmp9, %tmp8		; <i32> [#uses=1]
  ret i32 %tmp10
}


; This load can be moved above the call because the function won't write to it
; and the load provably can't trap.
define fastcc i32 @raise_load_2(ptr %a_arg, i32 %a_len_arg, i32 %start_arg) readonly {
; CHECK-LABEL: @raise_load_2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[TAILRECURSE:%.*]]
; CHECK:       tailrecurse:
; CHECK-NEXT:    [[ACCUMULATOR_TR:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[TMP10:%.*]], [[RECURSE:%.*]] ]
; CHECK-NEXT:    [[START_ARG_TR:%.*]] = phi i32 [ [[START_ARG:%.*]], [[ENTRY]] ], [ [[TMP7:%.*]], [[RECURSE]] ]
; CHECK-NEXT:    [[TMP2:%.*]] = icmp sge i32 [[START_ARG_TR]], [[A_LEN_ARG:%.*]]
; CHECK-NEXT:    br i1 [[TMP2]], label [[IF:%.*]], label [[ELSE:%.*]]
; CHECK:       if:
; CHECK-NEXT:    [[ACCUMULATOR_RET_TR:%.*]] = add i32 0, [[ACCUMULATOR_TR]]
; CHECK-NEXT:    ret i32 [[ACCUMULATOR_RET_TR]]
; CHECK:       else:
; CHECK-NEXT:    [[NULLCHECK:%.*]] = icmp eq ptr [[A_ARG:%.*]], null
; CHECK-NEXT:    br i1 [[NULLCHECK]], label [[UNWIND:%.*]], label [[RECURSE]]
; CHECK:       unwind:
; CHECK-NEXT:    unreachable
; CHECK:       recurse:
; CHECK-NEXT:    [[TMP7]] = add i32 [[START_ARG_TR]], 1
; CHECK-NEXT:    [[TMP9:%.*]] = load i32, ptr @global, align 4
; CHECK-NEXT:    [[TMP10]] = add i32 [[TMP9]], [[ACCUMULATOR_TR]]
; CHECK-NEXT:    br label [[TAILRECURSE]]
;
entry:
  %tmp2 = icmp sge i32 %start_arg, %a_len_arg		; <i1> [#uses=1]
  br i1 %tmp2, label %if, label %else

if:		; preds = %entry
  ret i32 0

else:		; preds = %entry
  %nullcheck = icmp eq ptr %a_arg, null		; <i1> [#uses=1]
  br i1 %nullcheck, label %unwind, label %recurse

unwind:		; preds = %else
  unreachable

recurse:		; preds = %else
  %tmp7 = add i32 %start_arg, 1		; <i32> [#uses=1]
  %tmp8 = call fastcc i32 @raise_load_2(ptr %a_arg, i32 %a_len_arg, i32 %tmp7)		; <i32> [#uses=1]
  %tmp9 = load i32, ptr @global		; <i32> [#uses=1]
  %tmp10 = add i32 %tmp9, %tmp8		; <i32> [#uses=1]
  ret i32 %tmp10
}


; This load can be safely moved above the call (even though it's from an
; extern_weak global) because the call has no side effects.
define fastcc i32 @raise_load_3(ptr %a_arg, i32 %a_len_arg, i32 %start_arg) nounwind readonly willreturn {
; CHECK-LABEL: @raise_load_3(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[TAILRECURSE:%.*]]
; CHECK:       tailrecurse:
; CHECK-NEXT:    [[ACCUMULATOR_TR:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[TMP10:%.*]], [[ELSE:%.*]] ]
; CHECK-NEXT:    [[START_ARG_TR:%.*]] = phi i32 [ [[START_ARG:%.*]], [[ENTRY]] ], [ [[TMP7:%.*]], [[ELSE]] ]
; CHECK-NEXT:    [[TMP2:%.*]] = icmp sge i32 [[START_ARG_TR]], [[A_LEN_ARG:%.*]]
; CHECK-NEXT:    br i1 [[TMP2]], label [[IF:%.*]], label [[ELSE]]
; CHECK:       if:
; CHECK-NEXT:    [[ACCUMULATOR_RET_TR:%.*]] = add i32 0, [[ACCUMULATOR_TR]]
; CHECK-NEXT:    ret i32 [[ACCUMULATOR_RET_TR]]
; CHECK:       else:
; CHECK-NEXT:    [[TMP7]] = add i32 [[START_ARG_TR]], 1
; CHECK-NEXT:    [[TMP9:%.*]] = load i32, ptr @extern_weak_global, align 4
; CHECK-NEXT:    [[TMP10]] = add i32 [[TMP9]], [[ACCUMULATOR_TR]]
; CHECK-NEXT:    br label [[TAILRECURSE]]
;
entry:
  %tmp2 = icmp sge i32 %start_arg, %a_len_arg		; <i1> [#uses=1]
  br i1 %tmp2, label %if, label %else

if:		; preds = %entry
  ret i32 0

else:		; preds = %entry
  %tmp7 = add i32 %start_arg, 1		; <i32> [#uses=1]
  %tmp8 = call fastcc i32 @raise_load_3(ptr %a_arg, i32 %a_len_arg, i32 %tmp7)		; <i32> [#uses=1]
  %tmp9 = load i32, ptr @extern_weak_global		; <i32> [#uses=1]
  %tmp10 = add i32 %tmp9, %tmp8		; <i32> [#uses=1]
  ret i32 %tmp10
}


; The second load can be safely moved above the call even though it's from an
; unknown pointer (which normally means it might trap) because the first load
; proves it doesn't trap.
define fastcc i32 @raise_load_4(ptr %a_arg, i32 %a_len_arg, i32 %start_arg) readonly {
; CHECK-LABEL: @raise_load_4(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[TAILRECURSE:%.*]]
; CHECK:       tailrecurse:
; CHECK-NEXT:    [[ACCUMULATOR_TR:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[TMP10:%.*]], [[RECURSE:%.*]] ]
; CHECK-NEXT:    [[A_LEN_ARG_TR:%.*]] = phi i32 [ [[A_LEN_ARG:%.*]], [[ENTRY]] ], [ [[FIRST:%.*]], [[RECURSE]] ]
; CHECK-NEXT:    [[START_ARG_TR:%.*]] = phi i32 [ [[START_ARG:%.*]], [[ENTRY]] ], [ [[TMP7:%.*]], [[RECURSE]] ]
; CHECK-NEXT:    [[TMP2:%.*]] = icmp sge i32 [[START_ARG_TR]], [[A_LEN_ARG_TR]]
; CHECK-NEXT:    br i1 [[TMP2]], label [[IF:%.*]], label [[ELSE:%.*]]
; CHECK:       if:
; CHECK-NEXT:    [[ACCUMULATOR_RET_TR:%.*]] = add i32 0, [[ACCUMULATOR_TR]]
; CHECK-NEXT:    ret i32 [[ACCUMULATOR_RET_TR]]
; CHECK:       else:
; CHECK-NEXT:    [[NULLCHECK:%.*]] = icmp eq ptr [[A_ARG:%.*]], null
; CHECK-NEXT:    br i1 [[NULLCHECK]], label [[UNWIND:%.*]], label [[RECURSE]]
; CHECK:       unwind:
; CHECK-NEXT:    unreachable
; CHECK:       recurse:
; CHECK-NEXT:    [[TMP7]] = add i32 [[START_ARG_TR]], 1
; CHECK-NEXT:    [[FIRST]] = load i32, ptr [[A_ARG]], align 4
; CHECK-NEXT:    [[SECOND:%.*]] = load i32, ptr [[A_ARG]], align 4
; CHECK-NEXT:    [[TMP10]] = add i32 [[SECOND]], [[ACCUMULATOR_TR]]
; CHECK-NEXT:    br label [[TAILRECURSE]]
;
entry:
  %tmp2 = icmp sge i32 %start_arg, %a_len_arg		; <i1> [#uses=1]
  br i1 %tmp2, label %if, label %else

if:		; preds = %entry
  ret i32 0

else:		; preds = %entry
  %nullcheck = icmp eq ptr %a_arg, null		; <i1> [#uses=1]
  br i1 %nullcheck, label %unwind, label %recurse

unwind:		; preds = %else
  unreachable

recurse:		; preds = %else
  %tmp7 = add i32 %start_arg, 1		; <i32> [#uses=1]
  %first = load i32, ptr %a_arg		; <i32> [#uses=1]
  %tmp8 = call fastcc i32 @raise_load_4(ptr %a_arg, i32 %first, i32 %tmp7)		; <i32> [#uses=1]
  %second = load i32, ptr %a_arg		; <i32> [#uses=1]
  %tmp10 = add i32 %second, %tmp8		; <i32> [#uses=1]
  ret i32 %tmp10
}

; This load can be moved above the call because the function won't write to it
; and the a_arg is dereferenceable.
define fastcc i32 @raise_load_5(ptr dereferenceable(4) align 4 %a_arg, i32 %a_len_arg, i32 %start_arg) readonly nofree nosync {
; CHECK-LABEL: @raise_load_5(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[TAILRECURSE:%.*]]
; CHECK:       tailrecurse:
; CHECK-NEXT:    [[ACCUMULATOR_TR:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[TMP10:%.*]], [[ELSE:%.*]] ]
; CHECK-NEXT:    [[START_ARG_TR:%.*]] = phi i32 [ [[START_ARG:%.*]], [[ENTRY]] ], [ [[TMP7:%.*]], [[ELSE]] ]
; CHECK-NEXT:    [[TMP2:%.*]] = icmp sge i32 [[START_ARG_TR]], [[A_LEN_ARG:%.*]]
; CHECK-NEXT:    br i1 [[TMP2]], label [[IF:%.*]], label [[ELSE]]
; CHECK:       if:
; CHECK-NEXT:    [[ACCUMULATOR_RET_TR:%.*]] = add i32 0, [[ACCUMULATOR_TR]]
; CHECK-NEXT:    ret i32 [[ACCUMULATOR_RET_TR]]
; CHECK:       else:
; CHECK-NEXT:    [[TMP7]] = add i32 [[START_ARG_TR]], 1
; CHECK-NEXT:    [[TMP9:%.*]] = load i32, ptr [[A_ARG:%.*]], align 4
; CHECK-NEXT:    [[TMP10]] = add i32 [[TMP9]], [[ACCUMULATOR_TR]]
; CHECK-NEXT:    br label [[TAILRECURSE]]
;
entry:
  %tmp2 = icmp sge i32 %start_arg, %a_len_arg		; <i1> [#uses=1]
  br i1 %tmp2, label %if, label %else

if:		; preds = %entry
  ret i32 0

else:		; preds = %entry
  %tmp7 = add i32 %start_arg, 1		; <i32> [#uses=1]
  %tmp8 = call fastcc i32 @raise_load_5(ptr %a_arg, i32 %a_len_arg, i32 %tmp7)		; <i32> [#uses=1]
  %tmp9 = load i32, ptr %a_arg		; <i32> [#uses=1]
  %tmp10 = add i32 %tmp9, %tmp8		; <i32> [#uses=1]
  ret i32 %tmp10
}

; This load can be moved above the call because the function call does not write to the memory the load
; is accessing and the load is safe to speculate.
define fastcc i32 @raise_load_6(ptr %a_arg, i32 %a_len_arg, i32 %start_arg) nounwind  {
; CHECK-LABEL: @raise_load_6(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[S:%.*]] = alloca i32, align 4
; CHECK-NEXT:    br label [[TAILRECURSE:%.*]]
; CHECK:       tailrecurse:
; CHECK-NEXT:    [[ACCUMULATOR_TR:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[TMP10:%.*]], [[ELSE:%.*]] ]
; CHECK-NEXT:    [[START_ARG_TR:%.*]] = phi i32 [ [[START_ARG:%.*]], [[ENTRY]] ], [ [[TMP7:%.*]], [[ELSE]] ]
; CHECK-NEXT:    store i32 4, ptr [[S]], align 4
; CHECK-NEXT:    [[TMP2:%.*]] = icmp sge i32 [[START_ARG_TR]], [[A_LEN_ARG:%.*]]
; CHECK-NEXT:    br i1 [[TMP2]], label [[IF:%.*]], label [[ELSE]]
; CHECK:       if:
; CHECK-NEXT:    store i32 1, ptr [[A_ARG:%.*]], align 4
; CHECK-NEXT:    [[ACCUMULATOR_RET_TR:%.*]] = add i32 0, [[ACCUMULATOR_TR]]
; CHECK-NEXT:    ret i32 [[ACCUMULATOR_RET_TR]]
; CHECK:       else:
; CHECK-NEXT:    [[TMP7]] = add i32 [[START_ARG_TR]], 1
; CHECK-NEXT:    [[TMP9:%.*]] = load i32, ptr [[S]], align 4
; CHECK-NEXT:    [[TMP10]] = add i32 [[TMP9]], [[ACCUMULATOR_TR]]
; CHECK-NEXT:    br label [[TAILRECURSE]]
;
entry:
  %s = alloca i32
  store i32 4, ptr %s
  %tmp2 = icmp sge i32 %start_arg, %a_len_arg		; <i1> [#uses=1]
  br i1 %tmp2, label %if, label %else

if:		; preds = %entry
  store i32 1, ptr %a_arg
  ret i32 0

else:		; preds = %entry
  %tmp7 = add i32 %start_arg, 1		; <i32> [#uses=1]
  %tmp8 = call fastcc i32 @raise_load_6(ptr %a_arg, i32 %a_len_arg, i32 %tmp7)		; <i32> [#uses=1]
  %tmp9 = load i32, ptr %s		; <i32> [#uses=1]
  %tmp10 = add i32 %tmp9, %tmp8		; <i32> [#uses=1]
  ret i32 %tmp10
}
