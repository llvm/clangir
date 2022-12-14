; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -passes=constraint-elimination -S %s | FileCheck %s

; Tests for using inbounds information from GEPs.

declare void @noundef(ptr noundef)

define i1 @inbounds_poison_is_ub1(ptr %src, i32 %n, i32 %idx) {
; CHECK-LABEL: @inbounds_poison_is_ub1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UPPER:%.*]] = getelementptr inbounds i32, ptr [[SRC:%.*]], i64 5
; CHECK-NEXT:    call void @noundef(ptr [[UPPER]])
; CHECK-NEXT:    [[CMP_IDX:%.*]] = icmp ult i32 [[IDX:%.*]], [[N:%.*]]
; CHECK-NEXT:    [[IDX_EXT:%.*]] = zext i32 [[IDX]] to i64
; CHECK-NEXT:    [[SRC_IDX_4:%.*]] = getelementptr i32, ptr [[SRC]], i64 4
; CHECK-NEXT:    [[CMP_UPPER_4:%.*]] = icmp ule ptr [[SRC_IDX_4]], [[UPPER]]
; CHECK-NEXT:    [[SRC_IDX_5:%.*]] = getelementptr i32, ptr [[SRC]], i64 5
; CHECK-NEXT:    [[CMP_UPPER_5:%.*]] = icmp ule ptr [[SRC_IDX_5]], [[UPPER]]
; CHECK-NEXT:    [[RES_0:%.*]] = xor i1 [[CMP_UPPER_4]], [[CMP_UPPER_5]]
; CHECK-NEXT:    [[SRC_IDX_6:%.*]] = getelementptr i32, ptr [[SRC]], i64 6
; CHECK-NEXT:    [[CMP_UPPER_6:%.*]] = icmp ule ptr [[SRC_IDX_6]], [[UPPER]]
; CHECK-NEXT:    [[RES_1:%.*]] = xor i1 [[RES_0]], [[CMP_UPPER_6]]
; CHECK-NEXT:    [[SRC_IDX_NEG_1:%.*]] = getelementptr i32, ptr [[SRC]], i64 -1
; CHECK-NEXT:    [[CMP_UPPER_NEG_1:%.*]] = icmp ule ptr [[SRC_IDX_NEG_1]], [[UPPER]]
; CHECK-NEXT:    [[RES_2:%.*]] = xor i1 [[RES_1]], [[CMP_UPPER_NEG_1]]
; CHECK-NEXT:    ret i1 [[RES_2]]
;
entry:
  %upper = getelementptr inbounds i32, ptr %src, i64 5
  call void @noundef(ptr %upper)
  %cmp.idx = icmp ult i32 %idx, %n
  %idx.ext = zext i32 %idx to i64
  %src.idx.4 = getelementptr i32, ptr %src, i64 4
  %cmp.upper.4 = icmp ule ptr %src.idx.4, %upper
  %src.idx.5 = getelementptr i32, ptr %src, i64 5
  %cmp.upper.5 = icmp ule ptr %src.idx.5, %upper
  %res.0 = xor i1 %cmp.upper.4, %cmp.upper.5

  %src.idx.6 = getelementptr i32, ptr %src, i64 6
  %cmp.upper.6 = icmp ule ptr %src.idx.6, %upper
  %res.1 = xor i1 %res.0, %cmp.upper.6

  %src.idx.neg.1 = getelementptr i32, ptr %src, i64 -1
  %cmp.upper.neg.1 = icmp ule ptr %src.idx.neg.1, %upper
  %res.2 = xor i1 %res.1, %cmp.upper.neg.1
  ret i1 %res.2
}

; %start + %n.ext is guaranteed to not overflow (due to inbounds).
; %start + %idx.ext does not overflow if %idx.ext <= %n.ext.
define i1 @inbounds_poison_is_ub2(ptr %src, i32 %n, i32 %idx) {
; CHECK-LABEL: @inbounds_poison_is_ub2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[N_EXT:%.*]] = zext i32 [[N:%.*]] to i64
; CHECK-NEXT:    [[UPPER:%.*]] = getelementptr inbounds i32, ptr [[SRC:%.*]], i64 [[N_EXT]]
; CHECK-NEXT:    call void @noundef(ptr [[UPPER]])
; CHECK-NEXT:    [[CMP_IDX:%.*]] = icmp ult i32 [[IDX:%.*]], [[N]]
; CHECK-NEXT:    [[IDX_EXT:%.*]] = zext i32 [[IDX]] to i64
; CHECK-NEXT:    [[SRC_IDX:%.*]] = getelementptr i32, ptr [[SRC]], i64 [[IDX_EXT]]
; CHECK-NEXT:    br i1 [[CMP_IDX]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    [[CMP_UPPER_1:%.*]] = icmp ule ptr [[SRC_IDX]], [[UPPER]]
; CHECK-NEXT:    ret i1 [[CMP_UPPER_1]]
; CHECK:       else:
; CHECK-NEXT:    [[CMP_UPPER_2:%.*]] = icmp ule ptr [[SRC_IDX]], [[UPPER]]
; CHECK-NEXT:    ret i1 [[CMP_UPPER_2]]
;
entry:
  %n.ext = zext i32 %n to i64
  %upper = getelementptr inbounds i32, ptr %src, i64 %n.ext
  call void @noundef(ptr %upper)
  %cmp.idx = icmp ult i32 %idx, %n
  %idx.ext = zext i32 %idx to i64
  %src.idx = getelementptr i32, ptr %src, i64 %idx.ext
  br i1 %cmp.idx, label %then, label %else

then:
  %cmp.upper.1 = icmp ule ptr %src.idx, %upper
  ret i1 %cmp.upper.1

else:
  %cmp.upper.2 = icmp ule ptr %src.idx, %upper
  ret i1 %cmp.upper.2
}

; Same as inbounds_poison_is_ub2, but with individual GEPs in the %then and
; %else blocks.
define i1 @inbounds_poison_is_ub3(ptr %src, i32 %n, i32 %idx) {
; CHECK-LABEL: @inbounds_poison_is_ub3(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[N_EXT:%.*]] = zext i32 [[N:%.*]] to i64
; CHECK-NEXT:    [[IDX_EXT:%.*]] = zext i32 [[IDX:%.*]] to i64
; CHECK-NEXT:    [[CMP_IDX:%.*]] = icmp ult i64 [[IDX_EXT]], [[N_EXT]]
; CHECK-NEXT:    br i1 [[CMP_IDX]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    [[UPPER_1:%.*]] = getelementptr inbounds i32, ptr [[SRC:%.*]], i64 [[N_EXT]]
; CHECK-NEXT:    call void @noundef(ptr [[UPPER_1]])
; CHECK-NEXT:    [[SRC_IDX_1:%.*]] = getelementptr i32, ptr [[SRC]], i64 [[IDX_EXT]]
; CHECK-NEXT:    [[CMP_UPPER_1:%.*]] = icmp ule ptr [[SRC_IDX_1]], [[UPPER_1]]
; CHECK-NEXT:    ret i1 [[CMP_UPPER_1]]
; CHECK:       else:
; CHECK-NEXT:    [[UPPER_2:%.*]] = getelementptr inbounds i32, ptr [[SRC]], i64 [[N_EXT]]
; CHECK-NEXT:    call void @noundef(ptr [[UPPER_2]])
; CHECK-NEXT:    [[SRC_IDX_2:%.*]] = getelementptr i32, ptr [[SRC]], i64 [[IDX_EXT]]
; CHECK-NEXT:    [[CMP_UPPER_2:%.*]] = icmp ule ptr [[SRC_IDX_2]], [[UPPER_2]]
; CHECK-NEXT:    ret i1 [[CMP_UPPER_2]]
;
entry:
  %n.ext = zext i32 %n to i64
  %idx.ext = zext i32 %idx to i64
  %cmp.idx = icmp ult i64 %idx.ext, %n.ext
  br i1 %cmp.idx, label %then, label %else

then:
  %upper.1 = getelementptr inbounds i32, ptr %src, i64 %n.ext
  call void @noundef(ptr %upper.1)
  %src.idx.1 = getelementptr i32, ptr %src, i64 %idx.ext
  %cmp.upper.1 = icmp ule ptr %src.idx.1, %upper.1
  ret i1 %cmp.upper.1

else:
  %upper.2 = getelementptr inbounds i32, ptr %src, i64 %n.ext
  call void @noundef(ptr %upper.2)
  %src.idx.2 = getelementptr i32, ptr %src, i64 %idx.ext
  %cmp.upper.2 = icmp ule ptr %src.idx.2, %upper.2
  ret i1 %cmp.upper.2
}

; The function does not have UB if %upper is poison because of an overflow. Do
; not simplify anything. In this particular case, the returned result will be
; poison in this case, so it could be simplified, but currently we cannot
; distinguish that case.
define i1 @inbounds_poison_does_not_cause_ub(ptr %src, i32 %n, i32 %idx) {
; CHECK-LABEL: @inbounds_poison_does_not_cause_ub(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[N_EXT:%.*]] = zext i32 [[N:%.*]] to i64
; CHECK-NEXT:    [[IDX_EXT:%.*]] = zext i32 [[IDX:%.*]] to i64
; CHECK-NEXT:    [[UPPER:%.*]] = getelementptr inbounds i32, ptr [[SRC:%.*]], i64 [[N_EXT]]
; CHECK-NEXT:    [[SRC_IDX:%.*]] = getelementptr i32, ptr [[SRC]], i64 [[IDX_EXT]]
; CHECK-NEXT:    [[CMP_IDX:%.*]] = icmp ult i64 [[IDX_EXT]], [[N_EXT]]
; CHECK-NEXT:    br i1 [[CMP_IDX]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    [[CMP_UPPER_1:%.*]] = icmp ule ptr [[SRC_IDX]], [[UPPER]]
; CHECK-NEXT:    ret i1 [[CMP_UPPER_1]]
; CHECK:       else:
; CHECK-NEXT:    [[CMP_UPPER_2:%.*]] = icmp ule ptr [[SRC_IDX]], [[UPPER]]
; CHECK-NEXT:    ret i1 [[CMP_UPPER_2]]
;
entry:
  %n.ext = zext i32 %n to i64
  %idx.ext = zext i32 %idx to i64
  %upper = getelementptr inbounds i32, ptr %src, i64 %n.ext
  %src.idx = getelementptr i32, ptr %src, i64 %idx.ext
  %cmp.idx = icmp ult i64 %idx.ext, %n.ext
  br i1 %cmp.idx, label %then, label %else

then:
  %cmp.upper.1 = icmp ule ptr %src.idx, %upper
  ret i1 %cmp.upper.1

else:
  %cmp.upper.2 = icmp ule ptr %src.idx, %upper
  ret i1 %cmp.upper.2
}

; Same as @inbounds_poison_does_not_cause_ub, but with separate GEPs in the
; %then and %else blocks.
define i1 @inbounds_poison_does_not_cause_ub2(ptr %src, i32 %n, i32 %idx) {
; CHECK-LABEL: @inbounds_poison_does_not_cause_ub2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[N_EXT:%.*]] = zext i32 [[N:%.*]] to i64
; CHECK-NEXT:    [[IDX_EXT:%.*]] = zext i32 [[IDX:%.*]] to i64
; CHECK-NEXT:    [[CMP_IDX:%.*]] = icmp ult i64 [[IDX_EXT]], [[N_EXT]]
; CHECK-NEXT:    br i1 [[CMP_IDX]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    [[UPPER_1:%.*]] = getelementptr inbounds i32, ptr [[SRC:%.*]], i64 [[N_EXT]]
; CHECK-NEXT:    [[SRC_IDX_1:%.*]] = getelementptr i32, ptr [[SRC]], i64 [[IDX_EXT]]
; CHECK-NEXT:    [[CMP_UPPER_1:%.*]] = icmp ule ptr [[SRC_IDX_1]], [[UPPER_1]]
; CHECK-NEXT:    ret i1 [[CMP_UPPER_1]]
; CHECK:       else:
; CHECK-NEXT:    [[SRC_IDX_2:%.*]] = getelementptr i32, ptr [[SRC]], i64 [[IDX_EXT]]
; CHECK-NEXT:    [[UPPER_2:%.*]] = getelementptr inbounds i32, ptr [[SRC]], i64 [[N_EXT]]
; CHECK-NEXT:    [[CMP_UPPER_2:%.*]] = icmp ule ptr [[SRC_IDX_2]], [[UPPER_2]]
; CHECK-NEXT:    ret i1 [[CMP_UPPER_2]]
;
entry:
  %n.ext = zext i32 %n to i64
  %idx.ext = zext i32 %idx to i64
  %cmp.idx = icmp ult i64 %idx.ext, %n.ext
  br i1 %cmp.idx, label %then, label %else

then:
  %upper.1 = getelementptr inbounds i32, ptr %src, i64 %n.ext
  %src.idx.1 = getelementptr i32, ptr %src, i64 %idx.ext
  %cmp.upper.1 = icmp ule ptr %src.idx.1, %upper.1
  ret i1 %cmp.upper.1

else:
  %src.idx.2 = getelementptr i32, ptr %src, i64 %idx.ext
  %upper.2 = getelementptr inbounds i32, ptr %src, i64 %n.ext
  %cmp.upper.2 = icmp ule ptr %src.idx.2, %upper.2
  ret i1 %cmp.upper.2
}

define i1 @no_zexts_indices_may_be_negative(ptr %src, i32 %n, i32 %idx) {
; CHECK-LABEL: @no_zexts_indices_may_be_negative(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UPPER:%.*]] = getelementptr inbounds i32, ptr [[SRC:%.*]], i32 [[N:%.*]]
; CHECK-NEXT:    call void @noundef(ptr [[UPPER]])
; CHECK-NEXT:    [[SRC_IDX:%.*]] = getelementptr i32, ptr [[SRC]], i32 [[IDX:%.*]]
; CHECK-NEXT:    [[CMP_IDX:%.*]] = icmp ult i32 [[IDX]], [[N]]
; CHECK-NEXT:    br i1 [[CMP_IDX]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    [[CMP_UPPER_1:%.*]] = icmp ule ptr [[SRC_IDX]], [[UPPER]]
; CHECK-NEXT:    ret i1 [[CMP_UPPER_1]]
; CHECK:       else:
; CHECK-NEXT:    [[CMP_UPPER_2:%.*]] = icmp ule ptr [[SRC_IDX]], [[UPPER]]
; CHECK-NEXT:    ret i1 [[CMP_UPPER_2]]
;
entry:
  %upper = getelementptr inbounds i32, ptr %src, i32 %n
  call void @noundef(ptr %upper)
  %src.idx = getelementptr i32, ptr %src, i32 %idx
  %cmp.idx = icmp ult i32 %idx, %n
  br i1 %cmp.idx, label %then, label %else

then:
  %cmp.upper.1 = icmp ule ptr %src.idx, %upper
  ret i1 %cmp.upper.1

else:
  %cmp.upper.2 = icmp ule ptr %src.idx, %upper
  ret i1 %cmp.upper.2
}

; Tests for multiple inbound GEPs, make sure the largest upper bound is used.
define i1 @multiple_upper_bounds(ptr %src, i32 %n, i32 %idx) {
; CHECK-LABEL: @multiple_upper_bounds(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[N_EXT:%.*]] = zext i32 [[N:%.*]] to i64
; CHECK-NEXT:    [[UPPER_1:%.*]] = getelementptr inbounds i32, ptr [[SRC:%.*]], i64 1
; CHECK-NEXT:    call void @noundef(ptr [[UPPER_1]])
; CHECK-NEXT:    [[UPPER_2:%.*]] = getelementptr inbounds i32, ptr [[SRC]], i64 [[N_EXT]]
; CHECK-NEXT:    call void @noundef(ptr [[UPPER_2]])
; CHECK-NEXT:    [[CMP_IDX:%.*]] = icmp ult i32 [[IDX:%.*]], [[N]]
; CHECK-NEXT:    [[IDX_EXT:%.*]] = zext i32 [[IDX]] to i64
; CHECK-NEXT:    [[SRC_IDX:%.*]] = getelementptr i32, ptr [[SRC]], i64 [[IDX_EXT]]
; CHECK-NEXT:    br i1 [[CMP_IDX]], label [[THEN:%.*]], label [[ELSE:%.*]]
; CHECK:       then:
; CHECK-NEXT:    [[CMP_UPPER_1:%.*]] = icmp ule ptr [[SRC_IDX]], [[UPPER_2]]
; CHECK-NEXT:    ret i1 [[CMP_UPPER_1]]
; CHECK:       else:
; CHECK-NEXT:    [[CMP_UPPER_2:%.*]] = icmp ule ptr [[SRC_IDX]], [[UPPER_2]]
; CHECK-NEXT:    ret i1 [[CMP_UPPER_2]]
;
entry:
  %n.ext = zext i32 %n to i64
  %upper.1 = getelementptr inbounds i32, ptr %src, i64 1
  call void @noundef(ptr %upper.1)
  %upper.2 = getelementptr inbounds i32, ptr %src, i64 %n.ext
  call void @noundef(ptr %upper.2)
  %cmp.idx = icmp ult i32 %idx, %n
  %idx.ext = zext i32 %idx to i64
  %src.idx = getelementptr i32, ptr %src, i64 %idx.ext
  br i1 %cmp.idx, label %then, label %else

then:
  %cmp.upper.1 = icmp ule ptr %src.idx, %upper.2
  ret i1 %cmp.upper.1

else:
  %cmp.upper.2 = icmp ule ptr %src.idx, %upper.2
  ret i1 %cmp.upper.2
}

define i1 @multiple_upper_bounds2(ptr %src, i32 %n, i32 %idx) {
; CHECK-LABEL: @multiple_upper_bounds2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[N_EXT:%.*]] = zext i32 [[N:%.*]] to i64
; CHECK-NEXT:    [[UPPER_1:%.*]] = getelementptr inbounds i32, ptr [[SRC:%.*]], i64 1
; CHECK-NEXT:    call void @noundef(ptr [[UPPER_1]])
; CHECK-NEXT:    [[UPPER_2:%.*]] = getelementptr inbounds i32, ptr [[SRC]], i64 4
; CHECK-NEXT:    call void @noundef(ptr [[UPPER_2]])
; CHECK-NEXT:    [[SRC_IDX:%.*]] = getelementptr i32, ptr [[SRC]], i64 4
; CHECK-NEXT:    [[CMP_UPPER_1:%.*]] = icmp ule ptr [[SRC_IDX]], [[UPPER_2]]
; CHECK-NEXT:    ret i1 [[CMP_UPPER_1]]
;
entry:
  %n.ext = zext i32 %n to i64
  %upper.1 = getelementptr inbounds i32, ptr %src, i64 1
  call void @noundef(ptr %upper.1)
  %upper.2 = getelementptr inbounds i32, ptr %src, i64 4
  call void @noundef(ptr %upper.2)
  %src.idx = getelementptr i32, ptr %src, i64 4
  %cmp.upper.1 = icmp ule ptr %src.idx, %upper.2
  ret i1 %cmp.upper.1
}

define i1 @multiple_upper_bounds3(ptr %src, i32 %n, i32 %idx) {
; CHECK-LABEL: @multiple_upper_bounds3(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[N_EXT:%.*]] = zext i32 [[N:%.*]] to i64
; CHECK-NEXT:    [[UPPER_1:%.*]] = getelementptr inbounds i32, ptr [[SRC:%.*]], i64 4
; CHECK-NEXT:    call void @noundef(ptr [[UPPER_1]])
; CHECK-NEXT:    [[UPPER_2:%.*]] = getelementptr inbounds i32, ptr [[SRC]], i64 1
; CHECK-NEXT:    call void @noundef(ptr [[UPPER_2]])
; CHECK-NEXT:    [[SRC_IDX:%.*]] = getelementptr i32, ptr [[SRC]], i64 4
; CHECK-NEXT:    [[CMP_UPPER_1:%.*]] = icmp ule ptr [[SRC_IDX]], [[UPPER_1]]
; CHECK-NEXT:    ret i1 [[CMP_UPPER_1]]
;
entry:
  %n.ext = zext i32 %n to i64
  %upper.1 = getelementptr inbounds i32, ptr %src, i64 4
  call void @noundef(ptr %upper.1)
  %upper.2 = getelementptr inbounds i32, ptr %src, i64 1
  call void @noundef(ptr %upper.2)
  %src.idx = getelementptr i32, ptr %src, i64 4
  %cmp.upper.1 = icmp ule ptr %src.idx, %upper.1
  ret i1 %cmp.upper.1
}

; %src.idx + 5 may overflow.
define i1 @multiple_upper_bounds4(ptr %src, i32 %n, i32 %idx) {
; CHECK-LABEL: @multiple_upper_bounds4(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[N_EXT:%.*]] = zext i32 [[N:%.*]] to i64
; CHECK-NEXT:    [[UPPER_1:%.*]] = getelementptr inbounds i32, ptr [[SRC:%.*]], i64 1
; CHECK-NEXT:    call void @noundef(ptr [[UPPER_1]])
; CHECK-NEXT:    [[UPPER_2:%.*]] = getelementptr inbounds i32, ptr [[SRC]], i64 4
; CHECK-NEXT:    call void @noundef(ptr [[UPPER_2]])
; CHECK-NEXT:    [[SRC_IDX:%.*]] = getelementptr i32, ptr [[SRC]], i64 5
; CHECK-NEXT:    [[CMP_UPPER_1:%.*]] = icmp ule ptr [[SRC_IDX]], [[UPPER_2]]
; CHECK-NEXT:    ret i1 [[CMP_UPPER_1]]
;
entry:
  %n.ext = zext i32 %n to i64
  %upper.1 = getelementptr inbounds i32, ptr %src, i64 1
  call void @noundef(ptr %upper.1)
  %upper.2 = getelementptr inbounds i32, ptr %src, i64 4
  call void @noundef(ptr %upper.2)
  %src.idx = getelementptr i32, ptr %src, i64 5
  %cmp.upper.1 = icmp ule ptr %src.idx, %upper.2
  ret i1 %cmp.upper.1
}
