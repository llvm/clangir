; This test contains extremely tricky call graph structures for the inliner to
; handle correctly. They form cycles where the inliner introduces code that is
; immediately or can eventually be transformed back into the original code. And
; each step changes the call graph and so will trigger iteration. This requires
; some out-of-band way to prevent infinitely re-inlining and re-transforming the
; code.
;
; RUN: opt < %s -passes='cgscc(inline,function(sroa,instcombine))' -inline-threshold=50 -S | FileCheck %s


; The `test1_*` collection of functions form a directly cycling pattern.

define void @test1_a(ptr %ptr) {
; CHECK-LABEL: define void @test1_a(
entry:
  call void @test1_b(ptr @test1_b, i1 false, i32 0)
; Inlining and simplifying this call will reliably produce the exact same call,
; over and over again. However, each inlining increments the count, and so we
; expect this test case to stop after one round of inlining with a final
; argument of '1'.
; CHECK-NOT:     call
; CHECK:         call void @test1_b(ptr nonnull @test1_b, i1 false, i32 1)
; CHECK-NOT:     call

  ret void
}

define void @test1_b(ptr %arg, i1 %flag, i32 %inline_count) {
; CHECK-LABEL: define void @test1_b(
entry:
  %a = alloca ptr
  store ptr %arg, ptr %a
; This alloca and store should remain through any optimization.
; CHECK:         %[[A:.*]] = alloca
; CHECK:         store ptr %arg, ptr %[[A]]

  br i1 %flag, label %bb1, label %bb2

bb1:
  call void @test1_a(ptr %a) noinline
  br label %bb2

bb2:
  %p = load ptr, ptr %a
  %inline_count_inc = add i32 %inline_count, 1
  call void %p(ptr %arg, i1 %flag, i32 %inline_count_inc)
; And we should continue to load and call indirectly through optimization.
; CHECK:         %[[P:.*]] = load ptr, ptr %[[A]]
; CHECK:         call void %[[P]](

  ret void
}

define void @test2_a(ptr %ptr) {
; CHECK-LABEL: define void @test2_a(
entry:
  call void @test2_b(ptr @test2_b, ptr @test2_c, i1 false, i32 0)
; Inlining and simplifying this call will reliably produce the exact same call,
; but only after doing two rounds if inlining, first from @test2_b then
; @test2_c. We check the exact number of inlining rounds before we cut off to
; break the cycle by inspecting the last paramater that gets incremented with
; each inlined function body.
; CHECK-NOT:     call
; CHECK:         call void @test2_b(ptr nonnull @test2_b, ptr nonnull @test2_c, i1 false, i32 2)
; CHECK-NOT:     call
  ret void
}

define void @test2_b(ptr %arg1, ptr %arg2, i1 %flag, i32 %inline_count) {
; CHECK-LABEL: define void @test2_b(
entry:
  %a = alloca ptr
  store ptr %arg2, ptr %a
; This alloca and store should remain through any optimization.
; CHECK:         %[[A:.*]] = alloca
; CHECK:         store ptr %arg2, ptr %[[A]]

  br i1 %flag, label %bb1, label %bb2

bb1:
  call void @test2_a(ptr %a) noinline
  br label %bb2

bb2:
  %p = load ptr, ptr %a
  %inline_count_inc = add i32 %inline_count, 1
  call void %p(ptr %arg1, ptr %arg2, i1 %flag, i32 %inline_count_inc)
; And we should continue to load and call indirectly through optimization.
; CHECK:         %[[P:.*]] = load ptr, ptr %[[A]]
; CHECK:         call void %[[P]](

  ret void
}

define void @test2_c(ptr %arg1, ptr %arg2, i1 %flag, i32 %inline_count) {
; CHECK-LABEL: define void @test2_c(
entry:
  %a = alloca ptr
  store ptr %arg1, ptr %a
; This alloca and store should remain through any optimization.
; CHECK:         %[[A:.*]] = alloca
; CHECK:         store ptr %arg1, ptr %[[A]]

  br i1 %flag, label %bb1, label %bb2

bb1:
  call void @test2_a(ptr %a) noinline
  br label %bb2

bb2:
  %p = load ptr, ptr %a
  %inline_count_inc = add i32 %inline_count, 1
  call void %p(ptr %arg1, ptr %arg2, i1 %flag, i32 %inline_count_inc)
; And we should continue to load and call indirectly through optimization.
; CHECK:         %[[P:.*]] = load ptr, ptr %[[A]]
; CHECK:         call void %[[P]](

  ret void
}

; Another infinite inlining case. The initial callgraph is like following:
;
;         test3_a <---> test3_b
;             |         ^
;             v         |
;         test3_c <---> test3_d
;
; For all the call edges in the call graph, only test3_c and test3_d can be
; inlined into test3_a, and no other call edge can be inlined.
;
; After test3_c is inlined into test3_a, the original call edge test3_a->test3_c
; will be removed, a new call edge will be added and the call graph becomes:
;
;            test3_a <---> test3_b
;                  \      ^
;                   v    /
;     test3_c <---> test3_d
; But test3_a, test3_b, test3_c and test3_d still belong to the same SCC.
;
; Then after test3_a->test3_d is inlined, when test3_a->test3_d is converted to
; a ref edge, the original SCC will be split into two: {test3_c, test3_d} and
; {test3_a, test3_b}, immediately after the newly added ref edge
; test3_a->test3_c will be converted to a call edge, and the two SCCs will be
; merged into the original one again. During this cycle, the original SCC will
; be added into UR.CWorklist again and this creates an infinite loop.

@a = global i64 0
@b = global i64 0

; Check test3_c is inlined into test3_a once and only once.
; CHECK-LABEL: @test3_a(
; CHECK: tail call void @test3_b()
; CHECK-NEXT: tail call void @test3_d(i32 5)
; CHECK-NEXT: %[[LD1:.*]] = load i64, ptr @a
; CHECK-NEXT: %[[ADD1:.*]] = add nsw i64 %[[LD1]], 1
; CHECK-NEXT: store i64 %[[ADD1]], ptr @a
; CHECK-NEXT: %[[LD2:.*]] = load i64, ptr @b
; CHECK-NEXT: %[[ADD2:.*]] = add nsw i64 %[[LD2]], 5
; CHECK-NEXT: store i64 %[[ADD2]], ptr @b
; CHECK-NEXT: ret void

; Function Attrs: noinline
define void @test3_a() #0 {
entry:
  tail call void @test3_b()
  tail call void @test3_c(i32 5)
  %t0 = load i64, ptr @b
  %add = add nsw i64 %t0, 5
  store i64 %add, ptr @b
  ret void
}

; Function Attrs: noinline
define void @test3_b() #0 {
entry:
  tail call void @test3_a()
  %t0 = load i64, ptr @a
  %add = add nsw i64 %t0, 2
  store i64 %add, ptr @a
  ret void
}

define void @test3_d(i32 %i) {
entry:
  %cmp = icmp eq i32 %i, 5
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %call = tail call i64 @random()
  %t0 = load i64, ptr @a
  %add = add nsw i64 %t0, %call
  store i64 %add, ptr @a
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  tail call void @test3_c(i32 %i)
  tail call void @test3_b()
  %t6 = load i64, ptr @a
  %add79 = add nsw i64 %t6, 3
  store i64 %add79, ptr @a
  ret void
}

define void @test3_c(i32 %i) {
entry:
  %cmp = icmp eq i32 %i, 5
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %call = tail call i64 @random()
  %t0 = load i64, ptr @a
  %add = add nsw i64 %t0, %call
  store i64 %add, ptr @a
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  tail call void @test3_d(i32 %i)
  %t6 = load i64, ptr @a
  %add85 = add nsw i64 %t6, 1
  store i64 %add85, ptr @a
  ret void
}

declare i64 @random()

attributes #0 = { noinline }
