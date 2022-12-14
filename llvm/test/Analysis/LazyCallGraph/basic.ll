; RUN: opt -disable-output -passes=print-lcg %s 2>&1 | FileCheck %s
;
; Basic validation of the call graph analysis used in the new pass manager.

define void @f() {
; CHECK-LABEL: Edges in function: f
; CHECK-NOT: ->

entry:
  ret void
}

; A bunch more functions just to make it easier to test several call edges at once.
define void @f1() {
  ret void
}
define void @f2() {
  ret void
}
define void @f3() {
  ret void
}
define void @f4() {
  ret void
}
define void @f5() {
  ret void
}
define void @f6() {
  ret void
}
define void @f7() {
  ret void
}
define void @f8() {
  ret void
}
define void @f9() {
  ret void
}
define void @f10() {
  ret void
}
define void @f11() {
  ret void
}
define void @f12() {
  ret void
}

declare i32 @__gxx_personality_v0(...)

define void @test0() {
; CHECK-LABEL: Edges in function: test0
; CHECK-NEXT: call -> f
; CHECK-NOT: ->

entry:
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  ret void
}

define ptr @test1(ptr %x) personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: Edges in function: test1
; CHECK-NEXT: call -> f6
; CHECK-NEXT: call -> f10
; CHECK-NEXT: ref -> f12
; CHECK-NEXT: ref -> f11
; CHECK-NEXT: ref -> f7
; CHECK-NEXT: ref -> f9
; CHECK-NEXT: ref -> f8
; CHECK-NEXT: ref -> f5
; CHECK-NEXT: ref -> f4
; CHECK-NEXT: ref -> f3
; CHECK-NEXT: ref -> f2
; CHECK-NEXT: ref -> f1
; CHECK-NOT: ->

entry:
  br label %next

dead:
  br label %next

next:
  phi ptr [ @f1, %entry ], [ @f2, %dead ]
  select i1 true, ptr @f3, ptr @f4
  store ptr @f5, ptr %x
  call void @f6()
  call void (ptr, ptr) @f7(ptr @f8, ptr @f9)
  invoke void @f10() to label %exit unwind label %unwind

exit:
  ret ptr @f11

unwind:
  %res = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } { ptr @f12, i32 42 }
}

@g = global ptr @f1
@g1 = global [4 x ptr] [ptr @f2, ptr @f3, ptr @f4, ptr @f5]
@g2 = global {i8, ptr, i8} {i8 1, ptr @f6, i8 2}
@h = constant ptr @f7

define void @test2() {
; CHECK-LABEL: Edges in function: test2
; CHECK-NEXT: ref -> f7
; CHECK-NEXT: ref -> f6
; CHECK-NEXT: ref -> f5
; CHECK-NEXT: ref -> f4
; CHECK-NEXT: ref -> f3
; CHECK-NEXT: ref -> f2
; CHECK-NEXT: ref -> f1
; CHECK-NOT: ->

  load ptr, ptr @g
  load ptr, ptr getelementptr ([4 x ptr], ptr @g1, i32 0, i32 2)
  load ptr, ptr getelementptr ({i8, ptr, i8}, ptr @g2, i32 0, i32 1)
  load ptr, ptr @h
  ret void
}

@test3_ptr = external global ptr

define void @test3_aa1() {
; CHECK-LABEL: Edges in function: test3_aa1
; CHECK-NEXT: call -> test3_aa2
; CHECK-NEXT: ref -> test3_ab1
; CHECK-NOT: ->

entry:
  call void @test3_aa2()
  store ptr @test3_ab1, ptr @test3_ptr
  ret void
}

define void @test3_aa2() {
; CHECK-LABEL: Edges in function: test3_aa2
; CHECK-NEXT: call -> test3_aa1
; CHECK-NEXT: call -> test3_ab2
; CHECK-NOT: ->

entry:
  call void @test3_aa1()
  call void @test3_ab2()
  ret void
}

define void @test3_ab1() {
; CHECK-LABEL: Edges in function: test3_ab1
; CHECK-NEXT: call -> test3_ab2
; CHECK-NEXT: call -> test3_ac1
; CHECK-NOT: ->

entry:
  call void @test3_ab2()
  call void @test3_ac1()
  ret void
}

define void @test3_ab2() {
; CHECK-LABEL: Edges in function: test3_ab2
; CHECK-NEXT: call -> test3_ab1
; CHECK-NEXT: call -> test3_ba1
; CHECK-NOT: ->

entry:
  call void @test3_ab1()
  call void @test3_ba1()
  ret void
}

define void @test3_ac1() {
; CHECK-LABEL: Edges in function: test3_ac1
; CHECK-NEXT: call -> test3_ac2
; CHECK-NEXT: ref -> test3_aa2
; CHECK-NOT: ->

entry:
  call void @test3_ac2()
  store ptr @test3_aa2, ptr @test3_ptr
  ret void
}

define void @test3_ac2() {
; CHECK-LABEL: Edges in function: test3_ac2
; CHECK-NEXT: call -> test3_ac1
; CHECK-NEXT: ref -> test3_ba1
; CHECK-NOT: ->

entry:
  call void @test3_ac1()
  store ptr @test3_ba1, ptr @test3_ptr
  ret void
}

define void @test3_ba1() {
; CHECK-LABEL: Edges in function: test3_ba1
; CHECK-NEXT: call -> test3_bb1
; CHECK-NEXT: ref -> test3_ca1
; CHECK-NOT: ->

entry:
  call void @test3_bb1()
  store ptr @test3_ca1, ptr @test3_ptr
  ret void
}

define void @test3_bb1() {
; CHECK-LABEL: Edges in function: test3_bb1
; CHECK-NEXT: call -> test3_ca2
; CHECK-NEXT: ref -> test3_ba1
; CHECK-NOT: ->

entry:
  call void @test3_ca2()
  store ptr @test3_ba1, ptr @test3_ptr
  ret void
}

define void @test3_ca1() {
; CHECK-LABEL: Edges in function: test3_ca1
; CHECK-NEXT: call -> test3_ca2
; CHECK-NOT: ->

entry:
  call void @test3_ca2()
  ret void
}

define void @test3_ca2() {
; CHECK-LABEL: Edges in function: test3_ca2
; CHECK-NEXT: call -> test3_ca3
; CHECK-NOT: ->

entry:
  call void @test3_ca3()
  ret void
}

define void @test3_ca3() {
; CHECK-LABEL: Edges in function: test3_ca3
; CHECK-NEXT: call -> test3_ca1
; CHECK-NOT: ->

entry:
  call void @test3_ca1()
  ret void
}

; Verify the SCCs formed.
;
; CHECK-LABEL: RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      f
;
; CHECK-LABEL: RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      f1
;
; CHECK-LABEL: RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      f2
;
; CHECK-LABEL: RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      f3
;
; CHECK-LABEL: RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      f4
;
; CHECK-LABEL: RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      f5
;
; CHECK-LABEL: RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      f6
;
; CHECK-LABEL: RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      f7
;
; CHECK-LABEL: RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      f8
;
; CHECK-LABEL: RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      f9
;
; CHECK-LABEL: RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      f10
;
; CHECK-LABEL: RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      f11
;
; CHECK-LABEL: RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      f12
;
; CHECK-LABEL: RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      test0
;
; CHECK-LABEL: RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      test1
;
; CHECK-LABEL: RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      test2
;
; CHECK-LABEL: RefSCC with 1 call SCCs:
; CHECK-NEXT:    SCC with 3 functions:
; CHECK-NEXT:      test3_ca2
; CHECK-NEXT:      test3_ca3
; CHECK-NEXT:      test3_ca1
;
; CHECK-LABEL: RefSCC with 2 call SCCs:
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      test3_bb1
; CHECK-NEXT:    SCC with 1 functions:
; CHECK-NEXT:      test3_ba1
;
; CHECK-LABEL: RefSCC with 3 call SCCs:
; CHECK-NEXT:    SCC with 2 functions:
; CHECK-NEXT:      test3_ac1
; CHECK-NEXT:      test3_ac2
; CHECK-NEXT:    SCC with 2 functions:
; CHECK-NEXT:      test3_ab2
; CHECK-NEXT:      test3_ab1
; CHECK-NEXT:    SCC with 2 functions:
; CHECK-NEXT:      test3_aa1
; CHECK-NEXT:      test3_aa2
