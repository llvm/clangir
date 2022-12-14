; RUN: llc < %s -mtriple=aarch64-apple-ios | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-apple-ios -enable-shrink-wrap=true | FileCheck %s
; Shrink wrapping currently does not kick in because we have a TLS CALL
; in the entry block and it will clobber the link register.

; RUN: llc < %s -mtriple=aarch64-apple-ios -O0 -fast-isel | FileCheck --check-prefix=CHECK-O0 %s

%struct.S = type { i8 }

@sg = internal thread_local global %struct.S zeroinitializer, align 1
@__dso_handle = external global i8
@__tls_guard = internal thread_local unnamed_addr global i1 false
@sum1 = internal thread_local global i32 0, align 4

declare ptr @_ZN1SC1Ev(ptr returned)
declare ptr @_ZN1SD1Ev(ptr returned)
declare i32 @_tlv_atexit(ptr, ptr, ptr)

define cxx_fast_tlscc nonnull ptr @_ZTW2sg() nounwind {
  %.b.i = load i1, ptr @__tls_guard, align 1
  br i1 %.b.i, label %__tls_init.exit, label %init.i

init.i:
  store i1 true, ptr @__tls_guard, align 1
  %call.i.i = tail call ptr @_ZN1SC1Ev(ptr nonnull @sg)
  %1 = tail call i32 @_tlv_atexit(ptr nonnull @_ZN1SD1Ev, ptr nonnull @sg, ptr nonnull @__dso_handle)
  br label %__tls_init.exit

__tls_init.exit:
  ret ptr @sg
}

; CHECK-LABEL: _ZTW2sg
; CHECK-NOT: stp d31, d30
; CHECK-NOT: stp d29, d28
; CHECK-NOT: stp d27, d26
; CHECK-NOT: stp d25, d24
; CHECK-NOT: stp d23, d22
; CHECK-NOT: stp d21, d20
; CHECK-NOT: stp d19, d18
; CHECK-NOT: stp d17, d16
; CHECK-NOT: stp d7, d6
; CHECK-NOT: stp d5, d4
; CHECK-NOT: stp d3, d2
; CHECK-NOT: stp d1, d0
; CHECK-NOT: stp x20, x19
; FIXME: The splitting logic in the register allocator fails to split along
;        control flow here, we used to get this right by accident before...
; CHECK-NOTXX: stp x14, x13
; CHECK-NOT: stp x12, x11
; CHECK-NOT: stp x10, x9
; CHECK-NOT: stp x8, x7
; CHECK-NOT: stp x6, x5
; CHECK-NOT: stp x4, x3
; CHECK-NOT: stp x2, x1
; CHECK: blr
; CHECK: tbnz w{{.*}}, #0, [[BB_end:.?LBB0_[0-9]+]]
; CHECK: blr
; CHECK: tlv_atexit
; CHECK: [[BB_end]]:
; CHECK: blr
; CHECK-NOT: ldp x2, x1
; CHECK-NOT: ldp x4, x3
; CHECK-NOT: ldp x6, x5
; CHECK-NOT: ldp x8, x7
; CHECK-NOT: ldp x10, x9
; CHECK-NOT: ldp x12, x11
; CHECK-NOTXX: ldp x14, x13
; CHECK-NOT: ldp x20, x19
; CHECK-NOT: ldp d1, d0
; CHECK-NOT: ldp d3, d2
; CHECK-NOT: ldp d5, d4
; CHECK-NOT: ldp d7, d6
; CHECK-NOT: ldp d17, d16
; CHECK-NOT: ldp d19, d18
; CHECK-NOT: ldp d21, d20
; CHECK-NOT: ldp d23, d22
; CHECK-NOT: ldp d25, d24
; CHECK-NOT: ldp d27, d26
; CHECK-NOT: ldp d29, d28
; CHECK-NOT: ldp d31, d30

; CHECK-O0-LABEL: _ZTW2sg
; CHECK-O0: stp d31, d30
; CHECK-O0: stp d29, d28
; CHECK-O0: stp d27, d26
; CHECK-O0: stp d25, d24
; CHECK-O0: stp d23, d22
; CHECK-O0: stp d21, d20
; CHECK-O0: stp d19, d18
; CHECK-O0: stp d17, d16
; CHECK-O0: stp d7, d6
; CHECK-O0: stp d5, d4
; CHECK-O0: stp d3, d2
; CHECK-O0: stp d1, d0
; CHECK-O0: str x14
; CHECK-O0: stp x13, x12
; CHECK-O0: stp x11, x10
; CHECK-O0: stp x8, x7
; CHECK-O0: stp x6, x5
; CHECK-O0: stp x4, x3
; CHECK-O0: stp x2, x1
; CHECK-O0: blr
; CHECK-O0: tbnz w{{.*}}, #0, [[BB_end:.?LBB0_[0-9]+]]
; CHECK-O0: blr
; CHECK-O0: tlv_atexit
; CHECK-O0: [[BB_end]]:
; CHECK-O0: blr
; CHECK-O0: ldp x2, x1
; CHECK-O0: ldp x4, x3
; CHECK-O0: ldp x6, x5
; CHECK-O0: ldp x8, x7
; CHECK-O0: ldp x11, x10
; CHECK-O0: ldp x13, x12
; CHECK-O0: ldr x14
; CHECK-O0: ldp d1, d0
; CHECK-O0: ldp d3, d2
; CHECK-O0: ldp d5, d4
; CHECK-O0: ldp d7, d6
; CHECK-O0: ldp d17, d16
; CHECK-O0: ldp d19, d18
; CHECK-O0: ldp d21, d20
; CHECK-O0: ldp d23, d22
; CHECK-O0: ldp d25, d24
; CHECK-O0: ldp d27, d26
; CHECK-O0: ldp d29, d28
; CHECK-O0: ldp d31, d30

; CHECK-LABEL: _ZTW4sum1
; CHECK-NOT: stp d31, d30
; CHECK-NOT: stp d29, d28
; CHECK-NOT: stp d27, d26
; CHECK-NOT: stp d25, d24
; CHECK-NOT: stp d23, d22
; CHECK-NOT: stp d21, d20
; CHECK-NOT: stp d19, d18
; CHECK-NOT: stp d17, d16
; CHECK-NOT: stp d7, d6
; CHECK-NOT: stp d5, d4
; CHECK-NOT: stp d3, d2
; CHECK-NOT: stp d1, d0
; CHECK-NOT: stp x20, x19
; CHECK-NOT: stp x14, x13
; CHECK-NOT: stp x12, x11
; CHECK-NOT: stp x10, x9
; CHECK-NOT: stp x8, x7
; CHECK-NOT: stp x6, x5
; CHECK-NOT: stp x4, x3
; CHECK-NOT: stp x2, x1
; CHECK: blr

; CHECK-O0-LABEL: _ZTW4sum1
; CHECK-O0-NOT: vstr
; CHECK-O0-NOT: vldr
define cxx_fast_tlscc nonnull ptr @_ZTW4sum1() nounwind {
  ret ptr @sum1
}

; Make sure at O0, we don't generate spilling/reloading of the CSRs.
; CHECK-O0-LABEL: tls_test2
; CHECK-O0-NOT: stp d31, d30
; CHECK-O0-NOT: stp d29, d28
; CHECK-O0-NOT: stp d27, d26
; CHECK-O0-NOT: stp d25, d24
; CHECK-O0-NOT: stp d23, d22
; CHECK-O0-NOT: stp d21, d20
; CHECK-O0-NOT: stp d19, d18
; CHECK-O0-NOT: stp d17, d16
; CHECK-O0-NOT: stp d7, d6
; CHECK-O0-NOT: stp d5, d4
; CHECK-O0-NOT: stp d3, d2
; CHECK-O0-NOT: stp d1, d0
; CHECK-O0-NOT: stp x20, x19
; CHECK-O0-NOT: stp x14, x13
; CHECK-O0-NOT: stp x12, x11
; CHECK-O0-NOT: stp x10, x9
; CHECK-O0-NOT: stp x8, x7
; CHECK-O0-NOT: stp x6, x5
; CHECK-O0-NOT: stp x4, x3
; CHECK-O0-NOT: stp x2, x1
; CHECK-O0: bl {{.*}}tls_helper
; CHECK-O0-NOT: ldp x2, x1
; CHECK-O0-NOT: ldp x4, x3
; CHECK-O0-NOT: ldp x6, x5
; CHECK-O0-NOT: ldp x8, x7
; CHECK-O0-NOT: ldp x10, x9
; CHECK-O0-NOT: ldp x12, x11
; CHECK-O0-NOT: ldp x14, x13
; CHECK-O0-NOT: ldp x20, x19
; CHECK-O0-NOT: ldp d1, d0
; CHECK-O0-NOT: ldp d3, d2
; CHECK-O0-NOT: ldp d5, d4
; CHECK-O0-NOT: ldp d7, d6
; CHECK-O0-NOT: ldp d17, d16
; CHECK-O0-NOT: ldp d19, d18
; CHECK-O0-NOT: ldp d21, d20
; CHECK-O0-NOT: ldp d23, d22
; CHECK-O0-NOT: ldp d25, d24
; CHECK-O0-NOT: ldp d27, d26
; CHECK-O0-NOT: ldp d29, d28
; CHECK-O0-NOT: ldp d31, d30
; CHECK-O0: ret
%class.C = type { i32 }
@tC = internal thread_local global %class.C zeroinitializer, align 4
declare cxx_fast_tlscc void @tls_helper()
define cxx_fast_tlscc ptr @tls_test2() #1 {
  call cxx_fast_tlscc void @tls_helper()
  ret ptr @tC
}

; Make sure we do not allow tail call when caller and callee have different
; calling conventions.
declare ptr @_ZN1CD1Ev(ptr readnone returned %this)
; CHECK-LABEL: tls_test
; CHECK: bl __tlv_atexit
define cxx_fast_tlscc void @__tls_test() {
entry:
  store i32 0, ptr @tC, align 4
  %0 = tail call i32 @_tlv_atexit(ptr @_ZN1CD1Ev, ptr @tC, ptr nonnull @__dso_handle) #1
  ret void
}

define cxx_fast_tlscc void @weird_prologue_regs(i32 %n) #1 {
; CHECK-LABEL: weird_prologue_regs:
; CHECK-NOT: str x9
; CHECK-NOT: stp{{.*}}x9{{.*}}[
; CHECK-NOT: str x19
; CHECK-NOT: stp{{.*}}x19{{.*}}[

; CHECK: sub x9, sp, #
; CHECK: and sp, x9, #0x
; CHECK: mov x19, sp

; CHECK-NOT: str x9
; CHECK-NOT: stp{{.*}}x9{{.*}}[
; CHECK-NOT: str x19
; CHECK-NOT: stp{{.*}}x19{{.*}}[

  %p0 = alloca i32, i32 200
  %p1 = alloca i32, align 32
  %p2 = alloca i32, i32 %n
  call void @callee(ptr %p0)
  call void @callee(ptr %p1)
  call void @callee(ptr %p2)
  ret void
}
declare void @callee(ptr)

attributes #0 = { nounwind "frame-pointer"="all" }
attributes #1 = { nounwind }
