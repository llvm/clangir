; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s
; RUN: llc < %s -mtriple=arm-linux-gnueabi -relocation-model=pic | \
; RUN:   FileCheck %s --check-prefix=PIC


; CHECK: i(TPOFF)
; CHECK: __aeabi_read_tp

; PIC: __tls_get_addr

@i = dso_local thread_local global i32 15		; <ptr> [#uses=2]

define dso_local i32 @f() {
entry:
	%tmp1 = load i32, ptr @i		; <i32> [#uses=1]
	ret i32 %tmp1
}

define dso_local ptr @g() {
entry:
	ret ptr @i
}
