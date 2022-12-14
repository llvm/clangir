; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc  -O0 -mtriple=mipsel-linux-gnu -global-isel  -verify-machineinstrs %s -o -| FileCheck %s -check-prefixes=MIPS32

define ptr @inttoptr(i32 %a) {
; MIPS32-LABEL: inttoptr:
; MIPS32:       # %bb.0: # %entry
; MIPS32-NEXT:    move $2, $4
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    nop
entry:
  %0 = inttoptr i32 %a to ptr
  ret ptr %0
}

define i32 @ptrtoint(ptr %a) {
; MIPS32-LABEL: ptrtoint:
; MIPS32:       # %bb.0: # %entry
; MIPS32-NEXT:    move $2, $4
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    nop
entry:
  %0 = ptrtoint ptr %a to i32
  ret i32 %0
}
