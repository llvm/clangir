; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc  -O0 -mtriple=mipsel-linux-gnu -global-isel  -verify-machineinstrs %s -o -| FileCheck %s -check-prefixes=MIPS32

@.str = private unnamed_addr constant [11 x i8] c"string %s\0A\00", align 1
declare void @llvm.va_start(ptr)
declare void @llvm.va_copy(ptr, ptr)
declare i32 @printf(ptr, ...)

define void @testVaCopyArg(ptr %fmt, ...) {
; MIPS32-LABEL: testVaCopyArg:
; MIPS32:       # %bb.0: # %entry
; MIPS32-NEXT:    addiu $sp, $sp, -40
; MIPS32-NEXT:    .cfi_def_cfa_offset 40
; MIPS32-NEXT:    sw $ra, 36($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    .cfi_offset 31, -4
; MIPS32-NEXT:    move $3, $4
; MIPS32-NEXT:    addiu $1, $sp, 44
; MIPS32-NEXT:    sw $5, 0($1)
; MIPS32-NEXT:    addiu $1, $sp, 48
; MIPS32-NEXT:    sw $6, 0($1)
; MIPS32-NEXT:    addiu $1, $sp, 52
; MIPS32-NEXT:    sw $7, 0($1)
; MIPS32-NEXT:    lui $1, %hi($.str)
; MIPS32-NEXT:    addiu $4, $1, %lo($.str)
; MIPS32-NEXT:    addiu $6, $sp, 32
; MIPS32-NEXT:    addiu $2, $sp, 28
; MIPS32-NEXT:    addiu $5, $sp, 24
; MIPS32-NEXT:    addiu $1, $sp, 20
; MIPS32-NEXT:    sw $3, 0($6)
; MIPS32-NEXT:    addiu $3, $sp, 44
; MIPS32-NEXT:    sw $3, 0($2)
; MIPS32-NEXT:    lw $2, 0($2)
; MIPS32-NEXT:    sw $2, 0($5)
; MIPS32-NEXT:    lw $2, 0($5)
; MIPS32-NEXT:    ori $3, $zero, 4
; MIPS32-NEXT:    addu $3, $2, $3
; MIPS32-NEXT:    sw $3, 0($5)
; MIPS32-NEXT:    lw $2, 0($2)
; MIPS32-NEXT:    sw $2, 0($1)
; MIPS32-NEXT:    lw $5, 0($1)
; MIPS32-NEXT:    jal printf
; MIPS32-NEXT:    nop
; MIPS32-NEXT:    lw $ra, 36($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    addiu $sp, $sp, 40
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    nop
entry:
  %fmt.addr = alloca ptr, align 4
  %ap = alloca ptr, align 4
  %aq = alloca ptr, align 4
  %s = alloca ptr, align 4
  store ptr %fmt, ptr %fmt.addr, align 4
  call void @llvm.va_start(ptr %ap)
  call void @llvm.va_copy(ptr %aq, ptr %ap)
  %argp.cur = load ptr, ptr %aq, align 4
  %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
  store ptr %argp.next, ptr %aq, align 4
  %0 = load ptr, ptr %argp.cur, align 4
  store ptr %0, ptr %s, align 4
  %1 = load ptr, ptr %s, align 4
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %1)
  ret void
}
