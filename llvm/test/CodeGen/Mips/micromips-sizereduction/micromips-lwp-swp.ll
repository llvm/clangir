; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=mipsel-unknown-linux-gnu -mattr=+micromips -mcpu=mips32r2 \
; RUN: -verify-machineinstrs < %s | FileCheck %s

; Function Attrs: nounwind
define i32 @fun(ptr %adr, i32 %val) {
; CHECK-LABEL: fun:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addiusp -32
; CHECK-NEXT:    .cfi_def_cfa_offset 32
; CHECK-NEXT:    sw $ra, 28($sp) # 4-byte Folded Spill
; CHECK-NEXT:    swp $16, 20($sp)
; CHECK-NEXT:    .cfi_offset 31, -4
; CHECK-NEXT:    .cfi_offset 17, -8
; CHECK-NEXT:    .cfi_offset 16, -12
; CHECK-NEXT:    move $17, $5
; CHECK-NEXT:    move $16, $4
; CHECK-NEXT:    jal fun1
; CHECK-NEXT:    nop
; CHECK-NEXT:    sw16 $17, 0($16)
; CHECK-NEXT:    li16 $2, 0
; CHECK-NEXT:    lwp $16, 20($sp)
; CHECK-NEXT:    lw $ra, 28($sp) # 4-byte Folded Reload
; CHECK-NEXT:    addiusp 32
; CHECK-NEXT:    jrc $ra
entry:
  %call1 =  call ptr @fun1()
  store i32 %val, ptr %adr, align 4
  ret i32 0
}

declare ptr @fun1()

