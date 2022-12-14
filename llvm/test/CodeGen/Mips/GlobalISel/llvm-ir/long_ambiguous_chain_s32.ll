; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc  -O0 -mtriple=mipsel-linux-gnu -global-isel  -verify-machineinstrs %s -o -| FileCheck %s -check-prefixes=MIPS32

define void @long_chain_ambiguous_i32_in_gpr(i1 %cnd0, i1 %cnd1, i1 %cnd2, ptr %a, ptr %b, ptr %c, ptr %result) {
; MIPS32-LABEL: long_chain_ambiguous_i32_in_gpr:
; MIPS32:       # %bb.0: # %entry
; MIPS32-NEXT:    addiu $sp, $sp, -48
; MIPS32-NEXT:    .cfi_def_cfa_offset 48
; MIPS32-NEXT:    sw $4, 20($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    sw $5, 24($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    sw $6, 28($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    sw $7, 32($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    addiu $1, $sp, 64
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 36($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    addiu $1, $sp, 68
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 40($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    addiu $1, $sp, 72
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 44($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    andi $1, $4, 1
; MIPS32-NEXT:    bnez $1, $BB0_12
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.1: # %entry
; MIPS32-NEXT:    j $BB0_2
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB0_2: # %pre.PHI.1
; MIPS32-NEXT:    lw $1, 24($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    bnez $1, $BB0_7
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.3: # %pre.PHI.1
; MIPS32-NEXT:    j $BB0_4
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB0_4: # %pre.PHI.1.0
; MIPS32-NEXT:    lw $1, 28($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    bnez $1, $BB0_8
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.5: # %pre.PHI.1.0
; MIPS32-NEXT:    j $BB0_6
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB0_6: # %b.PHI.1.0
; MIPS32-NEXT:    lw $1, 32($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 16($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB0_9
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB0_7: # %b.PHI.1.1
; MIPS32-NEXT:    lw $1, 36($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 16($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB0_9
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB0_8: # %b.PHI.1.2
; MIPS32-NEXT:    lw $1, 40($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 16($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB0_9
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB0_9: # %b.PHI.1
; MIPS32-NEXT:    lw $1, 28($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $2, 16($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    sw $2, 8($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    sw $2, 12($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    bnez $1, $BB0_11
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.10: # %b.PHI.1
; MIPS32-NEXT:    j $BB0_19
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB0_11: # %b.PHI.1.end
; MIPS32-NEXT:    lw $1, 8($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $2, 44($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    sw $1, 0($2)
; MIPS32-NEXT:    addiu $sp, $sp, 48
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB0_12: # %pre.PHI.2
; MIPS32-NEXT:    lw $1, 20($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    bnez $1, $BB0_14
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.13: # %pre.PHI.2
; MIPS32-NEXT:    j $BB0_15
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB0_14: # %b.PHI.2.0
; MIPS32-NEXT:    lw $1, 32($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 4($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB0_16
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB0_15: # %b.PHI.2.1
; MIPS32-NEXT:    lw $1, 36($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 4($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB0_16
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB0_16: # %b.PHI.2
; MIPS32-NEXT:    lw $1, 24($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $2, 4($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    sw $2, 0($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    sw $2, 12($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    bnez $1, $BB0_19
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.17: # %b.PHI.2
; MIPS32-NEXT:    j $BB0_18
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB0_18: # %b.PHI.2.end
; MIPS32-NEXT:    lw $1, 0($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $2, 44($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    sw $1, 0($2)
; MIPS32-NEXT:    addiu $sp, $sp, 48
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB0_19: # %b.PHI.3
; MIPS32-NEXT:    lw $2, 44($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $3, 24($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $5, 28($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 12($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    move $4, $1
; MIPS32-NEXT:    andi $5, $5, 1
; MIPS32-NEXT:    movn $4, $1, $5
; MIPS32-NEXT:    andi $5, $3, 1
; MIPS32-NEXT:    move $3, $1
; MIPS32-NEXT:    movn $3, $4, $5
; MIPS32-NEXT:    sw $3, 0($2)
; MIPS32-NEXT:    sw $1, 0($2)
; MIPS32-NEXT:    addiu $sp, $sp, 48
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    nop
entry:
  br i1 %cnd0, label %pre.PHI.2, label %pre.PHI.1

pre.PHI.1:
  br i1 %cnd1, label %b.PHI.1.1, label %pre.PHI.1.0

pre.PHI.1.0:
  br i1 %cnd2, label %b.PHI.1.2, label %b.PHI.1.0

b.PHI.1.0:
  %phi1.0 = load i32, ptr %a
  br label %b.PHI.1

b.PHI.1.1:
  %phi1.1 = load i32, ptr %b
  br label %b.PHI.1

b.PHI.1.2:
  %phi1.2 = load i32, ptr %c
  br label %b.PHI.1

b.PHI.1:
  %phi1 = phi i32 [ %phi1.0, %b.PHI.1.0 ], [ %phi1.1, %b.PHI.1.1 ], [ %phi1.2, %b.PHI.1.2 ]
  br i1 %cnd2, label %b.PHI.1.end, label %b.PHI.3

b.PHI.1.end:
  store i32 %phi1, ptr %result
  ret void

pre.PHI.2:
  br i1 %cnd0, label %b.PHI.2.0, label %b.PHI.2.1

b.PHI.2.0:
  %phi2.0 = load i32, ptr %a
  br label %b.PHI.2

b.PHI.2.1:
  %phi2.1 = load i32, ptr %b
  br label %b.PHI.2

b.PHI.2:
  %phi2 = phi i32 [ %phi2.0, %b.PHI.2.0 ], [ %phi2.1, %b.PHI.2.1 ]
   br i1 %cnd1, label %b.PHI.3, label %b.PHI.2.end

b.PHI.2.end:
  store i32 %phi2, ptr %result
  ret void

b.PHI.3:
  %phi3 = phi i32 [ %phi2, %b.PHI.2], [ %phi1, %b.PHI.1 ]
  %phi4 = phi i32 [ %phi2, %b.PHI.2], [ %phi1, %b.PHI.1 ]
  %sel_1.2 = select i1 %cnd2, i32 %phi3, i32 %phi4
  %sel_3_1.2 = select i1 %cnd1, i32 %sel_1.2, i32 %phi3
  store i32 %sel_3_1.2, ptr %result
  store i32 %phi3, ptr %result
  ret void

}

define void @long_chain_i32_in_gpr(i1 %cnd0, i1 %cnd1, i1 %cnd2, ptr %a, ptr %b, ptr %c, ptr %result) {
; MIPS32-LABEL: long_chain_i32_in_gpr:
; MIPS32:       # %bb.0: # %entry
; MIPS32-NEXT:    addiu $sp, $sp, -56
; MIPS32-NEXT:    .cfi_def_cfa_offset 56
; MIPS32-NEXT:    sw $4, 24($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    sw $5, 28($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    sw $6, 32($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    sw $7, 36($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    addiu $1, $sp, 72
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 40($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    addiu $1, $sp, 76
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 44($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    addiu $1, $sp, 80
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 48($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    ori $1, $zero, 0
; MIPS32-NEXT:    sw $1, 52($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    andi $1, $4, 1
; MIPS32-NEXT:    bnez $1, $BB1_12
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.1: # %entry
; MIPS32-NEXT:    j $BB1_2
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB1_2: # %pre.PHI.1
; MIPS32-NEXT:    lw $1, 28($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    bnez $1, $BB1_7
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.3: # %pre.PHI.1
; MIPS32-NEXT:    j $BB1_4
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB1_4: # %pre.PHI.1.0
; MIPS32-NEXT:    lw $1, 32($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    bnez $1, $BB1_8
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.5: # %pre.PHI.1.0
; MIPS32-NEXT:    j $BB1_6
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB1_6: # %b.PHI.1.0
; MIPS32-NEXT:    lw $1, 36($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 20($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB1_9
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB1_7: # %b.PHI.1.1
; MIPS32-NEXT:    lw $1, 40($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 20($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB1_9
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB1_8: # %b.PHI.1.2
; MIPS32-NEXT:    lw $1, 44($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 20($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB1_9
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB1_9: # %b.PHI.1
; MIPS32-NEXT:    lw $2, 52($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 32($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $3, 20($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    sw $3, 8($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    sw $3, 12($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    sw $2, 16($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    bnez $1, $BB1_11
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.10: # %b.PHI.1
; MIPS32-NEXT:    j $BB1_19
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB1_11: # %b.PHI.1.end
; MIPS32-NEXT:    lw $1, 8($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $2, 48($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    sw $1, 0($2)
; MIPS32-NEXT:    addiu $sp, $sp, 56
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB1_12: # %pre.PHI.2
; MIPS32-NEXT:    lw $1, 24($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    bnez $1, $BB1_14
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.13: # %pre.PHI.2
; MIPS32-NEXT:    j $BB1_15
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB1_14: # %b.PHI.2.0
; MIPS32-NEXT:    lw $1, 36($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 4($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB1_16
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB1_15: # %b.PHI.2.1
; MIPS32-NEXT:    lw $1, 40($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 4($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB1_16
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB1_16: # %b.PHI.2
; MIPS32-NEXT:    lw $1, 28($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $2, 4($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    sw $2, 0($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    move $3, $2
; MIPS32-NEXT:    sw $3, 12($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    sw $2, 16($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    bnez $1, $BB1_19
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.17: # %b.PHI.2
; MIPS32-NEXT:    j $BB1_18
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB1_18: # %b.PHI.2.end
; MIPS32-NEXT:    lw $1, 0($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $2, 48($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    sw $1, 0($2)
; MIPS32-NEXT:    addiu $sp, $sp, 56
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB1_19: # %b.PHI.3
; MIPS32-NEXT:    lw $2, 48($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $3, 28($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $5, 32($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 12($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $4, 16($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    andi $5, $5, 1
; MIPS32-NEXT:    movn $4, $1, $5
; MIPS32-NEXT:    andi $5, $3, 1
; MIPS32-NEXT:    move $3, $1
; MIPS32-NEXT:    movn $3, $4, $5
; MIPS32-NEXT:    sw $3, 0($2)
; MIPS32-NEXT:    sw $1, 0($2)
; MIPS32-NEXT:    addiu $sp, $sp, 56
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    nop
entry:
  br i1 %cnd0, label %pre.PHI.2, label %pre.PHI.1

pre.PHI.1:
  br i1 %cnd1, label %b.PHI.1.1, label %pre.PHI.1.0

pre.PHI.1.0:
  br i1 %cnd2, label %b.PHI.1.2, label %b.PHI.1.0

b.PHI.1.0:
  %phi1.0 = load i32, ptr %a
  br label %b.PHI.1

b.PHI.1.1:
  %phi1.1 = load i32, ptr %b
  br label %b.PHI.1

b.PHI.1.2:
  %phi1.2 = load i32, ptr %c
  br label %b.PHI.1

b.PHI.1:
  %phi1 = phi i32 [ %phi1.0, %b.PHI.1.0 ], [ %phi1.1, %b.PHI.1.1 ], [ %phi1.2, %b.PHI.1.2 ]
  br i1 %cnd2, label %b.PHI.1.end, label %b.PHI.3

b.PHI.1.end:
  store i32 %phi1, ptr %result
  ret void

pre.PHI.2:
  br i1 %cnd0, label %b.PHI.2.0, label %b.PHI.2.1

b.PHI.2.0:
  %phi2.0 = load i32, ptr %a
  br label %b.PHI.2

b.PHI.2.1:
  %phi2.1 = load i32, ptr %b
  br label %b.PHI.2

b.PHI.2:
  %phi2 = phi i32 [ %phi2.0, %b.PHI.2.0 ], [ %phi2.1, %b.PHI.2.1 ]
   br i1 %cnd1, label %b.PHI.3, label %b.PHI.2.end

b.PHI.2.end:
  store i32 %phi2, ptr %result
  ret void

b.PHI.3:
  %phi3 = phi i32 [ %phi2, %b.PHI.2], [ %phi1, %b.PHI.1 ]
  %phi4 = phi i32 [ %phi2, %b.PHI.2], [ 0, %b.PHI.1 ]
  %sel_1.2 = select i1 %cnd2, i32 %phi3, i32 %phi4
  %sel_3_1.2 = select i1 %cnd1, i32 %sel_1.2, i32 %phi3
  store i32 %sel_3_1.2, ptr %result
  store i32 %phi3, ptr %result
  ret void
}

define void @long_chain_ambiguous_float_in_fpr(i1 %cnd0, i1 %cnd1, i1 %cnd2, ptr %a, ptr %b, ptr %c, ptr %result) {
; MIPS32-LABEL: long_chain_ambiguous_float_in_fpr:
; MIPS32:       # %bb.0: # %entry
; MIPS32-NEXT:    addiu $sp, $sp, -48
; MIPS32-NEXT:    .cfi_def_cfa_offset 48
; MIPS32-NEXT:    sw $4, 20($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    sw $5, 24($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    sw $6, 28($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    sw $7, 32($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    addiu $1, $sp, 64
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 36($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    addiu $1, $sp, 68
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 40($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    addiu $1, $sp, 72
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 44($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    andi $1, $4, 1
; MIPS32-NEXT:    bnez $1, $BB2_12
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.1: # %entry
; MIPS32-NEXT:    j $BB2_2
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB2_2: # %pre.PHI.1
; MIPS32-NEXT:    lw $1, 24($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    bnez $1, $BB2_7
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.3: # %pre.PHI.1
; MIPS32-NEXT:    j $BB2_4
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB2_4: # %pre.PHI.1.0
; MIPS32-NEXT:    lw $1, 28($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    bnez $1, $BB2_8
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.5: # %pre.PHI.1.0
; MIPS32-NEXT:    j $BB2_6
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB2_6: # %b.PHI.1.0
; MIPS32-NEXT:    lw $1, 32($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 16($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB2_9
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB2_7: # %b.PHI.1.1
; MIPS32-NEXT:    lw $1, 36($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 16($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB2_9
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB2_8: # %b.PHI.1.2
; MIPS32-NEXT:    lw $1, 40($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 16($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB2_9
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB2_9: # %b.PHI.1
; MIPS32-NEXT:    lw $1, 28($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $2, 16($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    sw $2, 8($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    sw $2, 12($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    bnez $1, $BB2_11
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.10: # %b.PHI.1
; MIPS32-NEXT:    j $BB2_19
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB2_11: # %b.PHI.1.end
; MIPS32-NEXT:    lw $1, 8($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $2, 44($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    sw $1, 0($2)
; MIPS32-NEXT:    addiu $sp, $sp, 48
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB2_12: # %pre.PHI.2
; MIPS32-NEXT:    lw $1, 20($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    bnez $1, $BB2_14
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.13: # %pre.PHI.2
; MIPS32-NEXT:    j $BB2_15
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB2_14: # %b.PHI.2.0
; MIPS32-NEXT:    lw $1, 32($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 4($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB2_16
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB2_15: # %b.PHI.2.1
; MIPS32-NEXT:    lw $1, 36($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 4($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB2_16
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB2_16: # %b.PHI.2
; MIPS32-NEXT:    lw $1, 24($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $2, 4($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    sw $2, 0($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    sw $2, 12($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    bnez $1, $BB2_19
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.17: # %b.PHI.2
; MIPS32-NEXT:    j $BB2_18
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB2_18: # %b.PHI.2.end
; MIPS32-NEXT:    lw $1, 0($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $2, 44($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    sw $1, 0($2)
; MIPS32-NEXT:    addiu $sp, $sp, 48
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB2_19: # %b.PHI.3
; MIPS32-NEXT:    lw $2, 44($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $3, 24($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $5, 28($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 12($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    move $4, $1
; MIPS32-NEXT:    andi $5, $5, 1
; MIPS32-NEXT:    movn $4, $1, $5
; MIPS32-NEXT:    andi $5, $3, 1
; MIPS32-NEXT:    move $3, $1
; MIPS32-NEXT:    movn $3, $4, $5
; MIPS32-NEXT:    sw $3, 0($2)
; MIPS32-NEXT:    sw $1, 0($2)
; MIPS32-NEXT:    addiu $sp, $sp, 48
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    nop
entry:
  br i1 %cnd0, label %pre.PHI.2, label %pre.PHI.1

pre.PHI.1:
  br i1 %cnd1, label %b.PHI.1.1, label %pre.PHI.1.0

pre.PHI.1.0:
  br i1 %cnd2, label %b.PHI.1.2, label %b.PHI.1.0

b.PHI.1.0:
  %phi1.0 = load float, ptr %a
  br label %b.PHI.1

b.PHI.1.1:
  %phi1.1 = load float, ptr %b
  br label %b.PHI.1

b.PHI.1.2:
  %phi1.2 = load float, ptr %c
  br label %b.PHI.1

b.PHI.1:
  %phi1 = phi float [ %phi1.0, %b.PHI.1.0 ], [ %phi1.1, %b.PHI.1.1 ], [ %phi1.2, %b.PHI.1.2 ]
  br i1 %cnd2, label %b.PHI.1.end, label %b.PHI.3

b.PHI.1.end:
  store float %phi1, ptr %result
  ret void

pre.PHI.2:
  br i1 %cnd0, label %b.PHI.2.0, label %b.PHI.2.1

b.PHI.2.0:
  %phi2.0 = load float, ptr %a
  br label %b.PHI.2

b.PHI.2.1:
  %phi2.1 = load float, ptr %b
  br label %b.PHI.2

b.PHI.2:
  %phi2 = phi float [ %phi2.0, %b.PHI.2.0 ], [ %phi2.1, %b.PHI.2.1 ]
   br i1 %cnd1, label %b.PHI.3, label %b.PHI.2.end

b.PHI.2.end:
  store float %phi2, ptr %result
  ret void

b.PHI.3:
  %phi3 = phi float [ %phi2, %b.PHI.2], [ %phi1, %b.PHI.1 ]
  %phi4 = phi float [ %phi2, %b.PHI.2], [ %phi1, %b.PHI.1 ]
  %sel_1.2 = select i1 %cnd2, float %phi3, float %phi4
  %sel_3_1.2 = select i1 %cnd1, float %sel_1.2, float %phi3
  store float %sel_3_1.2, ptr %result
  store float %phi3, ptr %result
  ret void
}


define void @long_chain_float_in_fpr(i1 %cnd0, i1 %cnd1, i1 %cnd2, ptr %a, ptr %b, ptr %c, ptr %result) {
; MIPS32-LABEL: long_chain_float_in_fpr:
; MIPS32:       # %bb.0: # %entry
; MIPS32-NEXT:    addiu $sp, $sp, -56
; MIPS32-NEXT:    .cfi_def_cfa_offset 56
; MIPS32-NEXT:    sw $4, 24($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    sw $5, 28($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    sw $6, 32($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    sw $7, 36($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    addiu $1, $sp, 72
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 40($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    addiu $1, $sp, 76
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 44($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    addiu $1, $sp, 80
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    sw $1, 48($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    ori $1, $zero, 0
; MIPS32-NEXT:    mtc1 $1, $f0
; MIPS32-NEXT:    swc1 $f0, 52($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    andi $1, $4, 1
; MIPS32-NEXT:    bnez $1, $BB3_12
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.1: # %entry
; MIPS32-NEXT:    j $BB3_2
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB3_2: # %pre.PHI.1
; MIPS32-NEXT:    lw $1, 28($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    bnez $1, $BB3_7
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.3: # %pre.PHI.1
; MIPS32-NEXT:    j $BB3_4
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB3_4: # %pre.PHI.1.0
; MIPS32-NEXT:    lw $1, 32($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    bnez $1, $BB3_8
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.5: # %pre.PHI.1.0
; MIPS32-NEXT:    j $BB3_6
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB3_6: # %b.PHI.1.0
; MIPS32-NEXT:    lw $1, 36($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lwc1 $f0, 0($1)
; MIPS32-NEXT:    swc1 $f0, 20($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB3_9
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB3_7: # %b.PHI.1.1
; MIPS32-NEXT:    lw $1, 40($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lwc1 $f0, 0($1)
; MIPS32-NEXT:    swc1 $f0, 20($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB3_9
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB3_8: # %b.PHI.1.2
; MIPS32-NEXT:    lw $1, 44($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lwc1 $f0, 0($1)
; MIPS32-NEXT:    swc1 $f0, 20($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB3_9
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB3_9: # %b.PHI.1
; MIPS32-NEXT:    lwc1 $f0, 52($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 32($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lwc1 $f1, 20($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    swc1 $f1, 8($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    swc1 $f1, 12($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    swc1 $f0, 16($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    bnez $1, $BB3_11
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.10: # %b.PHI.1
; MIPS32-NEXT:    j $BB3_19
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB3_11: # %b.PHI.1.end
; MIPS32-NEXT:    lwc1 $f0, 8($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 48($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    swc1 $f0, 0($1)
; MIPS32-NEXT:    addiu $sp, $sp, 56
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB3_12: # %pre.PHI.2
; MIPS32-NEXT:    lw $1, 24($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    bnez $1, $BB3_14
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.13: # %pre.PHI.2
; MIPS32-NEXT:    j $BB3_15
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB3_14: # %b.PHI.2.0
; MIPS32-NEXT:    lw $1, 36($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lwc1 $f0, 0($1)
; MIPS32-NEXT:    swc1 $f0, 4($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB3_16
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB3_15: # %b.PHI.2.1
; MIPS32-NEXT:    lw $1, 40($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lwc1 $f0, 0($1)
; MIPS32-NEXT:    swc1 $f0, 4($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    j $BB3_16
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB3_16: # %b.PHI.2
; MIPS32-NEXT:    lw $1, 28($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lwc1 $f0, 4($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    swc1 $f0, 0($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    andi $1, $1, 1
; MIPS32-NEXT:    mov.s $f1, $f0
; MIPS32-NEXT:    swc1 $f1, 12($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    swc1 $f0, 16($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    bnez $1, $BB3_19
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  # %bb.17: # %b.PHI.2
; MIPS32-NEXT:    j $BB3_18
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB3_18: # %b.PHI.2.end
; MIPS32-NEXT:    lwc1 $f0, 0($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $1, 48($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    swc1 $f0, 0($1)
; MIPS32-NEXT:    addiu $sp, $sp, 56
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    nop
; MIPS32-NEXT:  $BB3_19: # %b.PHI.3
; MIPS32-NEXT:    lw $1, 48($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $2, 28($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $3, 32($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lwc1 $f0, 12($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lwc1 $f2, 16($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    andi $3, $3, 1
; MIPS32-NEXT:    movn.s $f2, $f0, $3
; MIPS32-NEXT:    andi $2, $2, 1
; MIPS32-NEXT:    mov.s $f1, $f0
; MIPS32-NEXT:    movn.s $f1, $f2, $2
; MIPS32-NEXT:    swc1 $f1, 0($1)
; MIPS32-NEXT:    swc1 $f0, 0($1)
; MIPS32-NEXT:    addiu $sp, $sp, 56
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    nop
entry:
  br i1 %cnd0, label %pre.PHI.2, label %pre.PHI.1

pre.PHI.1:
  br i1 %cnd1, label %b.PHI.1.1, label %pre.PHI.1.0

pre.PHI.1.0:
  br i1 %cnd2, label %b.PHI.1.2, label %b.PHI.1.0

b.PHI.1.0:
  %phi1.0 = load float, ptr %a
  br label %b.PHI.1

b.PHI.1.1:
  %phi1.1 = load float, ptr %b
  br label %b.PHI.1

b.PHI.1.2:
  %phi1.2 = load float, ptr %c
  br label %b.PHI.1

b.PHI.1:
  %phi1 = phi float [ %phi1.0, %b.PHI.1.0 ], [ %phi1.1, %b.PHI.1.1 ], [ %phi1.2, %b.PHI.1.2 ]
  br i1 %cnd2, label %b.PHI.1.end, label %b.PHI.3

b.PHI.1.end:
  store float %phi1, ptr %result
  ret void

pre.PHI.2:
  br i1 %cnd0, label %b.PHI.2.0, label %b.PHI.2.1

b.PHI.2.0:
  %phi2.0 = load float, ptr %a
  br label %b.PHI.2

b.PHI.2.1:
  %phi2.1 = load float, ptr %b
  br label %b.PHI.2

b.PHI.2:
  %phi2 = phi float [ %phi2.0, %b.PHI.2.0 ], [ %phi2.1, %b.PHI.2.1 ]
   br i1 %cnd1, label %b.PHI.3, label %b.PHI.2.end

b.PHI.2.end:
  store float %phi2, ptr %result
  ret void

b.PHI.3:
  %phi3 = phi float [ %phi2, %b.PHI.2], [ %phi1, %b.PHI.1 ]
  %phi4 = phi float [ %phi2, %b.PHI.2], [ 0.0, %b.PHI.1 ]
  %sel_1.2 = select i1 %cnd2, float %phi3, float %phi4
  %sel_3_1.2 = select i1 %cnd1, float %sel_1.2, float %phi3
  store float %sel_3_1.2, ptr %result
  store float %phi3, ptr %result
  ret void
}

