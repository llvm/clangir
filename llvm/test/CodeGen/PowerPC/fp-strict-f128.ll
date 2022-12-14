; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -verify-machineinstrs -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   < %s -mtriple=powerpc64le-unknown-linux -mcpu=pwr9 | FileCheck %s
; RUN: llc -verify-machineinstrs -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   < %s -mtriple=powerpc64le-unknown-linux -mcpu=pwr8 | FileCheck %s --check-prefix=CHECK-P8

declare fp128 @llvm.experimental.constrained.fadd.f128(fp128, fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.fsub.f128(fp128, fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.fmul.f128(fp128, fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.fdiv.f128(fp128, fp128, metadata, metadata)

declare fp128 @llvm.experimental.constrained.fma.f128(fp128, fp128, fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.sqrt.f128(fp128, metadata, metadata)

define fp128 @fadd_f128(fp128 %f1, fp128 %f2) #0 {
; CHECK-LABEL: fadd_f128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xsaddqp v2, v2, v3
; CHECK-NEXT:    blr
;
; CHECK-P8-LABEL: fadd_f128:
; CHECK-P8:       # %bb.0:
; CHECK-P8-NEXT:    mflr r0
; CHECK-P8-NEXT:    stdu r1, -32(r1)
; CHECK-P8-NEXT:    std r0, 48(r1)
; CHECK-P8-NEXT:    .cfi_def_cfa_offset 32
; CHECK-P8-NEXT:    .cfi_offset lr, 16
; CHECK-P8-NEXT:    bl __addkf3
; CHECK-P8-NEXT:    nop
; CHECK-P8-NEXT:    addi r1, r1, 32
; CHECK-P8-NEXT:    ld r0, 16(r1)
; CHECK-P8-NEXT:    mtlr r0
; CHECK-P8-NEXT:    blr
  %res = call fp128 @llvm.experimental.constrained.fadd.f128(
                        fp128 %f1, fp128 %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret fp128 %res
}

define fp128 @fsub_f128(fp128 %f1, fp128 %f2) #0 {
; CHECK-LABEL: fsub_f128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xssubqp v2, v2, v3
; CHECK-NEXT:    blr
;
; CHECK-P8-LABEL: fsub_f128:
; CHECK-P8:       # %bb.0:
; CHECK-P8-NEXT:    mflr r0
; CHECK-P8-NEXT:    stdu r1, -32(r1)
; CHECK-P8-NEXT:    std r0, 48(r1)
; CHECK-P8-NEXT:    .cfi_def_cfa_offset 32
; CHECK-P8-NEXT:    .cfi_offset lr, 16
; CHECK-P8-NEXT:    bl __subkf3
; CHECK-P8-NEXT:    nop
; CHECK-P8-NEXT:    addi r1, r1, 32
; CHECK-P8-NEXT:    ld r0, 16(r1)
; CHECK-P8-NEXT:    mtlr r0
; CHECK-P8-NEXT:    blr
  %res = call fp128 @llvm.experimental.constrained.fsub.f128(
                        fp128 %f1, fp128 %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret fp128 %res
}

define fp128 @fmul_f128(fp128 %f1, fp128 %f2) #0 {
; CHECK-LABEL: fmul_f128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xsmulqp v2, v2, v3
; CHECK-NEXT:    blr
;
; CHECK-P8-LABEL: fmul_f128:
; CHECK-P8:       # %bb.0:
; CHECK-P8-NEXT:    mflr r0
; CHECK-P8-NEXT:    stdu r1, -32(r1)
; CHECK-P8-NEXT:    std r0, 48(r1)
; CHECK-P8-NEXT:    .cfi_def_cfa_offset 32
; CHECK-P8-NEXT:    .cfi_offset lr, 16
; CHECK-P8-NEXT:    bl __mulkf3
; CHECK-P8-NEXT:    nop
; CHECK-P8-NEXT:    addi r1, r1, 32
; CHECK-P8-NEXT:    ld r0, 16(r1)
; CHECK-P8-NEXT:    mtlr r0
; CHECK-P8-NEXT:    blr
  %res = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %f1, fp128 %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret fp128 %res
}

define fp128 @fdiv_f128(fp128 %f1, fp128 %f2) #0 {
; CHECK-LABEL: fdiv_f128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xsdivqp v2, v2, v3
; CHECK-NEXT:    blr
;
; CHECK-P8-LABEL: fdiv_f128:
; CHECK-P8:       # %bb.0:
; CHECK-P8-NEXT:    mflr r0
; CHECK-P8-NEXT:    stdu r1, -32(r1)
; CHECK-P8-NEXT:    std r0, 48(r1)
; CHECK-P8-NEXT:    .cfi_def_cfa_offset 32
; CHECK-P8-NEXT:    .cfi_offset lr, 16
; CHECK-P8-NEXT:    bl __divkf3
; CHECK-P8-NEXT:    nop
; CHECK-P8-NEXT:    addi r1, r1, 32
; CHECK-P8-NEXT:    ld r0, 16(r1)
; CHECK-P8-NEXT:    mtlr r0
; CHECK-P8-NEXT:    blr
  %res = call fp128 @llvm.experimental.constrained.fdiv.f128(
                        fp128 %f1, fp128 %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret fp128 %res
}

define fp128 @fmadd_f128(fp128 %f0, fp128 %f1, fp128 %f2) #0 {
; CHECK-LABEL: fmadd_f128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xsmaddqp v4, v2, v3
; CHECK-NEXT:    vmr v2, v4
; CHECK-NEXT:    blr
;
; CHECK-P8-LABEL: fmadd_f128:
; CHECK-P8:       # %bb.0:
; CHECK-P8-NEXT:    mflr r0
; CHECK-P8-NEXT:    stdu r1, -32(r1)
; CHECK-P8-NEXT:    std r0, 48(r1)
; CHECK-P8-NEXT:    .cfi_def_cfa_offset 32
; CHECK-P8-NEXT:    .cfi_offset lr, 16
; CHECK-P8-NEXT:    bl fmaf128
; CHECK-P8-NEXT:    nop
; CHECK-P8-NEXT:    addi r1, r1, 32
; CHECK-P8-NEXT:    ld r0, 16(r1)
; CHECK-P8-NEXT:    mtlr r0
; CHECK-P8-NEXT:    blr
  %res = call fp128 @llvm.experimental.constrained.fma.f128(
                        fp128 %f0, fp128 %f1, fp128 %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret fp128 %res
}

define fp128 @fmsub_f128(fp128 %f0, fp128 %f1, fp128 %f2) #0 {
; CHECK-LABEL: fmsub_f128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xsmsubqp v4, v2, v3
; CHECK-NEXT:    vmr v2, v4
; CHECK-NEXT:    blr
;
; CHECK-P8-LABEL: fmsub_f128:
; CHECK-P8:       # %bb.0:
; CHECK-P8-NEXT:    mflr r0
; CHECK-P8-NEXT:    stdu r1, -48(r1)
; CHECK-P8-NEXT:    std r0, 64(r1)
; CHECK-P8-NEXT:    .cfi_def_cfa_offset 48
; CHECK-P8-NEXT:    .cfi_offset lr, 16
; CHECK-P8-NEXT:    xxswapd vs0, v4
; CHECK-P8-NEXT:    addi r3, r1, 32
; CHECK-P8-NEXT:    stxvd2x vs0, 0, r3
; CHECK-P8-NEXT:    lbz r4, 47(r1)
; CHECK-P8-NEXT:    xori r4, r4, 128
; CHECK-P8-NEXT:    stb r4, 47(r1)
; CHECK-P8-NEXT:    lxvd2x vs0, 0, r3
; CHECK-P8-NEXT:    xxswapd v4, vs0
; CHECK-P8-NEXT:    bl fmaf128
; CHECK-P8-NEXT:    nop
; CHECK-P8-NEXT:    addi r1, r1, 48
; CHECK-P8-NEXT:    ld r0, 16(r1)
; CHECK-P8-NEXT:    mtlr r0
; CHECK-P8-NEXT:    blr
  %neg = fneg fp128 %f2
  %res = call fp128 @llvm.experimental.constrained.fma.f128(
                        fp128 %f0, fp128 %f1, fp128 %neg,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret fp128 %res
}

define fp128 @fnmadd_f128(fp128 %f0, fp128 %f1, fp128 %f2) #0 {
; CHECK-LABEL: fnmadd_f128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xsnmaddqp v4, v2, v3
; CHECK-NEXT:    vmr v2, v4
; CHECK-NEXT:    blr
;
; CHECK-P8-LABEL: fnmadd_f128:
; CHECK-P8:       # %bb.0:
; CHECK-P8-NEXT:    mflr r0
; CHECK-P8-NEXT:    stdu r1, -48(r1)
; CHECK-P8-NEXT:    std r0, 64(r1)
; CHECK-P8-NEXT:    .cfi_def_cfa_offset 48
; CHECK-P8-NEXT:    .cfi_offset lr, 16
; CHECK-P8-NEXT:    bl fmaf128
; CHECK-P8-NEXT:    nop
; CHECK-P8-NEXT:    xxswapd vs0, v2
; CHECK-P8-NEXT:    addi r3, r1, 32
; CHECK-P8-NEXT:    stxvd2x vs0, 0, r3
; CHECK-P8-NEXT:    lbz r4, 47(r1)
; CHECK-P8-NEXT:    xori r4, r4, 128
; CHECK-P8-NEXT:    stb r4, 47(r1)
; CHECK-P8-NEXT:    lxvd2x vs0, 0, r3
; CHECK-P8-NEXT:    xxswapd v2, vs0
; CHECK-P8-NEXT:    addi r1, r1, 48
; CHECK-P8-NEXT:    ld r0, 16(r1)
; CHECK-P8-NEXT:    mtlr r0
; CHECK-P8-NEXT:    blr
  %fma = call fp128 @llvm.experimental.constrained.fma.f128(
                        fp128 %f0, fp128 %f1, fp128 %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %res = fneg fp128 %fma
  ret fp128 %res
}

define fp128 @fnmsub_f128(fp128 %f0, fp128 %f1, fp128 %f2) #0 {
; CHECK-LABEL: fnmsub_f128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xsnmsubqp v4, v2, v3
; CHECK-NEXT:    vmr v2, v4
; CHECK-NEXT:    blr
;
; CHECK-P8-LABEL: fnmsub_f128:
; CHECK-P8:       # %bb.0:
; CHECK-P8-NEXT:    mflr r0
; CHECK-P8-NEXT:    stdu r1, -64(r1)
; CHECK-P8-NEXT:    std r0, 80(r1)
; CHECK-P8-NEXT:    .cfi_def_cfa_offset 64
; CHECK-P8-NEXT:    .cfi_offset lr, 16
; CHECK-P8-NEXT:    xxswapd vs0, v4
; CHECK-P8-NEXT:    addi r3, r1, 32
; CHECK-P8-NEXT:    stxvd2x vs0, 0, r3
; CHECK-P8-NEXT:    lbz r4, 47(r1)
; CHECK-P8-NEXT:    xori r4, r4, 128
; CHECK-P8-NEXT:    stb r4, 47(r1)
; CHECK-P8-NEXT:    lxvd2x vs0, 0, r3
; CHECK-P8-NEXT:    xxswapd v4, vs0
; CHECK-P8-NEXT:    bl fmaf128
; CHECK-P8-NEXT:    nop
; CHECK-P8-NEXT:    xxswapd vs0, v2
; CHECK-P8-NEXT:    addi r3, r1, 48
; CHECK-P8-NEXT:    stxvd2x vs0, 0, r3
; CHECK-P8-NEXT:    lbz r4, 63(r1)
; CHECK-P8-NEXT:    xori r4, r4, 128
; CHECK-P8-NEXT:    stb r4, 63(r1)
; CHECK-P8-NEXT:    lxvd2x vs0, 0, r3
; CHECK-P8-NEXT:    xxswapd v2, vs0
; CHECK-P8-NEXT:    addi r1, r1, 64
; CHECK-P8-NEXT:    ld r0, 16(r1)
; CHECK-P8-NEXT:    mtlr r0
; CHECK-P8-NEXT:    blr
  %neg = fneg fp128 %f2
  %fma = call fp128 @llvm.experimental.constrained.fma.f128(
                        fp128 %f0, fp128 %f1, fp128 %neg,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %res = fneg fp128 %fma
  ret fp128 %res
}


define fp128 @fsqrt_f128(fp128 %f1) #0 {
; CHECK-LABEL: fsqrt_f128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xssqrtqp v2, v2
; CHECK-NEXT:    blr
;
; CHECK-P8-LABEL: fsqrt_f128:
; CHECK-P8:       # %bb.0:
; CHECK-P8-NEXT:    mflr r0
; CHECK-P8-NEXT:    stdu r1, -32(r1)
; CHECK-P8-NEXT:    std r0, 48(r1)
; CHECK-P8-NEXT:    .cfi_def_cfa_offset 32
; CHECK-P8-NEXT:    .cfi_offset lr, 16
; CHECK-P8-NEXT:    bl sqrtf128
; CHECK-P8-NEXT:    nop
; CHECK-P8-NEXT:    addi r1, r1, 32
; CHECK-P8-NEXT:    ld r0, 16(r1)
; CHECK-P8-NEXT:    mtlr r0
; CHECK-P8-NEXT:    blr
  %res = call fp128 @llvm.experimental.constrained.sqrt.f128(
                        fp128 %f1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret fp128 %res
}

attributes #0 = { strictfp }
