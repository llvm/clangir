; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mattr=+avx512bf16,+avx512vl < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define <32 x bfloat> @stack_fold_cvtne2ps2bf16(<16 x float> %a0, <16 x float> %a1) {
; CHECK-LABEL: stack_fold_cvtne2ps2bf16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %zmm1, {{[-0-9]+}}(%r{{[sb]}}p) # 64-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    vcvtne2ps2bf16 {{[-0-9]+}}(%r{{[sb]}}p), %zmm0, %zmm0 # 64-byte Folded Reload
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.512(<16 x float> %a0, <16 x float> %a1)
  ret <32 x bfloat> %2
}
declare <32 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.512(<16 x float>, <16 x float>)

define <32 x bfloat> @stack_fold_cvtne2ps2bf16_mask(<16 x float> %a0, <16 x float> %a1, ptr %passthru, i32 %U) {
; CHECK-LABEL: stack_fold_cvtne2ps2bf16_mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %zmm1, {{[-0-9]+}}(%r{{[sb]}}p) # 64-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    kmovd %esi, %k1
; CHECK-NEXT:    vmovaps (%rdi), %zmm2
; CHECK-NEXT:    vcvtne2ps2bf16 {{[-0-9]+}}(%r{{[sb]}}p), %zmm0, %zmm2 {%k1} # 64-byte Folded Reload
; CHECK-NEXT:    vmovaps %zmm2, %zmm0
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.512(<16 x float> %a0, <16 x float> %a1)
  %3 = bitcast i32 %U to <32 x i1>
  ; load needed to keep the operation from being scheduled above the asm block
  %4 = load <32 x bfloat>, ptr %passthru
  %5 = select <32 x i1> %3, <32 x bfloat> %2, <32 x bfloat> %4
  ret <32 x bfloat> %5
}

define <32 x bfloat> @stack_fold_cvtne2ps2bf16_maskz(<16 x float> %a0, <16 x float> %a1, i32 %U) {
; CHECK-LABEL: stack_fold_cvtne2ps2bf16_maskz:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %zmm1, {{[-0-9]+}}(%r{{[sb]}}p) # 64-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    kmovd %edi, %k1
; CHECK-NEXT:    vcvtne2ps2bf16 {{[-0-9]+}}(%r{{[sb]}}p), %zmm0, %zmm0 {%k1} {z} # 64-byte Folded Reload
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <32 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.512(<16 x float> %a0, <16 x float> %a1)
  %3 = bitcast i32 %U to <32 x i1>
  %4 = select <32 x i1> %3, <32 x bfloat> %2, <32 x bfloat> zeroinitializer
  ret <32 x bfloat> %4
}

define <16 x bfloat> @stack_fold_cvtneps2bf16(<16 x float> %a0) {
; CHECK-LABEL: stack_fold_cvtneps2bf16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %zmm0, {{[-0-9]+}}(%r{{[sb]}}p) # 64-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    vcvtneps2bf16 {{[-0-9]+}}(%r{{[sb]}}p), %ymm0 # 64-byte Folded Reload
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <16 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.512(<16 x float> %a0)
  ret <16 x bfloat> %2
}
declare <16 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.512(<16 x float>)

define <16 x bfloat> @stack_fold_cvtneps2bf16_mask(<16 x float> %a0, ptr %passthru, i16 %U) {
; CHECK-LABEL: stack_fold_cvtneps2bf16_mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %zmm0, {{[-0-9]+}}(%r{{[sb]}}p) # 64-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    kmovd %esi, %k1
; CHECK-NEXT:    vmovaps (%rdi), %ymm1
; CHECK-NEXT:    vcvtneps2bf16 {{[-0-9]+}}(%r{{[sb]}}p), %ymm1 {%k1} # 64-byte Folded Reload
; CHECK-NEXT:    vmovaps %ymm1, %ymm0
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <16 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.512(<16 x float> %a0)
  %3 = bitcast i16 %U to <16 x i1>
  ; load needed to keep the operation from being scheduled above the asm block
  %4 = load <16 x bfloat>, ptr %passthru
  %5 = select <16 x i1> %3, <16 x bfloat> %2, <16 x bfloat> %4
  ret <16 x bfloat> %5
}

define <16 x bfloat> @stack_fold_cvtneps2bf16_maskz(<16 x float> %a0, i16 %U) {
; CHECK-LABEL: stack_fold_cvtneps2bf16_maskz:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %zmm0, {{[-0-9]+}}(%r{{[sb]}}p) # 64-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    kmovd %edi, %k1
; CHECK-NEXT:    vcvtneps2bf16 {{[-0-9]+}}(%r{{[sb]}}p), %ymm0 {%k1} {z} # 64-byte Folded Reload
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <16 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.512(<16 x float> %a0)
  %3 = bitcast i16 %U to <16 x i1>
  %4 = select <16 x i1> %3, <16 x bfloat> %2, <16 x bfloat> zeroinitializer
  ret <16 x bfloat> %4
}

define <16 x float> @stack_fold_vdpbf16ps(<16 x float> %a0, <32 x bfloat> %a1, <32 x bfloat> %a2) {
; CHECK-LABEL: stack_fold_vdpbf16ps:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %zmm2, {{[-0-9]+}}(%r{{[sb]}}p) # 64-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    vdpbf16ps {{[-0-9]+}}(%r{{[sb]}}p), %zmm1, %zmm0 # 64-byte Folded Reload
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <16 x float> @llvm.x86.avx512bf16.dpbf16ps.512(<16 x float> %a0, <32 x bfloat> %a1, <32 x bfloat> %a2)
  ret <16 x float> %2
}
declare <16 x float> @llvm.x86.avx512bf16.dpbf16ps.512(<16 x float>, <32 x bfloat>, <32 x bfloat>)

define <16 x float> @stack_fold_vdpbf16ps_mask(ptr %a0, <32 x bfloat> %a1, <32 x bfloat> %a2, ptr %passthru, i16 %U) {
; CHECK-LABEL: stack_fold_vdpbf16ps_mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %zmm1, {{[-0-9]+}}(%r{{[sb]}}p) # 64-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    vmovaps (%rdi), %zmm2
; CHECK-NEXT:    kmovd %edx, %k1
; CHECK-NEXT:    vdpbf16ps {{[-0-9]+}}(%r{{[sb]}}p), %zmm0, %zmm2 {%k1} # 64-byte Folded Reload
; CHECK-NEXT:    vmovaps %zmm2, %zmm0
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ; load needed to keep the operation from being scheduled above the asm block
  %2 = load <16 x float>, ptr %a0
  %3 = tail call <16 x float> @llvm.x86.avx512bf16.dpbf16ps.512(<16 x float> %2, <32 x bfloat> %a1, <32 x bfloat> %a2)
  %4 = bitcast i16 %U to <16 x i1>
  %5 = select <16 x i1> %4, <16 x float> %3, <16 x float> %2
  ret <16 x float> %5
}

define <16 x float> @stack_fold_vdpbf16ps_maskz(<16 x float> %a0, <32 x bfloat> %a1, <32 x bfloat> %a2, ptr %U) {
; CHECK-LABEL: stack_fold_vdpbf16ps_maskz:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %zmm2, {{[-0-9]+}}(%r{{[sb]}}p) # 64-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    kmovw (%rdi), %k1
; CHECK-NEXT:    vdpbf16ps {{[-0-9]+}}(%r{{[sb]}}p), %zmm1, %zmm0 {%k1} {z} # 64-byte Folded Reload
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <16 x float> @llvm.x86.avx512bf16.dpbf16ps.512(<16 x float> %a0, <32 x bfloat> %a1, <32 x bfloat> %a2)
  %3 = load i16, ptr %U
  %4 = bitcast i16 %3 to <16 x i1>
  %5 = select <16 x i1> %4, <16 x float> %2, <16 x float> zeroinitializer
  ret <16 x float> %5
}



define <16 x bfloat> @stack_fold_cvtne2ps2bf16_ymm(<8 x float> %a0, <8 x float> %a1) {
; CHECK-LABEL: stack_fold_cvtne2ps2bf16_ymm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %ymm1, {{[-0-9]+}}(%r{{[sb]}}p) # 32-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    vcvtne2ps2bf16 {{[-0-9]+}}(%r{{[sb]}}p), %ymm0, %ymm0 # 32-byte Folded Reload
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.256(<8 x float> %a0, <8 x float> %a1)
  ret <16 x bfloat> %2
}
declare <16 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.256(<8 x float>, <8 x float>)

define <16 x bfloat> @stack_fold_cvtne2ps2bf16_mask_ymm(<8 x float> %a0, <8 x float> %a1, ptr %passthru, i16 %U) {
; CHECK-LABEL: stack_fold_cvtne2ps2bf16_mask_ymm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %ymm1, {{[-0-9]+}}(%r{{[sb]}}p) # 32-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    kmovd %esi, %k1
; CHECK-NEXT:    vmovaps (%rdi), %ymm2
; CHECK-NEXT:    vcvtne2ps2bf16 {{[-0-9]+}}(%r{{[sb]}}p), %ymm0, %ymm2 {%k1} # 32-byte Folded Reload
; CHECK-NEXT:    vmovaps %ymm2, %ymm0
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.256(<8 x float> %a0, <8 x float> %a1)
  %3 = bitcast i16 %U to <16 x i1>
  ; load needed to keep the operation from being scheduled above the asm block
  %4 = load <16 x bfloat>, ptr %passthru
  %5 = select <16 x i1> %3, <16 x bfloat> %2, <16 x bfloat> %4
  ret <16 x bfloat> %5
}

define <16 x bfloat> @stack_fold_cvtne2ps2bf16_maskz_ymm(<8 x float> %a0, <8 x float> %a1, i16 %U) {
; CHECK-LABEL: stack_fold_cvtne2ps2bf16_maskz_ymm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %ymm1, {{[-0-9]+}}(%r{{[sb]}}p) # 32-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    kmovd %edi, %k1
; CHECK-NEXT:    vcvtne2ps2bf16 {{[-0-9]+}}(%r{{[sb]}}p), %ymm0, %ymm0 {%k1} {z} # 32-byte Folded Reload
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <16 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.256(<8 x float> %a0, <8 x float> %a1)
  %3 = bitcast i16 %U to <16 x i1>
  %4 = select <16 x i1> %3, <16 x bfloat> %2, <16 x bfloat> zeroinitializer
  ret <16 x bfloat> %4
}

define <8 x bfloat> @stack_fold_cvtneps2bf16_ymm(<8 x float> %a0) {
; CHECK-LABEL: stack_fold_cvtneps2bf16_ymm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %ymm0, {{[-0-9]+}}(%r{{[sb]}}p) # 32-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    vcvtneps2bf16y {{[-0-9]+}}(%r{{[sb]}}p), %xmm0 # 32-byte Folded Reload
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <8 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.256(<8 x float> %a0)
  ret <8 x bfloat> %2
}
declare <8 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.256(<8 x float>)

define <8 x bfloat> @stack_fold_cvtneps2bf16_mask_ymm(<8 x float> %a0, ptr %passthru, i8 %U) {
; CHECK-LABEL: stack_fold_cvtneps2bf16_mask_ymm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %ymm0, {{[-0-9]+}}(%r{{[sb]}}p) # 32-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    kmovd %esi, %k1
; CHECK-NEXT:    vmovaps (%rdi), %xmm1
; CHECK-NEXT:    vcvtneps2bf16y {{[-0-9]+}}(%r{{[sb]}}p), %xmm1 {%k1} # 32-byte Folded Reload
; CHECK-NEXT:    vmovaps %xmm1, %xmm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <8 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.256(<8 x float> %a0)
  %3 = bitcast i8 %U to <8 x i1>
  ; load needed to keep the operation from being scheduled above the asm block
  %4 = load <8 x bfloat>, ptr %passthru
  %5 = select <8 x i1> %3, <8 x bfloat> %2, <8 x bfloat> %4
  ret <8 x bfloat> %5
}

define <8 x bfloat> @stack_fold_cvtneps2bf16_maskz_ymm(<8 x float> %a0, i8 %U) {
; CHECK-LABEL: stack_fold_cvtneps2bf16_maskz_ymm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %ymm0, {{[-0-9]+}}(%r{{[sb]}}p) # 32-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    kmovd %edi, %k1
; CHECK-NEXT:    vcvtneps2bf16y {{[-0-9]+}}(%r{{[sb]}}p), %xmm0 {%k1} {z} # 32-byte Folded Reload
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <8 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.256(<8 x float> %a0)
  %3 = bitcast i8 %U to <8 x i1>
  %4 = select <8 x i1> %3, <8 x bfloat> %2, <8 x bfloat> zeroinitializer
  ret <8 x bfloat> %4
}

define <8 x float> @stack_fold_vdpbf16ps_ymm(<8 x float> %a0, <16 x bfloat> %a1, <16 x bfloat> %a2) {
; CHECK-LABEL: stack_fold_vdpbf16ps_ymm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %ymm2, {{[-0-9]+}}(%r{{[sb]}}p) # 32-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    vdpbf16ps {{[-0-9]+}}(%r{{[sb]}}p), %ymm1, %ymm0 # 32-byte Folded Reload
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <8 x float> @llvm.x86.avx512bf16.dpbf16ps.256(<8 x float> %a0, <16 x bfloat> %a1, <16 x bfloat> %a2)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx512bf16.dpbf16ps.256(<8 x float>, <16 x bfloat>, <16 x bfloat>)

define <8 x float> @stack_fold_vdpbf16ps_mask_ymm(ptr %a0, <16 x bfloat> %a1, <16 x bfloat> %a2, ptr %passthru, i8 %U) {
; CHECK-LABEL: stack_fold_vdpbf16ps_mask_ymm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %ymm1, {{[-0-9]+}}(%r{{[sb]}}p) # 32-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    vmovaps (%rdi), %ymm2
; CHECK-NEXT:    kmovd %edx, %k1
; CHECK-NEXT:    vdpbf16ps {{[-0-9]+}}(%r{{[sb]}}p), %ymm0, %ymm2 {%k1} # 32-byte Folded Reload
; CHECK-NEXT:    vmovaps %ymm2, %ymm0
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ; load needed to keep the operation from being scheduled above the asm block
  %2 = load <8 x float>, ptr %a0
  %3 = tail call <8 x float> @llvm.x86.avx512bf16.dpbf16ps.256(<8 x float> %2, <16 x bfloat> %a1, <16 x bfloat> %a2)
  %4 = bitcast i8 %U to <8 x i1>
  %5 = select <8 x i1> %4, <8 x float> %3, <8 x float> %2
  ret <8 x float> %5
}

define <8 x float> @stack_fold_vdpbf16ps_maskz_ymm(<8 x float> %a0, <16 x bfloat> %a1, <16 x bfloat> %a2, ptr %U) {
; CHECK-LABEL: stack_fold_vdpbf16ps_maskz_ymm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovups %ymm2, {{[-0-9]+}}(%r{{[sb]}}p) # 32-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    movzbl (%rdi), %eax
; CHECK-NEXT:    kmovd %eax, %k1
; CHECK-NEXT:    vdpbf16ps {{[-0-9]+}}(%r{{[sb]}}p), %ymm1, %ymm0 {%k1} {z} # 32-byte Folded Reload
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <8 x float> @llvm.x86.avx512bf16.dpbf16ps.256(<8 x float> %a0, <16 x bfloat> %a1, <16 x bfloat> %a2)
  %3 = load i8, ptr %U
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = select <8 x i1> %4, <8 x float> %2, <8 x float> zeroinitializer
  ret <8 x float> %5
}




define <8 x bfloat> @stack_fold_cvtne2ps2bf16_xmm(<4 x float> %a0, <4 x float> %a1) {
; CHECK-LABEL: stack_fold_cvtne2ps2bf16_xmm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovaps %xmm1, {{[-0-9]+}}(%r{{[sb]}}p) # 16-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    vcvtne2ps2bf16 {{[-0-9]+}}(%r{{[sb]}}p), %xmm0, %xmm0 # 16-byte Folded Reload
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.128(<4 x float> %a0, <4 x float> %a1)
  ret <8 x bfloat> %2
}
declare <8 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.128(<4 x float>, <4 x float>)

define <8 x bfloat> @stack_fold_cvtne2ps2bf16_mask_xmm(<4 x float> %a0, <4 x float> %a1, ptr %passthru, i8 %U) {
; CHECK-LABEL: stack_fold_cvtne2ps2bf16_mask_xmm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovaps %xmm1, {{[-0-9]+}}(%r{{[sb]}}p) # 16-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    kmovd %esi, %k1
; CHECK-NEXT:    vmovaps (%rdi), %xmm2
; CHECK-NEXT:    vcvtne2ps2bf16 {{[-0-9]+}}(%r{{[sb]}}p), %xmm0, %xmm2 {%k1} # 16-byte Folded Reload
; CHECK-NEXT:    vmovaps %xmm2, %xmm0
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.128(<4 x float> %a0, <4 x float> %a1)
  %3 = bitcast i8 %U to <8 x i1>
  ; load needed to keep the operation from being scheduled above the asm block
  %4 = load <8 x bfloat>, ptr %passthru
  %5 = select <8 x i1> %3, <8 x bfloat> %2, <8 x bfloat> %4
  ret <8 x bfloat> %5
}

define <8 x bfloat> @stack_fold_cvtne2ps2bf16_maskz_xmm(<4 x float> %a0, <4 x float> %a1, i8 %U) {
; CHECK-LABEL: stack_fold_cvtne2ps2bf16_maskz_xmm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovaps %xmm1, {{[-0-9]+}}(%r{{[sb]}}p) # 16-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    kmovd %edi, %k1
; CHECK-NEXT:    vcvtne2ps2bf16 {{[-0-9]+}}(%r{{[sb]}}p), %xmm0, %xmm0 {%k1} {z} # 16-byte Folded Reload
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = call <8 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.128(<4 x float> %a0, <4 x float> %a1)
  %3 = bitcast i8 %U to <8 x i1>
  %4 = select <8 x i1> %3, <8 x bfloat> %2, <8 x bfloat> zeroinitializer
  ret <8 x bfloat> %4
}

define <8 x bfloat> @stack_fold_cvtneps2bf16_xmm(<4 x float> %a0) {
; CHECK-LABEL: stack_fold_cvtneps2bf16_xmm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovaps %xmm0, {{[-0-9]+}}(%r{{[sb]}}p) # 16-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    vcvtneps2bf16x {{[-0-9]+}}(%r{{[sb]}}p), %xmm0 # 16-byte Folded Reload
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <8 x bfloat> @llvm.x86.avx512bf16.mask.cvtneps2bf16.128(<4 x float> %a0, <8 x bfloat> undef, <4 x i1> <i1 true, i1 true, i1 true, i1 true>)
  ret <8 x bfloat> %2
}
declare <8 x bfloat> @llvm.x86.avx512bf16.mask.cvtneps2bf16.128(<4 x float>, <8 x bfloat>, <4 x i1>)

define <8 x bfloat> @stack_fold_cvtneps2bf16_mask_xmm(<4 x float> %a0, ptr %passthru, i8 %U) {
; CHECK-LABEL: stack_fold_cvtneps2bf16_mask_xmm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovaps %xmm0, {{[-0-9]+}}(%r{{[sb]}}p) # 16-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    vmovaps (%rdi), %xmm1
; CHECK-NEXT:    kmovd %esi, %k1
; CHECK-NEXT:    vcvtneps2bf16x {{[-0-9]+}}(%r{{[sb]}}p), %xmm1 {%k1} # 16-byte Folded Reload
; CHECK-NEXT:    vmovaps %xmm1, %xmm0
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = load <8 x bfloat>, ptr %passthru
  %3 = bitcast i8 %U to <8 x i1>
  %4 = shufflevector <8 x i1> %3, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %5 = tail call <8 x bfloat> @llvm.x86.avx512bf16.mask.cvtneps2bf16.128(<4 x float> %a0, <8 x bfloat> %2, <4 x i1> %4)
  ret <8 x bfloat> %5
}

define <8 x bfloat> @stack_fold_cvtneps2bf16_maskz_xmm(<4 x float> %a0, i8 %U) {
; CHECK-LABEL: stack_fold_cvtneps2bf16_maskz_xmm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovaps %xmm0, {{[-0-9]+}}(%r{{[sb]}}p) # 16-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    kmovd %edi, %k1
; CHECK-NEXT:    vcvtneps2bf16x {{[-0-9]+}}(%r{{[sb]}}p), %xmm0 {%k1} {z} # 16-byte Folded Reload
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = bitcast i8 %U to <8 x i1>
  %3 = shufflevector <8 x i1> %2, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %4 = tail call <8 x bfloat> @llvm.x86.avx512bf16.mask.cvtneps2bf16.128(<4 x float> %a0, <8 x bfloat> zeroinitializer, <4 x i1> %3)
  ret <8 x bfloat> %4
}

define <4 x float> @stack_fold_vdpbf16ps_xmm(<4 x float> %a0, <8 x bfloat> %a1, <8 x bfloat> %a2) {
; CHECK-LABEL: stack_fold_vdpbf16ps_xmm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovaps %xmm2, {{[-0-9]+}}(%r{{[sb]}}p) # 16-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    vdpbf16ps {{[-0-9]+}}(%r{{[sb]}}p), %xmm1, %xmm0 # 16-byte Folded Reload
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <4 x float> @llvm.x86.avx512bf16.dpbf16ps.128(<4 x float> %a0, <8 x bfloat> %a1, <8 x bfloat> %a2)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.avx512bf16.dpbf16ps.128(<4 x float>, <8 x bfloat>, <8 x bfloat>)

define <4 x float> @stack_fold_vdpbf16ps_mask_xmm(ptr %a0, <8 x bfloat> %a1, <8 x bfloat> %a2, ptr %passthru, i8 %U) {
; CHECK-LABEL: stack_fold_vdpbf16ps_mask_xmm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovaps %xmm1, {{[-0-9]+}}(%r{{[sb]}}p) # 16-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    vmovaps (%rdi), %xmm2
; CHECK-NEXT:    kmovd %edx, %k1
; CHECK-NEXT:    vdpbf16ps {{[-0-9]+}}(%r{{[sb]}}p), %xmm0, %xmm2 {%k1} # 16-byte Folded Reload
; CHECK-NEXT:    vmovaps %xmm2, %xmm0
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  ; load needed to keep the operation from being scheduled above the asm block
  %2 = load <4 x float>, ptr %a0
  %3 = tail call <4 x float> @llvm.x86.avx512bf16.dpbf16ps.128(<4 x float> %2, <8 x bfloat> %a1, <8 x bfloat> %a2)
  %4 = bitcast i8 %U to <8 x i1>
  %5 = shufflevector <8 x i1> %4, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %6 = select <4 x i1> %5, <4 x float> %3, <4 x float> %2
  ret <4 x float> %6
}

define <4 x float> @stack_fold_vdpbf16ps_maskz_xmm(<4 x float> %a0, <8 x bfloat> %a1, <8 x bfloat> %a2, ptr %U) {
; CHECK-LABEL: stack_fold_vdpbf16ps_maskz_xmm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovaps %xmm2, {{[-0-9]+}}(%r{{[sb]}}p) # 16-byte Spill
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    movzbl (%rdi), %eax
; CHECK-NEXT:    kmovd %eax, %k1
; CHECK-NEXT:    vdpbf16ps {{[-0-9]+}}(%r{{[sb]}}p), %xmm1, %xmm0 {%k1} {z} # 16-byte Folded Reload
; CHECK-NEXT:    retq
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{flags}"()
  %2 = tail call <4 x float> @llvm.x86.avx512bf16.dpbf16ps.128(<4 x float> %a0, <8 x bfloat> %a1, <8 x bfloat> %a2)
  %3 = load i8, ptr %U
  %4 = bitcast i8 %3 to <8 x i1>
  %5 = shufflevector <8 x i1> %4, <8 x i1> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %6 = select <4 x i1> %5, <4 x float> %2, <4 x float> zeroinitializer
  ret <4 x float> %6
}
