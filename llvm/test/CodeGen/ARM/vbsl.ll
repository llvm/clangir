; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

; rdar://12471808

define <8 x i8> @v_bsli8(ptr %A, ptr %B, ptr %C) nounwind {
; CHECK-LABEL: v_bsli8:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d18, [r0]
; CHECK-NEXT:    vldr d16, [r2]
; CHECK-NEXT:    vldr d17, [r1]
; CHECK-NEXT:    vbit d16, d17, d18
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = load <8 x i8>, ptr %B
	%tmp3 = load <8 x i8>, ptr %C
	%tmp4 = and <8 x i8> %tmp1, %tmp2
	%tmp5 = xor <8 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp6 = and <8 x i8> %tmp5, %tmp3
	%tmp7 = or <8 x i8> %tmp4, %tmp6
	ret <8 x i8> %tmp7
}

define <4 x i16> @v_bsli16(ptr %A, ptr %B, ptr %C) nounwind {
; CHECK-LABEL: v_bsli16:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d18, [r0]
; CHECK-NEXT:    vldr d16, [r2]
; CHECK-NEXT:    vldr d17, [r1]
; CHECK-NEXT:    vbit d16, d17, d18
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
	%tmp3 = load <4 x i16>, ptr %C
	%tmp4 = and <4 x i16> %tmp1, %tmp2
	%tmp5 = xor <4 x i16> %tmp1, < i16 -1, i16 -1, i16 -1, i16 -1 >
	%tmp6 = and <4 x i16> %tmp5, %tmp3
	%tmp7 = or <4 x i16> %tmp4, %tmp6
	ret <4 x i16> %tmp7
}

define <2 x i32> @v_bsli32(ptr %A, ptr %B, ptr %C) nounwind {
; CHECK-LABEL: v_bsli32:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d18, [r0]
; CHECK-NEXT:    vldr d16, [r2]
; CHECK-NEXT:    vldr d17, [r1]
; CHECK-NEXT:    vbit d16, d17, d18
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = load <2 x i32>, ptr %B
	%tmp3 = load <2 x i32>, ptr %C
	%tmp4 = and <2 x i32> %tmp1, %tmp2
	%tmp5 = xor <2 x i32> %tmp1, < i32 -1, i32 -1 >
	%tmp6 = and <2 x i32> %tmp5, %tmp3
	%tmp7 = or <2 x i32> %tmp4, %tmp6
	ret <2 x i32> %tmp7
}

define <1 x i64> @v_bsli64(ptr %A, ptr %B, ptr %C) nounwind {
; CHECK-LABEL: v_bsli64:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d18, [r0]
; CHECK-NEXT:    vldr d16, [r2]
; CHECK-NEXT:    vldr d17, [r1]
; CHECK-NEXT:    vbit d16, d17, d18
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <1 x i64>, ptr %A
	%tmp2 = load <1 x i64>, ptr %B
	%tmp3 = load <1 x i64>, ptr %C
	%tmp4 = and <1 x i64> %tmp1, %tmp2
	%tmp5 = xor <1 x i64> %tmp1, < i64 -1 >
	%tmp6 = and <1 x i64> %tmp5, %tmp3
	%tmp7 = or <1 x i64> %tmp4, %tmp6
	ret <1 x i64> %tmp7
}

define <16 x i8> @v_bslQi8(ptr %A, ptr %B, ptr %C) nounwind {
; CHECK-LABEL: v_bslQi8:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d20, d21}, [r0]
; CHECK-NEXT:    vld1.64 {d16, d17}, [r2]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r1]
; CHECK-NEXT:    vbit q8, q9, q10
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = load <16 x i8>, ptr %B
	%tmp3 = load <16 x i8>, ptr %C
	%tmp4 = and <16 x i8> %tmp1, %tmp2
	%tmp5 = xor <16 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp6 = and <16 x i8> %tmp5, %tmp3
	%tmp7 = or <16 x i8> %tmp4, %tmp6
	ret <16 x i8> %tmp7
}

define <8 x i16> @v_bslQi16(ptr %A, ptr %B, ptr %C) nounwind {
; CHECK-LABEL: v_bslQi16:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d20, d21}, [r0]
; CHECK-NEXT:    vld1.64 {d16, d17}, [r2]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r1]
; CHECK-NEXT:    vbit q8, q9, q10
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = load <8 x i16>, ptr %B
	%tmp3 = load <8 x i16>, ptr %C
	%tmp4 = and <8 x i16> %tmp1, %tmp2
	%tmp5 = xor <8 x i16> %tmp1, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1 >
	%tmp6 = and <8 x i16> %tmp5, %tmp3
	%tmp7 = or <8 x i16> %tmp4, %tmp6
	ret <8 x i16> %tmp7
}

define <4 x i32> @v_bslQi32(ptr %A, ptr %B, ptr %C) nounwind {
; CHECK-LABEL: v_bslQi32:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d20, d21}, [r0]
; CHECK-NEXT:    vld1.64 {d16, d17}, [r2]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r1]
; CHECK-NEXT:    vbit q8, q9, q10
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = load <4 x i32>, ptr %B
	%tmp3 = load <4 x i32>, ptr %C
	%tmp4 = and <4 x i32> %tmp1, %tmp2
	%tmp5 = xor <4 x i32> %tmp1, < i32 -1, i32 -1, i32 -1, i32 -1 >
	%tmp6 = and <4 x i32> %tmp5, %tmp3
	%tmp7 = or <4 x i32> %tmp4, %tmp6
	ret <4 x i32> %tmp7
}

define <2 x i64> @v_bslQi64(ptr %A, ptr %B, ptr %C) nounwind {
; CHECK-LABEL: v_bslQi64:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d20, d21}, [r0]
; CHECK-NEXT:    vld1.64 {d16, d17}, [r2]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r1]
; CHECK-NEXT:    vbit q8, q9, q10
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
	%tmp1 = load <2 x i64>, ptr %A
	%tmp2 = load <2 x i64>, ptr %B
	%tmp3 = load <2 x i64>, ptr %C
	%tmp4 = and <2 x i64> %tmp1, %tmp2
	%tmp5 = xor <2 x i64> %tmp1, < i64 -1, i64 -1 >
	%tmp6 = and <2 x i64> %tmp5, %tmp3
	%tmp7 = or <2 x i64> %tmp4, %tmp6
	ret <2 x i64> %tmp7
}

define <8 x i8> @f1(<8 x i8> %a, <8 x i8> %b, <8 x i8> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: f1:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [sp]
; CHECK-NEXT:    vmov d17, r2, r3
; CHECK-NEXT:    vmov d18, r0, r1
; CHECK-NEXT:    vbit d16, d17, d18
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %vbsl.i = tail call <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8> %a, <8 x i8> %b, <8 x i8> %c) nounwind
  ret <8 x i8> %vbsl.i
}

define <4 x i16> @f2(<4 x i16> %a, <4 x i16> %b, <4 x i16> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: f2:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [sp]
; CHECK-NEXT:    vmov d17, r2, r3
; CHECK-NEXT:    vmov d18, r0, r1
; CHECK-NEXT:    vbit d16, d17, d18
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %vbsl3.i = tail call <4 x i16> @llvm.arm.neon.vbsl.v4i16(<4 x i16> %a, <4 x i16> %b, <4 x i16> %c) nounwind
  ret <4 x i16> %vbsl3.i
}

define <2 x i32> @f3(<2 x i32> %a, <2 x i32> %b, <2 x i32> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: f3:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [sp]
; CHECK-NEXT:    vmov d17, r2, r3
; CHECK-NEXT:    vmov d18, r0, r1
; CHECK-NEXT:    vbit d16, d17, d18
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %vbsl3.i = tail call <2 x i32> @llvm.arm.neon.vbsl.v2i32(<2 x i32> %a, <2 x i32> %b, <2 x i32> %c) nounwind
  ret <2 x i32> %vbsl3.i
}

define <2 x float> @f4(<2 x float> %a, <2 x float> %b, <2 x float> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: f4:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [sp]
; CHECK-NEXT:    vmov d17, r2, r3
; CHECK-NEXT:    vmov d18, r0, r1
; CHECK-NEXT:    vbit d16, d17, d18
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %vbsl4.i = tail call <2 x float> @llvm.arm.neon.vbsl.v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c) nounwind
  ret <2 x float> %vbsl4.i
}

define <16 x i8> @g1(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: g1:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vmov d19, r2, r3
; CHECK-NEXT:    add r12, sp, #16
; CHECK-NEXT:    vmov d18, r0, r1
; CHECK-NEXT:    mov r0, sp
; CHECK-NEXT:    vld1.64 {d16, d17}, [r12]
; CHECK-NEXT:    vld1.64 {d20, d21}, [r0]
; CHECK-NEXT:    vbit q8, q10, q9
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %vbsl.i = tail call <16 x i8> @llvm.arm.neon.vbsl.v16i8(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) nounwind
  ret <16 x i8> %vbsl.i
}

define <8 x i16> @g2(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: g2:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vmov d19, r2, r3
; CHECK-NEXT:    add r12, sp, #16
; CHECK-NEXT:    vmov d18, r0, r1
; CHECK-NEXT:    mov r0, sp
; CHECK-NEXT:    vld1.64 {d16, d17}, [r12]
; CHECK-NEXT:    vld1.64 {d20, d21}, [r0]
; CHECK-NEXT:    vbit q8, q10, q9
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %vbsl3.i = tail call <8 x i16> @llvm.arm.neon.vbsl.v8i16(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c) nounwind
  ret <8 x i16> %vbsl3.i
}

define <4 x i32> @g3(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: g3:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vmov d19, r2, r3
; CHECK-NEXT:    add r12, sp, #16
; CHECK-NEXT:    vmov d18, r0, r1
; CHECK-NEXT:    mov r0, sp
; CHECK-NEXT:    vld1.64 {d16, d17}, [r12]
; CHECK-NEXT:    vld1.64 {d20, d21}, [r0]
; CHECK-NEXT:    vbit q8, q10, q9
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %vbsl3.i = tail call <4 x i32> @llvm.arm.neon.vbsl.v4i32(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) nounwind
  ret <4 x i32> %vbsl3.i
}

define <4 x float> @g4(<4 x float> %a, <4 x float> %b, <4 x float> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: g4:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vmov d19, r2, r3
; CHECK-NEXT:    add r12, sp, #16
; CHECK-NEXT:    vmov d18, r0, r1
; CHECK-NEXT:    mov r0, sp
; CHECK-NEXT:    vld1.64 {d16, d17}, [r12]
; CHECK-NEXT:    vld1.64 {d20, d21}, [r0]
; CHECK-NEXT:    vbit q8, q10, q9
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %vbsl4.i = tail call <4 x float> @llvm.arm.neon.vbsl.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c) nounwind
  ret <4 x float> %vbsl4.i
}

define <1 x i64> @test_vbsl_s64(<1 x i64> %a, <1 x i64> %b, <1 x i64> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: test_vbsl_s64:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [sp]
; CHECK-NEXT:    vmov d17, r2, r3
; CHECK-NEXT:    vmov d18, r0, r1
; CHECK-NEXT:    vbit d16, d17, d18
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %vbsl3.i = tail call <1 x i64> @llvm.arm.neon.vbsl.v1i64(<1 x i64> %a, <1 x i64> %b, <1 x i64> %c) nounwind
  ret <1 x i64> %vbsl3.i
}

define <1 x i64> @test_vbsl_u64(<1 x i64> %a, <1 x i64> %b, <1 x i64> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: test_vbsl_u64:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [sp]
; CHECK-NEXT:    vmov d17, r2, r3
; CHECK-NEXT:    vmov d18, r0, r1
; CHECK-NEXT:    vbit d16, d17, d18
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %vbsl3.i = tail call <1 x i64> @llvm.arm.neon.vbsl.v1i64(<1 x i64> %a, <1 x i64> %b, <1 x i64> %c) nounwind
  ret <1 x i64> %vbsl3.i
}

define <2 x i64> @test_vbslq_s64(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: test_vbslq_s64:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vmov d19, r2, r3
; CHECK-NEXT:    add r12, sp, #16
; CHECK-NEXT:    vmov d18, r0, r1
; CHECK-NEXT:    mov r0, sp
; CHECK-NEXT:    vld1.64 {d16, d17}, [r12]
; CHECK-NEXT:    vld1.64 {d20, d21}, [r0]
; CHECK-NEXT:    vbit q8, q10, q9
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %vbsl3.i = tail call <2 x i64> @llvm.arm.neon.vbsl.v2i64(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c) nounwind
  ret <2 x i64> %vbsl3.i
}

define <2 x i64> @test_vbslq_u64(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c) nounwind readnone optsize ssp {
; CHECK-LABEL: test_vbslq_u64:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vmov d19, r2, r3
; CHECK-NEXT:    add r12, sp, #16
; CHECK-NEXT:    vmov d18, r0, r1
; CHECK-NEXT:    mov r0, sp
; CHECK-NEXT:    vld1.64 {d16, d17}, [r12]
; CHECK-NEXT:    vld1.64 {d20, d21}, [r0]
; CHECK-NEXT:    vbit q8, q10, q9
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %vbsl3.i = tail call <2 x i64> @llvm.arm.neon.vbsl.v2i64(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c) nounwind
  ret <2 x i64> %vbsl3.i
}

declare <4 x i32> @llvm.arm.neon.vbsl.v4i32(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vbsl.v8i16(<8 x i16>, <8 x i16>, <8 x i16>) nounwind readnone
declare <16 x i8> @llvm.arm.neon.vbsl.v16i8(<16 x i8>, <16 x i8>, <16 x i8>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vbsl.v2i32(<2 x i32>, <2 x i32>, <2 x i32>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vbsl.v4i16(<4 x i16>, <4 x i16>, <4 x i16>) nounwind readnone
declare <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8>, <8 x i8>, <8 x i8>) nounwind readnone
declare <2 x float> @llvm.arm.neon.vbsl.v2f32(<2 x float>, <2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vbsl.v4f32(<4 x float>, <4 x float>, <4 x float>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vbsl.v2i64(<2 x i64>, <2 x i64>, <2 x i64>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vbsl.v1i64(<1 x i64>, <1 x i64>, <1 x i64>) nounwind readnone
