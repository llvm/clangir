; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=arm-eabi -float-abi=soft -mattr=+neon | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: llc < %s -mtriple=arm-eabi -float-abi=soft -mattr=+neon -regalloc=basic | FileCheck %s --check-prefixes=CHECK,BASIC

;Check the (default) alignment value.
define <8 x i8> @vld1lanei8(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld1lanei8:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vld1.8 {d16[3]}, [r0]
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <8 x i8>, ptr %B
  %tmp2 = load i8, ptr %A, align 8
  %tmp3 = insertelement <8 x i8> %tmp1, i8 %tmp2, i32 3
  ret <8 x i8> %tmp3
}

;Check the alignment value.  Max for this instruction is 16 bits:
define <4 x i16> @vld1lanei16(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld1lanei16:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vld1.16 {d16[2]}, [r0:16]
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <4 x i16>, ptr %B
  %tmp2 = load i16, ptr %A, align 8
  %tmp3 = insertelement <4 x i16> %tmp1, i16 %tmp2, i32 2
  ret <4 x i16> %tmp3
}

;Check the alignment value.  Max for this instruction is 32 bits:
define <2 x i32> @vld1lanei32(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld1lanei32:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vld1.32 {d16[1]}, [r0:32]
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <2 x i32>, ptr %B
  %tmp2 = load i32, ptr %A, align 8
  %tmp3 = insertelement <2 x i32> %tmp1, i32 %tmp2, i32 1
  ret <2 x i32> %tmp3
}

;Check the alignment value.  Legal values are none or :32.
define <2 x i32> @vld1lanei32a32(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld1lanei32a32:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vld1.32 {d16[1]}, [r0:32]
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <2 x i32>, ptr %B
  %tmp2 = load i32, ptr %A, align 4
  %tmp3 = insertelement <2 x i32> %tmp1, i32 %tmp2, i32 1
  ret <2 x i32> %tmp3
}

define <2 x float> @vld1lanef(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld1lanef:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vld1.32 {d16[1]}, [r0:32]
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <2 x float>, ptr %B
  %tmp2 = load float, ptr %A, align 4
  %tmp3 = insertelement <2 x float> %tmp1, float %tmp2, i32 1
  ret <2 x float> %tmp3
}

define <16 x i8> @vld1laneQi8(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld1laneQi8:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vld1.8 {d17[1]}, [r0]
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <16 x i8>, ptr %B
  %tmp2 = load i8, ptr %A, align 8
  %tmp3 = insertelement <16 x i8> %tmp1, i8 %tmp2, i32 9
  ret <16 x i8> %tmp3
}

define <8 x i16> @vld1laneQi16(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld1laneQi16:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vld1.16 {d17[1]}, [r0:16]
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <8 x i16>, ptr %B
  %tmp2 = load i16, ptr %A, align 8
  %tmp3 = insertelement <8 x i16> %tmp1, i16 %tmp2, i32 5
  ret <8 x i16> %tmp3
}

define <4 x i32> @vld1laneQi32(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld1laneQi32:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vld1.32 {d17[1]}, [r0:32]
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <4 x i32>, ptr %B
  %tmp2 = load i32, ptr %A, align 8
  %tmp3 = insertelement <4 x i32> %tmp1, i32 %tmp2, i32 3
  ret <4 x i32> %tmp3
}

define <4 x float> @vld1laneQf(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld1laneQf:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vld1.32 {d16[0]}, [r0:32]
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <4 x float>, ptr %B
  %tmp2 = load float, ptr %A
  %tmp3 = insertelement <4 x float> %tmp1, float %tmp2, i32 0
  ret <4 x float> %tmp3
}

%struct.__neon_int8x8x2_t = type { <8 x i8>,  <8 x i8> }
%struct.__neon_int16x4x2_t = type { <4 x i16>, <4 x i16> }
%struct.__neon_int32x2x2_t = type { <2 x i32>, <2 x i32> }
%struct.__neon_float32x2x2_t = type { <2 x float>, <2 x float> }

%struct.__neon_int16x8x2_t = type { <8 x i16>, <8 x i16> }
%struct.__neon_int32x4x2_t = type { <4 x i32>, <4 x i32> }
%struct.__neon_float32x4x2_t = type { <4 x float>, <4 x float> }

;Check the alignment value.  Max for this instruction is 16 bits:
define <8 x i8> @vld2lanei8(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld2lanei8:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vorr d17, d16, d16
; CHECK-NEXT:    vld2.8 {d16[1], d17[1]}, [r0:16]
; CHECK-NEXT:    vadd.i8 d16, d16, d17
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <8 x i8>, ptr %B
  %tmp2 = call %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2lane.v8i8.p0(ptr %A, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1, i32 4)
  %tmp3 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 1
  %tmp5 = add <8 x i8> %tmp3, %tmp4
  ret <8 x i8> %tmp5
}

;Check the alignment value.  Max for this instruction is 32 bits:
define <4 x i16> @vld2lanei16(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld2lanei16:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vorr d17, d16, d16
; CHECK-NEXT:    vld2.16 {d16[1], d17[1]}, [r0:32]
; CHECK-NEXT:    vadd.i16 d16, d16, d17
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <4 x i16>, ptr %B
  %tmp2 = call %struct.__neon_int16x4x2_t @llvm.arm.neon.vld2lane.v4i16.p0(ptr %A, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1, i32 8)
  %tmp3 = extractvalue %struct.__neon_int16x4x2_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int16x4x2_t %tmp2, 1
  %tmp5 = add <4 x i16> %tmp3, %tmp4
  ret <4 x i16> %tmp5
}

define <2 x i32> @vld2lanei32(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld2lanei32:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vorr d17, d16, d16
; CHECK-NEXT:    vld2.32 {d16[1], d17[1]}, [r0]
; CHECK-NEXT:    vadd.i32 d16, d16, d17
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <2 x i32>, ptr %B
  %tmp2 = call %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2lane.v2i32.p0(ptr %A, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1, i32 1)
  %tmp3 = extractvalue %struct.__neon_int32x2x2_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int32x2x2_t %tmp2, 1
  %tmp5 = add <2 x i32> %tmp3, %tmp4
  ret <2 x i32> %tmp5
}

;Check for a post-increment updating load.
define <2 x i32> @vld2lanei32_update(ptr %ptr, ptr %B) nounwind {
; DEFAULT-LABEL: vld2lanei32_update:
; DEFAULT:       @ %bb.0:
; DEFAULT-NEXT:    vldr d16, [r1]
; DEFAULT-NEXT:    ldr r3, [r0]
; DEFAULT-NEXT:    vorr d17, d16, d16
; DEFAULT-NEXT:    vld2.32 {d16[1], d17[1]}, [r3]!
; DEFAULT-NEXT:    vadd.i32 d16, d16, d17
; DEFAULT-NEXT:    str r3, [r0]
; DEFAULT-NEXT:    vmov r2, r1, d16
; DEFAULT-NEXT:    mov r0, r2
; DEFAULT-NEXT:    mov pc, lr
;
; BASIC-LABEL: vld2lanei32_update:
; BASIC:       @ %bb.0:
; BASIC-NEXT:    mov r2, r1
; BASIC-NEXT:    mov r1, r0
; BASIC-NEXT:    vldr d16, [r2]
; BASIC-NEXT:    ldr r0, [r0]
; BASIC-NEXT:    vorr d17, d16, d16
; BASIC-NEXT:    vld2.32 {d16[1], d17[1]}, [r0]!
; BASIC-NEXT:    vadd.i32 d16, d16, d17
; BASIC-NEXT:    str r0, [r1]
; BASIC-NEXT:    vmov r2, r3, d16
; BASIC-NEXT:    mov r0, r2
; BASIC-NEXT:    mov r1, r3
; BASIC-NEXT:    mov pc, lr
  %A = load ptr, ptr %ptr
  %tmp1 = load <2 x i32>, ptr %B
  %tmp2 = call %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2lane.v2i32.p0(ptr %A, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1, i32 1)
  %tmp3 = extractvalue %struct.__neon_int32x2x2_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int32x2x2_t %tmp2, 1
  %tmp5 = add <2 x i32> %tmp3, %tmp4
  %tmp6 = getelementptr i32, ptr %A, i32 2
  store ptr %tmp6, ptr %ptr
  ret <2 x i32> %tmp5
}

define <2 x i32> @vld2lanei32_odd_update(ptr %ptr, ptr %B) nounwind {
; DEFAULT-LABEL: vld2lanei32_odd_update:
; DEFAULT:       @ %bb.0:
; DEFAULT-NEXT:    vldr d16, [r1]
; DEFAULT-NEXT:    mov r1, #12
; DEFAULT-NEXT:    ldr r3, [r0]
; DEFAULT-NEXT:    vorr d17, d16, d16
; DEFAULT-NEXT:    vld2.32 {d16[1], d17[1]}, [r3], r1
; DEFAULT-NEXT:    vadd.i32 d16, d16, d17
; DEFAULT-NEXT:    str r3, [r0]
; DEFAULT-NEXT:    vmov r2, r1, d16
; DEFAULT-NEXT:    mov r0, r2
; DEFAULT-NEXT:    mov pc, lr
;
; BASIC-LABEL: vld2lanei32_odd_update:
; BASIC:       @ %bb.0:
; BASIC-NEXT:    mov r2, r1
; BASIC-NEXT:    mov r1, r0
; BASIC-NEXT:    vldr d16, [r2]
; BASIC-NEXT:    mov r2, #12
; BASIC-NEXT:    ldr r0, [r0]
; BASIC-NEXT:    vorr d17, d16, d16
; BASIC-NEXT:    vld2.32 {d16[1], d17[1]}, [r0], r2
; BASIC-NEXT:    vadd.i32 d16, d16, d17
; BASIC-NEXT:    str r0, [r1]
; BASIC-NEXT:    vmov r2, r3, d16
; BASIC-NEXT:    mov r0, r2
; BASIC-NEXT:    mov r1, r3
; BASIC-NEXT:    mov pc, lr
  %A = load ptr, ptr %ptr
  %tmp1 = load <2 x i32>, ptr %B
  %tmp2 = call %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2lane.v2i32.p0(ptr %A, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1, i32 1)
  %tmp3 = extractvalue %struct.__neon_int32x2x2_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int32x2x2_t %tmp2, 1
  %tmp5 = add <2 x i32> %tmp3, %tmp4
  %tmp6 = getelementptr i32, ptr %A, i32 3
  store ptr %tmp6, ptr %ptr
  ret <2 x i32> %tmp5
}

define <2 x float> @vld2lanef(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld2lanef:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vorr d17, d16, d16
; CHECK-NEXT:    vld2.32 {d16[1], d17[1]}, [r0]
; CHECK-NEXT:    vadd.f32 d16, d16, d17
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <2 x float>, ptr %B
  %tmp2 = call %struct.__neon_float32x2x2_t @llvm.arm.neon.vld2lane.v2f32.p0(ptr %A, <2 x float> %tmp1, <2 x float> %tmp1, i32 1, i32 1)
  %tmp3 = extractvalue %struct.__neon_float32x2x2_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_float32x2x2_t %tmp2, 1
  %tmp5 = fadd <2 x float> %tmp3, %tmp4
  ret <2 x float> %tmp5
}

;Check the (default) alignment.
define <8 x i16> @vld2laneQi16(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld2laneQi16:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vorr q9, q8, q8
; CHECK-NEXT:    vld2.16 {d17[1], d19[1]}, [r0]
; CHECK-NEXT:    vadd.i16 q8, q8, q9
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <8 x i16>, ptr %B
  %tmp2 = call %struct.__neon_int16x8x2_t @llvm.arm.neon.vld2lane.v8i16.p0(ptr %A, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 5, i32 1)
  %tmp3 = extractvalue %struct.__neon_int16x8x2_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int16x8x2_t %tmp2, 1
  %tmp5 = add <8 x i16> %tmp3, %tmp4
  ret <8 x i16> %tmp5
}

;Check the alignment value.  Max for this instruction is 64 bits:
define <4 x i32> @vld2laneQi32(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld2laneQi32:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vorr q9, q8, q8
; CHECK-NEXT:    vld2.32 {d17[0], d19[0]}, [r0:64]
; CHECK-NEXT:    vadd.i32 q8, q8, q9
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <4 x i32>, ptr %B
  %tmp2 = call %struct.__neon_int32x4x2_t @llvm.arm.neon.vld2lane.v4i32.p0(ptr %A, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 2, i32 16)
  %tmp3 = extractvalue %struct.__neon_int32x4x2_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int32x4x2_t %tmp2, 1
  %tmp5 = add <4 x i32> %tmp3, %tmp4
  ret <4 x i32> %tmp5
}

define <4 x float> @vld2laneQf(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld2laneQf:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vorr q9, q8, q8
; CHECK-NEXT:    vld2.32 {d16[1], d18[1]}, [r0]
; CHECK-NEXT:    vadd.f32 q8, q8, q9
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <4 x float>, ptr %B
  %tmp2 = call %struct.__neon_float32x4x2_t @llvm.arm.neon.vld2lane.v4f32.p0(ptr %A, <4 x float> %tmp1, <4 x float> %tmp1, i32 1, i32 1)
  %tmp3 = extractvalue %struct.__neon_float32x4x2_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_float32x4x2_t %tmp2, 1
  %tmp5 = fadd <4 x float> %tmp3, %tmp4
  ret <4 x float> %tmp5
}

declare %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2lane.v8i8.p0(ptr, <8 x i8>, <8 x i8>, i32, i32) nounwind readonly
declare %struct.__neon_int16x4x2_t @llvm.arm.neon.vld2lane.v4i16.p0(ptr, <4 x i16>, <4 x i16>, i32, i32) nounwind readonly
declare %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2lane.v2i32.p0(ptr, <2 x i32>, <2 x i32>, i32, i32) nounwind readonly
declare %struct.__neon_float32x2x2_t @llvm.arm.neon.vld2lane.v2f32.p0(ptr, <2 x float>, <2 x float>, i32, i32) nounwind readonly

declare %struct.__neon_int16x8x2_t @llvm.arm.neon.vld2lane.v8i16.p0(ptr, <8 x i16>, <8 x i16>, i32, i32) nounwind readonly
declare %struct.__neon_int32x4x2_t @llvm.arm.neon.vld2lane.v4i32.p0(ptr, <4 x i32>, <4 x i32>, i32, i32) nounwind readonly
declare %struct.__neon_float32x4x2_t @llvm.arm.neon.vld2lane.v4f32.p0(ptr, <4 x float>, <4 x float>, i32, i32) nounwind readonly

%struct.__neon_int8x8x3_t = type { <8 x i8>,  <8 x i8>,  <8 x i8> }
%struct.__neon_int16x4x3_t = type { <4 x i16>, <4 x i16>, <4 x i16> }
%struct.__neon_int32x2x3_t = type { <2 x i32>, <2 x i32>, <2 x i32> }
%struct.__neon_float32x2x3_t = type { <2 x float>, <2 x float>, <2 x float> }

%struct.__neon_int16x8x3_t = type { <8 x i16>, <8 x i16>, <8 x i16> }
%struct.__neon_int32x4x3_t = type { <4 x i32>, <4 x i32>, <4 x i32> }
%struct.__neon_float32x4x3_t = type { <4 x float>, <4 x float>, <4 x float> }

define <8 x i8> @vld3lanei8(ptr %A, ptr %B) nounwind {
; DEFAULT-LABEL: vld3lanei8:
; DEFAULT:       @ %bb.0:
; DEFAULT-NEXT:    vldr d16, [r1]
; DEFAULT-NEXT:    vorr d17, d16, d16
; DEFAULT-NEXT:    vorr d18, d16, d16
; DEFAULT-NEXT:    vld3.8 {d16[1], d17[1], d18[1]}, [r0]
; DEFAULT-NEXT:    vadd.i8 d20, d16, d17
; DEFAULT-NEXT:    vadd.i8 d16, d18, d20
; DEFAULT-NEXT:    vmov r0, r1, d16
; DEFAULT-NEXT:    mov pc, lr
;
; BASIC-LABEL: vld3lanei8:
; BASIC:       @ %bb.0:
; BASIC-NEXT:    vldr d18, [r1]
; BASIC-NEXT:    vorr d19, d18, d18
; BASIC-NEXT:    vorr d20, d18, d18
; BASIC-NEXT:    vld3.8 {d18[1], d19[1], d20[1]}, [r0]
; BASIC-NEXT:    vadd.i8 d16, d18, d19
; BASIC-NEXT:    vadd.i8 d16, d20, d16
; BASIC-NEXT:    vmov r0, r1, d16
; BASIC-NEXT:    mov pc, lr
  %tmp1 = load <8 x i8>, ptr %B
  %tmp2 = call %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3lane.v8i8.p0(ptr %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1, i32 1)
  %tmp3 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 1
  %tmp5 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 2
  %tmp6 = add <8 x i8> %tmp3, %tmp4
  %tmp7 = add <8 x i8> %tmp5, %tmp6
  ret <8 x i8> %tmp7
}

;Check the (default) alignment value.  VLD3 does not support alignment.
define <4 x i16> @vld3lanei16(ptr %A, ptr %B) nounwind {
; DEFAULT-LABEL: vld3lanei16:
; DEFAULT:       @ %bb.0:
; DEFAULT-NEXT:    vldr d16, [r1]
; DEFAULT-NEXT:    vorr d17, d16, d16
; DEFAULT-NEXT:    vorr d18, d16, d16
; DEFAULT-NEXT:    vld3.16 {d16[1], d17[1], d18[1]}, [r0]
; DEFAULT-NEXT:    vadd.i16 d20, d16, d17
; DEFAULT-NEXT:    vadd.i16 d16, d18, d20
; DEFAULT-NEXT:    vmov r0, r1, d16
; DEFAULT-NEXT:    mov pc, lr
;
; BASIC-LABEL: vld3lanei16:
; BASIC:       @ %bb.0:
; BASIC-NEXT:    vldr d18, [r1]
; BASIC-NEXT:    vorr d19, d18, d18
; BASIC-NEXT:    vorr d20, d18, d18
; BASIC-NEXT:    vld3.16 {d18[1], d19[1], d20[1]}, [r0]
; BASIC-NEXT:    vadd.i16 d16, d18, d19
; BASIC-NEXT:    vadd.i16 d16, d20, d16
; BASIC-NEXT:    vmov r0, r1, d16
; BASIC-NEXT:    mov pc, lr
  %tmp1 = load <4 x i16>, ptr %B
  %tmp2 = call %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3lane.v4i16.p0(ptr %A, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1, i32 8)
  %tmp3 = extractvalue %struct.__neon_int16x4x3_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int16x4x3_t %tmp2, 1
  %tmp5 = extractvalue %struct.__neon_int16x4x3_t %tmp2, 2
  %tmp6 = add <4 x i16> %tmp3, %tmp4
  %tmp7 = add <4 x i16> %tmp5, %tmp6
  ret <4 x i16> %tmp7
}

define <2 x i32> @vld3lanei32(ptr %A, ptr %B) nounwind {
; DEFAULT-LABEL: vld3lanei32:
; DEFAULT:       @ %bb.0:
; DEFAULT-NEXT:    vldr d16, [r1]
; DEFAULT-NEXT:    vorr d17, d16, d16
; DEFAULT-NEXT:    vorr d18, d16, d16
; DEFAULT-NEXT:    vld3.32 {d16[1], d17[1], d18[1]}, [r0]
; DEFAULT-NEXT:    vadd.i32 d20, d16, d17
; DEFAULT-NEXT:    vadd.i32 d16, d18, d20
; DEFAULT-NEXT:    vmov r0, r1, d16
; DEFAULT-NEXT:    mov pc, lr
;
; BASIC-LABEL: vld3lanei32:
; BASIC:       @ %bb.0:
; BASIC-NEXT:    vldr d18, [r1]
; BASIC-NEXT:    vorr d19, d18, d18
; BASIC-NEXT:    vorr d20, d18, d18
; BASIC-NEXT:    vld3.32 {d18[1], d19[1], d20[1]}, [r0]
; BASIC-NEXT:    vadd.i32 d16, d18, d19
; BASIC-NEXT:    vadd.i32 d16, d20, d16
; BASIC-NEXT:    vmov r0, r1, d16
; BASIC-NEXT:    mov pc, lr
  %tmp1 = load <2 x i32>, ptr %B
  %tmp2 = call %struct.__neon_int32x2x3_t @llvm.arm.neon.vld3lane.v2i32.p0(ptr %A, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1, i32 1)
  %tmp3 = extractvalue %struct.__neon_int32x2x3_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int32x2x3_t %tmp2, 1
  %tmp5 = extractvalue %struct.__neon_int32x2x3_t %tmp2, 2
  %tmp6 = add <2 x i32> %tmp3, %tmp4
  %tmp7 = add <2 x i32> %tmp5, %tmp6
  ret <2 x i32> %tmp7
}

define <2 x float> @vld3lanef(ptr %A, ptr %B) nounwind {
; DEFAULT-LABEL: vld3lanef:
; DEFAULT:       @ %bb.0:
; DEFAULT-NEXT:    vldr d16, [r1]
; DEFAULT-NEXT:    vorr d17, d16, d16
; DEFAULT-NEXT:    vorr d18, d16, d16
; DEFAULT-NEXT:    vld3.32 {d16[1], d17[1], d18[1]}, [r0]
; DEFAULT-NEXT:    vadd.f32 d20, d16, d17
; DEFAULT-NEXT:    vadd.f32 d16, d18, d20
; DEFAULT-NEXT:    vmov r0, r1, d16
; DEFAULT-NEXT:    mov pc, lr
;
; BASIC-LABEL: vld3lanef:
; BASIC:       @ %bb.0:
; BASIC-NEXT:    vldr d18, [r1]
; BASIC-NEXT:    vorr d19, d18, d18
; BASIC-NEXT:    vorr d20, d18, d18
; BASIC-NEXT:    vld3.32 {d18[1], d19[1], d20[1]}, [r0]
; BASIC-NEXT:    vadd.f32 d16, d18, d19
; BASIC-NEXT:    vadd.f32 d16, d20, d16
; BASIC-NEXT:    vmov r0, r1, d16
; BASIC-NEXT:    mov pc, lr
  %tmp1 = load <2 x float>, ptr %B
  %tmp2 = call %struct.__neon_float32x2x3_t @llvm.arm.neon.vld3lane.v2f32.p0(ptr %A, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, i32 1, i32 1)
  %tmp3 = extractvalue %struct.__neon_float32x2x3_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_float32x2x3_t %tmp2, 1
  %tmp5 = extractvalue %struct.__neon_float32x2x3_t %tmp2, 2
  %tmp6 = fadd <2 x float> %tmp3, %tmp4
  %tmp7 = fadd <2 x float> %tmp5, %tmp6
  ret <2 x float> %tmp7
}

;Check the (default) alignment value.  VLD3 does not support alignment.
define <8 x i16> @vld3laneQi16(ptr %A, ptr %B) nounwind {
; DEFAULT-LABEL: vld3laneQi16:
; DEFAULT:       @ %bb.0:
; DEFAULT-NEXT:    vld1.64 {d16, d17}, [r1]
; DEFAULT-NEXT:    vorr q9, q8, q8
; DEFAULT-NEXT:    vorr q10, q8, q8
; DEFAULT-NEXT:    vld3.16 {d16[1], d18[1], d20[1]}, [r0]
; DEFAULT-NEXT:    vadd.i16 q12, q8, q9
; DEFAULT-NEXT:    vadd.i16 q8, q10, q12
; DEFAULT-NEXT:    vmov r0, r1, d16
; DEFAULT-NEXT:    vmov r2, r3, d17
; DEFAULT-NEXT:    mov pc, lr
;
; BASIC-LABEL: vld3laneQi16:
; BASIC:       @ %bb.0:
; BASIC-NEXT:    vld1.64 {d18, d19}, [r1]
; BASIC-NEXT:    vorr q10, q9, q9
; BASIC-NEXT:    vorr q11, q9, q9
; BASIC-NEXT:    vld3.16 {d18[1], d20[1], d22[1]}, [r0]
; BASIC-NEXT:    vadd.i16 q8, q9, q10
; BASIC-NEXT:    vadd.i16 q8, q11, q8
; BASIC-NEXT:    vmov r0, r1, d16
; BASIC-NEXT:    vmov r2, r3, d17
; BASIC-NEXT:    mov pc, lr
  %tmp1 = load <8 x i16>, ptr %B
  %tmp2 = call %struct.__neon_int16x8x3_t @llvm.arm.neon.vld3lane.v8i16.p0(ptr %A, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 1, i32 8)
  %tmp3 = extractvalue %struct.__neon_int16x8x3_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int16x8x3_t %tmp2, 1
  %tmp5 = extractvalue %struct.__neon_int16x8x3_t %tmp2, 2
  %tmp6 = add <8 x i16> %tmp3, %tmp4
  %tmp7 = add <8 x i16> %tmp5, %tmp6
  ret <8 x i16> %tmp7
}

;Check for a post-increment updating load with register increment.
define <8 x i16> @vld3laneQi16_update(ptr %ptr, ptr %B, i32 %inc) nounwind {
; DEFAULT-LABEL: vld3laneQi16_update:
; DEFAULT:       @ %bb.0:
; DEFAULT-NEXT:    .save {r11, lr}
; DEFAULT-NEXT:    push {r11, lr}
; DEFAULT-NEXT:    vld1.64 {d16, d17}, [r1]
; DEFAULT-NEXT:    lsl r1, r2, #1
; DEFAULT-NEXT:    vorr q9, q8, q8
; DEFAULT-NEXT:    ldr lr, [r0]
; DEFAULT-NEXT:    vorr q10, q8, q8
; DEFAULT-NEXT:    vld3.16 {d16[1], d18[1], d20[1]}, [lr], r1
; DEFAULT-NEXT:    vadd.i16 q12, q8, q9
; DEFAULT-NEXT:    vadd.i16 q8, q10, q12
; DEFAULT-NEXT:    str lr, [r0]
; DEFAULT-NEXT:    vmov r12, r1, d16
; DEFAULT-NEXT:    vmov r2, r3, d17
; DEFAULT-NEXT:    mov r0, r12
; DEFAULT-NEXT:    pop {r11, lr}
; DEFAULT-NEXT:    mov pc, lr
;
; BASIC-LABEL: vld3laneQi16_update:
; BASIC:       @ %bb.0:
; BASIC-NEXT:    .save {r11, lr}
; BASIC-NEXT:    push {r11, lr}
; BASIC-NEXT:    vld1.64 {d18, d19}, [r1]
; BASIC-NEXT:    mov r3, r0
; BASIC-NEXT:    vorr q10, q9, q9
; BASIC-NEXT:    lsl r1, r2, #1
; BASIC-NEXT:    ldr r0, [r0]
; BASIC-NEXT:    vorr q11, q9, q9
; BASIC-NEXT:    vld3.16 {d18[1], d20[1], d22[1]}, [r0], r1
; BASIC-NEXT:    vadd.i16 q8, q9, q10
; BASIC-NEXT:    vadd.i16 q8, q11, q8
; BASIC-NEXT:    str r0, [r3]
; BASIC-NEXT:    vmov r1, lr, d16
; BASIC-NEXT:    vmov r2, r12, d17
; BASIC-NEXT:    mov r0, r1
; BASIC-NEXT:    mov r1, lr
; BASIC-NEXT:    mov r3, r12
; BASIC-NEXT:    pop {r11, lr}
; BASIC-NEXT:    mov pc, lr
  %A = load ptr, ptr %ptr
  %tmp1 = load <8 x i16>, ptr %B
  %tmp2 = call %struct.__neon_int16x8x3_t @llvm.arm.neon.vld3lane.v8i16.p0(ptr %A, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 1, i32 8)
  %tmp3 = extractvalue %struct.__neon_int16x8x3_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int16x8x3_t %tmp2, 1
  %tmp5 = extractvalue %struct.__neon_int16x8x3_t %tmp2, 2
  %tmp6 = add <8 x i16> %tmp3, %tmp4
  %tmp7 = add <8 x i16> %tmp5, %tmp6
  %tmp8 = getelementptr i16, ptr %A, i32 %inc
  store ptr %tmp8, ptr %ptr
  ret <8 x i16> %tmp7
}

define <4 x i32> @vld3laneQi32(ptr %A, ptr %B) nounwind {
; DEFAULT-LABEL: vld3laneQi32:
; DEFAULT:       @ %bb.0:
; DEFAULT-NEXT:    vld1.64 {d16, d17}, [r1]
; DEFAULT-NEXT:    vorr q9, q8, q8
; DEFAULT-NEXT:    vorr q10, q8, q8
; DEFAULT-NEXT:    vld3.32 {d17[1], d19[1], d21[1]}, [r0]
; DEFAULT-NEXT:    vadd.i32 q12, q8, q9
; DEFAULT-NEXT:    vadd.i32 q8, q10, q12
; DEFAULT-NEXT:    vmov r0, r1, d16
; DEFAULT-NEXT:    vmov r2, r3, d17
; DEFAULT-NEXT:    mov pc, lr
;
; BASIC-LABEL: vld3laneQi32:
; BASIC:       @ %bb.0:
; BASIC-NEXT:    vld1.64 {d18, d19}, [r1]
; BASIC-NEXT:    vorr q10, q9, q9
; BASIC-NEXT:    vorr q11, q9, q9
; BASIC-NEXT:    vld3.32 {d19[1], d21[1], d23[1]}, [r0]
; BASIC-NEXT:    vadd.i32 q8, q9, q10
; BASIC-NEXT:    vadd.i32 q8, q11, q8
; BASIC-NEXT:    vmov r0, r1, d16
; BASIC-NEXT:    vmov r2, r3, d17
; BASIC-NEXT:    mov pc, lr
  %tmp1 = load <4 x i32>, ptr %B
  %tmp2 = call %struct.__neon_int32x4x3_t @llvm.arm.neon.vld3lane.v4i32.p0(ptr %A, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 3, i32 1)
  %tmp3 = extractvalue %struct.__neon_int32x4x3_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int32x4x3_t %tmp2, 1
  %tmp5 = extractvalue %struct.__neon_int32x4x3_t %tmp2, 2
  %tmp6 = add <4 x i32> %tmp3, %tmp4
  %tmp7 = add <4 x i32> %tmp5, %tmp6
  ret <4 x i32> %tmp7
}

define <4 x float> @vld3laneQf(ptr %A, ptr %B) nounwind {
; DEFAULT-LABEL: vld3laneQf:
; DEFAULT:       @ %bb.0:
; DEFAULT-NEXT:    vld1.64 {d16, d17}, [r1]
; DEFAULT-NEXT:    vorr q9, q8, q8
; DEFAULT-NEXT:    vorr q10, q8, q8
; DEFAULT-NEXT:    vld3.32 {d16[1], d18[1], d20[1]}, [r0]
; DEFAULT-NEXT:    vadd.f32 q12, q8, q9
; DEFAULT-NEXT:    vadd.f32 q8, q10, q12
; DEFAULT-NEXT:    vmov r0, r1, d16
; DEFAULT-NEXT:    vmov r2, r3, d17
; DEFAULT-NEXT:    mov pc, lr
;
; BASIC-LABEL: vld3laneQf:
; BASIC:       @ %bb.0:
; BASIC-NEXT:    vld1.64 {d18, d19}, [r1]
; BASIC-NEXT:    vorr q10, q9, q9
; BASIC-NEXT:    vorr q11, q9, q9
; BASIC-NEXT:    vld3.32 {d18[1], d20[1], d22[1]}, [r0]
; BASIC-NEXT:    vadd.f32 q8, q9, q10
; BASIC-NEXT:    vadd.f32 q8, q11, q8
; BASIC-NEXT:    vmov r0, r1, d16
; BASIC-NEXT:    vmov r2, r3, d17
; BASIC-NEXT:    mov pc, lr
  %tmp1 = load <4 x float>, ptr %B
  %tmp2 = call %struct.__neon_float32x4x3_t @llvm.arm.neon.vld3lane.v4f32.p0(ptr %A, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, i32 1, i32 1)
  %tmp3 = extractvalue %struct.__neon_float32x4x3_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_float32x4x3_t %tmp2, 1
  %tmp5 = extractvalue %struct.__neon_float32x4x3_t %tmp2, 2
  %tmp6 = fadd <4 x float> %tmp3, %tmp4
  %tmp7 = fadd <4 x float> %tmp5, %tmp6
  ret <4 x float> %tmp7
}

declare %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3lane.v8i8.p0(ptr, <8 x i8>, <8 x i8>, <8 x i8>, i32, i32) nounwind readonly
declare %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3lane.v4i16.p0(ptr, <4 x i16>, <4 x i16>, <4 x i16>, i32, i32) nounwind readonly
declare %struct.__neon_int32x2x3_t @llvm.arm.neon.vld3lane.v2i32.p0(ptr, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32) nounwind readonly
declare %struct.__neon_float32x2x3_t @llvm.arm.neon.vld3lane.v2f32.p0(ptr, <2 x float>, <2 x float>, <2 x float>, i32, i32) nounwind readonly

declare %struct.__neon_int16x8x3_t @llvm.arm.neon.vld3lane.v8i16.p0(ptr, <8 x i16>, <8 x i16>, <8 x i16>, i32, i32) nounwind readonly
declare %struct.__neon_int32x4x3_t @llvm.arm.neon.vld3lane.v4i32.p0(ptr, <4 x i32>, <4 x i32>, <4 x i32>, i32, i32) nounwind readonly
declare %struct.__neon_float32x4x3_t @llvm.arm.neon.vld3lane.v4f32.p0(ptr, <4 x float>, <4 x float>, <4 x float>, i32, i32) nounwind readonly

%struct.__neon_int8x8x4_t = type { <8 x i8>,  <8 x i8>,  <8 x i8>,  <8 x i8> }
%struct.__neon_int16x4x4_t = type { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }
%struct.__neon_int32x2x4_t = type { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }
%struct.__neon_float32x2x4_t = type { <2 x float>, <2 x float>, <2 x float>, <2 x float> }

%struct.__neon_int16x8x4_t = type { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> }
%struct.__neon_int32x4x4_t = type { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }
%struct.__neon_float32x4x4_t = type { <4 x float>, <4 x float>, <4 x float>, <4 x float> }

;Check the alignment value.  Max for this instruction is 32 bits:
define <8 x i8> @vld4lanei8(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld4lanei8:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vorr d17, d16, d16
; CHECK-NEXT:    vorr d18, d16, d16
; CHECK-NEXT:    vorr d19, d16, d16
; CHECK-NEXT:    vld4.8 {d16[1], d17[1], d18[1], d19[1]}, [r0:32]
; CHECK-NEXT:    vadd.i8 d16, d16, d17
; CHECK-NEXT:    vadd.i8 d20, d18, d19
; CHECK-NEXT:    vadd.i8 d16, d16, d20
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <8 x i8>, ptr %B
  %tmp2 = call %struct.__neon_int8x8x4_t @llvm.arm.neon.vld4lane.v8i8.p0(ptr %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1, i32 8)
  %tmp3 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 1
  %tmp5 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 2
  %tmp6 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 3
  %tmp7 = add <8 x i8> %tmp3, %tmp4
  %tmp8 = add <8 x i8> %tmp5, %tmp6
  %tmp9 = add <8 x i8> %tmp7, %tmp8
  ret <8 x i8> %tmp9
}

;Check for a post-increment updating load.
define <8 x i8> @vld4lanei8_update(ptr %ptr, ptr %B) nounwind {
; DEFAULT-LABEL: vld4lanei8_update:
; DEFAULT:       @ %bb.0:
; DEFAULT-NEXT:    vldr d16, [r1]
; DEFAULT-NEXT:    vorr d17, d16, d16
; DEFAULT-NEXT:    ldr r3, [r0]
; DEFAULT-NEXT:    vorr d18, d16, d16
; DEFAULT-NEXT:    vorr d19, d16, d16
; DEFAULT-NEXT:    vld4.8 {d16[1], d17[1], d18[1], d19[1]}, [r3:32]!
; DEFAULT-NEXT:    vadd.i8 d16, d16, d17
; DEFAULT-NEXT:    vadd.i8 d20, d18, d19
; DEFAULT-NEXT:    str r3, [r0]
; DEFAULT-NEXT:    vadd.i8 d16, d16, d20
; DEFAULT-NEXT:    vmov r2, r1, d16
; DEFAULT-NEXT:    mov r0, r2
; DEFAULT-NEXT:    mov pc, lr
;
; BASIC-LABEL: vld4lanei8_update:
; BASIC:       @ %bb.0:
; BASIC-NEXT:    vldr d16, [r1]
; BASIC-NEXT:    mov r3, r0
; BASIC-NEXT:    vorr d17, d16, d16
; BASIC-NEXT:    ldr r0, [r0]
; BASIC-NEXT:    vorr d18, d16, d16
; BASIC-NEXT:    vorr d19, d16, d16
; BASIC-NEXT:    vld4.8 {d16[1], d17[1], d18[1], d19[1]}, [r0:32]!
; BASIC-NEXT:    vadd.i8 d16, d16, d17
; BASIC-NEXT:    vadd.i8 d20, d18, d19
; BASIC-NEXT:    str r0, [r3]
; BASIC-NEXT:    vadd.i8 d16, d16, d20
; BASIC-NEXT:    vmov r1, r2, d16
; BASIC-NEXT:    mov r0, r1
; BASIC-NEXT:    mov r1, r2
; BASIC-NEXT:    mov pc, lr
  %A = load ptr, ptr %ptr
  %tmp1 = load <8 x i8>, ptr %B
  %tmp2 = call %struct.__neon_int8x8x4_t @llvm.arm.neon.vld4lane.v8i8.p0(ptr %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1, i32 8)
  %tmp3 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 1
  %tmp5 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 2
  %tmp6 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 3
  %tmp7 = add <8 x i8> %tmp3, %tmp4
  %tmp8 = add <8 x i8> %tmp5, %tmp6
  %tmp9 = add <8 x i8> %tmp7, %tmp8
  %tmp10 = getelementptr i8, ptr %A, i32 4
  store ptr %tmp10, ptr %ptr
  ret <8 x i8> %tmp9
}

;Check that a power-of-two alignment smaller than the total size of the memory
;being loaded is ignored.
define <4 x i16> @vld4lanei16(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld4lanei16:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vorr d17, d16, d16
; CHECK-NEXT:    vorr d18, d16, d16
; CHECK-NEXT:    vorr d19, d16, d16
; CHECK-NEXT:    vld4.16 {d16[1], d17[1], d18[1], d19[1]}, [r0]
; CHECK-NEXT:    vadd.i16 d16, d16, d17
; CHECK-NEXT:    vadd.i16 d20, d18, d19
; CHECK-NEXT:    vadd.i16 d16, d16, d20
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <4 x i16>, ptr %B
  %tmp2 = call %struct.__neon_int16x4x4_t @llvm.arm.neon.vld4lane.v4i16.p0(ptr %A, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1, i32 4)
  %tmp3 = extractvalue %struct.__neon_int16x4x4_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int16x4x4_t %tmp2, 1
  %tmp5 = extractvalue %struct.__neon_int16x4x4_t %tmp2, 2
  %tmp6 = extractvalue %struct.__neon_int16x4x4_t %tmp2, 3
  %tmp7 = add <4 x i16> %tmp3, %tmp4
  %tmp8 = add <4 x i16> %tmp5, %tmp6
  %tmp9 = add <4 x i16> %tmp7, %tmp8
  ret <4 x i16> %tmp9
}

;Check the alignment value.  An 8-byte alignment is allowed here even though
;it is smaller than the total size of the memory being loaded.
define <2 x i32> @vld4lanei32(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld4lanei32:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vorr d17, d16, d16
; CHECK-NEXT:    vorr d18, d16, d16
; CHECK-NEXT:    vorr d19, d16, d16
; CHECK-NEXT:    vld4.32 {d16[1], d17[1], d18[1], d19[1]}, [r0:64]
; CHECK-NEXT:    vadd.i32 d16, d16, d17
; CHECK-NEXT:    vadd.i32 d20, d18, d19
; CHECK-NEXT:    vadd.i32 d16, d16, d20
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <2 x i32>, ptr %B
  %tmp2 = call %struct.__neon_int32x2x4_t @llvm.arm.neon.vld4lane.v2i32.p0(ptr %A, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1, i32 8)
  %tmp3 = extractvalue %struct.__neon_int32x2x4_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int32x2x4_t %tmp2, 1
  %tmp5 = extractvalue %struct.__neon_int32x2x4_t %tmp2, 2
  %tmp6 = extractvalue %struct.__neon_int32x2x4_t %tmp2, 3
  %tmp7 = add <2 x i32> %tmp3, %tmp4
  %tmp8 = add <2 x i32> %tmp5, %tmp6
  %tmp9 = add <2 x i32> %tmp7, %tmp8
  ret <2 x i32> %tmp9
}

define <2 x float> @vld4lanef(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld4lanef:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vorr d17, d16, d16
; CHECK-NEXT:    vorr d18, d16, d16
; CHECK-NEXT:    vorr d19, d16, d16
; CHECK-NEXT:    vld4.32 {d16[1], d17[1], d18[1], d19[1]}, [r0]
; CHECK-NEXT:    vadd.f32 d16, d16, d17
; CHECK-NEXT:    vadd.f32 d20, d18, d19
; CHECK-NEXT:    vadd.f32 d16, d16, d20
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <2 x float>, ptr %B
  %tmp2 = call %struct.__neon_float32x2x4_t @llvm.arm.neon.vld4lane.v2f32.p0(ptr %A, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, i32 1, i32 1)
  %tmp3 = extractvalue %struct.__neon_float32x2x4_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_float32x2x4_t %tmp2, 1
  %tmp5 = extractvalue %struct.__neon_float32x2x4_t %tmp2, 2
  %tmp6 = extractvalue %struct.__neon_float32x2x4_t %tmp2, 3
  %tmp7 = fadd <2 x float> %tmp3, %tmp4
  %tmp8 = fadd <2 x float> %tmp5, %tmp6
  %tmp9 = fadd <2 x float> %tmp7, %tmp8
  ret <2 x float> %tmp9
}

;Check the alignment value.  Max for this instruction is 64 bits:
define <8 x i16> @vld4laneQi16(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld4laneQi16:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vorr q9, q8, q8
; CHECK-NEXT:    vorr q10, q8, q8
; CHECK-NEXT:    vorr q11, q8, q8
; CHECK-NEXT:    vld4.16 {d16[1], d18[1], d20[1], d22[1]}, [r0:64]
; CHECK-NEXT:    vadd.i16 q8, q8, q9
; CHECK-NEXT:    vadd.i16 q12, q10, q11
; CHECK-NEXT:    vadd.i16 q8, q8, q12
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <8 x i16>, ptr %B
  %tmp2 = call %struct.__neon_int16x8x4_t @llvm.arm.neon.vld4lane.v8i16.p0(ptr %A, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 1, i32 16)
  %tmp3 = extractvalue %struct.__neon_int16x8x4_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int16x8x4_t %tmp2, 1
  %tmp5 = extractvalue %struct.__neon_int16x8x4_t %tmp2, 2
  %tmp6 = extractvalue %struct.__neon_int16x8x4_t %tmp2, 3
  %tmp7 = add <8 x i16> %tmp3, %tmp4
  %tmp8 = add <8 x i16> %tmp5, %tmp6
  %tmp9 = add <8 x i16> %tmp7, %tmp8
  ret <8 x i16> %tmp9
}

;Check the (default) alignment.
define <4 x i32> @vld4laneQi32(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld4laneQi32:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vorr q9, q8, q8
; CHECK-NEXT:    vorr q10, q8, q8
; CHECK-NEXT:    vorr q11, q8, q8
; CHECK-NEXT:    vld4.32 {d17[0], d19[0], d21[0], d23[0]}, [r0]
; CHECK-NEXT:    vadd.i32 q8, q8, q9
; CHECK-NEXT:    vadd.i32 q12, q10, q11
; CHECK-NEXT:    vadd.i32 q8, q8, q12
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <4 x i32>, ptr %B
  %tmp2 = call %struct.__neon_int32x4x4_t @llvm.arm.neon.vld4lane.v4i32.p0(ptr %A, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 2, i32 1)
  %tmp3 = extractvalue %struct.__neon_int32x4x4_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_int32x4x4_t %tmp2, 1
  %tmp5 = extractvalue %struct.__neon_int32x4x4_t %tmp2, 2
  %tmp6 = extractvalue %struct.__neon_int32x4x4_t %tmp2, 3
  %tmp7 = add <4 x i32> %tmp3, %tmp4
  %tmp8 = add <4 x i32> %tmp5, %tmp6
  %tmp9 = add <4 x i32> %tmp7, %tmp8
  ret <4 x i32> %tmp9
}

define <4 x float> @vld4laneQf(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vld4laneQf:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vorr q9, q8, q8
; CHECK-NEXT:    vorr q10, q8, q8
; CHECK-NEXT:    vorr q11, q8, q8
; CHECK-NEXT:    vld4.32 {d16[1], d18[1], d20[1], d22[1]}, [r0]
; CHECK-NEXT:    vadd.f32 q8, q8, q9
; CHECK-NEXT:    vadd.f32 q12, q10, q11
; CHECK-NEXT:    vadd.f32 q8, q8, q12
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <4 x float>, ptr %B
  %tmp2 = call %struct.__neon_float32x4x4_t @llvm.arm.neon.vld4lane.v4f32.p0(ptr %A, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, i32 1, i32 1)
  %tmp3 = extractvalue %struct.__neon_float32x4x4_t %tmp2, 0
  %tmp4 = extractvalue %struct.__neon_float32x4x4_t %tmp2, 1
  %tmp5 = extractvalue %struct.__neon_float32x4x4_t %tmp2, 2
  %tmp6 = extractvalue %struct.__neon_float32x4x4_t %tmp2, 3
  %tmp7 = fadd <4 x float> %tmp3, %tmp4
  %tmp8 = fadd <4 x float> %tmp5, %tmp6
  %tmp9 = fadd <4 x float> %tmp7, %tmp8
  ret <4 x float> %tmp9
}

declare %struct.__neon_int8x8x4_t @llvm.arm.neon.vld4lane.v8i8.p0(ptr, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, i32, i32) nounwind readonly
declare %struct.__neon_int16x4x4_t @llvm.arm.neon.vld4lane.v4i16.p0(ptr, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, i32, i32) nounwind readonly
declare %struct.__neon_int32x2x4_t @llvm.arm.neon.vld4lane.v2i32.p0(ptr, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32) nounwind readonly
declare %struct.__neon_float32x2x4_t @llvm.arm.neon.vld4lane.v2f32.p0(ptr, <2 x float>, <2 x float>, <2 x float>, <2 x float>, i32, i32) nounwind readonly

declare %struct.__neon_int16x8x4_t @llvm.arm.neon.vld4lane.v8i16.p0(ptr, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, i32, i32) nounwind readonly
declare %struct.__neon_int32x4x4_t @llvm.arm.neon.vld4lane.v4i32.p0(ptr, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i32, i32) nounwind readonly
declare %struct.__neon_float32x4x4_t @llvm.arm.neon.vld4lane.v4f32.p0(ptr, <4 x float>, <4 x float>, <4 x float>, <4 x float>, i32, i32) nounwind readonly

; Radar 8776599: If one of the operands to a QQQQ REG_SEQUENCE is a register
; in the QPR_VFP2 regclass, it needs to be copied to a QPR regclass because
; we don't currently have a QQQQ_VFP2 super-regclass.  (The "0" for the low
; part of %ins67 is supposed to be loaded by a VLDRS instruction in this test.)
define <8 x i16> @test_qqqq_regsequence_subreg([6 x i64] %b) nounwind {
; DEFAULT-LABEL: test_qqqq_regsequence_subreg:
; DEFAULT:       @ %bb.0:
; DEFAULT-NEXT:    add r0, sp, #24
; DEFAULT-NEXT:    vld1.32 {d21[0]}, [r0:32]
; DEFAULT-NEXT:    add r0, sp, #28
; DEFAULT-NEXT:    vmov.i32 d20, #0x0
; DEFAULT-NEXT:    vld1.32 {d21[1]}, [r0:32]
; DEFAULT-NEXT:    vld3.16 {d16[1], d18[1], d20[1]}, [r0]
; DEFAULT-NEXT:    vadd.i16 q12, q8, q9
; DEFAULT-NEXT:    vadd.i16 q8, q10, q12
; DEFAULT-NEXT:    vmov r0, r1, d16
; DEFAULT-NEXT:    vmov r2, r3, d17
; DEFAULT-NEXT:    mov pc, lr
;
; BASIC-LABEL: test_qqqq_regsequence_subreg:
; BASIC:       @ %bb.0:
; BASIC-NEXT:    add r0, sp, #24
; BASIC-NEXT:    vld1.32 {d23[0]}, [r0:32]
; BASIC-NEXT:    add r0, sp, #28
; BASIC-NEXT:    vmov.i32 d22, #0x0
; BASIC-NEXT:    vld1.32 {d23[1]}, [r0:32]
; BASIC-NEXT:    vld3.16 {d18[1], d20[1], d22[1]}, [r0]
; BASIC-NEXT:    vadd.i16 q8, q9, q10
; BASIC-NEXT:    vadd.i16 q8, q11, q8
; BASIC-NEXT:    vmov r0, r1, d16
; BASIC-NEXT:    vmov r2, r3, d17
; BASIC-NEXT:    mov pc, lr
  %tmp63 = extractvalue [6 x i64] %b, 5
  %tmp64 = zext i64 %tmp63 to i128
  %tmp65 = shl i128 %tmp64, 64
  %ins67 = or i128 %tmp65, 0
  %tmp78 = bitcast i128 %ins67 to <8 x i16>
  %vld3_lane = tail call %struct.__neon_int16x8x3_t @llvm.arm.neon.vld3lane.v8i16.p0(ptr undef, <8 x i16> undef, <8 x i16> undef, <8 x i16> %tmp78, i32 1, i32 2)
  %tmp3 = extractvalue %struct.__neon_int16x8x3_t %vld3_lane, 0
  %tmp4 = extractvalue %struct.__neon_int16x8x3_t %vld3_lane, 1
  %tmp5 = extractvalue %struct.__neon_int16x8x3_t %vld3_lane, 2
  %tmp6 = add <8 x i16> %tmp3, %tmp4
  %tmp7 = add <8 x i16> %tmp5, %tmp6
  ret <8 x i16> %tmp7
}

declare void @llvm.trap() nounwind
