; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s --mattr=+mve -o - | FileCheck %s

target triple = "thumbv8.1m.main-none-none-eabi"


; Expected to not transform
define arm_aapcs_vfpcc <2 x i8> @complex_add_v2i8(<2 x i8> %a, <2 x i8> %b) {
; CHECK-LABEL: complex_add_v2i8:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov r0, s0
; CHECK-NEXT:    vmov r1, s6
; CHECK-NEXT:    vmov r2, s4
; CHECK-NEXT:    add r0, r1
; CHECK-NEXT:    vmov r1, s2
; CHECK-NEXT:    subs r1, r2, r1
; CHECK-NEXT:    vmov q0[2], q0[0], r1, r0
; CHECK-NEXT:    bx lr
entry:
  %a.real = shufflevector <2 x i8> %a, <2 x i8> zeroinitializer, <1 x i32> <i32 0>
  %a.imag = shufflevector <2 x i8> %a, <2 x i8> zeroinitializer, <1 x i32> <i32 1>
  %b.real = shufflevector <2 x i8> %b, <2 x i8> zeroinitializer, <1 x i32> <i32 0>
  %b.imag = shufflevector <2 x i8> %b, <2 x i8> zeroinitializer, <1 x i32> <i32 1>
  %0 = sub <1 x i8> %b.real, %a.imag
  %1 = add <1 x i8> %b.imag, %a.real
  %interleaved.vec = shufflevector <1 x i8> %0, <1 x i8> %1, <2 x i32> <i32 0, i32 1>
  ret <2 x i8> %interleaved.vec
}

; Expected to not transform
define arm_aapcs_vfpcc <4 x i8> @complex_add_v4i8(<4 x i8> %a, <4 x i8> %b) {
; CHECK-LABEL: complex_add_v4i8:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vrev64.32 q2, q0
; CHECK-NEXT:    vmov r1, s6
; CHECK-NEXT:    vmov r0, s10
; CHECK-NEXT:    vrev64.32 q3, q1
; CHECK-NEXT:    vmov r2, s4
; CHECK-NEXT:    subs r0, r1, r0
; CHECK-NEXT:    vmov r1, s8
; CHECK-NEXT:    subs r1, r2, r1
; CHECK-NEXT:    vmov r2, s0
; CHECK-NEXT:    vmov q2[2], q2[0], r1, r0
; CHECK-NEXT:    vmov r0, s14
; CHECK-NEXT:    vmov r1, s2
; CHECK-NEXT:    add r0, r1
; CHECK-NEXT:    vmov r1, s12
; CHECK-NEXT:    add r1, r2
; CHECK-NEXT:    vmov q2[3], q2[1], r1, r0
; CHECK-NEXT:    vmov q0, q2
; CHECK-NEXT:    bx lr
entry:
  %a.real = shufflevector <4 x i8> %a, <4 x i8> zeroinitializer, <2 x i32> <i32 0, i32 2>
  %a.imag = shufflevector <4 x i8> %a, <4 x i8> zeroinitializer, <2 x i32> <i32 1, i32 3>
  %b.real = shufflevector <4 x i8> %b, <4 x i8> zeroinitializer, <2 x i32> <i32 0, i32 2>
  %b.imag = shufflevector <4 x i8> %b, <4 x i8> zeroinitializer, <2 x i32> <i32 1, i32 3>
  %0 = sub <2 x i8> %b.real, %a.imag
  %1 = add <2 x i8> %b.imag, %a.real
  %interleaved.vec = shufflevector <2 x i8> %0, <2 x i8> %1, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  ret <4 x i8> %interleaved.vec
}

; Expected to transform
define arm_aapcs_vfpcc <8 x i8> @complex_add_v8i8(<8 x i8> %a, <8 x i8> %b) {
; CHECK-LABEL: complex_add_v8i8:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vrev32.16 q2, q1
; CHECK-NEXT:    vadd.i32 q2, q2, q0
; CHECK-NEXT:    vrev32.16 q0, q0
; CHECK-NEXT:    vsub.i32 q0, q1, q0
; CHECK-NEXT:    vmovnt.i32 q0, q2
; CHECK-NEXT:    bx lr
entry:
  %a.real = shufflevector <8 x i8> %a, <8 x i8> zeroinitializer, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %a.imag = shufflevector <8 x i8> %a, <8 x i8> zeroinitializer, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %b.real = shufflevector <8 x i8> %b, <8 x i8> zeroinitializer, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %b.imag = shufflevector <8 x i8> %b, <8 x i8> zeroinitializer, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %0 = sub <4 x i8> %b.real, %a.imag
  %1 = add <4 x i8> %b.imag, %a.real
  %interleaved.vec = shufflevector <4 x i8> %0, <4 x i8> %1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  ret <8 x i8> %interleaved.vec
}

; Expected to transform
define arm_aapcs_vfpcc <16 x i8> @complex_add_v16i8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: complex_add_v16i8:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vcadd.i8 q0, q1, q0, #90
; CHECK-NEXT:    bx lr
entry:
  %a.real = shufflevector <16 x i8> %a, <16 x i8> zeroinitializer, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %a.imag = shufflevector <16 x i8> %a, <16 x i8> zeroinitializer, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %b.real = shufflevector <16 x i8> %b, <16 x i8> zeroinitializer, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %b.imag = shufflevector <16 x i8> %b, <16 x i8> zeroinitializer, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %0 = sub <8 x i8> %b.real, %a.imag
  %1 = add <8 x i8> %b.imag, %a.real
  %interleaved.vec = shufflevector <8 x i8> %0, <8 x i8> %1, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  ret <16 x i8> %interleaved.vec
}

; Expected to transform
define arm_aapcs_vfpcc <32 x i8> @complex_add_v32i8(<32 x i8> %a, <32 x i8> %b) {
; CHECK-LABEL: complex_add_v32i8:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vcadd.i8 q0, q2, q0, #90
; CHECK-NEXT:    vcadd.i8 q1, q3, q1, #90
; CHECK-NEXT:    bx lr
entry:
  %a.real = shufflevector <32 x i8> %a, <32 x i8> zeroinitializer, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %a.imag = shufflevector <32 x i8> %a, <32 x i8> zeroinitializer, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %b.real = shufflevector <32 x i8> %b, <32 x i8> zeroinitializer, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %b.imag = shufflevector <32 x i8> %b, <32 x i8> zeroinitializer, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %0 = sub <16 x i8> %b.real, %a.imag
  %1 = add <16 x i8> %b.imag, %a.real
  %interleaved.vec = shufflevector <16 x i8> %0, <16 x i8> %1, <32 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  ret <32 x i8> %interleaved.vec
}
