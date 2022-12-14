; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -O0 -mtriple=mipsel-linux-gnu -global-isel -mcpu=mips32r5 -mattr=+msa,+fp64,+nan2008 -verify-machineinstrs %s -o -| FileCheck %s -check-prefixes=P5600

declare <16 x i8> @llvm.mips.div.s.b(<16 x i8>, <16 x i8>)
define void @sdiv_v16i8_builtin(ptr %a, ptr %b, ptr %c) {
; P5600-LABEL: sdiv_v16i8_builtin:
; P5600:       # %bb.0: # %entry
; P5600-NEXT:    ld.b $w0, 0($4)
; P5600-NEXT:    ld.b $w1, 0($5)
; P5600-NEXT:    div_s.b $w0, $w0, $w1
; P5600-NEXT:    st.b $w0, 0($6)
; P5600-NEXT:    jr $ra
; P5600-NEXT:    nop
entry:
  %0 = load <16 x i8>, ptr %a, align 16
  %1 = load <16 x i8>, ptr %b, align 16
  %2 = tail call <16 x i8> @llvm.mips.div.s.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr %c, align 16
  ret void
}

declare <8 x i16> @llvm.mips.div.s.h(<8 x i16>, <8 x i16>)
define void @sdiv_v8i16_builtin(ptr %a, ptr %b, ptr %c) {
; P5600-LABEL: sdiv_v8i16_builtin:
; P5600:       # %bb.0: # %entry
; P5600-NEXT:    ld.h $w0, 0($4)
; P5600-NEXT:    ld.h $w1, 0($5)
; P5600-NEXT:    div_s.h $w0, $w0, $w1
; P5600-NEXT:    st.h $w0, 0($6)
; P5600-NEXT:    jr $ra
; P5600-NEXT:    nop
entry:
  %0 = load <8 x i16>, ptr %a, align 16
  %1 = load <8 x i16>, ptr %b, align 16
  %2 = tail call <8 x i16> @llvm.mips.div.s.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr %c, align 16
  ret void
}

declare <4 x i32> @llvm.mips.div.s.w(<4 x i32>, <4 x i32>)
define void @sdiv_v4i32_builtin(ptr %a, ptr %b, ptr %c) {
; P5600-LABEL: sdiv_v4i32_builtin:
; P5600:       # %bb.0: # %entry
; P5600-NEXT:    ld.w $w0, 0($4)
; P5600-NEXT:    ld.w $w1, 0($5)
; P5600-NEXT:    div_s.w $w0, $w0, $w1
; P5600-NEXT:    st.w $w0, 0($6)
; P5600-NEXT:    jr $ra
; P5600-NEXT:    nop
entry:
  %0 = load <4 x i32>, ptr %a, align 16
  %1 = load <4 x i32>, ptr %b, align 16
  %2 = tail call <4 x i32> @llvm.mips.div.s.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr %c, align 16
  ret void
}

declare <2 x i64> @llvm.mips.div.s.d(<2 x i64>, <2 x i64>)
define void @sdiv_v2i64_builtin(ptr %a, ptr %b, ptr %c) {
; P5600-LABEL: sdiv_v2i64_builtin:
; P5600:       # %bb.0: # %entry
; P5600-NEXT:    ld.d $w0, 0($4)
; P5600-NEXT:    ld.d $w1, 0($5)
; P5600-NEXT:    div_s.d $w0, $w0, $w1
; P5600-NEXT:    st.d $w0, 0($6)
; P5600-NEXT:    jr $ra
; P5600-NEXT:    nop
entry:
  %0 = load <2 x i64>, ptr %a, align 16
  %1 = load <2 x i64>, ptr %b, align 16
  %2 = tail call <2 x i64> @llvm.mips.div.s.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr %c, align 16
  ret void
}

declare <16 x i8> @llvm.mips.mod.s.b(<16 x i8>, <16 x i8>)
define void @smod_v16i8_builtin(ptr %a, ptr %b, ptr %c) {
; P5600-LABEL: smod_v16i8_builtin:
; P5600:       # %bb.0: # %entry
; P5600-NEXT:    ld.b $w0, 0($4)
; P5600-NEXT:    ld.b $w1, 0($5)
; P5600-NEXT:    mod_s.b $w0, $w0, $w1
; P5600-NEXT:    st.b $w0, 0($6)
; P5600-NEXT:    jr $ra
; P5600-NEXT:    nop
entry:
  %0 = load <16 x i8>, ptr %a, align 16
  %1 = load <16 x i8>, ptr %b, align 16
  %2 = tail call <16 x i8> @llvm.mips.mod.s.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr %c, align 16
  ret void
}

declare <8 x i16> @llvm.mips.mod.s.h(<8 x i16>, <8 x i16>)
define void @smod_v8i16_builtin(ptr %a, ptr %b, ptr %c) {
; P5600-LABEL: smod_v8i16_builtin:
; P5600:       # %bb.0: # %entry
; P5600-NEXT:    ld.h $w0, 0($4)
; P5600-NEXT:    ld.h $w1, 0($5)
; P5600-NEXT:    mod_s.h $w0, $w0, $w1
; P5600-NEXT:    st.h $w0, 0($6)
; P5600-NEXT:    jr $ra
; P5600-NEXT:    nop
entry:
  %0 = load <8 x i16>, ptr %a, align 16
  %1 = load <8 x i16>, ptr %b, align 16
  %2 = tail call <8 x i16> @llvm.mips.mod.s.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr %c, align 16
  ret void
}

declare <4 x i32> @llvm.mips.mod.s.w(<4 x i32>, <4 x i32>)
define void @smod_v4i32_builtin(ptr %a, ptr %b, ptr %c) {
; P5600-LABEL: smod_v4i32_builtin:
; P5600:       # %bb.0: # %entry
; P5600-NEXT:    ld.w $w0, 0($4)
; P5600-NEXT:    ld.w $w1, 0($5)
; P5600-NEXT:    mod_s.w $w0, $w0, $w1
; P5600-NEXT:    st.w $w0, 0($6)
; P5600-NEXT:    jr $ra
; P5600-NEXT:    nop
entry:
  %0 = load <4 x i32>, ptr %a, align 16
  %1 = load <4 x i32>, ptr %b, align 16
  %2 = tail call <4 x i32> @llvm.mips.mod.s.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr %c, align 16
  ret void
}

declare <2 x i64> @llvm.mips.mod.s.d(<2 x i64>, <2 x i64>)
define void @smod_v2i64_builtin(ptr %a, ptr %b, ptr %c) {
; P5600-LABEL: smod_v2i64_builtin:
; P5600:       # %bb.0: # %entry
; P5600-NEXT:    ld.d $w0, 0($4)
; P5600-NEXT:    ld.d $w1, 0($5)
; P5600-NEXT:    mod_s.d $w0, $w0, $w1
; P5600-NEXT:    st.d $w0, 0($6)
; P5600-NEXT:    jr $ra
; P5600-NEXT:    nop
entry:
  %0 = load <2 x i64>, ptr %a, align 16
  %1 = load <2 x i64>, ptr %b, align 16
  %2 = tail call <2 x i64> @llvm.mips.mod.s.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr %c, align 16
  ret void
}

declare <16 x i8> @llvm.mips.div.u.b(<16 x i8>, <16 x i8>)
define void @udiv_v16u8_builtin(ptr %a, ptr %b, ptr %c) {
; P5600-LABEL: udiv_v16u8_builtin:
; P5600:       # %bb.0: # %entry
; P5600-NEXT:    ld.b $w0, 0($4)
; P5600-NEXT:    ld.b $w1, 0($5)
; P5600-NEXT:    div_u.b $w0, $w0, $w1
; P5600-NEXT:    st.b $w0, 0($6)
; P5600-NEXT:    jr $ra
; P5600-NEXT:    nop
entry:
  %0 = load <16 x i8>, ptr %a, align 16
  %1 = load <16 x i8>, ptr %b, align 16
  %2 = tail call <16 x i8> @llvm.mips.div.u.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr %c, align 16
  ret void
}

declare <8 x i16> @llvm.mips.div.u.h(<8 x i16>, <8 x i16>)
define void @udiv_v8u16_builtin(ptr %a, ptr %b, ptr %c) {
; P5600-LABEL: udiv_v8u16_builtin:
; P5600:       # %bb.0: # %entry
; P5600-NEXT:    ld.h $w0, 0($4)
; P5600-NEXT:    ld.h $w1, 0($5)
; P5600-NEXT:    div_u.h $w0, $w0, $w1
; P5600-NEXT:    st.h $w0, 0($6)
; P5600-NEXT:    jr $ra
; P5600-NEXT:    nop
entry:
  %0 = load <8 x i16>, ptr %a, align 16
  %1 = load <8 x i16>, ptr %b, align 16
  %2 = tail call <8 x i16> @llvm.mips.div.u.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr %c, align 16
  ret void
}

declare <4 x i32> @llvm.mips.div.u.w(<4 x i32>, <4 x i32>)
define void @udiv_v4u32_builtin(ptr %a, ptr %b, ptr %c) {
; P5600-LABEL: udiv_v4u32_builtin:
; P5600:       # %bb.0: # %entry
; P5600-NEXT:    ld.w $w0, 0($4)
; P5600-NEXT:    ld.w $w1, 0($5)
; P5600-NEXT:    div_u.w $w0, $w0, $w1
; P5600-NEXT:    st.w $w0, 0($6)
; P5600-NEXT:    jr $ra
; P5600-NEXT:    nop
entry:
  %0 = load <4 x i32>, ptr %a, align 16
  %1 = load <4 x i32>, ptr %b, align 16
  %2 = tail call <4 x i32> @llvm.mips.div.u.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr %c, align 16
  ret void
}

declare <2 x i64> @llvm.mips.div.u.d(<2 x i64>, <2 x i64>)
define void @udiv_v2u64_builtin(ptr %a, ptr %b, ptr %c) {
; P5600-LABEL: udiv_v2u64_builtin:
; P5600:       # %bb.0: # %entry
; P5600-NEXT:    ld.d $w0, 0($4)
; P5600-NEXT:    ld.d $w1, 0($5)
; P5600-NEXT:    div_u.d $w0, $w0, $w1
; P5600-NEXT:    st.d $w0, 0($6)
; P5600-NEXT:    jr $ra
; P5600-NEXT:    nop
entry:
  %0 = load <2 x i64>, ptr %a, align 16
  %1 = load <2 x i64>, ptr %b, align 16
  %2 = tail call <2 x i64> @llvm.mips.div.u.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr %c, align 16
  ret void
}

declare <16 x i8> @llvm.mips.mod.u.b(<16 x i8>, <16 x i8>)
define void @umod_v16u8_builtin(ptr %a, ptr %b, ptr %c) {
; P5600-LABEL: umod_v16u8_builtin:
; P5600:       # %bb.0: # %entry
; P5600-NEXT:    ld.b $w0, 0($4)
; P5600-NEXT:    ld.b $w1, 0($5)
; P5600-NEXT:    mod_u.b $w0, $w0, $w1
; P5600-NEXT:    st.b $w0, 0($6)
; P5600-NEXT:    jr $ra
; P5600-NEXT:    nop
entry:
  %0 = load <16 x i8>, ptr %a, align 16
  %1 = load <16 x i8>, ptr %b, align 16
  %2 = tail call <16 x i8> @llvm.mips.mod.u.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr %c, align 16
  ret void
}

declare <8 x i16> @llvm.mips.mod.u.h(<8 x i16>, <8 x i16>)
define void @umod_v8u16_builtin(ptr %a, ptr %b, ptr %c) {
; P5600-LABEL: umod_v8u16_builtin:
; P5600:       # %bb.0: # %entry
; P5600-NEXT:    ld.h $w0, 0($4)
; P5600-NEXT:    ld.h $w1, 0($5)
; P5600-NEXT:    mod_u.h $w0, $w0, $w1
; P5600-NEXT:    st.h $w0, 0($6)
; P5600-NEXT:    jr $ra
; P5600-NEXT:    nop
entry:
  %0 = load <8 x i16>, ptr %a, align 16
  %1 = load <8 x i16>, ptr %b, align 16
  %2 = tail call <8 x i16> @llvm.mips.mod.u.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr %c, align 16
  ret void
}

declare <4 x i32> @llvm.mips.mod.u.w(<4 x i32>, <4 x i32>)
define void @umod_v4u32_builtin(ptr %a, ptr %b, ptr %c) {
; P5600-LABEL: umod_v4u32_builtin:
; P5600:       # %bb.0: # %entry
; P5600-NEXT:    ld.w $w0, 0($4)
; P5600-NEXT:    ld.w $w1, 0($5)
; P5600-NEXT:    mod_u.w $w0, $w0, $w1
; P5600-NEXT:    st.w $w0, 0($6)
; P5600-NEXT:    jr $ra
; P5600-NEXT:    nop
entry:
  %0 = load <4 x i32>, ptr %a, align 16
  %1 = load <4 x i32>, ptr %b, align 16
  %2 = tail call <4 x i32> @llvm.mips.mod.u.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr %c, align 16
  ret void
}

declare <2 x i64> @llvm.mips.mod.u.d(<2 x i64>, <2 x i64>)
define void @umod_v2u64_builtin(ptr %a, ptr %b, ptr %c) {
; P5600-LABEL: umod_v2u64_builtin:
; P5600:       # %bb.0: # %entry
; P5600-NEXT:    ld.d $w0, 0($4)
; P5600-NEXT:    ld.d $w1, 0($5)
; P5600-NEXT:    mod_u.d $w0, $w0, $w1
; P5600-NEXT:    st.d $w0, 0($6)
; P5600-NEXT:    jr $ra
; P5600-NEXT:    nop
entry:
  %0 = load <2 x i64>, ptr %a, align 16
  %1 = load <2 x i64>, ptr %b, align 16
  %2 = tail call <2 x i64> @llvm.mips.mod.u.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr %c, align 16
  ret void
}
