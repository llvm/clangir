; RUN: llc  < %s -mtriple=mipsel -relocation-model=static -mips-tail-calls=1 | FileCheck %s 

define void @foo0(ptr nocapture %f1) nounwind {
entry:
; CHECK: jr $25
  tail call void %f1(i32 13) nounwind
  ret void
}
