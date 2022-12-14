; RUN: llc < %s -mtriple=arm64-eabi -mcpu=cyclone | FileCheck %s

%struct.x = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }

@src = external dso_local global %struct.x
@dst = external dso_local global %struct.x

@.str1 = private unnamed_addr constant [31 x i8] c"DHRYSTONE PROGRAM, SOME STRING\00", align 1
@.str2 = private unnamed_addr constant [36 x i8] c"DHRYSTONE PROGRAM, SOME STRING BLAH\00", align 1
@.str3 = private unnamed_addr constant [24 x i8] c"DHRYSTONE PROGRAM, SOME\00", align 1
@.str4 = private unnamed_addr constant [18 x i8] c"DHRYSTONE PROGR  \00", align 1
@.str5 = private unnamed_addr constant [7 x i8] c"DHRYST\00", align 1
@.str6 = private unnamed_addr constant [14 x i8] c"/tmp/rmXXXXXX\00", align 1
@spool.splbuf = internal global [512 x i8] zeroinitializer, align 16

define i32 @t0() {
entry:
; CHECK-LABEL: t0:
; CHECK-DAG: ldur [[REG0:w[0-9]+]], [x[[BASEREG:[0-9]+]], #7]
; CHECK-DAG: stur [[REG0]], [x[[BASEREG2:[0-9]+]], #7]
; CHECK-DAG: ldr [[REG2:x[0-9]+]],
; CHECK-DAG: str [[REG2]],
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 @dst, ptr align 8 @src, i32 11, i1 false)
  ret i32 0
}

define void @t1(ptr nocapture %C) nounwind {
entry:
; CHECK-LABEL: t1:
; CHECK: ldr [[DEST:q[0-9]+]], [x[[BASEREG]]]
; CHECK: str [[DEST:q[0-9]+]], [x0]
; CHECK: ldur [[DEST:q[0-9]+]], [x[[BASEREG:[0-9]+]], #15]
; CHECK: stur [[DEST:q[0-9]+]], [x0, #15]
  tail call void @llvm.memcpy.p0.p0.i64(ptr %C, ptr @.str1, i64 31, i1 false)
  ret void
}

define void @t2(ptr nocapture %C) nounwind {
entry:
; CHECK-LABEL: t2:
; CHECK: mov [[REG3:w[0-9]+]]
; CHECK: movk [[REG3]],
; CHECK: str [[REG3]], [x0, #32]
; CHECK: ldp [[DEST1:q[0-9]+]], [[DEST2:q[0-9]+]], [x{{[0-9]+}}]
; CHECK: stp [[DEST1]], [[DEST2]], [x0]
  tail call void @llvm.memcpy.p0.p0.i64(ptr %C, ptr @.str2, i64 36, i1 false)
  ret void
}

define void @t3(ptr nocapture %C) nounwind {
entry:
; CHECK-LABEL: t3:
; CHECK: ldr [[DEST:q[0-9]+]], [x[[BASEREG]]]
; CHECK: str [[DEST]], [x0]
; CHECK: ldr [[REG4:x[0-9]+]], [x[[BASEREG:[0-9]+]], #16]
; CHECK: str [[REG4]], [x0, #16]
  tail call void @llvm.memcpy.p0.p0.i64(ptr %C, ptr @.str3, i64 24, i1 false)
  ret void
}

define void @t4(ptr nocapture %C) nounwind {
entry:
; CHECK-LABEL: t4:
; CHECK: mov [[REG5:w[0-9]+]], #32
; CHECK: strh [[REG5]], [x0, #16]
; CHECK: ldr [[REG6:q[0-9]+]], [x{{[0-9]+}}]
; CHECK: str [[REG6]], [x0]
  tail call void @llvm.memcpy.p0.p0.i64(ptr %C, ptr @.str4, i64 18, i1 false)
  ret void
}

define void @t5(ptr nocapture %C) nounwind {
entry:
; CHECK-LABEL: t5:
; CHECK: mov [[REG7:w[0-9]+]], #21337
; CHECK: movk [[REG7]],
; CHECK: stur [[REG7]], [x0, #3]
; CHECK: mov [[REG8:w[0-9]+]],
; CHECK: movk [[REG8]],
; CHECK: str [[REG8]], [x0]
  tail call void @llvm.memcpy.p0.p0.i64(ptr %C, ptr @.str5, i64 7, i1 false)
  ret void
}

define void @t6() nounwind {
entry:
; CHECK-LABEL: t6:
; CHECK-DAG: ldur [[REG9:x[0-9]+]], [x{{[0-9]+}}, #6]
; CHECK-DAG: stur [[REG9]], [x{{[0-9]+}}, #6]
; CHECK-DAG: ldr
; CHECK-DAG: str
  call void @llvm.memcpy.p0.p0.i64(ptr @spool.splbuf, ptr @.str6, i64 14, i1 false)
  ret void
}

%struct.Foo = type { i32, i32, i32, i32 }

define void @t7(ptr nocapture %a, ptr nocapture %b) nounwind {
entry:
; CHECK: t7
; CHECK: ldr [[REG10:q[0-9]+]], [x1]
; CHECK: str [[REG10]], [x0]
  tail call void @llvm.memcpy.p0.p0.i32(ptr align 4 %a, ptr align 4 %b, i32 16, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind
