; REQUIRES: asserts
; RUN: opt -mcpu=thunderx2t99 -passes=loop-unroll --debug-only=loop-unroll --debug-only=basicblock-utils -S -unroll-allow-partial < %s 2>&1 | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; CHECK: Loop Unroll: F[foo] Loop %loop.header
; CHECK:   Loop Size = 18
; CHECK:   Exiting block %loop.inc: TripCount=512, TripMultiple=0, BreakoutTrip=0
; CHECK: UNROLLING loop %loop.header by 4
; CHECK: Merging:
; CHECK: Loop Unroll: F[foo] Loop %loop.2.header
; CHECK:   Loop Size = 19
; CHECK:   Exiting block %loop.2.inc: TripCount=512, TripMultiple=0, BreakoutTrip=0
; CHECK: UNROLLING loop %loop.2.header by 4
; CHECK: Merging:
; CHECK: %counter = phi i32 [ 0, %entry ], [ %inc.3, %loop.inc.3 ]
; CHECK: %val = add nuw nsw i32 %counter, 5
; CHECK: %val1 = add nuw nsw i32 %counter, 6
; CHECK: %val2 = add nuw nsw i32 %counter, 7
; CHECK: %val3 = add nuw nsw i32 %counter, 8
; CHECK: %val4 = add nuw nsw i32 %counter, 9
; CHECK: %val5 = add nuw nsw i32 %counter, 10
; CHECK-NOT: %val = add i32 %counter, 5
; CHECK-NOT: %val = add i32 %counter, 6
; CHECK-NOT: %val = add i32 %counter, 7
; CHECK-NOT: %val = add i32 %counter, 8
; CHECK-NOT: %val = add i32 %counter, 9
; CHECK-NOT: %val = add i32 %counter, 10
; CHECK: %counter.2 = phi i32 [ 0, %exit.0 ], [ %inc.2.3, %loop.2.inc.3 ]

define void @foo(ptr %out) {
entry:
  %0 = alloca [1024 x i32]
  %x0 = alloca [1024 x i32]
  %x01 = alloca [1024 x i32]
  %x02 = alloca [1024 x i32]
  %x03 = alloca [1024 x i32]
  %x04 = alloca [1024 x i32]
  %x05 = alloca [1024 x i32]
  %x06 = alloca [1024 x i32]
  br label %loop.header

loop.header:
  %counter = phi i32 [0, %entry], [%inc, %loop.inc]
  br label %loop.body

loop.body:
  %ptr = getelementptr [1024 x i32], ptr %0, i32 0, i32 %counter
  store i32 %counter, ptr %ptr
  %val = add i32 %counter, 5
  %xptr = getelementptr [1024 x i32], ptr %x0, i32 0, i32 %counter
  store i32 %val, ptr %xptr
  %val1 = add i32 %counter, 6
  %xptr1 = getelementptr [1024 x i32], ptr %x01, i32 0, i32 %counter
  store i32 %val1, ptr %xptr1
  %val2 = add i32 %counter, 7
  %xptr2 = getelementptr [1024 x i32], ptr %x02, i32 0, i32 %counter
  store i32 %val2, ptr %xptr2
  %val3 = add i32 %counter, 8
  %xptr3 = getelementptr [1024 x i32], ptr %x03, i32 0, i32 %counter
  store i32 %val3, ptr %xptr3
  %val4 = add i32 %counter, 9
  %xptr4 = getelementptr [1024 x i32], ptr %x04, i32 0, i32 %counter
  store i32 %val4, ptr %xptr4
  %val5 = add i32 %counter, 10
  %xptr5 = getelementptr [1024 x i32], ptr %x05, i32 0, i32 %counter
  store i32 %val5, ptr %xptr5
  br label %loop.inc

loop.inc:
  %inc = add i32 %counter, 2
  %1 = icmp sge i32 %inc, 1023
  br i1 %1, label  %exit.0, label %loop.header

exit.0:
  %2 = getelementptr [1024 x i32], ptr %0, i32 0, i32 5
  %3 = load i32, ptr %2
  store i32 %3, ptr %out
  br label %loop.2.header


loop.2.header:
  %counter.2 = phi i32 [0, %exit.0], [%inc.2, %loop.2.inc]
  br label %loop.2.body

loop.2.body:
  %ptr.2 = getelementptr [1024 x i32], ptr %0, i32 0, i32 %counter.2
  store i32 %counter.2, ptr %ptr.2
  %val.2 = add i32 %counter.2, 5
  %xptr.2 = getelementptr [1024 x i32], ptr %x0, i32 0, i32 %counter.2
  store i32 %val.2, ptr %xptr.2
  %val1.2 = add i32 %counter.2, 6
  %xptr1.2 = getelementptr [1024 x i32], ptr %x01, i32 0, i32 %counter.2
  store i32 %val1, ptr %xptr1.2
  %val2.2 = add i32 %counter.2, 7
  %xptr2.2 = getelementptr [1024 x i32], ptr %x02, i32 0, i32 %counter.2
  store i32 %val2, ptr %xptr2.2
  %val3.2 = add i32 %counter.2, 8
  %xptr3.2 = getelementptr [1024 x i32], ptr %x03, i32 0, i32 %counter.2
  store i32 %val3.2, ptr %xptr3.2
  %val4.2 = add i32 %counter.2, 9
  %xptr4.2 = getelementptr [1024 x i32], ptr %x04, i32 0, i32 %counter.2
  store i32 %val4.2, ptr %xptr4.2
  %val5.2 = add i32 %counter.2, 10
  %xptr5.2 = getelementptr [1024 x i32], ptr %x05, i32 0, i32 %counter.2
  store i32 %val5.2, ptr %xptr5.2
  %xptr6.2 = getelementptr [1024 x i32], ptr %x06, i32 0, i32 %counter.2
  store i32 %val5.2, ptr %xptr6.2
  br label %loop.2.inc

loop.2.inc:
  %inc.2 = add i32 %counter.2, 2
  %4 = icmp sge i32 %inc.2, 1023
  br i1 %4, label  %exit.2, label %loop.2.header

exit.2:
  %x2 = getelementptr [1024 x i32], ptr %0, i32 0, i32 6
  %x3 = load i32, ptr %x2
  %out2 = getelementptr i32, ptr %out, i32 1
  store i32 %3, ptr %out2
  ret void
}
