; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; CHECK: test_pos1_ir_ne
; CHECK: bdnz
; a < b
define void @test_pos1_ir_ne(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 32623, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ 32623, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 1
  %cmp = icmp ne i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos2_ir_ne
; CHECK: bdnz
; a < b
define void @test_pos2_ir_ne(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 29554, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ 29554, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 2
  %cmp = icmp ne i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos4_ir_ne
; CHECK: bdnz
; a < b
define void @test_pos4_ir_ne(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 15692, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ 15692, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 4
  %cmp = icmp ne i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos8_ir_ne
; CHECK: bdnz
; a < b
define void @test_pos8_ir_ne(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 10449, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ 10449, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 8
  %cmp = icmp ne i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos16_ir_ne
; CHECK: bdnz
; a < b
define void @test_pos16_ir_ne(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 32087, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ 32087, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 16
  %cmp = icmp ne i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos1_ri_ne
; CHECK: bdnz
; a < b
define void @test_pos1_ri_ne(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, 3472
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 1
  %cmp = icmp ne i32 %inc, 3472
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos2_ri_ne
; CHECK: bdnz
; a < b
define void @test_pos2_ri_ne(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, 8730
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 2
  %cmp = icmp ne i32 %inc, 8730
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos4_ri_ne
; CHECK: bdnz
; a < b
define void @test_pos4_ri_ne(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, 1493
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 4
  %cmp = icmp ne i32 %inc, 1493
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos8_ri_ne
; CHECK: bdnz
; a < b
define void @test_pos8_ri_ne(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, 1706
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 8
  %cmp = icmp ne i32 %inc, 1706
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos16_ri_ne
; CHECK: bdnz
; a < b
define void @test_pos16_ri_ne(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, 1886
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 16
  %cmp = icmp ne i32 %inc, 1886
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos1_rr_ne
; CHECK: bdnz
; a < b
define void @test_pos1_rr_ne(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 1
  %cmp = icmp ne i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos2_rr_ne
; CHECK: bdnz
; a < b
define void @test_pos2_rr_ne(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 2
  %cmp = icmp ne i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos4_rr_ne
; CHECK: bdnz
; a < b
define void @test_pos4_rr_ne(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 4
  %cmp = icmp ne i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos8_rr_ne
; CHECK: bdnz
; a < b
define void @test_pos8_rr_ne(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 8
  %cmp = icmp ne i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; CHECK: test_pos16_rr_ne
; CHECK: bdnz
; a < b
define void @test_pos16_rr_ne(ptr nocapture %p, i32 %a, i32 %b) nounwind {
entry:
  %cmp3 = icmp slt i32 %a, %b
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.04 = phi i32 [ %a, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %i.04
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %add = add nsw i32 %conv, 1
  %conv1 = trunc i32 %add to i8
  store i8 %conv1, ptr %arrayidx, align 1
  %inc = add nsw i32 %i.04, 16
  %cmp = icmp ne i32 %inc, %b
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

