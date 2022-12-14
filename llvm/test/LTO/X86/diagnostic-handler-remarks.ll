; RUN: llvm-as < %s >%t.bc
; PR21108: Diagnostic handlers get pass remarks, even if they're not enabled.

; Confirm that there are -pass-remarks.
; RUN: llvm-lto \
; RUN:          -pass-remarks=inline \
; RUN:          -exported-symbol _func2 -pass-remarks-analysis=loop-vectorize \
; RUN:          -exported-symbol _main -o %t.o %t.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty -check-prefix=REMARKS
; RUN: llvm-nm %t.o | FileCheck %s -check-prefix NM

; RUN: llvm-lto \
; RUN:          -pass-remarks=inline -use-diagnostic-handler \
; RUN:          -exported-symbol _func2 -pass-remarks-analysis=loop-vectorize \
; RUN:          -exported-symbol _main -o %t.o %t.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty -check-prefix=REMARKS_DH
; RUN: llvm-nm %t.o | FileCheck %s -check-prefix NM

; Confirm that -pass-remarks are not printed by default.
; RUN: llvm-lto \
; RUN:         -exported-symbol _func2 \
; RUN:         -exported-symbol _main -o %t.o %t.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty
; RUN: llvm-nm %t.o | FileCheck %s -check-prefix NM

; RUN: llvm-lto \
; RUN:          -use-diagnostic-handler \
; RUN:          -exported-symbol _func2 \
; RUN:          -exported-symbol _main -o %t.o %t.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty
; RUN: llvm-nm %t.o | FileCheck %s -check-prefix NM

; Optimization records are collected regardless of the diagnostic handler
; RUN: rm -f %t.yaml
; RUN: llvm-lto \
; RUN:          -lto-pass-remarks-output=%t.yaml \
; RUN:          -exported-symbol _func2 \
; RUN:          -exported-symbol _main -o %t.o %t.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty
; RUN: cat %t.yaml | FileCheck %s -check-prefixes=YAML,YAML-NO-ANNOTATE

; Try again with `-annotate-inline-lto-phase`.
; RUN: rm -f %t.yaml
; RUN: llvm-lto \
; RUN:          -annotate-inline-phase \
; RUN:          -lto-pass-remarks-output=%t.yaml \
; RUN:          -exported-symbol _func2 \
; RUN:          -exported-symbol _main -o %t.o %t.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty
; RUN: cat %t.yaml | FileCheck %s -check-prefixes=YAML,YAML-ANNOTATE

; REMARKS: remark: {{.*}} 'foo' inlined into 'main'
; REMARKS: remark: {{.*}} the cost-model indicates that interleaving is not beneficial
; REMARKS_DH: llvm-lto: remark: {{.*}} 'foo' inlined into 'main'
; REMARKS_DH: llvm-lto: remark: {{.*}} the cost-model indicates that interleaving is not beneficial
; CHECK-NOT: remark:
; CHECK-NOT: llvm-lto:
; NM-NOT: foo
; NM: func2
; NM: main

; YAML:      --- !Passed
; YAML-NO-ANNOTATE-NEXT: Pass:            inline
; YAML-ANNOTATE-NEXT: Pass:            postlink-cgscc-inline
; YAML-NEXT: Name:            Inlined
; YAML-NEXT: Function:        main
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          ''''
; YAML-NEXT:   - Callee:          foo
; YAML-NEXT:   - String:          ''' inlined into '''
; YAML-NEXT:   - Caller:          main
; YAML-NEXT:   - String:          ''''
; YAML-NEXT:   - String:          ' with '
; YAML-NEXT:   - String:          '(cost='
; YAML-NEXT:   - Cost:            '-15000'
; YAML-NEXT:   - String:          ', threshold='
; YAML-NEXT:   - Threshold:       '337'
; YAML-NEXT:   - String:          ')'
; YAML-NEXT: ...

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

declare i32 @bar()

define i32 @foo() {
  %a = call i32 @bar()
  ret i32 %a
}

define i32 @main() {
  %i = call i32 @foo()
  ret i32 %i
}

define i32 @func2(ptr %out, ptr %out2, ptr %A, ptr %B, ptr %C, ptr %D, ptr %E, ptr %F) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.037 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %i.037
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %B, i64 %i.037
  %1 = load i32, ptr %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, ptr %C, i64 %i.037
  %2 = load i32, ptr %arrayidx2, align 4
  %add3 = add nsw i32 %add, %2
  %arrayidx4 = getelementptr inbounds i32, ptr %E, i64 %i.037
  %3 = load i32, ptr %arrayidx4, align 4
  %add5 = add nsw i32 %add3, %3
  %arrayidx6 = getelementptr inbounds i32, ptr %F, i64 %i.037
  %4 = load i32, ptr %arrayidx6, align 4
  %add7 = add nsw i32 %add5, %4
  %arrayidx8 = getelementptr inbounds i32, ptr %out, i64 %i.037
  store i32 %add7, ptr %arrayidx8, align 4
  %5 = load i32, ptr %arrayidx, align 4
  %6 = load i32, ptr %arrayidx1, align 4
  %add11 = add nsw i32 %6, %5
  %7 = load i32, ptr %arrayidx2, align 4
  %add13 = add nsw i32 %add11, %7
  %8 = load i32, ptr %arrayidx4, align 4
  %add15 = add nsw i32 %add13, %8
  %9 = load i32, ptr %arrayidx6, align 4
  %add17 = add nsw i32 %add15, %9
  %arrayidx18 = getelementptr inbounds i32, ptr %out2, i64 %i.037
  store i32 %add17, ptr %arrayidx18, align 4
  %inc = add i64 %i.037, 1
  %exitcond = icmp eq i64 %inc, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 undef
}
