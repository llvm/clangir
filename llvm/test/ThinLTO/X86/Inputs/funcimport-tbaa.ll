target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"


define i32 @main() {
entry:
  %unused = call float @globalfunc1(ptr null, ptr null)
  ret i32 0
}

declare float @globalfunc1(ptr, ptr)