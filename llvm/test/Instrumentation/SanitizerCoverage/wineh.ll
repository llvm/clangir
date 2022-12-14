; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc -S | FileCheck %s --check-prefix=CHECK

; Generated from this C++ source:
; $ clang -O2 t.cpp -S -emit-llvm
; void g();
; struct Foo { Foo(); ~Foo(); };
; int f() {
;   Foo v;
;   g();
;   try {
;     g();
;   } catch (int e) {
;     g();
;   } catch (...) {
;     g();
;   }
;   return 0;
; }

; FIXME: We need to do more than this. In particular, __sanitizer_cov callbacks
; in funclets need token bundles.

; CHECK-LABEL: define i32 @"\01?f@@YAHXZ"()
; CHECK: catch.dispatch:
; CHECK-NEXT: catchswitch within none [label %catch3, label %catch] unwind label %ehcleanup

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.10.24728"

%rtti.TypeDescriptor2 = type { ptr, ptr, [3 x i8] }
%struct.Foo = type { i8 }

$"\01??_R0H@8" = comdat any

@"\01??_7type_info@@6B@" = external constant ptr
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { ptr @"\01??_7type_info@@6B@", ptr null, [3 x i8] c".H\00" }, comdat

; Function Attrs: uwtable
define i32 @"\01?f@@YAHXZ"() local_unnamed_addr #0 personality ptr @__CxxFrameHandler3 {
entry:
  %v = alloca %struct.Foo, align 1
  %e = alloca i32, align 4
  call void @llvm.lifetime.start(i64 1, ptr nonnull %v) #4
  %call = call ptr @"\01??0Foo@@QEAA@XZ"(ptr nonnull %v)
  invoke void @"\01?g@@YAXXZ"()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  invoke void @"\01?g@@YAXXZ"()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %invoke.cont
  %0 = catchswitch within none [label %catch3, label %catch] unwind label %ehcleanup

catch3:                                           ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr @"\01??_R0H@8", i32 0, ptr %e]
  invoke void @"\01?g@@YAXXZ"() [ "funclet"(token %1) ]
          to label %invoke.cont4 unwind label %ehcleanup

invoke.cont4:                                     ; preds = %catch3
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %invoke.cont, %invoke.cont2, %invoke.cont4
  call void @"\01??1Foo@@QEAA@XZ"(ptr nonnull %v) #4
  call void @llvm.lifetime.end(i64 1, ptr nonnull %v) #4
  ret i32 0

catch:                                            ; preds = %catch.dispatch
  %2 = catchpad within %0 [ptr null, i32 64, ptr null]
  invoke void @"\01?g@@YAXXZ"() [ "funclet"(token %2) ]
          to label %invoke.cont2 unwind label %ehcleanup

invoke.cont2:                                     ; preds = %catch
  catchret from %2 to label %try.cont

ehcleanup:                                        ; preds = %catch3, %catch, %catch.dispatch, %entry
  %3 = cleanuppad within none []
  call void @"\01??1Foo@@QEAA@XZ"(ptr nonnull %v) #4 [ "funclet"(token %3) ]
  call void @llvm.lifetime.end(i64 1, ptr nonnull %v) #4
  cleanupret from %3 unwind to caller
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start(i64, ptr nocapture) #1

declare ptr @"\01??0Foo@@QEAA@XZ"(ptr returned) unnamed_addr #2

declare void @"\01?g@@YAXXZ"() local_unnamed_addr #2

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind
declare void @"\01??1Foo@@QEAA@XZ"(ptr) unnamed_addr #3

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(i64, ptr nocapture) #1

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 5.0.0 "}
