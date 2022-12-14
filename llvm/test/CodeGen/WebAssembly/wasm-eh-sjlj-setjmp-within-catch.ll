; RUN: not --crash opt < %s -wasm-lower-em-ehsjlj -wasm-enable-eh -wasm-enable-sjlj -S 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }
@_ZTIi = external constant ptr

define void @setjmp_within_catch() personality ptr @__gxx_wasm_personality_v0 {
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  invoke void @foo()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr @_ZTIi]
  %2 = call ptr @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i32 @llvm.eh.typeid.for(ptr @_ZTIi) #0
  %matches = icmp eq i32 %3, %4
  br i1 %matches, label %catch, label %rethrow

catch:                                            ; preds = %catch.start
  %5 = call ptr @__cxa_begin_catch(ptr %2) #0 [ "funclet"(token %1) ]
  %6 = load i32, ptr %5, align 4
; CHECK: LLVM ERROR: In function setjmp_within_catch: setjmp within a catch clause is not supported in Wasm EH
; CHECK-NEXT: %call = invoke i32 @setjmp
  %call = invoke i32 @setjmp(ptr noundef %buf) #2 [ "funclet"(token %1) ]
          to label %invoke.cont1 unwind label %ehcleanup

invoke.cont1:                                     ; preds = %catch
  call void @__cxa_end_catch() #0 [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

rethrow:                                          ; preds = %catch.start
  call void @llvm.wasm.rethrow() #1 [ "funclet"(token %1) ]
  unreachable

try.cont:                                         ; preds = %entry, %invoke.cont1
  ret void

ehcleanup:                                        ; preds = %catch
  %7 = cleanuppad within %1 []
  call void @__cxa_end_catch() #0 [ "funclet"(token %7) ]
  cleanupret from %7 unwind to caller
}

declare void @foo()
declare i32 @__gxx_wasm_personality_v0(...)
; Function Attrs: nounwind
declare ptr @llvm.wasm.get.exception(token) #0
; Function Attrs: nounwind
declare i32 @llvm.wasm.get.ehselector(token) #0
; Function Attrs: nounwind
declare i32 @llvm.eh.typeid.for(ptr) #0
; Function Attrs: noreturn
declare void @llvm.wasm.rethrow() #1
declare ptr @__cxa_begin_catch(ptr)
; Function Attrs: returns_twice
declare i32 @setjmp(ptr noundef) #2
declare void @__cxa_end_catch()

attributes #0 = { nounwind }
attributes #1 = { noreturn }
attributes #2 = { returns_twice }
