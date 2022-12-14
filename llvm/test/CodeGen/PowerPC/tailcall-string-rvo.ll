; RUN: llc -relocation-model=static -verify-machineinstrs -O2 < %s | FileCheck %s

; The call to function TestBar should be a tail call, when in C++ the string
; `ret` is RVO returned.
; string TestFoo() {
;   string ret = undef;
;   TestBar(&ret);  // tail call optimized
;   return ret;
; }

target triple = "powerpc64le-linux-gnu"

%class.basic_string.11.42.73 = type { %"class.__gnu_cxx::__versa_string.10.41.72" }
%"class.__gnu_cxx::__versa_string.10.41.72" = type { %"class.__gnu_cxx::__sso_string_base.9.40.71" }
%"class.__gnu_cxx::__sso_string_base.9.40.71" = type { %"struct.__gnu_cxx::__vstring_utility<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider.7.38.69", i64, %union.anon.8.39.70 }
%"struct.__gnu_cxx::__vstring_utility<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider.7.38.69" = type { ptr }
%union.anon.8.39.70 = type { i64, [8 x i8] }

declare void @TestBaz(ptr noalias sret(%class.basic_string.11.42.73) %arg)

define dso_local void @TestBar(ptr noalias sret(%class.basic_string.11.42.73) %arg) {
bb:
  call void @TestBaz(ptr noalias sret(%class.basic_string.11.42.73) %arg)
  ret void
}

define dso_local void @TestFoo(ptr noalias sret(%class.basic_string.11.42.73) %arg) {
; CHECK-LABEL: TestFoo:
; CHECK: #TC_RETURNd8 TestBar 0
bb:
  %tmp = getelementptr inbounds %class.basic_string.11.42.73, ptr %arg, i64 0, i32 0, i32 0, i32 2
  store ptr %tmp, ptr %arg, align 8
  tail call void @llvm.memcpy.p0.p0.i64(ptr %tmp, ptr nonnull undef, i64 13, i1 false)
  %tmp3 = getelementptr inbounds %class.basic_string.11.42.73, ptr %arg, i64 0, i32 0, i32 0, i32 1
  store i64 13, ptr %tmp3, align 8
  %tmp4 = getelementptr inbounds %class.basic_string.11.42.73, ptr %arg, i64 0, i32 0, i32 0, i32 2, i32 1, i64 5
  store i8 0, ptr %tmp4, align 1
  tail call void @TestBar(ptr noalias sret(%class.basic_string.11.42.73) %arg)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) #0

attributes #0 = { argmemonly nounwind }
