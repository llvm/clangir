// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=BEFORE
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=AFTER
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
double *a __attribute__((annotate("withargs", "21", 12 )));
int *b __attribute__((annotate("withargs", "21", 12 )));
void *c __attribute__((annotate("noargvar")));
void foo(int i) __attribute__((annotate("noargfunc"))) {
}
// redeclare with more annotate
void foo(int i) __attribute__((annotate("withargfunc", "os", 23 )));
void bar() __attribute__((annotate("withargfunc", "os", 22))) {
}

// BEFORE: cir.global  external @a = #cir.ptr<null> : !cir.ptr<!cir.double>
// BEFORE-SAME: [#cir.annotation<name = <"withargs" : !cir.array<!s8i x 8>>,
// BEFORE-SAME: unit = <"[[PATH:.*]]attribute-annotate-multiple.cpp" : 
// BEFORE-SAME: !cir.array<!s8i x [[FILENAME_LEN:[0-9]+]]>>, 
// BEFORE-SAME: lineno = <5>, args = #cir.const_struct<{#cir.global_view<@".str">
// BEFORE-SAME: : !cir.ptr<!cir.array<!s8i x 3>>, #cir.int<12> : !s32i}>
// BEFORE:  cir.global "private"  constant internal dsolocal @".str" =
// BEFORE-SAME: #cir.const_array<"21\00" : !cir.array<!s8i x 3>> :
// BEFORE-SAME: !cir.array<!s8i x 3> {alignment = 1 : i64}
// BEFORE: cir.global  external @b = #cir.ptr<null> : !cir.ptr<!s32i>
// BEFORE-SAME: [#cir.annotation<name = <"withargs" : !cir.array<!s8i x 8>>,
// BEFORE-SAME: unit = <"[[PATH]]attribute-annotate-multiple.cpp" : 
// BEFORE-SAME: !cir.array<!s8i x [[FILENAME_LEN]]>>, 
// BEFORE-SAME: lineno = <6>, args = #cir.const_struct<{#cir.global_view<@".str">
// BEFORE-SAME: : !cir.ptr<!cir.array<!s8i x 3>>, #cir.int<12> : !s32i}>
// BEFORE: cir.global  external @c = #cir.ptr<null> : !cir.ptr<!void>
// BEFORE-SAME: [#cir.annotation<name = <"noargvar" : !cir.array<!s8i x 8>>,
// BEFORE-SAME: unit = <"[[PATH]]attribute-annotate-multiple.cpp" : 
// BEFORE-SAME: !cir.array<!s8i x [[FILENAME_LEN]]>>, 
// BEFORE-SAME: lineno = <7>, args = #cir.zero : !cir.ptr<!void>>]
// BEFORE: cir.func  @_Z3fooi(%arg0: !s32i) attributes {annotates =
// BEFORE-SAME: [#cir.annotation<name = <"noargfunc" : !cir.array<!s8i x 9>>,
// BEFORE-SAME: unit = <"[[PATH]]attribute-annotate-multiple.cpp" : 
// BEFORE-SAME: !cir.array<!s8i x [[FILENAME_LEN]]>>, 
// BEFORE-SAME: lineno = <11>, args = #cir.zero : !cir.ptr<!void>>,
// BEFORE-SAME:  #cir.annotation<name = <"withargfunc" : !cir.array<!s8i x 11>>,
// BEFORE-SAME: unit = <"[[PATH]]attribute-annotate-multiple.cpp" : 
// BEFORE-SAME: !cir.array<!s8i x [[FILENAME_LEN]]>>, 
// BEFORE-SAME: lineno = <11>, args = #cir.const_struct<{#cir.global_view<@".str1"> :
// BEFORE-SAME: !cir.ptr<!cir.array<!s8i x 3>>, #cir.int<23> : !s32i}>
// BEFORE:  cir.func  @_Z3barv() attributes {annotates =
// BEFORE-SAME: [#cir.annotation<name = <"withargfunc" : !cir.array<!s8i x 11>>,
// BEFORE-SAME: unit = <"[[PATH]]attribute-annotate-multiple.cpp" : 
// BEFORE-SAME: !cir.array<!s8i x [[FILENAME_LEN]]>>, 
// BEFORE-SAME: lineno = <12>, args = #cir.const_struct<{#cir.global_view<@".str1"> :
// BEFORE-SAME: !cir.ptr<!cir.array<!s8i x 3>>, #cir.int<22> : !s32i}>
// BEFORE:  cir.global "private"  constant internal dsolocal @".str1"
// BEFORE-SAME: #cir.const_array<"os\00" : !cir.array<!s8i x 3>> :
// BEFORE-SAME: !cir.array<!s8i x 3> {alignment = 1 : i64}

// AFTER: ![[ANNO_STRUCT_T0:.*]] = !cir.struct<struct  {!cir.ptr<!cir.void>,
// AFTER-SAME: !cir.ptr<!cir.void>, !cir.ptr<!cir.void>, !cir.int<u, 32>, !cir.ptr<!cir.void>}>
// AFTER: ![[ANNO_STRUCT_T1:.*]] = !cir.struct<struct {!cir.ptr<!cir.array<!cir.int<s, 8> x 3>>, !cir.int<s, 32>}>
// AFTER: cir.global "private"  constant cir_private dsolocal @".str.0.llvm.metadata" = 
// AFTER-SAME: #cir.const_array<"withargs" : !cir.array<!s8i x 8>> :
// AFTER-SAME: !cir.array<!s8i x 8> {section = "llvm.metadata"}
// AFTER: cir.global "private" constant cir_private dsolocal @".str.1.llvm.metadata" =
// AFTER-SAME: #cir.const_array<"[[PATH:.*]]attribute-annotate-multiple.cpp" : 
// AFTER-SAME: !cir.array<!s8i x [[FILENAME_LEN:[0-9]+]]>> : 
// AFTER-SAME: !cir.array<!s8i x [[FILENAME_LEN]]> {section = "llvm.metadata"}
// AFTER: cir.global "private"  constant cir_private dsolocal @".args.0.llvm.metadata" =
// AFTER-SAME: #cir.const_struct<{#cir.global_view<@".str"> :
// AFTER-SAME: !cir.ptr<!cir.array<!s8i x 3>>, #cir.int<12> : !s32i}> :
// AFTER-SAME: ![[ANNO_STRUCT_T1]] {section = "llvm.metadata"}
// AFTER: cir.global "private"  constant cir_private dsolocal @".str.2.llvm.metadata" =
// AFTER-SAME: #cir.const_array<"noargvar" : !cir.array<!s8i x 8>> :
// AFTER-SAME: !cir.array<!s8i x 8> {section = "llvm.metadata"}
// AFTER: cir.global "private"  constant cir_private dsolocal @".str.3.llvm.metadata" = 
// AFTER-SAME: #cir.const_array<"noargfunc" : !cir.array<!s8i x 9>> :
// AFTER-SAME: !cir.array<!s8i x 9> {section = "llvm.metadata"} 
// AFTER: cir.global "private"  constant cir_private dsolocal @".str.4.llvm.metadata" = 
// AFTER-SAME: #cir.const_array<"withargfunc" : !cir.array<!s8i x 11>> :
// AFTER-SAME: !cir.array<!s8i x 11> {section = "llvm.metadata"}
// AFTER: cir.global "private"  constant cir_private dsolocal @".args.1.llvm.metadata" =
// AFTER-SAME: #cir.const_struct<{#cir.global_view<@".str1"> :
// AFTER-SAME: !cir.ptr<!cir.array<!s8i x 3>>, #cir.int<23> : !s32i}> :
// AFTER-SAME: ![[ANNO_STRUCT_T1]] {section = "llvm.metadata"}
// AFTER: cir.global "private"  constant cir_private dsolocal @".args.2.llvm.metadata" =
// AFTER-SAME: #cir.const_struct<{#cir.global_view<@".str1"> :
// AFTER-SAME: !cir.ptr<!cir.array<!s8i x 3>>, #cir.int<22> : !s32i}> :
// AFTER-SAME: ![[ANNO_STRUCT_T1]] {section = "llvm.metadata"}

// AFTER: cir.global  appending @llvm.global.annotations =
// AFTER-SAME: #cir.const_array<

// AFTER-SAME: [#cir.const_struct<{#cir.global_view<@a> : !cir.ptr<!void>,
// AFTER-SAME: #cir.global_view<@".str.0.llvm.metadata"> : !cir.ptr<!void>,
// AFTER-SAME: #cir.global_view<@".str.1.llvm.metadata"> : !cir.ptr<!void>,
// AFTER-SAME: #cir.int<5> : !u32i, 
// AFTER-SAME: #cir.global_view<@".args.0.llvm.metadata"> : !cir.ptr<!void>}> :
// AFTER-SAME: ![[ANNO_STRUCT_T0]],

// AFTER-SAME: #cir.const_struct<{#cir.global_view<@b> : !cir.ptr<!void>,
// AFTER-SAME: #cir.global_view<@".str.0.llvm.metadata"> : !cir.ptr<!void>,
// AFTER-SAME: #cir.global_view<@".str.1.llvm.metadata"> : !cir.ptr<!void>,
// AFTER-SAME: #cir.int<6> : !u32i, 
// AFTER-SAME: #cir.global_view<@".args.0.llvm.metadata"> : !cir.ptr<!void>}> :
// AFTER-SAME: ![[ANNO_STRUCT_T0]],

// AFTER-SAME: #cir.const_struct<{#cir.global_view<@c> : !cir.ptr<!void>,
// AFTER-SAME: #cir.global_view<@".str.2.llvm.metadata"> : !cir.ptr<!void>,
// AFTER-SAME: #cir.global_view<@".str.1.llvm.metadata"> : !cir.ptr<!void>,
// AFTER-SAME: #cir.int<7> : !u32i, 
// AFTER-SAME: #cir.zero : !cir.ptr<!void>}> :
// AFTER-SAME: ![[ANNO_STRUCT_T0]],

// AFTER-SAME: #cir.const_struct<{#cir.global_view<@_Z3fooi> : !cir.ptr<!void>,
// AFTER-SAME: #cir.global_view<@".str.3.llvm.metadata"> : !cir.ptr<!void>,
// AFTER-SAME: #cir.global_view<@".str.1.llvm.metadata"> : !cir.ptr<!void>,
// AFTER-SAME: #cir.int<11> : !u32i, 
// AFTER-SAME: #cir.zero : !cir.ptr<!void>}> :
// AFTER-SAME: ![[ANNO_STRUCT_T0]],

// AFTER-SAME: #cir.const_struct<{#cir.global_view<@_Z3fooi> : !cir.ptr<!void>,
// AFTER-SAME: #cir.global_view<@".str.4.llvm.metadata"> : !cir.ptr<!void>,
// AFTER-SAME: #cir.global_view<@".str.1.llvm.metadata"> : !cir.ptr<!void>,
// AFTER-SAME: #cir.int<11> : !u32i, 
// AFTER-SAME: #cir.global_view<@".args.1.llvm.metadata"> : !cir.ptr<!void>}> :
// AFTER-SAME: ![[ANNO_STRUCT_T0]],

// AFTER-SAME: #cir.const_struct<{#cir.global_view<@_Z3barv> : !cir.ptr<!void>,
// AFTER-SAME: #cir.global_view<@".str.4.llvm.metadata"> : !cir.ptr<!void>,
// AFTER-SAME: #cir.global_view<@".str.1.llvm.metadata"> : !cir.ptr<!void>,
// AFTER-SAME: #cir.int<12> : !u32i, 
// AFTER-SAME: #cir.global_view<@".args.2.llvm.metadata"> : !cir.ptr<!void>}> :
// AFTER-SAME: ![[ANNO_STRUCT_T0]]]> :
// AFTER-SAME: !cir.array<![[ANNO_STRUCT_T0]] x 6> {section = "llvm.metadata"}

// LLVM: @.str.0.llvm.metadata = private constant [8 x i8] c"withargs", section "llvm.metadata"
// LLVM: @.str.1.llvm.metadata = private constant [[[FILENAME_LEN:[0-9]+]] x i8] c"[[PATH:.*]]attribute-annotate-multiple.cpp", section "llvm.metadata"
// LLVM: @.args.0.llvm.metadata = private constant { ptr, i32 } { ptr @.str, i32 12 }, section "llvm.metadata"
// LLVM: @.str.2.llvm.metadata = private constant [8 x i8] c"noargvar", section "llvm.metadata"
// LLVM: @.str.3.llvm.metadata = private constant [9 x i8] c"noargfunc", section "llvm.metadata"
// LLVM: @.str.4.llvm.metadata = private constant [11 x i8] c"withargfunc", section "llvm.metadata"
// LLVM: @.args.1.llvm.metadata = private constant { ptr, i32 } { ptr @.str1, i32 23 }, section "llvm.metadata"
// LLVM: @.args.2.llvm.metadata = private constant { ptr, i32 } { ptr @.str1, i32 22 }, section "llvm.metadata"

// LLVM: @llvm.global.annotations = appending global [6 x { ptr, ptr, ptr, i32, ptr }]
// LLVM-SAME: [{ ptr, ptr, ptr, i32, ptr }
// LLVM-SAME: { ptr @a, ptr @.str.0.llvm.metadata, ptr @.str.1.llvm.metadata, i32 5, ptr @.args.0.llvm.metadata },

// LLVM-SAME: { ptr, ptr, ptr, i32, ptr }
// LLVM-SAME: { ptr @b, ptr @.str.0.llvm.metadata, ptr @.str.1.llvm.metadata, i32 6, ptr @.args.0.llvm.metadata },

// LLVM-SAME: { ptr, ptr, ptr, i32, ptr }
// LLVM-SAME: { ptr @c, ptr @.str.2.llvm.metadata, ptr @.str.1.llvm.metadata, i32 7, ptr null },

// LLVM-SAME: { ptr, ptr, ptr, i32, ptr }
// LLVM-SAME: { ptr @_Z3fooi, ptr @.str.3.llvm.metadata, ptr @.str.1.llvm.metadata, i32 11, ptr null },

// LLVM-SAME: { ptr, ptr, ptr, i32, ptr }
// LLVM-SAME: { ptr @_Z3fooi, ptr @.str.4.llvm.metadata, ptr @.str.1.llvm.metadata, i32 11, ptr @.args.1.llvm.metadata },

// LLVM-SAME: { ptr, ptr, ptr, i32, ptr }
// LLVM-SAME: { ptr @_Z3barv, ptr @.str.4.llvm.metadata, ptr @.str.1.llvm.metadata, i32 12, ptr @.args.2.llvm.metadata }],
// LLVM-SAME: section "llvm.metadata"

// LLVM: @a = global ptr null
// LLVM: @.str = internal constant [3 x i8] c"21\00"
// LLVM: @b = global ptr null
// LLVM: @c = global ptr null
// LLVM: @.str1 = internal constant [3 x i8] c"os\00"

// LLVM: define dso_local void @_Z3fooi(i32 %0)
// LLVM: define dso_local void @_Z3barv()
