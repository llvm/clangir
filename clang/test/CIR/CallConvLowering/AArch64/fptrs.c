// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu  -fclangir -fclangir-call-conv-lowering -emit-cir-flat -mmlir --mlir-print-ir-after=cir-call-conv-lowering %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu  -fclangir -emit-llvm %s -o %t.ll -fclangir-call-conv-lowering
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

typedef int (*myfptr_1)(int);

typedef struct {
  myfptr_1 f;
} A;

// CIR: cir.func @foo(%arg0: !s32i
// CIR: %[[#V0:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CIR: %[[#V1:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CIR: cir.store %arg0, %[[#V0]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[#V2:]] = cir.load %[[#V0]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store %[[#V2]], %[[#V1]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[#V3:]] = cir.load %[[#V1]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return %[[#V3]] : !s32i

// LLVM: i32 @foo(i32 %[[#V0:]])
// LLVM: %[[#V2:]] = alloca i32, i64 1, align 4
// LLVM: %[[#V3:]] = alloca i32, i64 1, align 4
// LLVM: store i32 %[[#V0]], ptr %[[#V2]], align 4
// LLVM: %[[#V4:]] = load i32, ptr %[[#V2]], align 4
// LLVM: store i32 %[[#V4]], ptr %[[#V3]], align 4
// LLVM: %[[#V5:]] = load i32, ptr %[[#V3]], align 4
// LLVM: ret i32 %[[#V5]]
int foo(int x) { return x; }

// CIR: cir.func @passA(%arg0: !u64i
// CIR: %[[#V0:]] = cir.alloca !ty_A, !cir.ptr<!ty_A>, [""] {alignment = 4 : i64}
// CIR: %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_A>), !cir.ptr<!u64i>
// CIR: cir.store %arg0, %[[#V1]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[#V2:]] = cir.get_global @foo : !cir.ptr<!cir.func<!s32i (!s32i)>>
// CIR: %[[#V3:]] = cir.get_member %[[#V0]][0] {name = "f"} : !cir.ptr<!ty_A> -> !cir.ptr<!cir.ptr<!cir.func<!s32i (!s32i)>>>
// CIR: cir.store %[[#V2]], %[[#V3]] : !cir.ptr<!cir.func<!s32i (!s32i)>>, !cir.ptr<!cir.ptr<!cir.func<!s32i (!s32i)>>>
// CIR: cir.return

// LLVM: void @passA(i64 %[[#V0:]])
// LLVM: %[[#V2:]] = alloca %struct.A, i64 1, align 4
// LLVM: store i64 %[[#V0]], ptr %[[#V2]], align 8
// LLVM: %[[#V3:]] = getelementptr %struct.A, ptr %[[#V2]], i32 0, i32 0
// LLVM: store ptr @foo, ptr %[[#V3]], align 8
// LLVM: ret void
void passA(A a) { a.f = foo; }

typedef void (*myfptr_2)();

typedef struct {
  myfptr_1 f1;
  myfptr_2 f2;
} B;

// CIR: cir.func @passB(%arg0: !cir.array<!u64i x 2>
// CIR: %[[#V0:]] = cir.alloca !ty_B, !cir.ptr<!ty_B>, [""] {alignment = 4 : i64}
// CIR: %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_B>), !cir.ptr<!cir.array<!u64i x 2>>
// CIR: cir.store %arg0, %[[#V1]] : !cir.array<!u64i x 2>, !cir.ptr<!cir.array<!u64i x 2>>
// CIR: cir.return

// LLVM: void @passB([2 x i64] %[[#V0:]])
// LLVM: %[[#V2:]] = alloca %struct.B, i64 1, align 4
// LLVM: store [2 x i64] %[[#V0]], ptr %[[#V2]], align 8
// LLVM: ret void
void passB(B b) {}

typedef int (*myfptr_3)();

typedef struct {
  myfptr_3 f;
} C;

// CIR: cir.func @passC(%arg0: !u64i
// CIR: %[[#V0:]] = cir.alloca !ty_C, !cir.ptr<!ty_C>, [""] {alignment = 4 : i64}
// CIR: %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_C>), !cir.ptr<!u64i>
// CIR: cir.store %arg0, %[[#V1]] : !u64i, !cir.ptr<!u64i>
// CIR: cir.return

// LLVM: void @passC(i64 %[[#V0:]])
// LLVM: %[[#V2:]] = alloca %struct.C, i64 1, align 4
// LLVM: store i64 %[[#V0]], ptr %[[#V2]], align 8
// LLVM: ret void
void passC(C c) {}

typedef long (*myfptr_4)(int, long, short);

typedef struct {
  myfptr_4 f;
} D;

// CIR: cir.func @passD(%arg0: !u64i
// CIR: %[[#V0:]] = cir.alloca !ty_D, !cir.ptr<!ty_D>, [""] {alignment = 4 : i64}
// CIR: %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_D>), !cir.ptr<!u64i>
// CIR: cir.store %arg0, %[[#V1]] : !u64i, !cir.ptr<!u64i>
// CIR: cir.return

// LLVM: define dso_local void @passD(i64 %[[#V0:]])
// LLVM: %[[#V2:]] = alloca %struct.D, i64 1, align 4
// LLVM: store i64 %[[#V0]], ptr %[[#V2]], align 8
// LLVM: ret void
void passD(D d) {}
