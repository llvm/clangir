// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -fclangir-call-conv-lowering
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

typedef struct {
  int a, b;
} S;

// LLVM: @init(i64 %[[#V0:]])
// LLVM: %[[#V2:]] = alloca %struct.S, i64 1, align 4
// LLVM: store i64 %[[#V0]], ptr %[[#V2]], align 8
// LLVM: %[[#V3:]] = alloca %struct.S, i64 1, align 4
// LLVM: %[[#V4:]] = getelementptr %struct.S, ptr %[[#V2]], i32 0, i32 0
// LLVM: store i32 1, ptr %[[#V4]], align 4
// LLVM: %[[#V5:]] = getelementptr %struct.S, ptr %[[#V2]], i32 0, i32 1
// LLVM: store i32 2, ptr %[[#V5]], align 4
// LLVM: call void @llvm.memcpy.p0.p0.i32(ptr %[[#V3]], ptr %[[#V2]], i32 8, i1 false)
// LLVM: %[[#V6:]] = load i64, ptr %[[#V3]], align 8
// LLVM: ret i64 %[[#V6]]
S init(S s) {
  s.a = 1;
  s.b = 2;
  return s;
}

// LLVM: @foo1()
// LLVM: %[[#V1:]] = alloca %struct.S, i64 1, align 4
// LLVM: %[[#V2:]] = alloca %struct.S, i64 1, align 4
// LLVM: %[[#V3:]] = load i64, ptr %[[#V1]], align 8
// LLVM: %[[#V4:]] = call i64 @init(i64 %[[#V3]])
// LLVM: store i64 %[[#V4]], ptr %[[#V2]], align 8
// LLVM: call void @llvm.memcpy.p0.p0.i32(ptr %[[#V1]], ptr %[[#V2]], i32 8, i1 false)
void foo1() {
  S s;
  s = init(s);
}

// LLVM: @foo2(i64 %[[#V0:]])
// LLVM: %[[#V2:]] = alloca %struct.S, i64 1, align 4
// LLVM: store i64 %[[#V0]], ptr %[[#V2]], align 8
// LLVM: %[[#V3:]] = alloca %struct.S, i64 1, align 4
// LLVM: %[[#V4:]] = alloca %struct.S, i64 1, align 4
// LLVM: %[[#V5:]] = alloca %struct.S, i64 1, align 4
// LLVM: store %struct.S { i32 1, i32 2 }, ptr %[[#V4]], align 4
// LLVM: %[[#V6:]] = load i64, ptr %[[#V2]], align 8
// LLVM: %[[#V7:]] = call i64 @foo2(i64 %[[#V6]])
// LLVM: store i64 %[[#V7]], ptr %[[#V5]], align 8
// LLVM: call void @llvm.memcpy.p0.p0.i32(ptr %[[#V2]], ptr %[[#V5]], i32 8, i1 false)
// LLVM: call void @llvm.memcpy.p0.p0.i32(ptr %[[#V3]], ptr %[[#V2]], i32 8, i1 false)
// LLVM: %[[#V8:]] = load i64, ptr %[[#V3]], align 8
// LLVM: ret i64 %[[#V8]]
S foo2(S s1) {
  S s2 = {1, 2};
  s1 = foo2(s1);
  return s1;
}

typedef struct {
  char a;
  char b;
} S2;

// LLVM: @init2(i16 %[[#V0:]])
// LLVM: %[[#V2:]] = alloca %struct.S2, i64 1, align 4
// LLVM: store i16 %[[#V0]], ptr %[[#V2]], align 2
// LLVM: %[[#V3:]] = alloca %struct.S2, i64 1, align 1
// LLVM: %[[#V4:]] = getelementptr %struct.S2, ptr %[[#V2]], i32 0, i32 0
// LLVM: store i8 1, ptr %[[#V4]], align 1
// LLVM: %[[#V5:]] = getelementptr %struct.S2, ptr %[[#V2]], i32 0, i32 1
// LLVM: store i8 2, ptr %[[#V5]], align 1
// LLVM: call void @llvm.memcpy.p0.p0.i32(ptr %[[#V3]], ptr %[[#V2]], i32 2, i1 false)
// LLVM: %[[#V6:]] = load i16, ptr %[[#V3]], align 2
// LLVM: ret i16 %[[#V6]]
S2 init2(S2 s) {
  s.a = 1;
  s.b = 2;
  return s;
}

// LLVM: @foo3()
// LLVM: %[[#V1:]] = alloca %struct.S2, i64 1, align 1
// LLVM: %[[#V2:]] = alloca %struct.S2, i64 1, align 1
// LLVM: %[[#V3:]] = load i16, ptr %[[#V1]], align 2
// LLVM: %[[#V4:]] = call i16 @init2(i16 %[[#V3]])
// LLVM: store i16 %[[#V4]], ptr %[[#V2]], align 2
// LLVM: call void @llvm.memcpy.p0.p0.i32(ptr %[[#V1]], ptr %[[#V2]], i32 2, i1 false)
void foo3() {
  S2 s;
  s = init2(s);
}