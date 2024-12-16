// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

typedef struct {
  char a;
} S_1;

typedef struct {
  long a, b;
} S_2;

typedef union {
  S_1 a;
  S_2 b;
} U;

void foo() { U arr[2] = {{.b = {1, 2}}, {.a = {1}}}; }

// LLVM: store { { %struct.S_2 }, { %struct.S_1, [15 x i8] } } { { %struct.S_2 } { %struct.S_2 { i64 1, i64 2 } }, { %struct.S_1, [15 x i8] } { %struct.S_1 { i8 1 }, [15 x i8] zeroinitializer } }
