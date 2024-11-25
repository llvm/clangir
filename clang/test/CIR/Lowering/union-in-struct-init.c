// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll 
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
typedef struct {
  union {
    int a;
    long b;
  };
} S;

S s = { .a = 1 };

// LLVM: @s = global { { i32, [4 x i8] } } { { i32, [4 x i8] } { i32 1, [4 x i8] zeroinitializer } }

typedef struct {
  union {
    int a;
    long b;
  };
} S2;

S2 s2 = { .b = 1 };

// LLVM: @s2 = global %struct.S2 { %union.anon.1 { i64 1 } }

typedef struct {
  union {
    int a;
    long b;
    long double c;
  };
} S3;

S3 s3 = { .a = 1 };

// LLVM: @s3 = global { { i32, [12 x i8] } } { { i32, [12 x i8] } { i32 1, [12 x i8] zeroinitializer } }
