// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s
struct def;
typedef struct def *decl;
struct def {
  int index;
};
struct def d;
int foo(unsigned char cond)
{
  if (cond)
    goto label;
  {
    decl b = &d;
    label:
      return b->index;
  }
  return 0;
}
// It is fine enough to check the LLVM IR are generated succesfully.
// CHECK: define {{.*}}i32 @foo
// CHECK: alloca ptr
// CHECK: alloca i8
