// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

struct nested
{
  union {
    const char *single;
    const char *const *multi;
  } output;
};
static const char * const test[] = {
  "test",
};
const struct nested data[] = 
{
    {
        {
            .multi = test,
        },
    },
    {
        {
            .single = "hello",
        },
    },
};

// CHECK: cir.global  constant external @data = #cir.const_array
// LLVM: @data = constant [2 x {{.*}}]
