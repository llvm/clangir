// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG

// Basic ifunc test: CPU feature detection
static void *resolve_foo(void) {
  return (void *)0;
}

int foo(void) __attribute__((ifunc("resolve_foo")));

void use_foo(void) {
  foo();
}

// CIR-DAG: cir.func.ifunc @foo resolver(@resolve_foo) {{.*}}
// CIR-DAG: cir.func {{.*}}@resolve_foo()
// CIR-DAG: cir.func {{.*}}@use_foo()
// CIR-DAG:   cir.call @foo()

// LLVM-DAG: @foo = ifunc i32 (), ptr @resolve_foo
// LLVM-DAG: define {{.*}}void @use_foo()
// LLVM-DAG:   call i32 @foo()
// LLVM-DAG: define internal ptr @resolve_foo()

// OGCG-DAG: @foo = ifunc i32 (), ptr @resolve_foo
// OGCG-DAG: define{{.*}} void @use_foo()
// OGCG-DAG:   call i32 @foo()
// OGCG-DAG: define internal ptr @resolve_foo()

// Test with multiple implementations
static int foo_impl_v1(void) { return 1; }
static int foo_impl_v2(void) { return 2; }

typedef int (*foo_func_t)(void);

static foo_func_t resolve_foo_multi(void) {
  return foo_impl_v1;
}

int foo_multi(void) __attribute__((ifunc("resolve_foo_multi")));

// CIR-DAG: cir.func.ifunc @foo_multi resolver(@resolve_foo_multi) {{.*}}
// CIR-DAG: cir.func {{.*}}@foo_impl_v1()
// CIR-DAG: cir.func {{.*}}@resolve_foo_multi()

// LLVM-DAG: @foo_multi = ifunc i32 (), ptr @resolve_foo_multi
// LLVM-DAG: define internal i32 @foo_impl_v1()
// LLVM-DAG: define internal ptr @resolve_foo_multi()

// OGCG-DAG: @foo_multi = ifunc i32 (), ptr @resolve_foo_multi
// OGCG-DAG: define internal i32 @foo_impl_v1()
// OGCG-DAG: define internal ptr @resolve_foo_multi()

// Test with extern declaration followed by ifunc
extern int bar(void);

void use_bar_before_def(void) {
  bar();
}

static void *resolve_bar(void) {
  return (void *)0;
}

int bar(void) __attribute__((ifunc("resolve_bar")));

// CIR-DAG: cir.func.ifunc @bar resolver(@resolve_bar) {{.*}}
// CIR-DAG: cir.func {{.*}}@use_bar_before_def()
// CIR-DAG:   cir.call @bar()
// CIR-DAG: cir.func {{.*}}@resolve_bar()

// LLVM-DAG: @bar = ifunc i32 (), ptr @resolve_bar
// LLVM-DAG: define {{.*}}void @use_bar_before_def()
// LLVM-DAG:   call i32 @bar()
// LLVM-DAG: define internal ptr @resolve_bar()

// OGCG-DAG: @bar = ifunc i32 (), ptr @resolve_bar
// OGCG-DAG: define{{.*}} void @use_bar_before_def()
// OGCG-DAG:   call i32 @bar()
// OGCG-DAG: define internal ptr @resolve_bar()
