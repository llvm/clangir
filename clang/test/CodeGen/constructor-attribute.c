// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=WITHOUTATEXIT %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fregister-global-dtors-with-atexit -debug-info-kind=line-tables-only -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=CXAATEXIT --check-prefix=WITHATEXIT %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fno-use-cxa-atexit -fregister-global-dtors-with-atexit -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=ATEXIT --check-prefix=WITHATEXIT %s

// WITHOUTATEXIT: global_ctors{{.*}}@A{{.*}}@C
// WITHOUTATEXIT: @llvm.global_dtors = appending global [5 x { i32, ptr, ptr }]{{.*}}@B{{.*}}@E{{.*}}@F{{.*}}@G{{.*}}@D
// WITHATEXIT: @llvm.global_ctors = appending global [5 x { i32, ptr, ptr }]{{.*}}i32 65535, ptr @A,{{.*}}i32 65535, ptr @C,{{.*}}i32 123, ptr @__GLOBAL_init_123,{{.*}}i32 789, ptr @[[GLOBAL_INIT_789:__GLOBAL_init_789.[0-9]+]],{{.*}}i32 65535, ptr @__GLOBAL_init_65535,
// WITHATEXIT-NOT: global_dtors

// CHECK: define{{.*}} void @A()
// CHECK: define{{.*}} void @B()
// CHECK: define internal void @E()
// CHECK: define internal void @F()
// CHECK: define internal void @G()
// CHECK: define{{.*}} i32 @__GLOBAL_init_789(i32 noundef %{{.*}})
// CHECK: define internal void @C()
// CHECK: define internal i32 @foo()
// CHECK: define internal void @D()
// CHECK: define{{.*}} i32 @main()
// WITHOUTATEXIT-NOT: define

// CXAATEXIT: define internal void @__GLOBAL_init_123(){{.*}}section "__TEXT,__StaticInit,regular,pure_instructions" !dbg ![[GLOBAL_INIT_SP:.*]] {
// ATEXIT: define internal void @__GLOBAL_init_123(){{.*}}section "__TEXT,__StaticInit,regular,pure_instructions"
// CXAATEXIT: call i32 @__cxa_atexit(ptr @E, ptr null, ptr @__dso_handle) {{.*}}, !dbg ![[GLOBAL_INIT_LOC:.*]]
// CXAATEXIT: call i32 @__cxa_atexit(ptr @G, ptr null, ptr @__dso_handle)
// ATEXIT: call i32 @atexit(ptr @E)
// ATEXIT: call i32 @atexit(ptr @G)

// WITHATEXIT: define internal void @[[GLOBAL_INIT_789]](){{.*}}section "__TEXT,__StaticInit,regular,pure_instructions"
// CXAATEXIT: call i32 @__cxa_atexit(ptr @F, ptr null, ptr @__dso_handle)
// ATEXIT: call i32 @atexit(ptr @F)

// WITHATEXIT: define internal void @__GLOBAL_init_65535(){{.*}}section "__TEXT,__StaticInit,regular,pure_instructions"
// CXAATEXIT: call i32 @__cxa_atexit(ptr @B, ptr null, ptr @__dso_handle)
// CXAATEXIT: call i32 @__cxa_atexit(ptr @D, ptr null, ptr @__dso_handle)
// ATEXIT: call i32 @atexit(ptr @B)
// ATEXIT: call i32 @atexit(ptr @D)

int printf(const char *, ...);

void A(void) __attribute__((constructor));
void B(void) __attribute__((destructor));

void A(void) {
  printf("A\n");
}

void B(void) {
  printf("B\n");
}

static void C(void) __attribute__((constructor));

static void D(void) __attribute__((destructor));

static __attribute__((destructor(123))) void E(void) {
}

static __attribute__((destructor(789))) void F(void) {
}

static __attribute__((destructor(123))) void G(void) {
}

// Test that this function doesn't collide with the synthesized constructor
// function for destructors with priority 789.
int __GLOBAL_init_789(int a) {
  return a * a;
}

static int foo(void) {
  return 10;
}

static void C(void) {
  printf("A: %d\n", foo());
}

static void D(void) {
  printf("B\n");
}

int main(void) {
  return 0;
}

// CXAATEXIT: ![[GLOBAL_INIT_SP]] = distinct !DISubprogram(linkageName: "__GLOBAL_init_123",
// CXAATEXIT: ![[GLOBAL_INIT_LOC]] = !DILocation(line: 0, scope: ![[GLOBAL_INIT_SP]])
