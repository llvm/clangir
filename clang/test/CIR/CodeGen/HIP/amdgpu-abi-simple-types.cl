// RUN: %clang_cc1 %s -fclangir -triple amdgcn-amd-amdhsa -emit-cir -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Test int return type
// CHECK: cir.func{{.*}} @return_int() -> !s32i
int return_int() { return 42; }

// Test void return type
// CHECK: cir.func{{.*}} @return_void()
void return_void() {}

// Test char argument
// CHECK: cir.func{{.*}} @char_arg(%arg{{[0-9]+}}: !s8i{{.*}})
int char_arg(char c) { return c; }

// Test short argument
// CHECK: cir.func{{.*}} @short_arg(%arg{{[0-9]+}}: !s16i{{.*}})
int short_arg(short s) { return s; }

// Test int argument
// CHECK: cir.func{{.*}} @int_arg(%arg{{[0-9]+}}: !s32i{{.*}})
int int_arg(int i) { return i; }

// Test long argument
// CHECK: cir.func{{.*}} @long_arg(%arg{{[0-9]+}}: !s64i{{.*}})
long long_arg(long l) { return l; }

// Test float argument
// CHECK: cir.func{{.*}} @float_arg(%arg{{[0-9]+}}: !cir.float{{.*}})
float float_arg(float f) { return f; }

// Test double argument
// CHECK: cir.func{{.*}} @double_arg(%arg{{[0-9]+}}: !cir.double{{.*}})
double double_arg(double d) { return d; }

// Test pointer argument
// CHECK: cir.func{{.*}} @ptr_arg(%arg{{[0-9]+}}: !cir.ptr<!s32i{{.*}}>{{.*}})
int* ptr_arg(int* p) { return p; }

// Test multiple arguments
// CHECK: cir.func{{.*}} @multi_arg(%arg{{[0-9]+}}: !s32i{{.*}}, %arg{{[0-9]+}}: !cir.float{{.*}}, %arg{{[0-9]+}}: !cir.ptr<!s32i{{.*}}>{{.*}})
int multi_arg(int a, float b, int* c) { return a; }
