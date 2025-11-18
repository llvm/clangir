// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Test __builtin_LINE
int test_builtin_LINE() {
  // CHECK-LABEL: cir.func {{.*}}@{{.*}}test_builtin_LINE
  // CHECK: %{{.*}} = cir.const #cir.int<8> : !u32i
  return __builtin_LINE();
}

// Test __builtin_FILE
const char* test_builtin_FILE() {
  // CHECK-LABEL: cir.func {{.*}}@{{.*}}test_builtin_FILE
  // CHECK: %{{.*}} = cir.const #cir.global_view<@".str{{.*}}"> : !cir.ptr<!s8i>
  return __builtin_FILE();
}

// Test __builtin_FUNCTION
const char* test_builtin_FUNCTION() {
  // CHECK-LABEL: cir.func {{.*}}@{{.*}}test_builtin_FUNCTION
  // CHECK: %{{.*}} = cir.const #cir.global_view<@".str{{.*}}"> : !cir.ptr<!s8i>
  return __builtin_FUNCTION();
}

// Test __builtin_COLUMN
int test_builtin_COLUMN() {
  // CHECK-LABEL: cir.func {{.*}}@{{.*}}test_builtin_COLUMN
  // The column number is the position of '__builtin_COLUMN'
  // CHECK: %{{.*}} = cir.const #cir.int<10> : !u32i
  return __builtin_COLUMN();
}

// Test in global context
#line 100 "test_file.cpp"
int global_line = __builtin_LINE();
// CHECK: cir.global external @global_line = #cir.int<100> : !s32i

// Test default argument
int get_line(int l = __builtin_LINE()) {
  return l;
}

void test_default_arg() {
  // CHECK-LABEL: cir.func {{.*}}@{{.*}}test_default_arg
  // The LINE should be from the call site, not the default argument definition
  #line 111
  int x = get_line();
  // CHECK: %{{.*}} = cir.const #cir.int<111> : !u32i
  // CHECK: %{{.*}} = cir.call @{{.*}}get_line{{.*}}({{.*}}) :
}

#line 200 "lambda-test.cpp"
// Test in lambda (this tests that source location correctly captures context)
void test_in_lambda() {
  // CHECK-LABEL: cir.func {{.*}}@{{.*}}test_in_lambda
  auto lambda = []() {
    return __builtin_LINE();
  };
  int x = lambda();
}

#line 214 "combined-test.cpp"
// Test multiple builtins in one expression
void test_combined() {
  // CHECK-LABEL: cir.func {{.*}}@{{.*}}test_combined
  const char* file = __builtin_FILE();
  int line = __builtin_LINE();
  const char* func = __builtin_FUNCTION();
  // All should produce constants
  // CHECK: cir.const
  // CHECK: cir.const
  // CHECK: cir.const
}
