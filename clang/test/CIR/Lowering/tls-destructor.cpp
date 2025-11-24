// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
//
// thread_local with non-trivial destructor - LLVM lowering test
// Verifies that lowered LLVM IR matches CodeGen for TLS destructors

namespace std {
struct string {
  char *data;
  unsigned long size;

  string(const char *s);
  ~string();

  unsigned long length() const { return size; }
};
}

thread_local std::string tls_string = "hello";

// LLVM: @__dso_handle = external global i8
// LLVM: @tls_string = thread_local global %"struct.std::string" zeroinitializer
// LLVM: declare void @__cxa_thread_atexit(ptr, ptr, ptr)

// LLVM: define internal void @__cxx_global_var_init()
// LLVM:   call void @_ZNSt6stringC1EPKc(ptr %{{.*}}, ptr @.str)
// LLVM:   call void @__cxa_thread_atexit(ptr @_ZNSt6stringD1Ev, ptr %{{.*}}, ptr @__dso_handle)
// LLVM:   ret void

int test() {
  return tls_string.length();
}
