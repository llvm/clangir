// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir.ll %s
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.og.ll %s
//
// Comparison test: Verify CIR lowering matches original CodeGen for TLS destructors
// Both implementations should call __cxa_thread_atexit with correct arguments

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

int test() {
  return tls_string.length();
}

// Both CIR and OGCG must have thread_local global variable
// CIR-DAG: @tls_string = thread_local global %"struct.std::string" zeroinitializer
// OGCG-DAG: @tls_string = thread_local global %"struct.std::string" zeroinitializer

// Both must declare __dso_handle
// CIR-DAG: @__dso_handle = external global i8
// OGCG-DAG: @__dso_handle = external hidden global i8

// Both must declare __cxa_thread_atexit
// CIR-DAG: declare void @__cxa_thread_atexit(ptr, ptr, ptr)
// OGCG-DAG: declare{{.*}} i32 @__cxa_thread_atexit(ptr, ptr, ptr)

// Both must call constructor for tls_string
// CIR-DAG: call void @_ZNSt6stringC1EPKc(ptr %{{.*}}, ptr @.str)
// OGCG-DAG: call void @_ZNSt6stringC1EPKc(ptr {{.*}} @tls_string, ptr {{.*}} @.str)

// Both must register destructor via __cxa_thread_atexit
// CIR-DAG: call void @__cxa_thread_atexit(ptr @_ZNSt6stringD1Ev, ptr %{{.*}}, ptr @__dso_handle)
// OGCG-DAG: call{{.*}} i32 @__cxa_thread_atexit(ptr @_ZNSt6stringD1Ev, ptr {{.*}}@tls_string{{.*}}, ptr @__dso_handle)
