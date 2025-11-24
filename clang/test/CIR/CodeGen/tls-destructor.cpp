// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
//
// thread_local with non-trivial destructor
// Tests TLS variable with destructor registration via __cxa_thread_atexit

namespace std {
struct string {
  char *data;
  unsigned long size;

  string(const char *s);
  ~string();

  unsigned long length() const { return size; }
};
}

// CHECK: cir.global "private" external @__dso_handle : i8
// CHECK: cir.func private @__cxa_thread_atexit(!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>)
// CHECK: cir.global external tls_dyn @tls_string

thread_local std::string tls_string = "hello";

// CHECK: cir.func internal private @__cxx_global_var_init()
// CHECK:   cir.get_global thread_local @tls_string
// CHECK:   cir.call @_ZNSt6stringC1EPKc
// CHECK:   %[[DTOR:.*]] = cir.get_global @_ZNSt6stringD1Ev
// CHECK:   %[[DTOR_VOID:.*]] = cir.cast bitcast %[[DTOR]]
// CHECK:   %[[OBJ:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!rec_std3A3Astring> -> !cir.ptr<!void>
// CHECK:   %[[DSO:.*]] = cir.get_global @__dso_handle
// CHECK:   cir.call @__cxa_thread_atexit(%[[DTOR_VOID]], %[[OBJ]], %[[DSO]])

int test() {
  return tls_string.length();
}
