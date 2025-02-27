// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -I%S/../Inputs -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#include "std-cxx.h"

void foo(const char* path) {
  std::string str = path;
  str = path;
  str = path;
}

// CHECK: cir
