// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-mem2reg -emit-llvm %s -o %t.ll

#include <stdbool.h>

bool test() {
  bool x = false;
  if (x) 
    return x;
  return x;
}