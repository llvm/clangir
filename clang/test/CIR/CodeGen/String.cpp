// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fenable-clangir -emit-cir %s -o - | FileCheck %s
// XFAIL: *

class String {
  char *storage;
  long size;
  long capacity;

public:
  String() : size{size} {}
};

void test() {
  String s;
}
