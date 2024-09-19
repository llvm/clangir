// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s

#define NULL ((void *)0)

char *foo() {
  return (char*)NULL + 1;
}
