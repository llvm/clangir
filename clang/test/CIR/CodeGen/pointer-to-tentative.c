// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s

int x;
int *px = &x;
// CHECK: cir.global external @px = @x

int a[100];
int *pa = a;
// CHECK: cir.global external @pa = @a


