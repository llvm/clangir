// RUN: %libomptarget-compile-generic -DLIBRARY -c -o %t.o
// RUN: ar rcs %t.a %t.o
// RUN: %libomptarget-compile-generic %t.a && %libomptarget-run-generic 2>&1 | %fcheck-generic

#ifdef LIBRARY
int x = 42;
#pragma omp declare target(x)

int foo() {
  int value;
#pragma omp target map(from : value)
  value = x;
  return value;
}
#else
#include <stdio.h>
int foo();

int main() {
  int x = foo();

  // CHECK: PASS
  if (x == 42)
    printf("PASS\n");
}
#endif
