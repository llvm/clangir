// RUN: %check_cir_tidy %s cir-lifetime-check %t --

int *p0() {
  int *p = nullptr;
  {
    int x = 0;
    p = &x;
    *p = 42;
  }
  *p = 42; // CHECK-MESSAGES: warning: use of invalid pointer 'p'
  return p;
}
