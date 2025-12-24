// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -fclangir-lifetime-check="history=invalid,null" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

namespace std {
template <typename T>
T&& move(T& t) {
  return static_cast<T&&>(t);
}
}

void consume_int(int&&);
void consume_double(double&&);
void consume_float(float&&);

// Test 1: Basic int move
void test_int_basic() {
  int a = 10;
  consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  int b = a; // expected-warning {{use of moved-from value 'a'}}
}

// Test 2: Reinitialization clears state
void test_reinit() {
  int a = 10;
  consume_int(std::move(a));
  a = 20; // Reinitialize
  int b = a; // OK - no warning
}

// Test 3: Multiple types
void test_double() {
  double d = 3.14;
  consume_double(std::move(d)); // expected-note {{moved here via std::move or rvalue reference}}
  double e = d; // expected-warning {{use of moved-from value 'd'}}
}

void test_float() {
  float f = 1.5f;
  consume_float(std::move(f)); // expected-note {{moved here via std::move or rvalue reference}}
  float g = f; // expected-warning {{use of moved-from value 'f'}}
}

// Test 4: Negative cases - NOT moves
void take_lvalue(int&);
void take_value(int);

void test_lvalue_ref() {
  int a = 10;
  take_lvalue(a); // Not a move
  int b = a; // OK
}

void test_by_value() {
  int a = 10;
  take_value(a); // Not a move (copies value)
  int b = a; // OK
}

// Test 5: Use in expressions
void test_use_in_expr() {
  int a = 10;
  consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  int b = a + 5; // expected-warning {{use of moved-from value 'a'}}
}

int test_use_in_return() {
  int a = 10;
  consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  return a; // expected-warning {{use of moved-from value 'a'}}
}

// Test 6: Multiple moves in sequence
void test_multiple_moves() {
  int a = 10;
  consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  int b = a; // expected-warning {{use of moved-from value 'a'}}

  a = 30; // Reinit
  consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  int c = a; // expected-warning {{use of moved-from value 'a'}}
}

// Test 7: Move in conditional
void test_move_in_if(bool cond) {
  int a = 10;
  if (cond) {
    consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  }
  int b = a; // expected-warning {{use of moved-from value 'a'}}
}

// Test 8: Move with different primitive types
void consume_char(char&&);
void consume_bool(bool&&);

void test_char() {
  char c = 'x';
  consume_char(std::move(c)); // expected-note {{moved here via std::move or rvalue reference}}
  char d = c; // expected-warning {{use of moved-from value 'c'}}
}

void test_bool() {
  bool b = true;
  consume_bool(std::move(b)); // expected-note {{moved here via std::move or rvalue reference}}
  bool c = b; // expected-warning {{use of moved-from value 'b'}}
}

// Test 9: Reinit after conditional move
void test_reinit_after_cond(bool cond) {
  int a = 10;
  if (cond) {
    consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  }
  int b = a; // expected-warning {{use of moved-from value 'a'}}
  a = 50; // Reinit
  int c = a; // OK - reinitialized
}

// Test 10: Lambda init-capture with move
void test_lambda_init_capture_move() {
  int a = 10;
  auto lambda = [b = std::move(a)]() { return b; }; // expected-note {{moved here via std::move or rvalue reference}}
  int c = a; // expected-warning {{use of moved-from value 'a'}}
}

// Test 11: Lambda init-capture without move (value)
void test_lambda_init_capture_value() {
  int a = 10;
  auto lambda = [b = a]() { return b; }; // Not a move
  int c = a; // OK - no warning
}

// Test 12: Lambda reference capture
void test_lambda_ref_capture() {
  int a = 10;
  auto lambda = [&a]() { return a; }; // Not a move
  int b = a; // OK - no warning
}

// Test 13: Lambda value capture
void test_lambda_value_capture() {
  int a = 10;
  auto lambda = [a]() { return a; }; // Not a move (copies)
  int b = a; // OK - no warning
}

// Test 14: Multiple captures with move
void test_lambda_multiple_captures() {
  int x = 10;
  int y = 20;
  auto lambda = [a = std::move(x), b = y]() { return a + b; }; // expected-note {{moved here via std::move or rvalue reference}}
  int z = x; // expected-warning {{use of moved-from value 'x'}}
  int w = y; // OK - y was copied, not moved
}

// Test 15: Move-after-move
void test_move_after_move() {
  int a = 10;
  consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  consume_int(std::move(a)); // expected-warning {{use of moved-from value 'a'}}
}

// Test 16: Function parameter move
void test_parameter_move(int a) {
  consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  int b = a; // expected-warning {{use of moved-from value 'a'}}
}

// Test 17: Loop with conditional move
void test_loop_with_move() {
  int a = 10;
  for (int i = 0; i < 3; i++) {
    if (i == 1) {
      consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
    }
    if (i == 2) {
      int b = a; // expected-warning {{use of moved-from value 'a'}}
    }
  }
}

// Test 18: Capture after move (explicit)
void test_capture_after_move() {
  int a = 10;
  consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  auto lambda = [a]() { return a; }; // expected-warning {{use of moved-from value 'a'}}
}

// Test 19: Capture after move (implicit =)
void test_implicit_capture_after_move() {
  int a = 10;
  consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  auto lambda = [=]() { return a; }; // expected-warning {{use of moved-from value 'a'}}
}

// Test 20: Switch with fallthrough
void test_switch_fallthrough(int cond) {
  int a = 10;
  switch (cond) {
  case 1:
    consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  case 2: // fallthrough
    int b = a; // expected-warning {{use of moved-from value 'a'}}
    break;
  }
}

// Test 21: Move in declaration
void test_move_in_declaration() {
  int a = 10;
  int b(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  int c = a; // expected-warning {{use of moved-from value 'a'}}
}

// Test 22: Only first use warned
void test_only_first_use_warned() {
  int a = 10;
  consume_int(std::move(a)); // expected-note {{moved here via std::move or rvalue reference}}
  int b = a; // expected-warning {{use of moved-from value 'a'}}
  int c = a; // NO warning (already warned once)
}
