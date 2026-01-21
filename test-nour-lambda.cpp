// Test case for NOUR_Constant - lambda capture of constexpr
void test() {
  static constexpr int x = 10;
  auto lambda = [&]() { return x; };
  lambda();
}
