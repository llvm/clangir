//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool any() const; // constexpr since C++23

#include <bitset>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

template <std::size_t N>
TEST_CONSTEXPR_CXX23 void test_any() {
    std::bitset<N> v;
    v.reset();
    assert(v.any() == false);
    v.set();
    assert(v.any() == (N != 0));
    if (v.size() > 1) {
        v[N/2] = false;
        assert(v.any() == true);
        v.reset();
        v[N/2] = true;
        assert(v.any() == true);
    }
}

TEST_CONSTEXPR_CXX23 bool test() {
  test_any<0>();
  test_any<1>();
  test_any<31>();
  test_any<32>();
  test_any<33>();
  test_any<63>();
  test_any<64>();
  test_any<65>();
  test_any<1000>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 20
  static_assert(test());
#endif

  return 0;
}
