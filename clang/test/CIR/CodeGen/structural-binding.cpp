// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct B { B(); };

struct A {
  B a;
  int b;
  char c;
};

struct C {
  C(int a, int b): a(a), b(b) {}
  template <unsigned>
  friend const int &get(const C&);
 private:
  int a;
  int b;
};

template <>
const int &get<0>(const C& c) { return c.a; }
template <>
const int &get<1>(const C& c) { return c.b; }

namespace std {

template <typename>
struct tuple_size;

template <>
struct tuple_size<C> { constexpr inline static unsigned value = 2; };

template <unsigned, typename>
struct tuple_element;

template <unsigned I>
struct tuple_element<I, C> { using type = const int; };

}

void f(A &a) {
  // CHECK: @_Z1fR1A

  // binding to data members
  auto &[x, y, z] = a;
  (x, y, z);
  // CHECK: %[[a:.*]] = cir.load %1 : !cir.ptr<!cir.ptr<!ty_22A22>>, !cir.ptr<!ty_22A22>
  // CHECK: {{.*}} = cir.get_member %[[a]][0] {name = "a"} : !cir.ptr<!ty_22A22> -> !cir.ptr<!ty_22B22>
  // CHECK: %[[a:.*]] = cir.load %1 : !cir.ptr<!cir.ptr<!ty_22A22>>, !cir.ptr<!ty_22A22>
  // CHECK: {{.*}} = cir.get_member %[[a]][1] {name = "b"} : !cir.ptr<!ty_22A22> -> !cir.ptr<!s32i>
  // CHECK: %[[a:.*]] = cir.load %1 : !cir.ptr<!cir.ptr<!ty_22A22>>, !cir.ptr<!ty_22A22>
  // CHECK: {{.*}} = cir.get_member %[[a]][2] {name = "c"} : !cir.ptr<!ty_22A22> -> !cir.ptr<!s8i>

  auto [x2, y2, z2] = a;
  (x2, y2, z2);
  // CHECK: cir.call @_ZN1AC1ERKS_(%2, {{.*}}) : (!cir.ptr<!ty_22A22>, !cir.ptr<!ty_22A22>) -> ()
  // CHECK: {{.*}} = cir.get_member %2[0] {name = "a"} : !cir.ptr<!ty_22A22> -> !cir.ptr<!ty_22B22>
  // CHECK: {{.*}} = cir.get_member %2[1] {name = "b"} : !cir.ptr<!ty_22A22> -> !cir.ptr<!s32i>
  // CHECK: {{.*}} = cir.get_member %2[2] {name = "c"} : !cir.ptr<!ty_22A22> -> !cir.ptr<!s8i>

  // for the rest, just expect the codegen does't crash
  auto &&[x3, y3, z3] = a;
  (x3, y3, z3);

  const auto &[x4, y4, z4] = a;
  (x4, y4, z4);

  const auto [x5, y5, z5] = a;
  (x5, y5, z5);

  // binding a tuple-like type
  C c(1, 2);

  auto [x8, y8] = c;
  (x8, y8);
  // CHECK: cir.call @_ZN1CC1ERKS_(%[[c:.*]], %6) : (!cir.ptr<!ty_22C22>, !cir.ptr<!ty_22C22>) -> ()
  // CHECK: %[[x8:.*]] = cir.call @_Z3getILj0EERKiRK1C(%[[c]]) : (!cir.ptr<!ty_22C22>) -> !cir.ptr<!s32i>
  // CHECK: cir.store %[[x8]], %[[x8p:.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CHECK: %[[x9:.*]] = cir.call @_Z3getILj1EERKiRK1C(%[[c]]) : (!cir.ptr<!ty_22C22>) -> !cir.ptr<!s32i>
  // CHECK: cir.store %[[x9]], %[[x9p:.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CHECK: {{.*}} = cir.load %[[x8p]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CHECK: {{.*}} = cir.load %[[x9p]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>

  auto &[x9, y9] = c;
  (x9, y9);
  // CHECK: cir.store %6, %[[cp:.*]] : !cir.ptr<!ty_22C22>, !cir.ptr<!cir.ptr<!ty_22C22>>
  // CHECK: %[[c:.*]] = cir.load %[[cp]] : !cir.ptr<!cir.ptr<!ty_22C22>>, !cir.ptr<!ty_22C22>
  // CHECK: %[[x8:.*]] = cir.call @_Z3getILj0EERKiRK1C(%[[c]]) : (!cir.ptr<!ty_22C22>) -> !cir.ptr<!s32i>
  // CHECK: cir.store %[[x8]], %[[x8p:.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CHECK: %[[c:.*]] = cir.load %[[cp]] : !cir.ptr<!cir.ptr<!ty_22C22>>, !cir.ptr<!ty_22C22>
  // CHECK: %[[x9:.*]] = cir.call @_Z3getILj1EERKiRK1C(%[[c]]) : (!cir.ptr<!ty_22C22>) -> !cir.ptr<!s32i>
  // CHECK: cir.store %[[x9]], %[[x9p:.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CHECK: {{.*}} = cir.load %[[x8p]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CHECK: {{.*}} = cir.load %[[x9p]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>

  
  // TODO: add test case for binding to an array type
  // after ArrayInitLoopExpr is supported
}
