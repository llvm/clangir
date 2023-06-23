// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// No-proto definition followed by a correct call.
int noProto0(x) int x; { return x; }
int test0(int x) {
  // CHECK: cir.func @test0
  return noProto0(x); // We know the definition. Should be a direct call.
  // CHECK: %{{.+}} = cir.call @noProto0(%{{.+}})
}

// Declaration without prototype followed by its definition, then a correct call.
//
// Call to no-proto is made after definition, so a direct call can be used.
int noProto1();
int noProto1(int x) { return x; }
// CHECK: cir.func @noProto1(%arg0: !s32i {{.+}}) -> !s32i {
int test1(int x) {
  // CHECK: cir.func @test1
  return noProto1(x);
  // CHECK: %{{.+}} = cir.call @noProto1(%{{[0-9]+}}) : (!s32i) -> !s32i
}

// Declaration without prototype followed by a correct call, then its definition.
//
// Call to no-proto is made before definition, so a variadic call that takes anything
// is created. Later, when the definition is found, no-proto is replaced.
int noProto2();
int test2(int x) {
  return noProto2(x);
  // CHECK: %{{.+}} = cir.call @noProto2(%{{[0-9]+}}) : (!s32i) -> !s32i
}
int noProto2(int x) { return x; }
// CHECK: cir.func @noProto2(%arg0: !s32i {{.+}}) -> !s32i {

// No-proto declaration without definition (any call here is "correct").
//
// Call to no-proto is made before definition, so a variadic call that takes anything
// is created. Definition is not in the translation unit, so it is left as is.
int noProto3();
// cir.func private @noProto3(...) -> !s32i
int test3(int x) {
// CHECK: cir.func @test3
  return noProto3(x);
  // CHECK: %{{.+}} = cir.call @noProto3(%{{[0-9]+}}) : (!s32i) -> !s32i
}

// TODO(cir): Handle incorrect calls to no-proto functions. It's mostly undefined
//            behaviour, but it should still compile.
