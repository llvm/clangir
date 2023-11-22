// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Yields void.
void test1() { ({ }); }
// CHECK: @test1
//     CHECK: cir.scope {
// CHECK-NOT:   cir.yield
//     CHECK: }

// Yields an l-value.
void test2(int x) { ({ x;}); }
// CHECK: @test2
// CHECK: %{{.+}} = cir.scope {
// CHECK:   %[[#V6:]] = cir.alloca !s32i, cir.ptr <!s32i>, ["tmp"]
// CHECK:   %[[#V7:]] = cir.load %{{.+}} : cir.ptr <!s32i>, !s32i
// CHECK:   cir.store %[[#V7]], %[[#V6]] : !s32i, cir.ptr <!s32i>
// CHECK:   cir.yield %[[#V6]] : !cir.ptr<!s32i>
// CHECK: } : !cir.ptr<!s32i>

// Yields an out-of-scope scalar.
void test3() { ({ int x = 3; x;}); }
// CHECK: @test3
// CHECK: %{{.+}} = cir.scope {
// CHECK:   %[[#V2:]] = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init]
// CHECK:   %[[#V3:]] = cir.alloca !s32i, cir.ptr <!s32i>, ["tmp"]
//          [...]
// CHECK:   %[[#V5:]] = cir.load %[[#V2]] : cir.ptr <!s32i>, !s32i
// CHECK:   cir.store %[[#V5]], %[[#V3]] : !s32i, cir.ptr <!s32i>
// CHECK:   cir.yield %[[#V3]] : !cir.ptr<!s32i>
// CHECK: } : !cir.ptr<!s32i>

// Yields an aggregate.
struct S { int x; };
void test4() { ({ struct S s = {1}; s; }); }
// CHECK: @test4
// CHECK: %[[#RET:]] = cir.alloca !ty_22S22, cir.ptr <!ty_22S22>
// CHECK: cir.scope {
// CHECK:   %[[#VAR:]] = cir.alloca !ty_22S22, cir.ptr <!ty_22S22>
//          [...]
// CHECK:   cir.copy %[[#VAR]] to %[[#RET]] : !cir.ptr<!ty_22S22>
// CHECK: }

// Expression is wrapped in an expression attribute.
void test5(int x) {
  ({[[gsl::suppress("foo")]] x;});
}
// CHECK: %{{.+}} = cir.scope {
//         [...]
// CHECK: } : !cir.ptr<!s32i>

// TODO(cir): Missing label support.
// // Expression is wrapped in a label.
// // void test5(int x) { x = ({ label: x; }); }
