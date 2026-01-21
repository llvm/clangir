// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -Wno-return-stack-address -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Global CHECK-DAGs must be at the top before any non-DAG CHECKs
// CHECK-DAG: cir.global "private" constant cir_private dso_local @".str" = #cir.const_array<"whatnow\00" : !cir.array<!s8i x 8>> : !cir.array<!s8i x 8> {{.*}}
// CHECK-DAG: cir.global external @globalNullArr = #cir.zero : !cir.array<!s32i x 2> {{.*}}
// CHECK-DAG: cir.global external @arr = #cir.const_array<[#cir.const_record<{#cir.int<1> : !s32i}> : !rec_S, #cir.zero : !rec_S, #cir.zero : !rec_S]> : !cir.array<!rec_S x 3> {{.*}}

void a0() {
  int a[10];
}

// CHECK: cir.func {{.*}} @_Z2a0v()
// CHECK-NEXT:   %0 = cir.alloca !cir.array<!s32i x 10>, !cir.ptr<!cir.array<!s32i x 10>>, ["a"] {alignment = 16 : i64}

void a1() {
  int a[10];
  a[0] = 1;
}

// CHECK: cir.func {{.*}} @_Z2a1v()
// CHECK-NEXT:  %0 = cir.alloca !cir.array<!s32i x 10>, !cir.ptr<!cir.array<!s32i x 10>>, ["a"] {alignment = 16 : i64}
// CHECK-NEXT:  %1 = cir.const #cir.int<1> : !s32i
// CHECK-NEXT:  %2 = cir.const #cir.int<0> : !s32i
// CHECK-NEXT:  %3 = cir.get_element %0[%2 : !s32i] : !cir.ptr<!cir.array<!s32i x 10>> -> !cir.ptr<!s32i>
// CHECK-NEXT:  cir.store{{.*}} %1, %3 : !s32i, !cir.ptr<!s32i>

int *a2() {
  int a[4];
  return &a[0];
}

// CHECK: cir.func {{.*}} @_Z2a2v() -> !cir.ptr<!s32i>
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["__retval"] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca !cir.array<!s32i x 4>, !cir.ptr<!cir.array<!s32i x 4>>, ["a"] {alignment = 16 : i64}
// CHECK-NEXT:   %2 = cir.const #cir.int<0> : !s32i
// CHECK-NEXT:   %3 = cir.get_element %1[%2 : !s32i] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!s32i>
// CHECK-NEXT:   cir.store{{.*}} %3, %0 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT:   %4 = cir.load{{.*}} %0 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT:   cir.return %4 : !cir.ptr<!s32i>

void local_stringlit() {
  const char *s = "whatnow";
}

// CHECK: cir.func {{.*}} @_Z15local_stringlitv()
// CHECK-NEXT:  %0 = cir.alloca !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>, ["s", init] {alignment = 8 : i64}
// CHECK-NEXT:  %1 = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 8>>
// CHECK-NEXT:  %2 = cir.cast array_to_ptrdecay %1 : !cir.ptr<!cir.array<!s8i x 8>> -> !cir.ptr<!s8i>
// CHECK-NEXT:  cir.store{{.*}} %2, %0 : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>

int multidim(int i, int j) {
  int arr[2][2];
  return arr[i][j];
}

// CHECK: %[[#ARR2D:]] = cir.alloca !cir.array<!cir.array<!s32i x 2> x 2>, !cir.ptr<!cir.array<!cir.array<!s32i x 2> x 2>>
// Both indices are loaded first, then used for indexing
// CHECK: %[[#JVAL:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// CHECK: %[[#IVAL:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// Index first dimension with i
// CHECK: %[[#DIM1:]] = cir.get_element %[[#ARR2D]][%[[#IVAL]] : !s32i] : !cir.ptr<!cir.array<!cir.array<!s32i x 2> x 2>> -> !cir.ptr<!cir.array<!s32i x 2>>
// Index second dimension with j
// CHECK: %[[#DIM2:]] = cir.get_element %[[#DIM1]][%[[#JVAL]] : !s32i] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!s32i>

// Should globally zero-initialize null arrays.
int globalNullArr[] = {0, 0};

// Should implicitly zero-initialize global array elements.
struct S {
  int i;
} arr[3] = {{1}};

void testPointerDecaySubscriptAccess(int arr[]) {
// CHECK: cir.func {{.*}} @{{.+}}testPointerDecaySubscriptAccess
  arr[1] = 2;
  // Constants are generated first, then load, then operations
  // CHECK: %[[#TWO:]] = cir.const #cir.int<2> : !s32i
  // CHECK: %[[#DIM1:]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[#BASE:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CHECK: %[[#ELEM:]] = cir.ptr_stride %[[#BASE]], %[[#DIM1]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CHECK: cir.store{{.*}} %[[#TWO]], %[[#ELEM]] : !s32i, !cir.ptr<!s32i>
}

void testPointerDecayedArrayMultiDimSubscriptAccess(int arr[][3]) {
// CHECK: cir.func {{.*}} @{{.+}}testPointerDecayedArrayMultiDimSubscriptAccess
  arr[1][2] = 3;
  // Constants are generated first, then load, then operations
  // CHECK: %[[#THREE:]] = cir.const #cir.int<3> : !s32i
  // CHECK: %[[#TWO:]] = cir.const #cir.int<2> : !s32i
  // CHECK: %[[#ONE:]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[#ARRAY:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!cir.array<!s32i x 3>>>, !cir.ptr<!cir.array<!s32i x 3>>
  // CHECK: %[[#OUTER:]] = cir.ptr_stride %[[#ARRAY]], %[[#ONE]] : (!cir.ptr<!cir.array<!s32i x 3>>, !s32i) -> !cir.ptr<!cir.array<!s32i x 3>>
  // CHECK: %[[#INNER:]] = cir.get_element %[[#OUTER]][%[[#TWO]] : !s32i] : !cir.ptr<!cir.array<!s32i x 3>> -> !cir.ptr<!s32i>
  // CHECK: cir.store{{.*}} %[[#THREE]], %[[#INNER]] : !s32i, !cir.ptr<!s32i>
}

void testArrayOfComplexType() { int _Complex a[4]; }

// CHECK: %[[ARR:.*]] = cir.alloca !cir.array<!cir.complex<!s32i> x 4>, !cir.ptr<!cir.array<!cir.complex<!s32i> x 4>>, ["a"]
