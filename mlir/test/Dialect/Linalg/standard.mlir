// RUN: mlir-opt %s -convert-linalg-to-std | FileCheck %s

func.func @dot(%arg0: memref<?xf32, strided<[1], offset: ?>>,
          %arg1: memref<?xf32, strided<[1], offset: ?>>,
          %arg2: memref<f32>) {
  linalg.dot ins(%arg0, %arg1: memref<?xf32, strided<[1], offset: ?>>,
                               memref<?xf32, strided<[1], offset: ?>>)
             outs(%arg2: memref<f32>)
  return
}
// CHECK-LABEL: func @dot(
//  CHECK-SAME: %[[arg0:[a-zA-z0-9]*]]: memref<?xf32, strided<[1], offset: ?>>,
//  CHECK-SAME: %[[arg1:[a-zA-z0-9]*]]: memref<?xf32, strided<[1], offset: ?>>,
//  CHECK-SAME: %[[arg2:[a-zA-z0-9]*]]: memref<f32>) {
//       CHECK:   %[[o0:.*]] = memref.cast %[[arg0]] :
//  CHECK-SAME:     memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
//       CHECK:   %[[o1:.*]] = memref.cast %[[arg1]] :
//  CHECK-SAME:     memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
//       CHECK:   %[[o2:.*]] = memref.cast %[[arg2]] :
//  CHECK-SAME:     memref<f32> to memref<f32, strided<[], offset: ?>>
//       CHECK:   call @linalg_dot_viewsxf32_viewsxf32_viewf32(
//  CHECK-SAME:     %[[o0]], %[[o1]], %[[o2]]) :
//  CHECK-SAME:   memref<?xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>>, memref<f32, strided<[], offset: ?>>

#matmul_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmul_trait = {
  iterator_types = ["parallel", "parallel", "reduction"],
  indexing_maps = #matmul_accesses,
  library_call = "external_outerproduct_matmul"
}

!vector_type_A = vector<4xf32>
!vector_type_B = vector<4xf32>
!vector_type_C = vector<4x4xf32>

!matrix_type_A = memref<?x?x!vector_type_A>
!matrix_type_B = memref<?x?x!vector_type_B>
!matrix_type_C = memref<?x?x!vector_type_C>

func.func @matmul_vec_impl(%A: !matrix_type_A, %B: !matrix_type_B, %C: !matrix_type_C) {
  linalg.generic #matmul_trait
      ins(%A, %B : !matrix_type_A, !matrix_type_B)
     outs(%C : !matrix_type_C) {
    ^bb0(%a: !vector_type_A, %b: !vector_type_B, %c: !vector_type_C):
      %d = vector.outerproduct %a, %b, %c: !vector_type_A, !vector_type_B
      linalg.yield %d: !vector_type_C
  }
  return
}
// CHECK-LABEL: func @matmul_vec_impl(
// CHECK:  call @external_outerproduct_matmul(%{{.*}}) :
