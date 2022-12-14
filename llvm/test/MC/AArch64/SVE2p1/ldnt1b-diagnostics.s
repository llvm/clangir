// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

ldnt1b {z0.b-z2.b}, pn8/z, [x0, x0, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ldnt1b {z0.b-z2.b}, pn8/z, [x0, x0, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldnt1b {z1.b-z4.b}, pn8/z, [x0, x0, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: ldnt1b {z1.b-z4.b}, pn8/z, [x0, x0, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldnt1b {z7.b-z8.b}, pn8/z, [x0, x0, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: ldnt1b {z7.b-z8.b}, pn8/z, [x0, x0, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid predicate-as-counter register

ldnt1b {z0.b-z1.b}, pn7/z, [x0, x0, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate-as-counter register expected pn8..pn15
// CHECK-NEXT: ldnt1b {z0.b-z1.b}, pn7/z, [x0, x0, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldnt1b {z0.b-z1.b}, pn8/m, [x13, #-8, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expecting 'z' predication
// CHECK-NEXT: ldnt1b {z0.b-z1.b}, pn8/m, [x13, #-8, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldnt1b {z0.b-z1.b}, pn8.b, [x13, #-8, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate-as-counter register expected pn8..pn15
// CHECK-NEXT: ldnt1b {z0.b-z1.b}, pn8.b, [x13, #-8, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid immediate range

ldnt1b {z0.b-z3.b}, pn8/z, [x0, #-9, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28]
// CHECK-NEXT: ldnt1b {z0.b-z3.b}, pn8/z, [x0, #-9, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldnt1b {z0.b-z3.b}, pn8/z, [x0, #-36, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28]
// CHECK-NEXT: ldnt1b {z0.b-z3.b}, pn8/z, [x0, #-36, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldnt1b {z0.b-z3.b}, pn8/z, [x0, #32, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28]
// CHECK-NEXT: ldnt1b {z0.b-z3.b}, pn8/z, [x0, #32, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
