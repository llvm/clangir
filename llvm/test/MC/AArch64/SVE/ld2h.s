// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ld2h    { z0.h, z1.h }, p0/z, [x0, x0, lsl #1]
// CHECK-INST: ld2h    { z0.h, z1.h }, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0xc0,0xa0,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a4a0c000 <unknown>

ld2h    { z5.h, z6.h }, p3/z, [x17, x16, lsl #1]
// CHECK-INST: ld2h    { z5.h, z6.h }, p3/z, [x17, x16, lsl #1]
// CHECK-ENCODING: [0x25,0xce,0xb0,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a4b0ce25 <unknown>

ld2h    { z0.h, z1.h }, p0/z, [x0]
// CHECK-INST: ld2h    { z0.h, z1.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0xe0,0xa0,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a4a0e000 <unknown>

ld2h    { z23.h, z24.h }, p3/z, [x13, #-16, mul vl]
// CHECK-INST: ld2h    { z23.h, z24.h }, p3/z, [x13, #-16, mul vl]
// CHECK-ENCODING: [0xb7,0xed,0xa8,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a4a8edb7 <unknown>

ld2h    { z21.h, z22.h }, p5/z, [x10, #10, mul vl]
// CHECK-INST: ld2h    { z21.h, z22.h }, p5/z, [x10, #10, mul vl]
// CHECK-ENCODING: [0x55,0xf5,0xa5,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a4a5f555 <unknown>
