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

ld1rqh  { z0.h }, p0/z, [x0]
// CHECK-INST: ld1rqh  { z0.h }, p0/z, [x0]
// CHECK-ENCODING: [0x00,0x20,0x80,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a4802000 <unknown>

ld1rqh  { z0.h }, p0/z, [x0, x0, lsl #1]
// CHECK-INST: ld1rqh  { z0.h }, p0/z, [x0, x0, lsl #1]
// CHECK-ENCODING: [0x00,0x00,0x80,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a4800000 <unknown>

ld1rqh  { z31.h }, p7/z, [sp, #-16]
// CHECK-INST: ld1rqh  { z31.h }, p7/z, [sp, #-16]
// CHECK-ENCODING: [0xff,0x3f,0x8f,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a48f3fff <unknown>

ld1rqh  { z23.h }, p3/z, [x13, #-128]
// CHECK-INST: ld1rqh  { z23.h }, p3/z, [x13, #-128]
// CHECK-ENCODING: [0xb7,0x2d,0x88,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a4882db7 <unknown>

ld1rqh  { z23.h }, p3/z, [x13, #112]
// CHECK-INST: ld1rqh  { z23.h }, p3/z, [x13, #112]
// CHECK-ENCODING: [0xb7,0x2d,0x87,0xa4]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a4872db7 <unknown>
