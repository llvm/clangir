// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN



orr     z5.b, z5.b, #0xf9
// CHECK-INST: orr     z5.b, z5.b, #0xf9
// CHECK-ENCODING: [0xa5,0x2e,0x00,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05002ea5 <unknown>

orr     z23.h, z23.h, #0xfff9
// CHECK-INST: orr     z23.h, z23.h, #0xfff9
// CHECK-ENCODING: [0xb7,0x6d,0x00,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05006db7 <unknown>

orr     z0.s, z0.s, #0xfffffff9
// CHECK-INST: orr     z0.s, z0.s, #0xfffffff9
// CHECK-ENCODING: [0xa0,0xeb,0x00,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0500eba0 <unknown>

orr     z0.d, z0.d, #0xfffffffffffffff9
// CHECK-INST: orr     z0.d, z0.d, #0xfffffffffffffff9
// CHECK-ENCODING: [0xa0,0xef,0x03,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0503efa0 <unknown>

orr     z5.b, z5.b, #0x6
// CHECK-INST: orr     z5.b, z5.b, #0x6
// CHECK-ENCODING: [0x25,0x3e,0x00,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05003e25 <unknown>

orr     z23.h, z23.h, #0x6
// CHECK-INST: orr     z23.h, z23.h, #0x6
// CHECK-ENCODING: [0x37,0x7c,0x00,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05007c37 <unknown>

orr     z0.s, z0.s, #0x6
// CHECK-INST: orr     z0.s, z0.s, #0x6
// CHECK-ENCODING: [0x20,0xf8,0x00,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0500f820 <unknown>

orr     z0.d, z0.d, #0x6
// CHECK-INST: orr     z0.d, z0.d, #0x6
// CHECK-ENCODING: [0x20,0xf8,0x03,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0503f820 <unknown>

orr     z0.d, z0.d, z0.d    // should use mov-alias
// CHECK-INST: mov     z0.d, z0.d
// CHECK-ENCODING: [0x00,0x30,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04603000 <unknown>

orr     z23.d, z13.d, z8.d  // should not use mov-alias
// CHECK-INST: orr     z23.d, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0x31,0x68,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 046831b7 <unknown>

orr     z31.b, p7/m, z31.b, z31.b
// CHECK-INST: orr     z31.b, p7/m, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x1f,0x18,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04181fff <unknown>

orr     z31.h, p7/m, z31.h, z31.h
// CHECK-INST: orr     z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x1f,0x58,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04581fff <unknown>

orr     z31.s, p7/m, z31.s, z31.s
// CHECK-INST: orr     z31.s, p7/m, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x1f,0x98,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04981fff <unknown>

orr     z31.d, p7/m, z31.d, z31.d
// CHECK-INST: orr     z31.d, p7/m, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x1f,0xd8,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d81fff <unknown>

orr     p0.b, p0/z, p0.b, p1.b
// CHECK-INST: orr     p0.b, p0/z, p0.b, p1.b
// CHECK-ENCODING: [0x00,0x40,0x81,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25814000 <unknown>

orr     p0.b, p0/z, p0.b, p0.b
// CHECK-INST: mov     p0.b, p0.b
// CHECK-ENCODING: [0x00,0x40,0x80,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25804000 <unknown>

orr     p15.b, p15/z, p15.b, p15.b
// CHECK-INST: mov     p15.b, p15.b
// CHECK-ENCODING: [0xef,0x7d,0x8f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 258f7def <unknown>


// --------------------------------------------------------------------------//
// Test aliases.

orr     z0.s, z0.s, z0.s
// CHECK-INST: mov     z0.d, z0.d
// CHECK-ENCODING: [0x00,0x30,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04603000 <unknown>

orr     z0.h, z0.h, z0.h
// CHECK-INST: mov     z0.d, z0.d
// CHECK-ENCODING: [0x00,0x30,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04603000 <unknown>

orr     z0.b, z0.b, z0.b
// CHECK-INST: mov     z0.d, z0.d
// CHECK-ENCODING: [0x00,0x30,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04603000 <unknown>

orr     z23.s, z13.s, z8.s  // should not use mov-alias
// CHECK-INST: orr     z23.d, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0x31,0x68,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 046831b7 <unknown>

orr     z23.h, z13.h, z8.h  // should not use mov-alias
// CHECK-INST: orr     z23.d, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0x31,0x68,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 046831b7 <unknown>

orr     z23.b, z13.b, z8.b  // should not use mov-alias
// CHECK-INST: orr     z23.d, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0x31,0x68,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 046831b7 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4.d, p7/z, z6.d
// CHECK-INST: movprfx	z4.d, p7/z, z6.d
// CHECK-ENCODING: [0xc4,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d03cc4 <unknown>

orr     z4.d, p7/m, z4.d, z31.d
// CHECK-INST: orr	z4.d, p7/m, z4.d, z31.d
// CHECK-ENCODING: [0xe4,0x1f,0xd8,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d81fe4 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcc4 <unknown>

orr     z4.d, p7/m, z4.d, z31.d
// CHECK-INST: orr	z4.d, p7/m, z4.d, z31.d
// CHECK-ENCODING: [0xe4,0x1f,0xd8,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d81fe4 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

orr     z0.d, z0.d, #0x6
// CHECK-INST: orr	z0.d, z0.d, #0x6
// CHECK-ENCODING: [0x20,0xf8,0x03,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0503f820 <unknown>
