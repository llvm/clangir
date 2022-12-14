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


// ---------------------------------------------------------------------------//
// Test 64-bit form (x0) and its aliases
// ---------------------------------------------------------------------------//

uqinch  x0
// CHECK-INST: uqinch  x0
// CHECK-ENCODING: [0xe0,0xf7,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f7e0 <unknown>

uqinch  x0, all
// CHECK-INST: uqinch  x0
// CHECK-ENCODING: [0xe0,0xf7,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f7e0 <unknown>

uqinch  x0, all, mul #1
// CHECK-INST: uqinch  x0
// CHECK-ENCODING: [0xe0,0xf7,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f7e0 <unknown>

uqinch  x0, all, mul #16
// CHECK-INST: uqinch  x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xf7,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 047ff7e0 <unknown>


// ---------------------------------------------------------------------------//
// Test 32-bit form (w0) and its aliases
// ---------------------------------------------------------------------------//

uqinch  w0
// CHECK-INST: uqinch  w0
// CHECK-ENCODING: [0xe0,0xf7,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460f7e0 <unknown>

uqinch  w0, all
// CHECK-INST: uqinch  w0
// CHECK-ENCODING: [0xe0,0xf7,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460f7e0 <unknown>

uqinch  w0, all, mul #1
// CHECK-INST: uqinch  w0
// CHECK-ENCODING: [0xe0,0xf7,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460f7e0 <unknown>

uqinch  w0, all, mul #16
// CHECK-INST: uqinch  w0, all, mul #16
// CHECK-ENCODING: [0xe0,0xf7,0x6f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 046ff7e0 <unknown>

uqinch  w0, pow2
// CHECK-INST: uqinch  w0, pow2
// CHECK-ENCODING: [0x00,0xf4,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460f400 <unknown>

uqinch  w0, pow2, mul #16
// CHECK-INST: uqinch  w0, pow2, mul #16
// CHECK-ENCODING: [0x00,0xf4,0x6f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 046ff400 <unknown>


// ---------------------------------------------------------------------------//
// Test vector form and aliases.
// ---------------------------------------------------------------------------//

uqinch  z0.h
// CHECK-INST: uqinch  z0.h
// CHECK-ENCODING: [0xe0,0xc7,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460c7e0 <unknown>

uqinch  z0.h, all
// CHECK-INST: uqinch  z0.h
// CHECK-ENCODING: [0xe0,0xc7,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460c7e0 <unknown>

uqinch  z0.h, all, mul #1
// CHECK-INST: uqinch  z0.h
// CHECK-ENCODING: [0xe0,0xc7,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460c7e0 <unknown>

uqinch  z0.h, all, mul #16
// CHECK-INST: uqinch  z0.h, all, mul #16
// CHECK-ENCODING: [0xe0,0xc7,0x6f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 046fc7e0 <unknown>

uqinch  z0.h, pow2
// CHECK-INST: uqinch  z0.h, pow2
// CHECK-ENCODING: [0x00,0xc4,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460c400 <unknown>

uqinch  z0.h, pow2, mul #16
// CHECK-INST: uqinch  z0.h, pow2, mul #16
// CHECK-ENCODING: [0x00,0xc4,0x6f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 046fc400 <unknown>


// ---------------------------------------------------------------------------//
// Test all patterns for 64-bit form
// ---------------------------------------------------------------------------//

uqinch  x0, pow2
// CHECK-INST: uqinch  x0, pow2
// CHECK-ENCODING: [0x00,0xf4,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f400 <unknown>

uqinch  x0, vl1
// CHECK-INST: uqinch  x0, vl1
// CHECK-ENCODING: [0x20,0xf4,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f420 <unknown>

uqinch  x0, vl2
// CHECK-INST: uqinch  x0, vl2
// CHECK-ENCODING: [0x40,0xf4,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f440 <unknown>

uqinch  x0, vl3
// CHECK-INST: uqinch  x0, vl3
// CHECK-ENCODING: [0x60,0xf4,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f460 <unknown>

uqinch  x0, vl4
// CHECK-INST: uqinch  x0, vl4
// CHECK-ENCODING: [0x80,0xf4,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f480 <unknown>

uqinch  x0, vl5
// CHECK-INST: uqinch  x0, vl5
// CHECK-ENCODING: [0xa0,0xf4,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f4a0 <unknown>

uqinch  x0, vl6
// CHECK-INST: uqinch  x0, vl6
// CHECK-ENCODING: [0xc0,0xf4,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f4c0 <unknown>

uqinch  x0, vl7
// CHECK-INST: uqinch  x0, vl7
// CHECK-ENCODING: [0xe0,0xf4,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f4e0 <unknown>

uqinch  x0, vl8
// CHECK-INST: uqinch  x0, vl8
// CHECK-ENCODING: [0x00,0xf5,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f500 <unknown>

uqinch  x0, vl16
// CHECK-INST: uqinch  x0, vl16
// CHECK-ENCODING: [0x20,0xf5,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f520 <unknown>

uqinch  x0, vl32
// CHECK-INST: uqinch  x0, vl32
// CHECK-ENCODING: [0x40,0xf5,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f540 <unknown>

uqinch  x0, vl64
// CHECK-INST: uqinch  x0, vl64
// CHECK-ENCODING: [0x60,0xf5,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f560 <unknown>

uqinch  x0, vl128
// CHECK-INST: uqinch  x0, vl128
// CHECK-ENCODING: [0x80,0xf5,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f580 <unknown>

uqinch  x0, vl256
// CHECK-INST: uqinch  x0, vl256
// CHECK-ENCODING: [0xa0,0xf5,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f5a0 <unknown>

uqinch  x0, #14
// CHECK-INST: uqinch  x0, #14
// CHECK-ENCODING: [0xc0,0xf5,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f5c0 <unknown>

uqinch  x0, #15
// CHECK-INST: uqinch  x0, #15
// CHECK-ENCODING: [0xe0,0xf5,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f5e0 <unknown>

uqinch  x0, #16
// CHECK-INST: uqinch  x0, #16
// CHECK-ENCODING: [0x00,0xf6,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f600 <unknown>

uqinch  x0, #17
// CHECK-INST: uqinch  x0, #17
// CHECK-ENCODING: [0x20,0xf6,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f620 <unknown>

uqinch  x0, #18
// CHECK-INST: uqinch  x0, #18
// CHECK-ENCODING: [0x40,0xf6,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f640 <unknown>

uqinch  x0, #19
// CHECK-INST: uqinch  x0, #19
// CHECK-ENCODING: [0x60,0xf6,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f660 <unknown>

uqinch  x0, #20
// CHECK-INST: uqinch  x0, #20
// CHECK-ENCODING: [0x80,0xf6,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f680 <unknown>

uqinch  x0, #21
// CHECK-INST: uqinch  x0, #21
// CHECK-ENCODING: [0xa0,0xf6,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f6a0 <unknown>

uqinch  x0, #22
// CHECK-INST: uqinch  x0, #22
// CHECK-ENCODING: [0xc0,0xf6,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f6c0 <unknown>

uqinch  x0, #23
// CHECK-INST: uqinch  x0, #23
// CHECK-ENCODING: [0xe0,0xf6,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f6e0 <unknown>

uqinch  x0, #24
// CHECK-INST: uqinch  x0, #24
// CHECK-ENCODING: [0x00,0xf7,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f700 <unknown>

uqinch  x0, #25
// CHECK-INST: uqinch  x0, #25
// CHECK-ENCODING: [0x20,0xf7,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f720 <unknown>

uqinch  x0, #26
// CHECK-INST: uqinch  x0, #26
// CHECK-ENCODING: [0x40,0xf7,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f740 <unknown>

uqinch  x0, #27
// CHECK-INST: uqinch  x0, #27
// CHECK-ENCODING: [0x60,0xf7,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f760 <unknown>

uqinch  x0, #28
// CHECK-INST: uqinch  x0, #28
// CHECK-ENCODING: [0x80,0xf7,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f780 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

uqinch  z0.h
// CHECK-INST: uqinch	z0.h
// CHECK-ENCODING: [0xe0,0xc7,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460c7e0 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

uqinch  z0.h, pow2, mul #16
// CHECK-INST: uqinch	z0.h, pow2, mul #16
// CHECK-ENCODING: [0x00,0xc4,0x6f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 046fc400 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

uqinch  z0.h, pow2
// CHECK-INST: uqinch	z0.h, pow2
// CHECK-ENCODING: [0x00,0xc4,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460c400 <unknown>
