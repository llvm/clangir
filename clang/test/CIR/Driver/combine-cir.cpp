// NOTE: This file exists only to host RUN lines; -cir-combine should reject
// positional inputs, so we never pass %s as an input.
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: printf "module {}\n" > %t/host.cir
// RUN: printf "module {}\n" > %t/device.cir

//------------------------------------------------------------------------------
// Positive / baseline: all required args present.
//------------------------------------------------------------------------------
//
// RUN: %clang_cc1 -fclangir -cir-combine \
// RUN:   -cir-host-input %t/host.cir \
// RUN:   -cir-device-input %t/device.cir \
// RUN:   -o %t/out.cir
// RUN: test -f %t/out.cir
// RUN: FileCheck %s --check-prefix=OK < %t/out.cir
//
// OK: cir.offload.container

//------------------------------------------------------------------------------
// Missing -o
//------------------------------------------------------------------------------
//
// RUN: not %clang_cc1 -fclangir -cir-combine \
// RUN:   -cir-host-input %t/host.cir \
// RUN:   -cir-device-input %t/device.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO_O
//
// NO_O: error: missing argument to '-o'

//------------------------------------------------------------------------------
// Missing host input
//------------------------------------------------------------------------------
//
// RUN: not %clang_cc1 -fclangir -cir-combine \
// RUN:   -cir-device-input %t/device.cir \
// RUN:   -o %t/out.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO_HOST
//
// NO_HOST: error:
// NO_HOST-SAME: -cir-host-input

//------------------------------------------------------------------------------
// Missing device input(s)
//------------------------------------------------------------------------------
//
// RUN: not %clang_cc1 -fclangir -cir-combine \
// RUN:   -cir-host-input %t/host.cir \
// RUN:   -o %t/out.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO_DEVICE
//
// NO_DEVICE: error:
// NO_DEVICE-SAME: -cir-device-input

//------------------------------------------------------------------------------
// Multiple host inputs (should be rejected)
//------------------------------------------------------------------------------
//
// RUN: not %clang_cc1 -fclangir -cir-combine \
// RUN:   -cir-host-input %t/host.cir \
// RUN:   -cir-host-input %t/host.cir \
// RUN:   -cir-device-input %t/device.cir \
// RUN:   -o %t/out.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MULTI_HOST
//
// MULTI_HOST: error:
// MULTI_HOST-SAME: -cir-host-input

//------------------------------------------------------------------------------
// Non-existent host file
//------------------------------------------------------------------------------
//
// RUN: not %clang_cc1 -fclangir -cir-combine \
// RUN:   -cir-host-input %t/does-not-exist.cir \
// RUN:   -cir-device-input %t/device.cir \
// RUN:   -o %t/out.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO_SUCH_HOST
//
// NO_SUCH_HOST: error: no such file or directory:

//------------------------------------------------------------------------------
// Non-existent device file
//------------------------------------------------------------------------------
//
// RUN: not %clang_cc1 -fclangir -cir-combine \
// RUN:   -cir-host-input %t/host.cir \
// RUN:   -cir-device-input %t/does-not-exist.cir \
// RUN:   -o %t/out.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO_SUCH_DEVICE
//
// NO_SUCH_DEVICE: error: no such file or directory:

//------------------------------------------------------------------------------
// CIR split outputs to host/device 
//------------------------------------------------------------------------------
// RUN: %clang_cc1 -fclangir -cir-combine -cir-emit-split \
// RUN:   -cir-host-input %t/host.cir \
// RUN:   -cir-device-input %t/device.cir \
// RUN:   -cir-host-output %t/host.out.cir \
// RUN:   -cir-device-output %t/device.out.cir
// RUN: test -f %t/host.out.cir
// RUN: test -f %t/device.out.cir

//------------------------------------------------------------------------------
// CIR split outputs missing device output 
//------------------------------------------------------------------------------
// RUN: not %clang_cc1 -fclangir -cir-combine -cir-emit-split \
// RUN:   -cir-host-input %t/host.cir \
// RUN:   -cir-device-input %t/device.cir \
// RUN:   -cir-host-output %t/host.out.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SPLIT_MISSING_DEVOUT
// SPLIT_MISSING_DEVOUT: error: argument to '-cir-device-output' is missing (expected 1 value)


//------------------------------------------------------------------------------
// CIR split outputs missing host output 
//------------------------------------------------------------------------------
// RUN: not %clang_cc1 -fclangir -cir-combine -cir-emit-split \
// RUN:   -cir-host-input %t/host.cir \
// RUN:   -cir-device-input %t/device.cir \
// RUN:   -cir-device-output %t/device.out.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SPLIT_MISSING_HOSTOUT
// SPLIT_MISSING_HOSTOUT: error: argument to '-cir-host-output' is missing (expected 1 value)


//------------------------------------------------------------------------------
// CIR split outputs to host/device, pass unused '-o'
//------------------------------------------------------------------------------
// RUN: %clang_cc1 -fclangir -cir-combine -cir-emit-split \
// RUN:   -cir-host-input %t/host.cir -o %t/out.cir \
// RUN:   -cir-device-input %t/device.cir \
// RUN:   -cir-host-output %t/host.out.cir \
// RUN:   -cir-device-output %t/device.out.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EXTRA_OUT
// RUN: test -f %t/host.out.cir
// RUN: test -f %t/device.out.cir
// EXTRA_OUT: warning: ignoring '-o' option as option '-cir-emit-split' overrides the behavior
