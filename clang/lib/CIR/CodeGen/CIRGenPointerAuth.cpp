//===--- CIRGenPointerAuth.cpp - CIR generation for ptr auth --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains common routines relating to the emission of
// pointer authentication operations.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"

using namespace clang;
using namespace cir;

Address CIRGenFunction::getAsNaturalAddressOf(Address Addr,
                                              QualType PointeeTy) {
  assert(!MissingFeatures::ptrAuth() && "NYI");
  return Addr;
}