//===- DataLayout.cpp - Data size & alignment routines ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics llvm/lib/IR/DataLayout.cpp. The queries are
// adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "DataLayout.h"

namespace mlir {
namespace cir {

void CIRDataLayout::reset(StringRef Desc) { clear(); }

void CIRDataLayout::clear() {}

} // end namespace cir
} // end namespace mlir
