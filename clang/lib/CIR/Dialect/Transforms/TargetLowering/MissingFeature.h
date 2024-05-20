//===--- MissingFeatures.cpp - Markers for missing C/C++ features in CIR --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Static methods declared here are spread throughout the CIR lowering codebase
// to track missing features in CIR that should eventually be implemented. Add
// new methods as needed.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
namespace cir {

struct MissingFeature {

  // Parameters may have additional attributes (e.g. [[noescape]]) that affect
  // the compiler. This is not yet supported in CIR.
  static bool extParamInfo() { return true; }

  // LangOpts may affect lowering, but we do not carry this information into CIR
  // just yet. Right now, it only instantiates the default lang options.
  static bool langOpts() { return true; }

  // Several type qualifiers are not yet supported in CIR, but important when
  // evaluating ABI-specific lowering.
  static bool qualifiedTypes() { return true; }

  // We're ignoring several details regarding ABI-halding for Swift.
  static bool swift() { return true; }

  // Despite carrying some information about variadics, we are currently
  // ignoring this to focus only on the code necessary to lower non-variadics.
  static bool variadicFunctions() { return true; }
};

} // namespace cir
} // namespace mlir
