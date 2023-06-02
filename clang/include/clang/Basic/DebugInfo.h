//===--- DebugInfo.h - DebugInfo for CIR or LLVM CodeGen --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the source-level debug info generator for llvm translation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_DEBUG_INFO_H
#define LLVM_CLANG_BASIC_DEBUG_INFO_H

#include "clang/Basic/LangOptions.h"
#include "clang/Basic/CodeGenOptions.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/Support/DataTypes.h"
#include <string>


namespace clang {
struct DebugCUInfo {
  unsigned LangTag;
  std::string Producer;
  bool IsOptimized;
  std::string DwarfDebugFlags;
  unsigned RuntimeVers;
  std::string SplitDwarfFileName;
  bool SplitDwarfInlining;
  bool DebugInfoForProfiling;
  llvm::DICompileUnit::DebugEmissionKind EmissionKind;
  llvm::DICompileUnit::DebugNameTableKind NameTableKind;
  bool DebugRangesBaseAddress;
};

DebugCUInfo getDebugCUInfoBundle(const CodeGenOptions &CGO,
                                 const LangOptions &LO);
} // end namespace clang

#endif
