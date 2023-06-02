//===--- DebugInfo.cpp - DebugInfo for CIR or LLVM CodeGen ------*- C++ -*-===//
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
#include "clang/Basic/DebugInfo.h"
#include "clang/Basic/Version.h"
#include "llvm/BinaryFormat/Dwarf.h"

namespace clang {

DebugCUInfo getDebugCUInfoBundle(const CodeGenOptions &CGO,
                                 const LangOptions &LO) {
  llvm::dwarf::SourceLanguage LangTag;
  if (LO.CPlusPlus) {
    if (LO.ObjC)
      LangTag = llvm::dwarf::DW_LANG_ObjC_plus_plus;
    else if (CGO.DebugStrictDwarf && CGO.DwarfVersion < 5)
      LangTag = llvm::dwarf::DW_LANG_C_plus_plus;
    else if (LO.CPlusPlus14)
      LangTag = llvm::dwarf::DW_LANG_C_plus_plus_14;
    else if (LO.CPlusPlus11)
      LangTag = llvm::dwarf::DW_LANG_C_plus_plus_11;
    else
      LangTag = llvm::dwarf::DW_LANG_C_plus_plus;
  } else if (LO.ObjC) {
    LangTag = llvm::dwarf::DW_LANG_ObjC;
  } else if (LO.OpenCL && (!CGO.DebugStrictDwarf || CGO.DwarfVersion >= 5)) {
    LangTag = llvm::dwarf::DW_LANG_OpenCL;
  } else if (LO.RenderScript) {
    LangTag = llvm::dwarf::DW_LANG_GOOGLE_RenderScript;
  } else if (LO.C11 && !(CGO.DebugStrictDwarf && CGO.DwarfVersion < 5)) {
    LangTag = llvm::dwarf::DW_LANG_C11;
  } else if (LO.C99) {
    LangTag = llvm::dwarf::DW_LANG_C99;
  } else {
    LangTag = llvm::dwarf::DW_LANG_C89;
  }

  // Figure out which version of the ObjC runtime we have.
  unsigned RuntimeVers = 0;
  if (LO.ObjC)
    RuntimeVers = LO.ObjCRuntime.isNonFragile() ? 2 : 1;

  llvm::DICompileUnit::DebugEmissionKind EmissionKind;
  switch (CGO.getDebugInfo()) {
  case llvm::codegenoptions::NoDebugInfo:
  case llvm::codegenoptions::LocTrackingOnly:
    EmissionKind = llvm::DICompileUnit::NoDebug;
    break;
  case llvm::codegenoptions::DebugLineTablesOnly:
    EmissionKind = llvm::DICompileUnit::LineTablesOnly;
    break;
  case llvm::codegenoptions::DebugDirectivesOnly:
    EmissionKind = llvm::DICompileUnit::DebugDirectivesOnly;
    break;
  case llvm::codegenoptions::DebugInfoConstructor:
  case llvm::codegenoptions::LimitedDebugInfo:
  case llvm::codegenoptions::FullDebugInfo:
  case llvm::codegenoptions::UnusedTypeInfo:
    EmissionKind = llvm::DICompileUnit::FullDebug;
    break;
  }

  DebugCUInfo Info;
  Info.LangTag = LangTag;
  Info.Producer = CGO.EmitVersionIdentMetadata ? getClangFullVersion() : "";
  Info.IsOptimized = LO.Optimize || CGO.PrepareForLTO || CGO.PrepareForThinLTO;
  Info.DwarfDebugFlags = CGO.DwarfDebugFlags;
  Info.RuntimeVers = RuntimeVers;
  Info.SplitDwarfFileName = CGO.SplitDwarfFile;
  Info.SplitDwarfInlining = CGO.SplitDwarfInlining;
  Info.DebugInfoForProfiling = CGO.DebugInfoForProfiling;
  Info.EmissionKind = EmissionKind;
  Info.NameTableKind =
      static_cast<llvm::DICompileUnit::DebugNameTableKind>(CGO.DebugNameTable);
  Info.DebugRangesBaseAddress = CGO.DebugRangesBaseAddress;
  return Info;
}

}  // end namespace clang
