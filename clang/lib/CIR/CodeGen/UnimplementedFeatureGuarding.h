//===---- UnimplementedFeatureGuarding.h - Checks against NYI ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file introduces some helper classes to guard against features that
// CodeGen supports that we do not have and also do not have great ways to
// assert against.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_UFG
#define LLVM_CLANG_LIB_CIR_UFG

namespace cir {
struct UnimplementedFeature {
  // TODO(CIR): Implement the CIRGenFunction::buildTypeCheck method that handles
  // sanitizer related type check features
  static bool buildTypeCheck() { return false; }
  static bool tbaa() { return false; }
  static bool cleanups() { return false; }
  // This is for whether or not we've implemented a cir::VectorType
  // corresponding to `llvm::VectorType`
  static bool cirVectorType() { return false; }

  // Address space related
  static bool addressSpace() { return false; }
  static bool addressSpaceInGlobalVar() { return false; }
  static bool getASTAllocaAddressSpace() { return false; }

  // Unhandled global/linkage information.
  static bool unnamedAddr() { return false; }
  static bool setComdat() { return false; }
  static bool setDSOLocal() { return false; }
  static bool threadLocal() { return false; }
  static bool setDLLStorageClass() { return false; }
  static bool setDLLImportDLLExport() { return false; }
  static bool setPartition() { return false; }
  static bool setGlobalVisibility() { return false; }
  static bool hiddenVisibility() { return false; }
  static bool protectedVisibility() { return false; }
  static bool addCompilerUsedGlobal() { return false; }

  // Sanitizers
  static bool reportGlobalToASan() { return false; }
  static bool emitAsanPrologueOrEpilogue() { return false; }
  static bool emitCheckedInBoundsGEP() { return false; }

  // ObjC
  static bool setObjCGCLValueClass() { return false; }

  // Debug info
  static bool generateDebugInfo() { return false; }

  // Coroutines
  static bool unhandledException() { return false; }

  static bool capturedByInit() { return false; }
  static bool tryEmitAsConstant() { return false; }
  static bool incrementProfileCounter() { return false; }
  static bool requiresReturnValueCheck() { return false; }
  static bool shouldEmitLifetimeMarkers() { return false; }
  static bool peepholeProtection() { return false; }
  static bool attributeNoBuiltin() { return false; }
  static bool CGCapturedStmtInfo() { return false; }
  static bool cxxABI() { return false; }
  static bool openCL() { return false; }
  static bool openMP() { return false; }
  static bool ehStack() { return false; }
  static bool isVarArg() { return false; }
};
} // namespace cir

#endif
