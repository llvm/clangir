//===---- TargetInfo.h - Encapsulate target details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_TARGETINFO_H
#define LLVM_CLANG_LIB_CIR_TARGETINFO_H

#include "ABIInfo.h"
#include "CIRGenValue.h"
#include "mlir/IR/Types.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Target/AArch64.h"
#include "clang/CIR/Target/x86.h"

#include <memory>

namespace clang::CIRGen {

/// isEmptyFieldForLayout - Return true if the field is "empty", that is,
/// either a zero-width bit-field or an isEmptyRecordForLayout.
bool isEmptyFieldForLayout(const ASTContext &Context, const FieldDecl *FD);

/// isEmptyRecordForLayout - Return true if a structure contains only empty
/// base classes (per  isEmptyRecordForLayout) and fields (per
/// isEmptyFieldForLayout). Note, C++ record fields are considered empty
/// if the [[no_unique_address]] attribute would have made them empty.
bool isEmptyRecordForLayout(const ASTContext &Context, QualType T);

class CIRGenFunction;
class CIRGenModule;
class CIRGenBuilderTy;

/// This class organizes various target-specific codegeneration issues, like
/// target-specific attributes, builtins and so on.
/// Equivalent to LLVM's TargetCodeGenInfo.
class TargetCIRGenInfo {
  std::unique_ptr<ABIInfo> Info = nullptr;

public:
  TargetCIRGenInfo(std::unique_ptr<ABIInfo> Info) : Info(std::move(Info)) {}

  /// Returns ABI info helper for the target.
  const ABIInfo &getABIInfo() const { return *Info; }

  /// Provides a convenient hook to handle extra target-specific attributes
  /// for the given global.
  /// In OG, the function receives an llvm::GlobalValue. However, functions
  /// and global variables are separate types in Clang IR, so we use a general
  /// mlir::Operation*.
  virtual void setTargetAttributes(const clang::Decl *decl,
                                   mlir::Operation *global,
                                   CIRGenModule &module) const {}

  virtual bool isScalarizableAsmOperand(CIRGenFunction &CGF,
                                        mlir::Type Ty) const {
    return false;
  }

  /// Performs a target specific test of a floating point value for things
  /// like IsNaN, Infinity, ... Nullptr is returned if no implementation
  /// exists.
  virtual mlir::Value testFPKind(mlir::Value V, unsigned BuiltinID,
                                 CIRGenBuilderTy &Builder,
                                 CIRGenModule &CGM) const {
    return {};
  }

  /// Corrects the MLIR type for a given constraint and "usual"
  /// type.
  ///
  /// \returns A new MLIR type, possibly the same as the original
  /// on success
  virtual mlir::Type adjustInlineAsmType(CIRGenFunction &CGF,
                                         llvm::StringRef Constraint,
                                         mlir::Type Ty) const {
    return Ty;
  }

  virtual void
  addReturnRegisterOutputs(CIRGenFunction &CGF, LValue ReturnValue,
                           std::string &Constraints,
                           std::vector<mlir::Type> &ResultRegTypes,
                           std::vector<mlir::Type> &ResultTruncRegTypes,
                           std::vector<LValue> &ResultRegDests,
                           std::string &AsmString, unsigned NumOutputs) const {}

  /// Get target favored AST address space of a global variable for languages
  /// other than OpenCL and CUDA.
  /// If \p D is nullptr, returns the default target favored address space
  /// for global variable.
  virtual clang::LangAS getGlobalVarAddressSpace(CIRGenModule &CGM,
                                                 const clang::VarDecl *D) const;

  /// Get the CIR address space for alloca.
  virtual cir::AddressSpace getCIRAllocaAddressSpace() const {
    return cir::AddressSpace::Default;
  }

  /// Perform address space cast of an expression of pointer type.
  /// \param V is the value to be casted to another address space.
  /// \param SrcAddr is the CIR address space of \p V.
  /// \param DestAddr is the targeted CIR address space.
  /// \param DestTy is the destination pointer type.
  /// \param IsNonNull is the flag indicating \p V is known to be non null.
  virtual mlir::Value performAddrSpaceCast(CIRGenFunction &CGF, mlir::Value V,
                                           cir::AddressSpace SrcAddr,
                                           cir::AddressSpace DestAddr,
                                           mlir::Type DestTy,
                                           bool IsNonNull = false) const;

  /// Get CIR calling convention for OpenCL kernel.
  virtual cir::CallingConv getOpenCLKernelCallingConv() const {
    // OpenCL kernels are called via an explicit runtime API with arguments
    // set with clSetKernelArg(), not as normal sub-functions.
    // Return SPIR_KERNEL by default as the kernel calling convention to
    // ensure the fingerprint is fixed such way that each OpenCL argument
    // gets one matching argument in the produced kernel function argument
    // list to enable feasible implementation of clSetKernelArg() with
    // aggregates etc. In case we would use the default C calling conv here,
    // clSetKernelArg() might break depending on the target-specific
    // conventions; different targets might split structs passed as values
    // to multiple function arguments etc.
    return cir::CallingConv::SpirKernel;
  }

  // Set calling convention for CUDA Kernels.
  // Some targets, such as AMD GPU or SPIRV, treat CUDA kernels as OpenCL
  // kernels. They should reset the calling convention to OpenCLKernel,
  // which will be further resolved by getOpenCLKernelCallingConv().
  virtual void setCUDAKernelCallingConvention(const FunctionType *&ft) const {}
  virtual mlir::Type getCUDADeviceBuiltinSurfaceDeviceType() const {
    return nullptr;
  }
  virtual mlir::Type getCUDADeviceBuiltinTextureDeviceType() const {
    return nullptr;
  }
  virtual ~TargetCIRGenInfo() {}
};

std::unique_ptr<TargetCIRGenInfo>
createAArch64TargetCIRGenInfo(CIRGenTypes &CGT, cir::AArch64ABIKind Kind);

std::unique_ptr<TargetCIRGenInfo>
createX86_64TargetCIRGenInfo(CIRGenTypes &CGT, cir::X86AVXABILevel AVXLevel);

std::unique_ptr<TargetCIRGenInfo> createSPIRVTargetCIRGenInfo(CIRGenTypes &CGT);

std::unique_ptr<TargetCIRGenInfo> createNVPTXTargetCIRGenInfo(CIRGenTypes &CGT);

std::unique_ptr<TargetCIRGenInfo>
createAMDGPUTargetCIRGenInfo(CIRGenTypes &CGT);

} // namespace clang::CIRGen

#endif
