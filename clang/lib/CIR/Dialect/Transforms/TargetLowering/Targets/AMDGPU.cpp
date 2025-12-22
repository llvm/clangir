//===- AMDGPU.cpp - TargetInfo for AMDGPU ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "LowerFunctionInfo.h"
#include "LowerTypes.h"
#include "TargetInfo.h"
#include "TargetLoweringInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using ABIArgInfo = cir::ABIArgInfo;
using MissingFeature = cir::MissingFeatures;

namespace cir {

//===----------------------------------------------------------------------===//
// AMDGPU ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class AMDGPUABIInfo : public ABIInfo {
public:
  AMDGPUABIInfo(LowerTypes &lt) : ABIInfo(lt) {}

private:
  static const unsigned maxNumRegsForArgsRet = 16;

  unsigned numRegsForType(mlir::Type ty) const;

  // Coerce HIP scalar pointer arguments from generic pointers to global ones.
  mlir::Type coerceKernelArgumentType(mlir::Type ty, unsigned fromAS,
                                      unsigned toAS) const;

  ABIArgInfo classifyReturnType(mlir::Type ty) const;
  ABIArgInfo classifyArgumentType(mlir::Type ty, bool variadic,
                                  unsigned &numRegsLeft) const;
  ABIArgInfo classifyKernelArgumentType(mlir::Type ty) const;
  void computeInfo(LowerFunctionInfo &fi) const override;
};

class AMDGPUTargetLoweringInfo : public TargetLoweringInfo {
public:
  AMDGPUTargetLoweringInfo(LowerTypes &lt)
      : TargetLoweringInfo(std::make_unique<AMDGPUABIInfo>(lt)) {}
  // Taken from here: https://llvm.org/docs/AMDGPUUsage.html#address-spaces
  unsigned getTargetAddrSpaceFromCIRAddrSpace(
      cir::LangAddressSpace addrSpace) const override {
    switch (addrSpace) {
    case cir::LangAddressSpace::OffloadPrivate:
      return 5;
    case cir::LangAddressSpace::OffloadLocal:
      return 3;
    case cir::LangAddressSpace::OffloadGlobal:
      return 1;
    case cir::LangAddressSpace::OffloadConstant:
      return 4;
    case cir::LangAddressSpace::OffloadGeneric:
      return 0;
    default:
      cir_cconv_unreachable("Unknown CIR address space for this target");
    }
  }
};

// Estimate the number of registers the type will use
unsigned AMDGPUABIInfo::numRegsForType(mlir::Type ty) const {
  if (isAggregateTypeForABI(ty)) {
    llvm_unreachable("numRegsForType for aggregate types is NYI for AMDGPU");
  }

  uint64_t size = getContext().getTypeSize(ty);
  return (size + 31) / 32;
}

// Coerce HIP scalar pointer arguments from generic pointers to global ones.
mlir::Type AMDGPUABIInfo::coerceKernelArgumentType(mlir::Type ty,
                                                   unsigned fromAS,
                                                   unsigned toAS) const {
  if (auto ptrTy = mlir::dyn_cast<cir::PointerType>(ty)) {
    mlir::Attribute addrSpaceAttr = ptrTy.getAddrSpace();
    unsigned currentAS = 0;
    // Get the current address space.
    if (auto targetAS = mlir::dyn_cast_if_present<cir::TargetAddressSpaceAttr>(
            addrSpaceAttr))
      currentAS = targetAS.getValue();
    // If currentAS is same as the FromAS, coerce it to the ToAS.
    if (currentAS == fromAS) {
      auto newAddrSpaceAttr =
          cir::TargetAddressSpaceAttr::get(ty.getContext(), toAS);
      return cir::PointerType::get(ptrTy.getPointee(), newAddrSpaceAttr);
    }
  }
  return ty;
}

ABIArgInfo AMDGPUABIInfo::classifyReturnType(mlir::Type ty) const {
  if (isAggregateTypeForABI(ty)) {
    llvm_unreachable(
        "classifyReturnType for aggregate types is NYI for AMDGPU");
  }

  return isPromotableIntegerTypeForABI(ty) ? ABIArgInfo::getExtend(ty)
                                           : ABIArgInfo::getDirect();
}

ABIArgInfo AMDGPUABIInfo::classifyArgumentType(mlir::Type ty, bool variadic,
                                               unsigned &numRegsLeft) const {
  assert(numRegsLeft <= maxNumRegsForArgsRet && "register estimate underflow");

  ty = useFirstFieldIfTransparentUnion(ty);

  if (isAggregateTypeForABI(ty)) {
    llvm_unreachable(
        "classifyArgumentType for aggregate types is NYI for AMDGPU");
  }

  if (variadic) {
    return ABIArgInfo::getDirect(nullptr, 0, nullptr, false, 0);
  }

  ABIArgInfo argInfo =
      (isPromotableIntegerTypeForABI(ty) ? ABIArgInfo::getExtend(ty)
                                         : ABIArgInfo::getDirect());

  // Track register usage
  if (!argInfo.isIndirect()) {
    unsigned numRegs = numRegsForType(ty);
    numRegsLeft -= std::min(numRegs, numRegsLeft);
  }

  return argInfo;
}

ABIArgInfo AMDGPUABIInfo::classifyKernelArgumentType(mlir::Type ty) const {
  ty = useFirstFieldIfTransparentUnion(ty);

  // Aggregate types are not yet supported
  if (isAggregateTypeForABI(ty)) {
    llvm_unreachable("Aggregate types NYI for AMDGPU kernel arguments");
  }

  mlir::Type origTy = ty;
  mlir::Type coercedTy = origTy;

  // Determine if the target is in HIP, based on the triple.
  // TODO: use getLangOpts().HIP instead.
  const auto &Triple = getTarget().getTriple();
  bool isHIP = Triple.getArch() == llvm::Triple::amdgcn &&
               Triple.getOS() == llvm::Triple::AMDHSA;

  // For HIP, coerce pointer arguments from generic to global
  if (isHIP) {
    unsigned genericAS =
        getTarget().getTargetAddressSpace(clang::LangAS::Default);
    unsigned globalAS =
        getTarget().getTargetAddressSpace(clang::LangAS::cuda_device);
    coercedTy = coerceKernelArgumentType(origTy, genericAS, globalAS);
  }

  // If we set CanBeFlattened to true, CodeGen will expand the struct to its
  // individual elements, which confuses the Clover OpenCL backend; therefore we
  // have to set it to false here. Other args of getDirect() are just defaults.
  return ABIArgInfo::getDirect(coercedTy, 0, nullptr, false);
}

void AMDGPUABIInfo::computeInfo(LowerFunctionInfo &fi) const {
  const unsigned cc = fi.getCallingConvention();

  if (!getCXXABI().classifyReturnType(fi))
    fi.getReturnInfo() = classifyReturnType(fi.getReturnType());

  unsigned argumentIndex = 0;
  const unsigned numFixedArguments = fi.getNumRequiredArgs();

  unsigned numRegsLeft = maxNumRegsForArgsRet;
  for (auto &arg : fi.arguments()) {
    if (cc == llvm::CallingConv::AMDGPU_KERNEL) {
      arg.info = classifyKernelArgumentType(arg.type);
    } else {
      bool fixedArgument = argumentIndex++ < numFixedArguments;
      arg.info = classifyArgumentType(arg.type, !fixedArgument, numRegsLeft);
    }
  }
}

} // namespace

std::unique_ptr<TargetLoweringInfo>
createAMDGPUTargetLoweringInfo(LowerModule &lowerModule) {
  return std::make_unique<AMDGPUTargetLoweringInfo>(lowerModule.getTypes());
}

} // namespace cir
