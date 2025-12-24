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
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
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
  static const unsigned MaxNumRegsForArgsRet = 16;

  ABIArgInfo classifyReturnType(mlir::Type ty) const;
  ABIArgInfo classifyArgumentType(mlir::Type Ty, bool Variadic,
                                  unsigned &NumRegsLeft) const;

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

} // namespace

ABIArgInfo AMDGPUABIInfo::classifyReturnType(mlir::Type ty) const {
  if (llvm::isa<VoidType>(ty))
    return ABIArgInfo::getIgnore();

  if (getContext().getLangOpts().OpenMP)
    llvm_unreachable("NYI");

  if (!isScalarType(ty))
    return ABIArgInfo::getDirect();

  // OG treats enums as their underlying type.
  // This has already been done for CIR.

  // Integers with size < 32 must be extended to 32 bits.
  // (See Section 3.3 of PTX ABI.)
  return (isPromotableIntegerTypeForABI(ty) ? ABIArgInfo::getExtend(ty)
                                            : ABIArgInfo::getDirect());
}

/// For kernels all parameters are really passed in a special buffer. It doesn't
/// make sense to pass anything byval, so everything must be direct.
ABIArgInfo AMDGPUABIInfo::classifyKernelArgumentType(mlir::Type ty) const {
  return ABIArgInfo::getDirect();
}

ABIArgInfo AMDGPUABIInfo::classifyArgumentType(mlir::Type ty, bool variadic,
                                               unsigned &numRegsLeft) const {
  assert(numRegsLeft <= MaxNumRegsForArgsRet && "register estimate underflow");

  ty = useFirstFieldIfTransparentUnion(ty);

  // Variadic arguments: always direct.
  if (variadic) {
    return ABIArgInfo::getDirect();
  }

  // Aggregate (struct/array) handling
  if (isAggregateTypeForABI(ty)) {
    llvm_unreachable("NYI");
  }

  // === Non-aggregate fallback ===
  ABIArgInfo Info = isPromotableIntegerTypeForABI(ty)
                        ? ABIArgInfo::getExtend(ty)
                        : ABIArgInfo::getDirect();

  return Info;
}

void AMDGPUABIInfo::computeInfo(LowerFunctionInfo &fi) const {
  llvm::CallingConv::ID cc = fi.getCallingConvention();

  if (!getCXXABI().classifyReturnType(fi))
    fi.getReturnInfo() = classifyReturnType(fi.getReturnType());

  unsigned argumentIndex = 0;
  const unsigned numFixedArguments = fi.getNumRequiredArgs();

  unsigned numRegsLeft = MaxNumRegsForArgsRet;
  for (auto &argument : fi.arguments()) {
    if (cc == llvm::CallingConv::AMDGPU_KERNEL) {
      argument.info = classifyKernelArgumentType(argument.type);
    } else {
      bool fixedArgument = argumentIndex++ < numFixedArguments;
      argument.info =
          classifyArgumentType(argument.type, !fixedArgument, numRegsLeft);
    }
  }
}

std::unique_ptr<TargetLoweringInfo>
createAMDGPUTargetLoweringInfo(LowerModule &lowerModule) {
  return std::make_unique<AMDGPUTargetLoweringInfo>(lowerModule.getTypes());
}

} // namespace cir
