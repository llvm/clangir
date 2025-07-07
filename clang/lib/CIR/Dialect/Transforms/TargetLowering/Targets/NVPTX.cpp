//===- NVPTX.cpp - TargetInfo for NVPTX -----------------------------------===//
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
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using ABIArgInfo = cir::ABIArgInfo;
using MissingFeature = cir::MissingFeatures;

namespace cir {

//===----------------------------------------------------------------------===//
// NVPTX ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class NVPTXABIInfo : public ABIInfo {
public:
  NVPTXABIInfo(LowerTypes &lt) : ABIInfo(lt) {}

private:
  ABIArgInfo classifyReturnType(mlir::Type ty) const;
  ABIArgInfo classifyArgumentType(mlir::Type ty) const;

  void computeInfo(LowerFunctionInfo &fi) const override;
};

class NVPTXTargetLoweringInfo : public TargetLoweringInfo {
public:
  NVPTXTargetLoweringInfo(LowerTypes &lt)
      : TargetLoweringInfo(std::make_unique<NVPTXABIInfo>(lt)) {}

  unsigned getTargetAddrSpaceFromCIRAddrSpace(
      cir::AddressSpace addrSpace) const override {
    switch (addrSpace) {
    case cir::AddressSpace::OffloadPrivate:
      return 0;
    case cir::AddressSpace::OffloadLocal:
      return 3;
    case cir::AddressSpace::OffloadGlobal:
      return 1;
    case cir::AddressSpace::OffloadConstant:
      return 4;
    case cir::AddressSpace::OffloadGeneric:
      return 0;
    default:
      cir_cconv_unreachable("Unknown CIR address space for this target");
    }
  }
};

} // namespace

ABIArgInfo NVPTXABIInfo::classifyReturnType(mlir::Type ty) const {
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

ABIArgInfo NVPTXABIInfo::classifyArgumentType(mlir::Type ty) const {
  if (isAggregateTypeForABI(ty))
    llvm_unreachable("NYI");

  if (auto intType = llvm::dyn_cast<IntType>(ty)) {
    if (intType.getWidth() > 128)
      llvm_unreachable("NYI");
  }

  return (isPromotableIntegerTypeForABI(ty) ? ABIArgInfo::getExtend(ty)
                                            : ABIArgInfo::getDirect());
}

void NVPTXABIInfo::computeInfo(LowerFunctionInfo &fi) const {
  if (!getCXXABI().classifyReturnType(fi))
    fi.getReturnInfo() = classifyReturnType(fi.getReturnType());

  for (auto &&[count, argument] : llvm::enumerate(fi.arguments()))
    argument.info = count < fi.getNumRequiredArgs()
                        ? classifyArgumentType(argument.type)
                        : ABIArgInfo::getDirect();
}

std::unique_ptr<TargetLoweringInfo>
createNVPTXTargetLoweringInfo(LowerModule &lowerModule) {
  return std::make_unique<NVPTXTargetLoweringInfo>(lowerModule.getTypes());
}

} // namespace cir
