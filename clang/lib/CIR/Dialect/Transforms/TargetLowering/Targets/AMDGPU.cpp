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
  void computeInfo(LowerFunctionInfo &fi) const override {
    llvm_unreachable("NYI");
  }
};

class AMDGPUTargetLoweringInfo : public TargetLoweringInfo {
public:
  AMDGPUTargetLoweringInfo(LowerTypes &lt)
      : TargetLoweringInfo(std::make_unique<AMDGPUABIInfo>(lt)) {}
  // Taken from here: https://llvm.org/docs/AMDGPUUsage.html#address-spaces
  unsigned getTargetAddrSpaceFromCIRAddrSpace(
      cir::AddressSpace addrSpace) const override {
    switch (addrSpace) {
    case cir::AddressSpace::OffloadPrivate:
      return 5;
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
std::unique_ptr<TargetLoweringInfo>
createAMDGPUTargetLoweringInfo(LowerModule &lowerModule) {
  return std::make_unique<AMDGPUTargetLoweringInfo>(lowerModule.getTypes());
}

} // namespace cir
