//===- AArch64.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Target/AArch64.h"
#include "ABIInfoImpl.h"
#include "LowerFunctionInfo.h"
#include "LowerTypes.h"
#include "TargetInfo.h"
#include "TargetLoweringInfo.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

using AArch64ABIKind = ::cir::AArch64ABIKind;
using ABIArgInfo = ::cir::ABIArgInfo;
using MissingFeature = ::cir::MissingFeatures;

namespace mlir {
namespace cir {

//===----------------------------------------------------------------------===//
// AArch64 ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class AArch64ABIInfo : public ABIInfo {
  AArch64ABIKind kind;

public:
  AArch64ABIInfo(LowerTypes &cgt, AArch64ABIKind kind)
      : ABIInfo(cgt), kind(kind) {}

private:
  AArch64ABIKind getABIKind() const { return kind; }
  bool isDarwinPCS() const { return kind == AArch64ABIKind::DarwinPCS; }

  ABIArgInfo classifyReturnType(Type retTy, bool isVariadic) const;
  ABIArgInfo classifyArgumentType(Type retTy, bool isVariadic,
                                  unsigned callingConvention) const;

  void computeInfo(LowerFunctionInfo &fi) const override {
    if (!::mlir::cir::classifyReturnType(getCXXABI(), fi, *this))
      fi.getReturnInfo() =
          classifyReturnType(fi.getReturnType(), fi.isVariadic());

    for (auto &it : fi.arguments())
      it.info = classifyArgumentType(it.type, fi.isVariadic(),
                                     fi.getCallingConvention());
  }
};

class AArch64TargetLoweringInfo : public TargetLoweringInfo {
public:
  AArch64TargetLoweringInfo(LowerTypes &lt, AArch64ABIKind kind)
      : TargetLoweringInfo(std::make_unique<AArch64ABIInfo>(lt, kind)) {
    assert(!MissingFeature::swift());
  }

  unsigned getTargetAddrSpaceFromCIRAddrSpace(
      mlir::cir::AddressSpaceAttr addressSpaceAttr) const override {
    using Kind = mlir::cir::AddressSpaceAttr::Kind;
    switch (addressSpaceAttr.getValue()) {
    case Kind::offload_private:
    case Kind::offload_local:
    case Kind::offload_global:
    case Kind::offload_constant:
    case Kind::offload_generic:
      return 0;
    default:
      llvm_unreachable("Unknown CIR address space for this target");
    }
  }
};

} // namespace

ABIArgInfo AArch64ABIInfo::classifyReturnType(Type retTy,
                                              bool isVariadic) const {
  if (isa<VoidType>(retTy))
    return ABIArgInfo::getIgnore();

  if (const auto _ = dyn_cast<VectorType>(retTy)) {
    llvm_unreachable("NYI");
  }

  // Large vector types should be returned via memory.
  if (isa<VectorType>(retTy) && getContext().getTypeSize(retTy) > 128)
    llvm_unreachable("NYI");

  if (!isAggregateTypeForABI(retTy)) {
    // NOTE(cir): Skip enum handling.

    if (MissingFeature::fixedSizeIntType())
      llvm_unreachable("NYI");

    return (isPromotableIntegerTypeForABI(retTy) && isDarwinPCS()
                ? ABIArgInfo::getExtend(retTy)
                : ABIArgInfo::getDirect());
  }

  llvm_unreachable("NYI");
}

ABIArgInfo
AArch64ABIInfo::classifyArgumentType(Type ty, bool isVariadic,
                                     unsigned callingConvention) const {
  ty = useFirstFieldIfTransparentUnion(ty);

  // TODO(cir): check for illegal vector types.
  if (MissingFeature::vectorType())
    llvm_unreachable("NYI");

  if (!isAggregateTypeForABI(ty)) {
    // NOTE(cir): Enum is IntType in CIR. Skip enum handling here.

    if (MissingFeature::fixedSizeIntType())
      llvm_unreachable("NYI");

    return (isPromotableIntegerTypeForABI(ty) && isDarwinPCS()
                ? ABIArgInfo::getExtend(ty)
                : ABIArgInfo::getDirect());
  }

  llvm_unreachable("NYI");
}

std::unique_ptr<TargetLoweringInfo>
createAArch64TargetLoweringInfo(LowerModule &cgm, AArch64ABIKind kind) {
  return std::make_unique<AArch64TargetLoweringInfo>(cgm.getTypes(), kind);
}

} // namespace cir
} // namespace mlir
