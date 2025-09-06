//===--- LowerTypes.cpp - Type translation to target-specific types -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CodeGenTypes.cpp. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "LowerTypes.h"
#include "CIRToCIRArgMapping.h"
#include "LowerModule.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

using namespace cir;

using ABIArgInfo = cir::ABIArgInfo;

unsigned LowerTypes::clangCallConvToLLVMCallConv(clang::CallingConv CC) {
  switch (CC) {
  case clang::CC_C:
    return llvm::CallingConv::C;
  default:
    cir_cconv_unreachable("calling convention NYI");
  }
}

LowerTypes::LowerTypes(LowerModule &LM)
    : LM(LM), context(LM.getContext()), Target(LM.getTarget()),
      CXXABI(LM.getCXXABI()),
      TheABIInfo(LM.getTargetLoweringInfo().getABIInfo()),
      mlirContext(LM.getMLIRContext()), DL(LM.getModule()) {}

/// Return the ABI-specific function type for a CIR function type.
FuncType LowerTypes::getFunctionType(const LowerFunctionInfo &FI) {

  mlir::Type resultType = {};
  const cir::ABIArgInfo &retAI = FI.getReturnInfo();
  switch (retAI.getKind()) {
  case ABIArgInfo::Extend:
  case ABIArgInfo::Direct:
    resultType = retAI.getCoerceToType();
    break;
  case cir::ABIArgInfo::Ignore:
  case cir::ABIArgInfo::Indirect:
    resultType = VoidType::get(getMLIRContext());
    break;
  default:
    cir_cconv_unreachable("Missing ABIArgInfo::Kind");
  }

  CIRToCIRArgMapping IRFunctionArgs(getContext(), FI, true);
  llvm::SmallVector<mlir::Type, 8> ArgTypes(IRFunctionArgs.totalIRArgs());

  // Add type for sret argument.
  if (IRFunctionArgs.hasSRetArg()) {
    mlir::Type ret = FI.getReturnType();
    ArgTypes[IRFunctionArgs.getSRetArgNo()] = cir::PointerType::get(ret);
  }

  // Add type for inalloca argument.
  cir_cconv_assert(!cir::MissingFeatures::inallocaArgs());

  // Add in all of the required arguments.
  unsigned ArgNo = 0;
  LowerFunctionInfo::const_arg_iterator it = FI.arg_begin(),
                                        ie = it + FI.getNumRequiredArgs();
  for (; it != ie; ++it, ++ArgNo) {
    const ABIArgInfo &ArgInfo = it->info;

    cir_cconv_assert(!cir::MissingFeatures::argumentPadding());

    unsigned FirstIRArg, NumIRArgs;
    std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);

    switch (ArgInfo.getKind()) {
    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {
      // Fast-isel and the optimizer generally like scalar values better than
      // FCAs, so we flatten them if this is safe to do for this argument.
      mlir::Type argType = ArgInfo.getCoerceToType();
      RecordType st = mlir::dyn_cast<RecordType>(argType);
      if (st && ArgInfo.isDirect() && ArgInfo.getCanBeFlattened()) {
        cir_cconv_assert(NumIRArgs == st.getNumElements());
        for (unsigned i = 0, e = st.getNumElements(); i != e; ++i)
          ArgTypes[FirstIRArg + i] = st.getMembers()[i];
      } else {
        cir_cconv_assert(NumIRArgs == 1);
        ArgTypes[FirstIRArg] = argType;
      }
      break;
    }
    case ABIArgInfo::Indirect: {
      mlir::Type argType = (FI.arg_begin() + ArgNo)->type;
      ArgTypes[FirstIRArg] = cir::PointerType::get(argType);
      break;
    }
    default:
      cir_cconv_unreachable("Missing ABIArgInfo::Kind");
    }
  }

  return FuncType::get(ArgTypes, resultType, FI.isVariadic());
}

static void dumpType(mlir::Type t) {
  auto *interface =
      llvm::dyn_cast<mlir::OpAsmDialectInterface>(&t.getDialect());
  auto printType = [&](mlir::Type typeToPrint, bool ptrLevel = 0) {
    llvm::SmallString<256> buffer;
    llvm::raw_svector_ostream out(buffer);
    mlir::OpAsmDialectInterface::AliasResult r =
        interface->getAlias(typeToPrint, out);
    llvm::errs() << "Missing default ABI-specific for type: ";
    if (ptrLevel)
      llvm::errs() << ptrLevel << " pointer indirection to ";
    if (r == mlir::OpAsmDialectInterface::AliasResult::NoAlias) {
      llvm::errs() << typeToPrint << "\n";
      return;
    }
    llvm::errs() << out.str() << "\n";
  };

  mlir::Type finalTy = t;
  int ptrInd = 0;
  auto ptrTy = llvm::dyn_cast<cir::PointerType>(t);
  while (ptrTy) {
    finalTy = ptrTy.getPointee();
    ptrInd++;
    ptrTy = llvm::dyn_cast<cir::PointerType>(finalTy);
  }
  printType(finalTy, ptrInd);
}

/// Convert a CIR type to its ABI-specific default form.
mlir::Type LowerTypes::convertType(mlir::Type t) {
  /// NOTE(cir): It the original codegen this method is used to get the default
  /// LLVM IR representation for a given AST type. When a the ABI-specific
  /// function info sets a nullptr for a return or argument type, the default
  /// type given by this method is used. In CIR's case, its types are already
  /// supposed to be ABI-specific, so this method is not really useful here.
  /// It's kept here for codegen parity's sake.

  // Certain CIR types are already ABI-specific, so we just return them.
  if (mlir::isa<BoolType, IntType, SingleType, DoubleType>(t)) {
    return t;
  }

  dumpType(t);
  cir_cconv_assert_or_abort(
      !cir::MissingFeatures::X86DefaultABITypeConvertion(),
      "TargetLowering convertType NYI");
  return t;
}
