#include "clang/Basic/DiagnosticSema.h"
#include "llvm/ADT/StringExtras.h"

#include "CIRGenFunction.h"
#include "TargetInfo.h"
#include "clang/CIR/MissingFeatures.h"

using namespace cir;
using namespace clang;
using namespace mlir::cir;

static bool isAggregateType(mlir::Type typ) {
  return isa<mlir::cir::StructType, mlir::cir::ArrayType>(typ);
}

static AsmFlavor inferFlavor(const CIRGenModule &cgm, const AsmStmt &s) {
  AsmFlavor gnuAsmFlavor =
      cgm.getCodeGenOpts().getInlineAsmDialect() == CodeGenOptions::IAD_ATT
          ? AsmFlavor::x86_att
          : AsmFlavor::x86_intel;

  return isa<MSAsmStmt>(&s) ? AsmFlavor::x86_intel : gnuAsmFlavor;
}

// FIXME(cir): This should be a common helper between CIRGen
// and traditional CodeGen
static std::string simplifyConstraint(
    const char *constraint, const TargetInfo &target,
    SmallVectorImpl<TargetInfo::ConstraintInfo> *outCons = nullptr) {
  std::string result;

  while (*constraint) {
    switch (*constraint) {
    default:
      result += target.convertConstraint(constraint);
      break;
    // Ignore these
    case '*':
    case '?':
    case '!':
    case '=': // Will see this and the following in mult-alt constraints.
    case '+':
      break;
    case '#': // Ignore the rest of the constraint alternative.
      while (constraint[1] && constraint[1] != ',')
        constraint++;
      break;
    case '&':
    case '%':
      result += *constraint;
      while (constraint[1] && constraint[1] == *constraint)
        constraint++;
      break;
    case ',':
      result += "|";
      break;
    case 'g':
      result += "imr";
      break;
    case '[': {
      assert(outCons &&
             "Must pass output names to constraints with a symbolic name");
      unsigned index;
      bool resolved = target.resolveSymbolicName(constraint, *outCons, index);
      assert(resolved && "Could not resolve symbolic name");
      (void)resolved;
      result += llvm::utostr(index);
      break;
    }
    }

    constraint++;
  }

  return result;
}

// FIXME(cir): This should be a common helper between CIRGen
// and traditional CodeGen
/// Look at AsmExpr and if it is a variable declared
/// as using a particular register add that as a constraint that will be used
/// in this asm stmt.
static std::string
addVariableConstraints(const std::string &constraint, const Expr &asmExpr,
                       const TargetInfo &target, CIRGenModule &cgm,
                       const AsmStmt &stmt, const bool earlyClobber,
                       std::string *gccReg = nullptr) {
  const DeclRefExpr *asmDeclRef = dyn_cast<DeclRefExpr>(&asmExpr);
  if (!asmDeclRef)
    return constraint;
  const ValueDecl &value = *asmDeclRef->getDecl();
  const VarDecl *variable = dyn_cast<VarDecl>(&value);
  if (!variable)
    return constraint;
  if (variable->getStorageClass() != SC_Register)
    return constraint;
  AsmLabelAttr *attr = variable->getAttr<AsmLabelAttr>();
  if (!attr)
    return constraint;
  StringRef Register = attr->getLabel();
  assert(target.isValidGCCRegisterName(Register));
  // We're using validateOutputConstraint here because we only care if
  // this is a register constraint.
  TargetInfo::ConstraintInfo info(constraint, "");
  if (target.validateOutputConstraint(info) && !info.allowsRegister()) {
    cgm.ErrorUnsupported(&stmt, "__asm__");
    return constraint;
  }
  // Canonicalize the register here before returning it.
  Register = target.getNormalizedGCCRegisterName(Register);
  if (gccReg != nullptr)
    *gccReg = Register.str();
  return (earlyClobber ? "&{" : "{") + Register.str() + "}";
}

static void collectClobbers(const CIRGenFunction &cgf, const AsmStmt &s,
                            std::string &constraints, bool &hasUnwindClobber,
                            bool &readOnly, bool readNone) {

  hasUnwindClobber = false;
  auto &cgm = cgf.getCIRGenModule();

  // Clobbers
  for (unsigned i = 0, e = s.getNumClobbers(); i != e; i++) {
    StringRef clobber = s.getClobber(i);
    if (clobber == "memory")
      readOnly = readNone = false;
    else if (clobber == "unwind") {
      hasUnwindClobber = true;
      continue;
    } else if (clobber != "cc") {
      clobber = cgf.getTarget().getNormalizedGCCRegisterName(clobber);
      if (cgm.getCodeGenOpts().StackClashProtector &&
          cgf.getTarget().isSPRegName(clobber)) {
        cgm.getDiags().Report(s.getAsmLoc(),
                              diag::warn_stack_clash_protection_inline_asm);
      }
    }

    if (isa<MSAsmStmt>(&s)) {
      if (clobber == "eax" || clobber == "edx") {
        if (constraints.find("=&A") != std::string::npos)
          continue;
        std::string::size_type position1 =
            constraints.find("={" + clobber.str() + "}");
        if (position1 != std::string::npos) {
          constraints.insert(position1 + 1, "&");
          continue;
        }
        std::string::size_type position2 = constraints.find("=A");
        if (position2 != std::string::npos) {
          constraints.insert(position2 + 1, "&");
          continue;
        }
      }
    }
    if (!constraints.empty())
      constraints += ',';

    constraints += "~{";
    constraints += clobber;
    constraints += '}';
  }

  // Add machine specific clobbers
  std::string_view machineClobbers = cgf.getTarget().getClobbers();
  if (!machineClobbers.empty()) {
    if (!constraints.empty())
      constraints += ',';
    constraints += machineClobbers;
  }
}

using constraintInfos = SmallVector<TargetInfo::ConstraintInfo, 4>;

static void collectInOutConstrainsInfos(const CIRGenFunction &cgf,
                                        const AsmStmt &s, constraintInfos &out,
                                        constraintInfos &in) {

  for (unsigned i = 0, e = s.getNumOutputs(); i != e; i++) {
    StringRef name;
    if (const GCCAsmStmt *gas = dyn_cast<GCCAsmStmt>(&s))
      name = gas->getOutputName(i);
    TargetInfo::ConstraintInfo info(s.getOutputConstraint(i), name);
    bool isValid = cgf.getTarget().validateOutputConstraint(info);
    (void)isValid;
    assert(isValid && "Failed to parse output constraint");
    out.push_back(info);
  }

  for (unsigned i = 0, e = s.getNumInputs(); i != e; i++) {
    StringRef name;
    if (const GCCAsmStmt *gas = dyn_cast<GCCAsmStmt>(&s))
      name = gas->getInputName(i);
    TargetInfo::ConstraintInfo info(s.getInputConstraint(i), name);
    bool isValid = cgf.getTarget().validateInputConstraint(out, info);
    assert(isValid && "Failed to parse input constraint");
    (void)isValid;
    in.push_back(info);
  }
}

std::pair<mlir::Value, mlir::Type> CIRGenFunction::buildAsmInputLValue(
    const TargetInfo::ConstraintInfo &info, LValue inputValue,
    QualType inputType, std::string &constraintStr, SourceLocation loc) {

  if (info.allowsRegister() || !info.allowsMemory()) {
    if (hasScalarEvaluationKind(inputType))
      return {buildLoadOfLValue(inputValue, loc).getScalarVal(), mlir::Type()};

    mlir::Type ty = convertType(inputType);
    uint64_t size = cgm.getDataLayout().getTypeSizeInBits(ty);
    if ((size <= 64 && llvm::isPowerOf2_64(size)) ||
        getTargetHooks().isScalarizableAsmOperand(*this, ty)) {
      ty = mlir::cir::IntType::get(builder.getContext(), size, false);

      return {builder.createLoad(getLoc(loc),
                                 inputValue.getAddress().withElementType(ty)),
              mlir::Type()};
    }
  }

  Address addr = inputValue.getAddress();
  constraintStr += '*';
  return {addr.getPointer(), addr.getElementType()};
}

std::pair<mlir::Value, mlir::Type>
CIRGenFunction::buildAsmInput(const TargetInfo::ConstraintInfo &info,
                              const Expr *inputExpr,
                              std::string &constraintStr) {
  auto loc = getLoc(inputExpr->getExprLoc());

  // If this can't be a register or memory, i.e., has to be a constant
  // (immediate or symbolic), try to emit it as such.
  if (!info.allowsRegister() && !info.allowsMemory()) {
    if (info.requiresImmediateConstant()) {
      Expr::EvalResult evResult;
      inputExpr->EvaluateAsRValue(evResult, getContext(), true);

      llvm::APSInt intResult;
      if (evResult.Val.toIntegralConstant(intResult, inputExpr->getType(),
                                          getContext()))
        return {builder.getConstAPSInt(loc, intResult), mlir::Type()};
    }

    Expr::EvalResult result;
    if (inputExpr->EvaluateAsInt(result, getContext()))
      return {builder.getConstAPSInt(loc, result.Val.getInt()), mlir::Type()};
  }

  if (info.allowsRegister() || !info.allowsMemory())
    if (CIRGenFunction::hasScalarEvaluationKind(inputExpr->getType()))
      return {buildScalarExpr(inputExpr), mlir::Type()};
  if (inputExpr->getStmtClass() == Expr::CXXThisExprClass)
    return {buildScalarExpr(inputExpr), mlir::Type()};
  inputExpr = inputExpr->IgnoreParenNoopCasts(getContext());
  LValue dest = buildLValue(inputExpr);
  return buildAsmInputLValue(info, dest, inputExpr->getType(), constraintStr,
                             inputExpr->getExprLoc());
}

static void buildAsmStores(CIRGenFunction &cgf, const AsmStmt &s,
                           const llvm::ArrayRef<mlir::Value> regResults,
                           const llvm::ArrayRef<mlir::Type> resultRegTypes,
                           const llvm::ArrayRef<mlir::Type> resultTruncRegTypes,
                           const llvm::ArrayRef<LValue> resultRegDests,
                           const llvm::ArrayRef<QualType> resultRegQualTys,
                           const llvm::BitVector &resultTypeRequiresCast,
                           const llvm::BitVector &resultRegIsFlagReg) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  CIRGenModule &cgm = cgf.cgm;
  auto *ctx = builder.getContext();

  assert(regResults.size() == resultRegTypes.size());
  assert(regResults.size() == resultTruncRegTypes.size());
  assert(regResults.size() == resultRegDests.size());
  // ResultRegDests can be also populated by addReturnRegisterOutputs() above,
  // in which case its size may grow.
  assert(resultTypeRequiresCast.size() <= resultRegDests.size());
  assert(resultRegIsFlagReg.size() <= resultRegDests.size());

  for (unsigned i = 0, e = regResults.size(); i != e; ++i) {
    mlir::Value tmp = regResults[i];
    mlir::Type truncTy = resultTruncRegTypes[i];

    if ((i < resultRegIsFlagReg.size()) && resultRegIsFlagReg[i]) {
      assert(!MissingFeatures::asmLLVMAssume());
    }

    // If the result type of the LLVM IR asm doesn't match the result type of
    // the expression, do the conversion.
    if (resultRegTypes[i] != truncTy) {

      // Truncate the integer result to the right size, note that TruncTy can be
      // a pointer.
      if (mlir::isa<mlir::FloatType>(truncTy))
        tmp = builder.createFloatingCast(tmp, truncTy);
      else if (isa<mlir::cir::PointerType>(truncTy) &&
               isa<mlir::cir::IntType>(tmp.getType())) {
        uint64_t resSize = cgm.getDataLayout().getTypeSizeInBits(truncTy);
        tmp = builder.createIntCast(
            tmp, mlir::cir::IntType::get(ctx, (unsigned)resSize, false));
        tmp = builder.createIntToPtr(tmp, truncTy);
      } else if (isa<mlir::cir::PointerType>(tmp.getType()) &&
                 isa<mlir::cir::IntType>(truncTy)) {
        uint64_t tmpSize = cgm.getDataLayout().getTypeSizeInBits(tmp.getType());
        tmp = builder.createPtrToInt(
            tmp, mlir::cir::IntType::get(ctx, (unsigned)tmpSize, false));
        tmp = builder.createIntCast(tmp, truncTy);
      } else if (isa<mlir::cir::IntType>(truncTy)) {
        tmp = builder.createIntCast(tmp, truncTy);
      } else if (false /*TruncTy->isVectorTy()*/) {
        assert(!MissingFeatures::asmVectorType());
      }
    }

    LValue dest = resultRegDests[i];
    // ResultTypeRequiresCast elements correspond to the first
    // ResultTypeRequiresCast.size() elements of RegResults.
    if ((i < resultTypeRequiresCast.size()) && resultTypeRequiresCast[i]) {
      unsigned size = cgf.getContext().getTypeSize(resultRegQualTys[i]);
      Address a = dest.getAddress().withElementType(resultRegTypes[i]);
      if (cgf.getTargetHooks().isScalarizableAsmOperand(cgf, truncTy)) {
        builder.createStore(cgf.getLoc(s.getAsmLoc()), tmp, a);
        continue;
      }

      QualType ty =
          cgf.getContext().getIntTypeForBitwidth(size, /*Signed=*/false);
      if (ty.isNull()) {
        const Expr *outExpr = s.getOutputExpr(i);
        cgm.getDiags().Report(outExpr->getExprLoc(),
                              diag::err_store_value_to_reg);
        return;
      }
      dest = cgf.makeAddrLValue(a, ty);
    }

    cgf.buildStoreThroughLValue(RValue::get(tmp), dest);
  }
}

mlir::LogicalResult CIRGenFunction::buildAsmStmt(const AsmStmt &s) {
  // Assemble the final asm string.
  std::string asmString = s.generateAsmString(getContext());

  // Get all the output and input constraints together.
  constraintInfos outputConstraintInfos;
  constraintInfos inputConstraintInfos;
  collectInOutConstrainsInfos(*this, s, outputConstraintInfos,
                              inputConstraintInfos);

  std::string constraints;
  std::vector<LValue> resultRegDests;
  std::vector<QualType> resultRegQualTys;
  std::vector<mlir::Type> resultRegTypes;
  std::vector<mlir::Type> resultTruncRegTypes;
  std::vector<mlir::Type> argTypes;
  std::vector<mlir::Type> argElemTypes;
  std::vector<mlir::Value> outArgs;
  std::vector<mlir::Value> inArgs;
  std::vector<mlir::Value> inOutArgs;
  std::vector<mlir::Value> args;
  llvm::BitVector resultTypeRequiresCast;
  llvm::BitVector resultRegIsFlagReg;

  // Keep track of input constraints.
  std::string inOutConstraints;
  std::vector<mlir::Type> inOutArgTypes;
  std::vector<mlir::Type> inOutArgElemTypes;

  // Keep track of out constraints for tied input operand.
  std::vector<std::string> outputConstraints;

  // Keep track of defined physregs.
  llvm::SmallSet<std::string, 8> physRegOutputs;

  // An inline asm can be marked readonly if it meets the following conditions:
  //  - it doesn't have any sideeffects
  //  - it doesn't clobber memory
  //  - it doesn't return a value by-reference
  // It can be marked readnone if it doesn't have any input memory constraints
  // in addition to meeting the conditions listed above.
  bool readOnly = true, readNone = true;

  for (unsigned i = 0, e = s.getNumOutputs(); i != e; i++) {
    TargetInfo::ConstraintInfo &info = outputConstraintInfos[i];

    // Simplify the output constraint.
    std::string outputConstraint(s.getOutputConstraint(i));
    outputConstraint = simplifyConstraint(outputConstraint.c_str() + 1,
                                          getTarget(), &outputConstraintInfos);

    const Expr *outExpr = s.getOutputExpr(i);
    outExpr = outExpr->IgnoreParenNoopCasts(getContext());

    std::string gccReg;
    outputConstraint =
        addVariableConstraints(outputConstraint, *outExpr, getTarget(), cgm, s,
                               info.earlyClobber(), &gccReg);

    // Give an error on multiple outputs to same physreg.
    if (!gccReg.empty() && !physRegOutputs.insert(gccReg).second)
      cgm.Error(s.getAsmLoc(), "multiple outputs to hard register: " + gccReg);

    outputConstraints.push_back(outputConstraint);
    LValue dest = buildLValue(outExpr);

    if (!constraints.empty())
      constraints += ',';

    // If this is a register output, then make the inline a sm return it
    // by-value.  If this is a memory result, return the value by-reference.
    QualType qTy = outExpr->getType();
    const bool isScalarOrAggregate =
        hasScalarEvaluationKind(qTy) || hasAggregateEvaluationKind(qTy);
    if (!info.allowsMemory() && isScalarOrAggregate) {
      constraints += "=" + outputConstraint;
      resultRegQualTys.push_back(qTy);
      resultRegDests.push_back(dest);

      bool isFlagReg = llvm::StringRef(outputConstraint).starts_with("{@cc");
      resultRegIsFlagReg.push_back(isFlagReg);

      mlir::Type ty = convertTypeForMem(qTy);
      const bool requiresCast =
          info.allowsRegister() &&
          (getTargetHooks().isScalarizableAsmOperand(*this, ty) ||
           isAggregateType(ty));

      resultTruncRegTypes.push_back(ty);
      resultTypeRequiresCast.push_back(requiresCast);

      if (requiresCast) {
        unsigned size = getContext().getTypeSize(qTy);
        ty = mlir::cir::IntType::get(builder.getContext(), size, false);
      }
      resultRegTypes.push_back(ty);
      // If this output is tied to an input, and if the input is larger, then
      // we need to set the actual result type of the inline asm node to be the
      // same as the input type.
      if (info.hasMatchingInput()) {
        unsigned inputNo;
        for (inputNo = 0; inputNo != s.getNumInputs(); ++inputNo) {
          TargetInfo::ConstraintInfo &input = inputConstraintInfos[inputNo];
          if (input.hasTiedOperand() && input.getTiedOperand() == i)
            break;
        }
        assert(inputNo != s.getNumInputs() && "Didn't find matching input!");

        QualType inputTy = s.getInputExpr(inputNo)->getType();
        QualType outputType = outExpr->getType();

        uint64_t inputSize = getContext().getTypeSize(inputTy);
        if (getContext().getTypeSize(outputType) < inputSize) {
          // Form the asm to return the value as a larger integer or fp type.
          resultRegTypes.back() = ConvertType(inputTy);
        }
      }
      if (mlir::Type adjTy = getTargetHooks().adjustInlineAsmType(
              *this, outputConstraint, resultRegTypes.back()))
        resultRegTypes.back() = adjTy;
      else {
        cgm.getDiags().Report(s.getAsmLoc(),
                              diag::err_asm_invalid_type_in_input)
            << outExpr->getType() << outputConstraint;
      }

      // Update largest vector width for any vector types.
      assert(!MissingFeatures::asmVectorType());
    } else {
      Address destAddr = dest.getAddress();

      // Matrix types in memory are represented by arrays, but accessed through
      // vector pointers, with the alignment specified on the access operation.
      // For inline assembly, update pointer arguments to use vector pointers.
      // Otherwise there will be a mis-match if the matrix is also an
      // input-argument which is represented as vector.
      if (isa<MatrixType>(outExpr->getType().getCanonicalType()))
        destAddr = destAddr.withElementType(ConvertType(outExpr->getType()));

      argTypes.push_back(destAddr.getType());
      argElemTypes.push_back(destAddr.getElementType());
      outArgs.push_back(destAddr.getPointer());
      args.push_back(destAddr.getPointer());
      constraints += "=*";
      constraints += outputConstraint;
      readOnly = readNone = false;
    }

    if (info.isReadWrite()) {
      inOutConstraints += ',';
      const Expr *inputExpr = s.getOutputExpr(i);

      mlir::Value arg;
      mlir::Type argElemType;
      std::tie(arg, argElemType) =
          buildAsmInputLValue(info, dest, inputExpr->getType(),
                              inOutConstraints, inputExpr->getExprLoc());

      if (mlir::Type adjTy = getTargetHooks().adjustInlineAsmType(
              *this, outputConstraint, arg.getType()))
        arg = builder.createBitcast(arg, adjTy);

      // Update largest vector width for any vector types.
      assert(!MissingFeatures::asmVectorType());

      // Only tie earlyclobber physregs.
      if (info.allowsRegister() && (gccReg.empty() || info.earlyClobber()))
        inOutConstraints += llvm::utostr(i);
      else
        inOutConstraints += outputConstraint;

      inOutArgTypes.push_back(arg.getType());
      inOutArgElemTypes.push_back(argElemType);
      inOutArgs.push_back(arg);
    }
  } // iterate over output operands

  // If this is a Microsoft-style asm blob, store the return registers (EAX:EDX)
  // to the return value slot. Only do this when returning in registers.
  if (isa<MSAsmStmt>(&s)) {
    const ABIArgInfo &retAi = CurFnInfo->getReturnInfo();
    if (retAi.isDirect() || retAi.isExtend()) {
      // Make a fake lvalue for the return value slot.
      LValue returnSlot = makeAddrLValue(ReturnValue, FnRetTy);
      cgm.getTargetCIRGenInfo().addReturnRegisterOutputs(
          *this, returnSlot, constraints, resultRegTypes, resultTruncRegTypes,
          resultRegDests, asmString, s.getNumOutputs());
      SawAsmBlock = true;
    }
  }

  for (unsigned i = 0, e = s.getNumInputs(); i != e; i++) {
    const Expr *inputExpr = s.getInputExpr(i);

    TargetInfo::ConstraintInfo &info = inputConstraintInfos[i];

    if (info.allowsMemory())
      readNone = false;

    if (!constraints.empty())
      constraints += ',';

    // Simplify the input constraint.
    std::string inputConstraint(s.getInputConstraint(i));
    inputConstraint = simplifyConstraint(inputConstraint.c_str(), getTarget(),
                                         &outputConstraintInfos);

    inputConstraint = addVariableConstraints(
        inputConstraint, *inputExpr->IgnoreParenNoopCasts(getContext()),
        getTarget(), cgm, s, false /* No EarlyClobber */);

    std::string replaceConstraint(inputConstraint);
    mlir::Value arg;
    mlir::Type argElemType;
    std::tie(arg, argElemType) = buildAsmInput(info, inputExpr, constraints);

    // If this input argument is tied to a larger output result, extend the
    // input to be the same size as the output.  The LLVM backend wants to see
    // the input and output of a matching constraint be the same size.  Note
    // that GCC does not define what the top bits are here.  We use zext because
    // that is usually cheaper, but LLVM IR should really get an anyext someday.
    if (info.hasTiedOperand()) {
      unsigned output = info.getTiedOperand();
      QualType outputType = s.getOutputExpr(output)->getType();
      QualType inputTy = inputExpr->getType();

      if (getContext().getTypeSize(outputType) >
          getContext().getTypeSize(inputTy)) {
        // Use ptrtoint as appropriate so that we can do our extension.
        if (isa<mlir::cir::PointerType>(arg.getType()))
          arg = builder.createPtrToInt(arg, UIntPtrTy);
        mlir::Type outputTy = convertType(outputType);
        if (isa<mlir::cir::IntType>(outputTy))
          arg = builder.createIntCast(arg, outputTy);
        else if (isa<mlir::cir::PointerType>(outputTy))
          arg = builder.createIntCast(arg, UIntPtrTy);
        else if (isa<mlir::FloatType>(outputTy))
          arg = builder.createFloatingCast(arg, outputTy);
      }

      // Deal with the tied operands' constraint code in adjustInlineAsmType.
      replaceConstraint = outputConstraints[output];
    }

    if (mlir::Type adjTy = getTargetHooks().adjustInlineAsmType(
            *this, replaceConstraint, arg.getType()))
      arg = builder.createBitcast(arg, adjTy);
    else
      cgm.getDiags().Report(s.getAsmLoc(), diag::err_asm_invalid_type_in_input)
          << inputExpr->getType() << inputConstraint;

    // Update largest vector width for any vector types.
    assert(!MissingFeatures::asmVectorType());

    argTypes.push_back(arg.getType());
    argElemTypes.push_back(argElemType);
    inArgs.push_back(arg);
    args.push_back(arg);
    constraints += inputConstraint;
  } // iterate over input operands

  // Append the "input" part of inout constraints.
  for (unsigned i = 0, e = inOutArgs.size(); i != e; i++) {
    args.push_back(inOutArgs[i]);
    argTypes.push_back(inOutArgTypes[i]);
    argElemTypes.push_back(inOutArgElemTypes[i]);
  }
  constraints += inOutConstraints;

  bool hasUnwindClobber = false;
  collectClobbers(*this, s, constraints, hasUnwindClobber, readOnly, readNone);

  mlir::Type resultType;

  if (resultRegTypes.size() == 1)
    resultType = resultRegTypes[0];
  else if (resultRegTypes.size() > 1) {
    auto sname = builder.getUniqueAnonRecordName();
    resultType =
        builder.getCompleteStructTy(resultRegTypes, sname, false, nullptr);
  }

  bool hasSideEffect = s.isVolatile() || s.getNumOutputs() == 0;
  std::vector<mlir::Value> regResults;

  llvm::SmallVector<mlir::ValueRange, 8> operands;
  operands.push_back(outArgs);
  operands.push_back(inArgs);
  operands.push_back(inOutArgs);

  auto ia = builder.create<mlir::cir::InlineAsmOp>(
      getLoc(s.getAsmLoc()), resultType, operands, asmString, constraints,
      hasSideEffect, inferFlavor(cgm, s), mlir::ArrayAttr());

  if (false /*IsGCCAsmGoto*/) {
    assert(!MissingFeatures::asmGoto());
  } else if (hasUnwindClobber) {
    assert(!MissingFeatures::asmUnwindClobber());
  } else {
    assert(!MissingFeatures::asmMemoryEffects());

    mlir::Value result;
    if (ia.getNumResults())
      result = ia.getResult(0);

    llvm::SmallVector<mlir::Attribute> operandAttrs;

    int i = 0;
    for (auto typ : argElemTypes) {
      if (typ) {
        auto op = args[i++];
        assert(mlir::isa<mlir::cir::PointerType>(op.getType()) &&
               "pointer type expected");
        assert(cast<mlir::cir::PointerType>(op.getType()).getPointee() == typ &&
               "element type differs from pointee type!");

        operandAttrs.push_back(mlir::UnitAttr::get(builder.getContext()));
      } else {
        // We need to add an attribute for every arg since later, during
        // the lowering to LLVM IR the attributes will be assigned to the
        // CallInsn argument by index, i.e. we can't skip null type here
        operandAttrs.push_back(mlir::Attribute());
      }
    }

    assert(args.size() == operandAttrs.size() &&
           "The number of attributes is not even with the number of operands");

    ia.setOperandAttrsAttr(builder.getArrayAttr(operandAttrs));

    if (resultRegTypes.size() == 1) {
      regResults.push_back(result);
    } else if (resultRegTypes.size() > 1) {
      auto alignment = CharUnits::One();
      auto sname = cast<mlir::cir::StructType>(resultType).getName();
      auto dest = buildAlloca(sname, resultType, getLoc(s.getAsmLoc()),
                              alignment, false);
      auto addr = Address(dest, alignment);
      builder.createStore(getLoc(s.getAsmLoc()), result, addr);

      for (unsigned i = 0, e = resultRegTypes.size(); i != e; ++i) {
        auto typ = builder.getPointerTo(resultRegTypes[i]);
        auto ptr =
            builder.createGetMember(getLoc(s.getAsmLoc()), typ, dest, "", i);
        auto tmp =
            builder.createLoad(getLoc(s.getAsmLoc()), Address(ptr, alignment));
        regResults.push_back(tmp);
      }
    }
  }

  buildAsmStores(*this, s, regResults, resultRegTypes, resultTruncRegTypes,
                 resultRegDests, resultRegQualTys, resultTypeRequiresCast,
                 resultRegIsFlagReg);

  return mlir::success();
}
