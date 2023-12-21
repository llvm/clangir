#include "clang/Basic/DiagnosticSema.h"
#include "llvm/ADT/StringExtras.h"

#include "CIRGenFunction.h"
#include "TargetInfo.h"

using namespace cir;
using namespace clang;
using namespace mlir::cir;

// The unimplemented features are too local to place them into the
// UnimplementedFeatureGuarding.h
struct AsmUnimplemented {
  static bool Goto() { return false; }
  static bool LLVMassume() { return false; }
  static bool unwindClobber() { return false; }
  static bool memoryEffects() { return false; }
  static bool vectorType() { return UnimplementedFeature::cirVectorType(); }
};

static bool isAggregateType(mlir::Type typ) {
  return isa<mlir::cir::StructType, mlir::cir::ArrayType>(typ);
}

static std::string SimplifyConstraint(
    const char *Constraint, const TargetInfo &Target,
    SmallVectorImpl<TargetInfo::ConstraintInfo> *OutCons = nullptr) {
  std::string Result;

  while (*Constraint) {
    switch (*Constraint) {
    default:
      Result += Target.convertConstraint(Constraint);
      break;
    // Ignore these
    case '*':
    case '?':
    case '!':
    case '=': // Will see this and the following in mult-alt constraints.
    case '+':
      break;
    case '#': // Ignore the rest of the constraint alternative.
      while (Constraint[1] && Constraint[1] != ',')
        Constraint++;
      break;
    case '&':
    case '%':
      Result += *Constraint;
      while (Constraint[1] && Constraint[1] == *Constraint)
        Constraint++;
      break;
    case ',':
      Result += "|";
      break;
    case 'g':
      Result += "imr";
      break;
    case '[': {
      assert(OutCons &&
             "Must pass output names to constraints with a symbolic name");
      unsigned Index;
      bool result = Target.resolveSymbolicName(Constraint, *OutCons, Index);
      assert(result && "Could not resolve symbolic name");
      (void)result;
      Result += llvm::utostr(Index);
      break;
    }
    }

    Constraint++;
  }

  return Result;
}

std::pair<mlir::Value, mlir::Type> CIRGenFunction::buildAsmInputLValue(
    const TargetInfo::ConstraintInfo &Info, LValue InputValue,
    QualType InputType, std::string &ConstraintStr, SourceLocation Loc) {

  if (Info.allowsRegister() || !Info.allowsMemory()) {
    if (hasScalarEvaluationKind(InputType))
      return {buildLoadOfLValue(InputValue, Loc).getScalarVal(), nullptr};

    mlir::Type Ty = convertType(InputType);
    uint64_t Size = CGM.getDataLayout().getTypeSizeInBits(Ty);
    if ((Size <= 64 && llvm::isPowerOf2_64(Size)) ||
        getTargetHooks().isScalarizableAsmOperand(*this, Ty)) {
      Ty = mlir::cir::IntType::get(builder.getContext(), Size, false);

      return {builder.createLoad(getLoc(Loc),
                                 InputValue.getAddress().withElementType(Ty)),
              nullptr};
    }
  }

  Address Addr = InputValue.getAddress();
  ConstraintStr += '*';
  return {Addr.getPointer(), Addr.getElementType()};
}

std::pair<mlir::Value, mlir::Type>
CIRGenFunction::buildAsmInput(const TargetInfo::ConstraintInfo &Info,
                              const Expr *InputExpr,
                              std::string &ConstraintStr) {
  auto loc = getLoc(InputExpr->getExprLoc());

  // If this can't be a register or memory, i.e., has to be a constant
  // (immediate or symbolic), try to emit it as such.
  if (!Info.allowsRegister() && !Info.allowsMemory()) {
    if (Info.requiresImmediateConstant()) {
      Expr::EvalResult EVResult;
      InputExpr->EvaluateAsRValue(EVResult, getContext(), true);

      llvm::APSInt IntResult;
      if (EVResult.Val.toIntegralConstant(IntResult, InputExpr->getType(),
                                          getContext()))
        return {builder.getConstAPSInt(loc, IntResult), mlir::Type()};
    }

    Expr::EvalResult Result;
    if (InputExpr->EvaluateAsInt(Result, getContext()))
      return {builder.getConstAPSInt(loc, Result.Val.getInt()), mlir::Type()};
  }

  if (Info.allowsRegister() || !Info.allowsMemory())
    if (CIRGenFunction::hasScalarEvaluationKind(InputExpr->getType()))
      return {buildScalarExpr(InputExpr), nullptr};
  if (InputExpr->getStmtClass() == Expr::CXXThisExprClass)
    return {buildScalarExpr(InputExpr), nullptr};
  InputExpr = InputExpr->IgnoreParenNoopCasts(getContext());
  LValue Dest = buildLValue(InputExpr);
  return buildAsmInputLValue(Info, Dest, InputExpr->getType(), ConstraintStr,
                             InputExpr->getExprLoc());
}

/// AddVariableConstraints - Look at AsmExpr and if it is a variable declared
/// as using a particular register add that as a constraint that will be used
/// in this asm stmt.
static std::string
AddVariableConstraints(const std::string &Constraint, const Expr &AsmExpr,
                       const TargetInfo &Target, CIRGenModule &CGM,
                       const AsmStmt &Stmt, const bool EarlyClobber,
                       std::string *GCCReg = nullptr) {
  const DeclRefExpr *AsmDeclRef = dyn_cast<DeclRefExpr>(&AsmExpr);
  if (!AsmDeclRef)
    return Constraint;
  const ValueDecl &Value = *AsmDeclRef->getDecl();
  const VarDecl *Variable = dyn_cast<VarDecl>(&Value);
  if (!Variable)
    return Constraint;
  if (Variable->getStorageClass() != SC_Register)
    return Constraint;
  AsmLabelAttr *Attr = Variable->getAttr<AsmLabelAttr>();
  if (!Attr)
    return Constraint;
  StringRef Register = Attr->getLabel();
  assert(Target.isValidGCCRegisterName(Register));
  // We're using validateOutputConstraint here because we only care if
  // this is a register constraint.
  TargetInfo::ConstraintInfo Info(Constraint, "");
  if (Target.validateOutputConstraint(Info) && !Info.allowsRegister()) {
    CGM.ErrorUnsupported(&Stmt, "__asm__");
    return Constraint;
  }
  // Canonicalize the register here before returning it.
  Register = Target.getNormalizedGCCRegisterName(Register);
  if (GCCReg != nullptr)
    *GCCReg = Register.str();
  return (EarlyClobber ? "&{" : "{") + Register.str() + "}";
}

static void buildAsmStores(CIRGenFunction &CGF, const AsmStmt &S,
                           const llvm::ArrayRef<mlir::Value> RegResults,
                           const llvm::ArrayRef<mlir::Type> ResultRegTypes,
                           const llvm::ArrayRef<mlir::Type> ResultTruncRegTypes,
                           const llvm::ArrayRef<LValue> ResultRegDests,
                           const llvm::ArrayRef<QualType> ResultRegQualTys,
                           const llvm::BitVector &ResultTypeRequiresCast,
                           const llvm::BitVector &ResultRegIsFlagReg) {
  CIRGenBuilderTy &Builder = CGF.getBuilder();
  CIRGenModule &CGM = CGF.CGM;
  auto CTX = Builder.getContext();

  assert(RegResults.size() == ResultRegTypes.size());
  assert(RegResults.size() == ResultTruncRegTypes.size());
  assert(RegResults.size() == ResultRegDests.size());
  // ResultRegDests can be also populated by addReturnRegisterOutputs() above,
  // in which case its size may grow.
  assert(ResultTypeRequiresCast.size() <= ResultRegDests.size());
  assert(ResultRegIsFlagReg.size() <= ResultRegDests.size());

  for (unsigned i = 0, e = RegResults.size(); i != e; ++i) {
    mlir::Value Tmp = RegResults[i];
    mlir::Type TruncTy = ResultTruncRegTypes[i];

    if ((i < ResultRegIsFlagReg.size()) && ResultRegIsFlagReg[i]) {
      assert(!AsmUnimplemented::LLVMassume());
    }

    // If the result type of the LLVM IR asm doesn't match the result type of
    // the expression, do the conversion.
    if (ResultRegTypes[i] != TruncTy) {

      // Truncate the integer result to the right size, note that TruncTy can be
      // a pointer.
      if (TruncTy.isa<mlir::FloatType>())
        Tmp = Builder.createFloatingCast(Tmp, TruncTy);
      else if (isa<mlir::cir::PointerType>(TruncTy) &&
               isa<mlir::cir::IntType>(Tmp.getType())) {
        uint64_t ResSize = CGM.getDataLayout().getTypeSizeInBits(TruncTy);
        Tmp = Builder.createIntCast(
            Tmp, mlir::cir::IntType::get(CTX, (unsigned)ResSize, false));
        Tmp = Builder.createIntToPtr(Tmp, TruncTy);
      } else if (isa<mlir::cir::PointerType>(Tmp.getType()) &&
                 isa<mlir::cir::IntType>(TruncTy)) {
        uint64_t TmpSize = CGM.getDataLayout().getTypeSizeInBits(Tmp.getType());
        Tmp = Builder.createPtrToInt(
            Tmp, mlir::cir::IntType::get(CTX, (unsigned)TmpSize, false));
        Tmp = Builder.createIntCast(Tmp, TruncTy);
      } else if (isa<mlir::cir::IntType>(TruncTy)) {
        Tmp = Builder.createIntCast(Tmp, TruncTy);
      } else if (false /*TruncTy->isVectorTy()*/) {
        assert(!AsmUnimplemented::vectorType());
      }
    }

    LValue Dest = ResultRegDests[i];
    // ResultTypeRequiresCast elements correspond to the first
    // ResultTypeRequiresCast.size() elements of RegResults.
    if ((i < ResultTypeRequiresCast.size()) && ResultTypeRequiresCast[i]) {
      unsigned Size = CGF.getContext().getTypeSize(ResultRegQualTys[i]);
      Address A = Dest.getAddress().withElementType(ResultRegTypes[i]);
      if (CGF.getTargetHooks().isScalarizableAsmOperand(CGF, TruncTy)) {
        Builder.createStore(CGF.getLoc(S.getAsmLoc()), Tmp, A);
        continue;
      }

      QualType Ty =
          CGF.getContext().getIntTypeForBitwidth(Size, /*Signed=*/false);
      if (Ty.isNull()) {
        const Expr *OutExpr = S.getOutputExpr(i);
        CGM.getDiags().Report(OutExpr->getExprLoc(),
                              diag::err_store_value_to_reg);
        return;
      }
      Dest = CGF.makeAddrLValue(A, Ty);
    }

    CGF.buildStoreThroughLValue(RValue::get(Tmp), Dest);
  }
}

static void collectClobbers(const CIRGenFunction &cgf, const AsmStmt &S,
                            std::string &constraints, bool &hasUnwindClobber,
                            bool &readOnly, bool readNone) {

  hasUnwindClobber = false;
  auto &cgm = cgf.getCIRGenModule();

  // Clobbers
  for (unsigned i = 0, e = S.getNumClobbers(); i != e; i++) {
    StringRef clobber = S.getClobber(i);

    if (clobber == "memory")
      readOnly = readNone = false;
    else if (clobber == "unwind") {
      hasUnwindClobber = true;
      continue;
    } else if (clobber != "cc") {
      clobber = cgf.getTarget().getNormalizedGCCRegisterName(clobber);
      if (cgm.getCodeGenOpts().StackClashProtector &&
          cgf.getTarget().isSPRegName(clobber)) {
        cgm.getDiags().Report(S.getAsmLoc(),
                              diag::warn_stack_clash_protection_inline_asm);
      }
    }

    if (isa<MSAsmStmt>(&S)) {
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
                                        const AsmStmt &S, constraintInfos &out,
                                        constraintInfos &in) {

  for (unsigned i = 0, e = S.getNumOutputs(); i != e; i++) {
    StringRef Name;
    if (const GCCAsmStmt *GAS = dyn_cast<GCCAsmStmt>(&S))
      Name = GAS->getOutputName(i);
    TargetInfo::ConstraintInfo Info(S.getOutputConstraint(i), Name);
    bool IsValid = cgf.getTarget().validateOutputConstraint(Info);
    (void)IsValid;
    assert(IsValid && "Failed to parse output constraint");
    out.push_back(Info);
  }

  for (unsigned i = 0, e = S.getNumInputs(); i != e; i++) {
    StringRef Name;
    if (const GCCAsmStmt *GAS = dyn_cast<GCCAsmStmt>(&S))
      Name = GAS->getInputName(i);
    TargetInfo::ConstraintInfo Info(S.getInputConstraint(i), Name);
    bool IsValid = cgf.getTarget().validateInputConstraint(out, Info);
    assert(IsValid && "Failed to parse input constraint");
    (void)IsValid;
    in.push_back(Info);
  }
}

static AsmDialect inferDialect(const CIRGenModule &cgm, const AsmStmt &S) {
  AsmDialect GnuAsmDialect =
      cgm.getCodeGenOpts().getInlineAsmDialect() == CodeGenOptions::IAD_ATT
          ? AsmDialect::AD_ATT
          : AsmDialect::AD_Intel;

  return isa<MSAsmStmt>(&S) ? AsmDialect::AD_Intel : GnuAsmDialect;
}

mlir::LogicalResult CIRGenFunction::buildAsmStmt(const AsmStmt &S) {

  // Assemble the final asm string.
  std::string AsmString = S.generateAsmString(getContext());

  // Get all the output and input constraints together.
  constraintInfos OutputConstraintInfos;
  constraintInfos InputConstraintInfos;
  collectInOutConstrainsInfos(*this, S, OutputConstraintInfos,
                              InputConstraintInfos);

  std::string Constraints;

  std::vector<LValue> ResultRegDests;
  std::vector<QualType> ResultRegQualTys;
  std::vector<mlir::Type> ResultRegTypes;
  std::vector<mlir::Type> ResultTruncRegTypes;
  std::vector<mlir::Type> ArgTypes;
  std::vector<mlir::Type> ArgElemTypes;
  std::vector<mlir::Value> Args;
  llvm::BitVector ResultTypeRequiresCast;
  llvm::BitVector ResultRegIsFlagReg;

  // Keep track of inout constraints.
  std::string InOutConstraints;
  std::vector<mlir::Value> InOutArgs;
  std::vector<mlir::Type> InOutArgTypes;
  std::vector<mlir::Type> InOutArgElemTypes;

  // Keep track of out constraints for tied input operand.
  std::vector<std::string> OutputConstraints;

  // Keep track of defined physregs.
  llvm::SmallSet<std::string, 8> PhysRegOutputs;

  // An inline asm can be marked readonly if it meets the following conditions:
  //  - it doesn't have any sideeffects
  //  - it doesn't clobber memory
  //  - it doesn't return a value by-reference
  // It can be marked readnone if it doesn't have any input memory constraints
  // in addition to meeting the conditions listed above.
  bool ReadOnly = true, ReadNone = true;

  for (unsigned i = 0, e = S.getNumOutputs(); i != e; i++) {
    TargetInfo::ConstraintInfo &Info = OutputConstraintInfos[i];

    // Simplify the output constraint.
    std::string OutputConstraint(S.getOutputConstraint(i));
    OutputConstraint = SimplifyConstraint(OutputConstraint.c_str() + 1,
                                          getTarget(), &OutputConstraintInfos);

    const Expr *OutExpr = S.getOutputExpr(i);
    OutExpr = OutExpr->IgnoreParenNoopCasts(getContext());

    std::string GCCReg;
    OutputConstraint =
        AddVariableConstraints(OutputConstraint, *OutExpr, getTarget(), CGM, S,
                               Info.earlyClobber(), &GCCReg);
    // Give an error on multiple outputs to same physreg.
    if (!GCCReg.empty() && !PhysRegOutputs.insert(GCCReg).second)
      CGM.Error(S.getAsmLoc(), "multiple outputs to hard register: " + GCCReg);

    OutputConstraints.push_back(OutputConstraint);

    LValue Dest = buildLValue(OutExpr);

    if (!Constraints.empty())
      Constraints += ',';

    // If this is a register output, then make the inline a sm return it
    // by-value.  If this is a memory result, return the value by-reference.
    QualType QTy = OutExpr->getType();
    const bool IsScalarOrAggregate =
        hasScalarEvaluationKind(QTy) || hasAggregateEvaluationKind(QTy);
    if (!Info.allowsMemory() && IsScalarOrAggregate) {
      Constraints += "=" + OutputConstraint;
      ResultRegQualTys.push_back(QTy);
      ResultRegDests.push_back(Dest);

      bool IsFlagReg = llvm::StringRef(OutputConstraint).starts_with("{@cc");
      ResultRegIsFlagReg.push_back(IsFlagReg);

      mlir::Type Ty = convertTypeForMem(QTy);
      const bool RequiresCast =
          Info.allowsRegister() &&
          (getTargetHooks().isScalarizableAsmOperand(*this, Ty) ||
           isAggregateType(Ty));

      ResultTruncRegTypes.push_back(Ty);
      ResultTypeRequiresCast.push_back(RequiresCast);

      if (RequiresCast) {
        unsigned Size = getContext().getTypeSize(QTy);
        Ty = mlir::cir::IntType::get(builder.getContext(), Size, false);
      }
      ResultRegTypes.push_back(Ty);
      // If this output is tied to an input, and if the input is larger, then
      // we need to set the actual result type of the inline asm node to be the
      // same as the input type.
      if (Info.hasMatchingInput()) {
        unsigned InputNo;
        for (InputNo = 0; InputNo != S.getNumInputs(); ++InputNo) {
          TargetInfo::ConstraintInfo &Input = InputConstraintInfos[InputNo];
          if (Input.hasTiedOperand() && Input.getTiedOperand() == i)
            break;
        }
        assert(InputNo != S.getNumInputs() && "Didn't find matching input!");

        QualType InputTy = S.getInputExpr(InputNo)->getType();
        QualType OutputType = OutExpr->getType();

        uint64_t InputSize = getContext().getTypeSize(InputTy);
        if (getContext().getTypeSize(OutputType) < InputSize) {
          // Form the asm to return the value as a larger integer or fp type.
          ResultRegTypes.back() = ConvertType(InputTy);
        }
      }
      if (mlir::Type AdjTy = getTargetHooks().adjustInlineAsmType(
              *this, OutputConstraint, ResultRegTypes.back()))
        ResultRegTypes.back() = AdjTy;
      else {
        CGM.getDiags().Report(S.getAsmLoc(),
                              diag::err_asm_invalid_type_in_input)
            << OutExpr->getType() << OutputConstraint;
      }

      // Update largest vector width for any vector types.
      assert(!AsmUnimplemented::vectorType());

    } else {
      Address DestAddr = Dest.getAddress();

      // Matrix types in memory are represented by arrays, but accessed through
      // vector pointers, with the alignment specified on the access operation.
      // For inline assembly, update pointer arguments to use vector pointers.
      // Otherwise there will be a mis-match if the matrix is also an
      // input-argument which is represented as vector.
      if (isa<MatrixType>(OutExpr->getType().getCanonicalType()))
        DestAddr = DestAddr.withElementType(ConvertType(OutExpr->getType()));

      ArgTypes.push_back(DestAddr.getType());
      ArgElemTypes.push_back(DestAddr.getElementType());
      Args.push_back(DestAddr.getPointer());
      Constraints += "=*";
      Constraints += OutputConstraint;
      ReadOnly = ReadNone = false;
    }

    if (Info.isReadWrite()) {
      InOutConstraints += ',';

      const Expr *InputExpr = S.getOutputExpr(i);
      mlir::Value Arg;
      mlir::Type ArgElemType;
      std::tie(Arg, ArgElemType) =
          buildAsmInputLValue(Info, Dest, InputExpr->getType(),
                              InOutConstraints, InputExpr->getExprLoc());

      if (mlir::Type AdjTy = getTargetHooks().adjustInlineAsmType(
              *this, OutputConstraint, Arg.getType()))
        Arg = builder.createBitcast(Arg, AdjTy);

      // Update largest vector width for any vector types.
      assert(!AsmUnimplemented::vectorType());

      // Only tie earlyclobber physregs.
      if (Info.allowsRegister() && (GCCReg.empty() || Info.earlyClobber()))
        InOutConstraints += llvm::utostr(i);
      else
        InOutConstraints += OutputConstraint;

      InOutArgTypes.push_back(Arg.getType());
      InOutArgElemTypes.push_back(ArgElemType);
      InOutArgs.push_back(Arg);
    }
  } // for loop

  // If this is a Microsoft-style asm blob, store the return registers (EAX:EDX)
  // to the return value slot. Only do this when returning in registers.
  if (isa<MSAsmStmt>(&S)) {
    const ABIArgInfo &RetAI = CurFnInfo->getReturnInfo();
    if (RetAI.isDirect() || RetAI.isExtend()) {
      // Make a fake lvalue for the return value slot.
      LValue ReturnSlot = makeAddrLValue(ReturnValue, FnRetTy);
      CGM.getTargetCIRGenInfo().addReturnRegisterOutputs(
          *this, ReturnSlot, Constraints, ResultRegTypes, ResultTruncRegTypes,
          ResultRegDests, AsmString, S.getNumOutputs());
      SawAsmBlock = true;
    }
  }

  for (unsigned i = 0, e = S.getNumInputs(); i != e; i++) {
    const Expr *InputExpr = S.getInputExpr(i);

    TargetInfo::ConstraintInfo &Info = InputConstraintInfos[i];

    if (Info.allowsMemory())
      ReadNone = false;

    if (!Constraints.empty())
      Constraints += ',';

    // Simplify the input constraint.
    std::string InputConstraint(S.getInputConstraint(i));
    InputConstraint = SimplifyConstraint(InputConstraint.c_str(), getTarget(),
                                         &OutputConstraintInfos);

    InputConstraint = AddVariableConstraints(
        InputConstraint, *InputExpr->IgnoreParenNoopCasts(getContext()),
        getTarget(), CGM, S, false /* No EarlyClobber */);

    std::string ReplaceConstraint(InputConstraint);
    mlir::Value Arg;
    mlir::Type ArgElemType;
    std::tie(Arg, ArgElemType) = buildAsmInput(Info, InputExpr, Constraints);

    // If this input argument is tied to a larger output result, extend the
    // input to be the same size as the output.  The LLVM backend wants to see
    // the input and output of a matching constraint be the same size.  Note
    // that GCC does not define what the top bits are here.  We use zext because
    // that is usually cheaper, but LLVM IR should really get an anyext someday.
    if (Info.hasTiedOperand()) {
      unsigned Output = Info.getTiedOperand();
      QualType OutputType = S.getOutputExpr(Output)->getType();
      QualType InputTy = InputExpr->getType();

      if (getContext().getTypeSize(OutputType) >
          getContext().getTypeSize(InputTy)) {
        // Use ptrtoint as appropriate so that we can do our extension.
        if (isa<mlir::cir::PointerType>(Arg.getType()))
          Arg = builder.createPtrToInt(Arg, UIntPtrTy);
        mlir::Type OutputTy = convertType(OutputType);
        if (isa<mlir::cir::IntType>(OutputTy))
          Arg = builder.createIntCast(Arg, OutputTy);
        else if (isa<mlir::cir::PointerType>(OutputTy))
          Arg = builder.createIntCast(Arg, UIntPtrTy);
        else if (isa<mlir::FloatType>(OutputTy))
          Arg = builder.createFloatingCast(Arg, OutputTy);
      }
      // Deal with the tied operands' constraint code in adjustInlineAsmType.
      ReplaceConstraint = OutputConstraints[Output];
    }
    if (mlir::Type AdjTy = getTargetHooks().adjustInlineAsmType(
            *this, ReplaceConstraint, Arg.getType()))
      Arg = builder.createBitcast(Arg, AdjTy);
    else
      CGM.getDiags().Report(S.getAsmLoc(), diag::err_asm_invalid_type_in_input)
          << InputExpr->getType() << InputConstraint;

    // Update largest vector width for any vector types.
    assert(!AsmUnimplemented::vectorType());

    ArgTypes.push_back(Arg.getType());
    ArgElemTypes.push_back(ArgElemType);
    Args.push_back(Arg);
    Constraints += InputConstraint;
  }

  // Append the "input" part of inout constraints.
  for (unsigned i = 0, e = InOutArgs.size(); i != e; i++) {
    ArgTypes.push_back(InOutArgTypes[i]);
    ArgElemTypes.push_back(InOutArgElemTypes[i]);
    Args.push_back(InOutArgs[i]);
  }
  Constraints += InOutConstraints;

  assert(!AsmUnimplemented::Goto());

  bool HasUnwindClobber = false;
  collectClobbers(*this, S, Constraints, HasUnwindClobber, ReadOnly, ReadNone);

  mlir::Type ResultType;

  if (ResultRegTypes.empty()) {
  //  ResultType = builder.getVoidTy();
  } else if (ResultRegTypes.size() == 1) {
    ResultType = ResultRegTypes[0];
  } else {
    auto sname = builder.getUniqueAnonRecordName();
    ResultType =
        builder.getCompleteStructTy(ResultRegTypes, sname, false, nullptr);
  }

  // mlir::cir::FuncType FTy =
  //     mlir::cir::FuncType::get(ArgTypes, ResultType, false);

  bool HasSideEffect = S.isVolatile() || S.getNumOutputs() == 0;
  AsmDialect AsmDialect = inferDialect(CGM, S);

  std::vector<mlir::Value> RegResults;

  auto IA = builder.create<mlir::cir::InlineAsmOp>(
      getLoc(S.getAsmLoc()), ResultType, Args, AsmString, Constraints,
      HasSideEffect,
      /* IsAlignStack */ false,
      AsmDialectAttr::get(builder.getContext(), AsmDialect), mlir::ArrayAttr());

  if (false /*IsGCCAsmGoto*/) {
    assert(!AsmUnimplemented::Goto());
  } else if (HasUnwindClobber) {
    assert(!AsmUnimplemented::unwindClobber());
  } else {
    assert(!AsmUnimplemented::memoryEffects());

    mlir::Value result;
    if (IA.getNumResults())
      result = IA.getResult(0);

    std::vector<mlir::Attribute> operandAttrs;
    auto attrName = mlir::cir::InlineAsmOp::getElementTypeAttrName();


    // this is for the lowering to LLVM from LLVm dialect. Otherwise, if we don't
    // have the result (i.e. void type as a result of operation), the element type
    // attribute will be attached to the whole instruction, but not to the operand
    if (!IA.getNumResults())
      operandAttrs.push_back(OptNoneAttr::get(builder.getContext()));

    for (auto typ : ArgElemTypes) {
      std::vector<mlir::NamedAttribute> attrs;
      if (typ) {
        auto typAttr = builder.getTypeAttr(typ);
        attrs.push_back(builder.getNamedAttr(attrName, typAttr));
        auto dict = builder.getDictionaryAttr(attrs);
        operandAttrs.push_back(dict);
      } else {
        // We need to add an attribute for every arg since later, during
        // the lowering to LLVM IR the attributes will be assigned to the
        // CallInsn argument by index, i.e. we can't skip null type here
        operandAttrs.push_back(OptNoneAttr::get(builder.getContext()));
      }
    }

    IA.setOperandAttrsAttr(builder.getArrayAttr(operandAttrs));

    if (ResultRegTypes.size() == 1) {
      RegResults.push_back(result);
    } else if (ResultRegTypes.size() > 1) {
      auto alignment = CharUnits::One();
      auto sname = cast<mlir::cir::StructType>(ResultType).getName();
      auto dest = buildAlloca(sname, ResultType, getLoc(S.getAsmLoc()),
                              alignment, false);
      auto addr = Address(dest, alignment);
      builder.createStore(getLoc(S.getAsmLoc()), result, addr);

      for (unsigned i = 0, e = ResultRegTypes.size(); i != e; ++i) {
        // TODO: double check. The point is the elt in the reg types should be
        // the same that is returned by getMember, i.e. should be a pointer.
        auto typ = builder.getPointerTo(ResultRegTypes[i]);

        auto ptr =
            builder.createGetMember(getLoc(S.getAsmLoc()), typ, dest, "", i);

        auto tmp =
            builder.createLoad(getLoc(S.getAsmLoc()), Address(ptr, alignment));
        // TODO: do we need to load after the getMember?
        RegResults.push_back(tmp);
      }
    }
  }

  buildAsmStores(*this, S, RegResults, ResultRegTypes, ResultTruncRegTypes,
                 ResultRegDests, ResultRegQualTys, ResultTypeRequiresCast,
                 ResultRegIsFlagReg);

  assert(!AsmUnimplemented::Goto());

  return mlir::success();
}
