#include "clang/Basic/DiagnosticSema.h"
#include "llvm/ADT/StringExtras.h"

#include "CIRGenFunction.h"
#include "TargetInfo.h"

using namespace cir;
using namespace clang;
using namespace mlir::cir;

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

  std::string Constraints;
  std::vector<mlir::Type> ResultRegTypes;
  std::vector<mlir::Value> Args;

  assert(!S.getNumOutputs() && "NYI");
  assert(!S.getNumInputs() && "NYI");
  assert(!S.getNumClobbers() && "NYI");

  mlir::Type ResultType;

  if (ResultRegTypes.empty())
    ResultType = builder.getVoidTy();
  else if (ResultRegTypes.size() == 1)
    ResultType = ResultRegTypes[0];
  else {
    auto sname = builder.getUniqueAnonRecordName();
    ResultType =
        builder.getCompleteStructTy(ResultRegTypes, sname, false, nullptr);
  }

  bool HasSideEffect = S.isVolatile() || S.getNumOutputs() == 0;
  AsmDialect AsmDialect = inferDialect(CGM, S);

  builder.create<mlir::cir::InlineAsmOp>(
      getLoc(S.getAsmLoc()), ResultType, Args, AsmString, Constraints,
      HasSideEffect,
      /* IsAlignStack */ false,
      AsmDialectAttr::get(builder.getContext(), AsmDialect), mlir::ArrayAttr());

  return mlir::success();
}