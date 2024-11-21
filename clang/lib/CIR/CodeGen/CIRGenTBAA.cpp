#include "CIRGenTBAA.h"
#include "CIRGenCXXABI.h"
#include "CIRGenTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "llvm/Support/ErrorHandling.h"
namespace clang::CIRGen {

cir::TBAAAttr tbaa_NYI(mlir::MLIRContext *ctx) {
  return cir::TBAAAttr::get(ctx);
}

CIRGenTBAA::CIRGenTBAA(mlir::MLIRContext *ctx, clang::ASTContext &context,
                       CIRGenTypes &types, mlir::ModuleOp moduleOp,
                       const clang::CodeGenOptions &codeGenOpts,
                       const clang::LangOptions &features)
    : ctx(ctx), context(context), types(types), moduleOp(moduleOp),
      codeGenOpts(codeGenOpts), features(features) {}

cir::TBAAAttr CIRGenTBAA::getTypeInfo(clang::QualType qty) {
  return tbaa_NYI(ctx);
}

TBAAAccessInfo CIRGenTBAA::getAccessInfo(clang::QualType accessType) {
  return TBAAAccessInfo();
}

TBAAAccessInfo CIRGenTBAA::getVTablePtrAccessInfo(mlir::Type vtablePtrType) {
  return TBAAAccessInfo();
}

mlir::ArrayAttr CIRGenTBAA::getTBAAStructInfo(clang::QualType qty) {
  return mlir::ArrayAttr::get(ctx, {});
}

cir::TBAAAttr CIRGenTBAA::getBaseTypeInfo(clang::QualType qty) {
  return tbaa_NYI(ctx);
}

mlir::ArrayAttr CIRGenTBAA::getAccessTagInfo(TBAAAccessInfo tbaaInfo) {
  return mlir::ArrayAttr::get(ctx, {tbaa_NYI(ctx)});
}

TBAAAccessInfo CIRGenTBAA::mergeTBAAInfoForCast(TBAAAccessInfo sourceInfo,
                                                TBAAAccessInfo targetInfo) {
  return TBAAAccessInfo();
}

TBAAAccessInfo
CIRGenTBAA::mergeTBAAInfoForConditionalOperator(TBAAAccessInfo infoA,
                                                TBAAAccessInfo infoB) {
  return TBAAAccessInfo();
}

TBAAAccessInfo
CIRGenTBAA::mergeTBAAInfoForMemoryTransfer(TBAAAccessInfo destInfo,
                                           TBAAAccessInfo srcInfo) {
  return TBAAAccessInfo();
}

} // namespace clang::CIRGen
