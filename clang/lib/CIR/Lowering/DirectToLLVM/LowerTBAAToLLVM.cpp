#include "LowerToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/AST/RecordLayout.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

namespace cir {
namespace direct {

class CIRToLLVMTBAAAttrLowering {
public:
  CIRToLLVMTBAAAttrLowering(mlir::MLIRContext *mlirContext,
                            const clang::LangOptions &features)
      : mlirContext(mlirContext), features(features) {}
  mlir::LLVM::TBAARootAttr getRoot() {
    if (features.CPlusPlus) {
      return createTBAARoot("Simple C++ TBAA");
    } else {
      return createTBAARoot("Simple C/C++ TBAA");
    }
  }

  mlir::LLVM::TBAATypeDescriptorAttr getChar() {
    return createScalarTypeNode("omnipotent char", getRoot(), 0);
  }

  mlir::LLVM::TBAATypeDescriptorAttr
  createScalarTypeNode(llvm::StringRef typeName,
                       mlir::LLVM::TBAANodeAttr parent, int64_t size) {
    llvm::SmallVector<mlir::LLVM::TBAAMemberAttr, 2> members;
    members.push_back(mlir::LLVM::TBAAMemberAttr::get(mlirContext, parent, 0));
    return mlir::LLVM::TBAATypeDescriptorAttr::get(
        mlirContext, typeName,
        llvm::ArrayRef<mlir::LLVM::TBAAMemberAttr>(members));
  }

private:
  mlir::LLVM::TBAARootAttr createTBAARoot(llvm::StringRef name) {
    return mlir::LLVM::TBAARootAttr::get(
        mlirContext, mlir::StringAttr::get(mlirContext, name));
  }

protected:
  mlir::MLIRContext *mlirContext;
  const clang::LangOptions &features;
};

class CIRToLLVMTBAAScalarAttrLowering : public CIRToLLVMTBAAAttrLowering {
public:
  CIRToLLVMTBAAScalarAttrLowering(mlir::MLIRContext *mlirContext,
                                  const clang::LangOptions &features)
      : CIRToLLVMTBAAAttrLowering(mlirContext, features) {}
  mlir::LLVM::TBAATypeDescriptorAttr
  lowerScalarType(cir::TBAAScalarAttr scalarAttr) {
    mlir::DataLayout layout;
    auto size = layout.getTypeSize(scalarAttr.getType());
    return createScalarTypeNode(scalarAttr.getId(), getChar(), size);
  }
};
mlir::LLVM::TBAATypeDescriptorAttr
lowerCIRTBAAAttrToLLVMTBAAAttr(mlir::MLIRContext *ctx, mlir::Attribute tbaa,
                               cir::LowerModule *lowerMod) {
  CIRToLLVMTBAAScalarAttrLowering scalarLower(
      ctx, lowerMod->getContext().getLangOpts());
  if (auto charAttr = mlir::dyn_cast<cir::TBAAOmnipotentCharAttr>(tbaa)) {
    return scalarLower.getChar();
  }
  if (auto scalarAttr = mlir::dyn_cast<cir::TBAAScalarAttr>(tbaa)) {
    return scalarLower.lowerScalarType(
        mlir::dyn_cast<cir::TBAAScalarAttr>(tbaa));
  }
  if (auto structAttr = mlir::dyn_cast<cir::TBAAStructAttr>(tbaa)) {
    llvm::SmallVector<mlir::LLVM::TBAAMemberAttr, 4> members;
    for (const auto &member : structAttr.getMembers()) {
      auto memberTypeDesc =
          lowerCIRTBAAAttrToLLVMTBAAAttr(ctx, member.getTypeDesc(), lowerMod);
      auto memberAttr = mlir::LLVM::TBAAMemberAttr::get(ctx, memberTypeDesc,
                                                        member.getOffset());
      members.push_back(memberAttr);
    }
    return mlir::LLVM::TBAATypeDescriptorAttr::get(ctx, structAttr.getId(),
                                                   members);
  }
  return nullptr;
}
mlir::ArrayAttr lowerCIRTBAAAttr(mlir::Attribute tbaa,
                                 mlir::ConversionPatternRewriter &rewriter,
                                 cir::LowerModule *lowerMod) {
  auto *ctx = rewriter.getContext();
  if (auto tbaaTag = mlir::dyn_cast<cir::TBAATagAttr>(tbaa)) {
    mlir::LLVM::TBAATypeDescriptorAttr accessType =
        lowerCIRTBAAAttrToLLVMTBAAAttr(ctx, tbaaTag.getAccess(), lowerMod);
    if (auto structAttr =
            mlir::dyn_cast<cir::TBAAStructAttr>(tbaaTag.getBase())) {
      auto baseType = lowerCIRTBAAAttrToLLVMTBAAAttr(ctx, structAttr, lowerMod);
      auto tag = mlir::LLVM::TBAATagAttr::get(baseType, accessType,
                                              tbaaTag.getOffset());
      return mlir::ArrayAttr::get(ctx, {tag});
    }
  } else {
    auto accessType = lowerCIRTBAAAttrToLLVMTBAAAttr(ctx, tbaa, lowerMod);
    if (accessType) {
      auto tag = mlir::LLVM::TBAATagAttr::get(accessType, accessType, 0);
      return mlir::ArrayAttr::get(ctx, {tag});
    }
  }
  return mlir::ArrayAttr();
}

} // namespace direct
} // namespace cir