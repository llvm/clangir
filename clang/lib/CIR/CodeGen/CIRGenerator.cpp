//===--- CIRGenerator.cpp - Emit CIR from ASTs ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This builds an AST and converts it to CIR.
//
//===----------------------------------------------------------------------===//

#include "CIRGenModule.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/CIR/CIRGenerator.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace cir;
using namespace clang;

void CIRGenerator::anchor() {}

CIRGenerator::CIRGenerator(clang::DiagnosticsEngine &diagsEngine,
                           llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs,
                           const CodeGenOptions &cgo)
    : diags(diagsEngine), fs(std::move(vfs)), codeGenOpts{cgo},
      handlingTopLevelDecls(0) {}
CIRGenerator::~CIRGenerator() {
  // There should normally not be any leftover inline method definitions.
  assert(deferredInlineMemberFuncDefs.empty() || diags.hasErrorOccurred());
}

static void setMLIRDataLayout(mlir::ModuleOp &mod, const llvm::DataLayout &dl) {
  auto *context = mod.getContext();
  mod->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
               mlir::StringAttr::get(context, dl.getStringRepresentation()));
  mlir::DataLayoutSpecInterface dlSpec = mlir::translateDataLayout(dl, context);
  mod->setAttr(mlir::DLTIDialect::kDataLayoutAttrName, dlSpec);
}

void CIRGenerator::Initialize(ASTContext &astCtx) {
  using namespace llvm;

  this->astCtx = &astCtx;

  mlirCtx = std::make_unique<mlir::MLIRContext>();
  mlirCtx->getOrLoadDialect<mlir::DLTIDialect>();
  mlirCtx->getOrLoadDialect<mlir::func::FuncDialect>();
  mlirCtx->getOrLoadDialect<mlir::cir::CIRDialect>();
  mlirCtx->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  mlirCtx->getOrLoadDialect<mlir::memref::MemRefDialect>();
  mlirCtx->getOrLoadDialect<mlir::omp::OpenMPDialect>();
  cgm = std::make_unique<CIRGenModule>(*mlirCtx, astCtx, codeGenOpts, diags);
  auto mod = cgm->getModule();
  auto layout = llvm::DataLayout(astCtx.getTargetInfo().getDataLayoutString());
  setMLIRDataLayout(mod, layout);
}

bool CIRGenerator::verifyModule() { return cgm->verifyModule(); }

bool CIRGenerator::emitFunction(const FunctionDecl *fd) {
  llvm_unreachable("NYI");
}

mlir::ModuleOp CIRGenerator::getModule() { return cgm->getModule(); }

bool CIRGenerator::HandleTopLevelDecl(DeclGroupRef d) {
  if (diags.hasErrorOccurred())
    return true;

  HandlingTopLevelDeclRAII handlingDecl(*this);

  for (auto &i : d) {
    cgm->buildTopLevelDecl(i);
  }

  return true;
}

void CIRGenerator::HandleTranslationUnit(ASTContext &c) {
  // Release the Builder when there is no error.
  if (!diags.hasErrorOccurred() && cgm)
    cgm->Release();

  // If there are errors before or when releasing the cgm, reset the module to
  // stop here before invoking the backend.
  if (diags.hasErrorOccurred()) {
    if (cgm)
      // TODO: cgm->clear();
      // TODO: M.reset();
      return;
  }
}

void CIRGenerator::HandleInlineFunctionDefinition(FunctionDecl *d) {
  if (diags.hasErrorOccurred())
    return;

  assert(d->doesThisDeclarationHaveABody());

  // We may want to emit this definition. However, that decision might be
  // based on computing the linkage, and we have to defer that in case we are
  // inside of something that will chagne the method's final linkage, e.g.
  //   typedef struct {
  //     void bar();
  //     void foo() { bar(); }
  //   } A;
  deferredInlineMemberFuncDefs.push_back(d);

  // Provide some coverage mapping even for methods that aren't emitted.
  // Don't do this for templated classes though, as they may not be
  // instantiable.
  if (!d->getLexicalDeclContext()->isDependentContext())
    cgm->AddDeferredUnusedCoverageMapping(d);
}

void CIRGenerator::buildDefaultMethods() { cgm->buildDefaultMethods(); }

void CIRGenerator::buildDeferredDecls() {
  if (deferredInlineMemberFuncDefs.empty())
    return;

  // Emit any deferred inline method definitions. Note that more deferred
  // methods may be added during this loop, since ASTConsumer callbacks can be
  // invoked if AST inspection results in declarations being added.
  HandlingTopLevelDeclRAII handlingDecls(*this);
  for (auto &deferredInlineMemberFuncDef : deferredInlineMemberFuncDefs)
    cgm->buildTopLevelDecl(deferredInlineMemberFuncDef);
  deferredInlineMemberFuncDefs.clear();
}

/// HandleTagDeclDefinition - This callback is invoked each time a TagDecl to
/// (e.g. struct, union, enum, class) is completed. This allows the client hack
/// on the type, which can occur at any point in the file (because these can be
/// defined in declspecs).
void CIRGenerator::HandleTagDeclDefinition(TagDecl *d) {
  if (diags.hasErrorOccurred())
    return;

  // Don't allow re-entrant calls to CIRGen triggered by PCH deserialization to
  // emit deferred decls.
  HandlingTopLevelDeclRAII handlingDecl(*this, /*EmitDeferred=*/false);

  cgm->UpdateCompletedType(d);

  // For MSVC compatibility, treat declarations of static data members with
  // inline initializers as definitions.
  if (astCtx->getTargetInfo().getCXXABI().isMicrosoft()) {
    llvm_unreachable("NYI");
  }
  // For OpenMP emit declare reduction functions, if required.
  if (astCtx->getLangOpts().OpenMP) {
    llvm_unreachable("NYI");
  }
}

void CIRGenerator::HandleTagDeclRequiredDefinition(const TagDecl *d) {
  if (diags.hasErrorOccurred())
    return;

  // Don't allow re-entrant calls to CIRGen triggered by PCH deserialization to
  // emit deferred decls.
  HandlingTopLevelDeclRAII handlingDecl(*this, /*EmitDeferred=*/false);

  if (cgm->getModuleDebugInfo())
    llvm_unreachable("NYI");
}

void CIRGenerator::HandleCXXStaticMemberVarInstantiation(VarDecl *d) {
  if (diags.hasErrorOccurred())
    return;

  cgm->HandleCXXStaticMemberVarInstantiation(d);
}

void CIRGenerator::CompleteTentativeDefinition(VarDecl *d) {
  if (diags.hasErrorOccurred())
    return;

  cgm->buildTentativeDefinition(d);
}
