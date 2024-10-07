//===- CIRGenerator.h - CIR Generation from Clang AST ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform CIR generation from Clang
// AST
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIRGENERATOR_H_
#define CLANG_CIRGENERATOR_H_

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/CodeGenOptions.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <memory>

namespace mlir {
class MLIRContext;
class ModuleOp;
class OwningModuleRef;
} // namespace mlir

namespace clang {
class ASTContext;
class DeclGroupRef;
class FunctionDecl;
} // namespace clang

namespace cir {
class CIRGenModule;
class CIRGenTypes;

class CIRGenerator : public clang::ASTConsumer {
  virtual void anchor();
  clang::DiagnosticsEngine &diags;
  clang::ASTContext *astCtx;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
      fs; // Only used for debug info.

  const clang::CodeGenOptions codeGenOpts; // Intentionally copied in.

  unsigned handlingTopLevelDecls;

  /// Use this when emitting decls to block re-entrant decl emission. It will
  /// emit all deferred decls on scope exit. Set EmitDeferred to false if decl
  /// emission must be deferred longer, like at the end of a tag definition.
  struct HandlingTopLevelDeclRAII {
    CIRGenerator &self;
    bool emitDeferred;
    HandlingTopLevelDeclRAII(CIRGenerator &self, bool emitDeferred = true)
        : self{self}, emitDeferred{emitDeferred} {
      ++self.handlingTopLevelDecls;
    }
    ~HandlingTopLevelDeclRAII() {
      unsigned level = --self.handlingTopLevelDecls;
      if (level == 0 && emitDeferred)
        self.buildDeferredDecls();
    }
  };

protected:
  std::unique_ptr<mlir::MLIRContext> mlirCtx;
  std::unique_ptr<CIRGenModule> cgm;

private:
  llvm::SmallVector<clang::FunctionDecl *, 8> deferredInlineMemberFuncDefs;

public:
  CIRGenerator(clang::DiagnosticsEngine &diags,
               llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
               const clang::CodeGenOptions &cgo);
  ~CIRGenerator() override;
  void Initialize(clang::ASTContext &astCtx) override;
  bool emitFunction(const clang::FunctionDecl *fd);

  bool HandleTopLevelDecl(clang::DeclGroupRef d) override;
  void HandleTranslationUnit(clang::ASTContext &ctx) override;
  void HandleInlineFunctionDefinition(clang::FunctionDecl *d) override;
  void HandleTagDeclDefinition(clang::TagDecl *d) override;
  void HandleTagDeclRequiredDefinition(const clang::TagDecl *d) override;
  void HandleCXXStaticMemberVarInstantiation(clang::VarDecl *d) override;
  void CompleteTentativeDefinition(clang::VarDecl *d) override;

  mlir::ModuleOp getModule();
  std::unique_ptr<mlir::MLIRContext> takeContext() {
    return std::move(mlirCtx);
  };

  bool verifyModule();

  void buildDeferredDecls();
  void buildDefaultMethods();
};

} // namespace cir

#endif // CLANG_CIRGENERATOR_H_
