
#include "clang/Frontend/FrontendAction.h"

#include <memory>

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace cir {
class CIRCombineAction : public clang::FrontendAction {
private:
  mlir::MLIRContext *mlirContext;

public:
  CIRCombineAction();
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    llvm::StringRef InFile) override {
    return std::make_unique<clang::ASTConsumer>();
  }

  void ExecuteAction() override;
  // We don't need a preprocessor-only mode.
  bool usesPreprocessorOnly() const override { return false; }
  virtual bool hasCIRSupport() const override { return true; }
};
} // namespace cir
