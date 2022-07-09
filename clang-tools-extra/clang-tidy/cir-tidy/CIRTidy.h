//===--- CIRTidy.h - cir-tidy -------------------------------*- C++ -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CIRTIDY_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CIRTIDY_H

#include "ClangTidyDiagnosticConsumer.h"
#include "clang/AST/ASTConsumer.h"
#include <vector>

namespace clang {
class CompilerInstance;
namespace tooling {
class CompilationDatabase;
}
} // namespace clang

namespace clang {
namespace tidy {

class CIRTidyASTConsumerFactory {
public:
  CIRTidyASTConsumerFactory(
      ClangTidyContext &Context,
      IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS = nullptr);

  std::unique_ptr<clang::ASTConsumer>
  createASTConsumer(clang::CompilerInstance &Compiler, StringRef File);

private:
  ClangTidyContext &Context;
  IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS;
};

std::vector<ClangTidyError>
runCIRTidy(clang::tidy::ClangTidyContext &Context,
           const tooling::CompilationDatabase &Compilations,
           ArrayRef<std::string> InputFiles,
           llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> BaseFS,
           bool ApplyAnyFix, bool EnableCheckProfile = false,
           llvm::StringRef StoreCheckProfile = StringRef());

} // end namespace tidy
} // end namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CIRTIDY_H
