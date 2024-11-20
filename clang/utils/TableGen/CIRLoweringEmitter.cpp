//===- CIRBuiltinsEmitter.cpp - Generate lowering of builtins --=-*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TableGenBackends.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

namespace {
std::string ClassDefinitions;
std::string ClassList;

void GenerateLowering(raw_ostream &OS, const Record *Operation) {
  using namespace std::string_literals;
  std::string Name = Operation->getName().str();
  std::string LLVMOp = Operation->getValueAsString("llvmOp").str();
  ClassDefinitions +=
      "class CIR" + Name +
      "Lowering : public mlir::OpConversionPattern<mlir::cir::" + Name +
      R"C++(> {
  public:
    using OpConversionPattern<mlir::cir::)C++" +
      Name + R"C++(>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::)C++" +
      Name +
      R"C++( op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::)C++" +
      LLVMOp + ">(op";

  auto ArgCount = Operation->getValueAsDag("arguments")->getNumArgs();
  for (size_t i = 0; i != ArgCount; ++i)
    ClassDefinitions += ", adaptor.getOperands()[" + std::to_string(i) + ']';

  ClassDefinitions += R"C++();
    return mlir::success();
  }
};
)C++";

  ClassList += ", CIR" + Name + "Lowering\n";
}
} // namespace

void clang::EmitCIRBuiltinsLowering(const RecordKeeper &Records,
                                    raw_ostream &OS) {
  emitSourceFileHeader("Lowering of ClangIR builtins to LLVM IR builtins", OS);
  for (const auto *Builtin :
       Records.getAllDerivedDefinitions("LLVMLoweringInfo")) {
    if (!Builtin->getValueAsString("llvmOp").empty())
      GenerateLowering(OS, Builtin);
  }

  OS << "#ifdef GET_BUILTIN_LOWERING_CLASSES\n"
     << ClassDefinitions << "\n#undef GET_BUILTIN_LOWERING_CLASSES\n#endif\n";
  OS << "#ifdef GET_BUILTIN_LOWERING_LIST\n"
     << ClassList << "\n#undef GET_BUILTIN_LOWERING_LIST\n#endif\n";
}
