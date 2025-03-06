//====- LowerToMLIR.h- Lowering from CIR to MLIR --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares functions for lowering CIR modules to MLIR.
//
//===----------------------------------------------------------------------===//
#ifndef CLANG_CIR_LOWERTOMLIR_H
#define CLANG_CIR_LOWERTOMLIR_H

<<<<<<< HEAD
#include "mlir/Transforms/DialectConversion.h"

=======
>>>>>>> 9a2a7a370a31 ([CIR][CUDA] Support for built-in CUDA surface type)
namespace cir {

void populateCIRLoopToSCFConversionPatterns(mlir::RewritePatternSet &patterns,
                                            mlir::TypeConverter &converter);
<<<<<<< HEAD

mlir::ModuleOp
lowerFromCIRToMLIRToLLVMDialect(mlir::ModuleOp theModule,
                                mlir::MLIRContext *mlirCtx = nullptr);
=======
>>>>>>> 9a2a7a370a31 ([CIR][CUDA] Support for built-in CUDA surface type)
} // namespace cir

#endif // CLANG_CIR_LOWERTOMLIR_H_
