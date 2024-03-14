//===--- CIRGenStmtOpenMP.cpp - Emit MLIR Code from OpenMP Statements -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit OpenMP Stmt nodes as MLIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "CIRGenOpenMPRuntime.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

using namespace cir;
using namespace clang;
using namespace mlir::omp;

mlir::LogicalResult
CIRGenFunction::buildOMPParallelDirective(const OMPParallelDirective &S) {
  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  // Create a `omp.parallel` op.
  auto parallelOp = builder.create<ParallelOp>(scopeLoc);
  mlir::Block &block = parallelOp.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPointToEnd(&block);
  // Create a scope for the OpenMP region.
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, scopeLoc, builder.getInsertionBlock()};
        // Emit the body of the region.
        if (buildStmt(S.getCapturedStmt(OpenMPDirectiveKind::OMPD_parallel)
                          ->getCapturedStmt(),
                      /*useCurrentScope=*/true)
                .failed())
          res = mlir::failure();
      });
  // Add the terminator for `omp.parallel`.
  builder.create<TerminatorOp>(getLoc(S.getSourceRange().getEnd()));
  return res;
}

mlir::LogicalResult 
CIRGenFunction::buildOMPTaskwaitDirective(const OMPTaskwaitDirective &S) {
  mlir::LogicalResult res = mlir::success();
  // Getting the source location information of AST node S scope
  auto scopeLoc = getLoc(S.getSourceRange());
  // Creation of an omp.taskwait operation
  auto taskwaitOp = builder.create<mlir::omp::TaskwaitOp>(scopeLoc);

  return res;

}
mlir::LogicalResult 
CIRGenFunction::buildOMPTaskyieldDirective(const OMPTaskyieldDirective &S){
  mlir::LogicalResult res = mlir::success();
  // Getting the source location information of AST node S scope
  auto scopeLoc = getLoc(S.getSourceRange());
  // Creation of an omp.taskyield operation
  auto taskyieldOp = builder.create<mlir::omp::TaskyieldOp>(scopeLoc);

  return res;
}
