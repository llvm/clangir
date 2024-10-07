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
#include "clang/AST/ASTFwd.h"
#include "clang/AST/StmtIterator.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/Basic/OpenMPKinds.h"

#include "CIRGenFunction.h"
#include "CIRGenOpenMPRuntime.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

using namespace cir;
using namespace clang;
using namespace mlir::omp;

static void buildDependences(const OMPExecutableDirective &s,
                             OMPTaskDataTy &data) {

  // First look for 'omp_all_memory' and add this first.
  bool ompAllMemory = false;
  if (llvm::any_of(
          s.getClausesOfKind<OMPDependClause>(), [](const OMPDependClause *c) {
            return c->getDependencyKind() == OMPC_DEPEND_outallmemory ||
                   c->getDependencyKind() == OMPC_DEPEND_inoutallmemory;
          })) {
    ompAllMemory = true;
    // Since both OMPC_DEPEND_outallmemory and OMPC_DEPEND_inoutallmemory are
    // equivalent to the runtime, always use OMPC_DEPEND_outallmemory to
    // simplify.
    OMPTaskDataTy::DependData &dd =
        data.Dependences.emplace_back(OMPC_DEPEND_outallmemory,
                                      /*IteratorExpr=*/nullptr);
    // Add a nullptr Expr to simplify the codegen in emitDependData.
    dd.DepExprs.push_back(nullptr);
  }
  // Add remaining dependences skipping any 'out' or 'inout' if they are
  // overridden by 'omp_all_memory'.
  for (const auto *c : s.getClausesOfKind<OMPDependClause>()) {
    OpenMPDependClauseKind kind = c->getDependencyKind();
    if (kind == OMPC_DEPEND_outallmemory || kind == OMPC_DEPEND_inoutallmemory)
      continue;
    if (ompAllMemory && (kind == OMPC_DEPEND_out || kind == OMPC_DEPEND_inout))
      continue;
    OMPTaskDataTy::DependData &dd =
        data.Dependences.emplace_back(c->getDependencyKind(), c->getModifier());
    dd.DepExprs.append(c->varlist_begin(), c->varlist_end());
  }
}

mlir::LogicalResult
CIRGenFunction::buildOMPParallelDirective(const OMPParallelDirective &s) {
  mlir::LogicalResult res = mlir::success();
  auto scopeLoc = getLoc(s.getSourceRange());
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
        if (buildStmt(s.getCapturedStmt(OpenMPDirectiveKind::OMPD_parallel)
                          ->getCapturedStmt(),
                      /*useCurrentScope=*/true)
                .failed())
          res = mlir::failure();
      });
  // Add the terminator for `omp.parallel`.
  builder.create<TerminatorOp>(getLoc(s.getSourceRange().getEnd()));
  return res;
}

mlir::LogicalResult
CIRGenFunction::buildOMPTaskwaitDirective(const OMPTaskwaitDirective &s) {
  mlir::LogicalResult res = mlir::success();
  OMPTaskDataTy data;
  buildDependences(s, data);
  data.HasNowaitClause = s.hasClausesOfKind<OMPNowaitClause>();
  cgm.getOpenMPRuntime().emitTaskWaitCall(builder, *this,
                                          getLoc(s.getSourceRange()), data);
  return res;
}
mlir::LogicalResult
CIRGenFunction::buildOMPTaskyieldDirective(const OMPTaskyieldDirective &s) {
  mlir::LogicalResult res = mlir::success();
  // Creation of an omp.taskyield operation
  cgm.getOpenMPRuntime().emitTaskyieldCall(builder, *this,
                                           getLoc(s.getSourceRange()));
  return res;
}

mlir::LogicalResult
CIRGenFunction::buildOMPBarrierDirective(const OMPBarrierDirective &s) {
  mlir::LogicalResult res = mlir::success();
  // Creation of an omp.barrier operation
  cgm.getOpenMPRuntime().emitBarrierCall(builder, *this,
                                         getLoc(s.getSourceRange()));
  return res;
}
