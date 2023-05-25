//===- CIRBuilder.cpp - MLIR Generation from a Toy AST --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting MLIR from a Module AST
// for the Toy language.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/CIRBuilder.h"
#include "clang/CIR/CIRCodeGenFunction.h"

#include "mlir/Dialect/CIR/IR/CIRDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using namespace mlir::cir;
using namespace cir;
using namespace clang;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

CIRContext::~CIRContext() {}
CIRContext::CIRContext(clang::ASTContext &AC) : astCtx(AC) { Init(); }

CIRCodeGenFunction::CIRCodeGenFunction() = default;
TypeEvaluationKind CIRCodeGenFunction::getEvaluationKind(QualType type) {
  type = type.getCanonicalType();
  while (true) {
    switch (type->getTypeClass()) {
#define TYPE(name, parent)
#define ABSTRACT_TYPE(name, parent)
#define NON_CANONICAL_TYPE(name, parent) case Type::name:
#define DEPENDENT_TYPE(name, parent) case Type::name:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(name, parent) case Type::name:
#include "clang/AST/TypeNodes.inc"
      llvm_unreachable("non-canonical or dependent type in IR-generation");

    case Type::Auto:
    case Type::DeducedTemplateSpecialization:
      llvm_unreachable("undeduced type in IR-generation");

    // Various scalar types.
    case Type::Builtin:
    case Type::Pointer:
    case Type::BlockPointer:
    case Type::LValueReference:
    case Type::RValueReference:
    case Type::MemberPointer:
    case Type::Vector:
    case Type::ExtVector:
    case Type::ConstantMatrix:
    case Type::FunctionProto:
    case Type::FunctionNoProto:
    case Type::Enum:
    case Type::ObjCObjectPointer:
    case Type::Pipe:
    case Type::BitInt:
      return TEK_Scalar;

    // Complexes.
    case Type::Complex:
      return TEK_Complex;

    // Arrays, records, and Objective-C objects.
    case Type::ConstantArray:
    case Type::IncompleteArray:
    case Type::VariableArray:
    case Type::Record:
    case Type::ObjCObject:
    case Type::ObjCInterface:
      return TEK_Aggregate;

    // We operate on atomic values according to their underlying type.
    case Type::Atomic:
      type = cast<AtomicType>(type)->getValueType();
      continue;
    }
    llvm_unreachable("unknown type kind!");
  }
}

namespace cir {

/// Implementation of a CIR/MLIR emission from Clang AST.
///
/// This will emit operations that are specific to C(++)/ObjC(++) language,
/// preserving the semantics of the language and (hopefully) allow to perform
/// accurate analysis and transformation based on these high level semantics.
class CIRBuildImpl {
public:
  CIRBuildImpl(mlir::MLIRContext &context, clang::ASTContext &astctx)
      : builder(&context), astCtx(astctx) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  }
  CIRBuildImpl(CIRBuildImpl &) = delete;
  CIRBuildImpl &operator=(CIRBuildImpl &) = delete;
  ~CIRBuildImpl() = default;

  using SymTableTy = llvm::ScopedHashTable<StringRef, mlir::Value>;
  using SymTableScopeTy = ScopedHashTableScope<llvm::StringRef, mlir::Value>;

private:
  /// A "module" matches a c/cpp source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The
  /// builder is stateful, in particular it keeps an "insertion point": this
  /// is where the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated,
  /// the scope is destroyed and the mappings created in this scope are
  /// dropped.
  SymTableTy symbolTable;

  /// Hold Clang AST information.
  clang::ASTContext &astCtx;

  /// Per-function codegen information. Updated everytime buildCIR is called
  /// for FunctionDecls's.
  CIRCodeGenFunction *CurCCGF = nullptr;

  /// Helper conversion from Clang source location to an MLIR location.
  mlir::Location getLoc(SourceLocation SLoc) {
    const SourceManager &SM = astCtx.getSourceManager();
    PresumedLoc PLoc = SM.getPresumedLoc(SLoc);
    StringRef Filename = PLoc.getFilename();
    return mlir::FileLineColLoc::get(builder.getStringAttr(Filename),
                                     PLoc.getLine(), PLoc.getColumn());
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(StringRef var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

public:
  mlir::ModuleOp getModule() { return theModule; }

  class ScalarExprEmitter : public StmtVisitor<ScalarExprEmitter, mlir::Value> {
    CIRCodeGenFunction &CGF;
    CIRBuildImpl &Builder;

  public:
    ScalarExprEmitter(CIRCodeGenFunction &cgf, CIRBuildImpl &builder)
        : CGF(cgf), Builder(builder) {
      (void)CGF;
    }

    mlir::Value Visit(Expr *E) {
      return StmtVisitor<ScalarExprEmitter, mlir::Value>::Visit(E);
    }

    class RawAddress {
      mlir::Value Pointer;
      CharUnits Alignment;

    public:
      RawAddress(mlir::Value pointer, CharUnits alignment)
          : Pointer(pointer), Alignment(alignment) {
        assert((!alignment.isZero() || pointer == nullptr) &&
               "creating valid address with invalid alignment");
      }

      static RawAddress invalid() { return RawAddress(nullptr, CharUnits()); }
      bool isValid() const { return Pointer != nullptr; }

      mlir::Value getPointer() const {
        // assert(isValid());
        return Pointer;
      }

      /// Return the alignment of this pointer.
      CharUnits getAlignment() const {
        // assert(isValid());
        return Alignment;
      }
    };
    class LValue {
    private:
      void Initialize(CharUnits Alignment, LValueBaseInfo BaseInfo) {
        // assert((!Alignment.isZero()) && // || Type->isIncompleteType()) &&
        //       "initializing l-value with zero alignment!");

        const unsigned MaxAlign = 1U << 31;
        this->Alignment = Alignment.getQuantity() <= MaxAlign
                              ? Alignment.getQuantity()
                              : MaxAlign;
        assert(this->Alignment == Alignment.getQuantity() &&
               "Alignment exceeds allowed max!");
        this->BaseInfo = BaseInfo;
      }

      // The alignment to use when accessing this lvalue. (For vector elements,
      // this is the alignment of the whole vector)
      unsigned Alignment;
      mlir::Value V;
      LValueBaseInfo BaseInfo;

    public:
      mlir::Value getPointer() const { return V; }

      CharUnits getAlignment() const {
        return CharUnits::fromQuantity(Alignment);
      }

      RawAddress getAddress() const {
        return RawAddress(getPointer(), getAlignment());
      }

      LValueBaseInfo getBaseInfo() const { return BaseInfo; }
      void setBaseInfo(LValueBaseInfo Info) { BaseInfo = Info; }

      static LValue makeAddr(RawAddress address,
                             AlignmentSource Source = AlignmentSource::Type) {
        LValue R;
        R.V = address.getPointer();
        R.Initialize(address.getAlignment(), LValueBaseInfo(Source));
        return R;
      }
    };

    LValue EmitDeclRefLValue(const DeclRefExpr *E) {
      const NamedDecl *ND = E->getDecl();

      assert(E->isNonOdrUse() != NOUR_Unevaluated &&
             "should not emit an unevaluated operand");

      if (const auto *VD = dyn_cast<VarDecl>(ND)) {
        // Global Named registers access via intrinsics only
        assert(VD->getStorageClass() != SC_Register && "not implemented");
        assert(E->isNonOdrUse() != NOUR_Constant && "not implemented");
        assert(!E->refersToEnclosingVariableOrCapture() && "not implemented");
        assert(!(VD->hasLinkage() || VD->isStaticDataMember()) &&
               "not implemented");
        assert(!VD->isEscapingByref() && "not implemented");
        assert(!VD->getType()->isReferenceType() && "not implemented");
        assert(Builder.symbolTable.count(VD->getName()) &&
               "should be already mapped");

        mlir::Value V = Builder.symbolTable.lookup(VD->getName());
        assert(V && "Name lookup must succeed");

        LValue LV = LValue::makeAddr(RawAddress(V, CharUnits::fromQuantity(4)),
                                     AlignmentSource::Decl);
        return LV;
      }

      llvm_unreachable("Unhandled DeclRefExpr?");
    }

    LValue EmitLValue(const Expr *E) {
      switch (E->getStmtClass()) {
      case Expr::DeclRefExprClass:
        return EmitDeclRefLValue(cast<DeclRefExpr>(E));
      default:
        emitError(Builder.getLoc(E->getExprLoc()),
                  "l-value not implemented for '")
            << E->getStmtClassName() << "'";
        break;
      }
      return LValue::makeAddr(RawAddress::invalid());
    }

    /// Emits the address of the l-value, then loads and returns the result.
    mlir::Value buildLoadOfLValue(const Expr *E) {
      LValue LV = EmitLValue(E);
      // mlir::Value V = EmitLoadOfLValue(LV, E->getExprLoc());

      // EmitLValueAlignmentAssumption(E, V);
      // return V;
      return LV.getPointer();
    }

    // Handle l-values.
    mlir::Value VisitDeclRefExpr(DeclRefExpr *E) {
      // FIXME: we could try to emit this as constant, similar to LLVM IR
      // codegen.
      return buildLoadOfLValue(E);
    }

    // Emit code for an explicit or implicit cast.  Implicit
    // casts have to handle a more broad range of conversions than explicit
    // casts, as they handle things like function to ptr-to-function decay
    // etc.
    mlir::Value VisitCastExpr(CastExpr *CE) {
      Expr *E = CE->getSubExpr();
      QualType DestTy = CE->getType();
      CastKind Kind = CE->getCastKind();
      switch (Kind) {
      case CK_LValueToRValue:
        assert(Builder.astCtx.hasSameUnqualifiedType(E->getType(), DestTy));
        assert(E->isGLValue() && "lvalue-to-rvalue applied to r-value!");
        return Visit(const_cast<Expr *>(E));
      default:
        emitError(Builder.getLoc(CE->getExprLoc()),
                  "cast kind not implemented: '")
            << CE->getCastKindName() << "'";
        return nullptr;
      }
    }

    mlir::Value VisitExpr(Expr *E) {
      emitError(Builder.getLoc(E->getExprLoc()), "scalar exp no implemented: '")
          << E->getStmtClassName() << "'";
      if (E->getType()->isVoidType())
        return nullptr;
      // FIXME: find a way to return "undef"...
      // return llvm::UndefValue::get(CGF.ConvertType(E->getType()));
      return nullptr;
    }
  };

  /// Emit the computation of the specified expression of scalar type,
  /// ignoring the result.
  mlir::Value buildScalarExpr(const Expr *E) {
    assert(E && CIRCodeGenFunction::hasScalarEvaluationKind(E->getType()) &&
           "Invalid scalar expression to emit");

    return ScalarExprEmitter(*CurCCGF, *this).Visit(const_cast<Expr *>(E));
  }

  mlir::LogicalResult buildReturnStmt(const ReturnStmt &S) {
    // Emit the result value, even if unused, to evaluate the side effects.
    const Expr *RV = S.getRetValue();
    assert(!isa<ExprWithCleanups>(RV) && "unimplemented");
    assert(!(astCtx.getLangOpts().ElideConstructors && S.getNRVOCandidate() &&
             S.getNRVOCandidate()->isNRVOVariable()) &&
           "unimplemented");
    assert(!CurCCGF->FnRetQualTy->isReferenceType() && "unimplemented");

    if (!RV) // Do nothing (return value is left uninitialized)
      return mlir::success();

    mlir::Value V = nullptr;
    switch (CIRCodeGenFunction::getEvaluationKind(RV->getType())) {
    case TEK_Scalar:
      V = buildScalarExpr(RV);
      // Builder.CreateStore(EmitScalarExpr(RV), ReturnValue);
      break;
    case TEK_Complex:
    case TEK_Aggregate:
      llvm::errs() << "ReturnStmt EvaluationKind not implemented\n";
      return mlir::failure();
    }

    CurCCGF->RetValue = V;
    // Otherwise, this return operation has zero operands.
    if (!V || (RV && RV->getType()->isVoidType())) {
      // FIXME: evaluate for side effects.
    }

    builder.create<ReturnOp>(getLoc(RV->getExprLoc()),
                             V ? ArrayRef(V) : ArrayRef<mlir::Value>());
    return mlir::success();
  }

  mlir::LogicalResult buildCompoundStmt(const CompoundStmt &S) {
    // Create a scope in the symbol table to hold variable declarations local
    // to this compound statement.
    SymTableScopeTy varScope(symbolTable);
    for (auto *CurStmt : S.body())
      if (buildStmt(CurStmt).failed())
        return mlir::failure();

    return mlir::success();
  }

  mlir::LogicalResult buildStmt(const Stmt *S) {
    switch (S->getStmtClass()) {
    default:
      llvm::errs() << "CIR codegen for '" << S->getStmtClassName()
                   << "' not implemented\n";
      return mlir::failure();
    case Stmt::CompoundStmtClass:
      return buildCompoundStmt(cast<CompoundStmt>(*S));
    case Stmt::ReturnStmtClass:
      return buildReturnStmt(cast<ReturnStmt>(*S));
    }

    return mlir::success();
  }

  // Emit a new function and add it to the MLIR module.
  mlir::FuncOp buildCIR(CIRCodeGenFunction *CCGF, const FunctionDecl *FD) {
    CurCCGF = CCGF;

    // Create a scope in the symbol table to hold variable declarations.
    SymTableScopeTy varScope(symbolTable);

    const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD);
    assert(!MD && "methods not implemented");
    auto loc = getLoc(FD->getLocation());

    // Create an MLIR function for the given prototype.
    llvm::SmallVector<mlir::Type, 4> argTypes;

    for (auto *Param : FD->parameters())
      argTypes.push_back(getType(Param->getType()));

    CurCCGF->FnRetQualTy = FD->getReturnType();
    CurCCGF->FnRetTy = getType(CurCCGF->FnRetQualTy);
    auto funcType = builder.getFunctionType(argTypes, CurCCGF->FnRetTy);
    mlir::FuncOp function = mlir::FuncOp::create(loc, FD->getName(), funcType);
    if (!function)
      return nullptr;

    // In MLIR the entry block of the function is special: it must have the
    // same argument list as the function itself.
    auto &entryBlock = *function.addEntryBlock();

    // Declare all the function arguments in the symbol table.
    for (const auto nameValue :
         llvm::zip(FD->parameters(), entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(nameValue)->getName(),
                         std::get<1>(nameValue))))
        return nullptr;
    }

    // Set the insertion point in the builder to the beginning of the
    // function body, it will be used throughout the codegen to create
    // operations in this function.
    builder.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    if (mlir::failed(buildStmt(FD->getBody()))) {
      function.erase();
      return nullptr;
    }

    ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = dyn_cast<ReturnOp>(entryBlock.back());
    if (!returnOp)
      builder.create<ReturnOp>(loc);

    assert(function.verifyBody().succeeded());
    theModule.push_back(function);
    return function;
  }

  mlir::Type getType(const QualType &type) {
    // FIXME: actually map into the appropriated types.
    return builder.getI32Type();
  }

  void verifyModule() {
    // Verify the module after we have finished constructing it, this will
    // check the structural properties of the IR and invoke any specific
    // verifiers we have on the CIR operations.
    if (failed(mlir::verify(theModule)))
      theModule.emitError("module verification error");
  }
};
} // namespace cir

void CIRContext::Init() {
  mlirCtx = std::make_unique<mlir::MLIRContext>();
  mlirCtx->getOrLoadDialect<mlir::cir::CIRDialect>();
  builder = std::make_unique<CIRBuildImpl>(*mlirCtx.get(), astCtx);
}

bool CIRContext::EmitFunction(const FunctionDecl *FD) {
  CIRCodeGenFunction CCGF{};
  auto func = builder->buildCIR(&CCGF, FD);
  assert(func && "should emit function");

  // FIXME: currently checked after emitting every function, should
  // only run when the consumer of the context shutdowns.
  builder->verifyModule();
  builder->getModule().dump();
  return true;
}
