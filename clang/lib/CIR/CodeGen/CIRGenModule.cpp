//===- CIRGenModule.cpp - Per-Module state for CIR generation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-translation-unit state used for CIR translation.
//
//===----------------------------------------------------------------------===//

#include "CIRGenModule.h"

#include "CIRGenCXXABI.h"
#include "CIRGenCstEmitter.h"
#include "CIRGenFunction.h"
#include "CIRGenOpenMPRuntime.h"
#include "CIRGenTBAA.h"
#include "CIRGenTypes.h"
#include "CIRGenValue.h"
#include "TargetInfo.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/Cuda.h"
#include "clang/CIR/MissingFeatures.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Basic/NoSanitizeList.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/CIR/CIRGenerator.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/Frontend/FrontendDiagnostic.h"
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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <numeric>

using namespace mlir::cir;
using namespace cir;
using namespace clang;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::SmallVector;
using llvm::StringRef;

static CIRGenCXXABI *createCXXABI(CIRGenModule &cgm) {
  switch (cgm.getASTContext().getCXXABIKind()) {
  case TargetCXXABI::GenericItanium:
  case TargetCXXABI::GenericAArch64:
  case TargetCXXABI::AppleARM64:
    return CreateCIRGenItaniumCXXABI(cgm);
  default:
    llvm_unreachable("invalid C++ ABI kind");
  }
}

CIRGenModule::CIRGenModule(mlir::MLIRContext &context,
                           clang::ASTContext &astctx,
                           const clang::CodeGenOptions &cgo,
                           DiagnosticsEngine &diags)
    : builder(context, *this), astCtx(astctx), langOpts(astctx.getLangOpts()),
      codeGenOpts(cgo),
      theModule{mlir::ModuleOp::create(builder.getUnknownLoc())}, Diags(diags),
      target(astCtx.getTargetInfo()), ABI(createCXXABI(*this)), genTypes{*this},
      VTables{*this}, openMPRuntime(new CIRGenOpenMPRuntime(*this)) {

  // Initialize CIR signed integer types cache.
  SInt8Ty =
      ::mlir::cir::IntType::get(builder.getContext(), 8, /*isSigned=*/true);
  SInt16Ty =
      ::mlir::cir::IntType::get(builder.getContext(), 16, /*isSigned=*/true);
  SInt32Ty =
      ::mlir::cir::IntType::get(builder.getContext(), 32, /*isSigned=*/true);
  SInt64Ty =
      ::mlir::cir::IntType::get(builder.getContext(), 64, /*isSigned=*/true);

  // Initialize CIR unsigned integer types cache.
  UInt8Ty =
      ::mlir::cir::IntType::get(builder.getContext(), 8, /*isSigned=*/false);
  UInt16Ty =
      ::mlir::cir::IntType::get(builder.getContext(), 16, /*isSigned=*/false);
  UInt32Ty =
      ::mlir::cir::IntType::get(builder.getContext(), 32, /*isSigned=*/false);
  UInt64Ty =
      ::mlir::cir::IntType::get(builder.getContext(), 64, /*isSigned=*/false);

  VoidTy = ::mlir::cir::VoidType::get(builder.getContext());

  // Initialize CIR pointer types cache.
  VoidPtrTy = ::mlir::cir::PointerType::get(builder.getContext(), VoidTy);

  FP16Ty = ::mlir::cir::FP16Type::get(builder.getContext());
  BFloat16Ty = ::mlir::cir::BF16Type::get(builder.getContext());
  FloatTy = ::mlir::cir::SingleType::get(builder.getContext());
  DoubleTy = ::mlir::cir::DoubleType::get(builder.getContext());
  FP80Ty = ::mlir::cir::FP80Type::get(builder.getContext());

  // TODO: PointerWidthInBits
  PointerAlignInBytes =
      astctx
          .toCharUnitsFromBits(
              astctx.getTargetInfo().getPointerAlign(LangAS::Default))
          .getQuantity();
  // TODO: SizeSizeInBytes
  // TODO: IntAlignInBytes
  UCharTy = ::mlir::cir::IntType::get(builder.getContext(),
                                      astCtx.getTargetInfo().getCharWidth(),
                                      /*isSigned=*/false);
  UIntTy = ::mlir::cir::IntType::get(builder.getContext(),
                                     astCtx.getTargetInfo().getIntWidth(),
                                     /*isSigned=*/false);
  UIntPtrTy = ::mlir::cir::IntType::get(
      builder.getContext(), astCtx.getTargetInfo().getMaxPointerWidth(),
      /*isSigned=*/false);
  UInt8PtrTy = builder.getPointerTo(UInt8Ty);
  UInt8PtrPtrTy = builder.getPointerTo(UInt8PtrTy);
  AllocaInt8PtrTy = UInt8PtrTy;
  // TODO: GlobalsInt8PtrTy
  // TODO: ConstGlobalsPtrTy
  CIRAllocaAddressSpace = getTargetCIRGenInfo().getCIRAllocaAddressSpace();

  PtrDiffTy = ::mlir::cir::IntType::get(
      builder.getContext(), astCtx.getTargetInfo().getMaxPointerWidth(),
      /*isSigned=*/true);

  if (langOpts.OpenCL) {
    createOpenCLRuntime();
  }

  mlir::cir::sob::SignedOverflowBehavior sob;
  switch (langOpts.getSignedOverflowBehavior()) {
  case clang::LangOptions::SignedOverflowBehaviorTy::SOB_Defined:
    sob = sob::SignedOverflowBehavior::defined;
    break;
  case clang::LangOptions::SignedOverflowBehaviorTy::SOB_Undefined:
    sob = sob::SignedOverflowBehavior::undefined;
    break;
  case clang::LangOptions::SignedOverflowBehaviorTy::SOB_Trapping:
    sob = sob::SignedOverflowBehavior::trapping;
    break;
  }

  // FIXME(cir): Implement a custom CIR Module Op and attributes to leverage
  // MLIR features.
  theModule->setAttr("cir.sob",
                     mlir::cir::SignedOverflowBehaviorAttr::get(&context, sob));
  auto lang = SourceLanguageAttr::get(&context, getCIRSourceLanguage());
  theModule->setAttr(
      "cir.lang", mlir::cir::LangAttr::get(&context, lang));
  theModule->setAttr("cir.triple", builder.getStringAttr(getTriple().str()));
  // Set the module name to be the name of the main file. TranslationUnitDecl
  // often contains invalid source locations and isn't a reliable source for the
  // module location.
  auto mainFileId = astctx.getSourceManager().getMainFileID();
  const FileEntry &mainFile =
      *astctx.getSourceManager().getFileEntryForID(mainFileId);
  auto path = mainFile.tryGetRealPathName();
  if (!path.empty()) {
    theModule.setSymName(path);
    theModule->setLoc(mlir::FileLineColLoc::get(&context, path,
                                                /*line=*/0,
                                                /*col=*/0));
  }
}

CIRGenModule::~CIRGenModule() = default;

bool CIRGenModule::isTypeConstant(QualType ty, bool excludeCtor,
                                  bool excludeDtor) {
  if (!ty.isConstant(astCtx) && !ty->isReferenceType())
    return false;

  if (astCtx.getLangOpts().CPlusPlus) {
    if (const CXXRecordDecl *record =
            astCtx.getBaseElementType(ty)->getAsCXXRecordDecl())
      return excludeCtor && !record->hasMutableFields() &&
             (record->hasTrivialDestructor() || excludeDtor);
  }

  return true;
}

/// FIXME: this could likely be a common helper and not necessarily related
/// with codegen.
/// Return the best known alignment for an unknown pointer to a
/// particular class.
CharUnits CIRGenModule::getClassPointerAlignment(const CXXRecordDecl *rd) {
  if (!rd->hasDefinition())
    return CharUnits::One(); // Hopefully won't be used anywhere.

  auto &layout = astCtx.getASTRecordLayout(rd);

  // If the class is final, then we know that the pointer points to an
  // object of that type and can use the full alignment.
  if (rd->isEffectivelyFinal())
    return layout.getAlignment();

  // Otherwise, we have to assume it could be a subclass.
  return layout.getNonVirtualAlignment();
}

/// FIXME: this could likely be a common helper and not necessarily related
/// with codegen.
CharUnits CIRGenModule::getNaturalPointeeTypeAlignment(
    QualType ty, LValueBaseInfo *baseInfo, TBAAAccessInfo *tbaaInfo) {
  return getNaturalTypeAlignment(ty->getPointeeType(), baseInfo, tbaaInfo,
                                 /* forPointeeType= */ true);
}

/// FIXME: this could likely be a common helper and not necessarily related
/// with codegen.
/// TODO: Add TBAAAccessInfo
CharUnits CIRGenModule::getNaturalTypeAlignment(QualType t,
                                                LValueBaseInfo *baseInfo,
                                                TBAAAccessInfo *tbaaInfo,
                                                bool forPointeeType) {
  // FIXME: This duplicates logic in ASTContext::getTypeAlignIfKnown. But
  // that doesn't return the information we need to compute BaseInfo.

  // Honor alignment typedef attributes even on incomplete types.
  // We also honor them straight for C++ class types, even as pointees;
  // there's an expressivity gap here.
  if (const auto *tt = t->getAs<TypedefType>()) {
    if (auto align = tt->getDecl()->getMaxAlignment()) {
      if (baseInfo)
        *baseInfo = LValueBaseInfo(AlignmentSource::AttributedType);
      return astCtx.toCharUnitsFromBits(align);
    }
  }

  bool alignForArray = t->isArrayType();

  // Analyze the base element type, so we don't get confused by incomplete
  // array types.
  t = astCtx.getBaseElementType(t);

  if (t->isIncompleteType()) {
    // We could try to replicate the logic from
    // ASTContext::getTypeAlignIfKnown, but nothing uses the alignment if the
    // type is incomplete, so it's impossible to test. We could try to reuse
    // getTypeAlignIfKnown, but that doesn't return the information we need
    // to set BaseInfo.  So just ignore the possibility that the alignment is
    // greater than one.
    if (baseInfo)
      *baseInfo = LValueBaseInfo(AlignmentSource::Type);
    return CharUnits::One();
  }

  if (baseInfo)
    *baseInfo = LValueBaseInfo(AlignmentSource::Type);

  CharUnits alignment;
  const CXXRecordDecl *rd;
  if (t.getQualifiers().hasUnaligned()) {
    alignment = CharUnits::One();
  } else if (forPointeeType && !alignForArray &&
             (rd = t->getAsCXXRecordDecl())) {
    // For C++ class pointees, we don't know whether we're pointing at a
    // base or a complete object, so we generally need to use the
    // non-virtual alignment.
    alignment = getClassPointerAlignment(rd);
  } else {
    alignment = astCtx.getTypeAlignInChars(t);
  }

  // Cap to the global maximum type alignment unless the alignment
  // was somehow explicit on the type.
  if (unsigned maxAlign = astCtx.getLangOpts().MaxTypeAlign) {
    if (alignment.getQuantity() > maxAlign && !astCtx.isAlignmentRequired(t))
      alignment = CharUnits::fromQuantity(maxAlign);
  }
  return alignment;
}

bool CIRGenModule::MustBeEmitted(const ValueDecl *global) {
  // Never defer when EmitAllDecls is specified.
  assert(!langOpts.EmitAllDecls && "EmitAllDecls NYI");
  assert(!codeGenOpts.KeepStaticConsts && "KeepStaticConsts NYI");

  return getASTContext().DeclMustBeEmitted(global);
}

bool CIRGenModule::MayBeEmittedEagerly(const ValueDecl *global) {
  // In OpenMP 5.0 variables and function may be marked as
  // device_type(host/nohost) and we should not emit them eagerly unless we sure
  // that they must be emitted on the host/device. To be sure we need to have
  // seen a declare target with an explicit mentioning of the function, we know
  // we have if the level of the declare target attribute is -1. Note that we
  // check somewhere else if we should emit this at all.
  if (langOpts.OpenMP >= 50 && !langOpts.OpenMPSimd) {
    std::optional<OMPDeclareTargetDeclAttr *> activeAttr =
        OMPDeclareTargetDeclAttr::getActiveAttr(global);
    if (!activeAttr || (*activeAttr)->getLevel() != (unsigned)-1)
      return false;
  }

  const auto *fd = dyn_cast<FunctionDecl>(global);
  if (fd) {
    // Implicit template instantiations may change linkage if they are later
    // explicitly instantiated, so they should not be emitted eagerly.
    // TODO(cir): do we care?
    assert(fd->getTemplateSpecializationKind() != TSK_ImplicitInstantiation &&
           "not implemented");
    assert(!fd->isTemplated() && "Templates NYI");
  }
  const auto *vd = dyn_cast<VarDecl>(global);
  if (vd)
    // A definition of an inline constexpr static data member may change
    // linkage later if it's redeclared outside the class.
    // TODO(cir): do we care?
    assert(astCtx.getInlineVariableDefinitionKind(vd) !=
               ASTContext::InlineVariableDefinitionKind::WeakUnknown &&
           "not implemented");

  // If OpenMP is enabled and threadprivates must be generated like TLS, delay
  // codegen for global variables, because they may be marked as threadprivate.
  if (langOpts.OpenMP && langOpts.OpenMPUseTLS &&
      getASTContext().getTargetInfo().isTLSSupported() &&
      isa<VarDecl>(global) &&
      !global->getType().isConstantStorage(getASTContext(), false, false) &&
      !OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(global))
    return false;

  assert((fd || vd) &&
         "Only FunctionDecl and VarDecl should hit this path so far.");
  return true;
}

static bool shouldAssumeDSOLocal(const CIRGenModule &cgm,
                                 CIRGlobalValueInterface gv) {
  if (gv.hasLocalLinkage())
    return true;

  if (!gv.hasDefaultVisibility() && !gv.hasExternalWeakLinkage()) {
    return true;
  }

  // DLLImport explicitly marks the GV as external.
  // so it shouldn't be dso_local
  // But we don't have the info set now
  assert(!MissingFeatures::setDLLImportDLLExport());

  const llvm::Triple &tt = cgm.getTriple();
  const auto &cgOpts = cgm.getCodeGenOpts();
  if (tt.isWindowsGNUEnvironment()) {
    // In MinGW, variables without DLLImport can still be automatically
    // imported from a DLL by the linker; don't mark variables that
    // potentially could come from another DLL as DSO local.

    // With EmulatedTLS, TLS variables can be autoimported from other DLLs
    // (and this actually happens in the public interface of libstdc++), so
    // such variables can't be marked as DSO local. (Native TLS variables
    // can't be dllimported at all, though.)
    llvm_unreachable("MinGW not supported here");
  }

  // On COFF, don't mark 'extern_weak' symbols as DSO local. If these symbols
  // remain unresolved in the link, they can be resolved to zero, which is
  // outside the current DSO.
  if (tt.isOSBinFormatCOFF() && gv.hasExternalWeakLinkage())
    return false;

  // Every other GV is local on COFF.
  // Make an exception for windows OS in the triple: Some firmware builds use
  // *-win32-macho triples. This (accidentally?) produced windows relocations
  // without GOT tables in older clang versions; Keep this behaviour.
  // FIXME: even thread local variables?
  if (tt.isOSBinFormatCOFF() || (tt.isOSWindows() && tt.isOSBinFormatMachO()))
    return true;

  // Only handle COFF and ELF for now.
  if (!tt.isOSBinFormatELF())
    return false;

  llvm::Reloc::Model rm = cgOpts.RelocationModel;
  const auto &lOpts = cgm.getLangOpts();
  if (rm != llvm::Reloc::Static && !lOpts.PIE) {
    // On ELF, if -fno-semantic-interposition is specified and the target
    // supports local aliases, there will be neither CC1
    // -fsemantic-interposition nor -fhalf-no-semantic-interposition. Set
    // dso_local on the function if using a local alias is preferable (can avoid
    // PLT indirection).
    if (!(isa<mlir::cir::FuncOp>(gv) && gv.canBenefitFromLocalAlias())) {
      return false;
    }
    return !(cgm.getLangOpts().SemanticInterposition ||
             cgm.getLangOpts().HalfNoSemanticInterposition);
  }

  // A definition cannot be preempted from an executable.
  if (!gv.isDeclarationForLinker())
    return true;

  // Most PIC code sequences that assume that a symbol is local cannot produce a
  // 0 if it turns out the symbol is undefined. While this is ABI and relocation
  // depended, it seems worth it to handle it here.
  if (rm == llvm::Reloc::PIC_ && gv.hasExternalWeakLinkage())
    return false;

  // PowerPC64 prefers TOC indirection to avoid copy relocations.
  if (tt.isPPC64())
    return false;

  if (cgOpts.DirectAccessExternalData) {
    llvm_unreachable("-fdirect-access-external-data not supported");
  }

  // If we can use copy relocations we can assume it is local.

  // Otherwise don't assume it is local.

  return false;
}

void CIRGenModule::setDSOLocal(CIRGlobalValueInterface gv) const {
  gv.setDSOLocal(shouldAssumeDSOLocal(*this, gv));
}

void CIRGenModule::buildGlobal(GlobalDecl gd) {
  llvm::TimeTraceScope scope("build CIR Global", [&]() -> std::string {
    auto *nd = dyn_cast<NamedDecl>(gd.getDecl());
    if (!nd)
      // TODO: How to print decls which is not named decl?
      return "Unnamed decl";

    std::string name;
    llvm::raw_string_ostream os(name);
    nd->getNameForDiagnostic(os, getASTContext().getPrintingPolicy(),
                             /*Qualified=*/true);
    return name;
  });

  const auto *global = cast<ValueDecl>(gd.getDecl());

  assert(!global->hasAttr<IFuncAttr>() && "NYI");
  assert(!global->hasAttr<CPUDispatchAttr>() && "NYI");
  assert(!langOpts.CUDA && "NYI");

  if (langOpts.OpenMP) {
    // If this is OpenMP, check if it is legal to emit this global normally.
    if (openMPRuntime && openMPRuntime->emitTargetGlobal(gd)) {
      assert(!MissingFeatures::openMPRuntime());
      return;
    }
    if (auto *drd = dyn_cast<OMPDeclareReductionDecl>(global)) {
      assert(!MissingFeatures::openMP());
      return;
    }
    if (auto *dmd = dyn_cast<OMPDeclareMapperDecl>(global)) {
      assert(!MissingFeatures::openMP());
      return;
    }
  }

  // Ignore declarations, they will be emitted on their first use.
  if (const auto *fd = dyn_cast<FunctionDecl>(global)) {
    // Update deferred annotations with the latest declaration if the function
    // was already used or defined.
    if (fd->hasAttr<AnnotateAttr>()) {
      StringRef mangledName = getMangledName(gd);
      if (getGlobalValue(mangledName))
        deferredAnnotations[mangledName] = fd;
    }
    // Forward declarations are emitted lazily on first use.
    if (!fd->doesThisDeclarationHaveABody()) {
      if (!fd->doesDeclarationForceExternallyVisibleDefinition())
        return;

      llvm::StringRef mangledName = getMangledName(gd);

      // Compute the function info and CIR type.
      const auto &fi = getTypes().arrangeGlobalDeclaration(gd);
      mlir::Type ty = getTypes().GetFunctionType(fi);

      GetOrCreateCIRFunction(mangledName, ty, gd, /*ForVTable=*/false,
                             /*DontDefer=*/false);
      return;
    }
  } else {
    const auto *vd = cast<VarDecl>(global);
    assert(vd->isFileVarDecl() && "Cannot emit local var decl as global.");
    if (vd->isThisDeclarationADefinition() != VarDecl::Definition &&
        !astCtx.isMSStaticDataMemberInlineDefinition(vd)) {
      if (langOpts.OpenMP) {
        // Emit declaration of the must-be-emitted declare target variable.
        if (std::optional<OMPDeclareTargetDeclAttr::MapTypeTy> res =
                OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(vd)) {
          assert(0 && "OMPDeclareTargetDeclAttr NYI");
        }
      }
      // If this declaration may have caused an inline variable definition to
      // change linkage, make sure that it's emitted.
      if (astCtx.getInlineVariableDefinitionKind(vd) ==
          ASTContext::InlineVariableDefinitionKind::Strong)
        getAddrOfGlobalVar(vd);
      return;
    }
  }

  // Defer code generation to first use when possible, e.g. if this is an inline
  // function. If the global mjust always be emitted, do it eagerly if possible
  // to benefit from cache locality.
  if (MustBeEmitted(global) && MayBeEmittedEagerly(global)) {
    // Emit the definition if it can't be deferred.
    buildGlobalDefinition(gd);
    return;
  }

  // If we're deferring emission of a C++ variable with an initializer, remember
  // the order in which it appeared on the file.
  if (getLangOpts().CPlusPlus && isa<VarDecl>(global) &&
      cast<VarDecl>(global)->hasInit()) {
    DelayedCXXInitPosition[global] = CXXGlobalInits.size();
    CXXGlobalInits.push_back(nullptr);
  }

  llvm::StringRef mangledName = getMangledName(gd);
  if (getGlobalValue(mangledName) != nullptr) {
    // The value has already been used and should therefore be emitted.
    addDeferredDeclToEmit(gd);
  } else if (MustBeEmitted(global)) {
    // The value must be emitted, but cannot be emitted eagerly.
    assert(!MayBeEmittedEagerly(global));
    addDeferredDeclToEmit(gd);
  } else {
    // Otherwise, remember that we saw a deferred decl with this name. The first
    // use of the mangled name will cause it to move into DeferredDeclsToEmit.
    DeferredDecls[mangledName] = gd;
  }
}

void CIRGenModule::buildGlobalFunctionDefinition(GlobalDecl gd,
                                                 mlir::Operation *op) {
  auto const *d = cast<FunctionDecl>(gd.getDecl());

  // Compute the function info and CIR type.
  const CIRGenFunctionInfo &fi = getTypes().arrangeGlobalDeclaration(gd);
  auto ty = getTypes().GetFunctionType(fi);

  // Get or create the prototype for the function.
  // if (!V || (V.getValueType() != Ty))
  // TODO(cir): Figure out what to do here? llvm uses a GlobalValue for the
  // FuncOp in mlir
  op = GetAddrOfFunction(gd, ty, /*ForVTable=*/false, /*DontDefer=*/true,
                         ForDefinition);

  auto globalVal = dyn_cast_or_null<mlir::cir::CIRGlobalValueInterface>(op);
  if (globalVal && !globalVal.isDeclaration()) {
    // Already emitted.
    return;
  }
  auto fn = cast<mlir::cir::FuncOp>(op);
  setFunctionLinkage(gd, fn);
  setGVProperties(op, d);
  // TODO(cir): MaubeHandleStaticInExternC
  // TODO(cir): maybeSetTrivialComdat
  // TODO(cir): setLLVMFunctionFEnvAttributes

  CIRGenFunction cgf{*this, builder};
  CurCGF = &cgf;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    cgf.generateCode(gd, fn, fi);
  }
  CurCGF = nullptr;

  setNonAliasAttributes(gd, op);
  setCIRFunctionAttributesForDefinition(d, fn);

  if (const ConstructorAttr *ca = d->getAttr<ConstructorAttr>())
    AddGlobalCtor(fn, ca->getPriority());
  if (const DestructorAttr *da = d->getAttr<DestructorAttr>())
    AddGlobalDtor(fn, da->getPriority(), true);

  if (d->getAttr<AnnotateAttr>())
    deferredAnnotations[getMangledName(gd)] = cast<ValueDecl>(d);
}

/// Track functions to be called before main() runs.
void CIRGenModule::AddGlobalCtor(mlir::cir::FuncOp ctor, int priority) {
  // FIXME(cir): handle LexOrder and Associated data upon testcases.
  //
  // Traditional LLVM codegen directly adds the function to the list of global
  // ctors. In CIR we just add a global_ctor attribute to the function. The
  // global list is created in LoweringPrepare.
  //
  // FIXME(from traditional LLVM): Type coercion of void()* types.
  ctor->setAttr(ctor.getGlobalCtorAttrName(),
                mlir::cir::GlobalCtorAttr::get(builder.getContext(),
                                               ctor.getName(), priority));
}

/// Add a function to the list that will be called when the module is unloaded.
void CIRGenModule::AddGlobalDtor(mlir::cir::FuncOp dtor, int priority,
                                 bool isDtorAttrFunc) {
  assert(isDtorAttrFunc && "NYI");
  if (codeGenOpts.RegisterGlobalDtorsWithAtExit &&
      (!getASTContext().getTargetInfo().getTriple().isOSAIX() ||
       isDtorAttrFunc)) {
    llvm_unreachable("NYI");
  }

  // FIXME(from traditional LLVM): Type coercion of void()* types.
  dtor->setAttr(dtor.getGlobalDtorAttrName(),
                mlir::cir::GlobalDtorAttr::get(builder.getContext(),
                                               dtor.getName(), priority));
}

mlir::Operation *CIRGenModule::getGlobalValue(StringRef name) {
  auto *global = mlir::SymbolTable::lookupSymbolIn(theModule, name);
  if (!global)
    return {};
  return global;
}

mlir::Value CIRGenModule::getGlobalValue(const Decl *d) {
  assert(CurCGF);
  return CurCGF->symbolTable.lookup(d);
}

mlir::cir::GlobalOp CIRGenModule::createGlobalOp(
    CIRGenModule &cgm, mlir::Location loc, StringRef name, mlir::Type t,
    bool isConstant, mlir::cir::AddressSpaceAttr addrSpace,
    mlir::Operation *insertPoint, mlir::cir::GlobalLinkageKind linkage) {
  mlir::cir::GlobalOp g;
  auto &builder = cgm.getBuilder();
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Some global emissions are triggered while emitting a function, e.g.
    // void s() { const char *s = "yolo"; ... }
    //
    // Be sure to insert global before the current function
    auto *curCGF = cgm.getCurrCIRGenFun();
    if (curCGF)
      builder.setInsertionPoint(curCGF->CurFn);

    g = builder.create<mlir::cir::GlobalOp>(loc, name, t, isConstant, linkage,
                                            addrSpace);
    if (!curCGF) {
      if (insertPoint)
        cgm.getModule().insert(insertPoint, g);
      else
        cgm.getModule().push_back(g);
    }

    // Default to private until we can judge based on the initializer,
    // since MLIR doesn't allow public declarations.
    mlir::SymbolTable::setSymbolVisibility(
        g, mlir::SymbolTable::Visibility::Private);
  }
  return g;
}

void CIRGenModule::setCommonAttributes(GlobalDecl gd, mlir::Operation *gv) {
  const Decl *d = gd.getDecl();
  if (isa_and_nonnull<NamedDecl>(d))
    setGVProperties(gv, dyn_cast<NamedDecl>(d));
  else
    assert(!MissingFeatures::setDefaultVisibility());

  if (d && d->hasAttr<UsedAttr>())
    assert(!MissingFeatures::addUsedOrCompilerUsedGlobal());

  if (const auto *vd = dyn_cast_if_present<VarDecl>(d);
      vd &&
      ((codeGenOpts.KeepPersistentStorageVariables &&
        (vd->getStorageDuration() == SD_Static ||
         vd->getStorageDuration() == SD_Thread)) ||
       (codeGenOpts.KeepStaticConsts && vd->getStorageDuration() == SD_Static &&
        vd->getType().isConstQualified())))
    assert(!MissingFeatures::addUsedOrCompilerUsedGlobal());
}

void CIRGenModule::setNonAliasAttributes(GlobalDecl gd, mlir::Operation *go) {
  const Decl *d = gd.getDecl();
  setCommonAttributes(gd, go);

  if (d) {
    auto gv = llvm::dyn_cast_or_null<mlir::cir::GlobalOp>(go);
    if (gv) {
      if (d->hasAttr<RetainAttr>())
        assert(!MissingFeatures::addUsedGlobal());
      if (auto *sa = d->getAttr<PragmaClangBSSSectionAttr>())
        assert(!MissingFeatures::addSectionAttributes());
      if (auto *sa = d->getAttr<PragmaClangDataSectionAttr>())
        assert(!MissingFeatures::addSectionAttributes());
      if (auto *sa = d->getAttr<PragmaClangRodataSectionAttr>())
        assert(!MissingFeatures::addSectionAttributes());
      if (auto *sa = d->getAttr<PragmaClangRelroSectionAttr>())
        assert(!MissingFeatures::addSectionAttributes());
    }
    auto f = llvm::dyn_cast_or_null<mlir::cir::FuncOp>(go);
    if (f) {
      if (d->hasAttr<RetainAttr>())
        assert(!MissingFeatures::addUsedGlobal());
      if (auto *sa = d->getAttr<PragmaClangTextSectionAttr>())
        if (!d->getAttr<SectionAttr>())
          assert(!MissingFeatures::setSectionForFuncOp());

      assert(!MissingFeatures::updateCPUAndFeaturesAttributes());
    }

    if (const auto *csa = d->getAttr<CodeSegAttr>()) {
      assert(!MissingFeatures::setSectionForFuncOp());
      if (gv)
        gv.setSection(csa->getName());
      if (f)
        assert(!MissingFeatures::setSectionForFuncOp());
    } else if (const auto *sa = d->getAttr<SectionAttr>())
      if (gv)
        gv.setSection(sa->getName());
    if (f)
      assert(!MissingFeatures::setSectionForFuncOp());
  }
  assert(!MissingFeatures::setTargetAttributes());
}

void CIRGenModule::replaceGlobal(mlir::cir::GlobalOp old,
                                 mlir::cir::GlobalOp New) {
  assert(old.getSymName() == New.getSymName() && "symbol names must match");

  // If the types does not match, update all references to Old to the new type.
  auto oldTy = old.getSymType();
  auto newTy = New.getSymType();
  mlir::cir::AddressSpaceAttr oldAS = old.getAddrSpaceAttr();
  mlir::cir::AddressSpaceAttr newAS = New.getAddrSpaceAttr();
  // TODO(cir): If the AS differs, we should also update all references.
  if (oldAS != newAS) {
    llvm_unreachable("NYI");
  }
  if (oldTy != newTy) {
    auto oldSymUses = old.getSymbolUses(theModule.getOperation());
    if (oldSymUses.has_value()) {
      for (auto use : *oldSymUses) {
        auto *userOp = use.getUser();
        assert((isa<mlir::cir::GetGlobalOp>(userOp) ||
                isa<mlir::cir::GlobalOp>(userOp)) &&
               "GlobalOp symbol user is neither a GetGlobalOp nor a GlobalOp");

        if (auto ggo = dyn_cast<mlir::cir::GetGlobalOp>(use.getUser())) {
          auto useOpResultValue = ggo.getAddr();
          useOpResultValue.setType(
              mlir::cir::PointerType::get(builder.getContext(), newTy));
        }
      }
    }
  }

  // Remove old global from the module.
  old.erase();
}

mlir::cir::TLS_Model CIRGenModule::GetDefaultCIRTLSModel() const {
  switch (getCodeGenOpts().getDefaultTLSModel()) {
  case CodeGenOptions::GeneralDynamicTLSModel:
    return mlir::cir::TLS_Model::GeneralDynamic;
  case CodeGenOptions::LocalDynamicTLSModel:
    return mlir::cir::TLS_Model::LocalDynamic;
  case CodeGenOptions::InitialExecTLSModel:
    return mlir::cir::TLS_Model::InitialExec;
  case CodeGenOptions::LocalExecTLSModel:
    return mlir::cir::TLS_Model::LocalExec;
  }
  llvm_unreachable("Invalid TLS model!");
}

void CIRGenModule::setTLSMode(mlir::Operation *op, const VarDecl &d) const {
  assert(d.getTLSKind() && "setting TLS mode on non-TLS var!");

  auto tlm = GetDefaultCIRTLSModel();

  // Override the TLS model if it is explicitly specified.
  if (const TLSModelAttr *attr = d.getAttr<TLSModelAttr>()) {
    llvm_unreachable("NYI");
  }

  auto global = dyn_cast<mlir::cir::GlobalOp>(op);
  assert(global && "NYI for other operations");
  global.setTlsModel(tlm);
}

/// If the specified mangled name is not in the module,
/// create and return an mlir GlobalOp with the specified type (TODO(cir):
/// address space).
///
/// TODO(cir):
/// 1. If there is something in the module with the specified name, return
/// it potentially bitcasted to the right type.
///
/// 2. If D is non-null, it specifies a decl that correspond to this.  This is
/// used to set the attributes on the global when it is first created.
///
/// 3. If IsForDefinition is true, it is guaranteed that an actual global with
/// type Ty will be returned, not conversion of a variable with the same
/// mangled name but some other type.
mlir::cir::GlobalOp
CIRGenModule::getOrCreateCIRGlobal(StringRef mangledName, mlir::Type ty,
                                   LangAS langAS, const VarDecl *d,
                                   ForDefinition_t isForDefinition) {
  // Lookup the entry, lazily creating it if necessary.
  mlir::cir::GlobalOp entry;
  if (auto *v = getGlobalValue(mangledName)) {
    assert(isa<mlir::cir::GlobalOp>(v) && "only supports GlobalOp for now");
    entry = dyn_cast_or_null<mlir::cir::GlobalOp>(v);
  }

  mlir::cir::AddressSpaceAttr cirAS = builder.getAddrSpaceAttr(langAS);
  if (entry) {
    auto entryCIRAS = entry.getAddrSpaceAttr();
    if (WeakRefReferences.erase(entry)) {
      if (d && !d->hasAttr<WeakAttr>()) {
        auto lt = mlir::cir::GlobalLinkageKind::ExternalLinkage;
        entry.setLinkageAttr(
            mlir::cir::GlobalLinkageKindAttr::get(builder.getContext(), lt));
        mlir::SymbolTable::setSymbolVisibility(entry, getMLIRVisibility(entry));
      }
    }

    // Handle dropped DLL attributes.
    if (d && !d->hasAttr<clang::DLLImportAttr>() &&
        !d->hasAttr<clang::DLLExportAttr>())
      assert(!MissingFeatures::setDLLStorageClass() && "NYI");

    if (langOpts.OpenMP && !langOpts.OpenMPSimd && d)
      getOpenMPRuntime().registerTargetGlobalVariable(d, entry);

    if (entry.getSymType() == ty && entryCIRAS == cirAS)
      return entry;

    // If there are two attempts to define the same mangled name, issue an
    // error.
    //
    // TODO(cir): look at mlir::GlobalValue::isDeclaration for all aspects of
    // recognizing the global as a declaration, for now only check if
    // initializer is present.
    if (isForDefinition && !entry.isDeclaration()) {
      GlobalDecl otherGd;
      const VarDecl *otherD;

      // Check that D is not yet in DiagnosedConflictingDefinitions is required
      // to make sure that we issue an error only once.
      if (d && lookupRepresentativeDecl(mangledName, otherGd) &&
          (d->getCanonicalDecl() != otherGd.getCanonicalDecl().getDecl()) &&
          (otherD = dyn_cast<VarDecl>(otherGd.getDecl())) &&
          otherD->hasInit() &&
          DiagnosedConflictingDefinitions.insert(d).second) {
        getDiags().Report(d->getLocation(), diag::err_duplicate_mangled_name)
            << mangledName;
        getDiags().Report(otherGd.getDecl()->getLocation(),
                          diag::note_previous_definition);
      }
    }

    // TODO(cir): LLVM codegen makes sure the result is of the correct type
    // by issuing a address space cast.
    if (entryCIRAS != cirAS)
      llvm_unreachable("NYI");

    // (If global is requested for a definition, we always need to create a new
    // global, not just return a bitcast.)
    if (!isForDefinition)
      return entry;
  }

  auto declCIRAS = builder.getAddrSpaceAttr(getGlobalVarAddressSpace(d));
  // TODO(cir): do we need to strip pointer casts for Entry?

  auto loc = getLoc(d->getSourceRange());

  // mlir::SymbolTable::Visibility::Public is the default, no need to explicitly
  // mark it as such.
  auto gv = CIRGenModule::createGlobalOp(*this, loc, mangledName, ty,
                                         /*isConstant=*/false,
                                         /*addrSpace=*/declCIRAS,
                                         /*insertPoint=*/entry.getOperation());

  // If we already created a global with the same mangled name (but different
  // type) before, replace it with the new global.
  if (entry) {
    replaceGlobal(entry, gv);
  }

  // This is the first use or definition of a mangled name.  If there is a
  // deferred decl with this name, remember that we need to emit it at the end
  // of the file.
  auto ddi = DeferredDecls.find(mangledName);
  if (ddi != DeferredDecls.end()) {
    // Move the potentially referenced deferred decl to the DeferredDeclsToEmit
    // list, and remove it from DeferredDecls (since we don't need it anymore).
    addDeferredDeclToEmit(ddi->second);
    DeferredDecls.erase(ddi);
  }

  // Handle things which are present even on external declarations.
  if (d) {
    if (langOpts.OpenMP && !langOpts.OpenMPSimd && d)
      getOpenMPRuntime().registerTargetGlobalVariable(d, entry);

    // FIXME: This code is overly simple and should be merged with other global
    // handling.
    gv.setAlignmentAttr(getSize(astCtx.getDeclAlign(d)));
    gv.setConstant(isTypeConstant(d->getType(), false, false));
    // TODO(cir): setLinkageForGV(GV, D);

    if (d->getTLSKind()) {
      if (d->getTLSKind() == VarDecl::TLS_Dynamic)
        llvm_unreachable("NYI");
      setTLSMode(gv, *d);
    }

    setGVProperties(gv, d);

    // If required by the ABI, treat declarations of static data members with
    // inline initializers as definitions.
    if (astCtx.isMSStaticDataMemberInlineDefinition(d)) {
      assert(0 && "not implemented");
    }

    // Emit section information for extern variables.
    if (d->hasExternalStorage()) {
      if (const SectionAttr *sa = d->getAttr<SectionAttr>())
        gv.setSectionAttr(builder.getStringAttr(sa->getName()));
    }

    gv.setGlobalVisibilityAttr(getGlobalVisibilityAttrFromDecl(d));

    // Handle XCore specific ABI requirements.
    if (getTriple().getArch() == llvm::Triple::xcore)
      assert(0 && "not implemented");

    // Check if we a have a const declaration with an initializer, we maybe
    // able to emit it as available_externally to expose it's value to the
    // optimizer.
    if (getLangOpts().CPlusPlus && gv.isPublic() &&
        d->getType().isConstQualified() && gv.isDeclaration() &&
        !d->hasDefinition() && d->hasInit() && !d->hasAttr<DLLImportAttr>()) {
      assert(0 && "not implemented");
    }
  }

  // TODO(cir): if this method is used to handle functions we must have
  // something closer to GlobalValue::isDeclaration instead of checking for
  // initializer.
  if (gv.isDeclaration()) {
    // TODO(cir): set target attributes

    // External HIP managed variables needed to be recorded for transformation
    // in both device and host compilations.
    if (getLangOpts().CUDA)
      assert(0 && "not implemented");
  }

  // TODO(cir): address space cast when needed for DAddrSpace.
  return gv;
}

mlir::cir::GlobalOp
CIRGenModule::getOrCreateCIRGlobal(const VarDecl *d, mlir::Type ty,
                                   ForDefinition_t isForDefinition) {
  assert(d->hasGlobalStorage() && "Not a global variable");
  QualType astTy = d->getType();
  if (!ty)
    ty = getTypes().convertTypeForMem(astTy);

  StringRef mangledName = getMangledName(d);
  return getOrCreateCIRGlobal(mangledName, ty, astTy.getAddressSpace(), d,
                              isForDefinition);
}

/// Return the mlir::Value for the address of the given global variable. If Ty
/// is non-null and if the global doesn't exist, then it will be created with
/// the specified type instead of whatever the normal requested type would be.
/// If IsForDefinition is true, it is guaranteed that an actual global with type
/// Ty will be returned, not conversion of a variable with the same mangled name
/// but some other type.
mlir::Value CIRGenModule::getAddrOfGlobalVar(const VarDecl *d, mlir::Type ty,
                                             ForDefinition_t isForDefinition) {
  assert(d->hasGlobalStorage() && "Not a global variable");
  QualType astTy = d->getType();
  if (!ty)
    ty = getTypes().convertTypeForMem(astTy);

  bool tlsAccess = d->getTLSKind() != VarDecl::TLS_None;
  auto g = getOrCreateCIRGlobal(d, ty, isForDefinition);
  auto ptrTy = builder.getPointerTo(g.getSymType(), g.getAddrSpaceAttr());
  return builder.create<mlir::cir::GetGlobalOp>(
      getLoc(d->getSourceRange()), ptrTy, g.getSymName(), tlsAccess);
}

mlir::cir::GlobalViewAttr
CIRGenModule::getAddrOfGlobalVarAttr(const VarDecl *d, mlir::Type ty,
                                     ForDefinition_t isForDefinition) {
  assert(d->hasGlobalStorage() && "Not a global variable");
  QualType astTy = d->getType();
  if (!ty)
    ty = getTypes().convertTypeForMem(astTy);

  auto globalOp = getOrCreateCIRGlobal(d, ty, isForDefinition);
  return builder.getGlobalViewAttr(builder.getPointerTo(ty), globalOp);
}

mlir::Operation *CIRGenModule::getWeakRefReference(const ValueDecl *vd) {
  const AliasAttr *aa = vd->getAttr<AliasAttr>();
  assert(aa && "No alias?");

  // See if there is already something with the target's name in the module.
  mlir::Operation *entry = getGlobalValue(aa->getAliasee());
  if (entry) {
    assert((isa<mlir::cir::GlobalOp>(entry) || isa<mlir::cir::FuncOp>(entry)) &&
           "weak ref should be against a global variable or function");
    return entry;
  }

  mlir::Type declTy = getTypes().convertTypeForMem(vd->getType());
  if (mlir::isa<mlir::cir::FuncType>(declTy)) {
    auto f = GetOrCreateCIRFunction(aa->getAliasee(), declTy,
                                    GlobalDecl(cast<FunctionDecl>(vd)),
                                    /*ForVtable=*/false);
    f.setLinkage(mlir::cir::GlobalLinkageKind::ExternalWeakLinkage);
    WeakRefReferences.insert(f);
    return f;
  }

  llvm_unreachable("GlobalOp NYI");
}

/// TODO(cir): looks like part of this code can be part of a common AST
/// helper betweem CIR and LLVM codegen.
template <typename SomeDecl>
void CIRGenModule::maybeHandleStaticInExternC(const SomeDecl *d,
                                              mlir::cir::GlobalOp gv) {
  if (!getLangOpts().CPlusPlus)
    return;

  // Must have 'used' attribute, or else inline assembly can't rely on
  // the name existing.
  if (!d->template hasAttr<UsedAttr>())
    return;

  // Must have internal linkage and an ordinary name.
  if (!d->getIdentifier() || d->getFormalLinkage() != Linkage::Internal)
    return;

  // Must be in an extern "C" context. Entities declared directly within
  // a record are not extern "C" even if the record is in such a context.
  const SomeDecl *first = d->getFirstDecl();
  if (first->getDeclContext()->isRecord() || !first->isInExternCContext())
    return;

  // TODO(cir):
  // OK, this is an internal linkage entity inside an extern "C" linkage
  // specification. Make a note of that so we can give it the "expected"
  // mangled name if nothing else is using that name.
  //
  // If we have multiple internal linkage entities with the same name
  // in extern "C" regions, none of them gets that name.
  assert(0 && "not implemented");
}

void CIRGenModule::buildGlobalVarDefinition(const clang::VarDecl *d,
                                            bool isTentative) {
  // TODO(cir):
  // OpenCL global variables of sampler type are translated to function calls,
  // therefore no need to be translated.
  // If this is OpenMP device, check if it is legal to emit this global
  // normally.
  QualType astTy = d->getType();
  if ((getLangOpts().OpenCL && astTy->isSamplerT()) ||
      getLangOpts().OpenMPIsTargetDevice)
    llvm_unreachable("not implemented");

  // TODO(cir): LLVM's codegen uses a llvm::TrackingVH here. Is that
  // necessary here for CIR gen?
  mlir::Attribute init;
  bool needsGlobalCtor = false;
  // Whether the definition of the variable is available externally.
  // If yes, we shouldn't emit the GloablCtor and GlobalDtor for the variable
  // since this is the job for its original source.
  bool isDefinitionAvailableExternally =
      astCtx.GetGVALinkageForVariable(d) == GVA_AvailableExternally;
  bool needsGlobalDtor =
      !isDefinitionAvailableExternally &&
      d->needsDestruction(astCtx) == QualType::DK_cxx_destructor;

  // It is helpless to emit the definition for an available_externally variable
  // which can't be marked as const.
  // We don't need to check if it needs global ctor or dtor. See the above
  // comment for ideas.
  if (isDefinitionAvailableExternally &&
      (!d->hasConstantInitialization() ||
       // TODO: Update this when we have interface to check constexpr
       // destructor.
       d->needsDestruction(getASTContext()) ||
       !d->getType().isConstantStorage(getASTContext(), true, true)))
    return;

  const VarDecl *initDecl;
  const Expr *initExpr = d->getAnyInitializer(initDecl);

  std::optional<ConstantEmitter> emitter;

  // CUDA E.2.4.1 "__shared__ variables cannot have an initialization
  // as part of their declaration."  Sema has already checked for
  // error cases, so we just need to set Init to UndefValue.
  bool isCudaSharedVar =
      getLangOpts().CUDAIsDevice && d->hasAttr<CUDASharedAttr>();
  // Shadows of initialized device-side global variables are also left
  // undefined.
  // Managed Variables should be initialized on both host side and device side.
  bool isCudaShadowVar =
      !getLangOpts().CUDAIsDevice && !d->hasAttr<HIPManagedAttr>() &&
      (d->hasAttr<CUDAConstantAttr>() || d->hasAttr<CUDADeviceAttr>() ||
       d->hasAttr<CUDASharedAttr>());
  bool isCudaDeviceShadowVar =
      getLangOpts().CUDAIsDevice && !d->hasAttr<HIPManagedAttr>() &&
      (d->getType()->isCUDADeviceBuiltinSurfaceType() ||
       d->getType()->isCUDADeviceBuiltinTextureType());
  if (getLangOpts().CUDA &&
      (isCudaSharedVar || isCudaShadowVar || isCudaDeviceShadowVar))
    assert(0 && "not implemented");
  else if (d->hasAttr<LoaderUninitializedAttr>())
    assert(0 && "not implemented");
  else if (!initExpr) {
    // This is a tentative definition; tentative definitions are
    // implicitly initialized with { 0 }.
    //
    // Note that tentative definitions are only emitted at the end of
    // a translation unit, so they should never have incomplete
    // type. In addition, EmitTentativeDefinition makes sure that we
    // never attempt to emit a tentative definition if a real one
    // exists. A use may still exists, however, so we still may need
    // to do a RAUW.
    assert(!astTy->isIncompleteType() && "Unexpected incomplete type");
    init = builder.getZeroInitAttr(getCIRType(d->getType()));
  } else {
    initializedGlobalDecl = GlobalDecl(d);
    emitter.emplace(*this);
    auto initializer = emitter->tryEmitForInitializer(*initDecl);
    if (!initializer) {
      QualType t = initExpr->getType();
      if (d->getType()->isReferenceType())
        t = d->getType();

      if (getLangOpts().CPlusPlus) {
        if (initDecl->hasFlexibleArrayInit(astCtx))
          ErrorUnsupported(d, "flexible array initializer");
        init = builder.getZeroInitAttr(getCIRType(t));
        if (!isDefinitionAvailableExternally)
          needsGlobalCtor = true;
      } else {
        ErrorUnsupported(d, "static initializer");
      }
    } else {
      init = initializer;
      // We don't need an initializer, so remove the entry for the delayed
      // initializer position (just in case this entry was delayed) if we
      // also don't need to register a destructor.
      if (getLangOpts().CPlusPlus && !needsGlobalDtor)
        DelayedCXXInitPosition.erase(d);
    }
  }

  mlir::Type initType;
  // If the initializer attribute is a SymbolRefAttr it means we are
  // initializing the global based on a global constant.
  //
  // TODO(cir): create another attribute to contain the final type and abstract
  // away SymbolRefAttr.
  if (auto symAttr = mlir::dyn_cast<mlir::SymbolRefAttr>(init)) {
    auto *cstGlobal = mlir::SymbolTable::lookupSymbolIn(theModule, symAttr);
    assert(isa<mlir::cir::GlobalOp>(cstGlobal) &&
           "unaware of other symbol providers");
    auto g = cast<mlir::cir::GlobalOp>(cstGlobal);
    auto arrayTy = mlir::dyn_cast<mlir::cir::ArrayType>(g.getSymType());
    // TODO(cir): pointer to array decay. Should this be modeled explicitly in
    // CIR?
    if (arrayTy)
      initType = mlir::cir::PointerType::get(builder.getContext(),
                                             arrayTy.getEltType());
  } else {
    assert(mlir::isa<mlir::TypedAttr>(init) && "This should have a type");
    auto typedInitAttr = mlir::cast<mlir::TypedAttr>(init);
    initType = typedInitAttr.getType();
  }
  assert(!mlir::isa<mlir::NoneType>(initType) && "Should have a type by now");

  auto entry = getOrCreateCIRGlobal(d, initType, ForDefinition_t(!isTentative));
  // TODO(cir): Strip off pointer casts from Entry if we get them?

  // TODO(cir): use GlobalValue interface
  assert(dyn_cast<GlobalOp>(&entry) && "FuncOp not supported here");
  auto gv = entry;

  // We have a definition after a declaration with the wrong type.
  // We must make a new GlobalVariable* and update everything that used OldGV
  // (a declaration or tentative definition) with the new GlobalVariable*
  // (which will be a definition).
  //
  // This happens if there is a prototype for a global (e.g.
  // "extern int x[];") and then a definition of a different type (e.g.
  // "int x[10];"). This also happens when an initializer has a different type
  // from the type of the global (this happens with unions).
  if (!gv || gv.getSymType() != initType) {
    // TODO(cir): this should include an address space check as well.
    assert(0 && "not implemented");
  }

  maybeHandleStaticInExternC(d, gv);

  if (d->hasAttr<AnnotateAttr>())
    addGlobalAnnotations(d, gv);

  // Set CIR's linkage type as appropriate.
  mlir::cir::GlobalLinkageKind linkage =
      getCIRLinkageVarDefinition(d, /*IsConstant=*/false);

  // TODO(cir):
  // CUDA B.2.1 "The __device__ qualifier declares a variable that resides on
  // the device. [...]"
  // CUDA B.2.2 "The __constant__ qualifier, optionally used together with
  // __device__, declares a variable that: [...]
  if (gv && getLangOpts().CUDA) {
    assert(0 && "not implemented");
  }

  // Set initializer and finalize emission
  CIRGenModule::setInitializer(gv, init);
  if (emitter)
    emitter->finalize(gv);

  // TODO(cir): If it is safe to mark the global 'constant', do so now.
  gv.setConstant(!needsGlobalCtor && !needsGlobalDtor &&
                 isTypeConstant(d->getType(), true, true));

  // If it is in a read-only section, mark it 'constant'.
  if (const SectionAttr *sa = d->getAttr<SectionAttr>())
    gv.setSectionAttr(builder.getStringAttr(sa->getName()));

  gv.setGlobalVisibilityAttr(getGlobalVisibilityAttrFromDecl(d));

  // TODO(cir):
  // GV->setAlignment(getContext().getDeclAlign(D).getAsAlign());

  // On Darwin, unlike other Itanium C++ ABI platforms, the thread-wrapper
  // function is only defined alongside the variable, not also alongside
  // callers. Normally, all accesses to a thread_local go through the
  // thread-wrapper in order to ensure initialization has occurred, underlying
  // variable will never be used other than the thread-wrapper, so it can be
  // converted to internal linkage.
  //
  // However, if the variable has the 'constinit' attribute, it _can_ be
  // referenced directly, without calling the thread-wrapper, so the linkage
  // must not be changed.
  //
  // Additionally, if the variable isn't plain external linkage, e.g. if it's
  // weak or linkonce, the de-duplication semantics are important to preserve,
  // so we don't change the linkage.
  if (d->getTLSKind() == VarDecl::TLS_Dynamic && gv.isPublic() &&
      astCtx.getTargetInfo().getTriple().isOSDarwin() &&
      !d->hasAttr<ConstInitAttr>()) {
    // TODO(cir): set to mlir::SymbolTable::Visibility::Private once we have
    // testcases.
    assert(0 && "not implemented");
  }

  // Set CIR linkage and DLL storage class.
  gv.setLinkage(linkage);
  // FIXME(cir): setLinkage should likely set MLIR's visibility automatically.
  gv.setVisibility(getMLIRVisibilityFromCIRLinkage(linkage));
  // TODO(cir): handle DLL storage classes in CIR?
  if (d->hasAttr<DLLImportAttr>())
    assert(!MissingFeatures::setDLLStorageClass());
  else if (d->hasAttr<DLLExportAttr>())
    assert(!MissingFeatures::setDLLStorageClass());
  else
    assert(!MissingFeatures::setDLLStorageClass());

  if (linkage == mlir::cir::GlobalLinkageKind::CommonLinkage) {
    // common vars aren't constant even if declared const.
    gv.setConstant(false);
    // Tentative definition of global variables may be initialized with
    // non-zero null pointers. In this case they should have weak linkage
    // since common linkage must have zero initializer and must not have
    // explicit section therefore cannot have non-zero initial value.
    auto initializer = gv.getInitialValue();
    if (initializer && !getBuilder().isNullValue(*initializer))
      gv.setLinkage(mlir::cir::GlobalLinkageKind::WeakAnyLinkage);
  }

  setNonAliasAttributes(d, gv);

  if (d->getTLSKind() && !gv.getTlsModelAttr()) {
    if (d->getTLSKind() == VarDecl::TLS_Dynamic)
      llvm_unreachable("NYI");
    setTLSMode(gv, *d);
  }

  maybeSetTrivialComdat(*d, gv);

  // TODO(cir):
  // Emit the initializer function if necessary.
  if (needsGlobalCtor || needsGlobalDtor) {
    globalOpContext = gv;
    buildCXXGlobalVarDeclInitFunc(d, gv, needsGlobalCtor);
    globalOpContext = nullptr;
  }

  // TODO(cir): sanitizers (reportGlobalToASan) and global variable debug
  // information.
  assert(!MissingFeatures::sanitizeOther());
  assert(!MissingFeatures::generateDebugInfo());
}

void CIRGenModule::buildGlobalDefinition(GlobalDecl gd, mlir::Operation *op) {
  const auto *d = cast<ValueDecl>(gd.getDecl());
  if (const auto *fd = dyn_cast<FunctionDecl>(d)) {
    // At -O0, don't generate CIR for functions with available_externally
    // linkage.
    if (!shouldEmitFunction(gd))
      return;

    if (const auto *method = dyn_cast<CXXMethodDecl>(d)) {
      // Make sure to emit the definition(s) before we emit the thunks. This is
      // necessary for the generation of certain thunks.
      if (isa<CXXConstructorDecl>(method) || isa<CXXDestructorDecl>(method))
        ABI->buildCXXStructor(gd);
      else if (fd->isMultiVersion())
        llvm_unreachable("NYI");
      else
        buildGlobalFunctionDefinition(gd, op);

      if (method->isVirtual())
        getVTables().buildThunks(gd);

      return;
    }

    if (fd->isMultiVersion())
      llvm_unreachable("NYI");
    buildGlobalFunctionDefinition(gd, op);
    return;
  }

  if (const auto *vd = dyn_cast<VarDecl>(d)) {
    return buildGlobalVarDefinition(vd, !vd->hasDefinition());
  }

  llvm_unreachable("Invalid argument to buildGlobalDefinition()");
}

mlir::Attribute
CIRGenModule::getConstantArrayFromStringLiteral(const StringLiteral *e) {
  assert(!e->getType()->isPointerType() && "Strings are always arrays");

  // Don't emit it as the address of the string, emit the string data itself
  // as an inline array.
  if (e->getCharByteWidth() == 1) {
    SmallString<64> str(e->getString());

    // Resize the string to the right size, which is indicated by its type.
    const ConstantArrayType *cat = astCtx.getAsConstantArrayType(e->getType());
    auto finalSize = cat->getSize().getZExtValue();
    str.resize(finalSize);

    auto eltTy = getTypes().ConvertType(cat->getElementType());
    return builder.getString(str, eltTy, finalSize);
  }

  auto arrayTy = mlir::dyn_cast<mlir::cir::ArrayType>(
      getTypes().ConvertType(e->getType()));
  assert(arrayTy && "string literals must be emitted as an array type");

  auto arrayEltTy = mlir::dyn_cast<mlir::cir::IntType>(arrayTy.getEltType());
  assert(arrayEltTy &&
         "string literal elements must be emitted as integral type");

  auto arraySize = arrayTy.getSize();
  auto literalSize = e->getLength();

  // Collect the code units.
  SmallVector<uint32_t, 32> elementValues;
  elementValues.reserve(arraySize);
  for (unsigned i = 0; i < literalSize; ++i)
    elementValues.push_back(e->getCodeUnit(i));
  elementValues.resize(arraySize);

  // If the string is full of null bytes, emit a #cir.zero instead.
  if (std::all_of(elementValues.begin(), elementValues.end(),
                  [](uint32_t x) { return x == 0; }))
    return builder.getZeroAttr(arrayTy);

  // Otherwise emit a constant array holding the characters.
  SmallVector<mlir::Attribute, 32> elements;
  elements.reserve(arraySize);
  for (uint64_t i = 0; i < arraySize; ++i)
    elements.push_back(mlir::cir::IntAttr::get(arrayEltTy, elementValues[i]));

  auto elementsAttr = mlir::ArrayAttr::get(builder.getContext(), elements);
  return builder.getConstArray(elementsAttr, arrayTy);
}

// TODO(cir): this could be a common AST helper for both CIR and LLVM codegen.
LangAS CIRGenModule::getGlobalConstantAddressSpace() const {
  // OpenCL v1.2 s6.5.3: a string literal is in the constant address space.
  if (getLangOpts().OpenCL)
    return LangAS::opencl_constant;
  if (getLangOpts().SYCLIsDevice)
    return LangAS::sycl_global;
  if (auto as = getTarget().getConstantAddressSpace())
    return as.value();
  return LangAS::Default;
}

// TODO(cir): this could be a common AST helper for both CIR and LLVM codegen.
LangAS CIRGenModule::getLangTempAllocaAddressSpace() const {
  if (getLangOpts().OpenCL)
    return LangAS::opencl_private;
  if (getLangOpts().SYCLIsDevice || getLangOpts().CUDAIsDevice ||
      (getLangOpts().OpenMP && getLangOpts().OpenMPIsTargetDevice))
    llvm_unreachable("NYI");
  return LangAS::Default;
}

static mlir::cir::GlobalOp
generateStringLiteral(mlir::Location loc, mlir::TypedAttr c,
                      mlir::cir::GlobalLinkageKind lt, CIRGenModule &cgm,
                      StringRef globalName, CharUnits alignment) {
  unsigned addrSpace = cgm.getASTContext().getTargetAddressSpace(
      cgm.getGlobalConstantAddressSpace());
  assert((addrSpace == 0 && !cir::MissingFeatures::addressSpaceInGlobalVar()) &&
         "NYI");

  // Create a global variable for this string
  // FIXME(cir): check for insertion point in module level.
  auto gv = CIRGenModule::createGlobalOp(cgm, loc, globalName, c.getType(),
                                         !cgm.getLangOpts().WritableStrings);

  // Set up extra information and add to the module
  gv.setAlignmentAttr(cgm.getSize(alignment));
  gv.setLinkageAttr(
      mlir::cir::GlobalLinkageKindAttr::get(cgm.getBuilder().getContext(), lt));
  CIRGenModule::setInitializer(gv, c);
  // TODO(cir)
  assert(!cir::MissingFeatures::threadLocal() && "NYI");
  assert(!cir::MissingFeatures::unnamedAddr() && "NYI");
  if (gv.isWeakForLinker()) {
    assert(cgm.supportsCOMDAT() && "Only COFF uses weak string literals");
    gv.setComdat(true);
  }
  cgm.setDSOLocal(static_cast<mlir::Operation *>(gv));
  return gv;
}

/// Return a pointer to a constant array for the given string literal.
mlir::cir::GlobalViewAttr
CIRGenModule::getAddrOfConstantStringFromLiteral(const StringLiteral *s,
                                                 StringRef name) {
  CharUnits alignment =
      astCtx.getAlignOfGlobalVarInChars(s->getType(), /*VD=*/nullptr);

  mlir::Attribute c = getConstantArrayFromStringLiteral(s);

  mlir::cir::GlobalOp gv;
  if (!getLangOpts().WritableStrings && ConstantStringMap.count(c)) {
    gv = ConstantStringMap[c];
    // The bigger alignment always wins.
    if (!gv.getAlignment() ||
        uint64_t(alignment.getQuantity()) > *gv.getAlignment())
      gv.setAlignmentAttr(getSize(alignment));
  } else {
    SmallString<256> stringNameBuffer = name;
    llvm::raw_svector_ostream out(stringNameBuffer);
    if (StringLiteralCnt)
      out << StringLiteralCnt;
    name = out.str();
    StringLiteralCnt++;

    SmallString<256> mangledNameBuffer;
    StringRef globalVariableName;
    auto lt = mlir::cir::GlobalLinkageKind::ExternalLinkage;

    // Mangle the string literal if that's how the ABI merges duplicate strings.
    // Don't do it if they are writable, since we don't want writes in one TU to
    // affect strings in another.
    if (getCXXABI().getMangleContext().shouldMangleStringLiteral(s) &&
        !getLangOpts().WritableStrings) {
      assert(0 && "not implemented");
    } else {
      lt = mlir::cir::GlobalLinkageKind::InternalLinkage;
      globalVariableName = name;
    }

    auto loc = getLoc(s->getSourceRange());
    auto typedC = llvm::dyn_cast<mlir::TypedAttr>(c);
    if (!typedC)
      llvm_unreachable("this should never be untyped at this point");
    gv = generateStringLiteral(loc, typedC, lt, *this, globalVariableName,
                               alignment);
    setDSOLocal(static_cast<mlir::Operation *>(gv));
    ConstantStringMap[c] = gv;

    assert(!cir::MissingFeatures::reportGlobalToASan() && "NYI");
  }

  auto arrayTy = mlir::dyn_cast<mlir::cir::ArrayType>(gv.getSymType());
  assert(arrayTy && "String literal must be array");
  auto ptrTy =
      mlir::cir::PointerType::get(builder.getContext(), arrayTy.getEltType());

  return builder.getGlobalViewAttr(ptrTy, gv);
}

void CIRGenModule::buildDeclContext(const DeclContext *dc) {
  for (auto *i : dc->decls()) {
    // Unlike other DeclContexts, the contents of an ObjCImplDecl at TU scope
    // are themselves considered "top-level", so EmitTopLevelDecl on an
    // ObjCImplDecl does not recursively visit them. We need to do that in
    // case they're nested inside another construct (LinkageSpecDecl /
    // ExportDecl) that does stop them from being considered "top-level".
    if (auto *oid = dyn_cast<ObjCImplDecl>(i))
      llvm_unreachable("NYI");

    buildTopLevelDecl(i);
  }
}

void CIRGenModule::buildLinkageSpec(const LinkageSpecDecl *lsd) {
  if (lsd->getLanguage() != LinkageSpecLanguageIDs::C &&
      lsd->getLanguage() != LinkageSpecLanguageIDs::CXX) {
    llvm_unreachable("unsupported linkage spec");
    return;
  }
  buildDeclContext(lsd);
}

mlir::Operation *
CIRGenModule::getAddrOfGlobalTemporary(const MaterializeTemporaryExpr *expr,
                                       const Expr *init) {
  assert((expr->getStorageDuration() == SD_Static ||
          expr->getStorageDuration() == SD_Thread) &&
         "not a global temporary");
  const auto *varDecl = cast<VarDecl>(expr->getExtendingDecl());

  // If we're not materializing a subobject of the temporay, keep the
  // cv-qualifiers from the type of the MaterializeTemporaryExpr.
  QualType materializedType = init->getType();
  if (init == expr->getSubExpr())
    materializedType = expr->getType();

  [[maybe_unused]] CharUnits align =
      getASTContext().getTypeAlignInChars(materializedType);

  auto insertResult = materializedGlobalTemporaryMap.insert({expr, nullptr});
  if (!insertResult.second) {
    llvm_unreachable("NYI");
  }

  // FIXME: If an externally-visible declaration extends multiple temporaries,
  // we need to give each temporary the same name in every translation unit (and
  // we also need to make the temporaries externally-visible).
  llvm::SmallString<256> name;
  llvm::raw_svector_ostream out(name);
  getCXXABI().getMangleContext().mangleReferenceTemporary(
      varDecl, expr->getManglingNumber(), out);

  APValue *value = nullptr;
  if (expr->getStorageDuration() == SD_Static && varDecl->evaluateValue()) {
    // If the initializer of the extending declaration is a constant
    // initializer, we should have a cached constant initializer for this
    // temporay. Note taht this m ight have a different value from the value
    // computed by evaluating the initializer if the surrounding constant
    // expression modifies the temporary.
    value = expr->getOrCreateValue(false);
  }

  // Try evaluating it now, it might have a constant initializer
  Expr::EvalResult evalResult;
  if (!value && init->EvaluateAsRValue(evalResult, getASTContext()) &&
      !evalResult.hasSideEffects())
    value = &evalResult.Val;

  LangAS addrSpace = getGlobalVarAddressSpace(varDecl);

  std::optional<ConstantEmitter> emitter;
  mlir::Attribute initialValue = nullptr;
  bool isConstant = false;
  mlir::Type type;
  if (value) {
    emitter.emplace(*this);
    initialValue =
        emitter->emitForInitializer(*value, addrSpace, materializedType);

    isConstant = materializedType.isConstantStorage(
        getASTContext(), /*ExcludeCtor*/ value, /*ExcludeDtor*/ false);

    type = mlir::cast<mlir::TypedAttr>(initialValue).getType();
  } else {
    // No initializer, the initialization will be provided when we initialize
    // the declaration which performed lifetime extension.
    llvm_unreachable("else value");
  }

  // Create a global variable for this lifetime-extended temporary.
  mlir::cir::GlobalLinkageKind linkage =
      getCIRLinkageVarDefinition(varDecl, false);
  if (linkage == mlir::cir::GlobalLinkageKind::ExternalLinkage) {
    const VarDecl *initVD;
    if (varDecl->isStaticDataMember() && varDecl->getAnyInitializer(initVD) &&
        isa<CXXRecordDecl>(initVD->getLexicalDeclContext())) {
      // Temporaries defined inside a class get linkonce_odr linkage because the
      // calss can be defined in multiple translation units.
      llvm_unreachable("staticdatamember NYI");
    } else {
      // There is no need for this temporary to have external linkage if the
      // VarDecl has external linkage.
      linkage = mlir::cir::GlobalLinkageKind::InternalLinkage;
    }
  }
  auto targetAS = builder.getAddrSpaceAttr(addrSpace);

  auto loc = getLoc(expr->getSourceRange());
  auto gv = createGlobalOp(*this, loc, name, type, isConstant, targetAS,
                           nullptr, linkage);
  gv.setInitialValueAttr(initialValue);

  if (emitter)
    emitter->finalize(gv);
  // Don't assign dllimport or dllexport to lcoal linkage globals
  if (!gv.hasLocalLinkage()) {
    llvm_unreachable("NYI");
  }
  gv.setAlignment(align.getAsAlign().value());
  if (supportsCOMDAT() && gv.isWeakForLinker())
    llvm_unreachable("NYI");
  if (varDecl->getTLSKind())
    llvm_unreachable("NYI");
  mlir::Operation *cv = gv;
  if (addrSpace != LangAS::Default)
    llvm_unreachable("NYI");

  // Update the map with the new temporay. If we created a placeholder above,
  // replace it with the new global now.
  mlir::Operation *&entry = materializedGlobalTemporaryMap[expr];
  if (entry) {
    entry->replaceAllUsesWith(cv);
    entry->erase();
  }
  entry = cv;

  return cv;
}

// Emit code for a single top level declaration.
void CIRGenModule::buildTopLevelDecl(Decl *decl) {
  // Ignore dependent declarations
  if (decl->isTemplated())
    return;

  // Consteval function shouldn't be emitted.
  if (auto *fd = dyn_cast<FunctionDecl>(decl))
    if (fd->isConsteval())
      return;

  switch (decl->getKind()) {
  default:
    llvm::errs() << "buildTopLevelDecl codegen for decl kind '"
                 << decl->getDeclKindName() << "' not implemented\n";
    assert(false && "Not yet implemented");

  case Decl::TranslationUnit: {
    // This path is CIR only - CIRGen handles TUDecls because
    // of clang-tidy checks, that operate on TU granularity.
    TranslationUnitDecl *tu = cast<TranslationUnitDecl>(decl);
    for (DeclContext::decl_iterator d = tu->decls_begin(),
                                    dEnd = tu->decls_end();
         d != dEnd; ++d)
      buildTopLevelDecl(*d);
    return;
  }
  case Decl::Var:
  case Decl::Decomposition:
  case Decl::VarTemplateSpecialization:
    buildGlobal(cast<VarDecl>(decl));
    assert(!isa<DecompositionDecl>(decl) && "not implemented");
    // if (auto *DD = dyn_cast<DecompositionDecl>(decl))
    //   for (auto *B : DD->bindings())
    //     if (auto *HD = B->getHoldingVar())
    //       EmitGlobal(HD);
    break;

  case Decl::CXXConversion:
  case Decl::CXXMethod:
  case Decl::Function:
    buildGlobal(cast<FunctionDecl>(decl));
    assert(!codeGenOpts.CoverageMapping && "Coverage Mapping NYI");
    break;
  // C++ Decls
  case Decl::Namespace:
    buildDeclContext(cast<NamespaceDecl>(decl));
    break;
  case Decl::ClassTemplateSpecialization: {
    // const auto *Spec = cast<ClassTemplateSpecializationDecl>(decl);
    assert(!MissingFeatures::generateDebugInfo() && "NYI");
  }
    [[fallthrough]];
  case Decl::CXXRecord: {
    CXXRecordDecl *crd = cast<CXXRecordDecl>(decl);
    // TODO: Handle debug info as CodeGenModule.cpp does
    for (auto *childDecl : crd->decls())
      if (isa<VarDecl>(childDecl) || isa<CXXRecordDecl>(childDecl))
        buildTopLevelDecl(childDecl);
    break;
  }
  // No code generation needed.
  case Decl::UsingShadow:
  case Decl::ClassTemplate:
  case Decl::VarTemplate:
  case Decl::Concept:
  case Decl::VarTemplatePartialSpecialization:
  case Decl::FunctionTemplate:
  case Decl::TypeAliasTemplate:
  case Decl::Block:
  case Decl::Empty:
  case Decl::Binding:
    break;
  case Decl::Using:     // using X; [C++]
  case Decl::UsingEnum: // using enum X; [C++]
  case Decl::NamespaceAlias:
  case Decl::UsingDirective: // using namespace X; [C++]
    assert(!MissingFeatures::generateDebugInfo() && "NYI");
    break;
  case Decl::CXXConstructor:
    getCXXABI().buildCXXConstructors(cast<CXXConstructorDecl>(decl));
    break;
  case Decl::CXXDestructor:
    getCXXABI().buildCXXDestructors(cast<CXXDestructorDecl>(decl));
    break;

  case Decl::StaticAssert:
    // Nothing to do.
    break;

  case Decl::LinkageSpec:
    buildLinkageSpec(cast<LinkageSpecDecl>(decl));
    break;

  case Decl::Typedef:
  case Decl::TypeAlias: // using foo = bar; [C++11]
  case Decl::Record:
  case Decl::Enum:
    assert(!MissingFeatures::generateDebugInfo() && "NYI");
    break;
  }
}

static bool shouldBeInCOMDAT(CIRGenModule &cgm, const Decl &d) {
  if (!cgm.supportsCOMDAT())
    return false;

  if (d.hasAttr<SelectAnyAttr>())
    return true;

  GVALinkage linkage;
  if (auto *vd = dyn_cast<VarDecl>(&d))
    linkage = cgm.getASTContext().GetGVALinkageForVariable(vd);
  else
    linkage =
        cgm.getASTContext().GetGVALinkageForFunction(cast<FunctionDecl>(&d));

  switch (linkage) {
  case clang::GVA_Internal:
  case clang::GVA_AvailableExternally:
  case clang::GVA_StrongExternal:
    return false;
  case clang::GVA_DiscardableODR:
  case clang::GVA_StrongODR:
    return true;
  }
  llvm_unreachable("No such linkage");
}

// TODO(cir): this could be a common method between LLVM codegen.
static bool isVarDeclStrongDefinition(const ASTContext &context,
                                      CIRGenModule &cgm, const VarDecl *d,
                                      bool noCommon) {
  // Don't give variables common linkage if -fno-common was specified unless it
  // was overridden by a NoCommon attribute.
  if ((noCommon || d->hasAttr<NoCommonAttr>()) && !d->hasAttr<CommonAttr>())
    return true;

  // C11 6.9.2/2:
  //   A declaration of an identifier for an object that has file scope without
  //   an initializer, and without a storage-class specifier or with the
  //   storage-class specifier static, constitutes a tentative definition.
  if (d->getInit() || d->hasExternalStorage())
    return true;

  // A variable cannot be both common and exist in a section.
  if (d->hasAttr<SectionAttr>())
    return true;

  // A variable cannot be both common and exist in a section.
  // We don't try to determine which is the right section in the front-end.
  // If no specialized section name is applicable, it will resort to default.
  if (d->hasAttr<PragmaClangBSSSectionAttr>() ||
      d->hasAttr<PragmaClangDataSectionAttr>() ||
      d->hasAttr<PragmaClangRelroSectionAttr>() ||
      d->hasAttr<PragmaClangRodataSectionAttr>())
    return true;

  // Thread local vars aren't considered common linkage.
  if (d->getTLSKind())
    return true;

  // Tentative definitions marked with WeakImportAttr are true definitions.
  if (d->hasAttr<WeakImportAttr>())
    return true;

  // A variable cannot be both common and exist in a comdat.
  if (shouldBeInCOMDAT(cgm, *d))
    return true;

  // Declarations with a required alignment do not have common linkage in MSVC
  // mode.
  if (context.getTargetInfo().getCXXABI().isMicrosoft()) {
    if (d->hasAttr<AlignedAttr>())
      return true;
    QualType varType = d->getType();
    if (context.isAlignmentRequired(varType))
      return true;

    if (const auto *rt = varType->getAs<RecordType>()) {
      const RecordDecl *rd = rt->getDecl();
      for (const FieldDecl *fd : rd->fields()) {
        if (fd->isBitField())
          continue;
        if (fd->hasAttr<AlignedAttr>())
          return true;
        if (context.isAlignmentRequired(fd->getType()))
          return true;
      }
    }
  }

  // Microsoft's link.exe doesn't support alignments greater than 32 bytes for
  // common symbols, so symbols with greater alignment requirements cannot be
  // common.
  // Other COFF linkers (ld.bfd and LLD) support arbitrary power-of-two
  // alignments for common symbols via the aligncomm directive, so this
  // restriction only applies to MSVC environments.
  if (context.getTargetInfo().getTriple().isKnownWindowsMSVCEnvironment() &&
      context.getTypeAlignIfKnown(d->getType()) >
          context.toBits(CharUnits::fromQuantity(32)))
    return true;

  return false;
}

void CIRGenModule::setInitializer(mlir::cir::GlobalOp &global,
                                  mlir::Attribute value) {
  // Recompute visibility when updating initializer.
  global.setInitialValueAttr(value);
  mlir::SymbolTable::setSymbolVisibility(
      global, CIRGenModule::getMLIRVisibility(global));
}

mlir::SymbolTable::Visibility
CIRGenModule::getMLIRVisibility(mlir::cir::GlobalOp op) {
  // MLIR doesn't accept public symbols declarations (only
  // definitions).
  if (op.isDeclaration())
    return mlir::SymbolTable::Visibility::Private;
  return getMLIRVisibilityFromCIRLinkage(op.getLinkage());
}

mlir::SymbolTable::Visibility CIRGenModule::getMLIRVisibilityFromCIRLinkage(
    mlir::cir::GlobalLinkageKind glk) {
  switch (glk) {
  case mlir::cir::GlobalLinkageKind::InternalLinkage:
  case mlir::cir::GlobalLinkageKind::PrivateLinkage:
    return mlir::SymbolTable::Visibility::Private;
  case mlir::cir::GlobalLinkageKind::ExternalLinkage:
  case mlir::cir::GlobalLinkageKind::ExternalWeakLinkage:
  case mlir::cir::GlobalLinkageKind::LinkOnceODRLinkage:
  case mlir::cir::GlobalLinkageKind::AvailableExternallyLinkage:
  case mlir::cir::GlobalLinkageKind::CommonLinkage:
  case mlir::cir::GlobalLinkageKind::WeakAnyLinkage:
  case mlir::cir::GlobalLinkageKind::WeakODRLinkage:
    return mlir::SymbolTable::Visibility::Public;
  default: {
    llvm::errs() << "visibility not implemented for '"
                 << stringifyGlobalLinkageKind(glk) << "'\n";
    assert(0 && "not implemented");
  }
  }
  llvm_unreachable("linkage should be handled above!");
}

mlir::cir::VisibilityKind
CIRGenModule::getGlobalVisibilityKindFromClangVisibility(
    clang::VisibilityAttr::VisibilityType visibility) {
  switch (visibility) {
  case clang::VisibilityAttr::VisibilityType::Default:
    return VisibilityKind::Default;
  case clang::VisibilityAttr::VisibilityType::Hidden:
    return VisibilityKind::Hidden;
  case clang::VisibilityAttr::VisibilityType::Protected:
    return VisibilityKind::Protected;
  }
}

mlir::cir::VisibilityAttr
CIRGenModule::getGlobalVisibilityAttrFromDecl(const Decl *decl) {
  const clang::VisibilityAttr *va = decl->getAttr<clang::VisibilityAttr>();
  mlir::cir::VisibilityAttr cirVisibility =
      mlir::cir::VisibilityAttr::get(builder.getContext());
  if (va) {
    cirVisibility = mlir::cir::VisibilityAttr::get(
        builder.getContext(),
        getGlobalVisibilityKindFromClangVisibility(va->getVisibility()));
  }
  return cirVisibility;
}

mlir::cir::GlobalLinkageKind CIRGenModule::getCIRLinkageForDeclarator(
    const DeclaratorDecl *d, GVALinkage linkage, bool isConstantVariable) {
  if (linkage == GVA_Internal)
    return mlir::cir::GlobalLinkageKind::InternalLinkage;

  if (d->hasAttr<WeakAttr>()) {
    if (isConstantVariable)
      return mlir::cir::GlobalLinkageKind::WeakODRLinkage;
    return mlir::cir::GlobalLinkageKind::WeakAnyLinkage;
  }

  if (const auto *fd = d->getAsFunction())
    if (fd->isMultiVersion() && linkage == GVA_AvailableExternally)
      return mlir::cir::GlobalLinkageKind::LinkOnceAnyLinkage;

  // We are guaranteed to have a strong definition somewhere else,
  // so we can use available_externally linkage.
  if (linkage == GVA_AvailableExternally)
    return mlir::cir::GlobalLinkageKind::AvailableExternallyLinkage;

  // Note that Apple's kernel linker doesn't support symbol
  // coalescing, so we need to avoid linkonce and weak linkages there.
  // Normally, this means we just map to internal, but for explicit
  // instantiations we'll map to external.

  // In C++, the compiler has to emit a definition in every translation unit
  // that references the function.  We should use linkonce_odr because
  // a) if all references in this translation unit are optimized away, we
  // don't need to codegen it.  b) if the function persists, it needs to be
  // merged with other definitions. c) C++ has the ODR, so we know the
  // definition is dependable.
  if (linkage == GVA_DiscardableODR)
    return !astCtx.getLangOpts().AppleKext
               ? mlir::cir::GlobalLinkageKind::LinkOnceODRLinkage
               : mlir::cir::GlobalLinkageKind::InternalLinkage;

  // An explicit instantiation of a template has weak linkage, since
  // explicit instantiations can occur in multiple translation units
  // and must all be equivalent. However, we are not allowed to
  // throw away these explicit instantiations.
  //
  // CUDA/HIP: For -fno-gpu-rdc case, device code is limited to one TU,
  // so say that CUDA templates are either external (for kernels) or internal.
  // This lets llvm perform aggressive inter-procedural optimizations. For
  // -fgpu-rdc case, device function calls across multiple TU's are allowed,
  // therefore we need to follow the normal linkage paradigm.
  if (linkage == GVA_StrongODR) {
    if (getLangOpts().AppleKext)
      return mlir::cir::GlobalLinkageKind::ExternalLinkage;
    if (getLangOpts().CUDA && getLangOpts().CUDAIsDevice &&
        !getLangOpts().GPURelocatableDeviceCode)
      return d->hasAttr<CUDAGlobalAttr>()
                 ? mlir::cir::GlobalLinkageKind::ExternalLinkage
                 : mlir::cir::GlobalLinkageKind::InternalLinkage;
    return mlir::cir::GlobalLinkageKind::WeakODRLinkage;
  }

  // C++ doesn't have tentative definitions and thus cannot have common
  // linkage.
  if (!getLangOpts().CPlusPlus && isa<VarDecl>(d) &&
      !isVarDeclStrongDefinition(astCtx, *this, cast<VarDecl>(d),
                                 getCodeGenOpts().NoCommon))
    return mlir::cir::GlobalLinkageKind::CommonLinkage;

  // selectany symbols are externally visible, so use weak instead of
  // linkonce.  MSVC optimizes away references to const selectany globals, so
  // all definitions should be the same and ODR linkage should be used.
  // http://msdn.microsoft.com/en-us/library/5tkz6s71.aspx
  if (d->hasAttr<SelectAnyAttr>())
    return mlir::cir::GlobalLinkageKind::WeakODRLinkage;

  // Otherwise, we have strong external linkage.
  assert(linkage == GVA_StrongExternal);
  return mlir::cir::GlobalLinkageKind::ExternalLinkage;
}

/// This function is called when we implement a function with no prototype, e.g.
/// "int foo() {}". If there are existing call uses of the old function in the
/// module, this adjusts them to call the new function directly.
///
/// This is not just a cleanup: the always_inline pass requires direct calls to
/// functions to be able to inline them.  If there is a bitcast in the way, it
/// won't inline them. Instcombine normally deletes these calls, but it isn't
/// run at -O0.
void CIRGenModule::ReplaceUsesOfNonProtoTypeWithRealFunction(
    mlir::Operation *old, mlir::cir::FuncOp newFn) {

  // If we're redefining a global as a function, don't transform it.
  auto oldFn = dyn_cast<mlir::cir::FuncOp>(old);
  if (!oldFn)
    return;

  // TODO(cir): this RAUW ignores the features below.
  assert(!MissingFeatures::exceptions() && "Call vs Invoke NYI");
  assert(!MissingFeatures::parameterAttributes());
  assert(!MissingFeatures::operandBundles());
  assert(oldFn->getAttrs().size() > 1 && "Attribute forwarding NYI");

  // Mark new function as originated from a no-proto declaration.
  newFn.setNoProtoAttr(oldFn.getNoProtoAttr());

  // Iterate through all calls of the no-proto function.
  auto symUses = oldFn.getSymbolUses(oldFn->getParentOp());
  for (auto use : symUses.value()) {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto noProtoCallOp = dyn_cast<mlir::cir::CallOp>(use.getUser())) {
      builder.setInsertionPoint(noProtoCallOp);

      // Patch call type with the real function type.
      auto realCallOp = builder.createCallOp(noProtoCallOp.getLoc(), newFn,
                                             noProtoCallOp.getOperands());

      // Replace old no proto call with fixed call.
      noProtoCallOp.replaceAllUsesWith(realCallOp);
      noProtoCallOp.erase();
    } else if (auto getGlobalOp =
                   dyn_cast<mlir::cir::GetGlobalOp>(use.getUser())) {
      // Replace type
      getGlobalOp.getAddr().setType(mlir::cir::PointerType::get(
          builder.getContext(), newFn.getFunctionType()));
    } else {
      llvm_unreachable("NIY");
    }
  }
}

mlir::cir::GlobalLinkageKind
CIRGenModule::getCIRLinkageVarDefinition(const VarDecl *vd, bool isConstant) {
  assert(!isConstant && "constant variables NYI");
  GVALinkage linkage = astCtx.GetGVALinkageForVariable(vd);
  return getCIRLinkageForDeclarator(vd, linkage, isConstant);
}

mlir::cir::GlobalLinkageKind CIRGenModule::getFunctionLinkage(GlobalDecl gd) {
  const auto *d = cast<FunctionDecl>(gd.getDecl());

  GVALinkage linkage = astCtx.GetGVALinkageForFunction(d);

  if (const auto *dtor = dyn_cast<CXXDestructorDecl>(d))
    return getCXXABI().getCXXDestructorLinkage(linkage, dtor, gd.getDtorType());

  if (isa<CXXConstructorDecl>(d) &&
      cast<CXXConstructorDecl>(d)->isInheritingConstructor() &&
      astCtx.getTargetInfo().getCXXABI().isMicrosoft()) {
    // Just like in LLVM codegen:
    // Our approach to inheriting constructors is fundamentally different from
    // that used by the MS ABI, so keep our inheriting constructor thunks
    // internal rather than trying to pick an unambiguous mangling for them.
    return mlir::cir::GlobalLinkageKind::InternalLinkage;
  }

  return getCIRLinkageForDeclarator(d, linkage, /*IsConstantVariable=*/false);
}

void CIRGenModule::buildAliasForGlobal(StringRef mangledName,
                                       mlir::Operation *op, GlobalDecl aliasGD,
                                       mlir::cir::FuncOp aliasee,
                                       mlir::cir::GlobalLinkageKind linkage) {
  auto *aliasFD = dyn_cast<FunctionDecl>(aliasGD.getDecl());
  assert(aliasFD && "expected FunctionDecl");
  auto alias =
      createCIRFunction(getLoc(aliasGD.getDecl()->getSourceRange()),
                        mangledName, aliasee.getFunctionType(), aliasFD);
  alias.setAliasee(aliasee.getName());
  alias.setLinkage(linkage);
  mlir::SymbolTable::setSymbolVisibility(
      alias, getMLIRVisibilityFromCIRLinkage(linkage));

  // Alias constructors and destructors are always unnamed_addr.
  assert(!MissingFeatures::unnamedAddr());

  // Switch any previous uses to the alias.
  if (op) {
    llvm_unreachable("NYI");
  } else {
    // Name already set by createCIRFunction
  }

  // Finally, set up the alias with its proper name and attributes.
  setCommonAttributes(aliasGD, alias);
}

mlir::Type CIRGenModule::getCIRType(const QualType &type) {
  return genTypes.ConvertType(type);
}

bool CIRGenModule::verifyModule() {
  // Verify the module after we have finished constructing it, this will
  // check the structural properties of the IR and invoke any specific
  // verifiers we have on the CIR operations.
  return mlir::verify(theModule).succeeded();
}

std::pair<mlir::cir::FuncType, mlir::cir::FuncOp>
CIRGenModule::getAddrAndTypeOfCXXStructor(GlobalDecl gd,
                                          const CIRGenFunctionInfo *fnInfo,
                                          mlir::cir::FuncType fnType,
                                          bool dontdefer,
                                          ForDefinition_t isForDefinition) {
  auto *md = cast<CXXMethodDecl>(gd.getDecl());

  if (isa<CXXDestructorDecl>(md)) {
    // Always alias equivalent complete destructors to base destructors in the
    // MS ABI.
    if (getTarget().getCXXABI().isMicrosoft() &&
        gd.getDtorType() == Dtor_Complete &&
        md->getParent()->getNumVBases() == 0)
      llvm_unreachable("NYI");
  }

  if (!fnType) {
    if (!fnInfo)
      fnInfo = &getTypes().arrangeCXXStructorDeclaration(gd);
    fnType = getTypes().GetFunctionType(*fnInfo);
  }

  auto fn = GetOrCreateCIRFunction(getMangledName(gd), fnType, gd,
                                   /*ForVtable=*/false, dontdefer,
                                   /*IsThunk=*/false, isForDefinition);

  return {fnType, fn};
}

mlir::cir::FuncOp
CIRGenModule::GetAddrOfFunction(clang::GlobalDecl gd, mlir::Type ty,
                                bool forVTable, bool dontDefer,
                                ForDefinition_t isForDefinition) {
  assert(!cast<FunctionDecl>(gd.getDecl())->isConsteval() &&
         "consteval function should never be emitted");

  if (!ty) {
    const auto *fd = cast<FunctionDecl>(gd.getDecl());
    ty = getTypes().ConvertType(fd->getType());
  }

  // Devirtualized destructor calls may come through here instead of via
  // getAddrOfCXXStructor. Make sure we use the MS ABI base destructor instead
  // of the complete destructor when necessary.
  if (const auto *dd = dyn_cast<CXXDestructorDecl>(gd.getDecl())) {
    if (getTarget().getCXXABI().isMicrosoft() &&
        gd.getDtorType() == Dtor_Complete &&
        dd->getParent()->getNumVBases() == 0)
      llvm_unreachable("NYI");
  }

  StringRef mangledName = getMangledName(gd);
  auto f = GetOrCreateCIRFunction(mangledName, ty, gd, forVTable, dontDefer,
                                  /*IsThunk=*/false, isForDefinition);

  assert(!langOpts.CUDA && "NYI");

  return f;
}

// Returns true if GD is a function decl with internal linkage and needs a
// unique suffix after the mangled name.
static bool isUniqueInternalLinkageDecl(GlobalDecl gd, CIRGenModule &cgm) {
  assert(cgm.getModuleNameHash().empty() &&
         "Unique internal linkage names NYI");

  return false;
}

static std::string getMangledNameImpl(CIRGenModule &cgm, GlobalDecl gd,
                                      const NamedDecl *nd,
                                      bool omitMultiVersionMangling = false) {
  assert(!omitMultiVersionMangling && "NYI");

  SmallString<256> buffer;

  llvm::raw_svector_ostream out(buffer);
  MangleContext &mc = cgm.getCXXABI().getMangleContext();

  assert(cgm.getModuleNameHash().empty() && "NYI");
  auto shouldMangle = mc.shouldMangleDeclName(nd);

  if (shouldMangle) {
    mc.mangleName(gd.getWithDecl(nd), out);
  } else {
    auto *ii = nd->getIdentifier();
    assert(ii && "Attempt to mangle unnamed decl.");

    const auto *fd = dyn_cast<FunctionDecl>(nd);

    if (fd &&
        fd->getType()->castAs<FunctionType>()->getCallConv() == CC_X86RegCall) {
      assert(0 && "NYI");
    } else if (fd && fd->hasAttr<CUDAGlobalAttr>() &&
               gd.getKernelReferenceKind() == KernelReferenceKind::Stub) {
      assert(0 && "NYI");
    } else {
      out << ii->getName();
    }
  }

  // Check if the module name hash should be appended for internal linkage
  // symbols. This should come before multi-version target suffixes are
  // appendded. This is to keep the name and module hash suffix of the internal
  // linkage function together. The unique suffix should only be added when name
  // mangling is done to make sure that the final name can be properly
  // demangled. For example, for C functions without prototypes, name mangling
  // is not done and the unique suffix should not be appended then.
  assert(!isUniqueInternalLinkageDecl(gd, cgm) && "NYI");

  if (const auto *fd = dyn_cast<FunctionDecl>(nd)) {
    assert(!fd->isMultiVersion() && "NYI");
  }
  assert(!cgm.getLangOpts().GPURelocatableDeviceCode && "NYI");

  return std::string(out.str());
}

StringRef CIRGenModule::getMangledName(GlobalDecl gd) {
  auto canonicalGd = gd.getCanonicalDecl();

  // Some ABIs don't have constructor variants. Make sure that base and complete
  // constructors get mangled the same.
  if (const auto *cd = dyn_cast<CXXConstructorDecl>(canonicalGd.getDecl())) {
    if (!getTarget().getCXXABI().hasConstructorVariants()) {
      assert(false && "NYI");
    }
  }

  assert(!langOpts.CUDAIsDevice && "NYI");

  // Keep the first result in the case of a mangling collision.
  const auto *nd = cast<NamedDecl>(gd.getDecl());
  std::string mangledName = getMangledNameImpl(*this, gd, nd);

  auto result = Manglings.insert(std::make_pair(mangledName, gd));
  return MangledDeclNames[canonicalGd] = result.first->first();
}

void CIRGenModule::buildTentativeDefinition(const VarDecl *d) {
  assert(!d->getInit() && "Cannot emit definite definitions here!");

  StringRef mangledName = getMangledName(d);
  auto *gv = getGlobalValue(mangledName);

  // TODO(cir): can a tentative definition come from something other than a
  // global op? If not, the assertion below is wrong and should be removed. If
  // so, getGlobalValue might be better of returining a global value interface
  // that alows use to manage different globals value types transparently.
  if (gv)
    assert(isa<mlir::cir::GlobalOp>(gv) &&
           "tentative definition can only be built from a cir.global_op");

  // We already have a definition, not declaration, with the same mangled name.
  // Emitting of declaration is not required (and actually overwrites emitted
  // definition).
  if (gv && !dyn_cast<mlir::cir::GlobalOp>(gv).isDeclaration())
    return;

  // If we have not seen a reference to this variable yet, place it into the
  // deferred declarations table to be emitted if needed later.
  if (!MustBeEmitted(d) && !gv) {
    DeferredDecls[mangledName] = d;
    return;
  }

  // The tentative definition is the only definition.
  buildGlobalVarDefinition(d);
}

void CIRGenModule::setGlobalVisibility(mlir::Operation *gv,
                                       const NamedDecl *d) const {
  assert(!MissingFeatures::setGlobalVisibility());
}

void CIRGenModule::setDSOLocal(mlir::Operation *op) const {
  assert(!MissingFeatures::setDSOLocal());
  if (auto globalValue = dyn_cast<mlir::cir::CIRGlobalValueInterface>(op)) {
    setDSOLocal(globalValue);
  }
}

void CIRGenModule::setGVProperties(mlir::Operation *op,
                                   const NamedDecl *d) const {
  assert(!MissingFeatures::setDLLImportDLLExport());
  setGVPropertiesAux(op, d);
}

void CIRGenModule::setGVPropertiesAux(mlir::Operation *op,
                                      const NamedDecl *d) const {
  setGlobalVisibility(op, d);
  setDSOLocal(op);
  assert(!MissingFeatures::setPartition());
}

bool CIRGenModule::lookupRepresentativeDecl(StringRef mangledName,
                                            GlobalDecl &result) const {
  auto res = Manglings.find(mangledName);
  if (res == Manglings.end())
    return false;
  result = res->getValue();
  return true;
}

mlir::cir::FuncOp
CIRGenModule::createCIRFunction(mlir::Location loc, StringRef name,
                                mlir::cir::FuncType ty,
                                const clang::FunctionDecl *fd) {
  // At the point we need to create the function, the insertion point
  // could be anywhere (e.g. callsite). Do not rely on whatever it might
  // be, properly save, find the appropriate place and restore.
  FuncOp f;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Some global emissions are triggered while emitting a function, e.g.
    // void s() { x.method() }
    //
    // Be sure to insert a new function before a current one.
    auto *curCGF = getCurrCIRGenFun();
    if (curCGF)
      builder.setInsertionPoint(curCGF->CurFn);

    f = builder.create<mlir::cir::FuncOp>(loc, name, ty);

    if (fd)
      f.setAstAttr(makeFuncDeclAttr(fd, builder.getContext()));

    if (fd && !fd->hasPrototype())
      f.setNoProtoAttr(builder.getUnitAttr());

    assert(f.isDeclaration() && "expected empty body");

    // A declaration gets private visibility by default, but external linkage
    // as the default linkage.
    f.setLinkageAttr(mlir::cir::GlobalLinkageKindAttr::get(
        builder.getContext(), mlir::cir::GlobalLinkageKind::ExternalLinkage));
    mlir::SymbolTable::setSymbolVisibility(
        f, mlir::SymbolTable::Visibility::Private);

    // Initialize with empty dict of extra attributes.
    f.setExtraAttrsAttr(mlir::cir::ExtraFuncAttributesAttr::get(
        builder.getContext(), builder.getDictionaryAttr({})));

    if (!curCGF)
      theModule.push_back(f);
  }
  return f;
}

mlir::cir::FuncOp CIRGenModule::createRuntimeFunction(
    mlir::cir::FuncType ty, StringRef name, mlir::ArrayAttr,
    [[maybe_unused]] bool local, bool assumeConvergent) {
  if (assumeConvergent) {
    llvm_unreachable("NYI");
  }

  auto entry = GetOrCreateCIRFunction(name, ty, GlobalDecl(),
                                      /*ForVtable=*/false);

  // Traditional codegen checks for a valid dyn_cast llvm::Function for `entry`,
  // no testcase that cover this path just yet though.
  if (!entry) {
    // Setup runtime CC, DLL support for windows and set dso local.
    llvm_unreachable("NYI");
  }

  return entry;
}

bool isDefaultedMethod(const clang::FunctionDecl *FD) {
  return FD->isDefaulted() && isa<CXXMethodDecl>(FD) &&
         (cast<CXXMethodDecl>(FD)->isCopyAssignmentOperator() ||
          cast<CXXMethodDecl>(FD)->isMoveAssignmentOperator());
}

mlir::Location CIRGenModule::getLocForFunction(const clang::FunctionDecl *fd) {
  bool invalidLoc = !fd || (fd->getSourceRange().getBegin().isInvalid() ||
                            fd->getSourceRange().getEnd().isInvalid());
  if (!invalidLoc)
    return getLoc(fd->getSourceRange());

  // Use the module location
  return theModule->getLoc();
}

/// Determines whether the language options require us to model
/// unwind exceptions.  We treat -fexceptions as mandating this
/// except under the fragile ObjC ABI with only ObjC exceptions
/// enabled.  This means, for example, that C with -fexceptions
/// enables this.
/// TODO(cir): can be shared with traditional LLVM codegen.
static bool hasUnwindExceptions(const LangOptions &langOpts) {
  // If exceptions are completely disabled, obviously this is false.
  if (!langOpts.Exceptions)
    return false;

  // If C++ exceptions are enabled, this is true.
  if (langOpts.CXXExceptions)
    return true;

  // If ObjC exceptions are enabled, this depends on the ABI.
  if (langOpts.ObjCExceptions) {
    return langOpts.ObjCRuntime.hasUnwindExceptions();
  }

  return true;
}

void CIRGenModule::setCIRFunctionAttributesForDefinition(const Decl *decl,
                                                         FuncOp f) {
  mlir::NamedAttrList attrs{f.getExtraAttrs().getElements().getValue()};

  if (!hasUnwindExceptions(getLangOpts())) {
    auto attr = mlir::cir::NoThrowAttr::get(builder.getContext());
    attrs.set(attr.getMnemonic(), attr);
  }

  if (!decl) {
    // If we don't have a declaration to control inlining, the function isn't
    // explicitly marked as alwaysinline for semantic reasons, and inlining is
    // disabled, mark the function as noinline.
    if (codeGenOpts.getInlining() == CodeGenOptions::OnlyAlwaysInlining) {
      auto attr = mlir::cir::InlineAttr::get(
          builder.getContext(), mlir::cir::InlineKind::AlwaysInline);
      attrs.set(attr.getMnemonic(), attr);
    }
  } else if (decl->hasAttr<NoInlineAttr>()) {
    // Add noinline if the function isn't always_inline.
    auto attr = mlir::cir::InlineAttr::get(builder.getContext(),
                                           mlir::cir::InlineKind::NoInline);
    attrs.set(attr.getMnemonic(), attr);
  } else if (decl->hasAttr<AlwaysInlineAttr>()) {
    // (noinline wins over always_inline, and we can't specify both in IR)
    auto attr = mlir::cir::InlineAttr::get(builder.getContext(),
                                           mlir::cir::InlineKind::AlwaysInline);
    attrs.set(attr.getMnemonic(), attr);
  } else if (codeGenOpts.getInlining() == CodeGenOptions::OnlyAlwaysInlining) {
    // If we're not inlining, then force everything that isn't always_inline
    // to carry an explicit noinline attribute.
    auto attr = mlir::cir::InlineAttr::get(builder.getContext(),
                                           mlir::cir::InlineKind::NoInline);
    attrs.set(attr.getMnemonic(), attr);
  } else {
    // Otherwise, propagate the inline hint attribute and potentially use its
    // absence to mark things as noinline.
    // Search function and template pattern redeclarations for inline.
    auto checkForInline = [](const FunctionDecl *decl) {
      auto checkRedeclForInline = [](const FunctionDecl *redecl) {
        return redecl->isInlineSpecified();
      };
      if (any_of(decl->redecls(), checkRedeclForInline))
        return true;
      const FunctionDecl *pattern = decl->getTemplateInstantiationPattern();
      if (!pattern)
        return false;
      return any_of(pattern->redecls(), checkRedeclForInline);
    };
    if (checkForInline(cast<FunctionDecl>(decl))) {
      auto attr = mlir::cir::InlineAttr::get(builder.getContext(),
                                             mlir::cir::InlineKind::InlineHint);
      attrs.set(attr.getMnemonic(), attr);
    } else if (codeGenOpts.getInlining() == CodeGenOptions::OnlyHintInlining) {
      auto attr = mlir::cir::InlineAttr::get(builder.getContext(),
                                             mlir::cir::InlineKind::NoInline);
      attrs.set(attr.getMnemonic(), attr);
    }
  }

  // Track whether we need to add the optnone attribute,
  // starting with the default for this optimization level.
  bool shouldAddOptNone =
      !codeGenOpts.DisableO0ImplyOptNone && codeGenOpts.OptimizationLevel == 0;
  if (decl) {
    shouldAddOptNone &= !decl->hasAttr<MinSizeAttr>();
    shouldAddOptNone &= !decl->hasAttr<AlwaysInlineAttr>();
    shouldAddOptNone |= decl->hasAttr<OptimizeNoneAttr>();
  }

  if (shouldAddOptNone) {
    auto optNoneAttr = mlir::cir::OptNoneAttr::get(builder.getContext());
    attrs.set(optNoneAttr.getMnemonic(), optNoneAttr);

    // OptimizeNone implies noinline; we should not be inlining such functions.
    auto noInlineAttr = mlir::cir::InlineAttr::get(
        builder.getContext(), mlir::cir::InlineKind::NoInline);
    attrs.set(noInlineAttr.getMnemonic(), noInlineAttr);
  }

  f.setExtraAttrsAttr(mlir::cir::ExtraFuncAttributesAttr::get(
      builder.getContext(), attrs.getDictionary(builder.getContext())));
}

void CIRGenModule::setCIRFunctionAttributes(GlobalDecl gd,
                                            const CIRGenFunctionInfo &info,
                                            mlir::cir::FuncOp func,
                                            bool isThunk) {
  // TODO(cir): More logic of constructAttributeList is needed.
  mlir::cir::CallingConv callingConv;

  // Initialize PAL with existing attributes to merge attributes.
  mlir::NamedAttrList pal{func.getExtraAttrs().getElements().getValue()};
  constructAttributeList(func.getName(), info, gd, pal, callingConv,
                         /*AttrOnCallSite=*/false, isThunk);
  func.setExtraAttrsAttr(mlir::cir::ExtraFuncAttributesAttr::get(
      builder.getContext(), pal.getDictionary(builder.getContext())));

  // TODO(cir): Check X86_VectorCall incompatibility with WinARM64EC

  func.setCallingConv(callingConv);
}

void CIRGenModule::setFunctionAttributes(GlobalDecl globalDecl,
                                         mlir::cir::FuncOp func,
                                         bool isIncompleteFunction,
                                         bool isThunk) {
  // NOTE(cir): Original CodeGen checks if this is an intrinsic. In CIR we
  // represent them in dedicated ops. The correct attributes are ensured during
  // translation to LLVM. Thus, we don't need to check for them here.

  if (!isIncompleteFunction) {
    setCIRFunctionAttributes(globalDecl,
                             getTypes().arrangeGlobalDeclaration(globalDecl),
                             func, isThunk);
  }

  // TODO(cir): Complete the remaining part of the function.
  assert(!MissingFeatures::setFunctionAttributes());
  const auto *decl = globalDecl.getDecl();
  func.setGlobalVisibilityAttr(getGlobalVisibilityAttrFromDecl(decl));
}

/// If the specified mangled name is not in the module,
/// create and return a CIR Function with the specified type. If there is
/// something in the module with the specified name, return it potentially
/// bitcasted to the right type.
///
/// If D is non-null, it specifies a decl that corresponded to this. This is
/// used to set the attributes on the function when it is first created.
mlir::cir::FuncOp CIRGenModule::GetOrCreateCIRFunction(
    StringRef mangledName, mlir::Type ty, GlobalDecl gd, bool forVTable,
    bool dontDefer, bool isThunk, ForDefinition_t isForDefinition,
    mlir::ArrayAttr extraAttrs) {
  assert(!isThunk && "NYI");

  const auto *d = gd.getDecl();

  // Any attempts to use a MultiVersion function should result in retrieving the
  // iFunc instead. Name mangling will handle the rest of the changes.
  if (const auto *fd = cast_or_null<FunctionDecl>(d)) {
    // For the device mark the function as one that should be emitted.
    if (getLangOpts().OpenMPIsTargetDevice && fd->isDefined() && !dontDefer &&
        !isForDefinition) {
      assert(0 && "OpenMP target functions NYI");
    }
    if (fd->isMultiVersion())
      llvm_unreachable("NYI");
  }

  // Lookup the entry, lazily creating it if necessary.
  mlir::Operation *entry = getGlobalValue(mangledName);
  if (entry) {
    assert(isa<mlir::cir::FuncOp>(entry) &&
           "not implemented, only supports FuncOp for now");

    if (WeakRefReferences.erase(entry)) {
      llvm_unreachable("NYI");
    }

    // Handle dropped DLL attributes.
    if (d && !d->hasAttr<DLLImportAttr>() && !d->hasAttr<DLLExportAttr>()) {
      // TODO(CIR): Entry->setDLLStorageClass
      setDSOLocal(entry);
    }

    // If there are two attempts to define the same mangled name, issue an
    // error.
    auto fn = cast<mlir::cir::FuncOp>(entry);
    if (isForDefinition && fn && !fn.isDeclaration()) {
      GlobalDecl otherGd;
      // CHeck that GD is not yet in DiagnosedConflictingDefinitions is required
      // to make sure that we issue and error only once.
      if (lookupRepresentativeDecl(mangledName, otherGd) &&
          (gd.getCanonicalDecl().getDecl()) &&
          DiagnosedConflictingDefinitions.insert(gd).second) {
        getDiags().Report(d->getLocation(), diag::err_duplicate_mangled_name)
            << mangledName;
        getDiags().Report(otherGd.getDecl()->getLocation(),
                          diag::note_previous_definition);
      }
    }

    if (fn && fn.getFunctionType() == ty) {
      return fn;
    }

    if (!isForDefinition) {
      return fn;
    }

    // TODO: clang checks here if this is a llvm::GlobalAlias... how will we
    // support this?
  }

  // This function doesn't have a complete type (for example, the return type is
  // an incomplete struct). Use a fake type instead, and make sure not to try to
  // set attributes.
  bool isIncompleteFunction = false;

  mlir::cir::FuncType fTy;
  if (mlir::isa<mlir::cir::FuncType>(ty)) {
    fTy = mlir::cast<mlir::cir::FuncType>(ty);
  } else {
    assert(false && "NYI");
    // FTy = mlir::FunctionType::get(VoidTy, false);
    isIncompleteFunction = true;
  }

  auto *fd = llvm::cast_or_null<FunctionDecl>(d);

  // TODO: CodeGen includeds the linkage (ExternalLinkage) and only passes the
  // mangledname if Entry is nullptr
  auto f = createCIRFunction(getLocForFunction(fd), mangledName, fTy, fd);

  // If we already created a function with the same mangled name (but different
  // type) before, take its name and add it to the list of functions to be
  // replaced with F at the end of CodeGen.
  //
  // This happens if there is a prototype for a function (e.g. "int f()") and
  // then a definition of a different type (e.g. "int f(int x)").
  if (entry) {

    // Fetch a generic symbol-defining operation and its uses.
    auto symbolOp = dyn_cast<mlir::SymbolOpInterface>(entry);
    assert(symbolOp && "Expected a symbol-defining operation");

    // TODO(cir): When can this symbol be something other than a function?
    assert(isa<mlir::cir::FuncOp>(entry) && "NYI");

    // This might be an implementation of a function without a prototype, in
    // which case, try to do special replacement of calls which match the new
    // prototype. The really key thing here is that we also potentially drop
    // arguments from the call site so as to make a direct call, which makes the
    // inliner happier and suppresses a number of optimizer warnings (!) about
    // dropping arguments.
    if (symbolOp.getSymbolUses(symbolOp->getParentOp())) {
      ReplaceUsesOfNonProtoTypeWithRealFunction(entry, f);
    }

    // Obliterate no-proto declaration.
    entry->erase();
  }

  if (d)
    setFunctionAttributes(gd, f, isIncompleteFunction, isThunk);
  if (extraAttrs) {
    llvm_unreachable("NYI");
  }

  if (!dontDefer) {
    // All MSVC dtors other than the base dtor are linkonce_odr and delegate to
    // each other bottoming out wiht the base dtor. Therefore we emit non-base
    // dtors on usage, even if there is no dtor definition in the TU.
    if (isa_and_nonnull<CXXDestructorDecl>(d) &&
        getCXXABI().useThunkForDtorVariant(cast<CXXDestructorDecl>(d),
                                           gd.getDtorType())) {
      llvm_unreachable("NYI"); // addDeferredDeclToEmit(GD);
    }

    // This is the first use or definition of a mangled name. If there is a
    // deferred decl with this name, remember that we need to emit it at the end
    // of the file.
    auto ddi = DeferredDecls.find(mangledName);
    if (ddi != DeferredDecls.end()) {
      // Move the potentially referenced deferred decl to the
      // DeferredDeclsToEmit list, and remove it from DeferredDecls (since we
      // don't need it anymore).
      addDeferredDeclToEmit(ddi->second);
      DeferredDecls.erase(ddi);

      // Otherwise, there are cases we have to worry about where we're using a
      // declaration for which we must emit a definition but where we might not
      // find a top-level definition.
      //   - member functions defined inline in their classes
      //   - friend functions defined inline in some class
      //   - special member functions with implicit definitions
      // If we ever change our AST traversal to walk into class methods, this
      // will be unnecessary.
      //
      // We also don't emit a definition for a function if it's going to be an
      // entry in a vtable, unless it's already marked as used.
    } else if (getLangOpts().CPlusPlus && d) {
      // Look for a declaration that's lexically in a record.
      for (const auto *fd = cast<FunctionDecl>(d)->getMostRecentDecl(); fd;
           fd = fd->getPreviousDecl()) {
        if (isa<CXXRecordDecl>(fd->getLexicalDeclContext())) {
          if (fd->doesThisDeclarationHaveABody()) {
            if (isDefaultedMethod(fd))
              addDefaultMethodsToEmit(gd.getWithDecl(fd));
            else
              addDeferredDeclToEmit(gd.getWithDecl(fd));
            break;
          }
        }
      }
    }
  }

  if (!isIncompleteFunction) {
    assert(f.getFunctionType() == ty);
    return f;
  }

  // TODO(cir): Might need bitcast to different address space.
  assert(!MissingFeatures::addressSpace());
  return f;
}

mlir::Location CIRGenModule::getLoc(SourceLocation sLoc) {
  assert(sLoc.isValid() && "expected valid source location");
  const SourceManager &sm = astCtx.getSourceManager();
  PresumedLoc pLoc = sm.getPresumedLoc(sLoc);
  StringRef filename = pLoc.getFilename();
  return mlir::FileLineColLoc::get(builder.getStringAttr(filename),
                                   pLoc.getLine(), pLoc.getColumn());
}

mlir::Location CIRGenModule::getLoc(SourceRange sLoc) {
  assert(sLoc.isValid() && "expected valid source location");
  mlir::Location b = getLoc(sLoc.getBegin());
  mlir::Location e = getLoc(sLoc.getEnd());
  SmallVector<mlir::Location, 2> locs = {b, e};
  mlir::Attribute metadata;
  return mlir::FusedLoc::get(locs, metadata, builder.getContext());
}

mlir::Location CIRGenModule::getLoc(mlir::Location lhs, mlir::Location rhs) {
  SmallVector<mlir::Location, 2> locs = {lhs, rhs};
  mlir::Attribute metadata;
  return mlir::FusedLoc::get(locs, metadata, builder.getContext());
}

void CIRGenModule::buildGlobalDecl(clang::GlobalDecl &d) {
  // We should call GetAddrOfGlobal with IsForDefinition set to true in order
  // to get a Value with exactly the type we need, not something that might
  // have been created for another decl with the same mangled name but
  // different type.
  auto *op = GetAddrOfGlobal(d, ForDefinition);

  // In case of different address spaces, we may still get a cast, even with
  // IsForDefinition equal to true. Query mangled names table to get
  // GlobalValue.
  if (!op) {
    op = getGlobalValue(getMangledName(d));
  }

  // In case of different address spaces, we may still get a cast, even with
  // IsForDefinition equal to true. Query mangled names table to get
  // GlobalValue.
  if (!op)
    llvm_unreachable("Address spaces NYI");

  // Make sure getGlobalValue returned non-null.
  assert(op);

  // Check to see if we've already emitted this. This is necessary for a
  // couple of reasons: first, decls can end up in deferred-decls queue
  // multiple times, and second, decls can end up with definitions in unusual
  // ways (e.g. by an extern inline function acquiring a strong function
  // redefinition). Just ignore those cases.
  // TODO: Not sure what to map this to for MLIR
  auto *globalValueOp = op;
  if (auto gv = dyn_cast<mlir::cir::GetGlobalOp>(op)) {
    auto *result =
        mlir::SymbolTable::lookupSymbolIn(getModule(), gv.getNameAttr());
    globalValueOp = result;
  }

  if (auto cirGlobalValue =
          dyn_cast<mlir::cir::CIRGlobalValueInterface>(globalValueOp)) {
    if (!cirGlobalValue.isDeclaration())
      return;
  }

  // If this is OpenMP, check if it is legal to emit this global normally.
  if (getLangOpts().OpenMP && openMPRuntime &&
      openMPRuntime->emitTargetGlobal(d))
    return;

  // Otherwise, emit the definition and move on to the next one.
  buildGlobalDefinition(d, op);
}

void CIRGenModule::buildDeferred(unsigned recursionLimit) {
  // Emit deferred declare target declarations
  if (getLangOpts().OpenMP && !getLangOpts().OpenMPSimd)
    getOpenMPRuntime().emitDeferredTargetDecls();

  // Emit code for any potentially referenced deferred decls. Since a previously
  // unused static decl may become used during the generation of code for a
  // static function, iterate until no changes are made.

  if (!DeferredVTables.empty()) {
    buildDeferredVTables();

    // Emitting a vtable doesn't directly cause more vtables to
    // become deferred, although it can cause functions to be
    // emitted that then need those vtables.
    assert(DeferredVTables.empty());
  }

  // Emit CUDA/HIP static device variables referenced by host code only. Note we
  // should not clear CUDADeviceVarODRUsedByHost since it is still needed for
  // further handling.
  if (getLangOpts().CUDA && getLangOpts().CUDAIsDevice) {
    llvm_unreachable("NYI");
  }

  // Stop if we're out of both deferred vtables and deferred declarations.
  if (DeferredDeclsToEmit.empty())
    return;

  // Grab the list of decls to emit. If buildGlobalDefinition schedules more
  // work, it will not interfere with this.
  std::vector<GlobalDecl> curDeclsToEmit;
  curDeclsToEmit.swap(DeferredDeclsToEmit);
  if (recursionLimit == 0)
    return;
  recursionLimit--;

  for (auto &d : curDeclsToEmit) {
    if (getCodeGenOpts().ClangIRSkipFunctionsFromSystemHeaders) {
      auto *decl = d.getDecl();
      assert(decl && "expected decl");
      if (astCtx.getSourceManager().isInSystemHeader(decl->getLocation()))
        continue;
    }

    buildGlobalDecl(d);

    // If we found out that we need to emit more decls, do that recursively.
    // This has the advantage that the decls are emitted in a DFS and related
    // ones are close together, which is convenient for testing.
    if (!DeferredVTables.empty() || !DeferredDeclsToEmit.empty()) {
      buildDeferred(recursionLimit);
      assert(DeferredVTables.empty() && DeferredDeclsToEmit.empty());
    }
  }
}

void CIRGenModule::buildDefaultMethods() {
  // Differently from DeferredDeclsToEmit, there's no recurrent use of
  // DefaultMethodsToEmit, so use it directly for emission.
  for (auto &d : DefaultMethodsToEmit)
    buildGlobalDecl(d);
}

mlir::IntegerAttr CIRGenModule::getSize(CharUnits size) {
  return builder.getSizeFromCharUnits(builder.getContext(), size);
}

mlir::Operation *
CIRGenModule::GetAddrOfGlobal(GlobalDecl gd, ForDefinition_t isForDefinition) {
  const Decl *d = gd.getDecl();

  if (isa<CXXConstructorDecl>(d) || isa<CXXDestructorDecl>(d))
    return getAddrOfCXXStructor(gd, /*FnInfo=*/nullptr, /*FnType=*/nullptr,
                                /*DontDefer=*/false, isForDefinition);

  if (isa<CXXMethodDecl>(d)) {
    const auto *fInfo =
        &getTypes().arrangeCXXMethodDeclaration(cast<CXXMethodDecl>(d));
    auto ty = getTypes().GetFunctionType(*fInfo);
    return GetAddrOfFunction(gd, ty, /*ForVTable=*/false, /*DontDefer=*/false,
                             isForDefinition);
  }

  if (isa<FunctionDecl>(d)) {
    const CIRGenFunctionInfo &fi = getTypes().arrangeGlobalDeclaration(gd);
    auto ty = getTypes().GetFunctionType(fi);
    return GetAddrOfFunction(gd, ty, /*ForVTable=*/false, /*DontDefer=*/false,
                             isForDefinition);
  }

  return getAddrOfGlobalVar(cast<VarDecl>(d), /*Ty=*/nullptr, isForDefinition)
      .getDefiningOp();
}

void CIRGenModule::Release() {
  buildDeferred(getCodeGenOpts().ClangIRBuildDeferredThreshold);
  // TODO: buildVTablesOpportunistically();
  // TODO: applyGlobalValReplacements();
  applyReplacements();
  // TODO: checkAliases();
  // TODO: buildMultiVersionFunctions();
  buildCXXGlobalInitFunc();
  // TODO: buildCXXGlobalCleanUpFunc();
  // TODO: registerGlobalDtorsWithAtExit();
  // TODO: buildCXXThreadLocalInitFunc();
  // TODO: ObjCRuntime
  if (astCtx.getLangOpts().CUDA) {
    llvm_unreachable("NYI");
  }
  // TODO: OpenMPRuntime
  // TODO: PGOReader
  // TODO: buildCtorList(GlobalCtors);
  // TODO: builtCtorList(GlobalDtors);
  buildGlobalAnnotations();
  // TODO: buildDeferredUnusedCoverageMappings();
  // TODO: CIRGenPGO
  // TODO: CoverageMapping
  if (getCodeGenOpts().SanitizeCfiCrossDso) {
    llvm_unreachable("NYI");
  }
  // TODO: buildAtAvailableLinkGuard();
  if (astCtx.getTargetInfo().getTriple().isWasm() &&
      !astCtx.getTargetInfo().getTriple().isOSEmscripten()) {
    llvm_unreachable("NYI");
  }

  // Emit reference of __amdgpu_device_library_preserve_asan_functions to
  // preserve ASAN functions in bitcode libraries.
  if (getLangOpts().Sanitize.has(SanitizerKind::Address)) {
    llvm_unreachable("NYI");
  }

  // TODO: buildLLVMUsed();
  // TODO: SanStats

  if (getCodeGenOpts().Autolink) {
    // TODO: buildModuleLinkOptions
  }

  // Emit OpenCL specific module metadata: OpenCL/SPIR version.
  if (langOpts.CUDAIsDevice && getTriple().isSPIRV())
    llvm_unreachable("CUDA SPIR-V NYI");
  if (langOpts.OpenCL) {
    buildOpenCLMetadata();
    // Emit SPIR version.
    if (getTriple().isSPIR())
      llvm_unreachable("SPIR target NYI");
  }

  // TODO: FINISH THE REST OF THIS
}

bool CIRGenModule::shouldEmitFunction(GlobalDecl gd) {
  // TODO: implement this -- requires defining linkage for CIR
  return true;
}

bool CIRGenModule::supportsCOMDAT() const {
  return getTriple().supportsCOMDAT();
}

void CIRGenModule::maybeSetTrivialComdat(const Decl &d, mlir::Operation *op) {
  if (!shouldBeInCOMDAT(*this, d))
    return;
  auto globalOp = dyn_cast_or_null<mlir::cir::GlobalOp>(op);
  if (globalOp)
    globalOp.setComdat(true);
  // Keep it as missing feature as we need to implement comdat for FuncOp.
  // in the future.
  assert(!MissingFeatures::setComdat() && "NYI");
}

bool CIRGenModule::isInNoSanitizeList(SanitizerMask kind, mlir::cir::FuncOp fn,
                                      SourceLocation loc) const {
  const auto &noSanitizeL = getASTContext().getNoSanitizeList();
  // NoSanitize by function name.
  if (noSanitizeL.containsFunction(kind, fn.getName()))
    llvm_unreachable("NYI");
  // NoSanitize by location.
  if (loc.isValid())
    return noSanitizeL.containsLocation(kind, loc);
  // If location is unknown, this may be a compiler-generated function. Assume
  // it's located in the main file.
  auto &sm = getASTContext().getSourceManager();
  FileEntryRef mainFile = *sm.getFileEntryRefForID(sm.getMainFileID());
  if (noSanitizeL.containsFile(kind, mainFile.getName()))
    return true;

  // Check "src" prefix.
  if (loc.isValid())
    return noSanitizeL.containsLocation(kind, loc);
  // If location is unknown, this may be a compiler-generated function. Assume
  // it's located in the main file.
  return noSanitizeL.containsFile(kind, mainFile.getName());
}

void CIRGenModule::AddDeferredUnusedCoverageMapping(Decl *d) {
  // Do we need to generate coverage mapping?
  if (!codeGenOpts.CoverageMapping)
    return;

  llvm_unreachable("NYI");
}

void CIRGenModule::UpdateCompletedType(const TagDecl *td) {
  // Make sure that this type is translated.
  genTypes.UpdateCompletedType(td);
}

void CIRGenModule::addReplacement(StringRef name, mlir::Operation *op) {
  Replacements[name] = op;
}

void CIRGenModule::applyReplacements() {
  for (auto &i : Replacements) {
    StringRef mangledName = i.first();
    mlir::Operation *replacement = i.second;
    auto *entry = getGlobalValue(mangledName);
    if (!entry)
      continue;
    assert(isa<mlir::cir::FuncOp>(entry) && "expected function");
    auto oldF = cast<mlir::cir::FuncOp>(entry);
    auto newF = dyn_cast<mlir::cir::FuncOp>(replacement);
    assert(newF && "not implemented");

    // Replace old with new, but keep the old order.
    if (oldF.replaceAllSymbolUses(newF.getSymNameAttr(), theModule).failed())
      llvm_unreachable("internal error, cannot RAUW symbol");
    if (newF) {
      newF->moveBefore(oldF);
      oldF->erase();
    }
  }
}

void CIRGenModule::buildExplicitCastExprType(const ExplicitCastExpr *e,
                                             CIRGenFunction *cgf) {
  // Bind VLAs in the cast type.
  if (cgf && e->getType()->isVariablyModifiedType())
    llvm_unreachable("NYI");

  assert(!MissingFeatures::generateDebugInfo() && "NYI");
}

void CIRGenModule::HandleCXXStaticMemberVarInstantiation(VarDecl *vd) {
  auto dk = vd->isThisDeclarationADefinition();
  if (dk == VarDecl::Definition && vd->hasAttr<DLLImportAttr>())
    return;

  TemplateSpecializationKind tsk = vd->getTemplateSpecializationKind();
  // If we have a definition, this might be a deferred decl. If the
  // instantiation is explicit, make sure we emit it at the end.
  if (vd->getDefinition() && tsk == TSK_ExplicitInstantiationDefinition) {
    llvm_unreachable("NYI");
  }

  buildTopLevelDecl(vd);
}

mlir::cir::GlobalOp CIRGenModule::createOrReplaceCXXRuntimeVariable(
    mlir::Location loc, StringRef name, mlir::Type ty,
    mlir::cir::GlobalLinkageKind linkage, clang::CharUnits alignment) {
  mlir::cir::GlobalOp oldGv{};
  auto gv = dyn_cast_or_null<mlir::cir::GlobalOp>(
      mlir::SymbolTable::lookupSymbolIn(getModule(), name));

  if (gv) {
    // Check if the variable has the right type.
    if (gv.getSymType() == ty)
      return gv;

    // Because C++ name mangling, the only way we can end up with an already
    // existing global with the same name is if it has been declared extern
    // "C".
    assert(gv.isDeclaration() && "Declaration has wrong type!");
    oldGv = gv;
  }

  // Create a new variable.
  gv = CIRGenModule::createGlobalOp(*this, loc, name, ty);

  // Set up extra information and add to the module
  gv.setLinkageAttr(
      mlir::cir::GlobalLinkageKindAttr::get(builder.getContext(), linkage));
  mlir::SymbolTable::setSymbolVisibility(gv,
                                         CIRGenModule::getMLIRVisibility(gv));

  if (oldGv) {
    // Replace occurrences of the old variable if needed.
    gv.setName(oldGv.getName());
    if (!oldGv->use_empty()) {
      // TODO(cir): remove erase call above and use replaceGlobal here.
      llvm_unreachable("NYI");
    }
    oldGv->erase();
  }

  if (supportsCOMDAT() && mlir::cir::isWeakForLinker(linkage) &&
      !gv.hasAvailableExternallyLinkage()) {
    gv.setComdat(true);
  }

  gv.setAlignmentAttr(getSize(alignment));
  setDSOLocal(static_cast<mlir::Operation *>(gv));
  return gv;
}

bool CIRGenModule::shouldOpportunisticallyEmitVTables() {
  if (codeGenOpts.OptimizationLevel != 0)
    llvm_unreachable("NYI");
  return codeGenOpts.OptimizationLevel > 0;
}

void CIRGenModule::buildVTableTypeMetadata(const CXXRecordDecl *rd,
                                           mlir::cir::GlobalOp vTable,
                                           const VTableLayout &vtLayout) {
  if (!getCodeGenOpts().LTOUnit)
    return;
  llvm_unreachable("NYI");
}

mlir::Attribute CIRGenModule::getAddrOfRTTIDescriptor(mlir::Location loc,
                                                      QualType ty, bool forEh) {
  // Return a bogus pointer if RTTI is disabled, unless it's for EH.
  // FIXME: should we even be calling this method if RTTI is disabled
  // and it's not for EH?
  if (!shouldEmitRTTI(forEh))
    return getBuilder().getConstNullPtrAttr(builder.getUInt8PtrTy());

  if (forEh && ty->isObjCObjectPointerType() &&
      getLangOpts().ObjCRuntime.isGNUFamily()) {
    llvm_unreachable("NYI");
  }

  return getCXXABI().getAddrOfRTTIDescriptor(loc, ty);
}

/// TODO(cir): once we have cir.module, add this as a convenience method there.
///
/// Look up the specified global in the module symbol table.
///   1. If it does not exist, add a declaration of the global and return it.
///   2. Else, the global exists but has the wrong type: return the function
///      with a constantexpr cast to the right type.
///   3. Finally, if the existing global is the correct declaration, return the
///      existing global.
mlir::cir::GlobalOp CIRGenModule::getOrInsertGlobal(
    mlir::Location loc, StringRef name, mlir::Type ty,
    llvm::function_ref<mlir::cir::GlobalOp()> createGlobalCallback) {
  // See if we have a definition for the specified global already.
  auto gv = dyn_cast_or_null<mlir::cir::GlobalOp>(getGlobalValue(name));
  if (!gv) {
    gv = createGlobalCallback();
  }
  assert(gv && "The CreateGlobalCallback is expected to create a global");

  // If the variable exists but has the wrong type, return a bitcast to the
  // right type.
  auto gvTy = gv.getSymType();
  assert(!MissingFeatures::addressSpace());
  auto pTy = builder.getPointerTo(ty);

  if (gvTy != pTy)
    llvm_unreachable("NYI");

  // Otherwise, we just found the existing function or a prototype.
  return gv;
}

// Overload to construct a global variable using its constructor's defaults.
mlir::cir::GlobalOp CIRGenModule::getOrInsertGlobal(mlir::Location loc,
                                                    StringRef name,
                                                    mlir::Type ty) {
  return getOrInsertGlobal(loc, name, ty, [&] {
    return CIRGenModule::createGlobalOp(*this, loc, name,
                                        builder.getPointerTo(ty));
  });
}

// TODO(cir): this can be shared with LLVM codegen.
CharUnits CIRGenModule::computeNonVirtualBaseClassOffset(
    const CXXRecordDecl *derivedClass, CastExpr::path_const_iterator start,
    CastExpr::path_const_iterator end) {
  CharUnits offset = CharUnits::Zero();

  const ASTContext &context = getASTContext();
  const CXXRecordDecl *rd = derivedClass;

  for (CastExpr::path_const_iterator i = start; i != end; ++i) {
    const CXXBaseSpecifier *base = *i;
    assert(!base->isVirtual() && "Should not see virtual bases here!");

    // Get the layout.
    const ASTRecordLayout &layout = context.getASTRecordLayout(rd);

    const auto *baseDecl =
        cast<CXXRecordDecl>(base->getType()->castAs<RecordType>()->getDecl());

    // Add the offset.
    offset += layout.getBaseClassOffset(baseDecl);

    rd = baseDecl;
  }

  return offset;
}

void CIRGenModule::Error(SourceLocation loc, StringRef message) {
  unsigned diagID = getDiags().getCustomDiagID(DiagnosticsEngine::Error, "%0");
  getDiags().Report(astCtx.getFullLoc(loc), diagID) << message;
}

/// Print out an error that codegen doesn't support the specified stmt yet.
void CIRGenModule::ErrorUnsupported(const Stmt *s, const char *type) {
  unsigned diagId = getDiags().getCustomDiagID(DiagnosticsEngine::Error,
                                               "cannot compile this %0 yet");
  std::string msg = type;
  getDiags().Report(astCtx.getFullLoc(s->getBeginLoc()), diagId)
      << msg << s->getSourceRange();
}

/// Print out an error that codegen doesn't support the specified decl yet.
void CIRGenModule::ErrorUnsupported(const Decl *d, const char *type) {
  unsigned diagId = getDiags().getCustomDiagID(DiagnosticsEngine::Error,
                                               "cannot compile this %0 yet");
  std::string msg = type;
  getDiags().Report(astCtx.getFullLoc(d->getLocation()), diagId) << msg;
}

mlir::cir::SourceLanguage CIRGenModule::getCIRSourceLanguage() {
  using ClangStd = clang::LangStandard;
  using CIRLang = mlir::cir::SourceLanguage;
  auto opts = getLangOpts();

  if (opts.OpenCL && !opts.OpenCLCPlusPlus)
    return CIRLang::OpenCLC;

  if (opts.CPlusPlus || opts.CPlusPlus11 || opts.CPlusPlus14 ||
      opts.CPlusPlus17 || opts.CPlusPlus20 || opts.CPlusPlus23 ||
      opts.CPlusPlus26)
    return CIRLang::CXX;
  if (opts.C99 || opts.C11 || opts.C17 || opts.C23 ||
      opts.LangStd == ClangStd::lang_c89 ||
      opts.LangStd == ClangStd::lang_gnu89)
    return CIRLang::C;

  // TODO(cir): support remaining source languages.
  llvm_unreachable("CIR does not yet support the given source language");
}

LangAS CIRGenModule::getGlobalVarAddressSpace(const VarDecl *d) {
  if (langOpts.OpenCL) {
    LangAS as = d ? d->getType().getAddressSpace() : LangAS::opencl_global;
    assert(as == LangAS::opencl_global || as == LangAS::opencl_global_device ||
           as == LangAS::opencl_global_host || as == LangAS::opencl_constant ||
           as == LangAS::opencl_local || as >= LangAS::FirstTargetAddressSpace);
    return as;
  }

  if (langOpts.SYCLIsDevice &&
      (!d || d->getType().getAddressSpace() == LangAS::Default))
    llvm_unreachable("NYI");

  if (langOpts.CUDA && langOpts.CUDAIsDevice)
    llvm_unreachable("NYI");

  if (langOpts.OpenMP)
    llvm_unreachable("NYI");

  return getTargetCIRGenInfo().getGlobalVarAddressSpace(*this, d);
}

mlir::ArrayAttr CIRGenModule::buildAnnotationArgs(AnnotateAttr *attr) {
  ArrayRef<Expr *> exprs = {attr->args_begin(), attr->args_size()};
  if (exprs.empty()) {
    return mlir::ArrayAttr::get(builder.getContext(), {});
  }
  llvm::FoldingSetNodeID id;
  for (Expr *e : exprs) {
    id.Add(cast<clang::ConstantExpr>(e)->getAPValueResult());
  }
  mlir::ArrayAttr &lookup = annotationArgs[id.ComputeHash()];
  if (lookup)
    return lookup;

  llvm::SmallVector<mlir::Attribute, 4> args;
  args.reserve(exprs.size());
  for (Expr *e : exprs) {
    auto &ce = *cast<clang::ConstantExpr>(e);
    if (auto *const strE =
            clang::dyn_cast<clang::StringLiteral>(ce.IgnoreParenCasts())) {
      // Add trailing null character as StringLiteral->getString() does not
      args.push_back(builder.getStringAttr(strE->getString()));
    } else if (ce.hasAPValueResult()) {
      // Handle case which can be evaluated to some numbers, not only literals
      const auto &ap = ce.getAPValueResult();
      if (ap.isInt()) {
        args.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(builder.getContext(),
                                   ap.getInt().getBitWidth()),
            ap.getInt()));
      } else {
        llvm_unreachable("NYI like float, fixed-point, array...");
      }
    } else {
      llvm_unreachable("NYI");
    }
  }

  lookup = builder.getArrayAttr(args);
  return lookup;
}

mlir::cir::AnnotationAttr
CIRGenModule::buildAnnotateAttr(clang::AnnotateAttr *aa) {
  mlir::StringAttr annoGV = builder.getStringAttr(aa->getAnnotation());
  mlir::ArrayAttr args = buildAnnotationArgs(aa);
  return mlir::cir::AnnotationAttr::get(builder.getContext(), annoGV, args);
}

void CIRGenModule::addGlobalAnnotations(const ValueDecl *d,
                                        mlir::Operation *gv) {
  assert(d->hasAttr<AnnotateAttr>() && "no annotate attribute");
  assert((isa<GlobalOp>(gv) || isa<FuncOp>(gv)) &&
         "annotation only on globals");
  llvm::SmallVector<mlir::Attribute, 4> annotations;
  for (auto *i : d->specific_attrs<AnnotateAttr>())
    annotations.push_back(buildAnnotateAttr(i));
  if (auto global = dyn_cast<mlir::cir::GlobalOp>(gv))
    global.setAnnotationsAttr(builder.getArrayAttr(annotations));
  else if (auto func = dyn_cast<mlir::cir::FuncOp>(gv))
    func.setAnnotationsAttr(builder.getArrayAttr(annotations));
}

void CIRGenModule::buildGlobalAnnotations() {
  for (const auto &[mangledName, vd] : deferredAnnotations) {
    mlir::Operation *gv = getGlobalValue(mangledName);
    if (gv)
      addGlobalAnnotations(vd, gv);
  }
  deferredAnnotations.clear();
}

TBAAAccessInfo CIRGenModule::getTBAAAccessInfo(QualType accessType) {
  if (!tbaa)
    return TBAAAccessInfo();
  llvm_unreachable("NYI");
}
