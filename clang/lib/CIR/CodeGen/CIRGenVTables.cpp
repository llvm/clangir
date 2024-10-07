//===--- CIRGenVTables.cpp - Emit CIR Code for C++ vtables ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of virtual tables.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "mlir/IR/Attributes.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/VTTBuilder.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "clang/CodeGen/ConstantInitBuilder.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <algorithm>
#include <cstdio>

using namespace clang;
using namespace cir;

CIRGenVTables::CIRGenVTables(CIRGenModule &cgm)
    : CGM(cgm), VTContext(cgm.getASTContext().getVTableContext()) {}

static bool useRelativeLayout(const CIRGenModule &cgm) {
  return cgm.getTarget().getCXXABI().isItaniumFamily() &&
         cgm.getItaniumVTableContext().isRelativeLayout();
}

bool CIRGenVTables::useRelativeLayout() const {
  return ::useRelativeLayout(CGM);
}

mlir::Type CIRGenModule::getVTableComponentType() {
  mlir::Type ptrTy = builder.getUInt8PtrTy();
  if (useRelativeLayout(*this))
    ptrTy = builder.getUInt32PtrTy();
  return ptrTy;
}

mlir::Type CIRGenVTables::getVTableComponentType() {
  return CGM.getVTableComponentType();
}

mlir::Type CIRGenVTables::getVTableType(const VTableLayout &layout) {
  SmallVector<mlir::Type, 4> tys;
  auto *ctx = CGM.getBuilder().getContext();
  auto componentType = getVTableComponentType();
  for (unsigned i = 0, e = layout.getNumVTables(); i != e; ++i)
    tys.push_back(
        mlir::cir::ArrayType::get(ctx, componentType, layout.getVTableSize(i)));

  // FIXME(cir): should VTableLayout be encoded like we do for some
  // AST nodes?
  return CGM.getBuilder().getAnonStructTy(tys, /*incomplete=*/false);
}

/// At this point in the translation unit, does it appear that can we
/// rely on the vtable being defined elsewhere in the program?
///
/// The response is really only definitive when called at the end of
/// the translation unit.
///
/// The only semantic restriction here is that the object file should
/// not contain a vtable definition when that vtable is defined
/// strongly elsewhere.  Otherwise, we'd just like to avoid emitting
/// vtables when unnecessary.
/// TODO(cir): this should be merged into common AST helper for codegen.
bool CIRGenVTables::isVTableExternal(const CXXRecordDecl *rd) {
  assert(rd->isDynamicClass() && "Non-dynamic classes have no VTable.");

  // We always synthesize vtables if they are needed in the MS ABI. MSVC doesn't
  // emit them even if there is an explicit template instantiation.
  if (CGM.getTarget().getCXXABI().isMicrosoft())
    return false;

  // If we have an explicit instantiation declaration (and not a
  // definition), the vtable is defined elsewhere.
  TemplateSpecializationKind tsk = rd->getTemplateSpecializationKind();
  if (tsk == TSK_ExplicitInstantiationDeclaration)
    return true;

  // Otherwise, if the class is an instantiated template, the
  // vtable must be defined here.
  if (tsk == TSK_ImplicitInstantiation ||
      tsk == TSK_ExplicitInstantiationDefinition)
    return false;

  // Otherwise, if the class doesn't have a key function (possibly
  // anymore), the vtable must be defined here.
  const CXXMethodDecl *keyFunction =
      CGM.getASTContext().getCurrentKeyFunction(rd);
  if (!keyFunction)
    return false;

  // Otherwise, if we don't have a definition of the key function, the
  // vtable must be defined somewhere else.
  return !keyFunction->hasBody();
}

static bool shouldEmitAvailableExternallyVTable(const CIRGenModule &cgm,
                                                const CXXRecordDecl *rd) {
  return cgm.getCodeGenOpts().OptimizationLevel > 0 &&
         cgm.getCXXABI().canSpeculativelyEmitVTable(rd);
}

/// Given that we're currently at the end of the translation unit, and
/// we've emitted a reference to the vtable for this class, should
/// we define that vtable?
static bool shouldEmitVTableAtEndOfTranslationUnit(CIRGenModule &cgm,
                                                   const CXXRecordDecl *rd) {
  // If vtable is internal then it has to be done.
  if (!cgm.getVTables().isVTableExternal(rd))
    return true;

  // If it's external then maybe we will need it as available_externally.
  return shouldEmitAvailableExternallyVTable(cgm, rd);
}

/// Given that at some point we emitted a reference to one or more
/// vtables, and that we are now at the end of the translation unit,
/// decide whether we should emit them.
void CIRGenModule::buildDeferredVTables() {
#ifndef NDEBUG
  // Remember the size of DeferredVTables, because we're going to assume
  // that this entire operation doesn't modify it.
  size_t savedSize = DeferredVTables.size();
#endif

  for (const CXXRecordDecl *rd : DeferredVTables)
    if (shouldEmitVTableAtEndOfTranslationUnit(*this, rd)) {
      VTables.GenerateClassData(rd);
    } else if (shouldOpportunisticallyEmitVTables()) {
      llvm_unreachable("NYI");
    }

  assert(savedSize == DeferredVTables.size() &&
         "deferred extra vtables during vtable emission?");
  DeferredVTables.clear();
}

void CIRGenVTables::GenerateClassData(const CXXRecordDecl *rd) {
  assert(!MissingFeatures::generateDebugInfo());

  if (rd->getNumVBases())
    CGM.getCXXABI().emitVirtualInheritanceTables(rd);

  CGM.getCXXABI().emitVTableDefinitions(*this, rd);
}

static void addPointerLayoutOffset(CIRGenModule &cgm,
                                   ConstantArrayBuilder &builder,
                                   CharUnits offset) {
  builder.add(cgm.getBuilder().getConstPtrAttr(cgm.getBuilder().getUInt8PtrTy(),
                                               offset.getQuantity()));
}

static void addRelativeLayoutOffset(CIRGenModule &cgm,
                                    ConstantArrayBuilder &builder,
                                    CharUnits offset) {
  llvm_unreachable("NYI");
  // builder.add(llvm::ConstantInt::get(CGM.Int32Ty, offset.getQuantity()));
}

void CIRGenVTables::addVTableComponent(ConstantArrayBuilder &builder,
                                       const VTableLayout &layout,
                                       unsigned componentIndex,
                                       mlir::Attribute rtti,
                                       unsigned &nextVTableThunkIndex,
                                       unsigned vtableAddressPoint,
                                       bool vtableHasLocalLinkage) {
  auto &component = layout.vtable_components()[componentIndex];

  auto addOffsetConstant =
      useRelativeLayout() ? addRelativeLayoutOffset : addPointerLayoutOffset;

  switch (component.getKind()) {
  case VTableComponent::CK_VCallOffset:
    return addOffsetConstant(CGM, builder, component.getVCallOffset());

  case VTableComponent::CK_VBaseOffset:
    return addOffsetConstant(CGM, builder, component.getVBaseOffset());

  case VTableComponent::CK_OffsetToTop:
    return addOffsetConstant(CGM, builder, component.getOffsetToTop());

  case VTableComponent::CK_RTTI:
    if (useRelativeLayout()) {
      llvm_unreachable("NYI");
      // return addRelativeComponent(builder, rtti, vtableAddressPoint,
      //                             vtableHasLocalLinkage,
      //                             /*isCompleteDtor=*/false);
    } else {
      assert((mlir::isa<mlir::cir::GlobalViewAttr>(rtti) ||
              mlir::isa<mlir::cir::ConstPtrAttr>(rtti)) &&
             "expected GlobalViewAttr or ConstPtrAttr");
      return builder.add(rtti);
    }

  case VTableComponent::CK_FunctionPointer:
  case VTableComponent::CK_CompleteDtorPointer:
  case VTableComponent::CK_DeletingDtorPointer: {
    GlobalDecl gd = component.getGlobalDecl();

    if (CGM.getLangOpts().CUDA) {
      llvm_unreachable("NYI");
    }

    auto getSpecialVirtualFn = [&](StringRef name) -> mlir::cir::FuncOp {
      // FIXME(PR43094): When merging comdat groups, lld can select a local
      // symbol as the signature symbol even though it cannot be accessed
      // outside that symbol's TU. The relative vtables ABI would make
      // __cxa_pure_virtual and __cxa_deleted_virtual local symbols, and
      // depending on link order, the comdat groups could resolve to the one
      // with the local symbol. As a temporary solution, fill these components
      // with zero. We shouldn't be calling these in the first place anyway.
      if (useRelativeLayout())
        llvm_unreachable("NYI");

      // For NVPTX devices in OpenMP emit special functon as null pointers,
      // otherwise linking ends up with unresolved references.
      if (CGM.getLangOpts().OpenMP && CGM.getLangOpts().OpenMPIsTargetDevice &&
          CGM.getTriple().isNVPTX())
        llvm_unreachable("NYI");

      mlir::cir::FuncType fnTy =
          CGM.getBuilder().getFuncType({}, CGM.getBuilder().getVoidTy());
      mlir::cir::FuncOp fnPtr = CGM.createRuntimeFunction(fnTy, name);
      // LLVM codegen handles unnamedAddr
      assert(!MissingFeatures::unnamedAddr());
      return fnPtr;
    };

    mlir::cir::FuncOp fnPtr;
    if (cast<CXXMethodDecl>(gd.getDecl())->isPureVirtual()) {
      // Pure virtual member functions.
      if (!PureVirtualFn)
        PureVirtualFn =
            getSpecialVirtualFn(CGM.getCXXABI().getPureVirtualCallName());
      fnPtr = PureVirtualFn;

    } else if (cast<CXXMethodDecl>(gd.getDecl())->isDeleted()) {
      // Deleted virtual member functions.
      if (!DeletedVirtualFn)
        DeletedVirtualFn =
            getSpecialVirtualFn(CGM.getCXXABI().getDeletedVirtualCallName());
      fnPtr = DeletedVirtualFn;

    } else if (nextVTableThunkIndex < layout.vtable_thunks().size() &&
               layout.vtable_thunks()[nextVTableThunkIndex].first ==
                   componentIndex) {
      // Thunks.
      llvm_unreachable("NYI");
      // auto &thunkInfo = layout.vtable_thunks()[nextVTableThunkIndex].second;

      // nextVTableThunkIndex++;
      // fnPtr = maybeEmitThunk(GD, thunkInfo, /*ForVTable=*/true);

    } else {
      // Otherwise we can use the method definition directly.
      auto fnTy = CGM.getTypes().GetFunctionTypeForVTable(gd);
      fnPtr = CGM.GetAddrOfFunction(gd, fnTy, /*ForVTable=*/true);
    }

    if (useRelativeLayout()) {
      llvm_unreachable("NYI");
    } else {
      return builder.add(mlir::cir::GlobalViewAttr::get(
          CGM.getBuilder().getUInt8PtrTy(),
          mlir::FlatSymbolRefAttr::get(fnPtr.getSymNameAttr())));
    }
  }

  case VTableComponent::CK_UnusedFunctionPointer:
    if (useRelativeLayout())
      llvm_unreachable("NYI");
    else {
      llvm_unreachable("NYI");
      // return builder.addNullPointer(CGM.Int8PtrTy);
    }
  }

  llvm_unreachable("Unexpected vtable component kind");
}

void CIRGenVTables::createVTableInitializer(ConstantStructBuilder &builder,
                                            const VTableLayout &layout,
                                            mlir::Attribute rtti,
                                            bool vtableHasLocalLinkage) {
  auto componentType = getVTableComponentType();

  const auto &addressPoints = layout.getAddressPointIndices();
  unsigned nextVTableThunkIndex = 0;
  for (unsigned vtableIndex = 0, endIndex = layout.getNumVTables();
       vtableIndex != endIndex; ++vtableIndex) {
    auto vtableElem = builder.beginArray(componentType);

    size_t vtableStart = layout.getVTableOffset(vtableIndex);
    size_t vtableEnd = vtableStart + layout.getVTableSize(vtableIndex);
    for (size_t componentIndex = vtableStart; componentIndex < vtableEnd;
         ++componentIndex) {
      addVTableComponent(vtableElem, layout, componentIndex, rtti,
                         nextVTableThunkIndex, addressPoints[vtableIndex],
                         vtableHasLocalLinkage);
    }
    vtableElem.finishAndAddTo(rtti.getContext(), builder);
  }
}

/// Compute the required linkage of the vtable for the given class.
///
/// Note that we only call this at the end of the translation unit.
mlir::cir::GlobalLinkageKind
CIRGenModule::getVTableLinkage(const CXXRecordDecl *rd) {
  if (!rd->isExternallyVisible())
    return mlir::cir::GlobalLinkageKind::InternalLinkage;

  // We're at the end of the translation unit, so the current key
  // function is fully correct.
  const CXXMethodDecl *keyFunction = astCtx.getCurrentKeyFunction(rd);
  if (keyFunction && !rd->hasAttr<DLLImportAttr>()) {
    // If this class has a key function, use that to determine the
    // linkage of the vtable.
    const FunctionDecl *def = nullptr;
    if (keyFunction->hasBody(def))
      keyFunction = cast<CXXMethodDecl>(def);

    switch (keyFunction->getTemplateSpecializationKind()) {
    case TSK_Undeclared:
    case TSK_ExplicitSpecialization:
      assert(
          (def || codeGenOpts.OptimizationLevel > 0 ||
           codeGenOpts.getDebugInfo() != llvm::codegenoptions::NoDebugInfo) &&
          "Shouldn't query vtable linkage without key function, "
          "optimizations, or debug info");
      if (!def && codeGenOpts.OptimizationLevel > 0)
        return mlir::cir::GlobalLinkageKind::AvailableExternallyLinkage;

      if (keyFunction->isInlined())
        return !astCtx.getLangOpts().AppleKext
                   ? mlir::cir::GlobalLinkageKind::LinkOnceODRLinkage
                   : mlir::cir::GlobalLinkageKind::InternalLinkage;

      return mlir::cir::GlobalLinkageKind::ExternalLinkage;

    case TSK_ImplicitInstantiation:
      return !astCtx.getLangOpts().AppleKext
                 ? mlir::cir::GlobalLinkageKind::LinkOnceODRLinkage
                 : mlir::cir::GlobalLinkageKind::InternalLinkage;

    case TSK_ExplicitInstantiationDefinition:
      return !astCtx.getLangOpts().AppleKext
                 ? mlir::cir::GlobalLinkageKind::WeakODRLinkage
                 : mlir::cir::GlobalLinkageKind::InternalLinkage;

    case TSK_ExplicitInstantiationDeclaration:
      llvm_unreachable("Should not have been asked to emit this");
    }
  }

  // -fapple-kext mode does not support weak linkage, so we must use
  // internal linkage.
  if (astCtx.getLangOpts().AppleKext)
    return mlir::cir::GlobalLinkageKind::InternalLinkage;

  auto discardableOdrLinkage = mlir::cir::GlobalLinkageKind::LinkOnceODRLinkage;
  auto nonDiscardableOdrLinkage = mlir::cir::GlobalLinkageKind::WeakODRLinkage;
  if (rd->hasAttr<DLLExportAttr>()) {
    // Cannot discard exported vtables.
    discardableOdrLinkage = nonDiscardableOdrLinkage;
  } else if (rd->hasAttr<DLLImportAttr>()) {
    // Imported vtables are available externally.
    discardableOdrLinkage =
        mlir::cir::GlobalLinkageKind::AvailableExternallyLinkage;
    nonDiscardableOdrLinkage =
        mlir::cir::GlobalLinkageKind::AvailableExternallyLinkage;
  }

  switch (rd->getTemplateSpecializationKind()) {
  case TSK_Undeclared:
  case TSK_ExplicitSpecialization:
  case TSK_ImplicitInstantiation:
    return discardableOdrLinkage;

  case TSK_ExplicitInstantiationDeclaration: {
    // Explicit instantiations in MSVC do not provide vtables, so we must emit
    // our own.
    if (getTarget().getCXXABI().isMicrosoft())
      return discardableOdrLinkage;
    auto r = shouldEmitAvailableExternallyVTable(*this, rd)
                 ? mlir::cir::GlobalLinkageKind::AvailableExternallyLinkage
                 : mlir::cir::GlobalLinkageKind::ExternalLinkage;
    assert(r == mlir::cir::GlobalLinkageKind::ExternalLinkage &&
           "available external NYI");
    return r;
  }

  case TSK_ExplicitInstantiationDefinition:
    return nonDiscardableOdrLinkage;
  }

  llvm_unreachable("Invalid TemplateSpecializationKind!");
}

mlir::cir::GlobalOp
getAddrOfVTTVTable(CIRGenVTables &cgvt, CIRGenModule &cgm,
                   const CXXRecordDecl *mostDerivedClass,
                   const VTTVTable &vtable,
                   mlir::cir::GlobalLinkageKind linkage,
                   VTableLayout::AddressPointsMapTy &addressPoints) {
  if (vtable.getBase() == mostDerivedClass) {
    assert(vtable.getBaseOffset().isZero() &&
           "Most derived class vtable must have a zero offset!");
    // This is a regular vtable.
    return cgm.getCXXABI().getAddrOfVTable(mostDerivedClass, CharUnits());
  }

  llvm_unreachable("generateConstructionVTable NYI");
}

mlir::cir::GlobalOp CIRGenVTables::getAddrOfVTT(const CXXRecordDecl *rd) {
  assert(rd->getNumVBases() && "Only classes with virtual bases need a VTT");

  SmallString<256> outName;
  llvm::raw_svector_ostream out(outName);
  cast<ItaniumMangleContext>(CGM.getCXXABI().getMangleContext())
      .mangleCXXVTT(rd, out);
  StringRef name = outName.str();

  // This will also defer the definition of the VTT.
  (void)CGM.getCXXABI().getAddrOfVTable(rd, CharUnits());

  VTTBuilder builder(CGM.getASTContext(), rd, /*GenerateDefinition=*/false);

  auto arrayType = mlir::cir::ArrayType::get(CGM.getBuilder().getContext(),
                                             CGM.getBuilder().getUInt8PtrTy(),
                                             builder.getVTTComponents().size());
  auto align =
      CGM.getDataLayout().getABITypeAlign(CGM.getBuilder().getUInt8PtrTy());
  auto vtt = CGM.createOrReplaceCXXRuntimeVariable(
      CGM.getLoc(rd->getSourceRange()), name, arrayType,
      mlir::cir::GlobalLinkageKind::ExternalLinkage,
      CharUnits::fromQuantity(align));
  CGM.setGVProperties(vtt, rd);
  return vtt;
}

/// Emit the definition of the given vtable.
void CIRGenVTables::buildVTTDefinition(mlir::cir::GlobalOp vtt,
                                       mlir::cir::GlobalLinkageKind linkage,
                                       const CXXRecordDecl *rd) {
  VTTBuilder Builder(CGM.getASTContext(), rd, /*GenerateDefinition=*/true);

  auto arrayType = mlir::cir::ArrayType::get(CGM.getBuilder().getContext(),
                                             CGM.getBuilder().getUInt8PtrTy(),
                                             Builder.getVTTComponents().size());

  SmallVector<mlir::cir::GlobalOp, 8> vTables;
  SmallVector<VTableAddressPointsMapTy, 8> vTableAddressPoints;
  for (auto i : Builder.getVTTVTables()) {
    vTableAddressPoints.push_back(VTableAddressPointsMapTy());
    vTables.push_back(getAddrOfVTTVTable(*this, CGM, rd, i, linkage,
                                         vTableAddressPoints.back()));
  }

  SmallVector<mlir::Attribute, 8> vttComponents;
  for (const VTTComponent *i = Builder.getVTTComponents().begin(),
                          *e = Builder.getVTTComponents().end();
       i != e; ++i) {
    const VTTVTable &vttvt = Builder.getVTTVTables()[i->VTableIndex];
    mlir::cir::GlobalOp vTable = vTables[i->VTableIndex];
    VTableLayout::AddressPointLocation addressPoint;
    if (vttvt.getBase() == rd) {
      // Just get the address point for the regular vtable.
      addressPoint =
          getItaniumVTableContext().getVTableLayout(rd).getAddressPoint(
              i->VTableBase);
    } else {
      addressPoint = vTableAddressPoints[i->VTableIndex].lookup(i->VTableBase);
      assert(addressPoint.AddressPointIndex != 0 &&
             "Did not find ctor vtable address point!");
    }

    mlir::Attribute idxs[3] = {
        CGM.getBuilder().getI32IntegerAttr(0),
        CGM.getBuilder().getI32IntegerAttr(addressPoint.VTableIndex),
        CGM.getBuilder().getI32IntegerAttr(addressPoint.AddressPointIndex),
    };

    auto indices = mlir::ArrayAttr::get(CGM.getBuilder().getContext(), idxs);
    auto init = CGM.getBuilder().getGlobalViewAttr(
        CGM.getBuilder().getUInt8PtrTy(), vTable, indices);

    vttComponents.push_back(init);
  }

  auto init = CGM.getBuilder().getConstArray(
      mlir::ArrayAttr::get(CGM.getBuilder().getContext(), vttComponents),
      arrayType);

  vtt.setInitialValueAttr(init);

  // Set the correct linkage.
  vtt.setLinkage(linkage);
  mlir::SymbolTable::setSymbolVisibility(vtt,
                                         CIRGenModule::getMLIRVisibility(vtt));

  if (CGM.supportsCOMDAT() && vtt.isWeakForLinker()) {
    assert(!MissingFeatures::setComdat());
  }
}

void CIRGenVTables::buildThunks(GlobalDecl gd) {
  const CXXMethodDecl *md =
      cast<CXXMethodDecl>(gd.getDecl())->getCanonicalDecl();

  // We don't need to generate thunks for the base destructor.
  if (isa<CXXDestructorDecl>(md) && gd.getDtorType() == Dtor_Base)
    return;

  const VTableContextBase::ThunkInfoVectorTy *thunkInfoVector =
      VTContext->getThunkInfo(gd);

  if (!thunkInfoVector)
    return;

  for ([[maybe_unused]] const ThunkInfo &thunk : *thunkInfoVector)
    llvm_unreachable("NYI");
}

bool CIRGenModule::AlwaysHasLTOVisibilityPublic(const CXXRecordDecl *rd) {
  if (rd->hasAttr<LTOVisibilityPublicAttr>() || rd->hasAttr<UuidAttr>() ||
      rd->hasAttr<DLLExportAttr>() || rd->hasAttr<DLLImportAttr>())
    return true;

  if (!getCodeGenOpts().LTOVisibilityPublicStd)
    return false;

  const DeclContext *dc = rd;
  while (true) {
    auto *d = cast<Decl>(dc);
    dc = dc->getParent();
    if (isa<TranslationUnitDecl>(dc->getRedeclContext())) {
      if (auto *nd = dyn_cast<NamespaceDecl>(d))
        if (const IdentifierInfo *ii = nd->getIdentifier())
          if (ii->isStr("std") || ii->isStr("stdext"))
            return true;
      break;
    }
  }

  return false;
}

bool CIRGenModule::HasHiddenLTOVisibility(const CXXRecordDecl *rd) {
  LinkageInfo lv = rd->getLinkageAndVisibility();
  if (!isExternallyVisible(lv.getLinkage()))
    return true;

  if (!getTriple().isOSBinFormatCOFF() &&
      lv.getVisibility() != HiddenVisibility)
    return false;

  return !AlwaysHasLTOVisibilityPublic(rd);
}
