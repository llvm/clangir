//===---- MissingFeatures.h - Checks for unimplemented features -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file introduces some helper classes to guard against features that
// CIR dialect supports that we do not have and also do not have great ways to
// assert against.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_MISSINGFEATURES_H
#define CLANG_CIR_MISSINGFEATURES_H

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>

constexpr bool cirCConvAssertionMode =
    true; // Change to `false` to use llvm_unreachable

#define CIR_CCONV_NOTE                                                         \
  " Target lowering is now required. To workaround use "                       \
  "-fno-clangir-call-conv-lowering. This flag is going to be removed at some"  \
  " point."

// Special assertion to be used in the target lowering library.
#define cir_cconv_assert(cond)                                                 \
  do {                                                                         \
    if (!(cond))                                                               \
      llvm::errs() << CIR_CCONV_NOTE << "\n";                                  \
    assert((cond));                                                            \
  } while (0)

// Special version of cir_cconv_unreachable to give more info to the user on how
// to temporarily disable target lowering.
#define cir_cconv_unreachable(msg)                                             \
  do {                                                                         \
    llvm_unreachable(msg CIR_CCONV_NOTE);                                      \
  } while (0)

// Some assertions knowingly generate incorrect code. This macro allows us to
// switch between using `assert` and `llvm_unreachable` for these cases.
#define cir_cconv_assert_or_abort(cond, msg)                                   \
  do {                                                                         \
    if (cirCConvAssertionMode) {                                               \
      assert((cond) && msg CIR_CCONV_NOTE);                                    \
    } else {                                                                   \
      llvm_unreachable(msg CIR_CCONV_NOTE);                                    \
    }                                                                          \
  } while (0)

namespace cir {

// As a way to track features that haven't yet been implemented this class
// explicitly contains a list of static fns that will return false that you
// can guard against. If and when a feature becomes implemented simply changing
// this return to true will cause compilation to fail at all the points in which
// we noted that we needed to address. This is a much more explicit way to
// handle "TODO"s.
struct MissingFeatures {
  // Address space related
  static bool addressSpace() { return false; }

  // Unhandled global/linkage information.
  static bool opGlobalThreadLocal() { return false; }
  static bool opGlobalWeakRef() { return false; }
  static bool opGlobalUnnamedAddr() { return false; }
  static bool opGlobalSection() { return false; }
  static bool opGlobalVisibility() { return false; }
  static bool opGlobalDLLImportExport() { return false; }
  static bool opGlobalPartition() { return false; }
  static bool opGlobalUsedOrCompilerUsed() { return false; }
  static bool opGlobalAnnotations() { return false; }
  static bool opGlobalCtorPriority() { return false; }
  static bool setDSOLocal() { return false; }
  static bool setComdat() { return false; }

  static bool supportIFuncAttr() { return false; }
  static bool supportVisibility() { return false; }
  static bool hiddenVisibility() { return false; }
  static bool protectedVisibility() { return false; }
  static bool defaultVisibility() { return false; }

  // Load/store attributes
  static bool opLoadStoreThreadLocal() { return false; }
  static bool opLoadEmitScalarRangeCheck() { return false; }
  static bool opLoadStoreNontemporal() { return false; }
  static bool opLoadStoreTbaa() { return false; }
  static bool opLoadStoreAtomic() { return false; }
  static bool opLoadStoreObjC() { return false; }

  // AllocaOp handling
  static bool opAllocaStaticLocal() { return false; }
  static bool opAllocaNonGC() { return false; }
  static bool opAllocaImpreciseLifetime() { return false; }
  static bool opAllocaPreciseLifetime() { return false; }
  static bool opAllocaTLS() { return false; }
  static bool opAllocaOpenMPThreadPrivate() { return false; }
  static bool opAllocaEscapeByReference() { return false; }
  static bool opAllocaReference() { return false; }
  static bool opAllocaAnnotations() { return false; }
  static bool opAllocaCaptureByInit() { return false; }

  // FuncOp handling
  static bool opFuncArmNewAttr() { return false; }
  static bool opFuncArmStreamingAttr() { return false; }
  static bool opFuncAstDeclAttr() { return false; }
  static bool opFuncCallingConv() { return false; }
  static bool opFuncColdHotAttr() { return false; }
  static bool opFuncCPUAndFeaturesAttributes() { return false; }
  static bool opFuncExceptions() { return false; }
  static bool opFuncExtraAttrs() { return false; }
  static bool opFuncMaybeHandleStaticInExternC() { return false; }
  static bool opFuncMinSizeAttr() { return false; }
  static bool opFuncMultipleReturnVals() { return false; }
  static bool opFuncNakedAttr() { return false; }
  static bool opFuncNoDuplicateAttr() { return false; }
  static bool opFuncNoUnwind() { return false; }
  static bool opFuncOpenCLKernelMetadata() { return false; }
  static bool opFuncOperandBundles() { return false; }
  static bool opFuncOptNoneAttr() { return false; }
  static bool opFuncParameterAttributes() { return false; }
  static bool opFuncReadOnly() { return false; }
  static bool opFuncSection() { return false; }
  static bool opFuncUnwindTablesAttr() { return false; }
  static bool opFuncWillReturn() { return false; }
  static bool opFuncNoReturn() { return false; }
  static bool setLLVMFunctionFEnvAttributes() { return false; }

  // CallOp handling
  static bool opCallAggregateArgs() { return false; }
  static bool opCallPaddingArgs() { return false; }
  static bool opCallABIExtendArg() { return false; }
  static bool opCallABIIndirectArg() { return false; }
  static bool opCallWidenArg() { return false; }
  static bool opCallBitcastArg() { return false; }
  static bool opCallImplicitObjectSizeArgs() { return false; }
  static bool opCallReturn() { return false; }
  static bool opCallArgEvaluationOrder() { return false; }
  static bool opCallCallConv() { return false; }
  static bool opCallSideEffect() { return false; }
  static bool opCallMustTail() { return false; }
  static bool opCallInAlloca() { return false; }
  static bool opCallAttrs() { return false; }
  static bool opCallASTAttr() { return false; }
  static bool opCallObjCMethod() { return false; }
  static bool opCallExtParameterInfo() { return false; }
  static bool opCallCIRGenFuncInfoParamInfo() { return false; }
  static bool opCallCIRGenFuncInfoExtParamInfo() { return false; }
  static bool opCallLandingPad() { return false; }
  static bool opCallContinueBlock() { return false; }
  static bool opCallChain() { return false; }
  static bool opCallExceptionAttr() { return false; }

  // FnInfoOpts -- This is used to track whether calls are chain calls or
  // instance methods. Classic codegen uses chain call to track and extra free
  // register for x86 and uses instance method as a condition for a thunk
  // generation special case. It's not clear that we need either of these in
  // pre-lowering CIR codegen.
  static bool opCallFnInfoOpts() { return false; }

  // ScopeOp handling
  static bool opScopeCleanupRegion() { return false; }

  // Unary operator handling
  static bool opUnaryPromotionType() { return false; }

  // SwitchOp handling
  static bool foldRangeCase() { return false; }

  // Clang early optimizations or things defered to LLVM lowering.
  static bool mayHaveIntegerOverflow() { return false; }
  static bool shouldReverseUnaryCondOnBoolExpr() { return false; }

  // RecordType
  static bool skippedLayout() { return false; }
  static bool astRecordDeclAttr() { return false; }
  static bool zeroSizeRecordMembers() { return false; }

  // Coroutines
  static bool coroEndBuiltinCall() { return false; }
  static bool emitBodyAndFallthrough() { return false; }
  static bool coroOutsideFrameMD() { return false; }
  static bool coroutineExceptions() { return false; };

  // Various handling of deferred processing in CIRGenModule.
  static bool cgmRelease() { return false; }
  static bool deferredVtables() { return false; }
  static bool deferredFuncDecls() { return false; }

  // CXXABI
  static bool cxxABI() { return false; }
  static bool cxxabiThisAlignment() { return false; }
  static bool cxxabiUseARMMethodPtrABI() { return false; }
  static bool cxxabiUseARMGuardVarABI() { return false; }
  static bool cxxabiAppleARM64CXXABI() { return false; }

  // Address class
  static bool addressOffset() { return false; }
  static bool addressIsKnownNonNull() { return false; }
  static bool addressPointerAuthInfo() { return false; }

  // Atomic
  static bool atomicExpr() { return false; }
  static bool atomicInfo() { return false; }
  static bool atomicInfoGetAtomicPointer() { return false; }
  static bool atomicInfoGetAtomicAddress() { return false; }
  static bool atomicScope() { return false; }
  static bool atomicSyncScopeID() { return false; }
  static bool atomicMapTargetSyncScope() { return false; }
  static bool atomicTypes() { return false; }
  static bool atomicUseLibCall() { return false; }
  static bool atomicMicrosoftVolatile() { return false; }
  static bool atomicOpenMP() { return false; }

  // Global ctor handling
  static bool globalCtorLexOrder() { return false; }
  static bool globalCtorAssociatedData() { return false; }

  // LowerModule handling
  static bool lowerModuleCodeGenOpts() { return false; }
  static bool lowerModuleLangOpts() { return false; }
  static bool targetLoweringInfo() { return false; }

  // Extra checks for lowerGetMethod in ItaniumCXXABI
  static bool emitCFICheck() { return false; }
  static bool emitVFEInfo() { return false; }
  static bool emitWPDInfo() { return false; }

  // Misc
  static bool aarch64SIMDIntrinsics() { return false; }
  static bool aarch64SMEIntrinsics() { return false; }
  static bool aarch64SVEIntrinsics() { return false; }
  static bool aarch64TblBuiltinExpr() { return false; }
  static bool abiArgInfo() { return false; }
  static bool addAutoInitAnnotation() { return false; }
  static bool addHeapAllocSiteMetadata() { return false; }
  static bool aggEmitFinalDestCopyRValue() { return false; }
  static bool aggValueSlot() { return false; }
  static bool aggValueSlotAlias() { return false; }
  static bool aggValueSlotDestructedFlag() { return false; }
  static bool aggValueSlotGC() { return false; }
  static bool aggValueSlotMayOverlap() { return false; }
  static bool aggValueSlotVolatile() { return false; }
  static bool alignCXXRecordDecl() { return false; }
  static bool allocToken() { return false; }
  static bool appleArm64CXXABI() { return false; }
  static bool appleKext() { return false; }
  static bool armComputeVolatileBitfields() { return false; }
  static bool asmGoto() { return false; }
  static bool asmLabelAttr() { return false; }
  static bool asmLLVMAssume() { return false; }
  static bool asmMemoryEffects() { return false; }
  static bool asmUnwindClobber() { return false; }
  static bool asmVectorType() { return false; }
  static bool assignMemcpyizer() { return false; }
  static bool astVarDeclInterface() { return false; }
  static bool attributeBuiltin() { return false; }
  static bool attributeNoBuiltin() { return false; }
  static bool bitfields() { return false; }
  static bool builtinCall() { return false; }
  static bool builtinCallF128() { return false; }
  static bool builtinCallMathErrno() { return false; }
  static bool builtinCheckKind() { return false; }
  static bool cgCapturedStmtInfo() { return false; }
  static bool countedBySize() { return false; }
  static bool cgFPOptionsRAII() { return false; }
  static bool checkBitfieldClipping() { return false; }
  static bool cirgenABIInfo() { return false; }
  static bool cleanupAfterErrorDiags() { return false; }
  static bool cleanupAppendInsts() { return false; }
  static bool cleanupBranchThrough() { return false; }
  static bool cleanupIndexAndBIAdjustment() { return false; }
  static bool cleanupWithPreservedValues() { return false; }
  static bool cleanupsToDeactivate() { return false; }
  static bool constEmitterAggILE() { return false; }
  static bool constEmitterArrayILE() { return false; }
  static bool constEmitterVectorILE() { return false; }
  static bool constantFoldSwitchStatement() { return false; }
  static bool constructABIArgDirectExtend() { return false; }
  static bool coverageMapping() { return false; }
  static bool createInvariantGroup() { return false; }
  static bool createProfileWeightsForLoop() { return false; }
  static bool ctorConstLvalueToRvalueConversion() { return false; }
  static bool ctorMemcpyizer() { return false; }
  static bool cudaSupport() { return false; }
  static bool dataLayoutTypeIsSized() { return false; }
  static bool dataLayoutTypeAllocSize() { return false; }
  static bool dataLayoutTypeStoreSize() { return false; }
  static bool dataLayoutPtrHandlingBasedOnLangAS() { return false; }
  static bool deferredCXXGlobalInit() { return false; }
  static bool deleteArray() { return false; }
  static bool devirtualizeDestructor() { return false; }
  static bool devirtualizeMemberFunction() { return false; }
  static bool dtorCleanups() { return false; }
  static bool ehCleanupActiveFlag() { return false; }
  static bool ehCleanupHasPrebranchedFallthrough() { return false; }
  static bool ehCleanupScope() { return false; }
  static bool ehCleanupScopeRequiresEHCleanup() { return false; }
  static bool ehCleanupBranchFixups() { return false; }
  static bool ehScopeFilter() { return false; }
  static bool ehstackBranches() { return false; }
  static bool emitBranchThroughCleanup() { return false; }
  static bool emitCheckedInBoundsGEP() { return false; }
  static bool emitCondLikelihoodViaExpectIntrinsic() { return false; }
  static bool emitConstrainedFPCall() { return false; }
  static bool emitLifetimeMarkers() { return false; }
  static bool emitLValueAlignmentAssumption() { return false; }
  static bool emitNullCheckForDeleteCalls() { return false; }
  static bool emitNullabilityCheck() { return false; }
  static bool emitTypeCheck() { return false; }
  static bool emitTypeMetadataCodeForVCall() { return false; }
  static bool fastMathFlags() { return false; }

  static bool fpConstraints() { return false; }
  static bool generateDebugInfo() { return false; }
  static bool getRuntimeFunctionDecl() { return false; }
  static bool globalViewIndices() { return false; }
  static bool globalViewIntLowering() { return false; }
  static bool handleBuiltinICEArguments() { return false; }
  static bool hip() { return false; }
  static bool incrementProfileCounter() { return false; }
  static bool innermostEHScope() { return false; }
  static bool insertBuiltinUnpredictable() { return false; }
  static bool instrumentation() { return false; }
  static bool intrinsicElementTypeSupport() { return false; }
  static bool intrinsics() { return false; }
  static bool isMemcpyEquivalentSpecialMember() { return false; }
  static bool isTrivialCtorOrDtor() { return false; }
  static bool lambdaCaptures() { return false; }
  static bool loopInfoStack() { return false; }
  static bool lowerAggregateLoadStore() { return false; }
  static bool lowerModeOptLevel() { return false; }
  static bool loweringPrepareX86CXXABI() { return false; }
  static bool loweringPrepareAArch64XXABI() { return false; }
  static bool makeTripleAlwaysPresent() { return false; }
  static bool maybeHandleStaticInExternC() { return false; }
  static bool mergeAllConstants() { return false; }
  static bool memberFuncPtrAuthInfo() { return false; }
  static bool memberFuncPtrCast() { return false; }
  static bool metaDataNode() { return false; }
  static bool moduleNameHash() { return false; }
  static bool msabi() { return false; }
  static bool neonSISDIntrinsics() { return false; }
  static bool nrvo() { return false; }
  static bool objCBlocks() { return false; }
  static bool objCGC() { return false; }
  static bool objCLifetime() { return false; }
  static bool hlsl() { return false; }
  static bool msvcBuiltins() { return false; }
  static bool openCL() { return false; }
  static bool openMP() { return false; }
  static bool opTBAA() { return false; }
  static bool peepholeProtection() { return false; }
  static bool pgoUse() { return false; }
  static bool pointerAuthentication() { return false; }
  static bool pointerOverflowSanitizer() { return false; }
  static bool preservedAccessIndexRegion() { return false; }
  static bool requiresCleanups() { return false; }
  static bool runCleanupsScope() { return false; }
  static bool sanitizers() { return false; }
  static bool setDLLStorageClass() { return false; }
  static bool setNonGC() { return false; }
  static bool setObjCGCLValueClass() { return false; }
  static bool setTargetAttributes() { return false; }
  static bool shouldCreateMemCpyFromGlobal() { return false; }
  static bool shouldSplitConstantStore() { return false; }
  static bool shouldUseBZeroPlusStoresToInitialize() { return false; }
  static bool shouldUseMemSetToInitialize() { return false; }
  static bool simplifyCleanupEntry() { return false; }
  static bool sourceLanguageCases() { return false; }
  static bool stackBase() { return false; }
  static bool stackSaveOp() { return false; }
  static bool stackProtector() { return false; }
  static bool targetCIRGenInfoArch() { return false; }
  static bool targetCIRGenInfoOS() { return false; }
  static bool targetCodeGenInfoGetNullPointer() { return false; }
  static bool thunks() { return false; }
  static bool tryEmitAsConstant() { return false; }
  static bool typeChecks() { return false; }
  static bool useEHCleanupForArray() { return false; }
  static bool vaArgABILowering() { return false; }
  static bool vectorConstants() { return false; }
  static bool virtualMethodAttr() { return false; }
  static bool vlas() { return false; }
  static bool vtableInitialization() { return false; }
  static bool vtableEmitMetadata() { return false; }
  static bool vtableRelativeLayout() { return false; }
  static bool weakRefReference() { return false; }
  static bool writebacks() { return false; }
  static bool msvcCXXPersonality() { return false; }
  static bool functionUsesSEHTry() { return false; }
  static bool nothrowAttr() { return false; }

  // Missing types
  static bool dataMemberType() { return false; }
  static bool matrixType() { return false; }
  static bool methodType() { return false; }
  static bool scalableVectors() { return false; }
  static bool unsizedTypes() { return false; }
  static bool vectorType() { return false; }
  static bool fixedPointType() { return false; }
  static bool stringTypeWithDifferentArraySize() { return false; }

  // Future CIR operations
  static bool awaitOp() { return false; }
  static bool callOp() { return false; }
  static bool ifOp() { return false; }
  static bool labelOp() { return false; }
  static bool ptrDiffOp() { return false; }
  static bool llvmLoweringPtrDiffConsidersPointee() { return false; }
  static bool ptrStrideOp() { return false; }
  static bool switchOp() { return false; }
  static bool throwOp() { return false; }
  static bool tryOp() { return false; }
  static bool vecTernaryOp() { return false; }
  static bool zextOp() { return false; }

  // Future CIR attributes
  static bool optInfoAttr() { return false; }

  // Maybe only needed for Windows exception handling
  static bool currentFuncletPad() { return false; }

  // Target lowering / CallConvLowering related
  static bool ABIAlignmentAttribute() { return false; }
  static bool ABIByValAttribute() { return false; }
  static bool ABIClangTypeKind() { return false; }
  static bool ABIFuncPtr() { return false; }
  static bool ABIInRegAttribute() { return false; }
  static bool ABINestedRecordLayout() { return false; }
  static bool ABINoAliasAttribute() { return false; }
  static bool ABINoProtoFunctions() { return false; }
  static bool ABIParameterCoercion() { return false; }
  static bool ABIPointerParameterAttrs() { return false; }
  static bool ABIPotentialArgAccess() { return false; }
  static bool ABITransparentUnionHandling() { return false; }
  static bool argumentPadding() { return false; }
  static bool astContextGetExternalSource() { return false; }
  static bool bitFieldPaddingDiagnostics() { return false; }
  static bool cacheRecordLayouts() { return false; }
  static bool chainCall() { return false; }
  static bool codeGenOpts() { return false; }
  static bool csmeCall() { return false; }
  static bool CUDA() { return false; }
  static bool CXXRecordDeclIsEmptyCXX11() { return false; }
  static bool CXXRecordDeclIsPOD() { return false; }
  static bool CXXRecordIsDynamicClass() { return false; }
  static bool declGetMaxAlignment() { return false; }
  static bool declHasAlignMac68kAttr() { return false; }
  static bool declHasAlignNaturalAttr() { return false; }
  static bool declHasMaxFieldAlignmentAttr() { return false; }
  static bool extParamInfo() { return false; }
  static bool fieldDeclAbstraction() { return false; }
  static bool fieldDeclGetMaxFieldAlignment() { return false; }
  static bool fieldDeclIsBitfield() { return false; }
  static bool fieldDeclIsPotentiallyOverlapping() { return false; }
  static bool fixedWidthIntegers() { return false; }
  static bool fixedSizeIntType() { return false; }
  static bool funcDeclIsCXXConstructorDecl() { return false; }
  static bool funcDeclIsCXXDestructorDecl() { return false; }
  static bool funcDeclIsCXXMethodDecl() { return false; }
  static bool funcDeclIsInlineBuiltinDeclaration() { return false; }
  static bool funcDeclIsReplaceableGlobalAllocationFunction() { return false; }
  static bool functionMemberPointerType() { return false; }
  static bool getCXXRecordBases() { return false; }
  static bool inallocaArgs() { return false; }
  static bool isCXXRecordDecl() { return false; }
  static bool isVarArg() { return false; }
  static bool langOpts() { return false; }
  static bool noFPClass() { return false; }
  static bool noReturn() { return false; }
  static bool objCIvarDecls() { return false; }
  static bool qualifiedTypes() { return false; }
  static bool qualTypeIsReferenceType() { return false; }
  static bool recordDeclHasAlignmentAttr() { return false; }
  static bool recordDeclHasFlexibleArrayMember() { return false; }
  static bool recordDeclIsCXXDecl() { return false; }
  static bool recordDeclIsMSStruct() { return false; }
  static bool recordDeclIsPacked() { return false; }
  static bool recordDeclMayInsertExtraPadding() { return false; }
  static bool setCallingConv() { return false; }
  static bool SPIRVABI() { return false; }
  static bool sretArgs() { return false; }
  static bool swift() { return false; }
  static bool tagTypeClassAbstraction() { return false; }
  static bool typeGetAsEnumType() { return false; }
  static bool typeIsCXXRecordDecl() { return false; }
  static bool X86DefaultABITypeConvertion() { return false; }
  static bool X86GetFPTypeAtOffset() { return false; }
  static bool X86TypeClassification() { return false; }
  static bool X86RetTypeClassification() { return false; }
  static bool X86ArgTypeClassification() { return false; }
  static bool fieldDeclisUnnamedBitField() { return false; }
  static bool regCall() { return false; }
  static bool recordDeclCanPassInRegisters() { return false; }

  // Additional CallConvLowering features from LowerFunction.cpp
  static bool argHasMaybeUndefAttr() { return false; }
  static bool cmseNonSecureCallAttr() { return false; }
  static bool emitEmptyRecordCheck() { return false; }
  static bool evaluationKind() { return false; }
  static bool returnValueDominatingStoreOptmiization() { return false; }
  static bool skipTempCopy() { return false; }
  static bool supportisHomogeneousAggregateQueryForAArch64() { return false; }
  static bool undef() { return false; }
  static bool varDeclIsKNRPromoted() { return false; }
  static bool volatileTypes() { return false; }
};

} // namespace cir

#endif // CLANG_CIR_MISSINGFEATURES_H
