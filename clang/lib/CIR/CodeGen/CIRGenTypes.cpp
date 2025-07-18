#include "CIRGenTypes.h"
#include "CIRGenCXXABI.h"
#include "CIRGenCall.h"
#include "CIRGenFunctionInfo.h"
#include "CIRGenModule.h"
#include "TargetInfo.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/RecordLayout.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/FnInfoOpts.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;

cir::CallingConv
CIRGenTypes::ClangCallConvToCIRCallConv(clang::CallingConv CC) {
  switch (CC) {
  case CC_C:
    return cir::CallingConv::C;
  case CC_OpenCLKernel:
    return CGM.getTargetCIRGenInfo().getOpenCLKernelCallingConv();
  case CC_SpirFunction:
    return cir::CallingConv::SpirFunction;
  default:
    llvm_unreachable("No other calling conventions implemented.");
  }
}

CIRGenTypes::CIRGenTypes(CIRGenModule &cgm)
    : astContext(cgm.getASTContext()), Builder(cgm.getBuilder()), CGM{cgm},
      Target(cgm.getTarget()), TheCXXABI(cgm.getCXXABI()),
      TheABIInfo(cgm.getTargetCIRGenInfo().getABIInfo()) {
  SkippedLayout = false;
}

CIRGenTypes::~CIRGenTypes() {
  for (llvm::FoldingSet<CIRGenFunctionInfo>::iterator I = FunctionInfos.begin(),
                                                      E = FunctionInfos.end();
       I != E;)
    delete &*I++;
}

// This is CIR's version of CIRGenTypes::addRecordTypeName
std::string CIRGenTypes::getRecordTypeName(const clang::RecordDecl *recordDecl,
                                           StringRef suffix) {
  llvm::SmallString<256> typeName;
  llvm::raw_svector_ostream outStream(typeName);

  PrintingPolicy policy = recordDecl->getASTContext().getPrintingPolicy();
  policy.SuppressInlineNamespace = false;
  policy.AlwaysIncludeTypeForTemplateArgument = true;
  policy.PrintCanonicalTypes = true;
  policy.SuppressTagKeyword = true;

  if (recordDecl->getIdentifier()) {
    astContext.getRecordType(recordDecl).print(outStream, policy);
  } else if (auto *typedefNameDecl = recordDecl->getTypedefNameForAnonDecl()) {
    typedefNameDecl->printQualifiedName(outStream, policy);
  } else {
    outStream << Builder.getUniqueAnonRecordName();
  }

  if (!suffix.empty())
    outStream << suffix;

  return Builder.getUniqueRecordName(std::string(typeName));
}

/// Return true if the specified type is already completely laid out.
bool CIRGenTypes::isRecordLayoutComplete(const Type *Ty) const {
  llvm::DenseMap<const Type *, cir::RecordType>::const_iterator I =
      recordDeclTypes.find(Ty);
  return I != recordDeclTypes.end() && I->second.isComplete();
}

static bool
isSafeToConvert(QualType T, CIRGenTypes &CGT,
                llvm::SmallPtrSet<const RecordDecl *, 16> &AlreadyChecked);

/// Return true if it is safe to convert the specified record decl to IR and lay
/// it out, false if doing so would cause us to get into a recursive compilation
/// mess.
static bool
isSafeToConvert(const RecordDecl *RD, CIRGenTypes &CGT,
                llvm::SmallPtrSet<const RecordDecl *, 16> &AlreadyChecked) {
  // If we have already checked this type (maybe the same type is used by-value
  // multiple times in multiple record fields, don't check again.
  if (!AlreadyChecked.insert(RD).second)
    return true;

  const Type *Key = CGT.getContext().getTagDeclType(RD).getTypePtr();

  // If this type is already laid out, converting it is a noop.
  if (CGT.isRecordLayoutComplete(Key))
    return true;

  // If this type is currently being laid out, we can't recursively compile it.
  if (CGT.isRecordBeingLaidOut(Key))
    return false;

  // If this type would require laying out bases that are currently being laid
  // out, don't do it.  This includes virtual base classes which get laid out
  // when a class is translated, even though they aren't embedded by-value into
  // the class.
  if (const CXXRecordDecl *CRD = dyn_cast<CXXRecordDecl>(RD)) {
    for (const auto &I : CRD->bases())
      if (!isSafeToConvert(I.getType()->castAs<RecordType>()->getDecl(), CGT,
                           AlreadyChecked))
        return false;
  }

  // If this type would require laying out members that are currently being laid
  // out, don't do it.
  for (const auto *I : RD->fields())
    if (!isSafeToConvert(I->getType(), CGT, AlreadyChecked))
      return false;

  // If there are no problems, lets do it.
  return true;
}

/// Return true if it is safe to convert this field type, which requires the
/// record elements contained by-value to all be recursively safe to convert.
static bool
isSafeToConvert(QualType T, CIRGenTypes &CGT,
                llvm::SmallPtrSet<const RecordDecl *, 16> &AlreadyChecked) {
  // Strip off atomic type sugar.
  if (const auto *AT = T->getAs<AtomicType>())
    T = AT->getValueType();

  // If this is a record, check it.
  if (const auto *RT = T->getAs<RecordType>())
    return isSafeToConvert(RT->getDecl(), CGT, AlreadyChecked);

  // If this is an array, check the elements, which are embedded inline.
  if (const auto *AT = CGT.getContext().getAsArrayType(T))
    return isSafeToConvert(AT->getElementType(), CGT, AlreadyChecked);

  // Otherwise, there is no concern about transforming this. We only care about
  // things that are contained by-value in a record that can have another
  // record as a member.
  return true;
}

// Return true if it is safe to convert the specified record decl to CIR and lay
// it out, false if doing so would cause us to get into a recursive compilation
// mess.
static bool isSafeToConvert(const RecordDecl *RD, CIRGenTypes &CGT) {
  // If no records are being laid out, we can certainly do this one.
  if (CGT.noRecordsBeingLaidOut())
    return true;

  llvm::SmallPtrSet<const RecordDecl *, 16> AlreadyChecked;
  return isSafeToConvert(RD, CGT, AlreadyChecked);
}

/// Lay out a tagged decl type like record or union.
mlir::Type CIRGenTypes::convertRecordDeclType(const clang::RecordDecl *RD) {
  // TagDecl's are not necessarily unique, instead use the (clang) type
  // connected to the decl.
  const auto *key = astContext.getTagDeclType(RD).getTypePtr();
  cir::RecordType entry = recordDeclTypes[key];

  // Handle forward decl / incomplete types.
  if (!entry) {
    auto name = getRecordTypeName(RD, "");
    entry = Builder.getIncompleteRecordTy(name, RD);
    recordDeclTypes[key] = entry;
  }

  RD = RD->getDefinition();
  if (!RD || !RD->isCompleteDefinition() || entry.isComplete())
    return entry;

  // If converting this type would cause us to infinitely loop, don't do it!
  if (!isSafeToConvert(RD, *this)) {
    DeferredRecords.push_back(RD);
    return entry;
  }

  // Okay, this is a definition of a type. Compile the implementation now.
  bool InsertResult = RecordsBeingLaidOut.insert(key).second;
  (void)InsertResult;
  assert(InsertResult && "Recursively compiling a record?");

  // Force conversion of non-virtual base classes recursively.
  if (const auto *cxxRecordDecl = dyn_cast<CXXRecordDecl>(RD)) {
    for (const auto &I : cxxRecordDecl->bases()) {
      if (I.isVirtual())
        continue;
      convertRecordDeclType(I.getType()->castAs<RecordType>()->getDecl());
    }
  }

  // Layout fields.
  std::unique_ptr<CIRGenRecordLayout> Layout = computeRecordLayout(RD, &entry);
  recordDeclTypes[key] = entry;
  CIRGenRecordLayouts[key] = std::move(Layout);

  // We're done laying out this record.
  bool EraseResult = RecordsBeingLaidOut.erase(key);
  (void)EraseResult;
  assert(EraseResult && "record not in RecordsBeingLaidOut set?");

  // If this record blocked a FunctionType conversion, then recompute whatever
  // was derived from that.
  // FIXME: This is hugely overconservative.
  if (SkippedLayout)
    TypeCache.clear();

  // If we're done converting the outer-most record, then convert any deferred
  // records as well.
  if (RecordsBeingLaidOut.empty())
    while (!DeferredRecords.empty())
      convertRecordDeclType(DeferredRecords.pop_back_val());

  return entry;
}

mlir::Type CIRGenTypes::convertTypeForMem(clang::QualType qualType,
                                          bool forBitField) {
  assert(!qualType->isConstantMatrixType() && "Matrix types NYI");

  mlir::Type convertedType = convertType(qualType);

  assert(!forBitField && "Bit fields NYI");

  // If this is a bit-precise integer type in a bitfield representation, map
  // this integer to the target-specified size.
  if (forBitField && qualType->isBitIntType())
    assert(!qualType->isBitIntType() && "Bit field with type _BitInt NYI");

  return convertedType;
}

mlir::MLIRContext &CIRGenTypes::getMLIRContext() const {
  return *Builder.getContext();
}

mlir::Type CIRGenTypes::convertFunctionTypeInternal(QualType QFT) {
  assert(QFT.isCanonical());
  const Type *Ty = QFT.getTypePtr();
  const FunctionType *FT = cast<FunctionType>(QFT.getTypePtr());
  // First, check whether we can build the full function type. If the function
  // type depends on an incomplete type (e.g. a record or enum), we cannot lower
  // the function type.
  assert(isFuncTypeConvertible(FT) && "NYI");

  // The function type can be built; call the appropriate routines to build it
  const CIRGenFunctionInfo *FI;
  if (const auto *FPT = dyn_cast<FunctionProtoType>(FT)) {
    FI = &arrangeFreeFunctionType(
        CanQual<FunctionProtoType>::CreateUnsafe(QualType(FPT, 0)));
  } else {
    const FunctionNoProtoType *FNPT = cast<FunctionNoProtoType>(FT);
    FI = &arrangeFreeFunctionType(
        CanQual<FunctionNoProtoType>::CreateUnsafe(QualType(FNPT, 0)));
  }

  mlir::Type ResultType = nullptr;
  // If there is something higher level prodding our CIRGenFunctionInfo, then
  // don't recurse into it again.
  assert(!FunctionsBeingProcessed.count(FI) && "NYI");

  // Otherwise, we're good to go, go ahead and convert it.
  ResultType = GetFunctionType(*FI);

  RecordsBeingLaidOut.erase(Ty);

  assert(!SkippedLayout && "Shouldn't have skipped anything yet");

  if (RecordsBeingLaidOut.empty())
    while (!DeferredRecords.empty())
      convertRecordDeclType(DeferredRecords.pop_back_val());

  return ResultType;
}

/// Return true if the specified type in a function parameter or result position
/// can be converted to a CIR type at this point. This boils down to being
/// whether it is complete, as well as whether we've temporarily deferred
/// expanding the type because we're in a recursive context.
bool CIRGenTypes::isFuncParamTypeConvertible(clang::QualType Ty) {
  // Some ABIs cannot have their member pointers represented in LLVM IR unless
  // certain circumstances have been reached.
  if (const auto *mpt = Ty->getAs<MemberPointerType>())
    return getCXXABI().isMemberPointerConvertible(mpt);

  // If this isn't a tagged type, we can convert it!
  const TagType *TT = Ty->getAs<TagType>();
  if (!TT)
    return true;

  // Incomplete types cannot be converted.
  return !TT->isIncompleteType();
}

/// Code to verify a given function type is complete, i.e. the return type and
/// all of the parameter types are complete. Also check to see if we are in a
/// RS_RecordPointer context, and if so whether any record types have been
/// pended. If so, we don't want to ask the ABI lowering code to handle a type
/// that cannot be converted to a CIR type.
bool CIRGenTypes::isFuncTypeConvertible(const FunctionType *FT) {
  if (!isFuncParamTypeConvertible(FT->getReturnType()))
    return false;

  if (const auto *FPT = dyn_cast<FunctionProtoType>(FT))
    for (unsigned i = 0, e = FPT->getNumParams(); i != e; i++)
      if (!isFuncParamTypeConvertible(FPT->getParamType(i)))
        return false;

  return true;
}

/// convertType - Convert the specified type to its MLIR form.
mlir::Type CIRGenTypes::convertType(QualType T) {
  T = astContext.getCanonicalType(T);
  const Type *Ty = T.getTypePtr();

  // For the device-side compilation, CUDA device builtin surface/texture types
  // may be represented in different types.
  // NOTE: CUDAIsDevice is true when building also HIP code.
  //  1. There is no SurfaceType on HIP,
  //  2. There is Texture memory on HIP but accessing the memory goes through
  //  calls to the runtime. e.g. for a 2D: `tex2D<float>(tex, x, y);`
  if (astContext.getLangOpts().CUDAIsDevice) {
    if (T->isCUDADeviceBuiltinSurfaceType()) {
      if (mlir::Type Ty =
              CGM.getTargetCIRGenInfo().getCUDADeviceBuiltinSurfaceDeviceType())
        return Ty;
      llvm_unreachable("NYI");
    } else if (T->isCUDADeviceBuiltinTextureType()) {
      if (mlir::Type Ty =
              CGM.getTargetCIRGenInfo().getCUDADeviceBuiltinTextureDeviceType())
        return Ty;
      llvm_unreachable("NYI");
    }
  }

  if (const auto *recordType = dyn_cast<RecordType>(T))
    return convertRecordDeclType(recordType->getDecl());

  // See if type is already cached.
  TypeCacheTy::iterator TCI = TypeCache.find(Ty);
  // If type is found in map then use it. Otherwise, convert type T.
  if (TCI != TypeCache.end())
    return TCI->second;

  // If we don't have it in the cache, convert it now.
  mlir::Type ResultType = nullptr;
  switch (Ty->getTypeClass()) {
  case Type::Record: // Handled above.
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.inc"
    llvm_unreachable("Non-canonical or dependent types aren't possible.");

  case Type::ArrayParameter:
  case Type::HLSLAttributedResource:
    llvm_unreachable("NYI");

  case Type::Builtin: {
    switch (cast<BuiltinType>(Ty)->getKind()) {
    case BuiltinType::HLSLResource:
      llvm_unreachable("NYI");
    case BuiltinType::SveMFloat8:
    case BuiltinType::SveMFloat8x2:
    case BuiltinType::SveMFloat8x3:
    case BuiltinType::SveMFloat8x4:
    case BuiltinType::MFloat8:
    case BuiltinType::SveBoolx2:
    case BuiltinType::SveBoolx4:
    case BuiltinType::SveCount:
      llvm_unreachable("NYI");
    case BuiltinType::Void:
      // TODO(cir): how should we model this?
      ResultType = CGM.VoidTy;
      break;

    case BuiltinType::ObjCId:
    case BuiltinType::ObjCClass:
    case BuiltinType::ObjCSel:
      // TODO(cir): probably same as BuiltinType::Void
      assert(0 && "not implemented");
      break;

    case BuiltinType::Bool:
      ResultType = cir::BoolType::get(&getMLIRContext());
      break;

    // Signed types.
    case BuiltinType::Accum:
    case BuiltinType::Char_S:
    case BuiltinType::Fract:
    case BuiltinType::Int:
    case BuiltinType::Long:
    case BuiltinType::LongAccum:
    case BuiltinType::LongFract:
    case BuiltinType::LongLong:
    case BuiltinType::SChar:
    case BuiltinType::Short:
    case BuiltinType::ShortAccum:
    case BuiltinType::ShortFract:
    case BuiltinType::WChar_S:
    // Saturated signed types.
    case BuiltinType::SatAccum:
    case BuiltinType::SatFract:
    case BuiltinType::SatLongAccum:
    case BuiltinType::SatLongFract:
    case BuiltinType::SatShortAccum:
    case BuiltinType::SatShortFract:
      ResultType =
          cir::IntType::get(&getMLIRContext(), astContext.getTypeSize(T),
                            /*isSigned=*/true);
      break;
    // Unsigned types.
    case BuiltinType::Char16:
    case BuiltinType::Char32:
    case BuiltinType::Char8:
    case BuiltinType::Char_U:
    case BuiltinType::UAccum:
    case BuiltinType::UChar:
    case BuiltinType::UFract:
    case BuiltinType::UInt:
    case BuiltinType::ULong:
    case BuiltinType::ULongAccum:
    case BuiltinType::ULongFract:
    case BuiltinType::ULongLong:
    case BuiltinType::UShort:
    case BuiltinType::UShortAccum:
    case BuiltinType::UShortFract:
    case BuiltinType::WChar_U:
    // Saturated unsigned types.
    case BuiltinType::SatUAccum:
    case BuiltinType::SatUFract:
    case BuiltinType::SatULongAccum:
    case BuiltinType::SatULongFract:
    case BuiltinType::SatUShortAccum:
    case BuiltinType::SatUShortFract:
      ResultType =
          cir::IntType::get(Builder.getContext(), astContext.getTypeSize(T),
                            /*isSigned=*/false);
      break;

    case BuiltinType::Float16:
      ResultType = CGM.FP16Ty;
      break;
    case BuiltinType::Half:
      if (astContext.getLangOpts().NativeHalfType ||
          !astContext.getTargetInfo().useFP16ConversionIntrinsics())
        ResultType = CGM.FP16Ty;
      else
        llvm_unreachable("NYI");
      break;
    case BuiltinType::BFloat16:
      ResultType = CGM.BFloat16Ty;
      break;
    case BuiltinType::Float:
      ResultType = CGM.FloatTy;
      break;
    case BuiltinType::Double:
      ResultType = CGM.DoubleTy;
      break;
    case BuiltinType::LongDouble:
      ResultType = Builder.getLongDoubleTy(astContext.getFloatTypeSemantics(T));
      break;
    case BuiltinType::Float128:
      ResultType = CGM.FP128Ty;
      break;
    case BuiltinType::Ibm128:
      // FIXME: look at astContext.getFloatTypeSemantics(T) and getTypeForFormat
      // on LLVM codegen.
      assert(0 && "not implemented");
      break;

    case BuiltinType::NullPtr:
      // Add proper CIR type for it? this looks mostly useful for sema related
      // things (like for overloads accepting void), for now, given that
      // `sizeof(std::nullptr_t)` is equal to `sizeof(void *)`, model
      // std::nullptr_t as !cir.ptr<!void>
      ResultType = Builder.getVoidPtrTy();
      break;

    case BuiltinType::UInt128:
      ResultType = CGM.UInt128Ty;
      break;
    case BuiltinType::Int128:
      ResultType = CGM.SInt128Ty;
      break;

#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix)                   \
  case BuiltinType::Id:
#include "clang/Basic/OpenCLImageTypes.def"
#define EXT_OPAQUE_TYPE(ExtType, Id, Ext) case BuiltinType::Id:
#include "clang/Basic/OpenCLExtensionTypes.def"
    case BuiltinType::OCLSampler:
    case BuiltinType::OCLEvent:
    case BuiltinType::OCLClkEvent:
    case BuiltinType::OCLQueue:
    case BuiltinType::OCLReserveID:
      assert(0 && "not implemented");
      break;
    case BuiltinType::SveInt8:
    case BuiltinType::SveUint8:
    case BuiltinType::SveInt8x2:
    case BuiltinType::SveUint8x2:
    case BuiltinType::SveInt8x3:
    case BuiltinType::SveUint8x3:
    case BuiltinType::SveInt8x4:
    case BuiltinType::SveUint8x4:
    case BuiltinType::SveInt16:
    case BuiltinType::SveUint16:
    case BuiltinType::SveInt16x2:
    case BuiltinType::SveUint16x2:
    case BuiltinType::SveInt16x3:
    case BuiltinType::SveUint16x3:
    case BuiltinType::SveInt16x4:
    case BuiltinType::SveUint16x4:
    case BuiltinType::SveInt32:
    case BuiltinType::SveUint32:
    case BuiltinType::SveInt32x2:
    case BuiltinType::SveUint32x2:
    case BuiltinType::SveInt32x3:
    case BuiltinType::SveUint32x3:
    case BuiltinType::SveInt32x4:
    case BuiltinType::SveUint32x4:
    case BuiltinType::SveInt64:
    case BuiltinType::SveUint64:
    case BuiltinType::SveInt64x2:
    case BuiltinType::SveUint64x2:
    case BuiltinType::SveInt64x3:
    case BuiltinType::SveUint64x3:
    case BuiltinType::SveInt64x4:
    case BuiltinType::SveUint64x4:
    case BuiltinType::SveBool:
    case BuiltinType::SveFloat16:
    case BuiltinType::SveFloat16x2:
    case BuiltinType::SveFloat16x3:
    case BuiltinType::SveFloat16x4:
    case BuiltinType::SveFloat32:
    case BuiltinType::SveFloat32x2:
    case BuiltinType::SveFloat32x3:
    case BuiltinType::SveFloat32x4:
    case BuiltinType::SveFloat64:
    case BuiltinType::SveFloat64x2:
    case BuiltinType::SveFloat64x3:
    case BuiltinType::SveFloat64x4:
    case BuiltinType::SveBFloat16:
    case BuiltinType::SveBFloat16x2:
    case BuiltinType::SveBFloat16x3:
    case BuiltinType::SveBFloat16x4: {
      assert(0 && "not implemented");
      break;
    }
#define PPC_VECTOR_TYPE(Name, Id, Size)                                        \
  case BuiltinType::Id:                                                        \
    assert(0 && "not implemented");                                            \
    break;
#include "clang/Basic/PPCTypes.def"
#define RVV_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/RISCVVTypes.def"
      {
        assert(0 && "not implemented");
        break;
      }
#define WASM_REF_TYPE(Name, MangledName, Id, SingletonId, AS)                  \
  case BuiltinType::Id: {                                                      \
    llvm_unreachable("NYI");                                                   \
  } break;
#include "clang/Basic/WebAssemblyReferenceTypes.def"
#define AMDGPU_TYPE(Name, Id, SingletonId, Width, Align)                       \
  case BuiltinType::Id:                                                        \
    llvm_unreachable("NYI");
#include "clang/Basic/AMDGPUTypes.def"
    case BuiltinType::Dependent:
#define BUILTIN_TYPE(Id, SingletonId)
#define PLACEHOLDER_TYPE(Id, SingletonId) case BuiltinType::Id:
#include "clang/AST/BuiltinTypes.def"
      llvm_unreachable("Unexpected placeholder builtin type!");
    }
    break;
  }
  case Type::Auto:
  case Type::DeducedTemplateSpecialization:
    llvm_unreachable("Unexpected undeduced type!");
  case Type::Complex: {
    const ComplexType *CT = cast<ComplexType>(Ty);
    auto ElementTy = convertType(CT->getElementType());
    ResultType = cir::ComplexType::get(ElementTy);
    break;
  }
  case Type::LValueReference:
  case Type::RValueReference: {
    const ReferenceType *RTy = cast<ReferenceType>(Ty);
    QualType ETy = RTy->getPointeeType();
    auto PointeeType = convertTypeForMem(ETy);
    ResultType = Builder.getPointerTo(PointeeType, ETy.getAddressSpace());
    assert(ResultType && "Cannot get pointer type?");
    break;
  }
  case Type::Pointer: {
    const PointerType *PTy = cast<PointerType>(Ty);
    QualType ETy = PTy->getPointeeType();
    assert(!ETy->isConstantMatrixType() && "not implemented");

    mlir::Type PointeeType = convertType(ETy);

    // Treat effectively as a *i8.
    // if (PointeeType->isVoidTy())
    //  PointeeType = Builder.getI8Type();

    ResultType = Builder.getPointerTo(PointeeType, ETy.getAddressSpace());
    assert(ResultType && "Cannot get pointer type?");
    break;
  }

  case Type::VariableArray: {
    const VariableArrayType *A = cast<VariableArrayType>(Ty);
    assert(A->getIndexTypeCVRQualifiers() == 0 &&
           "FIXME: We only handle trivial array types so far!");
    // VLAs resolve to the innermost element type; this matches
    // the return of alloca, and there isn't any obviously better choice.
    ResultType = convertTypeForMem(A->getElementType());
    break;
  }
  case Type::IncompleteArray: {
    const IncompleteArrayType *A = cast<IncompleteArrayType>(Ty);
    assert(A->getIndexTypeCVRQualifiers() == 0 &&
           "FIXME: We only handle trivial array types so far!");
    // int X[] -> [0 x int], unless the element type is not sized.  If it is
    // unsized (e.g. an incomplete record) just use [0 x i8].
    ResultType = convertTypeForMem(A->getElementType());
    if (!cir::isSized(ResultType)) {
      SkippedLayout = true;
      ResultType = Builder.getUInt8Ty();
    }
    ResultType = cir::ArrayType::get(ResultType, 0);
    break;
  }
  case Type::ConstantArray: {
    const ConstantArrayType *A = cast<ConstantArrayType>(Ty);
    auto EltTy = convertTypeForMem(A->getElementType());

    // FIXME: In LLVM, "lower arrays of undefined struct type to arrays of
    // i8 just to have a concrete type". Not sure this makes sense in CIR yet.
    assert(cir::isSized(EltTy) && "not implemented");
    ResultType = cir::ArrayType::get(EltTy, A->getSize().getZExtValue());
    break;
  }
  case Type::ExtVector:
  case Type::Vector: {
    const VectorType *V = cast<VectorType>(Ty);
    auto ElementType = convertTypeForMem(V->getElementType());
    ResultType = cir::VectorType::get(ElementType, V->getNumElements());
    break;
  }
  case Type::ConstantMatrix: {
    assert(0 && "not implemented");
    break;
  }
  case Type::FunctionNoProto:
  case Type::FunctionProto:
    ResultType = convertFunctionTypeInternal(T);
    break;
  case Type::ObjCObject:
    assert(0 && "not implemented");
    break;

  case Type::ObjCInterface: {
    assert(0 && "not implemented");
    break;
  }

  case Type::ObjCObjectPointer: {
    assert(0 && "not implemented");
    break;
  }

  case Type::Enum: {
    const EnumDecl *ED = cast<EnumType>(Ty)->getDecl();
    if (ED->isCompleteDefinition() || ED->isFixed())
      return convertType(ED->getIntegerType());
    // Return a placeholder 'i32' type.  This can be changed later when the
    // type is defined (see UpdateCompletedType), but is likely to be the
    // "right" answer.
    ResultType = CGM.UInt32Ty;
    break;
  }

  case Type::BlockPointer: {
    assert(0 && "not implemented");
    break;
  }

  case Type::MemberPointer: {
    const auto *MPT = cast<MemberPointerType>(Ty);

    auto memberTy = convertType(MPT->getPointeeType());
    auto clsTy =
        mlir::cast<cir::RecordType>(convertType(QualType(MPT->getClass(), 0)));
    if (MPT->isMemberDataPointer())
      ResultType = cir::DataMemberType::get(memberTy, clsTy);
    else {
      auto memberFuncTy = mlir::cast<cir::FuncType>(memberTy);
      ResultType = cir::MethodType::get(memberFuncTy, clsTy);
    }
    break;
  }

  case Type::Atomic: {
    QualType valueType = cast<AtomicType>(Ty)->getValueType();
    ResultType = convertTypeForMem(valueType);

    // Pad out to the inflated size if necessary.
    uint64_t valueSize = astContext.getTypeSize(valueType);
    uint64_t atomicSize = astContext.getTypeSize(Ty);
    if (valueSize != atomicSize) {
      llvm_unreachable("NYI");
    }
    break;
  }
  case Type::Pipe: {
    assert(0 && "not implemented");
    break;
  }
  case Type::BitInt: {
    const auto *bitIntTy = cast<BitIntType>(Ty);
    ResultType = cir::IntType::get(Builder.getContext(), bitIntTy->getNumBits(),
                                   bitIntTy->isSigned());
    break;
  }
  }

  assert(ResultType && "Didn't convert a type?");

  TypeCache[Ty] = ResultType;
  return ResultType;
}

const CIRGenFunctionInfo &CIRGenTypes::arrangeCIRFunctionInfo(
    CanQualType resultType, cir::FnInfoOpts opts,
    llvm::ArrayRef<CanQualType> argTypes, FunctionType::ExtInfo info,
    llvm::ArrayRef<FunctionProtoType::ExtParameterInfo> paramInfos,
    RequiredArgs required) {
  assert(llvm::all_of(argTypes,
                      [](CanQualType T) { return T.isCanonicalAsParam(); }));
  bool instanceMethod = opts == cir::FnInfoOpts::IsInstanceMethod;
  bool chainCall = opts == cir::FnInfoOpts::IsChainCall;

  // Lookup or create unique function info.
  llvm::FoldingSetNodeID ID;
  CIRGenFunctionInfo::Profile(ID, instanceMethod, chainCall, info, paramInfos,
                              required, resultType, argTypes);

  void *insertPos = nullptr;
  CIRGenFunctionInfo *FI = FunctionInfos.FindNodeOrInsertPos(ID, insertPos);
  if (FI)
    return *FI;

  cir::CallingConv CC = ClangCallConvToCIRCallConv(info.getCC());

  // Construction the function info. We co-allocate the ArgInfos.
  FI = CIRGenFunctionInfo::create(CC, instanceMethod, chainCall, info,
                                  paramInfos, resultType, argTypes, required);
  FunctionInfos.InsertNode(FI, insertPos);

  bool inserted = FunctionsBeingProcessed.insert(FI).second;
  (void)inserted;
  assert(inserted && "Recursively being processed?");

  bool erased = FunctionsBeingProcessed.erase(FI);
  (void)erased;
  assert(erased && "Not in set?");

  return *FI;
}

const CIRGenFunctionInfo &CIRGenTypes::arrangeGlobalDeclaration(GlobalDecl GD) {
  assert(!dyn_cast<ObjCMethodDecl>(GD.getDecl()) &&
         "This is reported as a FIXME in LLVM codegen");
  const auto *FD = cast<FunctionDecl>(GD.getDecl());

  if (isa<CXXConstructorDecl>(GD.getDecl()) ||
      isa<CXXDestructorDecl>(GD.getDecl()))
    return arrangeCXXStructorDeclaration(GD);

  return arrangeFunctionDeclaration(FD);
}

// When we find the full definition for a TagDecl, replace the 'opaque' type we
// previously made for it if applicable.
void CIRGenTypes::UpdateCompletedType(const TagDecl *TD) {
  // If this is an enum being completed, then we flush all non-struct types
  // from the cache. This allows function types and other things that may be
  // derived from the enum to be recomputed.
  if (const auto *ED = dyn_cast<EnumDecl>(TD)) {
    // Classic codegen clears the type cache if it contains an entry for this
    // enum type that doesn't use i32 as the underlying type, but I can't find
    // a test case that meets that condition. C++ doesn't allow forward
    // declaration of enums, and C doesn't allow an incomplete forward
    // declaration with a non-default type.
    assert(
        !TypeCache.count(ED->getTypeForDecl()) ||
        (convertType(ED->getIntegerType()) == TypeCache[ED->getTypeForDecl()]));
    // If necessary, provide the full definition of a type only used with a
    // declaration so far.
    assert(!cir::MissingFeatures::generateDebugInfo());
    return;
  }

  // If we completed a RecordDecl that we previously used and converted to an
  // anonymous type, then go ahead and complete it now.
  const auto *RD = cast<RecordDecl>(TD);
  if (RD->isDependentType())
    return;

  // Only complete if we converted it already. If we haven't converted it yet,
  // we'll just do it lazily.
  if (recordDeclTypes.count(astContext.getTagDeclType(RD).getTypePtr()))
    convertRecordDeclType(RD);

  // If necessary, provide the full definition of a type only used with a
  // declaration so far.
  if (CGM.getModuleDebugInfo())
    llvm_unreachable("NYI");
}

/// Return record layout info for the given record decl.
const CIRGenRecordLayout &
CIRGenTypes::getCIRGenRecordLayout(const RecordDecl *RD) {
  const auto *Key = astContext.getTagDeclType(RD).getTypePtr();

  auto I = CIRGenRecordLayouts.find(Key);
  if (I != CIRGenRecordLayouts.end())
    return *I->second;

  // Compute the type information.
  convertRecordDeclType(RD);

  // Now try again.
  I = CIRGenRecordLayouts.find(Key);

  assert(I != CIRGenRecordLayouts.end() &&
         "Unable to find record layout information for type");
  return *I->second;
}

bool CIRGenTypes::isPointerZeroInitializable(clang::QualType T) {
  assert((T->isAnyPointerType() || T->isBlockPointerType()) && "Invalid type");
  return isZeroInitializable(T);
}

bool CIRGenTypes::isZeroInitializable(QualType T) {
  if (T->getAs<PointerType>())
    return astContext.getTargetNullPointerValue(T) == 0;

  if (const auto *AT = astContext.getAsArrayType(T)) {
    if (isa<IncompleteArrayType>(AT))
      return true;
    if (const auto *CAT = dyn_cast<ConstantArrayType>(AT))
      if (astContext.getConstantArrayElementCount(CAT) == 0)
        return true;
    T = astContext.getBaseElementType(T);
  }

  // Records are non-zero-initializable if they contain any
  // non-zero-initializable subobjects.
  if (const RecordType *RT = T->getAs<RecordType>()) {
    const RecordDecl *RD = RT->getDecl();
    return isZeroInitializable(RD);
  }

  // We have to ask the ABI about member pointers.
  if (const MemberPointerType *MPT = T->getAs<MemberPointerType>())
    return TheCXXABI.isZeroInitializable(MPT);

  // Everything else is okay.
  return true;
}

bool CIRGenTypes::isZeroInitializable(const RecordDecl *RD) {
  return getCIRGenRecordLayout(RD).isZeroInitializable();
}
