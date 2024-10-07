#include "CIRGenTypes.h"
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
using namespace cir;

mlir::cir::CallingConv
CIRGenTypes::ClangCallConvToCIRCallConv(clang::CallingConv cc) {
  switch (cc) {
  case CC_C:
    return mlir::cir::CallingConv::C;
  case CC_OpenCLKernel:
    return CGM.getTargetCIRGenInfo().getOpenCLKernelCallingConv();
  case CC_SpirFunction:
    return mlir::cir::CallingConv::SpirFunction;
  default:
    llvm_unreachable("No other calling conventions implemented.");
  }
}

CIRGenTypes::CIRGenTypes(CIRGenModule &cgm)
    : Context(cgm.getASTContext()), Builder(cgm.getBuilder()), CGM{cgm},
      Target(cgm.getTarget()), TheCXXABI(cgm.getCXXABI()),
      TheABIInfo(cgm.getTargetCIRGenInfo().getABIInfo()) {
  SkippedLayout = false;
}

CIRGenTypes::~CIRGenTypes() {
  for (llvm::FoldingSet<CIRGenFunctionInfo>::iterator i = FunctionInfos.begin(),
                                                      e = FunctionInfos.end();
       i != e;)
    delete &*i++;
}

// This is CIR's version of CIRGenTypes::addRecordTypeName
std::string CIRGenTypes::getRecordTypeName(const clang::RecordDecl *recordDecl,
                                           StringRef suffix) {
  llvm::SmallString<256> typeName;
  llvm::raw_svector_ostream outStream(typeName);

  PrintingPolicy policy = recordDecl->getASTContext().getPrintingPolicy();
  policy.SuppressInlineNamespace = false;

  if (recordDecl->getIdentifier()) {
    if (recordDecl->getDeclContext())
      recordDecl->printQualifiedName(outStream, policy);
    else
      recordDecl->printName(outStream, policy);

    // Ensure each template specialization has a unique name.
    if (auto *templateSpecialization =
            llvm::dyn_cast<ClassTemplateSpecializationDecl>(recordDecl)) {
      outStream << '<';
      const auto args = templateSpecialization->getTemplateArgs().asArray();
      const auto printer = [&policy, &outStream](const TemplateArgument &arg) {
        /// Print this template argument to the given output stream.
        arg.print(policy, outStream, /*IncludeType=*/true);
      };
      llvm::interleaveComma(args, outStream, printer);
      outStream << '>';
    }

  } else if (auto *typedefNameDecl = recordDecl->getTypedefNameForAnonDecl()) {
    if (typedefNameDecl->getDeclContext())
      typedefNameDecl->printQualifiedName(outStream, policy);
    else
      typedefNameDecl->printName(outStream);
  } else {
    outStream << Builder.getUniqueAnonRecordName();
  }

  if (!suffix.empty())
    outStream << suffix;

  return Builder.getUniqueRecordName(std::string(typeName));
}

/// Return true if the specified type is already completely laid out.
bool CIRGenTypes::isRecordLayoutComplete(const Type *ty) const {
  llvm::DenseMap<const Type *, mlir::cir::StructType>::const_iterator i =
      recordDeclTypes.find(ty);
  return i != recordDeclTypes.end() && i->second.isComplete();
}

static bool
isSafeToConvert(QualType t, CIRGenTypes &cgt,
                llvm::SmallPtrSet<const RecordDecl *, 16> &alreadyChecked);

/// Return true if it is safe to convert the specified record decl to IR and lay
/// it out, false if doing so would cause us to get into a recursive compilation
/// mess.
static bool
isSafeToConvert(const RecordDecl *rd, CIRGenTypes &cgt,
                llvm::SmallPtrSet<const RecordDecl *, 16> &alreadyChecked) {
  // If we have already checked this type (maybe the same type is used by-value
  // multiple times in multiple structure fields, don't check again.
  if (!alreadyChecked.insert(rd).second)
    return true;

  const Type *key = cgt.getContext().getTagDeclType(rd).getTypePtr();

  // If this type is already laid out, converting it is a noop.
  if (cgt.isRecordLayoutComplete(key))
    return true;

  // If this type is currently being laid out, we can't recursively compile it.
  if (cgt.isRecordBeingLaidOut(key))
    return false;

  // If this type would require laying out bases that are currently being laid
  // out, don't do it.  This includes virtual base classes which get laid out
  // when a class is translated, even though they aren't embedded by-value into
  // the class.
  if (const CXXRecordDecl *crd = dyn_cast<CXXRecordDecl>(rd)) {
    for (const auto &i : crd->bases())
      if (!isSafeToConvert(i.getType()->castAs<RecordType>()->getDecl(), cgt,
                           alreadyChecked))
        return false;
  }

  // If this type would require laying out members that are currently being laid
  // out, don't do it.
  for (const auto *i : rd->fields())
    if (!isSafeToConvert(i->getType(), cgt, alreadyChecked))
      return false;

  // If there are no problems, lets do it.
  return true;
}

/// Return true if it is safe to convert this field type, which requires the
/// structure elements contained by-value to all be recursively safe to convert.
static bool
isSafeToConvert(QualType t, CIRGenTypes &cgt,
                llvm::SmallPtrSet<const RecordDecl *, 16> &alreadyChecked) {
  // Strip off atomic type sugar.
  if (const auto *at = t->getAs<AtomicType>())
    t = at->getValueType();

  // If this is a record, check it.
  if (const auto *rt = t->getAs<RecordType>())
    return isSafeToConvert(rt->getDecl(), cgt, alreadyChecked);

  // If this is an array, check the elements, which are embedded inline.
  if (const auto *at = cgt.getContext().getAsArrayType(t))
    return isSafeToConvert(at->getElementType(), cgt, alreadyChecked);

  // Otherwise, there is no concern about transforming this. We only care about
  // things that are contained by-value in a structure that can have another
  // structure as a member.
  return true;
}

// Return true if it is safe to convert the specified record decl to CIR and lay
// it out, false if doing so would cause us to get into a recursive compilation
// mess.
static bool isSafeToConvert(const RecordDecl *rd, CIRGenTypes &cgt) {
  // If no structs are being laid out, we can certainly do this one.
  if (cgt.noRecordsBeingLaidOut())
    return true;

  llvm::SmallPtrSet<const RecordDecl *, 16> alreadyChecked;
  return isSafeToConvert(rd, cgt, alreadyChecked);
}

/// Lay out a tagged decl type like struct or union.
mlir::Type CIRGenTypes::convertRecordDeclType(const clang::RecordDecl *rd) {
  // TagDecl's are not necessarily unique, instead use the (clang) type
  // connected to the decl.
  const auto *key = Context.getTagDeclType(rd).getTypePtr();
  mlir::cir::StructType entry = recordDeclTypes[key];

  // Handle forward decl / incomplete types.
  if (!entry) {
    auto name = getRecordTypeName(rd, "");
    entry = Builder.getIncompleteStructTy(name, rd);
    recordDeclTypes[key] = entry;
  }

  rd = rd->getDefinition();
  if (!rd || !rd->isCompleteDefinition() || entry.isComplete())
    return entry;

  // If converting this type would cause us to infinitely loop, don't do it!
  if (!isSafeToConvert(rd, *this)) {
    DeferredRecords.push_back(rd);
    return entry;
  }

  // Okay, this is a definition of a type. Compile the implementation now.
  bool insertResult = RecordsBeingLaidOut.insert(key).second;
  (void)insertResult;
  assert(insertResult && "Recursively compiling a struct?");

  // Force conversion of non-virtual base classes recursively.
  if (const auto *cxxRecordDecl = dyn_cast<CXXRecordDecl>(rd)) {
    for (const auto &i : cxxRecordDecl->bases()) {
      if (i.isVirtual())
        continue;
      convertRecordDeclType(i.getType()->castAs<RecordType>()->getDecl());
    }
  }

  // Layout fields.
  std::unique_ptr<CIRGenRecordLayout> layout = computeRecordLayout(rd, &entry);
  recordDeclTypes[key] = entry;
  CIRGenRecordLayouts[key] = std::move(layout);

  // We're done laying out this struct.
  bool eraseResult = RecordsBeingLaidOut.erase(key);
  (void)eraseResult;
  assert(eraseResult && "struct not in RecordsBeingLaidOut set?");

  // If this struct blocked a FunctionType conversion, then recompute whatever
  // was derived from that.
  // FIXME: This is hugely overconservative.
  if (SkippedLayout)
    TypeCache.clear();

  // If we're done converting the outer-most record, then convert any deferred
  // structs as well.
  if (RecordsBeingLaidOut.empty())
    while (!DeferredRecords.empty())
      convertRecordDeclType(DeferredRecords.pop_back_val());

  return entry;
}

mlir::Type CIRGenTypes::convertTypeForMem(clang::QualType qualType,
                                          bool forBitField) {
  assert(!qualType->isConstantMatrixType() && "Matrix types NYI");

  mlir::Type convertedType = ConvertType(qualType);

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

mlir::Type CIRGenTypes::ConvertFunctionTypeInternal(QualType qft) {
  assert(qft.isCanonical());
  const Type *ty = qft.getTypePtr();
  const FunctionType *ft = cast<FunctionType>(qft.getTypePtr());
  // First, check whether we can build the full fucntion type. If the function
  // type depends on an incomplete type (e.g. a struct or enum), we cannot lower
  // the function type.
  assert(isFuncTypeConvertible(ft) && "NYI");

  // The function type can be built; call the appropriate routines to build it
  const CIRGenFunctionInfo *fi;
  if (const auto *fpt = dyn_cast<FunctionProtoType>(ft)) {
    fi = &arrangeFreeFunctionType(
        CanQual<FunctionProtoType>::CreateUnsafe(QualType(fpt, 0)));
  } else {
    const FunctionNoProtoType *fnpt = cast<FunctionNoProtoType>(ft);
    fi = &arrangeFreeFunctionType(
        CanQual<FunctionNoProtoType>::CreateUnsafe(QualType(fnpt, 0)));
  }

  mlir::Type resultType = nullptr;
  // If there is something higher level prodding our CIRGenFunctionInfo, then
  // don't recurse into it again.
  assert(!FunctionsBeingProcessed.count(fi) && "NYI");

  // Otherwise, we're good to go, go ahead and convert it.
  resultType = GetFunctionType(*fi);

  RecordsBeingLaidOut.erase(ty);

  assert(!SkippedLayout && "Shouldn't have skipped anything yet");

  if (RecordsBeingLaidOut.empty())
    while (!DeferredRecords.empty())
      convertRecordDeclType(DeferredRecords.pop_back_val());

  return resultType;
}

/// Return true if the specified type in a function parameter or result position
/// can be converted to a CIR type at this point. This boils down to being
/// whether it is complete, as well as whether we've temporarily deferred
/// expanding the type because we're in a recursive context.
bool CIRGenTypes::isFuncParamTypeConvertible(clang::QualType ty) {
  // Some ABIs cannot have their member pointers represented in LLVM IR unless
  // certain circumstances have been reached.
  assert(!ty->getAs<MemberPointerType>() && "NYI");

  // If this isn't a tagged type, we can convert it!
  const TagType *tt = ty->getAs<TagType>();
  if (!tt)
    return true;

  // Incomplete types cannot be converted.
  if (tt->isIncompleteType())
    return false;

  // If this is an enum, then it is always safe to convert.
  const RecordType *rt = dyn_cast<RecordType>(tt);
  if (!rt)
    return true;

  // Otherwise, we have to be careful.  If it is a struct that we're in the
  // process of expanding, then we can't convert the function type.  That's ok
  // though because we must be in a pointer context under the struct, so we can
  // just convert it to a dummy type.
  //
  // We decide this by checking whether ConvertRecordDeclType returns us an
  // opaque type for a struct that we know is defined.
  return isSafeToConvert(rt->getDecl(), *this);
}

/// Code to verify a given function type is complete, i.e. the return type and
/// all of the parameter types are complete. Also check to see if we are in a
/// RS_StructPointer context, and if so whether any struct types have been
/// pended. If so, we don't want to ask the ABI lowering code to handle a type
/// that cannot be converted to a CIR type.
bool CIRGenTypes::isFuncTypeConvertible(const FunctionType *ft) {
  if (!isFuncParamTypeConvertible(ft->getReturnType()))
    return false;

  if (const auto *fpt = dyn_cast<FunctionProtoType>(ft))
    for (unsigned i = 0, e = fpt->getNumParams(); i != e; i++)
      if (!isFuncParamTypeConvertible(fpt->getParamType(i)))
        return false;

  return true;
}

/// ConvertType - Convert the specified type to its MLIR form.
mlir::Type CIRGenTypes::ConvertType(QualType t) {
  t = Context.getCanonicalType(t);
  const Type *ty = t.getTypePtr();

  // For the device-side compilation, CUDA device builtin surface/texture types
  // may be represented in different types.
  assert(!Context.getLangOpts().CUDAIsDevice && "not implemented");

  if (const auto *recordType = dyn_cast<RecordType>(t))
    return convertRecordDeclType(recordType->getDecl());

  // See if type is already cached.
  TypeCacheTy::iterator tci = TypeCache.find(ty);
  // If type is found in map then use it. Otherwise, convert type T.
  if (tci != TypeCache.end())
    return tci->second;

  // If we don't have it in the cache, convert it now.
  mlir::Type resultType = nullptr;
  switch (ty->getTypeClass()) {
  case Type::Record: // Handled above.
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.inc"
    llvm_unreachable("Non-canonical or dependent types aren't possible.");

  case Type::ArrayParameter:
    llvm_unreachable("NYI");

  case Type::Builtin: {
    switch (cast<BuiltinType>(ty)->getKind()) {
    case BuiltinType::HLSLResource:
      llvm_unreachable("NYI");
    case BuiltinType::SveBoolx2:
    case BuiltinType::SveBoolx4:
    case BuiltinType::SveCount:
      llvm_unreachable("NYI");
    case BuiltinType::Void:
      // TODO(cir): how should we model this?
      resultType = CGM.VoidTy;
      break;

    case BuiltinType::ObjCId:
    case BuiltinType::ObjCClass:
    case BuiltinType::ObjCSel:
      // TODO(cir): probably same as BuiltinType::Void
      assert(0 && "not implemented");
      break;

    case BuiltinType::Bool:
      resultType = ::mlir::cir::BoolType::get(Builder.getContext());
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
      resultType =
          mlir::cir::IntType::get(Builder.getContext(), Context.getTypeSize(t),
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
      resultType =
          mlir::cir::IntType::get(Builder.getContext(), Context.getTypeSize(t),
                                  /*isSigned=*/false);
      break;

    case BuiltinType::Float16:
      resultType = CGM.FP16Ty;
      break;
    case BuiltinType::Half:
      // Should be the same as above?
      assert(0 && "not implemented");
      break;
    case BuiltinType::BFloat16:
      resultType = CGM.BFloat16Ty;
      break;
    case BuiltinType::Float:
      resultType = CGM.FloatTy;
      break;
    case BuiltinType::Double:
      resultType = CGM.DoubleTy;
      break;
    case BuiltinType::LongDouble:
      resultType = Builder.getLongDoubleTy(Context.getFloatTypeSemantics(t));
      break;
    case BuiltinType::Float128:
    case BuiltinType::Ibm128:
      // FIXME: look at Context.getFloatTypeSemantics(T) and getTypeForFormat
      // on LLVM codegen.
      assert(0 && "not implemented");
      break;

    case BuiltinType::NullPtr:
      // Add proper CIR type for it? this looks mostly useful for sema related
      // things (like for overloads accepting void), for now, given that
      // `sizeof(std::nullptr_t)` is equal to `sizeof(void *)`, model
      // std::nullptr_t as !cir.ptr<!void>
      resultType = Builder.getVoidPtrTy();
      break;

    case BuiltinType::UInt128:
    case BuiltinType::Int128:
      assert(0 && "not implemented");
      // FIXME: ResultType = Builder.getIntegerType(128);
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
#define AMDGPU_OPAQUE_PTR_TYPE(Name, MangledName, AS, Width, Align, Id,        \
                               SingletonId)                                    \
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
    const ComplexType *ct = cast<ComplexType>(ty);
    auto elementTy = ConvertType(ct->getElementType());
    resultType = ::mlir::cir::ComplexType::get(Builder.getContext(), elementTy);
    break;
  }
  case Type::LValueReference:
  case Type::RValueReference: {
    const ReferenceType *rTy = cast<ReferenceType>(ty);
    QualType eTy = rTy->getPointeeType();
    auto pointeeType = convertTypeForMem(eTy);
    resultType = Builder.getPointerTo(pointeeType, eTy.getAddressSpace());
    assert(resultType && "Cannot get pointer type?");
    break;
  }
  case Type::Pointer: {
    const PointerType *pTy = cast<PointerType>(ty);
    QualType eTy = pTy->getPointeeType();
    assert(!eTy->isConstantMatrixType() && "not implemented");

    mlir::Type pointeeType = ConvertType(eTy);

    // Treat effectively as a *i8.
    // if (PointeeType->isVoidTy())
    //  PointeeType = Builder.getI8Type();

    resultType = Builder.getPointerTo(pointeeType, eTy.getAddressSpace());
    assert(resultType && "Cannot get pointer type?");
    break;
  }

  case Type::VariableArray: {
    const VariableArrayType *a = cast<VariableArrayType>(ty);
    assert(a->getIndexTypeCVRQualifiers() == 0 &&
           "FIXME: We only handle trivial array types so far!");
    // VLAs resolve to the innermost element type; this matches
    // the return of alloca, and there isn't any obviously better choice.
    resultType = convertTypeForMem(a->getElementType());
    break;
  }
  case Type::IncompleteArray: {
    const IncompleteArrayType *a = cast<IncompleteArrayType>(ty);
    assert(a->getIndexTypeCVRQualifiers() == 0 &&
           "FIXME: We only handle trivial array types so far!");
    // int X[] -> [0 x int], unless the element type is not sized.  If it is
    // unsized (e.g. an incomplete struct) just use [0 x i8].
    resultType = convertTypeForMem(a->getElementType());
    if (!Builder.isSized(resultType)) {
      SkippedLayout = true;
      resultType = Builder.getUInt8Ty();
    }
    resultType = Builder.getArrayType(resultType, 0);
    break;
  }
  case Type::ConstantArray: {
    const ConstantArrayType *a = cast<ConstantArrayType>(ty);
    auto eltTy = convertTypeForMem(a->getElementType());

    // FIXME: In LLVM, "lower arrays of undefined struct type to arrays of
    // i8 just to have a concrete type". Not sure this makes sense in CIR yet.
    assert(Builder.isSized(eltTy) && "not implemented");
    resultType = ::mlir::cir::ArrayType::get(Builder.getContext(), eltTy,
                                             a->getSize().getZExtValue());
    break;
  }
  case Type::ExtVector:
  case Type::Vector: {
    const VectorType *v = cast<VectorType>(ty);
    auto elementType = convertTypeForMem(v->getElementType());
    resultType = ::mlir::cir::VectorType::get(Builder.getContext(), elementType,
                                              v->getNumElements());
    break;
  }
  case Type::ConstantMatrix: {
    assert(0 && "not implemented");
    break;
  }
  case Type::FunctionNoProto:
  case Type::FunctionProto:
    resultType = ConvertFunctionTypeInternal(t);
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
    const EnumDecl *ed = cast<EnumType>(ty)->getDecl();
    if (ed->isCompleteDefinition() || ed->isFixed())
      return ConvertType(ed->getIntegerType());
    // Return a placeholder 'i32' type.  This can be changed later when the
    // type is defined (see UpdateCompletedType), but is likely to be the
    // "right" answer.
    resultType = CGM.UInt32Ty;
    break;
  }

  case Type::BlockPointer: {
    assert(0 && "not implemented");
    break;
  }

  case Type::MemberPointer: {
    const auto *mpt = cast<MemberPointerType>(ty);

    auto memberTy = ConvertType(mpt->getPointeeType());
    auto clsTy = mlir::cast<mlir::cir::StructType>(
        ConvertType(QualType(mpt->getClass(), 0)));
    if (mpt->isMemberDataPointer())
      resultType =
          mlir::cir::DataMemberType::get(Builder.getContext(), memberTy, clsTy);
    else {
      auto memberFuncTy = mlir::cast<mlir::cir::FuncType>(memberTy);
      resultType =
          mlir::cir::MethodType::get(Builder.getContext(), memberFuncTy, clsTy);
    }
    break;
  }

  case Type::Atomic: {
    QualType valueType = cast<AtomicType>(ty)->getValueType();
    resultType = convertTypeForMem(valueType);

    // Pad out to the inflated size if necessary.
    uint64_t valueSize = Context.getTypeSize(valueType);
    uint64_t atomicSize = Context.getTypeSize(ty);
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
    const auto *bitIntTy = cast<BitIntType>(ty);
    resultType = mlir::cir::IntType::get(
        Builder.getContext(), bitIntTy->getNumBits(), bitIntTy->isSigned());
    break;
  }
  }

  assert(resultType && "Didn't convert a type?");

  TypeCache[ty] = resultType;
  return resultType;
}

const CIRGenFunctionInfo &CIRGenTypes::arrangeCIRFunctionInfo(
    CanQualType resultType, FnInfoOpts opts,
    llvm::ArrayRef<CanQualType> argTypes, FunctionType::ExtInfo info,
    llvm::ArrayRef<FunctionProtoType::ExtParameterInfo> paramInfos,
    RequiredArgs required) {
  assert(llvm::all_of(argTypes,
                      [](CanQualType T) { return T.isCanonicalAsParam(); }));
  bool instanceMethod = opts == FnInfoOpts::IsInstanceMethod;
  bool chainCall = opts == FnInfoOpts::IsChainCall;

  // Lookup or create unique function info.
  llvm::FoldingSetNodeID id;
  CIRGenFunctionInfo::Profile(id, instanceMethod, chainCall, info, paramInfos,
                              required, resultType, argTypes);

  void *insertPos = nullptr;
  CIRGenFunctionInfo *fi = FunctionInfos.FindNodeOrInsertPos(id, insertPos);
  if (fi)
    return *fi;

  mlir::cir::CallingConv cc = ClangCallConvToCIRCallConv(info.getCC());

  // Construction the function info. We co-allocate the ArgInfos.
  fi = CIRGenFunctionInfo::create(cc, instanceMethod, chainCall, info,
                                  paramInfos, resultType, argTypes, required);
  FunctionInfos.InsertNode(fi, insertPos);

  bool inserted = FunctionsBeingProcessed.insert(fi).second;
  (void)inserted;
  assert(inserted && "Recursively being processed?");

  // Compute ABI information.
  if (cc == mlir::cir::CallingConv::SpirKernel) {
    // Force target independent argument handling for the host visible
    // kernel functions.
    computeSPIRKernelABIInfo(CGM, *fi);
  } else if (info.getCC() == CC_Swift || info.getCC() == CC_SwiftAsync) {
    llvm_unreachable("Swift NYI");
  } else {
    getABIInfo().computeInfo(*fi);
  }

  // Loop over all of the computed argument and return value info. If any of
  // them are direct or extend without a specified coerce type, specify the
  // default now.
  ABIArgInfo &retInfo = fi->getReturnInfo();
  if (retInfo.canHaveCoerceToType() && retInfo.getCoerceToType() == nullptr)
    retInfo.setCoerceToType(ConvertType(fi->getReturnType()));

  for (auto &i : fi->arguments())
    if (i.info.canHaveCoerceToType() && i.info.getCoerceToType() == nullptr)
      i.info.setCoerceToType(ConvertType(i.type));

  bool erased = FunctionsBeingProcessed.erase(fi);
  (void)erased;
  assert(erased && "Not in set?");

  return *fi;
}

const CIRGenFunctionInfo &CIRGenTypes::arrangeGlobalDeclaration(GlobalDecl gd) {
  assert(!dyn_cast<ObjCMethodDecl>(gd.getDecl()) &&
         "This is reported as a FIXME in LLVM codegen");
  const auto *fd = cast<FunctionDecl>(gd.getDecl());

  if (isa<CXXConstructorDecl>(gd.getDecl()) ||
      isa<CXXDestructorDecl>(gd.getDecl()))
    return arrangeCXXStructorDeclaration(gd);

  return arrangeFunctionDeclaration(fd);
}

// When we find the full definition for a TagDecl, replace the 'opaque' type we
// previously made for it if applicable.
void CIRGenTypes::UpdateCompletedType(const TagDecl *td) {
  // If this is an enum being completed, then we flush all non-struct types
  // from the cache. This allows function types and other things that may be
  // derived from the enum to be recomputed.
  if (const auto *ed = dyn_cast<EnumDecl>(td)) {
    // Only flush the cache if we've actually already converted this type.
    if (TypeCache.count(ed->getTypeForDecl())) {
      // Okay, we formed some types based on this.  We speculated that the enum
      // would be lowered to i32, so we only need to flush the cache if this
      // didn't happen.
      if (!ConvertType(ed->getIntegerType()).isInteger(32))
        TypeCache.clear();
    }
    // If necessary, provide the full definition of a type only used with a
    // declaration so far.
    assert(!MissingFeatures::generateDebugInfo());
    return;
  }

  // If we completed a RecordDecl that we previously used and converted to an
  // anonymous type, then go ahead and complete it now.
  const auto *rd = cast<RecordDecl>(td);
  if (rd->isDependentType())
    return;

  // Only complete if we converted it already. If we haven't converted it yet,
  // we'll just do it lazily.
  if (recordDeclTypes.count(Context.getTagDeclType(rd).getTypePtr()))
    convertRecordDeclType(rd);

  // If necessary, provide the full definition of a type only used with a
  // declaration so far.
  if (CGM.getModuleDebugInfo())
    llvm_unreachable("NYI");
}

/// Return record layout info for the given record decl.
const CIRGenRecordLayout &
CIRGenTypes::getCIRGenRecordLayout(const RecordDecl *rd) {
  const auto *key = Context.getTagDeclType(rd).getTypePtr();

  auto i = CIRGenRecordLayouts.find(key);
  if (i != CIRGenRecordLayouts.end())
    return *i->second;

  // Compute the type information.
  convertRecordDeclType(rd);

  // Now try again.
  i = CIRGenRecordLayouts.find(key);

  assert(i != CIRGenRecordLayouts.end() &&
         "Unable to find record layout information for type");
  return *i->second;
}

bool CIRGenTypes::isPointerZeroInitializable(clang::QualType t) {
  assert((t->isAnyPointerType() || t->isBlockPointerType()) && "Invalid type");
  return isZeroInitializable(t);
}

bool CIRGenTypes::isZeroInitializable(QualType t) {
  if (t->getAs<PointerType>())
    return Context.getTargetNullPointerValue(t) == 0;

  if (const auto *at = Context.getAsArrayType(t)) {
    if (isa<IncompleteArrayType>(at))
      return true;
    if (const auto *cat = dyn_cast<ConstantArrayType>(at))
      if (Context.getConstantArrayElementCount(cat) == 0)
        return true;
    t = Context.getBaseElementType(t);
  }

  // Records are non-zero-initializable if they contain any
  // non-zero-initializable subobjects.
  if (const RecordType *rt = t->getAs<RecordType>()) {
    const RecordDecl *rd = rt->getDecl();
    return isZeroInitializable(rd);
  }

  // We have to ask the ABI about member pointers.
  if (const MemberPointerType *mpt = t->getAs<MemberPointerType>())
    llvm_unreachable("NYI");

  // Everything else is okay.
  return true;
}

bool CIRGenTypes::isZeroInitializable(const RecordDecl *rd) {
  return getCIRGenRecordLayout(rd).isZeroInitializable();
}
