#include "TargetInfo.h"
#include "ABIInfo.h"
#include "CIRGenCXXABI.h"
#include "CIRGenFunctionInfo.h"
#include "CIRGenTypes.h"

#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/MissingFeatures.h"
#include "clang/CIR/Target/x86.h"

using namespace cir;
using namespace clang;

static bool testIfIsVoidTy(QualType ty) {
  const auto *bt = ty->getAs<BuiltinType>();
  if (!bt)
    return false;

  BuiltinType::Kind k = bt->getKind();
  return k == BuiltinType::Void;
}

static bool isAggregateTypeForABI(QualType t) {
  return !CIRGenFunction::hasScalarEvaluationKind(t) ||
         t->isMemberFunctionPointerType();
}

/// Pass transparent unions as if they were the type of the first element. Sema
/// should ensure that all elements of the union have the same "machine type".
static QualType useFirstFieldIfTransparentUnion(QualType ty) {
  assert(!ty->getAsUnionType() && "NYI");
  return ty;
}

namespace {

/// The default implementation for ABI specific
/// details. This implementation provides information which results in
/// self-consistent and sensible LLVM IR generation, but does not
/// conform to any particular ABI.
class DefaultABIInfo : public ABIInfo {
public:
  DefaultABIInfo(CIRGenTypes &cgt) : ABIInfo(cgt) {}

  ~DefaultABIInfo() override = default;

  ABIArgInfo classifyReturnType(QualType retTy) const {
    if (retTy->isVoidType())
      return ABIArgInfo::getIgnore();

    if (isAggregateTypeForABI(retTy))
      llvm_unreachable("NYI");

    // Treat an enum type as its underlying type.
    if (const EnumType *enumTy = retTy->getAs<EnumType>())
      llvm_unreachable("NYI");

    if (const auto *eit = retTy->getAs<BitIntType>())
      llvm_unreachable("NYI");

    return (isPromotableIntegerTypeForABI(retTy) ? ABIArgInfo::getExtend(retTy)
                                                 : ABIArgInfo::getDirect());
  }

  ABIArgInfo classifyArgumentType(QualType ty) const {
    ty = useFirstFieldIfTransparentUnion(ty);

    if (isAggregateTypeForABI(ty)) {
      llvm_unreachable("NYI");
    }

    // Treat an enum type as its underlying type.
    if (const EnumType *enumTy = ty->getAs<EnumType>())
      llvm_unreachable("NYI");

    if (const auto *eit = ty->getAs<BitIntType>())
      llvm_unreachable("NYI");

    return (isPromotableIntegerTypeForABI(ty) ? ABIArgInfo::getExtend(ty)
                                              : ABIArgInfo::getDirect());
  }

  void computeInfo(CIRGenFunctionInfo &fi) const override {
    if (!getCXXABI().classifyReturnType(fi))
      fi.getReturnInfo() = classifyReturnType(fi.getReturnType());
    for (auto &i : fi.arguments())
      i.info = classifyArgumentType(i.type);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// AArch64 ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class AArch64ABIInfo : public ABIInfo {
public:
  enum ABIKind { AAPCS = 0, DarwinPCS, Win64 };

private:
  ABIKind kind;

public:
  AArch64ABIInfo(CIRGenTypes &cgt, ABIKind kind) : ABIInfo(cgt), kind(kind) {}
  bool allowBFloatArgsAndRet() const override {
    // TODO: Should query target info instead of hardcoding.
    assert(!cir::MissingFeatures::useTargetLoweringABIInfo());
    return true;
  }

private:
  ABIKind getABIKind() const { return kind; }
  bool isDarwinPCS() const { return kind == DarwinPCS; }

  ABIArgInfo classifyReturnType(QualType retTy, bool isVariadic) const;
  ABIArgInfo classifyArgumentType(QualType retTy, bool isVariadic,
                                  unsigned callingConvention) const;

  void computeInfo(CIRGenFunctionInfo &fi) const override {
    // Top leevl CIR has unlimited arguments and return types. Lowering for ABI
    // specific concerns should happen during a lowering phase. Assume
    // everything is direct for now.
    for (CIRGenFunctionInfo::arg_iterator it = fi.arg_begin(),
                                          ie = fi.arg_end();
         it != ie; ++it) {
      if (testIfIsVoidTy(it->type))
        it->info = ABIArgInfo::getIgnore();
      else
        it->info = ABIArgInfo::getDirect(CGT.ConvertType(it->type));
    }
    auto retTy = fi.getReturnType();
    if (testIfIsVoidTy(retTy))
      fi.getReturnInfo() = ABIArgInfo::getIgnore();
    else
      fi.getReturnInfo() = ABIArgInfo::getDirect(CGT.ConvertType(retTy));
  }
};

class AArch64TargetCIRGenInfo : public TargetCIRGenInfo {
public:
  AArch64TargetCIRGenInfo(CIRGenTypes &cgt, AArch64ABIInfo::ABIKind kind)
      : TargetCIRGenInfo(std::make_unique<AArch64ABIInfo>(cgt, kind)) {}
};

} // namespace

//===----------------------------------------------------------------------===//
// X86 ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

/// The AVX ABI leel for X86 targets.
using X86AVXABILevel = ::cir::X86AVXABILevel;

class X8664AbiInfo : public ABIInfo {
  using Class = X86ArgClass;

  // X86AVXABILevel AVXLevel;
  // Some ABIs (e.g. X32 ABI and Native Client OS) use 32 bit pointers on 64-bit
  // hardware.
  // bool Has64BitPointers;

public:
  X8664AbiInfo(CIRGenTypes &cgt, X86AVXABILevel avxLevel)
      : ABIInfo(cgt)
  // , AVXLevel(AVXLevel)
  // , Has64BitPointers(CGT.getDataLayout().getPointeSize(0) == 8)
  {}

  void computeInfo(CIRGenFunctionInfo &fi) const override;

  /// classify - Determine the x86_64 register classes in which the given type T
  /// should be passed.
  ///
  /// \param Lo - The classification for the parts of the type residing in the
  /// low word of the containing object.
  ///
  /// \param Hi - The classification for the parts of the type residing in the
  /// high word of the containing object.
  ///
  /// \param OffsetBase - The bit offset of this type in the containing object.
  /// Some parameters are classified different depending on whether they
  /// straddle an eightbyte boundary.
  ///
  /// \param isNamedArg - Whether the argument in question is a "named"
  /// argument, as used in AMD64-ABI 3.5.7.
  ///
  /// If a word is unused its result will be NoClass; if a type should be passed
  /// in Memory then at least the classification of \arg Lo will be Memory.
  ///
  /// The \arg Lo class will be NoClass iff the argument is ignored.
  ///
  /// If the \arg Lo class is ComplexX87, then the \arg Hi class will also be
  /// ComplexX87.
  void classify(clang::QualType t, uint64_t offsetBase, Class &lo, Class &hi,
                bool isNamedArg) const;

  mlir::Type getSseTypeAtOffset(mlir::Type cirType, unsigned cirOffset,
                                clang::QualType sourceTy,
                                unsigned sourceOffset) const;

  ABIArgInfo classifyReturnType(QualType retTy) const;

  ABIArgInfo classifyArgumentType(clang::QualType ty, unsigned freeIntRegs,
                                  unsigned &neededInt, unsigned &neededSSE,
                                  bool isNamedArg) const;

  mlir::Type getIntegerTypeAtOffset(mlir::Type cirType, unsigned cirOffset,
                                    QualType sourceTy,
                                    unsigned sourceOffset) const;

  /// getIndirectResult - Give a source type \arg Ty, return a suitable result
  /// such that the argument will be passed in memory.
  ///
  /// \param freeIntRegs - The number of free integer registers remaining
  /// available.
  ABIArgInfo getIndirectResult(QualType ty, unsigned freeIntRegs) const;
};

class X8664TargetCirGenInfo : public TargetCIRGenInfo {
public:
  X8664TargetCirGenInfo(CIRGenTypes &cgt, X86AVXABILevel avxLevel)
      : TargetCIRGenInfo(std::make_unique<X8664AbiInfo>(cgt, avxLevel)) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// Base ABI and target codegen info implementation common between SPIR and
// SPIR-V.
//===----------------------------------------------------------------------===//

namespace {
class CommonSPIRABIInfo : public DefaultABIInfo {
public:
  CommonSPIRABIInfo(CIRGenTypes &cgt) : DefaultABIInfo(cgt) {}
};

class SPIRVABIInfo : public CommonSPIRABIInfo {
public:
  SPIRVABIInfo(CIRGenTypes &cgt) : CommonSPIRABIInfo(cgt) {}
  void computeInfo(CIRGenFunctionInfo &fi) const override {
    // The logic is same as in DefaultABIInfo with an exception on the kernel
    // arguments handling.
    mlir::cir::CallingConv cc = fi.getCallingConvention();

    bool cxxabiHit = getCXXABI().classifyReturnType(fi);
    assert(!cxxabiHit && "C++ ABI not considered");

    fi.getReturnInfo() = classifyReturnType(fi.getReturnType());

    for (auto &i : fi.arguments()) {
      if (cc == mlir::cir::CallingConv::SpirKernel) {
        i.info = classifyKernelArgumentType(i.type);
      } else {
        i.info = classifyArgumentType(i.type);
      }
    }
  }

private:
  ABIArgInfo classifyKernelArgumentType(QualType ty) const {
    assert(!getContext().getLangOpts().CUDAIsDevice && "NYI");
    return classifyArgumentType(ty);
  }
};
} // namespace

namespace cir {
void computeSPIRKernelABIInfo(CIRGenModule &cgm, CIRGenFunctionInfo &fi) {
  if (cgm.getTarget().getTriple().isSPIRV())
    SPIRVABIInfo(cgm.getTypes()).computeInfo(fi);
  else
    CommonSPIRABIInfo(cgm.getTypes()).computeInfo(fi);
}
} // namespace cir

namespace {

class CommonSPIRTargetCIRGenInfo : public TargetCIRGenInfo {
public:
  CommonSPIRTargetCIRGenInfo(std::unique_ptr<ABIInfo> abiInfo)
      : TargetCIRGenInfo(std::move(abiInfo)) {}

  mlir::cir::AddressSpaceAttr getCIRAllocaAddressSpace() const override {
    return mlir::cir::AddressSpaceAttr::get(
        &getABIInfo().CGT.getMLIRContext(),
        mlir::cir::AddressSpaceAttr::Kind::offload_private);
  }

  mlir::cir::CallingConv getOpenCLKernelCallingConv() const override {
    return mlir::cir::CallingConv::SpirKernel;
  }
};

class SPIRVTargetCIRGenInfo : public CommonSPIRTargetCIRGenInfo {
public:
  SPIRVTargetCIRGenInfo(CIRGenTypes &cgt)
      : CommonSPIRTargetCIRGenInfo(std::make_unique<SPIRVABIInfo>(cgt)) {}
};

} // namespace

// TODO(cir): remove the attribute once this gets used.
LLVM_ATTRIBUTE_UNUSED
static bool classifyReturnType(const CIRGenCXXABI &cxxabi,
                               CIRGenFunctionInfo &fi, const ABIInfo &info) {
  QualType ty = fi.getReturnType();

  assert(!ty->getAs<RecordType>() && "RecordType returns NYI");

  return cxxabi.classifyReturnType(fi);
}

CIRGenCXXABI &ABIInfo::getCXXABI() const { return CGT.getCXXABI(); }

clang::ASTContext &ABIInfo::getContext() const { return CGT.getContext(); }

ABIArgInfo X8664AbiInfo::getIndirectResult(QualType ty,
                                           unsigned freeIntRegs) const {
  assert(false && "NYI");
}

void X8664AbiInfo::computeInfo(CIRGenFunctionInfo &fi) const {
  // Top level CIR has unlimited arguments and return types. Lowering for ABI
  // specific concerns should happen during a lowering phase. Assume everything
  // is direct for now.
  for (CIRGenFunctionInfo::arg_iterator it = fi.arg_begin(), ie = fi.arg_end();
       it != ie; ++it) {
    if (testIfIsVoidTy(it->type))
      it->info = ABIArgInfo::getIgnore();
    else
      it->info = ABIArgInfo::getDirect(CGT.ConvertType(it->type));
  }
  auto retTy = fi.getReturnType();
  if (testIfIsVoidTy(retTy))
    fi.getReturnInfo() = ABIArgInfo::getIgnore();
  else
    fi.getReturnInfo() = ABIArgInfo::getDirect(CGT.ConvertType(retTy));
}

/// GetINTEGERTypeAtOffset - The ABI specifies that a value should be passed in
/// an 8-byte GPR. This means that we either have a scalar or we are talking
/// about the high or low part of an up-to-16-byte struct. This routine picks
/// the best CIR type to represent this, which may be i64 or may be anything
/// else that the backend will pass in a GPR that works better (e.g. i8, %foo*,
/// etc).
///
/// PrefType is a CIR type that corresponds to (part of) the IR type for the
/// source type. CIROffset is an offset in bytes into the CIR type taht the
/// 8-byte value references. PrefType may be null.
///
/// SourceTy is the source-level type for the entire argument. SourceOffset is
/// an offset into this that we're processing (which is always either 0 or 8).
///
mlir::Type X8664AbiInfo::getIntegerTypeAtOffset(mlir::Type cirType,
                                                unsigned cirOffset,
                                                QualType sourceTy,
                                                unsigned sourceOffset) const {
  // TODO: entirely stubbed out
  assert(cirOffset == 0 && "NYI");
  assert(sourceOffset == 0 && "NYI");
  return cirType;
}

ABIArgInfo X8664AbiInfo::classifyArgumentType(QualType ty,
                                              unsigned int freeIntRegs,
                                              unsigned int &neededInt,
                                              unsigned int &neededSSE,
                                              bool isNamedArg) const {
  ty = useFirstFieldIfTransparentUnion(ty);

  X8664AbiInfo::Class lo, hi;
  classify(ty, 0, lo, hi, isNamedArg);

  // Check some invariants
  // FIXME: Enforce these by construction.
  assert((hi != Memory || lo == Memory) && "Invalid memory classification.");
  assert((hi != SSEUp || lo == SSE) && "Invalid SSEUp classification.");

  neededInt = 0;
  neededSSE = 0;
  mlir::Type resType = nullptr;
  switch (lo) {
  default:
    assert(false && "NYI");

  // AMD64-ABI 3.2.3p3: Rule 2. If the class is INTEGER, the next available
  // register of the sequence %rdi, %rsi, %rdx, %rcx, %r8 and %r9 is used.
  case Integer:
    ++neededInt;

    // Pick an 8-byte type based on the preferred type.
    resType = getIntegerTypeAtOffset(CGT.ConvertType(ty), 0, ty, 0);

    // If we have a sign or zero extended integer, make sure to return Extend so
    // that the parameter gets the right LLVM IR attributes.
    if (hi == NoClass && mlir::isa<mlir::cir::IntType>(resType)) {
      assert(!ty->getAs<EnumType>() && "NYI");
      if (ty->isSignedIntegerOrEnumerationType() &&
          isPromotableIntegerTypeForABI(ty))
        return ABIArgInfo::getExtend(ty);
    }

    break;

    // AMD64-ABI 3.2.3p3: Rule 3. If the class is SSE, the next available SSE
    // register is used, the registers are taken in the order from %xmm0 to
    // %xmm7.
  case SSE: {
    mlir::Type cirType = CGT.ConvertType(ty);
    resType = getSseTypeAtOffset(cirType, 0, ty, 0);
    ++neededSSE;
    break;
  }
  }

  mlir::Type highPart = nullptr;
  switch (hi) {
  default:
    assert(false && "NYI");
  case NoClass:
    break;
  }

  assert(!highPart && "NYI");

  return ABIArgInfo::getDirect(resType);
}

ABIInfo::~ABIInfo() = default;

bool ABIInfo::isPromotableIntegerTypeForABI(QualType ty) const {
  if (getContext().isPromotableIntegerType(ty))
    return true;

  assert(!ty->getAs<BitIntType>() && "NYI");

  return false;
}

void X8664AbiInfo::classify(QualType ty, uint64_t offsetBase, Class &lo,
                            Class &hi, bool isNamedArg) const {
  // FIXME: This code can be simplified by introducing a simple value class for
  // Class pairs with appropriate constructor methods for the various
  // situations.

  // FIXME: Some of the split computations are wrong; unaligned vectors
  // shouldn't be passed in registers for example, so there is no chance they
  // can straddle an eightbyte. Verify & simplify.

  lo = hi = NoClass;
  Class &current = offsetBase < 64 ? lo : hi;
  current = Memory;

  if (const auto *bt = ty->getAs<BuiltinType>()) {
    BuiltinType::Kind k = bt->getKind();
    if (k == BuiltinType::Void) {
      current = NoClass;
    } else if (k == BuiltinType::Int128 || k == BuiltinType::UInt128) {
      assert(false && "NYI");
      lo = Integer;
      hi = Integer;
    } else if (k >= BuiltinType::Bool && k <= BuiltinType::LongLong) {
      current = Integer;
    } else if (k == BuiltinType::Float || k == BuiltinType::Double ||
               k == BuiltinType::Float16) {
      current = SSE;
    } else if (k == BuiltinType::LongDouble) {
      assert(false && "NYI");
    } else
      assert(false &&
             "Only void and Integer supported so far for builtin types");
    // FIXME: _Decimal32 and _Decimal64 are SSE.
    // FIXME: _float128 and _Decimal128 are (SSE, SSEUp).
    return;
  }

  assert(!ty->getAs<EnumType>() && "Enums NYI");
  if (ty->hasPointerRepresentation()) {
    current = Integer;
    return;
  }

  assert(false && "Nothing else implemented yet");
}

/// GetSSETypeAtOffset - Return a type that will be passed by the backend in the
/// low 8 bytes of an XMM register, corresponding to the SSE class.
mlir::Type X8664AbiInfo::getSseTypeAtOffset(mlir::Type cirType,
                                            unsigned int cirOffset,
                                            clang::QualType sourceTy,
                                            unsigned int sourceOffset) const {
  // TODO: entirely stubbed out
  assert(cirOffset == 0 && "NYI");
  assert(sourceOffset == 0 && "NYI");
  return cirType;
}

ABIArgInfo X8664AbiInfo::classifyReturnType(QualType retTy) const {
  // AMD64-ABI 3.2.3p4: Rule 1. Classify the return type with the classification
  // algorithm.
  X8664AbiInfo::Class lo, hi;
  classify(retTy, 0, lo, hi, /*isNamedArg*/ true);

  // Check some invariants.
  assert((hi != Memory || lo == Memory) && "Invalid memory classification.");
  assert((hi != SSEUp || lo == SSE) && "Invalid SSEUp classification.");

  mlir::Type resType = nullptr;
  assert(lo == NoClass || lo == Integer ||
         lo == SSE && "Only NoClass and Integer supported so far");

  switch (lo) {
  case NoClass:
    assert(hi == NoClass && "Only NoClass supported so far for Hi");
    return ABIArgInfo::getIgnore();

  // AMD64-ABI 3.2.3p4: Rule 3. If the class is INTEGER, the next available
  // register of the sequence %rax, %rdx is used.
  case Integer:
    resType = getIntegerTypeAtOffset(CGT.ConvertType(retTy), 0, retTy, 0);

    // If we have a sign or zero extended integer, make sure to return Extend so
    // that the parameter gets the right LLVM IR attributes.
    // TODO: extend the above consideration to MLIR
    if (hi == NoClass && mlir::isa<mlir::cir::IntType>(resType)) {
      // Treat an enum type as its underlying type.
      if (const auto *enumTy = retTy->getAs<EnumType>())
        retTy = enumTy->getDecl()->getIntegerType();

      if (retTy->isIntegralOrEnumerationType() &&
          isPromotableIntegerTypeForABI(retTy)) {
        return ABIArgInfo::getExtend(retTy);
      }
    }
    break;

    // AMD64-ABI 3.2.3p4: Rule 4. If the class is SSE, the next available SSE
    // register of the sequence %xmm0, %xmm1 is used.
  case SSE:
    resType = getSseTypeAtOffset(CGT.ConvertType(retTy), 0, retTy, 0);
    break;

  default:
    llvm_unreachable("NYI");
  }

  mlir::Type highPart = nullptr;

  if (highPart)
    assert(false && "NYI");

  return ABIArgInfo::getDirect(resType);
}

clang::LangAS
TargetCIRGenInfo::getGlobalVarAddressSpace(cir::CIRGenModule &cgm,
                                           const clang::VarDecl *d) const {
  assert(!cgm.getLangOpts().OpenCL &&
         !(cgm.getLangOpts().CUDA && cgm.getLangOpts().CUDAIsDevice) &&
         "Address space agnostic languages only");
  return d ? d->getType().getAddressSpace() : LangAS::Default;
}

mlir::Value TargetCIRGenInfo::performAddrSpaceCast(
    CIRGenFunction &cgf, mlir::Value src, mlir::cir::AddressSpaceAttr srcAddr,
    mlir::cir::AddressSpaceAttr destAddr, mlir::Type destTy,
    bool isNonNull) const {
  // Since target may map different address spaces in AST to the same address
  // space, an address space conversion may end up as a bitcast.
  if (auto globalOp = src.getDefiningOp<mlir::cir::GlobalOp>())
    llvm_unreachable("Global ops addrspace cast NYI");
  // Try to preserve the source's name to make IR more readable.
  return cgf.getBuilder().createAddrSpaceCast(src, destTy);
}

const TargetCIRGenInfo &CIRGenModule::getTargetCIRGenInfo() {
  if (TheTargetCIRGenInfo)
    return *TheTargetCIRGenInfo;

  // Helper to set the unique_ptr while still keeping the return value.
  auto setCirGenInfo = [&](TargetCIRGenInfo *p) -> const TargetCIRGenInfo & {
    this->TheTargetCIRGenInfo.reset(p);
    return *p;
  };

  const llvm::Triple &triple = getTarget().getTriple();

  switch (triple.getArch()) {
  default:
    assert(false && "Target not yet supported!");

  case llvm::Triple::aarch64_be:
  case llvm::Triple::aarch64: {
    AArch64ABIInfo::ABIKind kind = AArch64ABIInfo::AAPCS;
    assert(getTarget().getABI() == "aapcs" ||
           getTarget().getABI() == "darwinpcs" &&
               "Only Darwin supported for aarch64");
    kind = AArch64ABIInfo::DarwinPCS;
    return setCirGenInfo(new AArch64TargetCIRGenInfo(genTypes, kind));
  }

  case llvm::Triple::x86_64: {
    StringRef abi = getTarget().getABI();
    X86AVXABILevel avxLevel = (abi == "avx512" ? X86AVXABILevel::AVX512
                               : abi == "avx"  ? X86AVXABILevel::AVX
                                               : X86AVXABILevel::None);

    switch (triple.getOS()) {
    default:
      assert(false && "OSType NYI");
    case llvm::Triple::Linux:
      return setCirGenInfo(new X8664TargetCirGenInfo(genTypes, avxLevel));
    }
  }

  case llvm::Triple::spirv64: {
    return setCirGenInfo(new SPIRVTargetCIRGenInfo(genTypes));
  }
  }
}
