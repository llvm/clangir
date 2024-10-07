
#include "clang/CIR/Target/x86.h"
#include "ABIInfo.h"
#include "ABIInfoImpl.h"
#include "LowerModule.h"
#include "LowerTypes.h"
#include "TargetInfo.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

using X86AVXABILevel = ::cir::X86AVXABILevel;
using ABIArgInfo = ::cir::ABIArgInfo;

namespace mlir {
namespace cir {

namespace {

/// \p returns the size in bits of the largest (native) vector for \p AVXLevel.
unsigned getNativeVectorSizeForAVXABI(X86AVXABILevel avxLevel) {
  switch (avxLevel) {
  case X86AVXABILevel::AVX512:
    return 512;
  case X86AVXABILevel::AVX:
    return 256;
  case X86AVXABILevel::None:
    return 128;
  }
  llvm_unreachable("Unknown AVXLevel");
}

/// Return true if the specified [start,end) bit range is known to either be
/// off the end of the specified type or being in alignment padding.  The user
/// type specified is known to be at most 128 bits in size, and have passed
/// through X86_64ABIInfo::classify with a successful classification that put
/// one of the two halves in the INTEGER class.
///
/// It is conservatively correct to return false.
static bool bitsContainNoUserData(Type ty, unsigned startBit, unsigned endBit,
                                  CIRLowerContext &context) {
  // If the bytes being queried are off the end of the type, there is no user
  // data hiding here.  This handles analysis of builtins, vectors and other
  // types that don't contain interesting padding.
  unsigned tySize = (unsigned)context.getTypeSize(ty);
  if (tySize <= startBit)
    return true;

  if (auto arrTy = llvm::dyn_cast<ArrayType>(ty)) {
    llvm_unreachable("NYI");
  }

  if (auto structTy = llvm::dyn_cast<StructType>(ty)) {
    const CIRRecordLayout &layout = context.getCIRRecordLayout(ty);

    // If this is a C++ record, check the bases first.
    if (::cir::MissingFeatures::isCXXRecordDecl() ||
        ::cir::MissingFeatures::getCXXRecordBases()) {
      llvm_unreachable("NYI");
    }

    // Verify that no field has data that overlaps the region of interest. Yes
    // this could be sped up a lot by being smarter about queried fields,
    // however we're only looking at structs up to 16 bytes, so we don't care
    // much.
    unsigned idx = 0;
    for (auto type : structTy.getMembers()) {
      unsigned fieldOffset = (unsigned)layout.getFieldOffset(idx);

      // If we found a field after the region we care about, then we're done.
      if (fieldOffset >= endBit)
        break;

      unsigned fieldStart = fieldOffset < startBit ? startBit - fieldOffset : 0;
      if (!bitsContainNoUserData(type, fieldStart, endBit - fieldOffset,
                                 context))
        return false;

      ++idx;
    }

    // If nothing in this record overlapped the area of interest, we're good.
    return true;
  }

  return false;
}

/// Return a floating point type at the specified offset.
Type getFPTypeAtOffset(Type irType, unsigned irOffset,
                       const ::cir::CIRDataLayout &td) {
  if (irOffset == 0 && isa<SingleType, DoubleType>(irType))
    return irType;

  llvm_unreachable("NYI");
}

} // namespace

class X8664AbiInfo : public ABIInfo {
  using Class = ::cir::X86ArgClass;

  /// Implement the X86_64 ABI merging algorithm.
  ///
  /// Merge an accumulating classification \arg Accum with a field
  /// classification \arg Field.
  ///
  /// \param Accum - The accumulating classification. This should
  /// always be either NoClass or the result of a previous merge
  /// call. In addition, this should never be Memory (the caller
  /// should just return Memory for the aggregate).
  static Class merge(Class accum, Class field);

  /// Implement the X86_64 ABI post merging algorithm.
  ///
  /// Post merger cleanup, reduces a malformed Hi and Lo pair to
  /// final MEMORY or SSE classes when necessary.
  ///
  /// \param AggregateSize - The size of the current aggregate in
  /// the classification process.
  ///
  /// \param Lo - The classification for the parts of the type
  /// residing in the low word of the containing object.
  ///
  /// \param Hi - The classification for the parts of the type
  /// residing in the higher words of the containing object.
  ///
  void postMerge(unsigned aggregateSize, Class &lo, Class &hi) const;

  /// Determine the x86_64 register classes in which the given type T should be
  /// passed.
  ///
  /// \param Lo - The classification for the parts of the type
  /// residing in the low word of the containing object.
  ///
  /// \param Hi - The classification for the parts of the type
  /// residing in the high word of the containing object.
  ///
  /// \param OffsetBase - The bit offset of this type in the
  /// containing object.  Some parameters are classified different
  /// depending on whether they straddle an eightbyte boundary.
  ///
  /// \param isNamedArg - Whether the argument in question is a "named"
  /// argument, as used in AMD64-ABI 3.5.7.
  ///
  /// \param IsRegCall - Whether the calling conversion is regcall.
  ///
  /// If a word is unused its result will be NoClass; if a type should
  /// be passed in Memory then at least the classification of \arg Lo
  /// will be Memory.
  ///
  /// The \arg Lo class will be NoClass iff the argument is ignored.
  ///
  /// If the \arg Lo class is ComplexX87, then the \arg Hi class will
  /// also be ComplexX87.
  void classify(Type t, uint64_t offsetBase, Class &lo, Class &hi,
                bool isNamedArg, bool isRegCall = false) const;

  Type getSseTypeAtOffset(Type irType, unsigned irOffset, Type sourceTy,
                          unsigned sourceOffset) const;

  Type getIntegerTypeAtOffset(Type destTy, unsigned irOffset, Type sourceTy,
                              unsigned sourceOffset) const;

  /// The 0.98 ABI revision clarified a lot of ambiguities,
  /// unfortunately in ways that were not always consistent with
  /// certain previous compilers.  In particular, platforms which
  /// required strict binary compatibility with older versions of GCC
  /// may need to exempt themselves.
  bool honorsRevision098() const {
    return !getTarget().getTriple().isOSDarwin();
  }

  X86AVXABILevel avxLevel;

public:
  X8664AbiInfo(LowerTypes &cgt, X86AVXABILevel avxLevel)
      : ABIInfo(cgt), avxLevel(avxLevel) {}

  ::cir::ABIArgInfo classifyReturnType(Type retTy) const;

  ABIArgInfo classifyArgumentType(Type ty, unsigned freeIntRegs,
                                  unsigned &neededInt, unsigned &neededSSE,
                                  bool isNamedArg, bool isRegCall) const;

  void computeInfo(LowerFunctionInfo &fi) const override;
};

class X8664TargetLoweringInfo : public TargetLoweringInfo {
public:
  X8664TargetLoweringInfo(LowerTypes &lm, X86AVXABILevel avxLevel)
      : TargetLoweringInfo(std::make_unique<X8664AbiInfo>(lm, avxLevel)) {
    assert(!::cir::MissingFeatures::swift());
  }

  unsigned getTargetAddrSpaceFromCIRAddrSpace(
      mlir::cir::AddressSpaceAttr addressSpaceAttr) const override {
    using Kind = mlir::cir::AddressSpaceAttr::Kind;
    switch (addressSpaceAttr.getValue()) {
    case Kind::offload_private:
    case Kind::offload_local:
    case Kind::offload_global:
    case Kind::offload_constant:
    case Kind::offload_generic:
      return 0;
    default:
      llvm_unreachable("Unknown CIR address space for this target");
    }
  }
};

void X8664AbiInfo::classify(Type ty, uint64_t offsetBase, Class &lo, Class &hi,
                            bool isNamedArg, bool isRegCall) const {
  // FIXME: This code can be simplified by introducing a simple value class
  // for Class pairs with appropriate constructor methods for the various
  // situations.

  // FIXME: Some of the split computations are wrong; unaligned vectors
  // shouldn't be passed in registers for example, so there is no chance they
  // can straddle an eightbyte. Verify & simplify.

  lo = hi = Class::NoClass;

  Class &current = offsetBase < 64 ? lo : hi;
  current = Class::Memory;

  // FIXME(cir): There's currently no direct way to identify if a type is a
  // builtin.
  if (/*isBuitinType=*/true) {
    if (isa<VoidType>(ty)) {
      current = Class::NoClass;
    } else if (isa<IntType>(ty)) {

      // FIXME(cir): Clang's BuiltinType::Kind allow comparisons (GT, LT, etc).
      // We should implement this in CIR to simplify the conditions below.
      // Hence, Comparisons below might not be truly equivalent to the ones in
      // Clang.
      if (isa<IntType>(ty)) {
        current = Class::Integer;
      }
      return;

    } else if (isa<SingleType>(ty) || isa<DoubleType>(ty)) {
      current = Class::SSE;
      return;

    } else if (isa<BoolType>(ty)) {
      current = Class::Integer;
    } else if (const auto rt = dyn_cast<StructType>(ty)) {
      uint64_t size = getContext().getTypeSize(ty);

      // AMD64-ABI 3.2.3p2: Rule 1. If the size of an object is larger
      // than eight eightbytes, ..., it has class MEMORY.
      if (size > 512)
        llvm_unreachable("NYI");

      // AMD64-ABI 3.2.3p2: Rule 2. If a C++ object has either a non-trivial
      // copy constructor or a non-trivial destructor, it is passed by invisible
      // reference.
      if (getRecordArgABI(rt, getCXXABI()))
        llvm_unreachable("NYI");

      // Assume variable sized types are passed in memory.
      if (::cir::MissingFeatures::recordDeclHasFlexibleArrayMember())
        llvm_unreachable("NYI");

      const auto &layout = getContext().getCIRRecordLayout(ty);

      // Reset Lo class, this will be recomputed.
      current = Class::NoClass;

      // If this is a C++ record, classify the bases first.
      assert(!::cir::MissingFeatures::isCXXRecordDecl() &&
             !::cir::MissingFeatures::getCXXRecordBases());

      // Classify the fields one at a time, merging the results.
      bool useClang11Compat = getContext().getLangOpts().getClangABICompat() <=
                                  clang::LangOptions::ClangABI::Ver11 ||
                              getContext().getTargetInfo().getTriple().isPS();
      bool isUnion = rt.isUnion() && !useClang11Compat;

      // FIXME(cir): An interface to handle field declaration might be needed.
      assert(!::cir::MissingFeatures::fieldDeclAbstraction());
      for (auto [idx, FT] : llvm::enumerate(rt.getMembers())) {
        uint64_t offset = offsetBase + layout.getFieldOffset(idx);
        assert(!::cir::MissingFeatures::fieldDeclIsBitfield());
        bool bitField = false;

        // Ignore padding bit-fields.
        if (bitField && !::cir::MissingFeatures::fieldDeclisUnnamedBitField())
          llvm_unreachable("NYI");

        // AMD64-ABI 3.2.3p2: Rule 1. If the size of an object is larger than
        // eight eightbytes, or it contains unaligned fields, it has class
        // MEMORY.
        //
        // The only case a 256-bit or a 512-bit wide vector could be used is
        // when the struct contains a single 256-bit or 512-bit element. Early
        // check and fallback to memory.
        //
        // FIXME: Extended the Lo and Hi logic properly to work for size wider
        // than 128.
        if (size > 128 && ((!isUnion && size != getContext().getTypeSize(FT)) ||
                           size > getNativeVectorSizeForAVXABI(avxLevel))) {
          llvm_unreachable("NYI");
        }
        // Note, skip this test for bit-fields, see below.
        if (!bitField && offset % getContext().getTypeAlign(rt)) {
          llvm_unreachable("NYI");
        }

        // Classify this field.
        //
        // AMD64-ABI 3.2.3p2: Rule 3. If the size of the aggregate
        // exceeds a single eightbyte, each is classified
        // separately. Each eightbyte gets initialized to class
        // NO_CLASS.
        Class fieldLo, fieldHi;

        // Bit-fields require special handling, they do not force the
        // structure to be passed in memory even if unaligned, and
        // therefore they can straddle an eightbyte.
        if (bitField) {
          llvm_unreachable("NYI");
        } else {
          classify(FT, offset, fieldLo, fieldHi, isNamedArg);
        }
        lo = merge(lo, fieldLo);
        hi = merge(hi, fieldHi);
        if (lo == Class::Memory || hi == Class::Memory)
          break;
      }

      postMerge(size, lo, hi);
    } else {
      llvm::outs() << "Missing X86 classification for type " << ty << "\n";
      llvm_unreachable("NYI");
    }
    // FIXME: _Decimal32 and _Decimal64 are SSE.
    // FIXME: _float128 and _Decimal128 are (SSE, SSEUp).
    return;
  }

  llvm::outs() << "Missing X86 classification for non-builtin types\n";
  llvm_unreachable("NYI");
}

/// Return a type that will be passed by the backend in the low 8 bytes of an
/// XMM register, corresponding to the SSE class.
Type X8664AbiInfo::getSseTypeAtOffset(Type irType, unsigned irOffset,
                                      Type sourceTy,
                                      unsigned sourceOffset) const {
  const ::cir::CIRDataLayout &td = getDataLayout();
  unsigned sourceSize =
      (unsigned)getContext().getTypeSize(sourceTy) / 8 - sourceOffset;
  Type t0 = getFPTypeAtOffset(irType, irOffset, td);
  if (!t0 || isa<Float64Type>(t0))
    return t0; // NOTE(cir): Not sure if this is correct.

  Type t1 = {};
  unsigned t0Size = td.getTypeAllocSize(t0);
  if (sourceSize > t0Size)
    llvm_unreachable("NYI");
  if (t1 == nullptr) {
    // Check if IRType is a half/bfloat + float. float type will be in
    // IROffset+4 due to its alignment.
    if (isa<Float16Type>(t0) && sourceSize > 4)
      llvm_unreachable("NYI");
    // If we can't get a second FP type, return a simple half or float.
    // avx512fp16-abi.c:pr51813_2 shows it works to return float for
    // {float, i8} too.
    if (t1 == nullptr)
      return t0;
  }

  llvm_unreachable("NYI");
}

/// The ABI specifies that a value should be passed in an 8-byte GPR.  This
/// means that we either have a scalar or we are talking about the high or low
/// part of an up-to-16-byte struct.  This routine picks the best CIR type
/// to represent this, which may be i64 or may be anything else that the
/// backend will pass in a GPR that works better (e.g. i8, %foo*, etc).
///
/// PrefType is an CIR type that corresponds to (part of) the IR type for
/// the source type.  IROffset is an offset in bytes into the CIR type that
/// the 8-byte value references.  PrefType may be null.
///
/// SourceTy is the source-level type for the entire argument.  SourceOffset
/// is an offset into this that we're processing (which is always either 0 or
/// 8).
///
Type X8664AbiInfo::getIntegerTypeAtOffset(Type destTy, unsigned irOffset,
                                          Type sourceTy,
                                          unsigned sourceOffset) const {
  // If we're dealing with an un-offset CIR type, then it means that we're
  // returning an 8-byte unit starting with it. See if we can safely use it.
  if (irOffset == 0) {
    // Pointers and int64's always fill the 8-byte unit.
    assert(!isa<PointerType>(destTy) && "Ptrs are NYI");

    // If we have a 1/2/4-byte integer, we can use it only if the rest of the
    // goodness in the source type is just tail padding.  This is allowed to
    // kick in for struct {double,int} on the int, but not on
    // struct{double,int,int} because we wouldn't return the second int.  We
    // have to do this analysis on the source type because we can't depend on
    // unions being lowered a specific way etc.
    if (auto intTy = dyn_cast<IntType>(destTy)) {
      if (intTy.getWidth() == 8 || intTy.getWidth() == 16 ||
          intTy.getWidth() == 32) {
        unsigned bitWidth = intTy.getWidth();
        if (bitsContainNoUserData(sourceTy, sourceOffset * 8 + bitWidth,
                                  sourceOffset * 8 + 64, getContext()))
          return destTy;
      }
    }
  }

  if (auto rt = dyn_cast<StructType>(destTy)) {
    // If this is a struct, recurse into the field at the specified offset.
    const ::cir::StructLayout *sl = getDataLayout().getStructLayout(rt);
    if (irOffset < sl->getSizeInBytes()) {
      unsigned fieldIdx = sl->getElementContainingOffset(irOffset);
      irOffset -= sl->getElementOffset(fieldIdx);

      return getIntegerTypeAtOffset(rt.getMembers()[fieldIdx], irOffset,
                                    sourceTy, sourceOffset);
    }
  }

  // Okay, we don't have any better idea of what to pass, so we pass this in
  // an integer register that isn't too big to fit the rest of the struct.
  unsigned tySizeInBytes =
      (unsigned)getContext().getTypeSizeInChars(sourceTy).getQuantity();

  assert(tySizeInBytes != sourceOffset && "Empty field?");

  // It is always safe to classify this as an integer type up to i64 that
  // isn't larger than the structure.
  // FIXME(cir): Perhaps we should have the concept of singless integers in
  // CIR, mostly because coerced types should carry sign. On the other hand,
  // this might not make a difference in practice. For now, we just preserve the
  // sign as is to avoid unecessary bitcasts.
  bool isSigned = false;
  if (auto intTy = dyn_cast<IntType>(sourceTy))
    isSigned = intTy.isSigned();
  return IntType::get(LT.getMLIRContext(),
                      std::min(tySizeInBytes - sourceOffset, 8U) * 8, isSigned);
}

::cir::ABIArgInfo X8664AbiInfo::classifyReturnType(Type retTy) const {
  // AMD64-ABI 3.2.3p4: Rule 1. Classify the return type with the
  // classification algorithm.
  X8664AbiInfo::Class lo, hi;
  classify(retTy, 0, lo, hi, true);

  // Check some invariants.
  assert((hi != Class::Memory || lo == Class::Memory) &&
         "Invalid memory classification.");
  assert((hi != Class::SSEUp || lo == Class::SSE) &&
         "Invalid SSEUp classification.");

  Type resType = {};
  switch (lo) {
  case Class::NoClass:
    if (hi == Class::NoClass)
      return ABIArgInfo::getIgnore();
    break;

  case Class::Integer:
    resType = getIntegerTypeAtOffset(retTy, 0, retTy, 0);

    // If we have a sign or zero extended integer, make sure to return Extend
    // so that the parameter gets the right LLVM IR attributes.
    if (hi == Class::NoClass && isa<IntType>(resType)) {
      // NOTE(cir): We skip enum types handling here since CIR represents
      // enums directly as their unerlying integer types. NOTE(cir): For some
      // reason, Clang does not set the coerce type here and delays it to
      // arrangeLLVMFunctionInfo. We do the same to keep parity.
      if (isa<IntType, BoolType>(retTy) && isPromotableIntegerTypeForABI(retTy))
        return ABIArgInfo::getExtend(retTy);
    }
    break;

    // AMD64-ABI 3.2.3p4: Rule 4. If the class is SSE, the next
    // available SSE register of the sequence %xmm0, %xmm1 is used.
  case Class::SSE:
    resType = getSseTypeAtOffset(retTy, 0, retTy, 0);
    break;

  default:
    llvm_unreachable("NYI");
  }

  Type highPart = {};
  switch (hi) {

  case Class::NoClass:
    break;

  default:
    llvm_unreachable("NYI");
  }

  // If a high part was specified, merge it together with the low part.  It is
  // known to pass in the high eightbyte of the result.  We do this by forming
  // a first class struct aggregate with the high and low part: {low, high}
  if (highPart)
    llvm_unreachable("NYI");

  return ABIArgInfo::getDirect(resType);
}

ABIArgInfo X8664AbiInfo::classifyArgumentType(Type ty, unsigned freeIntRegs,
                                              unsigned &neededInt,
                                              unsigned &neededSSE,
                                              bool isNamedArg,
                                              bool isRegCall = false) const {
  ty = useFirstFieldIfTransparentUnion(ty);

  X8664AbiInfo::Class lo, hi;
  classify(ty, 0, lo, hi, isNamedArg, isRegCall);

  // Check some invariants.
  // FIXME: Enforce these by construction.
  assert((hi != Class::Memory || lo == Class::Memory) &&
         "Invalid memory classification.");
  assert((hi != Class::SSEUp || lo == Class::SSE) &&
         "Invalid SSEUp classification.");

  neededInt = 0;
  neededSSE = 0;
  Type resType = {};
  switch (lo) {
    // AMD64-ABI 3.2.3p3: Rule 2. If the class is INTEGER, the next
    // available register of the sequence %rdi, %rsi, %rdx, %rcx, %r8
    // and %r9 is used.
  case Class::Integer:
    ++neededInt;

    // Pick an 8-byte type based on the preferred type.
    resType = getIntegerTypeAtOffset(ty, 0, ty, 0);

    // If we have a sign or zero extended integer, make sure to return Extend
    // so that the parameter gets the right LLVM IR attributes.
    if (hi == Class::NoClass && isa<IntType>(resType)) {
      // NOTE(cir): We skip enum types handling here since CIR represents
      // enums directly as their unerlying integer types. NOTE(cir): For some
      // reason, Clang does not set the coerce type here and delays it to
      // arrangeLLVMFunctionInfo. We do the same to keep parity.
      if (isa<IntType, BoolType>(ty) && isPromotableIntegerTypeForABI(ty))
        return ABIArgInfo::getExtend(ty);
    }

    break;

    // AMD64-ABI 3.2.3p3: Rule 3. If the class is SSE, the next
    // available SSE register is used, the registers are taken in the
    // order from %xmm0 to %xmm7.
  case Class::SSE: {
    resType = getSseTypeAtOffset(ty, 0, ty, 0);
    ++neededSSE;
    break;
  }
  default:
    llvm_unreachable("NYI");
  }

  Type highPart = {};
  switch (hi) {
  case Class::NoClass:
    break;
  default:
    llvm_unreachable("NYI");
  }

  if (highPart)
    llvm_unreachable("NYI");

  return ABIArgInfo::getDirect(resType);
}

void X8664AbiInfo::computeInfo(LowerFunctionInfo &fi) const {
  const unsigned callingConv = fi.getCallingConvention();
  // It is possible to force Win64 calling convention on any x86_64 target by
  // using __attribute__((ms_abi)). In such case to correctly emit Win64
  // compatible code delegate this call to WinX86_64ABIInfo::computeInfo.
  if (callingConv == llvm::CallingConv::Win64) {
    llvm_unreachable("Win64 CC is NYI");
  }

  bool isRegCall = callingConv == llvm::CallingConv::X86_RegCall;

  // Keep track of the number of assigned registers.
  unsigned freeIntRegs = isRegCall ? 11 : 6;
  unsigned freeSseRegs = isRegCall ? 16 : 8;
  unsigned neededInt = 0, neededSse = 0, maxVectorWidth = 0;

  if (!::mlir::cir::classifyReturnType(getCXXABI(), fi, *this)) {
    if (isRegCall || ::cir::MissingFeatures::regCall()) {
      llvm_unreachable("RegCall is NYI");
    } else
      fi.getReturnInfo() = classifyReturnType(fi.getReturnType());
  }

  // If the return value is indirect, then the hidden argument is consuming
  // one integer register.
  if (fi.getReturnInfo().isIndirect())
    llvm_unreachable("NYI");
  else if (neededSse && maxVectorWidth)
    llvm_unreachable("NYI");

  // The chain argument effectively gives us another free register.
  if (::cir::MissingFeatures::chainCall())
    llvm_unreachable("NYI");

  unsigned numRequiredArgs = fi.getNumRequiredArgs();
  // AMD64-ABI 3.2.3p3: Once arguments are classified, the registers
  // get assigned (in left-to-right order) for passing as follows...
  unsigned argNo = 0;
  for (LowerFunctionInfo::arg_iterator it = fi.arg_begin(), ie = fi.arg_end();
       it != ie; ++it, ++argNo) {
    bool isNamedArg = argNo < numRequiredArgs;

    if (isRegCall && ::cir::MissingFeatures::regCall())
      llvm_unreachable("NYI");
    else
      it->info = classifyArgumentType(it->type, freeIntRegs, neededInt,
                                      neededSse, isNamedArg);

    // AMD64-ABI 3.2.3p3: If there are no registers available for any
    // eightbyte of an argument, the whole argument is passed on the
    // stack. If registers have already been assigned for some
    // eightbytes of such an argument, the assignments get reverted.
    if (freeIntRegs >= neededInt && freeSseRegs >= neededSse) {
      freeIntRegs -= neededInt;
      freeSseRegs -= neededSse;
      if (::cir::MissingFeatures::vectorType())
        llvm_unreachable("NYI");
    } else {
      llvm_unreachable("Indirect results are NYI");
    }
  }
}

X8664AbiInfo::Class X8664AbiInfo::merge(Class accum, Class field) {
  // AMD64-ABI 3.2.3p2: Rule 4. Each field of an object is
  // classified recursively so that always two fields are
  // considered. The resulting class is calculated according to
  // the classes of the fields in the eightbyte:
  //
  // (a) If both classes are equal, this is the resulting class.
  //
  // (b) If one of the classes is NO_CLASS, the resulting class is
  // the other class.
  //
  // (c) If one of the classes is MEMORY, the result is the MEMORY
  // class.
  //
  // (d) If one of the classes is INTEGER, the result is the
  // INTEGER.
  //
  // (e) If one of the classes is X87, X87UP, COMPLEX_X87 class,
  // MEMORY is used as class.
  //
  // (f) Otherwise class SSE is used.

  // Accum should never be memory (we should have returned) or
  // ComplexX87 (because this cannot be passed in a structure).
  assert((accum != Class::Memory && accum != Class::ComplexX87) &&
         "Invalid accumulated classification during merge.");
  if (accum == field || field == Class::NoClass)
    return accum;
  if (field == Class::Memory)
    return Class::Memory;
  if (accum == Class::NoClass)
    return field;
  if (accum == Class::Integer || field == Class::Integer)
    return Class::Integer;
  if (field == Class::X87 || field == Class::X87Up ||
      field == Class::ComplexX87 || accum == Class::X87 ||
      accum == Class::X87Up)
    return Class::Memory;
  return Class::SSE;
}

void X8664AbiInfo::postMerge(unsigned aggregateSize, Class &lo,
                             Class &hi) const {
  // AMD64-ABI 3.2.3p2: Rule 5. Then a post merger cleanup is done:
  //
  // (a) If one of the classes is Memory, the whole argument is passed in
  //     memory.
  //
  // (b) If X87UP is not preceded by X87, the whole argument is passed in
  //     memory.
  //
  // (c) If the size of the aggregate exceeds two eightbytes and the first
  //     eightbyte isn't SSE or any other eightbyte isn't SSEUP, the whole
  //     argument is passed in memory. NOTE: This is necessary to keep the
  //     ABI working for processors that don't support the __m256 type.
  //
  // (d) If SSEUP is not preceded by SSE or SSEUP, it is converted to SSE.
  //
  // Some of these are enforced by the merging logic.  Others can arise
  // only with unions; for example:
  //   union { _Complex double; unsigned; }
  //
  // Note that clauses (b) and (c) were added in 0.98.
  //
  if (hi == Class::Memory)
    lo = Class::Memory;
  if (hi == Class::X87Up && lo != Class::X87 && honorsRevision098())
    lo = Class::Memory;
  if (aggregateSize > 128 && (lo != Class::SSE || hi != Class::SSEUp))
    lo = Class::Memory;
  if (hi == Class::SSEUp && lo != Class::SSE)
    hi = Class::SSE;
}

std::unique_ptr<TargetLoweringInfo>
createX86_64TargetLoweringInfo(LowerModule &lm, X86AVXABILevel avxLevel) {
  return std::make_unique<X8664TargetLoweringInfo>(lm.getTypes(), avxLevel);
}

} // namespace cir
} // namespace mlir
