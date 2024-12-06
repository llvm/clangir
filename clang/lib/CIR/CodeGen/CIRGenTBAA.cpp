#include "CIRGenTBAA.h"
#include "CIRGenCXXABI.h"
#include "CIRGenTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "llvm/Support/ErrorHandling.h"
namespace clang::CIRGen {

CIRGenTBAA::CIRGenTBAA(mlir::MLIRContext *ctx, clang::ASTContext &context,
                       CIRGenTypes &types, mlir::ModuleOp moduleOp,
                       const clang::CodeGenOptions &codeGenOpts,
                       const clang::LangOptions &features)
    : ctx(ctx), context(context), types(types), moduleOp(moduleOp),
      codeGenOpts(codeGenOpts), features(features) {}

cir::TBAAAttr CIRGenTBAA::createScalarTypeNode(llvm::StringRef typeName,
                                               cir::TBAAAttr parent,
                                               std::size_t offset) {
  return cir::TBAAScalarTypeDescriptorAttr::get(ctx, typeName);
}

cir::TBAAAttr CIRGenTBAA::getChar() {
  return cir::TBAAScalarTypeDescriptorAttr::get(ctx, "omnipotent char");
}

static bool typeHasMayAlias(clang::QualType qty) {
  // Tagged types have declarations, and therefore may have attributes.
  if (auto *td = qty->getAsTagDecl())
    if (td->hasAttr<MayAliasAttr>())
      return true;

  // Also look for may_alias as a declaration attribute on a typedef.
  // FIXME: We should follow GCC and model may_alias as a type attribute
  // rather than as a declaration attribute.
  while (auto *tt = qty->getAs<TypedefType>()) {
    if (tt->getDecl()->hasAttr<MayAliasAttr>())
      return true;
    qty = tt->desugar();
  }
  return false;
}

/// Check if the given type is a valid base type to be used in access tags.
static bool isValidBaseType(clang::QualType qty) {
  if (const clang::RecordType *tty = qty->getAs<clang::RecordType>()) {
    const clang::RecordDecl *rd = tty->getDecl()->getDefinition();
    // Incomplete types are not valid base access types.
    if (!rd)
      return false;
    if (rd->hasFlexibleArrayMember())
      return false;
    // rd can be struct, union, class, interface or enum.
    // For now, we only handle struct and class.
    if (rd->isStruct() || rd->isClass())
      return true;
  }
  return false;
}

cir::TBAAAttr CIRGenTBAA::getTypeInfoHelper(const clang::Type *ty) {
  using namespace clang;
  uint64_t size = context.getTypeSizeInChars(ty).getQuantity();
  // Handle builtin types.
  if (const BuiltinType *bty = dyn_cast<BuiltinType>(ty)) {
    switch (bty->getKind()) {
    // Character types are special and can alias anything.
    // In C++, this technically only includes "char" and "unsigned char",
    // and not "signed char". In C, it includes all three. For now,
    // the risk of exploiting this detail in C++ seems likely to outweigh
    // the benefit.
    case BuiltinType::Char_U:
    case BuiltinType::Char_S:
    case BuiltinType::UChar:
    case BuiltinType::SChar:
      return getChar();

    // Unsigned types can alias their corresponding signed types.
    case BuiltinType::UShort:
      return getTypeInfo(context.ShortTy);
    case BuiltinType::UInt:
      return getTypeInfo(context.IntTy);
    case BuiltinType::ULong:
      return getTypeInfo(context.LongTy);
    case BuiltinType::ULongLong:
      return getTypeInfo(context.LongLongTy);
    case BuiltinType::UInt128:
      return getTypeInfo(context.Int128Ty);

    case BuiltinType::UShortFract:
      return getTypeInfo(context.ShortFractTy);
    case BuiltinType::UFract:
      return getTypeInfo(context.FractTy);
    case BuiltinType::ULongFract:
      return getTypeInfo(context.LongFractTy);

    case BuiltinType::SatUShortFract:
      return getTypeInfo(context.SatShortFractTy);
    case BuiltinType::SatUFract:
      return getTypeInfo(context.SatFractTy);
    case BuiltinType::SatULongFract:
      return getTypeInfo(context.SatLongFractTy);

    case BuiltinType::UShortAccum:
      return getTypeInfo(context.ShortAccumTy);
    case BuiltinType::UAccum:
      return getTypeInfo(context.AccumTy);
    case BuiltinType::ULongAccum:
      return getTypeInfo(context.LongAccumTy);

    case BuiltinType::SatUShortAccum:
      return getTypeInfo(context.SatShortAccumTy);
    case BuiltinType::SatUAccum:
      return getTypeInfo(context.SatAccumTy);
    case BuiltinType::SatULongAccum:
      return getTypeInfo(context.SatLongAccumTy);

    // Treat all other builtin types as distinct types. This includes
    // treating wchar_t, char16_t, and char32_t as distinct from their
    // "underlying types".
    default:
      return createScalarTypeNode(bty->getName(features), getChar(), size);
    }
  }

  // C++1z [basic.lval]p10: "If a program attempts to access the stored value of
  // an object through a glvalue of other than one of the following types the
  // behavior is undefined: [...] a char, unsigned char, or std::byte type."
  if (ty->isStdByteType())
    return getChar();

  // Handle pointers and references.
  //
  // C has a very strict rule for pointer aliasing. C23 6.7.6.1p2:
  //     For two pointer types to be compatible, both shall be identically
  //     qualified and both shall be pointers to compatible types.
  //
  // This rule is impractically strict; we want to at least ignore CVR
  // qualifiers. Distinguishing by CVR qualifiers would make it UB to
  // e.g. cast a `char **` to `const char * const *` and dereference it,
  // which is too common and useful to invalidate. C++'s similar types
  // rule permits qualifier differences in these nested positions; in fact,
  // C++ even allows that cast as an implicit conversion.
  //
  // Other qualifiers could theoretically be distinguished, especially if
  // they involve a significant representation difference.  We don't
  // currently do so, however.
  //
  // Computing the pointee type string recursively is implicitly more
  // forgiving than the standards require.  Effectively, we are turning
  // the question "are these types compatible/similar" into "are
  // accesses to these types allowed to alias".  In both C and C++,
  // the latter question has special carve-outs for signedness
  // mismatches that only apply at the top level.  As a result, we are
  // allowing e.g. `int *` l-values to access `unsigned *` objects.
  if (ty->isPointerType() || ty->isReferenceType()) {
    auto anyPtr = createScalarTypeNode("any pointer", getChar(), size);
    if (!codeGenOpts.PointerTBAA)
      return anyPtr;
    llvm_unreachable("NYI");
  }

  // Accesses to arrays are accesses to objects of their element types.
  if (codeGenOpts.NewStructPathTBAA && ty->isArrayType())
    return getTypeInfo(cast<ArrayType>(ty)->getElementType());

  // Enum types are distinct types. In C++ they have "underlying types",
  // however they aren't related for TBAA.
  if (const EnumType *ety = dyn_cast<EnumType>(ty)) {
    if (!features.CPlusPlus)
      return getTypeInfo(ety->getDecl()->getIntegerType());

    // In C++ mode, types have linkage, so we can rely on the ODR and
    // on their mangled names, if they're external.
    // TODO: Is there a way to get a program-wide unique name for a
    // decl with local linkage or no linkage?
    if (!ety->getDecl()->isExternallyVisible())
      return getChar();

    SmallString<256> outName;
    llvm::raw_svector_ostream out(outName);
    types.getCXXABI().getMangleContext().mangleCanonicalTypeName(
        QualType(ety, 0), out);
    return createScalarTypeNode(outName, getChar(), size);
  }

  if (const auto *eit = dyn_cast<BitIntType>(ty)) {
    SmallString<256> outName;
    llvm::raw_svector_ostream out(outName);
    // Don't specify signed/unsigned since integer types can alias despite sign
    // differences.
    out << "_BitInt(" << eit->getNumBits() << ')';
    return createScalarTypeNode(outName, getChar(), size);
  }

  // For now, handle any other kind of type conservatively.
  return getChar();
}
cir::TBAAAttr CIRGenTBAA::getTypeInfo(clang::QualType qty) {
  // At -O0 or relaxed aliasing, TBAA is not emitted for regular types.
  if (codeGenOpts.OptimizationLevel == 0 || codeGenOpts.RelaxedAliasing) {
    return nullptr;
  }

  // If the type has the may_alias attribute (even on a typedef), it is
  // effectively in the general char alias class.
  if (typeHasMayAlias(qty)) {
    return getChar();
  }
  // We need this function to not fall back to returning the "omnipotent char"
  // type node for aggregate and union types. Otherwise, any dereference of an
  // aggregate will result into the may-alias access descriptor, meaning all
  // subsequent accesses to direct and indirect members of that aggregate will
  // be considered may-alias too.
  // TODO: Combine getTypeInfo() and getValidBaseTypeInfo() into a single
  // function.
  if (isValidBaseType(qty)) {
    return getValidBaseTypeInfo(qty);
  }

  const clang::Type *ty = context.getCanonicalType(qty).getTypePtr();
  if (auto attr = metadataCache[ty]) {
    return attr;
  }

  // Note that the following helper call is allowed to add new nodes to the
  // cache, which invalidates all its previously obtained iterators. So we
  // first generate the node for the type and then add that node to the cache.
  auto typeNode = getTypeInfoHelper(ty);
  return metadataCache[ty] = typeNode;
}

TBAAAccessInfo CIRGenTBAA::getAccessInfo(clang::QualType accessType) {
  // Pointee values may have incomplete types, but they shall never be
  // dereferenced.
  if (accessType->isIncompleteType()) {
    return TBAAAccessInfo::getIncompleteInfo();
  }

  if (typeHasMayAlias(accessType)) {
    return TBAAAccessInfo::getMayAliasInfo();
  }

  uint64_t size = context.getTypeSizeInChars(accessType).getQuantity();
  return TBAAAccessInfo(getTypeInfo(accessType), size);
}

TBAAAccessInfo CIRGenTBAA::getVTablePtrAccessInfo(mlir::Type vtablePtrType) {
  // TODO(cir): support vtable ptr
  return TBAAAccessInfo();
}

mlir::ArrayAttr CIRGenTBAA::getTBAAStructInfo(clang::QualType qty) {
  return mlir::ArrayAttr::get(ctx, {});
}

cir::TBAAAttr CIRGenTBAA::getBaseTypeInfo(clang::QualType qty) {
  return isValidBaseType(qty) ? getValidBaseTypeInfo(qty) : nullptr;
}

mlir::ArrayAttr CIRGenTBAA::getAccessTagInfo(TBAAAccessInfo tbaaInfo) {
  assert(!tbaaInfo.isIncomplete() &&
         "Access to an object of an incomplete type!");

  if (tbaaInfo.isMayAlias())
    tbaaInfo = TBAAAccessInfo(getChar(), tbaaInfo.size);

  if (!tbaaInfo.accessType) {
    return nullptr;
  }

  if (!codeGenOpts.StructPathTBAA)
    tbaaInfo = TBAAAccessInfo(tbaaInfo.accessType, tbaaInfo.size);

  // auto &N = AccessTagMetadataCache[tbaaInfo];
  if (!tbaaInfo.baseType) {
    tbaaInfo.baseType = tbaaInfo.accessType;
    assert(!tbaaInfo.offset &&
           "Nonzero offset for an access with no base type!");
  }
  if (codeGenOpts.NewStructPathTBAA) {
    llvm_unreachable("NYI");
  }
  if (tbaaInfo.baseType == tbaaInfo.accessType) {
    return mlir::ArrayAttr::get(ctx, {tbaaInfo.accessType});
  }
  return mlir::ArrayAttr::get(
      ctx, {cir::TBAATagAttr::get(ctx, tbaaInfo.baseType, tbaaInfo.accessType,
                                  tbaaInfo.offset, tbaaInfo.size)});
}

cir::TBAAAttr CIRGenTBAA::getBaseTypeInfoHelper(const clang::Type *ty) {
  using namespace clang;
  if (auto *tty = dyn_cast<RecordType>(ty)) {
    const RecordDecl *rd = tty->getDecl()->getDefinition();
    const ASTRecordLayout &layout = context.getASTRecordLayout(rd);
    SmallVector<cir::TBAAMemberAttr, 4> fields;
    if (const CXXRecordDecl *cxxrd = dyn_cast<CXXRecordDecl>(rd)) {
      // Handle C++ base classes. Non-virtual bases can treated a kind of
      // field. Virtual bases are more complex and omitted, but avoid an
      // incomplete view for NewStructPathTBAA.
      if (codeGenOpts.NewStructPathTBAA && cxxrd->getNumVBases() != 0)
        return nullptr;
      for (const CXXBaseSpecifier &cxxBaseSpecifier : cxxrd->bases()) {
        if (cxxBaseSpecifier.isVirtual())
          continue;
        QualType baseQTy = cxxBaseSpecifier.getType();
        const CXXRecordDecl *baseRD = baseQTy->getAsCXXRecordDecl();
        if (baseRD->isEmpty())
          continue;
        auto typeNode = isValidBaseType(baseQTy) ? getValidBaseTypeInfo(baseQTy)
                                                 : getTypeInfo(baseQTy);
        if (!typeNode)
          return nullptr;
        uint64_t offset = layout.getBaseClassOffset(baseRD).getQuantity();
        uint64_t size =
            context.getASTRecordLayout(baseRD).getDataSize().getQuantity();
        fields.push_back(cir::TBAAMemberAttr::get(typeNode, offset, size));
      }
      // The order in which base class subobjects are allocated is unspecified,
      // so may differ from declaration order. In particular, Itanium ABI will
      // allocate a primary base first.
      // Since we exclude empty subobjects, the objects are not overlapping and
      // their offsets are unique.
      llvm::sort(fields, [](const cir::TBAAMemberAttr &lhs,
                            const cir::TBAAMemberAttr &rhs) {
        return lhs.getOffset() < rhs.getOffset();
      });
    }
    for (FieldDecl *field : rd->fields()) {
      if (field->isZeroSize(context) || field->isUnnamedBitField())
        continue;
      QualType fieldQTy = field->getType();
      auto typeNode = isValidBaseType(fieldQTy) ? getValidBaseTypeInfo(fieldQTy)
                                                : getTypeInfo(fieldQTy);
      if (!typeNode)
        return nullptr;

      uint64_t bitOffset = layout.getFieldOffset(field->getFieldIndex());
      uint64_t offset = context.toCharUnitsFromBits(bitOffset).getQuantity();
      uint64_t size = context.getTypeSizeInChars(fieldQTy).getQuantity();
      fields.push_back(cir::TBAAMemberAttr::get(typeNode, offset, size));
    }

    SmallString<256> outName;
    if (features.CPlusPlus) {
      // Don't use the mangler for C code.
      llvm::raw_svector_ostream out(outName);
      types.getCXXABI().getMangleContext().mangleCanonicalTypeName(
          QualType(ty, 0), out);
    } else {
      outName = rd->getName();
    }

    if (codeGenOpts.NewStructPathTBAA) {
      llvm_unreachable("NYI");
    }
    return cir::TBAAStructTypeDescriptorAttr::get(ctx, outName, fields);
  }
  return nullptr;
}

cir::TBAAAttr CIRGenTBAA::getValidBaseTypeInfo(clang::QualType qty) {
  assert(isValidBaseType(qty) && "Must be a valid base type");

  const clang::Type *ty = context.getCanonicalType(qty).getTypePtr();

  // nullptr is a valid value in the cache, so use find rather than []
  auto iter = baseTypeMetadataCache.find(ty);
  if (iter != baseTypeMetadataCache.end())
    return iter->second;

  // First calculate the metadata, before recomputing the insertion point, as
  // the helper can recursively call us.
  auto typeNode = getBaseTypeInfoHelper(ty);
  LLVM_ATTRIBUTE_UNUSED auto inserted =
      baseTypeMetadataCache.insert({ty, typeNode});
  assert(inserted.second && "BaseType metadata was already inserted");

  return typeNode;
}
TBAAAccessInfo CIRGenTBAA::mergeTBAAInfoForCast(TBAAAccessInfo sourceInfo,
                                                TBAAAccessInfo targetInfo) {
  if (sourceInfo.isMayAlias() || targetInfo.isMayAlias())
    return TBAAAccessInfo::getMayAliasInfo();
  return targetInfo;
}

TBAAAccessInfo
CIRGenTBAA::mergeTBAAInfoForConditionalOperator(TBAAAccessInfo infoA,
                                                TBAAAccessInfo infoB) {
  if (infoA == infoB)
    return infoA;

  if (!infoA || !infoB)
    return TBAAAccessInfo();

  if (infoA.isMayAlias() || infoB.isMayAlias())
    return TBAAAccessInfo::getMayAliasInfo();

  // TODO: Implement the rest of the logic here. For example, two accesses
  // with same final access types result in an access to an object of that
  // final access type regardless of their base types.
  return TBAAAccessInfo::getMayAliasInfo();
}

TBAAAccessInfo
CIRGenTBAA::mergeTBAAInfoForMemoryTransfer(TBAAAccessInfo destInfo,
                                           TBAAAccessInfo srcInfo) {
  if (destInfo == srcInfo)
    return destInfo;

  if (!destInfo || !srcInfo)
    return TBAAAccessInfo();

  if (destInfo.isMayAlias() || srcInfo.isMayAlias())
    return TBAAAccessInfo::getMayAliasInfo();

  // TODO: Implement the rest of the logic here. For example, two accesses
  // with same final access types result in an access to an object of that
  // final access type regardless of their base types.
  return TBAAAccessInfo::getMayAliasInfo();
}

} // namespace clang::CIRGen