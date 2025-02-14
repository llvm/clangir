#include "LowerToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/AST/RecordLayout.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

namespace cir {
namespace direct {

class CIRToLLVMTBAAAttrLowering {
public:
  CIRToLLVMTBAAAttrLowering(mlir::MLIRContext *mlirContext,
                            const clang::LangOptions &features)
      : mlirContext(mlirContext), features(features) {}
  mlir::LLVM::TBAARootAttr getRoot() {
    if (features.CPlusPlus) {
      return createTBAARoot("Simple C++ TBAA");
    } else {
      return createTBAARoot("Simple C/C++ TBAA");
    }
  }

  mlir::LLVM::TBAATypeDescriptorAttr getChar() {
    return createScalarTypeNode("omnipotent char", getRoot(), 0);
  }

  mlir::LLVM::TBAATypeDescriptorAttr
  createScalarTypeNode(llvm::StringRef typeName,
                       mlir::LLVM::TBAANodeAttr parent, int64_t size) {
    llvm::SmallVector<mlir::LLVM::TBAAMemberAttr, 2> members;
    members.push_back(mlir::LLVM::TBAAMemberAttr::get(mlirContext, parent, 0));
    return mlir::LLVM::TBAATypeDescriptorAttr::get(
        mlirContext, typeName,
        llvm::ArrayRef<mlir::LLVM::TBAAMemberAttr>(members));
  }

private:
  mlir::LLVM::TBAARootAttr createTBAARoot(llvm::StringRef name) {
    return mlir::LLVM::TBAARootAttr::get(
        mlirContext, mlir::StringAttr::get(mlirContext, name));
  }

protected:
  mlir::MLIRContext *mlirContext;
  const clang::LangOptions &features;
};

class CIRToLLVMTBAAScalarAttrLowering : public CIRToLLVMTBAAAttrLowering {
public:
  CIRToLLVMTBAAScalarAttrLowering(mlir::MLIRContext *mlirContext,
                                  const clang::LangOptions &features)
      : CIRToLLVMTBAAAttrLowering(mlirContext, features) {}
  mlir::LLVM::TBAATypeDescriptorAttr
  lowerScalarType(cir::TBAAScalarAttr scalarAttr) {
    mlir::DataLayout layout;
    auto size = layout.getTypeSize(scalarAttr.getType());
    return createScalarTypeNode(scalarAttr.getId(), getChar(), size);
  }
};

class CIRToLLVMTBAAStructAttrLowering : public CIRToLLVMTBAAAttrLowering {
public:
  CIRToLLVMTBAAStructAttrLowering(mlir::MLIRContext *mlirContext,
                                  clang::ASTContext &astContext,
                                  const clang::CodeGenOptions &codeGenOpts,
                                  const clang::LangOptions &features)
      : CIRToLLVMTBAAAttrLowering(mlirContext, features),
        astContext(astContext), codeGenOpts(codeGenOpts) {}

  mlir::LLVM::TBAATypeDescriptorAttr lowerStructType(const clang::Type *ty) {
    return getBaseTypeInfoHelper(ty);
  }

  mlir::LLVM::TBAATypeDescriptorAttr getTypeInfoHelper(const clang::Type *ty) {
    using namespace clang;
    uint64_t size = astContext.getTypeSizeInChars(ty).getQuantity();
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
        return getTypeInfo(astContext.ShortTy);
      case BuiltinType::UInt:
        return getTypeInfo(astContext.IntTy);
      case BuiltinType::ULong:
        return getTypeInfo(astContext.LongTy);
      case BuiltinType::ULongLong:
        return getTypeInfo(astContext.LongLongTy);
      case BuiltinType::UInt128:
        return getTypeInfo(astContext.Int128Ty);

      case BuiltinType::UShortFract:
        return getTypeInfo(astContext.ShortFractTy);
      case BuiltinType::UFract:
        return getTypeInfo(astContext.FractTy);
      case BuiltinType::ULongFract:
        return getTypeInfo(astContext.LongFractTy);

      case BuiltinType::SatUShortFract:
        return getTypeInfo(astContext.SatShortFractTy);
      case BuiltinType::SatUFract:
        return getTypeInfo(astContext.SatFractTy);
      case BuiltinType::SatULongFract:
        return getTypeInfo(astContext.SatLongFractTy);

      case BuiltinType::UShortAccum:
        return getTypeInfo(astContext.ShortAccumTy);
      case BuiltinType::UAccum:
        return getTypeInfo(astContext.AccumTy);
      case BuiltinType::ULongAccum:
        return getTypeInfo(astContext.LongAccumTy);

      case BuiltinType::SatUShortAccum:
        return getTypeInfo(astContext.SatShortAccumTy);
      case BuiltinType::SatUAccum:
        return getTypeInfo(astContext.SatAccumTy);
      case BuiltinType::SatULongAccum:
        return getTypeInfo(astContext.SatLongAccumTy);

      // Treat all other builtin types as distinct types. This includes
      // treating wchar_t, char16_t, and char32_t as distinct from their
      // "underlying types".
      default:
        return createScalarTypeNode(bty->getName(features), getChar(), size);
      }
    }

    // C++1z [basic.lval]p10: "If a program attempts to access the stored value
    // of an object through a glvalue of other than one of the following types
    // the behavior is undefined: [...] a char, unsigned char, or std::byte
    // type."
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
      assert(!cir::MissingFeatures::tbaaPointer());
      return nullptr;
    }

    // Accesses to arrays are accesses to objects of their element types.
    if (codeGenOpts.NewStructPathTBAA && ty->isArrayType())
      return getTypeInfo(cast<clang::ArrayType>(ty)->getElementType());

    // Enum types are distinct types. In C++ they have "underlying types",
    // however they aren't related for TBAA.
    if (const clang::EnumType *ety = dyn_cast<clang::EnumType>(ty)) {
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
      mangleCanonicalTypeName(ety, out);
      return createScalarTypeNode(outName, getChar(), size);
    }

    if (const auto *eit = dyn_cast<BitIntType>(ty)) {
      SmallString<256> outName;
      llvm::raw_svector_ostream out(outName);
      // Don't specify signed/unsigned since integer types can alias despite
      // sign differences.
      out << "_BitInt(" << eit->getNumBits() << ')';
      return createScalarTypeNode(outName, getChar(), size);
    }

    // For now, handle any other kind of type conservatively.
    return getChar();
  }
  mlir::LLVM::TBAATypeDescriptorAttr getTypeInfo(const clang::Type *ty) {
    return getTypeInfo(clang::QualType(ty, 0));
  }
  mlir::LLVM::TBAATypeDescriptorAttr getTypeInfo(clang::QualType qty) {
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

    const clang::Type *ty = astContext.getCanonicalType(qty).getTypePtr();
    if (auto attr = metadataCache[ty]) {
      return attr;
    }

    // Note that the following helper call is allowed to add new nodes to the
    // cache, which invalidates all its previously obtained iterators. So we
    // first generate the node for the type and then add that node to the cache.
    auto typeNode = getTypeInfoHelper(ty);
    return metadataCache[ty] = typeNode;
  }

private:
  mlir::LLVM::TBAATypeDescriptorAttr
  getBaseTypeInfoHelper(const clang::Type *ty) {
    using namespace clang;
    if (auto *tty = mlir::dyn_cast<clang::RecordType>(ty)) {
      const clang::RecordDecl *rd = tty->getDecl()->getDefinition();
      const ASTRecordLayout &layout = astContext.getASTRecordLayout(rd);
      SmallVector<mlir::LLVM::TBAAMemberAttr, 4> fields;
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
          auto typeNode = isValidBaseType(baseQTy)
                              ? getValidBaseTypeInfo(baseQTy)
                              : getTypeInfo(baseQTy);
          if (!typeNode)
            return nullptr;
          uint64_t offset = layout.getBaseClassOffset(baseRD).getQuantity();
          [[maybe_unused]] uint64_t size =
              astContext.getASTRecordLayout(baseRD).getDataSize().getQuantity();
          fields.push_back(mlir::LLVM::TBAAMemberAttr::get(typeNode, offset));
        }
        // The order in which base class subobjects are allocated is
        // unspecified, so may differ from declaration order. In particular,
        // Itanium ABI will allocate a primary base first. Since we exclude
        // empty subobjects, the objects are not overlapping and their offsets
        // are unique.
        llvm::sort(fields, [](const mlir::LLVM::TBAAMemberAttr &lhs,
                              const mlir::LLVM::TBAAMemberAttr &rhs) {
          return lhs.getOffset() < rhs.getOffset();
        });
      }
      for (FieldDecl *field : rd->fields()) {
        if (field->isZeroSize(astContext) || field->isUnnamedBitField())
          continue;
        QualType fieldQTy = field->getType();
        auto typeNode = isValidBaseType(fieldQTy)
                            ? getValidBaseTypeInfo(fieldQTy)
                            : getTypeInfo(fieldQTy);
        if (!typeNode)
          return nullptr;

        uint64_t bitOffset = layout.getFieldOffset(field->getFieldIndex());
        uint64_t offset =
            astContext.toCharUnitsFromBits(bitOffset).getQuantity();
        [[maybe_unused]] uint64_t size =
            astContext.getTypeSizeInChars(fieldQTy).getQuantity();
        fields.push_back(mlir::LLVM::TBAAMemberAttr::get(typeNode, offset));
      }

      SmallString<256> outName;
      if (features.CPlusPlus) {
        // Don't use the mangler for C code.
        llvm::raw_svector_ostream out(outName);
        mangleCanonicalTypeName(ty, out);
      } else {
        outName = rd->getName();
      }

      if (codeGenOpts.NewStructPathTBAA) {
        assert(!cir::MissingFeatures::tbaaNewStructPath());
        return nullptr;
      }
      return mlir::LLVM::TBAATypeDescriptorAttr::get(mlirContext, outName,
                                                     fields);
    }
    return nullptr;
  }

  mlir::LLVM::TBAATypeDescriptorAttr getValidBaseTypeInfo(clang::QualType qty) {
    assert(isValidBaseType(qty) && "Must be a valid base type");

    const clang::Type *ty = astContext.getCanonicalType(qty).getTypePtr();

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
  static bool typeHasMayAlias(clang::QualType qty) {
    // Tagged types have declarations, and therefore may have attributes.
    if (auto *td = qty->getAsTagDecl())
      if (td->hasAttr<clang::MayAliasAttr>())
        return true;

    // Also look for may_alias as a declaration attribute on a typedef.
    // FIXME: We should follow GCC and model may_alias as a type attribute
    // rather than as a declaration attribute.
    // auto
    while (auto *tt = qty->getAs<clang::TypedefType>()) {
      if (tt->getDecl()->hasAttr<clang::MayAliasAttr>())
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

  void mangleCanonicalTypeName(const clang::Type *qty, llvm::raw_ostream &out) {
    astContext.createMangleContext()->mangleCanonicalTypeName(
        clang::QualType(qty, 0), out);
  }

  clang::ASTContext &astContext;
  const clang::CodeGenOptions &codeGenOpts;
  llvm::DenseMap<const clang::Type *, mlir::LLVM::TBAATypeDescriptorAttr>
      metadataCache;
  llvm::DenseMap<const clang::Type *, mlir::LLVM::TBAATypeDescriptorAttr>
      baseTypeMetadataCache;
};

mlir::ArrayAttr lowerCIRTBAAAttr(mlir::Attribute tbaa,
                                 mlir::ConversionPatternRewriter &rewriter,
                                 cir::LowerModule *lowerMod) {
  auto *ctx = rewriter.getContext();
  CIRToLLVMTBAAScalarAttrLowering scalarLower(
      ctx, lowerMod->getContext().getLangOpts());
  if (auto charAttr = mlir::dyn_cast<cir::TBAAOmnipotentCharAttr>(tbaa)) {
    auto accessType = scalarLower.getChar();
    auto tag = mlir::LLVM::TBAATagAttr::get(accessType, accessType, 0);
    return mlir::ArrayAttr::get(ctx, {tag});
  }
  if (auto scalarAttr = mlir::dyn_cast<cir::TBAAScalarAttr>(tbaa)) {
    auto accessType = scalarLower.lowerScalarType(scalarAttr);
    auto tag = mlir::LLVM::TBAATagAttr::get(accessType, accessType, 0);
    return mlir::ArrayAttr::get(ctx, {tag});
  }
  if (auto tbaaTag = mlir::dyn_cast<cir::TBAATagAttr>(tbaa)) {
    mlir::LLVM::TBAATypeDescriptorAttr accessType;
    if (mlir::isa<cir::TBAAOmnipotentCharAttr>(tbaaTag.getAccess())) {
      accessType = scalarLower.getChar();
    } else if (auto scalarAttr =
                   mlir::dyn_cast<cir::TBAAScalarAttr>(tbaaTag.getAccess())) {
      accessType = scalarLower.lowerScalarType(
          mlir::dyn_cast<cir::TBAAScalarAttr>(tbaaTag.getAccess()));
    } else {
      return nullptr;
    }
    if (auto structAttr =
            mlir::dyn_cast<cir::TBAAStructAttr>(tbaaTag.getBase())) {
      cir::StructType structType =
          mlir::dyn_cast<cir::StructType>(structAttr.getType());
      auto ast = structType.getAst();
      CIRToLLVMTBAAStructAttrLowering structLower(
          rewriter.getContext(), ast.getRawDecl()->getASTContext(),
          lowerMod->getContext().getCodeGenOpts(),
          lowerMod->getContext().getLangOpts());
      auto baseType =
          structLower.lowerStructType(ast.getRawDecl()->getTypeForDecl());
      if (!baseType) {
        return nullptr;
      }

      auto tag = mlir::LLVM::TBAATagAttr::get(baseType, accessType,
                                              tbaaTag.getOffset());
      return mlir::ArrayAttr::get(ctx, {tag});
    }
  }
  return mlir::ArrayAttr();
}

} // namespace direct
} // namespace cir