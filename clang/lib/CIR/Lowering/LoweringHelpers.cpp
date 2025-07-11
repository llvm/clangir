//====- LoweringHelpers.cpp - Lowering helper functions -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains helper functions for lowering from CIR to LLVM or MLIR.
//
//===----------------------------------------------------------------------===//
#include "clang/CIR/LoweringHelpers.h"

mlir::DenseElementsAttr
convertStringAttrToDenseElementsAttr(cir::ConstArrayAttr attr,
                                     mlir::Type type) {
  auto values = llvm::SmallVector<mlir::APInt, 8>{};
  auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getElts());
  assert(stringAttr && "expected string attribute here");
  for (auto element : stringAttr)
    values.push_back({8, (uint64_t)element});
  auto arrayTy = mlir::dyn_cast<cir::ArrayType>(attr.getType());
  assert(arrayTy && "String attribute must have an array type");
  if (arrayTy.getSize() != stringAttr.size())
    llvm_unreachable("array type of the length not equal to that of the string "
                     "attribute is not supported yet");
  return mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({(int64_t)values.size()}, type),
      llvm::ArrayRef(values));
}

template <> mlir::APInt getZeroInitFromType(mlir::Type Ty) {
  assert(mlir::isa<cir::IntType>(Ty) && "expected int type");
  auto IntTy = mlir::cast<cir::IntType>(Ty);
  return mlir::APInt::getZero(IntTy.getWidth());
}

template <> mlir::APFloat getZeroInitFromType(mlir::Type Ty) {
  assert((mlir::isa<cir::SingleType, cir::DoubleType>(Ty)) &&
         "only float and double supported");
  if (Ty.isF32() || mlir::isa<cir::SingleType>(Ty))
    return mlir::APFloat(0.f);
  if (Ty.isF64() || mlir::isa<cir::DoubleType>(Ty))
    return mlir::APFloat(0.0);
  llvm_unreachable("NYI");
}

/// \param attr the ConstArrayAttr to convert
/// \param values the output parameter, the values array to fill
/// \param currentDims the shpae of tensor we're going to convert to
/// \param dimIndex the current dimension we're processing
/// \param currentIndex the current index in the values array
template <typename AttrTy, typename StorageTy>
void convertToDenseElementsAttrImpl(
    cir::ConstArrayAttr attr, llvm::SmallVectorImpl<StorageTy> &values,
    const llvm::SmallVectorImpl<int64_t> &currentDims, int64_t dimIndex,
    int64_t currentIndex) {
  if (auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getElts())) {
    if (auto arrayType = mlir::dyn_cast<cir::ArrayType>(attr.getType())) {
      for (auto element : stringAttr) {
        auto intAttr = cir::IntAttr::get(arrayType.getElementType(), element);
        values[currentIndex++] = mlir::dyn_cast<AttrTy>(intAttr).getValue();
      }
      return;
    }
  }

  dimIndex++;
  std::size_t elementsSizeInCurrentDim = 1;
  for (std::size_t i = dimIndex; i < currentDims.size(); i++)
    elementsSizeInCurrentDim *= currentDims[i];

  auto arrayAttr = mlir::cast<mlir::ArrayAttr>(attr.getElts());
  for (auto eltAttr : arrayAttr) {
    if (auto valueAttr = mlir::dyn_cast<AttrTy>(eltAttr)) {
      values[currentIndex++] = valueAttr.getValue();
      continue;
    }

    if (auto subArrayAttr = mlir::dyn_cast<cir::ConstArrayAttr>(eltAttr)) {
      convertToDenseElementsAttrImpl<AttrTy>(subArrayAttr, values, currentDims,
                                             dimIndex, currentIndex);
      currentIndex += elementsSizeInCurrentDim;
      continue;
    }

    if (mlir::isa<cir::ZeroAttr, cir::UndefAttr>(eltAttr)) {
      currentIndex += elementsSizeInCurrentDim;
      continue;
    }

    llvm_unreachable("unknown element in ConstArrayAttr");
  }
}

template <typename AttrTy, typename StorageTy>
void convertToDenseElementsAttrImpl(
    cir::ConstVectorAttr attr, llvm::SmallVectorImpl<StorageTy> &values,
    const llvm::SmallVectorImpl<int64_t> &currentDims, int64_t dimIndex,
    int64_t currentIndex) {
  dimIndex++;
  std::size_t elementsSizeInCurrentDim = 1;
  for (std::size_t i = dimIndex; i < currentDims.size(); i++)
    elementsSizeInCurrentDim *= currentDims[i];

  auto arrayAttr = mlir::cast<mlir::ArrayAttr>(attr.getElts());
  for (auto eltAttr : arrayAttr) {
    if (auto valueAttr = mlir::dyn_cast<AttrTy>(eltAttr)) {
      values[currentIndex++] = valueAttr.getValue();
      continue;
    }

    if (auto subArrayAttr = mlir::dyn_cast<cir::ConstArrayAttr>(eltAttr)) {
      convertToDenseElementsAttrImpl<AttrTy>(subArrayAttr, values, currentDims,
                                             dimIndex, currentIndex);
      currentIndex += elementsSizeInCurrentDim;
      continue;
    }

    if (mlir::isa<cir::ZeroAttr, cir::UndefAttr>(eltAttr)) {
      currentIndex += elementsSizeInCurrentDim;
      continue;
    }

    llvm_unreachable("unknown element in ConstArrayAttr");
  }
}

template <typename AttrTy, typename StorageTy>
void convertToDenseElementsAttrImpl(
    cir::ComplexAttr attr, llvm::SmallVectorImpl<StorageTy> &values,
    const llvm::SmallVectorImpl<int64_t> &currentDims, int64_t dimIndex,
    int64_t currentIndex) {
  dimIndex++;
  std::size_t elementsSizeInCurrentDim = 1;
  for (std::size_t i = dimIndex; i < currentDims.size(); i++)
    elementsSizeInCurrentDim *= currentDims[i];

  auto attrArray =
      mlir::ArrayAttr::get(attr.getContext(), {attr.getImag(), attr.getReal()});
  for (auto eltAttr : attrArray) {
    if (auto valueAttr = mlir::dyn_cast<AttrTy>(eltAttr)) {
      values[currentIndex++] = valueAttr.getValue();
      continue;
    }

    if (mlir::isa<cir::ZeroAttr, cir::UndefAttr>(eltAttr)) {
      currentIndex += elementsSizeInCurrentDim;
      continue;
    }

    llvm_unreachable("unknown element in ComplexAttr");
  }
}

template <typename AttrTy, typename StorageTy>
mlir::DenseElementsAttr convertToDenseElementsAttr(
    cir::ConstArrayAttr attr, const llvm::SmallVectorImpl<int64_t> &dims,
    mlir::Type elementType, mlir::Type convertedElementType) {
  unsigned vector_size = 1;
  for (auto dim : dims)
    vector_size *= dim;
  auto values = llvm::SmallVector<StorageTy, 8>(
      vector_size, getZeroInitFromType<StorageTy>(elementType));
  convertToDenseElementsAttrImpl<AttrTy>(attr, values, dims, /*currentDim=*/0,
                                         /*initialIndex=*/0);
  return mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get(dims, convertedElementType),
      llvm::ArrayRef(values));
}

template <typename AttrTy, typename StorageTy>
mlir::DenseElementsAttr convertToDenseElementsAttr(
    cir::ConstVectorAttr attr, const llvm::SmallVectorImpl<int64_t> &dims,
    mlir::Type elementType, mlir::Type convertedElementType) {
  unsigned vector_size = 1;
  for (auto dim : dims)
    vector_size *= dim;
  auto values = llvm::SmallVector<StorageTy, 8>(
      vector_size, getZeroInitFromType<StorageTy>(elementType));
  convertToDenseElementsAttrImpl<AttrTy>(attr, values, dims, /*currentDim=*/0,
                                         /*initialIndex=*/0);
  return mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get(dims, convertedElementType),
      llvm::ArrayRef(values));
}

template <typename AttrTy, typename StorageTy>
mlir::DenseElementsAttr convertToDenseElementsAttr(
    cir::ComplexAttr attr, const llvm::SmallVectorImpl<int64_t> &dims,
    mlir::Type elementType, mlir::Type convertedElementType) {
  unsigned array_size = 2;
  auto values = llvm::SmallVector<StorageTy, 8>(
      array_size, getZeroInitFromType<StorageTy>(elementType));
  convertToDenseElementsAttrImpl<AttrTy>(attr, values, dims, /*currentDim=*/0,
                                         /*initialIndex=*/0);
  return mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get(dims, convertedElementType),
      llvm::ArrayRef(values));
}

std::optional<mlir::Attribute>
lowerConstArrayAttr(cir::ConstArrayAttr constArr,
                    const mlir::TypeConverter *converter) {

  // Ensure ConstArrayAttr has a type.
  auto typedConstArr = mlir::dyn_cast<mlir::TypedAttr>(constArr);
  assert(typedConstArr && "cir::ConstArrayAttr is not a mlir::TypedAttr");

  // Ensure ConstArrayAttr type is a ArrayType.
  auto cirArrayType = mlir::dyn_cast<cir::ArrayType>(typedConstArr.getType());
  assert(cirArrayType && "cir::ConstArrayAttr is not a cir::ArrayType");

  // Is a ConstArrayAttr with an cir::ArrayType: fetch element type.
  mlir::Type type = cirArrayType;
  auto dims = llvm::SmallVector<int64_t, 2>{};
  while (auto arrayType = mlir::dyn_cast<cir::ArrayType>(type)) {
    dims.push_back(arrayType.getSize());
    type = arrayType.getElementType();
  }

  if (mlir::isa<mlir::StringAttr>(constArr.getElts()))
    return convertStringAttrToDenseElementsAttr(constArr,
                                                converter->convertType(type));
  if (mlir::isa<cir::IntType>(type))
    return convertToDenseElementsAttr<cir::IntAttr, mlir::APInt>(
        constArr, dims, type, converter->convertType(type));
  if (mlir::isa<cir::FPTypeInterface>(type))
    return convertToDenseElementsAttr<cir::FPAttr, mlir::APFloat>(
        constArr, dims, type, converter->convertType(type));

  return std::nullopt;
}

std::optional<mlir::Attribute>
lowerConstComplexAttr(cir::ComplexAttr constComplex,
                      const mlir::TypeConverter *converter) {

  // Ensure ComplexAttr has a type.
  auto typedConstArr = mlir::dyn_cast<mlir::TypedAttr>(constComplex);
  assert(typedConstArr && "cir::ComplexAttr is not a mlir::TypedAttr");

  mlir::Type type = constComplex.getType();
  auto dims = llvm::SmallVector<int64_t, 2>{2};

  if (mlir::isa<cir::IntType>(type))
    return convertToDenseElementsAttr<cir::IntAttr, mlir::APInt>(
        constComplex, dims, type, converter->convertType(type));
  if (mlir::isa<cir::FPTypeInterface>(type))
    return convertToDenseElementsAttr<cir::FPAttr, mlir::APFloat>(
        constComplex, dims, type, converter->convertType(type));

  return std::nullopt;
}

std::optional<mlir::Attribute>
lowerConstVectorAttr(cir::ConstVectorAttr constArr,
                     const mlir::TypeConverter *converter) {

  // Ensure ConstArrayAttr has a type.
  auto typedConstArr = mlir::dyn_cast<mlir::TypedAttr>(constArr);
  assert(typedConstArr && "cir::ConstArrayAttr is not a mlir::TypedAttr");

  // Ensure ConstArrayAttr type is a ArrayType.
  auto cirArrayType = mlir::dyn_cast<cir::VectorType>(typedConstArr.getType());
  assert(cirArrayType && "cir::ConstArrayAttr is not a cir::ArrayType");

  // Is a ConstArrayAttr with an cir::ArrayType: fetch element type.
  mlir::Type type = cirArrayType;
  auto dims = llvm::SmallVector<int64_t, 2>{};
  while (auto arrayType = mlir::dyn_cast<cir::ArrayType>(type)) {
    dims.push_back(arrayType.getSize());
    type = arrayType.getElementType();
  }

  if (mlir::isa<cir::IntType>(type))
    return convertToDenseElementsAttr<cir::IntAttr, mlir::APInt>(
        constArr, dims, type, converter->convertType(type));
  if (mlir::isa<cir::FPTypeInterface>(type))
    return convertToDenseElementsAttr<cir::FPAttr, mlir::APFloat>(
        constArr, dims, type, converter->convertType(type));

  return std::nullopt;
}
