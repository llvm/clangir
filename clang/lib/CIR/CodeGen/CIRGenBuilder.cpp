//===-- CIRGenBuilder.cpp - CIRBuilder implementation  ------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "CIRGenBuilder.h"

namespace cir {

mlir::Value CIRGenBuilderTy::maybeBuildArrayDecay(mlir::Location loc,
                                                  mlir::Value arrayPtr,
                                                  mlir::Type eltTy) {
  auto arrayPtrTy =
      ::mlir::dyn_cast<::mlir::cir::PointerType>(arrayPtr.getType());
  assert(arrayPtrTy && "expected pointer type");
  auto arrayTy =
      ::mlir::dyn_cast<::mlir::cir::ArrayType>(arrayPtrTy.getPointee());

  if (arrayTy) {
    mlir::cir::PointerType flatPtrTy =
        mlir::cir::PointerType::get(getContext(), arrayTy.getEltType());
    return create<mlir::cir::CastOp>(
        loc, flatPtrTy, mlir::cir::CastKind::array_to_ptrdecay, arrayPtr);
  }

  assert(arrayPtrTy.getPointee() == eltTy &&
         "flat pointee type must match original array element type");
  return arrayPtr;
}

mlir::Value CIRGenBuilderTy::buildArrayAccessOp(
    mlir::Location arrayLocBegin, mlir::Location arrayLocEnd,
    mlir::Value arrayPtr, mlir::Type eltTy, mlir::Value idx, bool shouldDecay) {
  mlir::Value basePtr = arrayPtr;
  if (shouldDecay)
    basePtr = maybeBuildArrayDecay(arrayLocBegin, arrayPtr, eltTy);
  mlir::Type flatPtrTy = basePtr.getType();
  return create<mlir::cir::PtrStrideOp>(arrayLocEnd, flatPtrTy, basePtr, idx);
}
}; // namespace cir
