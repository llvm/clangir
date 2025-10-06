//====- LowerCIRToMLIR.cpp - Lowering from CIR to MLIR --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of CIR operations to MLIR.
//
//===----------------------------------------------------------------------===//

#include "LowerToMLIRHelpers.h"
#define DEBUG_TYPE "cir-lowering"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include <atomic>
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Interfaces/CIRLoopOpInterface.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/CIR/LowerToMLIR.h"
#include "clang/CIR/LoweringHelpers.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TimeProfiler.h"
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

using namespace cir;
using namespace llvm;

namespace cir {

static constexpr llvm::StringLiteral kMemrefReinterpretCastName(
    "memref.reinterpret_cast");

static mlir::Operation *getMemrefReinterpretCastOp(mlir::Value value) {
  mlir::Operation *op = value.getDefiningOp();
  if (!op || !op->getBlock() || !op->getBlock()->getParent())
    return nullptr;
  if (op->getName().getStringRef() != kMemrefReinterpretCastName)
    return nullptr;
  return op;
}

class CIRReturnLowering : public mlir::OpConversionPattern<cir::ReturnOp> {
public:
  using OpConversionPattern<cir::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ReturnOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Use adapted operands which have already been converted to MLIR types
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, adaptor.getOperands());
    return mlir::LogicalResult::success();
  }
};

struct ConvertCIRToMLIRPass
    : public mlir::PassWrapper<ConvertCIRToMLIRPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                    mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::scf::SCFDialect, mlir::math::MathDialect,
                    mlir::vector::VectorDialect, mlir::LLVM::LLVMDialect>();
  }
  void runOnOperation() final;

  StringRef getDescription() const override {
    return "Convert the CIR dialect module to MLIR standard dialects";
  }

  StringRef getArgument() const override { return "cir-to-mlir"; }
};

class CIRCallOpLowering : public mlir::OpConversionPattern<cir::CallOp> {
public:
  using OpConversionPattern<cir::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SmallVector<mlir::Type> types;
    if (mlir::failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), types)))
      return mlir::failure();

    if (!op.isIndirect()) {
      // Currently variadic functions are not supported by the builtin func
      // dialect. For now only basic call to printf are supported by using the
      // llvmir dialect.
      // TODO: remove this and add support for variadic function calls once
      // TODO: supported by the func dialect
      if (op.getCallee()->equals_insensitive("printf")) {
        SmallVector<mlir::Type> operandTypes =
            llvm::to_vector(adaptor.getOperands().getTypes());

        // Drop the initial memref operand type (we replace the memref format
        // string with equivalent llvm.mlir ops)
        operandTypes.erase(operandTypes.begin());

        // Check that the printf attributes can be used in llvmir dialect (i.e
        // they have integer/float type)
        // Attempt a best-effort handling of non LLVM-compatible varargs
        // (e.g. memref<> coming from pointer-to-char) by dropping them.
        // This preserves pipeline progress instead of hard failing.
        bool hasNullType =
            llvm::any_of(operandTypes, [](mlir::Type ty) { return !ty; });
        if (hasNullType) {
          op.emitRemark() << "printf lowering: encountered null converted "
                             "vararg type; conservatively dropping all varargs";
        }
        bool needsSalvage =
            !hasNullType && llvm::any_of(operandTypes, [](mlir::Type ty) {
              return !mlir::LLVM::isCompatibleType(ty);
            });
        if (needsSalvage)
          op.emitRemark() << "printf lowering: attempting to salvage non-LLVM "
                             "varargs (memref -> pointer)";

        // Currently only versions of printf are supported where the format
        // string is defined inside the printf ==> the lowering of the cir ops
        // will match:
        // %global = memref.get_global %frm_str
        // %* = memref.reinterpret_cast (%global, 0)
        if (auto *reinterpretCastOp =
                getMemrefReinterpretCastOp(adaptor.getOperands()[0])) {
          if (auto getGlobalOp =
                  reinterpretCastOp->getOperand(0)
                      .getDefiningOp<mlir::memref::GetGlobalOp>()) {
            mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();

            auto context = rewriter.getContext();

            // Find the memref.global op defining the frm_str
            auto globalOp = parentModule.lookupSymbol<mlir::memref::GlobalOp>(
                getGlobalOp.getNameAttr());

            rewriter.setInsertionPoint(globalOp);

            // Reconstruct the format string from the dense char array.
            auto initialvalueAttr =
                mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(
                    globalOp.getInitialValueAttr());
            std::string fmt;
            if (initialvalueAttr) {
              for (auto ap : initialvalueAttr.getValues<mlir::APInt>()) {
                char ch = static_cast<char>(ap.getZExtValue());
                if (ch == '\0')
                  break;
                fmt.push_back(ch);
              }
            }

            // Parse printf style specifiers, capturing each argument-consuming
            // entity in order: '*' (dynamic width/precision) and the final
            // conversion letter. This lets us map vararg index -> expected
            // spec kind (e.g. 's', 'p', etc.).
            std::vector<char> argKinds; // sequence of argument markers
            for (size_t i = 0; i < fmt.size(); ++i) {
              if (fmt[i] != '%')
                continue;
              size_t j = i + 1;
              if (j < fmt.size() && fmt[j] == '%') { // escaped %%
                i = j;
                continue;
              }
              // Flags
              while (j < fmt.size() && strchr("-+ #0", fmt[j]))
                j++;
              // Width
              if (j < fmt.size() && fmt[j] == '*') {
                argKinds.push_back('*');
                ++j;
              } else {
                while (j < fmt.size() &&
                       std::isdigit(static_cast<unsigned char>(fmt[j])))
                  j++;
              }
              // Precision
              if (j < fmt.size() && fmt[j] == '.') {
                ++j;
                if (j < fmt.size() && fmt[j] == '*') {
                  argKinds.push_back('*');
                  ++j;
                } else {
                  while (j < fmt.size() &&
                         std::isdigit(static_cast<unsigned char>(fmt[j])))
                    j++;
                }
              }
              // Length modifiers (simplified)
              auto startsWith = [&](const char *s) {
                size_t L = strlen(s);
                return j + L <= fmt.size() && strncmp(&fmt[j], s, L) == 0;
              };
              if (startsWith("hh"))
                j += 2;
              else if (startsWith("ll"))
                j += 2;
              else if (j < fmt.size() && strchr("hljztL", fmt[j]))
                j++;
              if (j < fmt.size()) {
                argKinds.push_back(fmt[j]);
                i = j;
              } else {
                break; // truncated spec
              }
            }

            // Insert an equivalent llvm.mlir.global (reuse earlier
            // initialvalueAttr)

            auto type = mlir::LLVM::LLVMArrayType::get(
                mlir::IntegerType::get(context, 8),
                initialvalueAttr.getNumElements());

            auto llvmglobalOp = rewriter.create<mlir::LLVM::GlobalOp>(
                globalOp->getLoc(), type, true, mlir::LLVM::Linkage::Internal,
                "printf_format_" + globalOp.getSymName().str(),
                initialvalueAttr, 0);

            rewriter.setInsertionPoint(getGlobalOp);

            // Insert llvmir dialect ops to retrive the !llvm.ptr of the global
            auto globalPtrOp = rewriter.create<mlir::LLVM::AddressOfOp>(
                getGlobalOp->getLoc(), llvmglobalOp);

            mlir::Value cst0 = rewriter.create<mlir::LLVM::ConstantOp>(
                getGlobalOp->getLoc(), rewriter.getI8Type(),
                rewriter.getIndexAttr(0));
            auto gepPtrOp = rewriter.create<mlir::LLVM::GEPOp>(
                getGlobalOp->getLoc(),
                mlir::LLVM::LLVMPointerType::get(context),
                llvmglobalOp.getType(), globalPtrOp,
                ArrayRef<mlir::Value>({cst0, cst0}));

            mlir::ValueRange operands = adaptor.getOperands();

            // Replace the old memref operand with the !llvm.ptr for the frm_str
            mlir::SmallVector<mlir::Value> newOperands;
            newOperands.push_back(gepPtrOp);
            auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(context);
            unsigned varArgIndex = 0; // index into argKinds for varargs
            for (auto it = operands.begin() + 1; it != operands.end();
                 ++it, ++varArgIndex) {
              mlir::Value val = *it;
              mlir::Type vty = val.getType();
              if (hasNullType) {
                // Drop all additional varargs to keep well-formed call.
                if (std::getenv("CLANGIR_DEBUG_PRINTF"))
                  llvm::errs()
                      << "[clangir] dropping vararg (null type scenario) index "
                      << varArgIndex << "\n";
                continue;
              }
              char kind = (varArgIndex < argKinds.size())
                              ? argKinds[varArgIndex]
                              : '\0';
              if (mlir::LLVM::isCompatibleType(vty)) {
                newOperands.push_back(val);
                continue;
              }
              if (auto mrTy = mlir::dyn_cast<mlir::MemRefType>(vty)) {
                bool treatAsString = (kind == 's');
                bool treatAsPointer = (kind == 'p');
                if (treatAsString &&
                    mrTy.getElementType() != rewriter.getI8Type()) {
                  // Mismatch: expected char element for %s.
                  treatAsString =
                      false; // fall back to %p semantics if possible
                  treatAsPointer = true;
                }
                if (treatAsString || treatAsPointer) {
                  mlir::Location loc = val.getLoc();
                  mlir::Value addrIdx =
                      rewriter
                          .create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
                              loc, val);
                  mlir::Value intVal = addrIdx;
                  if (!intVal.getType().isInteger(64)) {
                    if (intVal.getType().isIndex())
                      intVal = rewriter.create<mlir::arith::IndexCastOp>(
                          loc, rewriter.getI64Type(), intVal);
                    else if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(
                                 intVal.getType());
                             intTy && intTy.getWidth() < 64)
                      intVal = rewriter.create<mlir::arith::ExtUIOp>(
                          loc, rewriter.getI64Type(), intVal);
                  }
                  mlir::Value rawPtr = rewriter.create<mlir::LLVM::IntToPtrOp>(
                      loc, llvmPtrTy, intVal);
                  if (mrTy.getElementType() != rewriter.getI8Type())
                    rawPtr = rewriter.create<mlir::LLVM::BitcastOp>(
                        loc, llvmPtrTy, rawPtr);
                  newOperands.push_back(rawPtr);
                  if (std::getenv("CLANGIR_DEBUG_PRINTF"))
                    llvm::errs()
                        << "[clangir] salvaged memref arg for printf (%"
                        << (treatAsString ? 's' : 'p') << "): " << vty << "\n";
                  continue;
                }
              }
              if (std::getenv("CLANGIR_DEBUG_PRINTF"))
                llvm::errs()
                    << "[clangir] dropping unsupported printf arg at position "
                    << varArgIndex << " with type: " << vty << " (format kind '"
                    << kind << "')\n";
            }

            // Create the llvmir dialect function type for printf
            auto llvmI32Ty = mlir::IntegerType::get(context, 32);
            auto llvmFnType =
                mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                                  /*isVarArg=*/true);

            rewriter.setInsertionPoint(op);

            // Insert an llvm.call op with the updated operands to printf
            rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
                op, llvmFnType, op.getCalleeAttr(), newOperands);

            // Cleanup printf frm_str memref ops
            rewriter.eraseOp(reinterpretCastOp);
            rewriter.eraseOp(getGlobalOp);
            rewriter.eraseOp(globalOp);

            return mlir::LogicalResult::success();
          }
        }

        // Fallback path: format string not recognized as a local global literal
        // pattern. Degrade by treating first operand as the format pointer and
        // salvaging remaining operands generically.
        op.emitRemark() << "printf lowering: fallback generic path "
                           "(unrecognized format literal pattern)";
        mlir::ValueRange operands = adaptor.getOperands();
        if (operands.empty())
          return op.emitError() << "printf call with no operands";
        auto context = rewriter.getContext();
        auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(context);
        mlir::SmallVector<mlir::Value> newOperands;
        auto salvagePtr = [&](mlir::Value v) -> mlir::Value {
          mlir::Type ty = v.getType();
          if (mlir::LLVM::isCompatibleType(ty))
            return v; // already good (likely a pointer/integer)
          if (auto mrTy = mlir::dyn_cast<mlir::MemRefType>(ty)) {
            mlir::Location loc = v.getLoc();
            mlir::Value idx =
                rewriter.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
                    loc, v);
            mlir::Value intVal = idx;
            if (!intVal.getType().isInteger(64)) {
              if (intVal.getType().isIndex())
                intVal = rewriter.create<mlir::arith::IndexCastOp>(
                    loc, rewriter.getI64Type(), intVal);
              else if (auto intTy =
                           mlir::dyn_cast<mlir::IntegerType>(intVal.getType());
                       intTy && intTy.getWidth() < 64)
                intVal = rewriter.create<mlir::arith::ExtUIOp>(
                    loc, rewriter.getI64Type(), intVal);
            }
            mlir::Value raw =
                rewriter.create<mlir::LLVM::IntToPtrOp>(loc, llvmPtrTy, intVal);
            if (mrTy.getElementType() != rewriter.getI8Type())
              raw = rewriter.create<mlir::LLVM::BitcastOp>(loc, llvmPtrTy, raw);
            return raw;
          }
          // Unsupported exotic type: drop it.
          if (std::getenv("CLANGIR_DEBUG_PRINTF"))
            llvm::errs()
                << "[clangir] dropping unsupported printf vararg in fallback: "
                << ty << "\n";
          return {};
        };
        // First operand -> format pointer salvage.
        mlir::Value fmtPtr = salvagePtr(operands.front());
        if (!fmtPtr)
          return op.emitError() << "unable to salvage printf format operand";
        newOperands.push_back(fmtPtr);
        for (auto it = operands.begin() + 1; it != operands.end(); ++it) {
          if (mlir::Value v = salvagePtr(*it))
            newOperands.push_back(v);
        }
        auto llvmI32Ty = mlir::IntegerType::get(context, 32);
        auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(
            llvmI32Ty, llvmPtrTy, /*isVarArg=*/true);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
            op, llvmFnType, op.getCalleeAttr(), newOperands);
        return mlir::LogicalResult::success();
      }

      rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
          op, op.getCalleeAttr(), types, adaptor.getOperands());
      return mlir::LogicalResult::success();

    } else {
      // TODO: support lowering of indirect calls via func.call_indirect op
      return op.emitError() << "lowering of indirect calls not supported yet";
    }
  }
};

/// Given a type convertor and a data layout, convert the given type to a type
/// that is suitable for memory operations. For example, this can be used to
/// lower cir.bool accesses to i8.
static mlir::Type convertTypeForMemory(const mlir::TypeConverter &converter,
                                       mlir::Type type) {
  // TODO(cir): Handle other types similarly to clang's codegen
  // convertTypeForMemory
  if (isa<cir::BoolType>(type)) {
    // TODO: Use datalayout to get the size of bool
    return mlir::IntegerType::get(type.getContext(), 8);
  }

  if (isa<cir::PointerType>(type)) {
    // Model pointer memory slots as 64-bit integers to keep memref element
    // types legal (memref element cannot be an llvm.ptr) and avoid creating
    // memref<llvm.ptr> which is invalid. Later we translate loads/stores back
    // to llvm.ptr with inttoptr/ptrtoint.
    // TODO: derive width from target datalayout, currently fixed at 64.
    return mlir::IntegerType::get(type.getContext(), 64);
  }

  return converter.convertType(type);
}

static mlir::LLVM::AtomicOrdering
getLLVMMemOrder(std::optional<cir::MemOrder> memorder) {
  if (!memorder)
    return mlir::LLVM::AtomicOrdering::not_atomic;
  switch (*memorder) {
  case cir::MemOrder::Relaxed:
    return mlir::LLVM::AtomicOrdering::monotonic;
  case cir::MemOrder::Consume:
  case cir::MemOrder::Acquire:
    return mlir::LLVM::AtomicOrdering::acquire;
  case cir::MemOrder::Release:
    return mlir::LLVM::AtomicOrdering::release;
  case cir::MemOrder::AcquireRelease:
    return mlir::LLVM::AtomicOrdering::acq_rel;
  case cir::MemOrder::SequentiallyConsistent:
    return mlir::LLVM::AtomicOrdering::seq_cst;
  }
  llvm_unreachable("unknown memory order");
}

static llvm::DenseMap<mlir::Value, mlir::Value> PointerBackingMemrefs;

static void clearPointerBackingMemrefs() { PointerBackingMemrefs.clear(); }

static void registerPointerBackingMemref(mlir::Value pointer,
                                         mlir::Value memref) {
  if (!pointer || !memref)
    return;
  PointerBackingMemrefs[pointer] = memref;
}

static mlir::Value lookupPointerBackingMemref(mlir::Value pointer) {
  auto it = PointerBackingMemrefs.find(pointer);
  if (it == PointerBackingMemrefs.end())
    return {};
  return it->second;
}

struct PointerMemRefView {
  mlir::Value memref;
  mlir::Operation *bridgingCast = nullptr;
};

static std::optional<PointerMemRefView>
unwrapPointerLikeToMemRefImpl(mlir::Value value,
                              mlir::ConversionPatternRewriter &rewriter,
                              llvm::SmallPtrSetImpl<mlir::Value> &visited) {
  if (!value || visited.contains(value))
    return std::nullopt;
  visited.insert(value);

  if (auto memrefTy =
          mlir::dyn_cast_if_present<mlir::MemRefType>(value.getType()))
    return PointerMemRefView{value, nullptr};

  if (auto cached = lookupPointerBackingMemref(value))
    return PointerMemRefView{cached, nullptr};

  if (mlir::Value remapped = rewriter.getRemappedValue(value))
    if (auto cached = lookupPointerBackingMemref(remapped))
      return PointerMemRefView{cached, nullptr};

  if (auto castOp = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    for (mlir::Value input : castOp.getInputs()) {
      if (auto memrefTy =
              mlir::dyn_cast_if_present<mlir::MemRefType>(input.getType()))
        return PointerMemRefView{input, castOp};
      if (auto cached = lookupPointerBackingMemref(input))
        return PointerMemRefView{cached, castOp};
      if (auto nested = unwrapPointerLikeToMemRefImpl(input, rewriter, visited))
        return PointerMemRefView{nested->memref, castOp};
      if (mlir::Value remappedInput = rewriter.getRemappedValue(input)) {
        if (auto cached = lookupPointerBackingMemref(remappedInput))
          return PointerMemRefView{cached, castOp};
        if (auto nested = unwrapPointerLikeToMemRefImpl(remappedInput, rewriter,
                                                        visited))
          return PointerMemRefView{nested->memref, castOp};
      }
    }
  }

  return std::nullopt;
}

static std::optional<PointerMemRefView>
unwrapPointerLikeToMemRef(mlir::Value value,
                          mlir::ConversionPatternRewriter &rewriter) {
  llvm::SmallPtrSet<mlir::Value, 4> visited;
  return unwrapPointerLikeToMemRefImpl(value, rewriter, visited);
}

static mlir::LogicalResult replaceLoadWithSentinel(
    cir::LoadOp op, mlir::PatternRewriter &rewriter,
    const mlir::TypeConverter *converter, llvm::StringRef reason) {
  if (op->getBlock())
    op.emitRemark() << reason;
  else
    LLVM_DEBUG(llvm::dbgs() << "[cir][lowering] load sentinel reason: " << reason
                            << " (op detached)\n");

  if (!converter) {
    rewriter.eraseOp(op);
    return mlir::success();
  }

  mlir::Type mlirResTy = converter->convertType(op.getType());
  if (!mlirResTy) {
    rewriter.eraseOp(op);
    return mlir::success();
  }

  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(mlirResTy)) {
    auto allOnes = rewriter.getIntegerAttr(
        mlirResTy, llvm::APInt::getAllOnes(intTy.getWidth()));
    auto cst = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), mlirResTy,
                                                        allOnes);
    rewriter.replaceOp(op, cst.getResult());
    return mlir::success();
  }

  if (auto fTy = mlir::dyn_cast<mlir::FloatType>(mlirResTy)) {
    llvm::APFloat nan = llvm::APFloat::getQNaN(fTy.getFloatSemantics());
    auto attr = rewriter.getFloatAttr(fTy, nan);
    auto cst =
        rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), fTy, attr);
    rewriter.replaceOp(op, cst.getResult());
    return mlir::success();
  }

  if (auto ptrTy =
          mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(mlirResTy)) {
    auto undef = rewriter.create<mlir::LLVM::UndefOp>(op.getLoc(), ptrTy);
    rewriter.replaceOp(op, undef.getResult());
    return mlir::success();
  }

  if (auto vecTy = mlir::dyn_cast<mlir::VectorType>(mlirResTy)) {
    if (auto elemInt =
            mlir::dyn_cast<mlir::IntegerType>(vecTy.getElementType())) {
      llvm::SmallVector<llvm::APInt> vals(
          vecTy.getNumElements(), llvm::APInt::getAllOnes(elemInt.getWidth()));
      auto dense = mlir::DenseIntElementsAttr::get(vecTy, vals);
      auto cst = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), vecTy,
                                                          dense);
      rewriter.replaceOp(op, cst.getResult());
      return mlir::success();
    }
  }

  rewriter.eraseOp(op);
  return mlir::success();
}

// Helper: if 'maybeTy' is (or can be wrapped into) a MemRefType return it;
// otherwise emit a remark on 'tag' and forward the original base value by
// replacing the op. Returns std::nullopt if a forward happened.
static std::optional<mlir::MemRefType>
ensureMemRefOrForward(mlir::Location loc, mlir::Type maybeTy, mlir::Value base,
                      mlir::Operation *originalOp,
                      mlir::PatternRewriter &rewriter, llvm::StringRef tag) {
  auto dumpKind = [&](llvm::StringRef prefix, mlir::Type ty) {
    llvm::errs() << "[cir][lowering] " << tag << " " << prefix << "=";
    if (ty)
      ty.print(llvm::errs());
    else
      llvm::errs() << "<null>";
    llvm::errs() << '\n';
  };
  dumpKind("maybe-type", maybeTy);
  dumpKind("base-type", base.getType());

  if (auto mr = mlir::dyn_cast_if_present<mlir::MemRefType>(maybeTy))
    return mr;
  // Attempt to wrap a bare scalar (non-shaped, non-pointer) type in a rank-0
  // memref to preserve memref-based downstream assumptions. If pointer or
  // already a shaped/memref type, forward instead.
  if (maybeTy && !mlir::isa<mlir::MemRefType>(maybeTy) &&
      !mlir::isa<mlir::ShapedType>(maybeTy) &&
      !mlir::isa<mlir::LLVM::LLVMPointerType>(maybeTy)) {
    return mlir::MemRefType::get({}, maybeTy);
  }
  originalOp->emitRemark()
      << tag << " lowered as value forward (no memref representation)";
  rewriter.replaceOp(originalOp, base);
  return std::nullopt;
}

/// Emits the value from memory as expected by its users. Should be called when
/// the memory represetnation of a CIR type is not equal to its scalar
/// representation.
static mlir::Value emitFromMemory(mlir::ConversionPatternRewriter &rewriter,
                                  cir::LoadOp op, mlir::Value value) {

  // TODO(cir): Handle other types similarly to clang's codegen EmitFromMemory
  if (isa<cir::BoolType>(op.getType())) {
    // Create trunc of value from i8 to i1
    // TODO: Use datalayout to get the size of bool
    assert(value.getType().isInteger(8));
    return createIntCast(rewriter, value, rewriter.getI1Type());
  }

  if (isa<cir::PointerType>(op.getType())) {
    // Memory slot holds integer; rebuild pointer with inttoptr.
    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(value.getType())) {
      if (intTy.getWidth() != 64) {
        // Extend or truncate to 64 then cast (defensive; shouldn't happen now)
        auto i64Ty = rewriter.getI64Type();
        if (intTy.getWidth() < 64)
          value =
              rewriter.create<mlir::arith::ExtUIOp>(op.getLoc(), i64Ty, value);
        else if (intTy.getWidth() > 64)
          value =
              rewriter.create<mlir::arith::TruncIOp>(op.getLoc(), i64Ty, value);
      }
      auto ptrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
      return rewriter.create<mlir::LLVM::IntToPtrOp>(op.getLoc(), ptrTy, value);
    }
    return value;
  }

  return value;
}

/// Emits a value to memory with the expected scalar type. Should be called when
/// the memory represetnation of a CIR type is not equal to its scalar
/// representation.
static mlir::Value emitToMemory(mlir::ConversionPatternRewriter &rewriter,
                                cir::StoreOp op, mlir::Value value) {

  // TODO(cir): Handle other types similarly to clang's codegen EmitToMemory
  if (isa<cir::BoolType>(op.getValue().getType())) {
    // Create zext of value from i1 to i8
    // TODO: Use datalayout to get the size of bool
    return createIntCast(rewriter, value, rewriter.getI8Type());
  }

  if (isa<cir::PointerType>(op.getValue().getType())) {
    // Convert pointer to integer for memory representation.
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType())) {
      auto i64Ty = rewriter.getI64Type();
      return rewriter.create<mlir::LLVM::PtrToIntOp>(op.getLoc(), i64Ty, value);
    }
    return value;
  }

  return value;
}

class CIRAllocaOpLowering : public mlir::OpConversionPattern<cir::AllocaOp> {
public:
  using OpConversionPattern<cir::AllocaOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::AllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::Type mlirType =
        convertTypeForMemory(*getTypeConverter(), adaptor.getAllocaType());

    // FIXME: Some types can not be converted yet (e.g. struct)
    if (!mlirType)
      return mlir::LogicalResult::failure();

    // If the lowered memory type is an LLVM pointer (opaque), fall back to an
    // i64 slot alloca (consistent with pointer memory model elsewhere) then
    // treat loads/stores via bridging ops.
    mlir::MemRefType memrefTy;
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(mlirType)) {
      auto i64Ty = rewriter.getI64Type();
      memrefTy = mlir::MemRefType::get({}, i64Ty);
    } else {
      memrefTy = mlir::dyn_cast<mlir::MemRefType>(mlirType);
      if (!(memrefTy && mlir::isa<cir::ArrayType>(adaptor.getAllocaType())))
        memrefTy = mlir::MemRefType::get({}, mlirType);
    }

    auto loc = op.getLoc();
    auto memrefAlloca = rewriter.create<mlir::memref::AllocaOp>(
        loc, memrefTy, op.getAlignmentAttr());

    auto loweredResultTy = getTypeConverter()->convertType(op.getType());
    llvm::errs() << "[cir][lowering] alloca result convert " << op.getType()
                 << " -> ";
    if (loweredResultTy)
      loweredResultTy.print(llvm::errs());
    else
      llvm::errs() << "<null>";
    llvm::errs() << '\n';
    if (!loweredResultTy || loweredResultTy == memrefAlloca.getType()) {
      rewriter.replaceOp(op, memrefAlloca.getResult());
      return mlir::success();
    }

    auto bridge = rewriter.create<mlir::UnrealizedConversionCastOp>(
        loc, loweredResultTy, memrefAlloca.getResult());
    registerPointerBackingMemref(bridge.getResult(0), memrefAlloca.getResult());
    rewriter.replaceOp(op, bridge.getResults());
    return mlir::success();
  }
};

// Find base and indices from memref.reinterpret_cast
// and put it into eraseList.
static bool findBaseAndIndices(mlir::Value addr, mlir::Value &base,
                               SmallVector<mlir::Value> &indices,
                               SmallVector<mlir::Operation *> &eraseList,
                               mlir::ConversionPatternRewriter &rewriter) {
  while (auto *addrOp = getMemrefReinterpretCastOp(addr)) {
    if (addrOp->getNumOperands() > 1)
      indices.push_back(addrOp->getOperand(1));
    else
      break;
    addr = addrOp->getOperand(0);
    eraseList.push_back(addrOp);
  }
  base = addr;
  if (indices.size() == 0)
    return false;
  std::reverse(indices.begin(), indices.end());
  return true;
}

// If the memref.reinterpret_cast has multiple users (i.e the original
// cir.ptr_stride op has multiple users), only erase the operation after the
// last load or store has been generated.
static void eraseIfSafe(mlir::Value oldAddr, mlir::Value newAddr,
                        SmallVector<mlir::Operation *> &eraseList,
                        mlir::ConversionPatternRewriter &rewriter) {
  if (eraseList.empty())
    return; // Nothing to erase / no reinterpret_cast chain discovered.

  unsigned oldUsedNum =
      std::distance(oldAddr.getUses().begin(), oldAddr.getUses().end());
  unsigned newUsedNum = 0;
  // Count the uses of the newAddr (the result of the original base alloca) in
  // load/store ops using an forwarded offset from the current
  // memref.reinterpret_cast op
  mlir::Operation *anchor = eraseList.back();
  // If the anchor reinterpret_cast op was already removed (stale), bail out.
  if (!anchor || !anchor->getBlock() ||
      anchor->getName().getStringRef() != kMemrefReinterpretCastName) {
    eraseList.clear();
    return;
  }
  // Derive the anchor index operand directly. Earlier code attempted to call
  // getOffsets()/get<Value>() which is not part of the current
  // ReinterpretCastOp API here. For our purposes we only need to match the
  // single dynamic offset operand (pushed in findBaseAndIndices as operand(1)).
  mlir::Value anchorIndex;
  if (anchor->getNumOperands() > 1)
    anchorIndex = anchor->getOperand(1);
  for (auto *user : newAddr.getUsers()) {
    if (auto loadOpUser = mlir::dyn_cast_or_null<mlir::memref::LoadOp>(*user)) {
      if (!loadOpUser.getIndices().empty()) {
        auto strideVal = loadOpUser.getIndices()[0];
        if (anchorIndex && strideVal == anchorIndex)
          ++newUsedNum;
      }
    } else if (auto storeOpUser =
                   mlir::dyn_cast_or_null<mlir::memref::StoreOp>(*user)) {
      if (!storeOpUser.getIndices().empty()) {
        auto strideVal = storeOpUser.getIndices()[0];
        if (anchorIndex && strideVal == anchorIndex)
          ++newUsedNum;
      }
    }
  }
  // If all load/store ops using forwarded offsets from the current
  // memref.reinterpret_cast ops erase the memref.reinterpret_cast ops
  if (oldUsedNum == newUsedNum) {
    for (auto *op : eraseList)
      rewriter.eraseOp(op);
  }
}

class CIRLoadOpLowering : public mlir::OpConversionPattern<cir::LoadOp> {
public:
  using OpConversionPattern<cir::LoadOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value ptr = adaptor.getAddr();
    if (!ptr)
      return replaceLoadWithSentinel(
          op, rewriter, getTypeConverter(),
          "load lowering: missing converted address operand; producing undef "
          "surrogate");

    mlir::Type llvmTy =
        convertTypeForMemory(*getTypeConverter(), op.getType());
    if (!llvmTy)
      return replaceLoadWithSentinel(
          op, rewriter, getTypeConverter(),
          "load lowering: unable to derive memory element type; producing "
          "undef sentinel");

    unsigned alignment = 0;
    if (auto align = op.getAlignment())
      alignment = *align;

    auto ordering = getLLVMMemOrder(op.getMemOrder());

    auto load = rewriter.create<mlir::LLVM::LoadOp>(
        op.getLoc(), llvmTy, ptr, alignment, op.getIsVolatile(),
        op.getIsNontemporal(), /*invariant=*/false,
        /*invariantGroup=*/false, ordering);

    mlir::Value result = emitFromMemory(rewriter, op, load.getResult());
    rewriter.replaceOp(op, result);
    return mlir::LogicalResult::success();
  }
};

class CIRStoreOpLowering : public mlir::OpConversionPattern<cir::StoreOp> {
public:
  using OpConversionPattern<cir::StoreOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value ptr = adaptor.getAddr();
    if (!ptr) {
      op.emitRemark()
          << "store lowering: missing converted address operand; dropping "
             "store (no side effect)";
      rewriter.eraseOp(op);
      return mlir::success();
    }

    mlir::Value value = emitToMemory(rewriter, op, adaptor.getValue());

    unsigned alignment = 0;
    if (auto align = op.getAlignment())
      alignment = *align;

    auto ordering = getLLVMMemOrder(op.getMemOrder());

    rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(
        op, value, ptr, alignment, op.getIsVolatile(),
        op.getIsNontemporal(), /*invariantGroup=*/false, ordering);
    return mlir::LogicalResult::success();
  }
};

/// Converts CIR unary math ops (e.g., cir::SinOp) to their MLIR equivalents
/// (e.g., math::SinOp) using a generic template to avoid redundant boilerplate
/// matchAndRewrite definitions.

template <typename CIROp, typename MLIROp>
class CIRUnaryMathOpLowering : public mlir::OpConversionPattern<CIROp> {
public:
  using mlir::OpConversionPattern<CIROp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CIROp op,
                  typename mlir::OpConversionPattern<CIROp>::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<MLIROp>(op, adaptor.getSrc());
    return mlir::LogicalResult::success();
  }
};

using CIRASinOpLowering =
    CIRUnaryMathOpLowering<cir::ASinOp, mlir::math::AsinOp>;
using CIRSinOpLowering = CIRUnaryMathOpLowering<cir::SinOp, mlir::math::SinOp>;
using CIRExp2OpLowering =
    CIRUnaryMathOpLowering<cir::Exp2Op, mlir::math::Exp2Op>;
using CIRExpOpLowering = CIRUnaryMathOpLowering<cir::ExpOp, mlir::math::ExpOp>;
using CIRRoundOpLowering =
    CIRUnaryMathOpLowering<cir::RoundOp, mlir::math::RoundOp>;
using CIRLog2OpLowering =
    CIRUnaryMathOpLowering<cir::Log2Op, mlir::math::Log2Op>;
using CIRLogOpLowering = CIRUnaryMathOpLowering<cir::LogOp, mlir::math::LogOp>;
using CIRLog10OpLowering =
    CIRUnaryMathOpLowering<cir::Log10Op, mlir::math::Log10Op>;
using CIRCeilOpLowering =
    CIRUnaryMathOpLowering<cir::CeilOp, mlir::math::CeilOp>;
using CIRFloorOpLowering =
    CIRUnaryMathOpLowering<cir::FloorOp, mlir::math::FloorOp>;
using CIRAbsOpLowering = CIRUnaryMathOpLowering<cir::AbsOp, mlir::math::AbsIOp>;
using CIRFAbsOpLowering =
    CIRUnaryMathOpLowering<cir::FAbsOp, mlir::math::AbsFOp>;
using CIRSqrtOpLowering =
    CIRUnaryMathOpLowering<cir::SqrtOp, mlir::math::SqrtOp>;
using CIRCosOpLowering = CIRUnaryMathOpLowering<cir::CosOp, mlir::math::CosOp>;
using CIRATanOpLowering =
    CIRUnaryMathOpLowering<cir::ATanOp, mlir::math::AtanOp>;
using CIRACosOpLowering =
    CIRUnaryMathOpLowering<cir::ACosOp, mlir::math::AcosOp>;
using CIRTanOpLowering = CIRUnaryMathOpLowering<cir::TanOp, mlir::math::TanOp>;

class CIRShiftOpLowering : public mlir::OpConversionPattern<cir::ShiftOp> {
public:
  using mlir::OpConversionPattern<cir::ShiftOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(cir::ShiftOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto cirAmtTy = mlir::dyn_cast<cir::IntType>(op.getAmount().getType());
    auto cirValTy = mlir::dyn_cast<cir::IntType>(op.getValue().getType());
    auto mlirTy = getTypeConverter()->convertType(op.getType());
    mlir::Value amt = adaptor.getAmount();
    mlir::Value val = adaptor.getValue();

    assert(cirValTy && cirAmtTy && "non-integer shift is NYI");
    assert(cirValTy == op.getType() && "inconsistent operands' types NYI");

    // Ensure shift amount is the same type as the value. Some undefined
    // behavior might occur in the casts below as per [C99 6.5.7.3].
    amt = createIntCast(rewriter, amt, mlirTy, cirAmtTy.isSigned());

    // Lower to the proper arith shift operation.
    if (op.getIsShiftleft())
      rewriter.replaceOpWithNewOp<mlir::arith::ShLIOp>(op, mlirTy, val, amt);
    else {
      if (cirValTy.isUnsigned())
        rewriter.replaceOpWithNewOp<mlir::arith::ShRUIOp>(op, mlirTy, val, amt);
      else
        rewriter.replaceOpWithNewOp<mlir::arith::ShRSIOp>(op, mlirTy, val, amt);
    }

    return mlir::success();
  }
};

template <typename CIROp, typename MLIROp>
class CIRCountZerosBitOpLowering : public mlir::OpConversionPattern<CIROp> {
public:
  using mlir::OpConversionPattern<CIROp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CIROp op,
                  typename mlir::OpConversionPattern<CIROp>::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<MLIROp>(op, adaptor.getInput());
    return mlir::LogicalResult::success();
  }
};

using CIRBitClzOpLowering =
    CIRCountZerosBitOpLowering<cir::BitClzOp, mlir::math::CountLeadingZerosOp>;
using CIRBitCtzOpLowering =
    CIRCountZerosBitOpLowering<cir::BitCtzOp, mlir::math::CountTrailingZerosOp>;

class CIRBitClrsbOpLowering
    : public mlir::OpConversionPattern<cir::BitClrsbOp> {
public:
  using OpConversionPattern<cir::BitClrsbOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitClrsbOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto inputTy = adaptor.getInput().getType();
    auto zero = getConst(rewriter, op.getLoc(), inputTy, 0);
    auto isNeg = rewriter.create<mlir::arith::CmpIOp>(
        op.getLoc(),
        mlir::arith::CmpIPredicateAttr::get(rewriter.getContext(),
                                            mlir::arith::CmpIPredicate::slt),
        adaptor.getInput(), zero);

    auto negOne = getConst(rewriter, op.getLoc(), inputTy, -1);
    auto flipped = rewriter.create<mlir::arith::XOrIOp>(
        op.getLoc(), adaptor.getInput(), negOne);

    auto select = rewriter.create<mlir::arith::SelectOp>(
        op.getLoc(), isNeg, flipped, adaptor.getInput());

    auto clz =
        rewriter.create<mlir::math::CountLeadingZerosOp>(op->getLoc(), select);

    auto one = getConst(rewriter, op.getLoc(), inputTy, 1);
    auto res = rewriter.create<mlir::arith::SubIOp>(op.getLoc(), clz, one);
    rewriter.replaceOp(op, res);

    return mlir::LogicalResult::success();
  }
};

class CIRBitFfsOpLowering : public mlir::OpConversionPattern<cir::BitFfsOp> {
public:
  using OpConversionPattern<cir::BitFfsOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitFfsOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto inputTy = adaptor.getInput().getType();
    auto ctz = rewriter.create<mlir::math::CountTrailingZerosOp>(
        op.getLoc(), adaptor.getInput());

    auto one = getConst(rewriter, op.getLoc(), inputTy, 1);
    auto ctzAddOne =
        rewriter.create<mlir::arith::AddIOp>(op.getLoc(), ctz, one);

    auto zero = getConst(rewriter, op.getLoc(), inputTy, 0);
    auto isZero = rewriter.create<mlir::arith::CmpIOp>(
        op.getLoc(),
        mlir::arith::CmpIPredicateAttr::get(rewriter.getContext(),
                                            mlir::arith::CmpIPredicate::eq),
        adaptor.getInput(), zero);

    auto res = rewriter.create<mlir::arith::SelectOp>(op.getLoc(), isZero, zero,
                                                      ctzAddOne);
    rewriter.replaceOp(op, res);

    return mlir::LogicalResult::success();
  }
};

class CIRBitPopcountOpLowering
    : public mlir::OpConversionPattern<cir::BitPopcountOp> {
public:
  using mlir::OpConversionPattern<cir::BitPopcountOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitPopcountOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::math::CtPopOp>(op, adaptor.getInput());
    return mlir::LogicalResult::success();
  }
};

class CIRBitParityOpLowering
    : public mlir::OpConversionPattern<cir::BitParityOp> {
public:
  using OpConversionPattern<cir::BitParityOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitParityOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto count =
        rewriter.create<mlir::math::CtPopOp>(op.getLoc(), adaptor.getInput());
    auto countMod2 = rewriter.create<mlir::arith::AndIOp>(
        op.getLoc(), count,
        getConst(rewriter, op.getLoc(), count.getType(), 1));
    rewriter.replaceOp(op, countMod2);
    return mlir::LogicalResult::success();
  }
};

class CIRConstantOpLowering
    : public mlir::OpConversionPattern<cir::ConstantOp> {
public:
  using OpConversionPattern<cir::ConstantOp>::OpConversionPattern;

private:
  // This code is in a separate function rather than part of matchAndRewrite
  // because it is recursive.  There is currently only one level of recursion;
  // when lowing a vector attribute the attributes for the elements also need
  // to be lowered.
  mlir::TypedAttr
  lowerCirAttrToMlirAttr(mlir::Attribute cirAttr,
                         mlir::ConversionPatternRewriter &rewriter) const {
    assert(mlir::isa<mlir::TypedAttr>(cirAttr) &&
           "Can't lower a non-typed attribute");
    auto mlirType = getTypeConverter()->convertType(
        mlir::cast<mlir::TypedAttr>(cirAttr).getType());
    if (auto vecAttr = mlir::dyn_cast<cir::ConstVectorAttr>(cirAttr)) {
      assert(mlir::isa<mlir::VectorType>(mlirType) &&
             "MLIR type for CIR vector attribute is not mlir::VectorType");
      assert(mlir::isa<mlir::ShapedType>(mlirType) &&
             "mlir::VectorType is not a mlir::ShapedType ??");
      SmallVector<mlir::Attribute> mlirValues;
      for (auto elementAttr : vecAttr.getElts()) {
        mlirValues.push_back(
            this->lowerCirAttrToMlirAttr(elementAttr, rewriter));
      }
      return mlir::DenseElementsAttr::get(
          mlir::cast<mlir::ShapedType>(mlirType), mlirValues);
    } else if (auto boolAttr = mlir::dyn_cast<cir::BoolAttr>(cirAttr)) {
      return rewriter.getIntegerAttr(mlirType, boolAttr.getValue());
    } else if (auto floatAttr = mlir::dyn_cast<cir::FPAttr>(cirAttr)) {
      return rewriter.getFloatAttr(mlirType, floatAttr.getValue());
    } else if (auto intAttr = mlir::dyn_cast<cir::IntAttr>(cirAttr)) {
      return rewriter.getIntegerAttr(mlirType, intAttr.getValue());
    } else {
      // Support a few more common CIR constant attribute forms conservatively
      // and fall back to a zero initializer instead of crashing. This keeps
      // overall lowering progressing while we incrementally add precise
      // semantics for each attribute kind.
      if (auto zeroAttr = mlir::dyn_cast<cir::ZeroAttr>(cirAttr)) {
        // Use MLIR's generic zero attribute if possible.
        if (auto zero = rewriter.getZeroAttr(mlirType))
          return mlir::cast<mlir::TypedAttr>(zero);
        // Fallback: integer 0 bitcast style for unsupported zero forms.
        if (mlir::isa<mlir::IntegerType>(mlirType))
          return rewriter.getIntegerAttr(mlirType, 0);
        if (mlir::isa<mlir::FloatType>(mlirType))
          return rewriter.getFloatAttr(mlirType, 0.0);
      } else if (auto undefAttr = mlir::dyn_cast<cir::UndefAttr>(cirAttr)) {
        // Treat undef conservatively as zero.
        if (mlir::isa<mlir::IntegerType>(mlirType))
          return rewriter.getIntegerAttr(mlirType, 0);
        if (mlir::isa<mlir::FloatType>(mlirType))
          return rewriter.getFloatAttr(mlirType, 0.0);
      } else if (auto poisonAttr = mlir::dyn_cast<cir::PoisonAttr>(cirAttr)) {
        // Map poison to zero for now; a future improvement could thread a
        // distinct poison/undef dialect value.
        if (mlir::isa<mlir::IntegerType>(mlirType))
          return rewriter.getIntegerAttr(mlirType, 0);
        if (mlir::isa<mlir::FloatType>(mlirType))
          return rewriter.getFloatAttr(mlirType, 0.0);
      } else if (auto ptrAttr = mlir::dyn_cast<cir::ConstPtrAttr>(cirAttr)) {
        // Pointer constants currently appear as integer address payloads in
        // CIR. Attempt to materialize as an integer attribute matching the
        // lowered pointer bit-width, defaulting to zero when unavailable.
        if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(mlirType))
          return rewriter.getIntegerAttr(intTy, 0); // TODO: propagate value.
        // For opaque LLVM pointers, we can't use an integer attribute.
        // The caller should handle this specially by not using arith.constant.
        // Return empty TypedAttr to signal this needs special handling.
        if (mlir::isa<mlir::LLVM::LLVMPointerType>(mlirType))
          return mlir::TypedAttr();
      } else if (auto boolLike = mlir::dyn_cast<cir::BoolAttr>(cirAttr)) {
        return rewriter.getIntegerAttr(mlirType, boolLike.getValue());
      } else if (auto fpLike = mlir::dyn_cast<cir::FPAttr>(cirAttr)) {
        return rewriter.getFloatAttr(mlirType, fpLike.getValue());
      }
      // Generic final fallback: try to build a zero attribute; if that fails,
      // emit a remark and return an empty typed attr (caller will drop op).
      if (auto zero = rewriter.getZeroAttr(mlirType))
        return mlir::cast<mlir::TypedAttr>(zero);
      if (auto *ctx = mlirType.getContext()) {
        mlir::emitRemark(mlir::UnknownLoc::get(ctx))
            << "conservative fallback: unsupported CIR constant attribute kind";
      }
      return mlir::TypedAttr();
    }
  }

public:
  mlir::LogicalResult
  matchAndRewrite(cir::ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto convertedType = getTypeConverter()->convertType(op.getType());
    auto mlirAttr = this->lowerCirAttrToMlirAttr(op.getValue(), rewriter);

    // Special case: null pointer constant for LLVM opaque pointers
    if (!mlirAttr && mlir::isa<mlir::LLVM::LLVMPointerType>(convertedType) &&
        mlir::isa<cir::ConstPtrAttr>(op.getValue())) {
      // Create an llvm.mlir.zero for null pointer
      auto nullPtr = rewriter.create<mlir::LLVM::ZeroOp>(op.getLoc(), convertedType);
      rewriter.replaceOp(op, nullPtr.getResult());
      return mlir::LogicalResult::success();
    }

    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
        op, convertedType, mlirAttr);
    return mlir::LogicalResult::success();
  }
};

class CIRFuncOpLowering : public mlir::OpConversionPattern<cir::FuncOp> {
public:
  using OpConversionPattern<cir::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto fnType = op.getFunctionType();

    if (fnType.isVarArg()) {
      // TODO: once the func dialect supports variadic functions rewrite this
      // For now only insert special handling of printf via the llvmir dialect
      if (op.getSymName().equals_insensitive("printf")) {
        auto context = rewriter.getContext();
        // Create a llvmir dialect function declaration for printf, the
        // signature is: i32 (!llvm.ptr, ...)
        auto llvmI32Ty = mlir::IntegerType::get(context, 32);
        auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(context);
        auto llvmFnType =
            mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                              /*isVarArg=*/true);
        auto printfFunc = rewriter.create<mlir::LLVM::LLVMFuncOp>(
            op.getLoc(), "printf", llvmFnType);
        rewriter.replaceOp(op, printfFunc);
      } else {
        rewriter.eraseOp(op);
        return op.emitError() << "lowering of variadic functions (except "
                                 "printf) not supported yet";
      }
    } else {
      mlir::TypeConverter::SignatureConversion signatureConversion(
          fnType.getNumInputs());

      for (const auto &argType : enumerate(fnType.getInputs())) {
        auto convertedType = typeConverter->convertType(argType.value());
        if (!convertedType)
          return mlir::failure();
        signatureConversion.addInputs(argType.index(), convertedType);
      }

      SmallVector<mlir::NamedAttribute, 2> passThroughAttrs;

      if (auto symVisibilityAttr = op.getSymVisibilityAttr())
        passThroughAttrs.push_back(
            rewriter.getNamedAttr("sym_visibility", symVisibilityAttr));

      mlir::Type resultType =
          getTypeConverter()->convertType(fnType.getReturnType());
      auto fn = rewriter.create<mlir::func::FuncOp>(
          op.getLoc(), op.getName(),
          rewriter.getFunctionType(signatureConversion.getConvertedTypes(),
                                   resultType ? mlir::TypeRange(resultType)
                                              : mlir::TypeRange()),
          passThroughAttrs);

      // Convert types on the original region, then inline
      if (failed(rewriter.convertRegionTypes(&op.getBody(), *typeConverter,
                                             &signatureConversion)))
        return mlir::failure();
      rewriter.inlineRegionBefore(op.getBody(), fn.getBody(), fn.end());

      // Manually convert cir.return operations to func.return
      llvm::SmallVector<cir::ReturnOp> pendingReturns;
      fn.walk([&](cir::ReturnOp retOp) { pendingReturns.push_back(retOp); });
      llvm::ArrayRef<mlir::Type> expectedResults =
          fn.getFunctionType().getResults();

      for (cir::ReturnOp retOp : pendingReturns) {
        // Set insertion point FIRST to ensure all operations are created
        // inside the function body
        rewriter.setInsertionPoint(retOp);

        llvm::SmallVector<mlir::Value> retOperands;
        retOperands.reserve(retOp.getNumOperands());
        for (mlir::Value operand : retOp.getOperands()) {
          // Unwrap any redundant conversion casts
          if (auto castOp =
                  operand.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
            if (castOp->getNumOperands() == 1 &&
                castOp->getNumResults() == 1) {
              auto srcTy = castOp->getOperand(0).getType();
              auto dstTy = castOp->getResult(0).getType();
              if ((mlir::isa<cir::IntType>(dstTy) &&
                   mlir::isa<mlir::IntegerType>(srcTy)) ||
                  (mlir::isa<cir::SingleType, cir::DoubleType>(dstTy) &&
                   mlir::isa<mlir::FloatType>(srcTy))) {
                operand = castOp->getOperand(0);
                if (castOp->use_empty())
                  castOp->erase();
              }
            }
          }
          retOperands.push_back(operand);
        }

        // Create bridge casts if needed to match expected return types
        // These are created at the current insertion point (before retOp)
        if (expectedResults.size() == retOperands.size()) {
          for (auto [idx, value] : llvm::enumerate(retOperands)) {
            auto expectedTy = expectedResults[idx];
            if (value.getType() == expectedTy)
              continue;
            auto bridge = rewriter.create<mlir::UnrealizedConversionCastOp>(
                retOp.getLoc(), expectedTy, value);
            retOperands[idx] = bridge.getResult(0);
          }
        }

        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(retOp, retOperands);
      }

      rewriter.eraseOp(op);
    }
    return mlir::LogicalResult::success();
  }
};

class CIRUnaryOpLowering : public mlir::OpConversionPattern<cir::UnaryOp> {
public:
  using OpConversionPattern<cir::UnaryOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::UnaryOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getInput();
    auto type = getTypeConverter()->convertType(op.getType());

    switch (op.getKind()) {
    case cir::UnaryOpKind::Inc: {
      auto One = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), mlir::IntegerAttr::get(type, 1));
      rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, type, input, One);
      break;
    }
    case cir::UnaryOpKind::Dec: {
      auto One = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), mlir::IntegerAttr::get(type, 1));
      rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, type, input, One);
      break;
    }
    case cir::UnaryOpKind::Plus: {
      rewriter.replaceOp(op, op.getInput());
      break;
    }
    case cir::UnaryOpKind::Minus: {
      auto Zero = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), mlir::IntegerAttr::get(type, 0));
      rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, type, Zero, input);
      break;
    }
    case cir::UnaryOpKind::Not: {
      auto MinusOne = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), mlir::IntegerAttr::get(type, -1));
      rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(op, type, MinusOne,
                                                       input);
      break;
    }
    }

    return mlir::LogicalResult::success();
  }
};

class CIRBinOpLowering : public mlir::OpConversionPattern<cir::BinOp> {
public:
  using OpConversionPattern<cir::BinOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BinOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert((adaptor.getLhs().getType() == adaptor.getRhs().getType()) &&
           "inconsistent operands' types not supported yet");
    mlir::Type mlirType = getTypeConverter()->convertType(op.getType());
    assert((mlir::isa<mlir::IntegerType>(mlirType) ||
            mlir::isa<mlir::FloatType>(mlirType) ||
            mlir::isa<mlir::VectorType>(mlirType)) &&
           "operand type not supported yet");

    auto type = op.getLhs().getType();
    if (auto VecType = mlir::dyn_cast<cir::VectorType>(type)) {
      type = VecType.getElementType();
    }

    switch (op.getKind()) {
    case cir::BinOpKind::Add:
      if (mlir::isa<cir::IntType>(type))
        rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case cir::BinOpKind::Sub:
      if (mlir::isa<cir::IntType>(type))
        rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::SubFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case cir::BinOpKind::Mul:
      if (mlir::isa<cir::IntType>(type))
        rewriter.replaceOpWithNewOp<mlir::arith::MulIOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::MulFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case cir::BinOpKind::Div:
      if (auto ty = mlir::dyn_cast<cir::IntType>(type)) {
        if (ty.isUnsigned())
          rewriter.replaceOpWithNewOp<mlir::arith::DivUIOp>(
              op, mlirType, adaptor.getLhs(), adaptor.getRhs());
        else
          rewriter.replaceOpWithNewOp<mlir::arith::DivSIOp>(
              op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      } else
        rewriter.replaceOpWithNewOp<mlir::arith::DivFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case cir::BinOpKind::Rem:
      if (auto ty = mlir::dyn_cast<cir::IntType>(type)) {
        if (ty.isUnsigned())
          rewriter.replaceOpWithNewOp<mlir::arith::RemUIOp>(
              op, mlirType, adaptor.getLhs(), adaptor.getRhs());
        else
          rewriter.replaceOpWithNewOp<mlir::arith::RemSIOp>(
              op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      } else
        rewriter.replaceOpWithNewOp<mlir::arith::RemFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case cir::BinOpKind::And:
      rewriter.replaceOpWithNewOp<mlir::arith::AndIOp>(
          op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case cir::BinOpKind::Or:
      rewriter.replaceOpWithNewOp<mlir::arith::OrIOp>(
          op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case cir::BinOpKind::Xor:
      rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(
          op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case cir::BinOpKind::Max:
      llvm_unreachable("BinOpKind::Max lowering through MLIR NYI");
      break;
    }

    return mlir::LogicalResult::success();
  }
};

class CIRCmpOpLowering : public mlir::OpConversionPattern<cir::CmpOp> {
public:
  using OpConversionPattern<cir::CmpOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::CmpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = op.getLhs().getType();

    if (auto ty = mlir::dyn_cast<cir::IntType>(type)) {
      auto kind = convertCmpKindToCmpIPredicate(op.getKind(), ty.isSigned());
      rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
          op, kind, adaptor.getLhs(), adaptor.getRhs());
    } else if (auto ty = mlir::dyn_cast<cir::FPTypeInterface>(type)) {
      auto kind = convertCmpKindToCmpFPredicate(op.getKind());
      rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
          op, kind, adaptor.getLhs(), adaptor.getRhs());
    } else if (auto ty = mlir::dyn_cast<cir::PointerType>(type)) {
      op.emitRemark()
          << "pointer comparison lowered via address compare (conservative)";
      auto loc = op.getLoc();
      auto i64Ty = rewriter.getI64Type();
      auto toInt = [&](mlir::Value v) -> mlir::Value {
        if (mlir::isa<mlir::LLVM::LLVMPointerType>(v.getType()))
          return rewriter.create<mlir::LLVM::PtrToIntOp>(loc, i64Ty, v);
        if (v.getType().isIndex())
          return rewriter.create<mlir::arith::IndexCastOp>(loc, i64Ty, v);
        if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(v.getType());
            intTy && intTy.getWidth() < 64)
          return rewriter.create<mlir::arith::ExtUIOp>(loc, i64Ty, v);
        return v;
      };
      mlir::Value lhsAddr = toInt(adaptor.getLhs());
      mlir::Value rhsAddr = toInt(adaptor.getRhs());
      auto pred =
          convertCmpKindToCmpIPredicate(op.getKind(), /*isSigned=*/false);
      rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(op, pred, lhsAddr,
                                                       rhsAddr);
    } else {
      return op.emitError() << "unsupported type for CmpOp: " << type;
    }

    return mlir::LogicalResult::success();
  }
};

class CIRBrOpLowering : public mlir::OpRewritePattern<cir::BrOp> {
public:
  using OpRewritePattern<cir::BrOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BrOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, op.getDest());
    return mlir::LogicalResult::success();
  }
};

class CIRScopeOpLowering : public mlir::OpConversionPattern<cir::ScopeOp> {
  using mlir::OpConversionPattern<cir::ScopeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ScopeOp scopeOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Empty scope: just remove it.
    // TODO: Remove this logic once CIR uses MLIR infrastructure to remove
    // trivially dead operations
    if (scopeOp.isEmpty()) {
      rewriter.eraseOp(scopeOp);
      return mlir::success();
    }

    // Check if the scope is empty (no operations)
    auto &scopeRegion = scopeOp.getScopeRegion();
    if (scopeRegion.empty() ||
        (scopeRegion.front().empty() ||
         (scopeRegion.front().getOperations().size() == 1 &&
          isa<cir::YieldOp>(scopeRegion.front().front())))) {
      // Drop empty scopes
      rewriter.eraseOp(scopeOp);
      return mlir::LogicalResult::success();
    }

    // For scopes without results, use memref.alloca_scope
    if (scopeOp.getNumResults() == 0) {
      auto allocaScope = rewriter.create<mlir::memref::AllocaScopeOp>(
          scopeOp.getLoc(), mlir::TypeRange{});

      // Convert region types before inlining to handle cir.yield properly
      mlir::Region &scopeRegion = scopeOp.getScopeRegion();
      if (failed(rewriter.convertRegionTypes(&scopeRegion, *getTypeConverter())))
        return mlir::failure();

      rewriter.inlineRegionBefore(scopeOp.getScopeRegion(),
                                  allocaScope.getBodyRegion(),
                                  allocaScope.getBodyRegion().end());
      rewriter.eraseOp(scopeOp);
    } else {
      // For scopes with results, use scf.execute_region
      SmallVector<mlir::Type> types;
      if (mlir::failed(getTypeConverter()->convertTypes(
              scopeOp->getResultTypes(), types)))
        return mlir::failure();
      auto exec =
          rewriter.create<mlir::scf::ExecuteRegionOp>(scopeOp.getLoc(), types);

      // Convert region types before inlining to handle cir.yield properly
      mlir::Region &scopeRegion = scopeOp.getScopeRegion();
      if (failed(rewriter.convertRegionTypes(&scopeRegion, *getTypeConverter())))
        return mlir::failure();

      rewriter.inlineRegionBefore(scopeOp.getScopeRegion(), exec.getRegion(),
                                  exec.getRegion().end());
      rewriter.replaceOp(scopeOp, exec.getResults());
    }
    return mlir::LogicalResult::success();
  }
};

struct CIRBrCondOpLowering : public mlir::OpConversionPattern<cir::BrCondOp> {
  using mlir::OpConversionPattern<cir::BrCondOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BrCondOp brOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        brOp, adaptor.getCond(), brOp.getDestTrue(),
        adaptor.getDestOperandsTrue(), brOp.getDestFalse(),
        adaptor.getDestOperandsFalse());

    return mlir::success();
  }
};

class CIRTernaryOpLowering : public mlir::OpConversionPattern<cir::TernaryOp> {
public:
  using OpConversionPattern<cir::TernaryOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::TernaryOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    SmallVector<mlir::Type> resultTypes;
    if (mlir::failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)))
      return mlir::failure();

    auto ifOp = rewriter.create<mlir::scf::IfOp>(op.getLoc(), resultTypes,
                                                 adaptor.getCond(), true);
    auto *thenBlock = &ifOp.getThenRegion().front();
    auto *elseBlock = &ifOp.getElseRegion().front();
    rewriter.inlineBlockBefore(&op.getTrueRegion().front(), thenBlock,
                               thenBlock->end());
    rewriter.inlineBlockBefore(&op.getFalseRegion().front(), elseBlock,
                               elseBlock->end());

    rewriter.replaceOp(op, ifOp);
    return mlir::success();
  }
};

class CIRYieldOpLowering : public mlir::OpConversionPattern<cir::YieldOp> {
public:
  using OpConversionPattern<cir::YieldOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(cir::YieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *parentOp = op->getParentOp();
    return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(parentOp)
        .Case<mlir::scf::IfOp, mlir::scf::ForOp, mlir::scf::WhileOp, mlir::scf::ExecuteRegionOp>([&](auto) {
          rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(
              op, adaptor.getOperands());
          return mlir::success();
        })
        .Case<mlir::memref::AllocaScopeOp>([&](auto) {
          rewriter.replaceOpWithNewOp<mlir::memref::AllocaScopeReturnOp>(
              op, adaptor.getOperands());
          return mlir::success();
        })
        .Default([](auto) { return mlir::failure(); });
  }
};

class CIRConditionOpLowering
    : public mlir::OpConversionPattern<cir::ConditionOp> {
public:
  using OpConversionPattern<cir::ConditionOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(cir::ConditionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // cir.condition is only valid in scf.while before region
    // Convert to scf.condition with the boolean operand
    rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
        op, adaptor.getCondition(), adaptor.getOperands());
    return mlir::success();
  }
};

class CIRBreakOpLowering : public mlir::OpConversionPattern<cir::BreakOp> {
public:
  using OpConversionPattern<cir::BreakOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(cir::BreakOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // cir.break should have been converted by SCF preparation pass.
    // If we see it here, check parent and use appropriate yield
    auto *parentOp = op->getParentOp();
    if (mlir::isa<mlir::scf::IfOp, mlir::scf::ForOp, mlir::scf::WhileOp, mlir::scf::ExecuteRegionOp>(parentOp)) {
      rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, mlir::ValueRange{});
    } else if (mlir::isa<mlir::memref::AllocaScopeOp>(parentOp)) {
      rewriter.replaceOpWithNewOp<mlir::memref::AllocaScopeReturnOp>(op, mlir::ValueRange{});
    } else {
      rewriter.replaceOpWithNewOp<cir::YieldOp>(op, mlir::ValueRange{});
    }
    return mlir::success();
  }
};

class CIRContinueOpLowering : public mlir::OpConversionPattern<cir::ContinueOp> {
public:
  using OpConversionPattern<cir::ContinueOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(cir::ContinueOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // cir.continue should have been converted by SCF preparation pass.
    // If we see it here, check parent and use appropriate yield
    auto *parentOp = op->getParentOp();
    if (mlir::isa<mlir::scf::IfOp, mlir::scf::ForOp, mlir::scf::WhileOp, mlir::scf::ExecuteRegionOp>(parentOp)) {
      rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, mlir::ValueRange{});
    } else if (mlir::isa<mlir::memref::AllocaScopeOp>(parentOp)) {
      rewriter.replaceOpWithNewOp<mlir::memref::AllocaScopeReturnOp>(op, mlir::ValueRange{});
    } else {
      rewriter.replaceOpWithNewOp<cir::YieldOp>(op, mlir::ValueRange{});
    }
    return mlir::success();
  }
};

class CIRIfOpLowering : public mlir::OpConversionPattern<cir::IfOp> {
public:
  using mlir::OpConversionPattern<cir::IfOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::IfOp ifop, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto newIfOp = rewriter.create<mlir::scf::IfOp>(
        ifop->getLoc(), ifop->getResultTypes(), adaptor.getCondition());
    auto *thenBlock = rewriter.createBlock(&newIfOp.getThenRegion());
    rewriter.inlineBlockBefore(&ifop.getThenRegion().front(), thenBlock,
                               thenBlock->end());
    if (!ifop.getElseRegion().empty()) {
      auto *elseBlock = rewriter.createBlock(&newIfOp.getElseRegion());
      rewriter.inlineBlockBefore(&ifop.getElseRegion().front(), elseBlock,
                                 elseBlock->end());
    }
    rewriter.replaceOp(ifop, newIfOp);
    return mlir::success();
  }
};

class CIRGlobalOpLowering : public mlir::OpConversionPattern<cir::GlobalOp> {
public:
  using OpConversionPattern<cir::GlobalOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::GlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp)
      return mlir::failure();

    mlir::OpBuilder b(moduleOp.getContext());

    const auto cirSymType = op.getSymType();
    // For globals, first try regular type conversion to preserve pointer types.
    // convertTypeForMemory is inappropriate here because it converts pointers to i64
    // for memref compatibility, but globals can directly be llvm.ptr.
    auto convertedType = getTypeConverter()->convertType(cirSymType);
    if (!convertedType)
      return mlir::failure();
    // For constant arrays (like string literals), create llvm.mlir.global instead of memref.global
    // This allows get_global operations that expect pointers to work correctly
    bool isConstantArray = op.getConstant() && mlir::isa<mlir::MemRefType>(convertedType) &&
                          op.getInitialValue() &&
                          mlir::isa<cir::ConstArrayAttr>(*op.getInitialValue());

    if (isConstantArray) {
      auto memrefType = mlir::cast<mlir::MemRefType>(convertedType);
      auto elemType = memrefType.getElementType();
      auto shape = memrefType.getShape();

      // Create an LLVM array type
      mlir::Type llvmArrayType = elemType;
      for (auto dim : llvm::reverse(shape)) {
        llvmArrayType = mlir::LLVM::LLVMArrayType::get(llvmArrayType, dim);
      }

      // Get the initializer
      auto constArr = mlir::cast<cir::ConstArrayAttr>(*op.getInitialValue());
      auto init = lowerConstArrayAttr(constArr, getTypeConverter());

      auto linkage = mlir::LLVM::Linkage::Internal;  // String literals are typically internal
      auto nameAttr = b.getStringAttr(op.getSymName());

      rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
          op, llvmArrayType, /*isConstant=*/true, linkage,
          nameAttr.getValue(), init.value_or(mlir::Attribute()),
          /*alignment=*/op.getAlignment().value_or(1));
      return mlir::success();
    }

    // If the lowered element type is already an LLVM pointer (opaque or typed),
    // prefer emitting an llvm.global directly instead of wrapping in a memref.
    if (auto llvmPtrTy =
            mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(convertedType)) {
      // Build initializer if present (only handle simple scalar
      // zero/int/float/bool cases now).
      mlir::Attribute initAttr; // (unused for now; pointer scalar init NYI)
      if (op.getInitialValue()) {
        auto iv = *op.getInitialValue();
        if (auto intAttr = mlir::dyn_cast<cir::IntAttr>(iv)) {
          // auto i64Ty = mlir::IntegerType::get(b.getContext(), 64);
          auto val = intAttr.getValue();
          // Truncate or extend through APInt then cast constant pointer via
          // inttoptr at use sites; here we just store integer as data by
          // emitting a zero-initialized pointer (no direct ptr const model
          // yet).
          (void)val; // placeholder; pointer constants not yet materialized.
        } else if (mlir::isa<cir::ZeroAttr>(iv)) {
          // Nothing needed; default zeroinitializer is fine.
        } else if (auto boolAttr = mlir::dyn_cast<cir::BoolAttr>(iv)) {
          (void)boolAttr; // ignore, keep default null pointer.
        } else {
          op.emitRemark()
              << "pointer global initializer kind unsupported; using null";
        }
      }
      auto linkage = mlir::LLVM::Linkage::External;
      if (op.isPrivate())
        linkage = mlir::LLVM::Linkage::Internal;
      auto nameAttr = b.getStringAttr(op.getSymName());
      rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
          op, llvmPtrTy, /*isConstant=*/op.getConstant(), linkage,
          nameAttr.getValue(), /*initializer=*/mlir::Attribute(),
          /*alignment=*/0); // alignment currently ignored in direct path
      return mlir::success();
    }

    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(convertedType);
    if (!memrefType)
      memrefType = mlir::MemRefType::get({}, convertedType);
    // Add an optional alignment to the global memref.
    mlir::IntegerAttr memrefAlignment =
        op.getAlignment()
            ? mlir::IntegerAttr::get(b.getI64Type(), op.getAlignment().value())
            : mlir::IntegerAttr();
    // Add an optional initial value to the global memref.
    mlir::Attribute initialValue = mlir::Attribute();
    std::optional<mlir::Attribute> init = op.getInitialValue();
    if (init.has_value()) {
      if (auto constArr = mlir::dyn_cast<cir::ConstArrayAttr>(init.value())) {
        init = lowerConstArrayAttr(constArr, getTypeConverter());
        if (init.has_value()) {
          initialValue = init.value();
        } else {
          op.emitRemark()
              << "global lowering: unsupported constant array initializer; "
                 "emitting zero-initialized fallback";
          // Best-effort zero fallback (scalar) if element type is integral/FP.
          if (auto elemTy = memrefType.getElementType();
              mlir::isa<mlir::IntegerType>(elemTy)) {
            auto rtt = mlir::RankedTensorType::get({}, elemTy);
            initialValue = mlir::DenseIntElementsAttr::get(rtt, 0);
          } else if (auto fTy = mlir::dyn_cast<mlir::FloatType>(elemTy)) {
            auto rtt = mlir::RankedTensorType::get({}, fTy);
            initialValue = mlir::DenseFPElementsAttr::get(
                rtt, mlir::FloatAttr::get(fTy, 0.0).getValue());
          }
        }
      } else if (auto zeroAttr = mlir::dyn_cast<cir::ZeroAttr>(init.value())) {
        (void)zeroAttr; // unused variable silence
        auto shape = memrefType.getShape();
        auto elementType = memrefType.getElementType();
        auto buildZeroTensor = [&](mlir::Type elemTy, mlir::Type tensorElemTy) {
          if (!shape.empty()) {
            auto rtt = mlir::RankedTensorType::get(shape, tensorElemTy);
            if (mlir::isa<mlir::IntegerType>(tensorElemTy)) {
              initialValue = mlir::DenseIntElementsAttr::get(rtt, 0);
            } else if (auto fTy =
                           mlir::dyn_cast<mlir::FloatType>(tensorElemTy)) {
              initialValue = mlir::DenseFPElementsAttr::get(
                  rtt, mlir::FloatAttr::get(fTy, 0.0).getValue());
            } else {
              op.emitRemark() << "global lowering: unsupported element type in "
                                 "zero initializer; leaving uninitialized";
            }
          } else {
            auto rtt = mlir::RankedTensorType::get({}, tensorElemTy);
            if (mlir::isa<mlir::IntegerType>(tensorElemTy)) {
              initialValue = mlir::DenseIntElementsAttr::get(rtt, 0);
            } else if (auto fTy =
                           mlir::dyn_cast<mlir::FloatType>(tensorElemTy)) {
              initialValue = mlir::DenseFPElementsAttr::get(
                  rtt, mlir::FloatAttr::get(tensorElemTy, 0.0).getValue());
            } else {
              op.emitRemark() << "global lowering: unsupported scalar type in "
                                 "zero initializer; leaving uninitialized";
            }
          }
        };
        buildZeroTensor(elementType, elementType);
      } else if (auto intAttr = mlir::dyn_cast<cir::IntAttr>(init.value())) {
        auto rtt = mlir::RankedTensorType::get({}, convertedType);
        initialValue = mlir::DenseIntElementsAttr::get(rtt, intAttr.getValue());
      } else if (auto fltAttr = mlir::dyn_cast<cir::FPAttr>(init.value())) {
        auto rtt = mlir::RankedTensorType::get({}, convertedType);
        initialValue = mlir::DenseFPElementsAttr::get(rtt, fltAttr.getValue());
      } else if (auto boolAttr = mlir::dyn_cast<cir::BoolAttr>(init.value())) {
        auto rtt = mlir::RankedTensorType::get({}, convertedType);
        initialValue =
            mlir::DenseIntElementsAttr::get(rtt, (char)boolAttr.getValue());
      } else {
        op.emitRemark() << "global lowering: unsupported initializer kind; "
                           "leaving uninitialized";
      }
    }

    // Add symbol visibility
    std::string sym_visibility = op.isPrivate() ? "private" : "public";

    rewriter.replaceOpWithNewOp<mlir::memref::GlobalOp>(
        op, b.getStringAttr(op.getSymName()),
        /*sym_visibility=*/b.getStringAttr(sym_visibility),
        /*type=*/memrefType, initialValue,
        /*constant=*/op.getConstant(),
        /*alignment=*/memrefAlignment);

    return mlir::success();
  }
};

class CIRGetGlobalOpLowering
    : public mlir::OpConversionPattern<cir::GetGlobalOp> {
public:
  using OpConversionPattern<cir::GetGlobalOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::GetGlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // FIXME(cir): Premature DCE to avoid lowering stuff we're not using.
    // CIRGen should mitigate this and not emit the get_global.
    if (op->getUses().empty()) {
      rewriter.eraseOp(op);
      return mlir::success();
    }
    auto type = getTypeConverter()->convertType(op.getType());
    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();

    // Pointer-aware path: if the symbol refers to an llvm.global (created by
    // pointer global refinement), emit an llvm.address_of producing the
    // pointer directly instead of a memref.get_global.
    if (auto llvmGlob =
            module.lookupSymbol<mlir::LLVM::GlobalOp>(op.getName())) {
      // llvm.address_of returns a pointer to the global
      // For LLVM opaque pointers, all pointers have the same type
      auto ctx = rewriter.getContext();
      auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);
      auto addrOp = rewriter.create<mlir::LLVM::AddressOfOp>(
          op.getLoc(), ptrType, llvmGlob.getSymName());
      mlir::Value addrVal = addrOp.getResult();
      // The result is already the correct pointer type for opaque pointers
      rewriter.replaceOp(op, addrVal);
      return mlir::success();
    }

    auto symbol = op.getName();

    // If the converted type is an LLVM pointer but we haven't found an llvm.global above,
    // this is an error - we can't create memref.get_global for pointer types
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(type)) {
      return op.emitError() << "cannot lower get_global of pointer type without corresponding llvm.global";
    }

    rewriter.replaceOpWithNewOp<mlir::memref::GetGlobalOp>(op, type, symbol);
    return mlir::success();
  }
};

class CIRVectorCreateLowering
    : public mlir::OpConversionPattern<cir::VecCreateOp> {
public:
  using OpConversionPattern<cir::VecCreateOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecCreateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto vecTy = mlir::dyn_cast<cir::VectorType>(op.getType());
    assert(vecTy && "result type of cir.vec.create op is not VectorType");
    auto elementTy = typeConverter->convertType(vecTy.getElementType());
    auto loc = op.getLoc();
    auto zeroElement = rewriter.getZeroAttr(elementTy);
    mlir::Value result = rewriter.create<mlir::arith::ConstantOp>(
        loc,
        mlir::DenseElementsAttr::get(
            mlir::VectorType::get(vecTy.getSize(), elementTy), zeroElement));
    assert(vecTy.getSize() == op.getElements().size() &&
           "cir.vec.create op count doesn't match vector type elements count");
    for (uint64_t i = 0; i < vecTy.getSize(); ++i) {
      mlir::Value indexValue =
          getConst(rewriter, loc, rewriter.getI64Type(), i);
      result = rewriter.create<mlir::vector::InsertElementOp>(
          loc, adaptor.getElements()[i], result, indexValue);
    }
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class CIRVectorInsertLowering
    : public mlir::OpConversionPattern<cir::VecInsertOp> {
public:
  using OpConversionPattern<cir::VecInsertOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecInsertOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::vector::InsertElementOp>(
        op, adaptor.getValue(), adaptor.getVec(), adaptor.getIndex());
    return mlir::success();
  }
};

class CIRVectorExtractLowering
    : public mlir::OpConversionPattern<cir::VecExtractOp> {
public:
  using OpConversionPattern<cir::VecExtractOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecExtractOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::vector::ExtractElementOp>(
        op, adaptor.getVec(), adaptor.getIndex());
    return mlir::success();
  }
};

class CIRVectorCmpOpLowering : public mlir::OpConversionPattern<cir::VecCmpOp> {
public:
  using OpConversionPattern<cir::VecCmpOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecCmpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert(mlir::isa<cir::VectorType>(op.getType()) &&
           mlir::isa<cir::VectorType>(op.getLhs().getType()) &&
           mlir::isa<cir::VectorType>(op.getRhs().getType()) &&
           "Vector compare with non-vector type");
    auto elementType =
        mlir::cast<cir::VectorType>(op.getLhs().getType()).getElementType();
    mlir::Value bitResult;
    if (auto intType = mlir::dyn_cast<cir::IntType>(elementType)) {
      bitResult = rewriter.create<mlir::arith::CmpIOp>(
          op.getLoc(),
          convertCmpKindToCmpIPredicate(op.getKind(), intType.isSigned()),
          adaptor.getLhs(), adaptor.getRhs());
    } else if (mlir::isa<cir::FPTypeInterface>(elementType)) {
      bitResult = rewriter.create<mlir::arith::CmpFOp>(
          op.getLoc(), convertCmpKindToCmpFPredicate(op.getKind()),
          adaptor.getLhs(), adaptor.getRhs());
    } else {
      return op.emitError() << "unsupported type for VecCmpOp: " << elementType;
    }
    rewriter.replaceOpWithNewOp<mlir::arith::ExtSIOp>(
        op, typeConverter->convertType(op.getType()), bitResult);
    return mlir::success();
  }
};

class CIRCastOpLowering : public mlir::OpConversionPattern<cir::CastOp> {
public:
  using OpConversionPattern<cir::CastOp>::OpConversionPattern;

  inline mlir::Type convertTy(mlir::Type ty) const {
    return getTypeConverter()->convertType(ty);
  }

  mlir::LogicalResult
  matchAndRewrite(cir::CastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (isa<cir::VectorType>(op.getSrc().getType()))
      llvm_unreachable("CastOp lowering for vector type is not supported yet");
    auto src = adaptor.getSrc();
    auto dstType = op.getType();
    using CIR = cir::CastKind;
    switch (op.getKind()) {
    case CIR::array_to_ptrdecay: {
      auto converted = convertTy(dstType);
      if (auto mr = mlir::dyn_cast_or_null<mlir::MemRefType>(converted)) {
        rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
            op, mr, src, 0, ArrayRef<int64_t>{}, ArrayRef<int64_t>{},
            ArrayRef<mlir::NamedAttribute>{});
      } else {
        // Pointer decay to a raw pointer (llvm.ptr) no longer needs an
        // intermediate memref wrapper; just forward the operand (bitcast
        // semantics are already captured earlier in lowering pipeline).
        op.emitRemark()
            << "array_to_ptrdecay lowered as value forward (no memref)";
        rewriter.replaceOp(op, src);
      }
      return mlir::success();
    }
    case CIR::int_to_bool: {
      auto zero = rewriter.create<cir::ConstantOp>(
          src.getLoc(), op.getSrc().getType(),
          cir::IntAttr::get(op.getSrc().getType(), 0));
      rewriter.replaceOpWithNewOp<cir::CmpOp>(
          op, cir::BoolType::get(getContext()), cir::CmpOpKind::ne, op.getSrc(),
          zero);
      return mlir::success();
    }
    case CIR::integral: {
      auto newDstType = convertTy(dstType);
      auto srcType = op.getSrc().getType();
      cir::IntType srcIntType = mlir::cast<cir::IntType>(srcType);
      auto newOp =
          createIntCast(rewriter, src, newDstType, srcIntType.isSigned());
      rewriter.replaceOp(op, newOp);
      return mlir::success();
    }
    case CIR::floating: {
      auto newDstType = convertTy(dstType);
      auto srcTy = op.getSrc().getType();
      auto dstTy = op.getType();

      if (!mlir::isa<cir::FPTypeInterface>(dstTy) ||
          !mlir::isa<cir::FPTypeInterface>(srcTy))
        return op.emitError() << "NYI cast from " << srcTy << " to " << dstTy;

      auto getFloatWidth = [](mlir::Type ty) -> unsigned {
        return mlir::cast<cir::FPTypeInterface>(ty).getWidth();
      };

      if (getFloatWidth(srcTy) > getFloatWidth(dstTy))
        rewriter.replaceOpWithNewOp<mlir::arith::TruncFOp>(op, newDstType, src);
      else
        rewriter.replaceOpWithNewOp<mlir::arith::ExtFOp>(op, newDstType, src);
      return mlir::success();
    }
    case CIR::float_to_bool: {
      auto kind = mlir::arith::CmpFPredicate::UNE;

      // Check if float is not equal to zero.
      auto zeroFloat = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), src.getType(), mlir::FloatAttr::get(src.getType(), 0.0));

      rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(op, kind, src,
                                                       zeroFloat);
      return mlir::success();
    }
    case CIR::bool_to_int: {
      auto dstTy = mlir::cast<cir::IntType>(op.getType());
      auto newDstType = mlir::cast<mlir::IntegerType>(convertTy(dstTy));
      auto newOp = createIntCast(rewriter, src, newDstType);
      rewriter.replaceOp(op, newOp);
      return mlir::success();
    }
    case CIR::bool_to_float: {
      auto dstTy = op.getType();
      auto newDstType = convertTy(dstTy);
      rewriter.replaceOpWithNewOp<mlir::arith::UIToFPOp>(op, newDstType, src);
      return mlir::success();
    }
    case CIR::int_to_float: {
      auto dstTy = op.getType();
      auto newDstType = convertTy(dstTy);
      if (mlir::cast<cir::IntType>(op.getSrc().getType()).isSigned())
        rewriter.replaceOpWithNewOp<mlir::arith::SIToFPOp>(op, newDstType, src);
      else
        rewriter.replaceOpWithNewOp<mlir::arith::UIToFPOp>(op, newDstType, src);
      return mlir::success();
    }
    case CIR::float_to_int: {
      auto dstTy = op.getType();
      auto newDstType = convertTy(dstTy);
      if (mlir::cast<cir::IntType>(op.getType()).isSigned())
        rewriter.replaceOpWithNewOp<mlir::arith::FPToSIOp>(op, newDstType, src);
      else
        rewriter.replaceOpWithNewOp<mlir::arith::FPToUIOp>(op, newDstType, src);
      return mlir::success();
    }
    case CIR::ptr_to_int: {
      // Pointer to integer conversion (e.g., for pointer arithmetic)
      auto loc = op.getLoc();
      auto srcType = src.getType();
      auto newDstType = convertTy(dstType);

      if (mlir::isa<mlir::LLVM::LLVMPointerType>(srcType)) {
        // LLVM pointer: use llvm.ptrtoint
        rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(op, newDstType, src);
      } else if (mlir::isa<mlir::IntegerType>(srcType)) {
        // Already an integer (pointer represented as intptr_t)
        // Just extend/truncate to target size if needed
        if (srcType == newDstType) {
          rewriter.replaceOp(op, src);
        } else {
          auto srcWidth = mlir::cast<mlir::IntegerType>(srcType).getWidth();
          auto dstWidth = mlir::cast<mlir::IntegerType>(newDstType).getWidth();
          if (srcWidth < dstWidth) {
            // Extend
            rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(op, newDstType, src);
          } else {
            // Truncate
            rewriter.replaceOpWithNewOp<mlir::arith::TruncIOp>(op, newDstType, src);
          }
        }
      } else {
        return op.emitError() << "ptr_to_int cast from unsupported type: " << srcType;
      }
      return mlir::success();
    }
    case CIR::ptr_to_bool: {
      // Pointer to boolean conversion: compare pointer against null
      auto loc = op.getLoc();
      auto srcType = src.getType();
      mlir::Value cmpResult;

      if (mlir::isa<mlir::LLVM::LLVMPointerType>(srcType)) {
        // LLVM pointer: compare against null pointer
        auto nullPtr = rewriter.create<mlir::LLVM::ZeroOp>(loc, srcType);
        cmpResult = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::ne, src, nullPtr);
      } else if (mlir::isa<mlir::IntegerType>(srcType)) {
        // Integer (pointer represented as intptr_t): compare against zero
        auto zero = rewriter.create<mlir::arith::ConstantOp>(
            loc, srcType, rewriter.getIntegerAttr(srcType, 0));
        cmpResult = rewriter.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::ne, src, zero);
      } else {
        return op.emitError() << "ptr_to_bool cast from unsupported type: " << srcType;
      }

      // The result is i1, convert to target boolean type if needed
      auto newDstType = convertTy(dstType);
      if (newDstType == cmpResult.getType()) {
        rewriter.replaceOp(op, cmpResult);
      } else {
        // Extend i1 to target integer type
        rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(op, newDstType, cmpResult);
      }
      return mlir::success();
    }
    case CIR::bitcast: {
      // Generic conservative lowering: if source and destination types lower
      // to the same MLIR type just forward the value. If both are memrefs but
      // with differing element types/ranks, attempt a memref.cast which is a
      // no-op if layout-compatible. If incompatible, keep the original value
      // (best-effort) to avoid aborting the pipeline.
      auto newDstType = convertTy(dstType);
      auto newSrcType = src.getType();
      if (newDstType == newSrcType) {
        rewriter.replaceOp(op, src);
        return mlir::success();
      }
      if (mlir::isa_and_nonnull<mlir::MemRefType>(newDstType) &&
          mlir::isa<mlir::MemRefType>(newSrcType)) {
        // memref.cast enforces layout compatibility; if it fails verification
        // downstream we still avoided leaving an illegal CIR op behind.
        rewriter.replaceOpWithNewOp<mlir::memref::CastOp>(op, newDstType, src);
        return mlir::success();
      }
      // Fallback: emit remark and forward value unchanged.
      op.emitRemark() << "conservative bitcast fallback from " << newSrcType
                      << " to " << newDstType;
      rewriter.replaceOp(op, src);
      return mlir::success();
    }
    case CIR::int_to_ptr: {
      auto loc = op.getLoc();
      auto newDstType = convertTy(dstType);

      if (mlir::isa<mlir::IntegerType>(src.getType())) {
        // Integer to pointer: use llvm.inttoptr
        rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(op, newDstType, src);
      } else if (mlir::isa<mlir::LLVM::LLVMPointerType>(src.getType())) {
        // Already a pointer: just forward it (bitcast-like)
        rewriter.replaceOp(op, src);
      } else {
        return mlir::failure();
      }
      return mlir::success();
    }
    default:
      break;
    }
    return mlir::failure();
  }
};

class CIRSelectOpLowering : public mlir::OpConversionPattern<cir::SelectOp> {
public:
  using mlir::OpConversionPattern<cir::SelectOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::SelectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto condition = adaptor.getCondition();
    auto trueValue = adaptor.getTrueValue();
    auto falseValue = adaptor.getFalseValue();

    // Convert result type
    auto resultType = this->getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return mlir::failure();

    // Use arith.select for the ternary operator
    rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(
        op, resultType, condition, trueValue, falseValue);
    return mlir::success();
  }
};

class CIRCopyOpLowering : public mlir::OpConversionPattern<cir::CopyOp> {
public:
  using mlir::OpConversionPattern<cir::CopyOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::CopyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = adaptor.getSrc();
    auto dst = adaptor.getDst();

    // Get the size of the type being copied using DataLayout
    auto cirPtrType = mlir::dyn_cast<cir::PointerType>(op.getSrc().getType());
    if (!cirPtrType)
      return mlir::failure();

    auto pointeeType = cirPtrType.getPointee();
    auto module = op->getParentOfType<mlir::ModuleOp>();
    mlir::DataLayout dataLayout(module);
    llvm::TypeSize typeSize = dataLayout.getTypeSize(pointeeType);
    uint64_t copySize = typeSize.getFixedValue();

    // Create constant for the copy size
    auto i64Type = rewriter.getI64Type();
    auto sizeConst = rewriter.create<mlir::arith::ConstantOp>(
        loc, i64Type, rewriter.getIntegerAttr(i64Type, copySize));

    // Use llvm.memcpy for the copy operation
    // memcpy(dst, src, size, isVolatile)
    auto i1Type = rewriter.getI1Type();
    auto falseVal = rewriter.create<mlir::arith::ConstantOp>(
        loc, i1Type, rewriter.getBoolAttr(false));

    rewriter.replaceOpWithNewOp<mlir::LLVM::MemcpyOp>(
        op, dst, src, sizeConst, falseVal);
    return mlir::success();
  }
};

class CIRGetElementOpLowering
    : public mlir::OpConversionPattern<cir::GetElementOp> {
  using mlir::OpConversionPattern<cir::GetElementOp>::OpConversionPattern;

  bool isLoadStoreOrGetProducer(cir::GetElementOp op) const {
    for (auto *user : op->getUsers()) {
      if (!op->isBeforeInBlock(user))
        return false;
      if (isa<cir::LoadOp, cir::StoreOp, cir::GetElementOp>(*user))
        continue;
      return false;
    }
    return true;
  }

  // Rewrite
  //        cir.get_element(%base[%index])
  // to
  //        memref.reinterpret_cast (%base, %stride)
  //
  // MemRef Dialect doesn't have GEP-like operation. memref.reinterpret_cast
  // only been used to propagate %base and %index to memref.load/store and
  // should be erased after the conversion.
  mlir::LogicalResult
  matchAndRewrite(cir::GetElementOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Only rewrite if all users are load/stores.
    if (!isLoadStoreOrGetProducer(op))
      return mlir::failure();

    // Cast the index to the index type, if needed.
    auto index = adaptor.getIndex();
    auto indexType = rewriter.getIndexType();
    if (index.getType() != indexType)
      index = rewriter.create<mlir::arith::IndexCastOp>(op.getLoc(), indexType,
                                                        index);

    // Convert the destination type using helper.
    auto converted = getTypeConverter()->convertType(op.getType());
    if (auto memrefTy =
            mlir::dyn_cast_if_present<mlir::MemRefType>(converted)) {
      auto tryMemRef =
          ensureMemRefOrForward(op.getLoc(), memrefTy, adaptor.getBase(), op,
                                rewriter, "get_element");
      if (!tryMemRef)
        return mlir::success();
      rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
          op, *tryMemRef, adaptor.getBase(),
          /* offset */ index,
          /* sizes */ ArrayRef<mlir::OpFoldResult>{},
          /* strides */ ArrayRef<mlir::OpFoldResult>{},
          /* attr */ ArrayRef<mlir::NamedAttribute>{});
      return mlir::success();
    }

    if (mlir::isa_and_nonnull<mlir::LLVM::LLVMPointerType>(converted)) {
      auto pointerView = unwrapPointerLikeToMemRef(adaptor.getBase(), rewriter);
      if (!pointerView) {
        // Raw pointer arithmetic: scale index by element size if known.
        auto elemTy = op.getType();
        auto memElemTy = convertTypeForMemory(*getTypeConverter(), elemTy);
        mlir::Value scaledIndex = index;
        if (memElemTy && !memElemTy.isInteger(1)) {
          // Attempt to get size in bytes for element.
          uint64_t elemSizeBytes = 1;
          if (auto llvmInt = mlir::dyn_cast<mlir::IntegerType>(memElemTy))
            elemSizeBytes = llvmInt.getWidth() / 8;
          else if (auto fty = mlir::dyn_cast<mlir::FloatType>(memElemTy))
            elemSizeBytes = fty.getWidth() / 8;
          else if (mlir::isa<mlir::LLVM::LLVMPointerType>(memElemTy))
            elemSizeBytes = 8; // assume 64-bit pointer (TODO: datalayout)
          if (elemSizeBytes > 1) {
            auto idxTy = rewriter.getIndexType();
            auto cst = rewriter.create<mlir::arith::ConstantIndexOp>(
                op.getLoc(), elemSizeBytes);
            // Multiply index * elemSizeBytes
            scaledIndex =
                rewriter.create<mlir::arith::MulIOp>(op.getLoc(), index, cst);
          }
        }
        // ptr + scaledIndex (byte offset) via ptrtoint/add/inttoptr sequence.
        auto i64Ty = rewriter.getI64Type();
        auto baseInt = rewriter.create<mlir::LLVM::PtrToIntOp>(
            op.getLoc(), i64Ty, adaptor.getBase());
        auto idxInt = rewriter.create<mlir::arith::IndexCastOp>(
            op.getLoc(), i64Ty, scaledIndex);
        auto sum =
            rewriter.create<mlir::arith::AddIOp>(op.getLoc(), baseInt, idxInt);
        auto newPtr = rewriter.create<mlir::LLVM::IntToPtrOp>(
            op.getLoc(), converted, sum.getResult());
        rewriter.replaceOp(op, newPtr.getResult());
        return mlir::success();
      }

      auto memElemTy = convertTypeForMemory(*getTypeConverter(), op.getType());
      if (!memElemTy)
        return op.emitError()
               << "unable to derive memory element type for pointer result";

      auto memrefTy = mlir::MemRefType::get({}, memElemTy);
      auto reinterpret = rewriter.create<mlir::memref::ReinterpretCastOp>(
          op.getLoc(), memrefTy, pointerView->memref,
          /*offset*/ index, ArrayRef<mlir::OpFoldResult>{},
          ArrayRef<mlir::OpFoldResult>{}, ArrayRef<mlir::NamedAttribute>{});
      auto castBack = rewriter.create<mlir::UnrealizedConversionCastOp>(
          op.getLoc(), converted, reinterpret.getResult());
      rewriter.replaceOp(op, castBack.getResults());
      registerPointerBackingMemref(castBack.getResult(0), reinterpret.getResult());

      if (pointerView->bridgingCast && pointerView->bridgingCast->use_empty())
        rewriter.eraseOp(pointerView->bridgingCast);

      return mlir::success();
    }

    return op.emitError() << "get_element lowering: unsupported converted type"
                          << converted;
  }
};

class CIRGetMemberOpLowering
    : public mlir::OpConversionPattern<cir::GetMemberOp> {
public:
  using mlir::OpConversionPattern<cir::GetMemberOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::GetMemberOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Get the converted result type (should be llvm.ptr)
    auto convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return mlir::failure();

    auto basePtr = adaptor.getAddr();
    if (!basePtr) {
      // Input hasn't been converted yet, conversion framework will retry later
      return mlir::failure();
    }

    if (!basePtr.getType())
      return mlir::failure();

    // Get the struct type from the original (unconverted) base pointer
    auto basePtrType = mlir::dyn_cast<cir::PointerType>(op.getAddr().getType());
    if (!basePtrType) {
      return op.emitError("get_member base is not a CIR pointer type");
    }

    auto recordType = mlir::dyn_cast<cir::RecordType>(basePtrType.getPointee());
    if (!recordType) {
      return op.emitError("get_member base does not point to a record type");
    }

    // Get the member index
    uint64_t memberIndex = op.getIndex();

    auto members = recordType.getMembers();
    if (memberIndex >= members.size())
      return mlir::failure();

    // Get the module to access DataLayout
    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();

    // Create DataLayout for accurate size/alignment queries
    mlir::DataLayout dataLayout(module);

    // Calculate byte offset with proper alignment
    // C struct layout: each member is aligned to its natural alignment
    uint64_t byteOffset = 0;

    for (uint64_t i = 0; i < memberIndex; i++) {
      auto memberType = members[i];

      // Get the size and alignment for this member
      llvm::TypeSize typeSize = dataLayout.getTypeSize(memberType);
      uint64_t alignment = dataLayout.getTypeABIAlignment(memberType);

      // Align the current offset to the member's alignment requirement
      byteOffset = (byteOffset + alignment - 1) / alignment * alignment;

      // Add the size of this member
      if (typeSize.isScalable())
        return op.emitError("scalable types not supported in struct layout");

      byteOffset += typeSize.getFixedValue();
    }

    // Align to the target member's alignment before computing its address
    if (memberIndex < members.size()) {
      auto targetMemberType = members[memberIndex];
      uint64_t targetAlignment = dataLayout.getTypeABIAlignment(targetMemberType);
      byteOffset = (byteOffset + targetAlignment - 1) / targetAlignment * targetAlignment;
    }

    // Use llvm.getelementptr with byte offset
    // For opaque pointers, we can use GEP with i8 element type
    auto i8Type = rewriter.getI8Type();

    auto gepOp = rewriter.create<mlir::LLVM::GEPOp>(
        loc, convertedType, i8Type, basePtr,
        mlir::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(byteOffset)},
        mlir::LLVM::GEPNoWrapFlags::none);

    rewriter.replaceOp(op, gepOp.getResult());
    return mlir::success();
  }
};

class CIRPtrStrideOpLowering
    : public mlir::OpConversionPattern<cir::PtrStrideOp> {
public:
  using mlir::OpConversionPattern<cir::PtrStrideOp>::OpConversionPattern;

  // Return true if PtrStrideOp is produced by cast with array_to_ptrdecay kind
  // and they are in the same block.
  inline bool isCastArrayToPtrConsumer(cir::PtrStrideOp op) const {
    auto castOp = op->getOperand(0).getDefiningOp<cir::CastOp>();
    if (!castOp)
      return false;
    if (castOp.getKind() != cir::CastKind::array_to_ptrdecay)
      return false;
    if (!castOp->hasOneUse())
      return false;
    if (!castOp->isBeforeInBlock(op))
      return false;
    return true;
  }

  // Return true if all the PtrStrideOp users are load, store or cast
  // with array_to_ptrdecay kind and they are in the same block.
  inline bool isLoadStoreOrCastArrayToPtrProduer(cir::PtrStrideOp op) const {
    if (op.use_empty())
      return false;
    for (auto *user : op->getUsers()) {
      if (!op->isBeforeInBlock(user))
        return false;
      if (isa<cir::LoadOp, cir::StoreOp, cir::GetElementOp>(*user))
        continue;
      auto castOp = dyn_cast<cir::CastOp>(*user);
      if (castOp && (castOp.getKind() == cir::CastKind::array_to_ptrdecay))
        continue;
      return false;
    }
    return true;
  }

  inline mlir::Type convertTy(mlir::Type ty) const {
    return getTypeConverter()->convertType(ty);
  }

  // Rewrite
  //        %0 = cir.cast(array_to_ptrdecay, %base)
  //        cir.ptr_stride(%0, %stride)
  // to
  //        memref.reinterpret_cast (%base, %stride)
  //
  // MemRef Dialect doesn't have GEP-like operation. memref.reinterpret_cast
  // only been used to propogate %base and %stride to memref.load/store and
  // should be erased after the conversion.
  mlir::LogicalResult
  matchAndRewrite(cir::PtrStrideOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Check if this is the special array-to-pointer-decay case
    bool isArrayPtrDecayCase = isCastArrayToPtrConsumer(op) &&
                                isLoadStoreOrCastArrayToPtrProduer(op);

    if (!isArrayPtrDecayCase) {
      // General pointer arithmetic case: use GEP for element-wise stride
      auto loc = op.getLoc();
      auto base = adaptor.getBase();
      auto stride = adaptor.getStride();
      auto dstType = convertTy(op.getType());

      if (mlir::isa<mlir::LLVM::LLVMPointerType>(base.getType())) {
        // Base is LLVM pointer: use llvm.getelementptr with byte-based indexing
        // Get the pointee type to calculate element size
        auto cirPtrType = mlir::dyn_cast<cir::PointerType>(op.getBase().getType());
        if (!cirPtrType)
          return op.emitError() << "ptr_stride base is not a CIR pointer type";

        auto pointeeType = cirPtrType.getPointee();

        // Get element size using DataLayout
        auto module = op->getParentOfType<mlir::ModuleOp>();
        if (!module)
          return mlir::failure();
        mlir::DataLayout dataLayout(module);
        llvm::TypeSize typeSize = dataLayout.getTypeSize(pointeeType);
        if (typeSize.isScalable())
          return op.emitError() << "scalable types not supported in ptr_stride";

        uint64_t elemSize = typeSize.getFixedValue();

        // Convert stride from element count to byte offset
        mlir::Value byteStride;
        if (elemSize == 1) {
          // Already in bytes
          byteStride = stride;
        } else {
          // Multiply stride by element size
          auto elemSizeVal = rewriter.create<mlir::arith::ConstantOp>(
              loc, stride.getType(),
              rewriter.getIntegerAttr(stride.getType(), elemSize));
          byteStride = rewriter.create<mlir::arith::MulIOp>(
              loc, stride, elemSizeVal);
        }

        // Create GEP with i8 element type (byte-based addressing)
        auto i8Type = rewriter.getI8Type();
        rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
            op, dstType, i8Type, base,
            mlir::ArrayRef<mlir::LLVM::GEPArg>{byteStride},
            mlir::LLVM::GEPNoWrapFlags::none);
        return mlir::success();
      } else if (mlir::isa<mlir::IntegerType>(base.getType())) {
        // Base is integer (pointer represented as intptr_t)
        // Convert to byte arithmetic
        auto cirPtrType = mlir::dyn_cast<cir::PointerType>(op.getBase().getType());
        if (!cirPtrType)
          return op.emitError() << "ptr_stride base is not a CIR pointer type";

        auto pointeeType = cirPtrType.getPointee();

        // Get element size
        auto module = op->getParentOfType<mlir::ModuleOp>();
        if (!module)
          return mlir::failure();
        mlir::DataLayout dataLayout(module);
        llvm::TypeSize typeSize = dataLayout.getTypeSize(pointeeType);
        if (typeSize.isScalable())
          return op.emitError() << "scalable types not supported in ptr_stride";

        uint64_t elemSize = typeSize.getFixedValue();

        // Compute byte offset: stride * elemSize
        auto elemSizeVal = rewriter.create<mlir::arith::ConstantOp>(
            loc, stride.getType(),
            rewriter.getIntegerAttr(stride.getType(), elemSize));
        auto byteOffset = rewriter.create<mlir::arith::MulIOp>(
            loc, stride, elemSizeVal);

        // Add to base pointer
        auto result = rewriter.create<mlir::arith::AddIOp>(loc, base, byteOffset);
        rewriter.replaceOp(op, result);
        return mlir::success();
      } else {
        return op.emitError() << "ptr_stride with unsupported base type: " << base.getType();
      }
    }

    // Original array-to-pointer-decay special case handling follows:
    auto baseVal = adaptor.getBase();
    auto baseOp = getMemrefReinterpretCastOp(baseVal);
    auto dstType = op.getType();
    auto converted = convertTy(dstType);
    std::optional<mlir::MemRefType> maybeDst;
    mlir::Value baseMemRef;
    mlir::Operation *castToErase = nullptr;

    if (auto memrefTy =
            mlir::dyn_cast_if_present<mlir::MemRefType>(converted)) {
      if (!baseOp)
        return mlir::failure();
      baseMemRef = baseOp->getOperand(0);
      maybeDst = ensureMemRefOrForward(op.getLoc(), memrefTy, baseVal, op,
                                       rewriter, "ptr_stride");
      if (!maybeDst)
        return mlir::success();
    } else if (mlir::isa_and_nonnull<mlir::LLVM::LLVMPointerType>(converted)) {
      auto pointerView = unwrapPointerLikeToMemRef(baseVal, rewriter);
      if (!pointerView) {
        rewriter.replaceOp(op, baseVal);
        return mlir::success();
      }
      baseMemRef = pointerView->memref;
      castToErase = pointerView->bridgingCast;
      auto memElemTy = convertTypeForMemory(*getTypeConverter(), dstType);
      if (!memElemTy)
        return op.emitError()
               << "unable to derive memory element type for pointer result";
      maybeDst = mlir::MemRefType::get({}, memElemTy);
    } else {
      return op.emitError() << "ptr_stride lowering: unsupported converted"
                            << " type " << converted;
    }

    auto newDstType = *maybeDst;
    auto stride = adaptor.getStride();
    auto indexType = rewriter.getIndexType();
    // Generate casting if the stride is not index type.
    if (stride.getType() != indexType)
      stride = rewriter.create<mlir::arith::IndexCastOp>(op.getLoc(), indexType,
                                                         stride);

    auto reinterpret = rewriter.create<mlir::memref::ReinterpretCastOp>(
        op.getLoc(), newDstType, baseMemRef, stride, mlir::ValueRange{},
        mlir::ValueRange{}, llvm::ArrayRef<mlir::NamedAttribute>{});

    if (mlir::isa<mlir::LLVM::LLVMPointerType>(converted)) {
      // Provide element-size aware byte stride for raw pointers when a memref
      // view exists: reinterpret cast indexes units-of-elements; convert to
      // byte offset via (stride * elemSizeBytes).
      auto memElemTy = convertTypeForMemory(*getTypeConverter(), op.getType());
      uint64_t elemSizeBytes = 1;
      if (memElemTy) {
        if (auto it = mlir::dyn_cast<mlir::IntegerType>(memElemTy))
          elemSizeBytes = it.getWidth() / 8;
        else if (auto ft = mlir::dyn_cast<mlir::FloatType>(memElemTy))
          elemSizeBytes = ft.getWidth() / 8;
        else if (mlir::isa<mlir::LLVM::LLVMPointerType>(memElemTy))
          elemSizeBytes = 8; // TODO: compute from data layout
      }
      auto loc = op.getLoc();
      auto i64Ty = rewriter.getI64Type();
      // baseMemRef is a memref value; we need an llvm.ptr to the buffer. We
      // fall back to reusing the prior reinterpret cast result's bridging
      // pointer if available; otherwise we cannot form raw pointer arithmetic
      // here (conservatively skip scaling path and just forward original
      // reinterpret result via unrealized cast). For now, bail out if we can't
      // detect an existing pointer origin.
      if (!mlir::isa<mlir::MemRefType>(baseMemRef.getType())) {
        return op.emitError() << "expected memref base for ptr stride lowering";
      }
      // Degrade: just produce the original reinterpret result cast to pointer
      // if stride is zero; else approximate by not adjusting.
      auto zeroIdx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      if (stride != zeroIdx) {
        // Without a direct buffer pointer, fallback to previous behavior: cast
        // reinterpret result back.
        auto castBack = rewriter.create<mlir::UnrealizedConversionCastOp>(
            loc, converted, reinterpret.getResult());
        rewriter.replaceOp(op, castBack.getResults());
        return mlir::success();
      }
      auto castBack = rewriter.create<mlir::UnrealizedConversionCastOp>(
          loc, converted, reinterpret.getResult());
      rewriter.replaceOp(op, castBack.getResults());
      registerPointerBackingMemref(castBack.getResult(0), reinterpret.getResult());
      return mlir::success();
    } else {
      rewriter.replaceOp(op, reinterpret.getResult());
    }

    if (baseOp && baseOp->use_empty())
      rewriter.eraseOp(baseOp);
    if (castToErase && castToErase->use_empty())
      rewriter.eraseOp(castToErase);
    return mlir::success();
  }
};

class CIRUnreachableOpLowering
    : public mlir::OpConversionPattern<cir::UnreachableOp> {
public:
  using OpConversionPattern<cir::UnreachableOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::UnreachableOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op);
    return mlir::success();
  }
};

class CIRTrapOpLowering : public mlir::OpConversionPattern<cir::TrapOp> {
public:
  using OpConversionPattern<cir::TrapOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::TrapOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPointAfter(op);
    auto trapIntrinsicName = rewriter.getStringAttr("llvm.trap");
    rewriter.create<mlir::LLVM::CallIntrinsicOp>(op.getLoc(), trapIntrinsicName,
                                                 /*args=*/mlir::ValueRange());
    rewriter.create<mlir::LLVM::UnreachableOp>(op.getLoc());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

void populateCIRToMLIRConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &converter) {
  patterns.add<CIRReturnLowering, CIRBrOpLowering>(patterns.getContext());

  patterns.add<
      CIRATanOpLowering, CIRCmpOpLowering, CIRCallOpLowering,
      CIRUnaryOpLowering, CIRBinOpLowering, CIRLoadOpLowering,
      CIRConstantOpLowering, CIRStoreOpLowering, CIRAllocaOpLowering,
      CIRFuncOpLowering, CIRBrCondOpLowering, CIRTernaryOpLowering,
      CIRYieldOpLowering, CIRCosOpLowering, CIRGlobalOpLowering,
      CIRGetGlobalOpLowering, CIRCastOpLowering, CIRPtrStrideOpLowering,
      CIRGetElementOpLowering, CIRGetMemberOpLowering, CIRSqrtOpLowering,
      CIRCeilOpLowering, CIRExp2OpLowering, CIRExpOpLowering, CIRFAbsOpLowering,
      CIRAbsOpLowering, CIRFloorOpLowering, CIRLog10OpLowering,
      CIRLog2OpLowering, CIRLogOpLowering, CIRRoundOpLowering, CIRSinOpLowering,
      CIRShiftOpLowering, CIRBitClzOpLowering, CIRBitCtzOpLowering,
      CIRBitPopcountOpLowering, CIRBitClrsbOpLowering, CIRBitFfsOpLowering,
      CIRBitParityOpLowering, CIRIfOpLowering, CIRVectorCreateLowering,
      CIRVectorInsertLowering, CIRVectorExtractLowering, CIRVectorCmpOpLowering,
      CIRACosOpLowering, CIRASinOpLowering, CIRUnreachableOpLowering,
      CIRTanOpLowering, CIRTrapOpLowering>(converter, patterns.getContext());
}

static mlir::TypeConverter prepareTypeConverter() {
  mlir::TypeConverter converter;
  converter.addConversion([&](cir::PointerType type) -> mlir::Type {
    // Represent CIR raw pointers as opaque LLVM pointers. This avoids forcing
    // them through memref descriptors (which complicated comparisons and
    // allocator call lowering) and eliminates unresolved materialization
    // issues when pointer values are only tested for null / compared.
    auto *ctx = type.getContext();
    auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    // Special-case array-to-pointer decay: if pointee is an array we rely on
    // prior lowering producing a memref for the array itself; the pointer to
    // array then decays naturally via existing cast ops. For simplicity still
    // return a pointer here; subsequent decay uses bitcast.
    return llvmPtrTy;
  });
  converter.addConversion(
      [&](mlir::IntegerType type) -> mlir::Type { return type; });
  converter.addConversion(
      [&](mlir::FloatType type) -> mlir::Type { return type; });
  converter.addConversion([&](cir::VoidType type) -> mlir::Type { return {}; });
  converter.addConversion([&](cir::IntType type) -> mlir::Type {
    // arith dialect ops doesn't take signed integer -- drop cir sign here
    return mlir::IntegerType::get(
        type.getContext(), type.getWidth(),
        mlir::IntegerType::SignednessSemantics::Signless);
  });
  converter.addConversion([&](cir::BoolType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(), 1);
  });
  converter.addConversion([&](cir::SingleType type) -> mlir::Type {
    return mlir::Float32Type::get(type.getContext());
  });
  converter.addConversion([&](cir::DoubleType type) -> mlir::Type {
    return mlir::Float64Type::get(type.getContext());
  });
  converter.addConversion([&](cir::FP80Type type) -> mlir::Type {
    return mlir::Float80Type::get(type.getContext());
  });
  converter.addConversion([&](cir::LongDoubleType type) -> mlir::Type {
    return converter.convertType(type.getUnderlying());
  });
  converter.addConversion([&](cir::FP128Type type) -> mlir::Type {
    return mlir::Float128Type::get(type.getContext());
  });
  converter.addConversion([&](cir::FP16Type type) -> mlir::Type {
    return mlir::Float16Type::get(type.getContext());
  });
  converter.addConversion([&](cir::BF16Type type) -> mlir::Type {
    return mlir::BFloat16Type::get(type.getContext());
  });
  converter.addConversion([&](cir::ArrayType type) -> mlir::Type {
    SmallVector<int64_t> shape;
    mlir::Type curType = type;
    while (auto arrayType = dyn_cast<cir::ArrayType>(curType)) {
      shape.push_back(arrayType.getSize());
      curType = arrayType.getElementType();
    }
    auto elementType = converter.convertType(curType);
    // FIXME: The element type might not be converted (e.g. struct)
    if (!elementType)
      return nullptr;
    return mlir::MemRefType::get(shape, elementType);
  });
  converter.addConversion([&](cir::VectorType type) -> mlir::Type {
    auto ty = converter.convertType(type.getElementType());
    return mlir::VectorType::get(type.getSize(), ty);
  });

  // Add source materialization to handle unconverted CIR operations
  // that need to use values from converted operations
  converter.addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return nullptr;

        // Only create materialization if the input is valid
        auto input = inputs[0];
        if (!input || !input.getType())
          return nullptr;

        // Just create an unrealized conversion cast for any needed conversions
        return builder.create<mlir::UnrealizedConversionCastOp>(
            loc, resultType, input).getResult(0);
      });

  return converter;
}

void ConvertCIRToMLIRPass::runOnOperation() {
  mlir::MLIRContext *context = &getContext();
  mlir::ModuleOp theModule = getOperation();

  clearPointerBackingMemrefs();

  auto converter = prepareTypeConverter();

  mlir::RewritePatternSet patterns(&getContext());

  populateCIRLoopToSCFConversionPatterns(patterns, converter);
  populateCIRToMLIRConversionPatterns(patterns, converter);

  mlir::ConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp, mlir::UnrealizedConversionCastOp>();
  target.addLegalDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                         mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                         mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
                         mlir::math::MathDialect, mlir::vector::VectorDialect,
                         mlir::LLVM::LLVMDialect>();

  // Mark the entire CIR dialect as illegal to force conversion
  target.addIllegalDialect<cir::CIRDialect>();

  // Keep control flow markers legal only when inside CIR operations
  target.addDynamicallyLegalOp<cir::YieldOp>([](cir::YieldOp op) {
    auto *parentOp = op->getParentOp();
    // Legal only if parent is still a CIR operation
    return parentOp->getDialect()->getNamespace() == "cir";
  });

  target.addDynamicallyLegalOp<cir::ConditionOp>([](cir::ConditionOp op) {
    auto *parentOp = op->getParentOp();
    // Legal only if parent is still a CIR operation
    return parentOp->getDialect()->getNamespace() == "cir";
  });

  // cir.continue and cir.break should be lowered, not kept legal
  // They are erased as they should have been handled by SCF preparation

  patterns.add<CIRCastOpLowering, CIRSelectOpLowering, CIRCopyOpLowering,
               CIRIfOpLowering, CIRScopeOpLowering, CIRYieldOpLowering,
               CIRConditionOpLowering, CIRBreakOpLowering, CIRContinueOpLowering>(converter, context);

  // Use partial conversion - this allows intermediate states during conversion
  if (mlir::failed(mlir::applyPartialConversion(theModule, target,
                                                 std::move(patterns)))) {
    signalPassFailure();
  }
}

mlir::ModuleOp lowerFromCIRToMLIRToLLVMDialect(mlir::ModuleOp theModule,
                                               mlir::MLIRContext *mlirCtx) {
  llvm::TimeTraceScope scope("Lower from CIR to MLIR To LLVM Dialect");
  if (!mlirCtx) {
    mlirCtx = theModule.getContext();
  }

  mlir::PassManager pm(mlirCtx);

  pm.addPass(createConvertCIRToMLIRPass());
  pm.addPass(createConvertMLIRToLLVMPass());

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    report_fatal_error(
        "The pass manager failed to lower CIR to LLVMIR dialect!");

  // Now that we ran all the lowering passes, verify the final output.
  if (theModule.verify().failed())
    report_fatal_error("Verification of the final LLVMIR dialect failed!");

  return theModule;
}

std::unique_ptr<llvm::Module>
lowerFromCIRToMLIRToLLVMIR(mlir::ModuleOp theModule,
                           std::unique_ptr<mlir::MLIRContext> mlirCtx,
                           llvm::LLVMContext &llvmCtx) {
  llvm::TimeTraceScope scope("Lower from CIR to MLIR To LLVM");

  lowerFromCIRToMLIRToLLVMDialect(theModule, mlirCtx.get());

  mlir::registerBuiltinDialectTranslation(*mlirCtx);
  mlir::registerLLVMDialectTranslation(*mlirCtx);
  mlir::registerOpenMPDialectTranslation(*mlirCtx);

  auto llvmModule = mlir::translateModuleToLLVMIR(theModule, llvmCtx);

  if (!llvmModule)
    report_fatal_error("Lowering from LLVMIR dialect to llvm IR failed!");

  return llvmModule;
}

mlir::ModuleOp lowerDirectlyFromCIRToLLVMIR(mlir::ModuleOp theModule,
                                            mlir::MLIRContext *mlirCtx) {
  auto llvmModule = lowerFromCIRToMLIR(theModule, mlirCtx);
  if (!llvmModule.getOperation())
    return {};

  mlir::PassManager pm(mlirCtx);

  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());

  if (mlir::failed(pm.run(llvmModule))) {
    llvmModule.emitError("The pass manager failed to lower the module");
    return {};
  }

  return llvmModule;
}

std::unique_ptr<mlir::Pass> createConvertCIRToMLIRPass() {
  return std::make_unique<ConvertCIRToMLIRPass>();
}

mlir::ModuleOp lowerFromCIRToMLIR(mlir::ModuleOp theModule,
                                  mlir::MLIRContext *mlirCtx) {
  llvm::TimeTraceScope scope("Lower CIR To MLIR");

  mlir::PassManager pm(mlirCtx);
  pm.addPass(createConvertCIRToMLIRPass());

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    report_fatal_error(
        "The pass manager failed to lower CIR to MLIR standard dialects!");
  // Now that we ran all the lowering passes, verify the final output.
  if (theModule.verify().failed())
    report_fatal_error(
        "Verification of the final MLIR in standard dialects failed!");

  return theModule;
}

} // namespace cir
