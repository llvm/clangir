//===- CIROpInterfaces.td - CIR Op Interface Definitions --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_OP_INTERFACES
#define MLIR_CIR_OP_INTERFACES

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

let cppNamespace = "::cir" in {
  // The CIRCallOpInterface must be used instead of CallOpInterface when looking
  // at arguments and other bits of CallOp. This creates a level of abstraction
  // that's useful for handling indirect calls and other details.
  def CIRCallOpInterface
      : OpInterface<"CIRCallOpInterface", [CallOpInterface]> {
    let methods = [
      InterfaceMethod<"", "mlir::Operation::operand_iterator",
                      "arg_operand_begin", (ins)>,
      InterfaceMethod<"", "mlir::Operation::operand_iterator",
                      "arg_operand_end", (ins)>,
      InterfaceMethod<
          "Return the operand at index 'i', accounts for indirect call or "
          "exception info",
          "mlir::Value", "getArgOperand",
          (ins "unsigned"
           : $i)>,
      InterfaceMethod<
          "Return the number of operands, accounts for indirect call or "
          "exception info",
          "unsigned", "getNumArgOperands", (ins)>,
      InterfaceMethod<
          "Return the calling convention of the call operation",
          "cir::CallingConv", "getCallingConv", (ins)>,
      InterfaceMethod<
          "Return the side effects of the call operation",
          "cir::SideEffect", "getSideEffect", (ins)>,
    ];
  }

  def CIRCallableOpInterface
      : OpInterface<"CIRCallableOpInterface"> {
    let description = [{
       The CIRCallableOpInterface is created due to the flexibility that 
       either AliasOp or FuncOp is returned from the same function.

       It also handles all the default and similar methods that AliasOp and FuncOp might have so
       we don't have to define them twice.

       Notice that even though we are declaring methods from CallableOpInterface and
       FunctionOpInterface, we cannot add them to the inheritance in OpInterface<...>, as
       it will produce clashing of definitions.
    }];
    let methods = [
      InterfaceMethod<
          "Returns the region on the current operation that is callable. This may "
          "return null in the case of an external callable object, e.g. an external "
          "function.",
          "::mlir::Region*", "getCallableRegion", (ins)>,
      InterfaceMethod<
          "Returns the type of this function.",
          "cir::FuncType", "getFunctionType",
          (ins)>,
      InterfaceMethod<
          "Returns the results types that the callable region produces when "
          "executed.",
          "llvm::ArrayRef<mlir::Type>", "getCallableResults",
          (ins), [{}],
      /*defaultImplementation=*/[{
          return $_op.getFunctionType().getReturnTypes();
      }]>,
      InterfaceMethod<
          "Returns the result attributes for all callable region results or null if "
          "there are none.",
          "::mlir::ArrayAttr", "getCallableResAttrs",
          (ins), [{}],
      /*defaultImplementation=*/[{
          return $_op.getResAttrs().value_or(nullptr);
      }]>,
      InterfaceMethod<
          "Returns the argument types of this function.",
          "llvm::ArrayRef<mlir::Type>", "getArgumentTypes",
          (ins), [{}],
      /*defaultImplementation=*/[{
       return $_op.getFunctionType().getInputs();
      }]>,
      InterfaceMethod<
          "Returns 0 or 1 result type of this function (0 in the case of a function "
          "returing void).",
          "llvm::ArrayRef<mlir::Type>", "getResultTypes",
          (ins), [{}],
      /*defaultImplementation=*/[{
       return $_op.getFunctionType().getReturnTypes();
      }]>,
      InterfaceMethod<
          "Return the calling convention of the called operation",
          "cir::CallingConv", "getCallingConv", (ins)>,
      InterfaceMethod<
          "Return if the callable is prototype defined or not",
          "bool", "getNoProto", (ins)>,

    ];
  }

  def CIRGlobalValueInterface
      : OpInterface<"CIRGlobalValueInterface", [Symbol]> {

    let methods = [
      InterfaceMethod<"",
      "bool", "hasExternalLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isExternalLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasAvailableExternallyLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isAvailableExternallyLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasLinkOnceLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isLinkOnceLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasLinkOnceAnyLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isLinkOnceAnyLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasLinkOnceODRLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isLinkOnceODRLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasWeakLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isWeakLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasWeakAnyLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isWeakAnyLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasWeakODRLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isWeakODRLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasInternalLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isInternalLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasPrivateLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isPrivateLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasLocalLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isLocalLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasExternalWeakLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isExternalWeakLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasCommonLinkage", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isCommonLinkage($_op.getLinkage());
      }]
      >,
      InterfaceMethod<"",
      "bool", "isDeclarationForLinker", (ins), [{}],
      /*defaultImplementation=*/[{
        if ($_op.hasAvailableExternallyLinkage())
          return true;
        return $_op.isDeclaration();
      }]
      >,
      InterfaceMethod<"",
      "bool", "hasComdat", (ins), [{}],
      /*defaultImplementation=*/[{
        return $_op.getComdat();
      }]
      >,
      InterfaceMethod<"",
      "void", "setDSOLocal", (ins "bool":$val), [{}],
      /*defaultImplementation=*/[{
        $_op.setDsoLocal(val);
      }]
      >,
      InterfaceMethod<"",
      "bool", "isDSOLocal", (ins), [{}],
      /*defaultImplementation=*/[{
        return $_op.getDsoLocal();
      }]
      >,
      InterfaceMethod<"",
      "bool", "isWeakForLinker", (ins), [{}],
      /*defaultImplementation=*/[{
        return cir::isWeakForLinker($_op.getLinkage());
      }]
      >
    ];
    let extraClassDeclaration = [{
      bool hasDefaultVisibility();
      bool canBenefitFromLocalAlias();
    }];
  }

} // namespace cir

#endif // MLIR_CIR_OP_INTERFACES
