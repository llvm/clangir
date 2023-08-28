//===- ASTAttrInterfaces.td - CIR AST Interface Definitions -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_INTERFACES_AST_ATTR_INTERFACES
#define MLIR_CIR_INTERFACES_AST_ATTR_INTERFACES

include "mlir/IR/OpBase.td"

let cppNamespace = "::mlir::cir" in {

    def ASTDeclInterface : AttrInterface< "ASTDeclInterface" > {
        let methods = [
            // hasAttr, cannot be templated :(
            InterfaceMethod< "", "bool", "hasOwnerAttr", (ins),
                /* methodBody = */ [{}],
                /* defaultImplementation= */[{
                    return $_attr.getDecl()->template hasAttr< clang::OwnerAttr >();
                }]
            >,
            InterfaceMethod< "", "bool", "hasPointerAttr", (ins),
                /* methodBody = */ [{}],
                /* defaultImplementation= */[{
                    return $_attr.getDecl()->template hasAttr< clang::PointerAttr >();
                }]
            >
        ];
    }

    // def ASTNamedDeclInterface : AttrInterface< "ASTNamedDeclInterface", [ASTDeclInterface] > {}

    // def ASTTypeDeclInterface : AttrInterface< "ASTTypeDeclInterface", [ASTNamedDeclInterface] > {}

    // def ASTTagDeclInterface : AttrInterface< "ASTTagDeclInterface", [ASTTypeDeclInterface] > {}

    // def ASTRecordDeclInterface : AttrInterface< "ASTRecordDeclInterface", [ASTTagDeclInterface] > {}

    def ASTCXXRecorDeclInterface : AttrInterface< "ASTCXXRecorDeclInterface", [ASTDeclInterface] > {
        let methods = [
            InterfaceMethod< "", "bool", "isLambda", (ins), [{}],
                /*defaultImplementation=*/ [{
                    return $_attr.getDecl()->isLambda();
                }]
            >
        ];
    }

    def ASTVarDeclInterface : AttrInterface< "ASTVarDeclInterface", [ASTDeclInterface] > {}

    def ASTFunctionDeclInterface : AttrInterface< "ASTFunctionDeclInterface", [ASTDeclInterface] > {
        let methods = [
            InterfaceMethod< "", "bool", "isOverloadedOperator", (ins), [{}],
                /*defaultImplementation=*/ [{
                    return $_attr.getDecl()->isOverloadedOperator();
                }]
            >,
            // TODO move to ASTMethodDeclInterface
            InterfaceMethod< "", "bool", "isCopyAssignmentOperator", (ins), [{}],
                /*defaultImplementation=*/ [{
                    if (auto decl = dyn_cast< clang::CXXMethodDecl >($_attr.getDecl()))
                        return decl->isCopyAssignmentOperator();
                    return false;
                }]
            >,
            // TODO move to ASTMethodDeclInterface
            InterfaceMethod< "", "bool", "isMoveAssignmentOperator", (ins), [{}],
                /*defaultImplementation=*/ [{
                    if (auto decl = dyn_cast< clang::CXXMethodDecl >($_attr.getDecl()))
                        return decl->isMoveAssignmentOperator();
                    return false;
                }]
            >
        ];
    }

    def ASTCXXMethodDeclInterface : AttrInterface< "ASTCXXMethodDeclInterface", [ASTDeclInterface] > {}
}

#endif // MLIR_CIR_INTERFACES_AST_ATTR_INTERFACES
