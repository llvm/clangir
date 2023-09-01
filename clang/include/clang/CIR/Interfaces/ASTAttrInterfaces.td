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
            InterfaceMethod< "", "bool", "hasOwnerAttr", (ins), [{}],
                /*defaultImplementation=*/ [{
                    return $_attr.getAstDecl()->template hasAttr< clang::OwnerAttr >();
                }]
            >,
            InterfaceMethod< "", "bool", "hasPointerAttr", (ins), [{}],
                /*defaultImplementation=*/ [{
                    return $_attr.getAstDecl()->template hasAttr< clang::PointerAttr >();
                }]
            >,
            InterfaceMethod< "", "ASTDeclInterface", "getNextDeclInContext", (ins), [{}],
                /*defaultImplementation=*/ [{
                    return $_attr.getNextDeclInContext();
                }]
            >
        ];
    }

    def ASTDeclContextInterface : AttrInterface< "ASTDeclContextInterface" > {}

    def ASTNamedDeclInterface : AttrInterface< "ASTNamedDeclInterface",
        [ASTDeclInterface]
    > {
        let methods = [
            InterfaceMethod< "", "clang::DeclarationName", "getDeclName", (ins), [{}],
                /*defaultImplementation=*/ [{
                    return $_attr.getAstDecl()->getDeclName();
                }]
            >,
            InterfaceMethod< "", "llvm::StringRef", "getName", (ins), [{}],
                /*defaultImplementation=*/ [{
                    return $_attr.getAstDecl()->getName();
                }]
            >
        ];
    }

    def ASTValueDeclInterface : AttrInterface< "ASTValueDeclInterface",
        [ASTNamedDeclInterface]
    > {}

    def ASTDeclaratorDeclInterface : AttrInterface< "ASTDeclaratorDeclInterface",
        [ASTValueDeclInterface]
    > {}

    def ASTVarDeclInterface : AttrInterface< "ASTVarDeclInterface",
        [ASTDeclaratorDeclInterface]
    > {
        let methods = [
            InterfaceMethod< "", "void", "mangleDynamicInitializer", (ins "llvm::raw_ostream&":$Out), [{}],
                /*defaultImplementation=*/ [{
                    std::unique_ptr<clang::MangleContext> MangleCtx(
                        $_attr.getAstDecl()->getASTContext().createMangleContext());
                    MangleCtx->mangleDynamicInitializer($_attr.getAstDecl(), Out);
                }]
            >
        ];
    }

    def ASTFunctionDeclInterface : AttrInterface< "ASTFunctionDeclInterface",
        [ASTDeclaratorDeclInterface, ASTDeclContextInterface]
    > {
        let methods = [
            InterfaceMethod< "", "bool", "isOverloadedOperator", (ins), [{}],
                /*defaultImplementation=*/ [{
                    return $_attr.getAstDecl()->isOverloadedOperator();
                }]
            >
        ];
    }

    def ASTCXXMethodDeclInterface : AttrInterface< "ASTCXXMethodDeclInterface",
        [ASTFunctionDeclInterface]
    > {
        let methods = [
            InterfaceMethod< "", "bool", "isCopyAssignmentOperator", (ins), [{}],
                /*defaultImplementation=*/ [{
                    if (auto decl = dyn_cast< clang::CXXMethodDecl >($_attr.getAstDecl()))
                        return decl->isCopyAssignmentOperator();
                    return false;
                }]
            >,
            InterfaceMethod< "", "bool", "isMoveAssignmentOperator", (ins), [{}],
                /*defaultImplementation=*/ [{
                    if (auto decl = dyn_cast< clang::CXXMethodDecl >($_attr.getAstDecl()))
                        return decl->isMoveAssignmentOperator();
                    return false;
                }]
            >
        ];
    }

    def ASTCXXConstructDeclInterface : AttrInterface< "ASTCXXConstructDeclInterface",
        [ASTCXXMethodDeclInterface]
    > {}

    def ASTCXXConversionDeclInterface : AttrInterface< "ASTCXXConversionDeclInterface",
        [ASTCXXMethodDeclInterface]
    > {}

    def ASTCXXDestructDeclInterface : AttrInterface< "ASTCXXDestructDeclInterface",
        [ASTCXXMethodDeclInterface]
    > {}

    def ASTTypeDeclInterface : AttrInterface< "ASTTypeDeclInterface",
        [ASTNamedDeclInterface]
    > {}

    def ASTTagDeclInterface : AttrInterface< "ASTTagDeclInterface",
        [ASTTypeDeclInterface, ASTDeclContextInterface]
    > {}

    def ASTEnumDeclInterface : AttrInterface< "ASTEnumDeclInterface",
        [ASTTagDeclInterface]
    > {}

    def ASTRecordDeclInterface : AttrInterface< "ASTRecordDeclInterface",
        [ASTTagDeclInterface]
    > {}

    def ASTCXXRecordDeclInterface : AttrInterface< "ASTCXXRecordDeclInterface",
        [ASTRecordDeclInterface]
    > {
        let methods = [
            InterfaceMethod< "", "bool", "isLambda", (ins), [{}],
                /*defaultImplementation=*/ [{
                    return $_attr.getAstDecl()->isLambda();
                }]
            >,
            InterfaceMethod< "",
                "llvm::SmallVector< ASTCXXRecordDeclInterface, 4 >", "CXXRecordDecls"
            >
        ];
    }

    def ASTClassTemplateSpecializationDeclInterface
        : AttrInterface< "ASTClassTemplateSpecializationDeclInterface",
            [ASTCXXRecordDeclInterface]
        >
    {}

    def ASTClassTemplatePartialSpecializationDeclInterface
        : AttrInterface< "ASTClassTemplatePartialSpecializationDeclInterface",
            [ASTClassTemplateSpecializationDeclInterface]
        >
    {}
}

#endif // MLIR_CIR_INTERFACES_AST_ATTR_INTERFACES
