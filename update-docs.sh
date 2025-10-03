#!/bin/bash

# If generating doc, update the BUILD dir below. This is usually used
# from another clone of llvm/clangir.

# Note that this build path is outside this repo checkout, update to
# your needs if using this hacky script.
# This should be up-to-date, use `ninja clang-cir-doc`
BUILD="${BUILD_DIR:-../clangir/Build+Release+Libcxx+Assert}/tools/clang/docs"

TEMPLATE="---\nparent: CIR Dialect\nnav_order: POSITION\n---\n\n# TITLE\n\n* toc\n{:toc}\n\n---\n"

echo -e $TEMPLATE | sed -e "s@POSITION@1@g" -e "s@TITLE@Operations@g" > Dialect/ops.md && cat ${BUILD}/Dialects/CIROps.md >> Dialect/ops.md
echo -e $TEMPLATE | sed -e "s@POSITION@2@g" -e "s@TITLE@Types@g" > Dialect/types.md && cat ${BUILD}/Dialects/CIRTypes.md >> Dialect/types.md
echo -e $TEMPLATE | sed -e "s@POSITION@3@g" -e "s@TITLE@Attributes@g" > Dialect/attrs.md && cat ${BUILD}/Dialects/CIRAttrs.md >> Dialect/attrs.md
echo -e $TEMPLATE | sed -e "s@POSITION@4@g" -e "s@TITLE@Passes@g" > Dialect/passes.md && cat ${BUILD}/CIRPasses.md >> Dialect/passes.md
