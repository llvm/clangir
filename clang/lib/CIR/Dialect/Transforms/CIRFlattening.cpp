#include "clang/CIR/Dialect/Passes.h"
#include "mlir/Pass/PassManager.h"


void mlir::populateCIRFlatteningPasses(mlir::OpPassManager &pm) {
    pm.addPass(mlir::createStructuredCFGPass());
}