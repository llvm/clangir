#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

#include <iostream>
#include <set>


using namespace mlir;
using namespace mlir::cir;

namespace {

//TODO: there is something bad here 
static bool has_one_user(mlir::Value val) {
  unsigned users = 0;      
  for (auto it = val.user_begin(); it != val.user_end(); ++it)  
    users++;

  if (!val.hasOneUse() && users == 1)
    std::cout << "ALARM1!!!\n";

  if (val.hasOneUse() && users != 1)
    std::cout << "ALARM2!!!\n";

  return users == 1;  
}

struct CIRSimplifyPass : public CIRSimplifyBase<CIRSimplifyPass> {

  CIRSimplifyPass() = default;
  void runOnOperation() override;
};

class SimplifyBoolCasts
    : public mlir::OpRewritePattern<mlir::cir::CastOp> {

  typedef std::vector<mlir::cir::CastOp> chain_of_casts;

  bool is_bool_aimed(mlir::cir::CastOp op) const {
    return op.getKind() == mlir::cir::CastKind::int_to_bool;
  }

  bool is_int_or_bool_cast(mlir::cir::CastOp op) const {
    auto kind = op.getKind();
    return kind == mlir::cir::CastKind::bool_to_int 
          || kind == mlir::cir::CastKind::int_to_bool
          || kind == mlir::cir::CastKind::integral;
  }

  // makes a chain of casts with the next properties:
  // - each member is a cast of one of the next kinds: integral, bool_to_int, int_to_bool
  // - each member except the last one may have only one user - the next member in the chain
  // - last member may have one or many users
  void collect_casts(chain_of_casts& casts, 
                     mlir::cir::CastOp op) const {    
    auto val = op.getResult();     
    if (is_int_or_bool_cast(op)) {
      casts.push_back(op);
      if (val.hasOneUse()) 
        if (auto next = dyn_cast<mlir::cir::CastOp>(*val.user_begin()))
          collect_casts(casts, next);        
    }
  }

public:
  using mlir::OpRewritePattern<mlir::cir::CastOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CastOp op,
                  mlir::PatternRewriter &rewriter) const final {
      // std::cout << "SimplifyBoolCasts: we are here\n";
      // return mlir::success();

      chain_of_casts casts;
      collect_casts(casts, op);

      if (casts.size() < 2 || !is_bool_aimed(casts.back()))
        return mlir::success();

      auto first = casts.front();  
      auto last = casts.back();

      mlir::Value actual;
      auto shift = 0;

      // The next two cases are covered:
      // 1) bool_to_int -> ... -> int_to_bool: remove all the casts, 
      //    bool value was already obtained before all the conversions
      // 2) int_to_bool -> ... -> int_to_bool: take the result of the first op, 
      //    erase all others
      if (first.getKind() == mlir::cir::CastKind::bool_to_int) { 
        actual = first.getSrc();
      } else if (first.getKind() == mlir::cir::CastKind::int_to_bool) {
        actual = first.getResult();
        shift++;
      } else { // unexpected case
        return mlir::success();  
      }
      std::cout << "replace\n ";
      rewriter.replaceAllUsesWith(last, actual);
      auto it = casts.begin();
      for (auto rit = casts.end() - 1; rit >= it + shift; --rit) {
        std::cout << "erase\n";
        rit->dump();
        rewriter.eraseOp(*rit);
      }

      return mlir::success();
  }
};


class SimplifyUnaryNot : public mlir::OpRewritePattern<mlir::cir::UnaryOp> {

  bool is_unary_not(mlir::cir::UnaryOp op) const {
    return op.getKind() == mlir::cir::UnaryOpKind::Not;
  } 

public:
  using mlir::OpRewritePattern<mlir::cir::UnaryOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::UnaryOp op,
                  mlir::PatternRewriter &rewriter) const final {
    if (!is_unary_not(op) || !op.getResult().hasOneUse())
      return mlir::success();
  
    auto val = op.getResult();    
    auto user = *val.user_begin();
   
    if (auto next = dyn_cast<mlir::cir::UnaryOp>(user)) {
      next->dump();
      if (!is_unary_not(next))
        return mlir::failure();               
      rewriter.replaceAllUsesWith(next, op.getInput());
      rewriter.eraseOp(next);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    return mlir::failure();
  }
};



  typedef std::vector<mlir::cir::CastOp> chain_of_casts;

  bool is_bool_aimed(mlir::cir::CastOp op)  {
    return op.getKind() == mlir::cir::CastKind::int_to_bool;
  }

  bool is_int_or_bool_cast(mlir::cir::CastOp op)  {
    auto kind = op.getKind();
    return kind == mlir::cir::CastKind::bool_to_int 
          || kind == mlir::cir::CastKind::int_to_bool
          || kind == mlir::cir::CastKind::integral;
  }

  // makes a chain of casts with the next properties:
  // - each member is a cast of one of the next kinds: integral, bool_to_int, int_to_bool
  // - each member except the last one may have only one user - the next member in the chain
  // - last member may have one or many users
  void collect_casts(chain_of_casts& casts, 
                     mlir::cir::CastOp op)  {    
    auto val = op.getResult();     
    if (is_int_or_bool_cast(op)) {
      casts.push_back(op);
      if (val.hasOneUse()) 
        if (auto next = dyn_cast<mlir::cir::CastOp>(*val.user_begin()))
          collect_casts(casts, next);        
    }
  }

bool runCastOp(mlir::cir::CastOp op, std::set<mlir::Operation*>& erased)  {
      
      chain_of_casts casts;
      collect_casts(casts, op);

      if (casts.size() < 2 || !is_bool_aimed(casts.back()))
        return false;

      auto first = casts.front();  
      auto last = casts.back();

      mlir::Value actual;
      auto shift = 0;

      // The next two cases are covered:
      // 1) bool_to_int -> ... -> int_to_bool: remove all the casts, 
      //    bool value was already obtained before all the conversions
      // 2) int_to_bool -> ... -> int_to_bool: take the result of the first op, 
      //    erase all others
      if (first.getKind() == mlir::cir::CastKind::bool_to_int) { 
        actual = first.getSrc();
      } else if (first.getKind() == mlir::cir::CastKind::int_to_bool) {
        actual = first.getResult();
        shift++;
      } else { // unexpected case
        return false;  
      }
      std::cout << "replace\n";
      last.dump();
      actual.dump();
      last.replaceAllUsesWith(actual);
      auto it = casts.begin();
      for (auto rit = casts.end() - 1; rit >= it + shift; --rit) {
        std::cout << "erase1\n";
        erased.emplace(*rit);
       // (*rit)->erase();
      }

      return true;
}

// void CIRSimplifyPass::runOnOperation() {
//   RewritePatternSet patterns(&getContext());
//   patterns.add<SimplifyUnaryNot>(patterns.getContext());

//   // Collect operations to apply patterns.
//   SmallVector<Operation *, 16> ops;
//   getOperation()->walk([&](Operation *op) {
//     if (isa<mlir::cir::UnaryOp>(op))
//       ops.push_back(op);
//   });

//   GreedyRewriteConfig config;
//   config.enableRegionSimplification = false;

//   // Apply patterns.
//   if (applyOpPatternsAndFold(ops, std::move(patterns), config).failed()) 
//     signalPassFailure();

// }




  static bool is_unary_not(mlir::cir::UnaryOp op) {
    return op.getKind() == mlir::cir::UnaryOpKind::Not;
  } 

  static bool runUnOp(mlir::cir::UnaryOp op, std::set<mlir::Operation*>& erased) {
    if (!is_unary_not(op) || !op.getResult().hasOneUse())
      return false;
    std::cout << "label1\n";
    auto val = op.getResult();    
    auto user = *val.user_begin();
    std::cout << "label2\n";
    user->dump();
    std::cout << "label3\n";

    if (auto next = dyn_cast_or_null<mlir::cir::UnaryOp>(user)) {
      std::cout << "label4\n";
  
      if (!is_unary_not(next))
        return false;
      std::cout << "replace\n";
      next.dump();
      op.getInput().dump();
      next.replaceAllUsesWith(op.getInput());
      std::cout << "erase2\n";
      // next->erase();
      // op->erase();
      erased.emplace(op);
      erased.emplace(next);
      return true;
    }
}


void CIRSimplifyPass::runOnOperation() {
  // Collect operations to apply patterns.
  SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](Operation *op) {
    if (isa<mlir::cir::UnaryOp, mlir::cir::CastOp>(op))
      ops.push_back(op);
  });

  std::set<mlir::Operation*> erased;

  bool changed = true;
  while (changed) {
    changed = false;  
    std::cout << "iter\n";
    for (auto op : ops) {
      if (erased.count(op))
        continue;
      
      if (auto uop = dyn_cast<UnaryOp>(op)) {
        changed |= runUnOp(uop, erased);
      }
      else if (auto cast = dyn_cast<CastOp>(op)) {
        changed |= runCastOp(cast, erased);
      }
    }
  }

  for (auto it = erased.rbegin(); it != erased.rend(); ++it)
    (*it)->erase();
}






} // namespace

std::unique_ptr<Pass> mlir::createCIRSimplifyPass() {
  return std::make_unique<CIRSimplifyPass>();
}