" Vim syntax file
" Language:   cir
" Maintainer: The CIR team
" Version:      $Revision$
" Some parts adapted from the LLVM and MLIR vim syntax file.

if version < 600
  syntax clear
elseif exists("b:current_syntax")
  finish
endif

syn case match

" Types.
"
syn keyword mlirType index
syn match mlirType /\<f\d\+\>/
syn match mlirType /\<bf\d\+\>/
" Signless integer types.
syn match mlirType /\<i\d\+\>/
" Unsigned integer types.
syn match mlirType /\<ui\d\+\>/
" Signed integer types.
syn match mlirType /\<si\d\+\>/

" Elemental types inside memref, tensor, or vector types.
syn match mlirType /x\s*\zs\(bf16|f16\|f32\|f64\|i\d\+\|ui\d\+\|si\d\+\)/

" Shaped types.
syn match mlirType /\<memref\ze\s*<.*>/
syn match mlirType /\<tensor\ze\s*<.*>/
syn match mlirType /\<vector\ze\s*<.*>/

" vector types inside memref or tensor.
syn match mlirType /x\s*\zsvector/

" Misc syntax.

syn match   mlirNumber /-\?\<\d\+\>/
" Match numbers even in shaped types.
syn match   mlirNumber /-\?\<\d\+\ze\s*x/
syn match   mlirNumber /x\s*\zs-\?\d\+\ze\s*x/

syn match   mlirFloat  /-\?\<\d\+\.\d*\(e[+-]\d\+\)\?\>/
syn match   mlirFloat  /\<0x\x\+\>/
syn keyword mlirBoolean true false
" Spell checking is enabled only in comments by default.
syn match   mlirComment /\/\/.*$/ contains=@Spell
syn region  mlirString start=/"/ skip=/\\"/ end=/"/
syn match   mlirLabel /[-a-zA-Z$._][-a-zA-Z$._0-9]*:/
" Prefixed identifiers usually used for ssa values and symbols.
syn match   mlirIdentifier /[%@][a-zA-Z$._-][a-zA-Z0-9$._-]*/
syn match   mlirIdentifier /[%@]\d\+\>/
" Prefixed identifiers usually used for blocks.
syn match   mlirBlockIdentifier /\^[a-zA-Z$._-][a-zA-Z0-9$._-]*/
syn match   mlirBlockIdentifier /\^\d\+\>/
" Prefixed identifiers usually used for types.
syn match   mlirTypeIdentifier /![a-zA-Z$._-][a-zA-Z0-9$._-]*/
syn match   mlirTypeIdentifier /!\d\+\>/
" Prefixed identifiers usually used for attribute aliases and result numbers.
syn match   mlirAttrIdentifier /#[a-zA-Z$._-][a-zA-Z0-9$._-]*/
syn match   mlirAttrIdentifier /#\d\+\>/

" Syntax-highlight lit test commands and bug numbers.
" Old highlighting version of MLIR is faulty, any misamount of whitespace before or after on a line
" will not highlight the special comments anymore.
syn match mlirSpecialComment /^\s*\/\/\s*RUN:.*$/
syn match mlirSpecialComment /^\s*\/\/\s*\(CHECK\|CIR\|MLIR\|LLVM\|OGCG\):.*$/
syn match mlirSpecialComment /^\s*\/\/\s*\(CHECK\|CIR\|MLIR\|LLVM\|OGCG\)-\(NEXT\|NOT\|DAG\|SAME\|LABEL\):.*$/
syn match mlirSpecialComment /^\s*\/\/\s*expected-error.*$/
syn match mlirSpecialComment /^\s*\/\/\s*expected-remark.*$/
syn match mlirSpecialComment /^\s*;\s*XFAIL:.*$/
syn match mlirSpecialComment /^\s*\/\/\s*PR\d*\s*$/
syn match mlirSpecialComment /^\s*\/\/\s*REQUIRES:.*$/

"""""""""""" CIR SECTION """"""""""""
" CIR blanket operations
syn match cirOps /\vcir(\.\w+)+/

syn keyword cirKeyword
      \ module attributes
      \ from to cond body step
      \ align alignment init
      \ cond exception
      \ null
      \ coroutine suspend resume cleanup
      \ class union struct
      \ do while
      \ if then else
      \ nsw nuw
      \ constant
      \ equal default anyof
      \ internal external available_externally private linkonce linkonce_odr 
      \ weak weak_odr extern_weak common cir_private


syn keyword cirType bytes


if version >= 508 || !exists("did_c_syn_inits")
  if version < 508
    let did_c_syn_inits = 1
    command -nargs=+ HiLink hi link <args>
  else
    command -nargs=+ HiLink hi def link <args>
  endif

  HiLink mlirType Type
  HiLink mlirNumber Number
  HiLink mlirComment Comment
  HiLink mlirString String
  HiLink mlirLabel Label
  HiLink mlirBoolean Boolean
  HiLink mlirFloat Float
  HiLink mlirSpecialComment SpecialComment
  HiLink mlirIdentifier Identifier
  HiLink mlirBlockIdentifier Label
  HiLink mlirTypeIdentifier Type
  HiLink mlirAttrIdentifier PreProc

  HiLink cirType Type
  HiLink cirOps Statement
  HiLink cirKeyword Keyword
  delcommand HiLink
endif

let b:current_syntax = "cir"
