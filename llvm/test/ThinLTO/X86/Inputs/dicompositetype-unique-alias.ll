target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-scei-ps4"

%struct.CFVS = type { %struct.Vec }
%struct.Vec = type { i8 }
%struct.S = type { i8 }

@_ZN4CFVSD1Ev = alias void (ptr), ptr @_ZN4CFVSD2Ev

define void @_ZN4CFVSD2Ev(ptr %this) unnamed_addr align 2 !dbg !8 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}

declare dereferenceable(1) ptr @_Z3Getv()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 (trunk 321360) (llvm/trunk 321359)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "bz188598-b.cpp", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!8 = distinct !DISubprogram(name: "~CFVS", linkageName: "_ZN4CFVSD2Ev", scope: !9, file: !1, line: 2, type: !28, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !27, retainedNodes: !2)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CFVS", file: !10, line: 7, size: 8, elements: !11, identifier: "_ZTS4CFVS")
!10 = !DIFile(filename: "./bz188598.h", directory: "")
!11 = !{!35}
!27 = !DISubprogram(name: "~CFVS", scope: !9, file: !10, line: 8, type: !28, isLocal: false, isDefinition: false, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false)
!28 = !DISubroutineType(types: !29)
!29 = !{null, !30}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!35 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
