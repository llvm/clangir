; RUN: llc -mtriple=i686-pc-mingw32 -accel-tables=Apple -filetype=asm -O0 < %s | FileCheck %s
; RUN: llc -mtriple=i686-pc-cygwin -accel-tables=Apple -filetype=asm -O0 < %s | FileCheck %s
; RUN: llc -mtriple=i686-w64-mingw32 -accel-tables=Apple -filetype=asm -O0 < %s | FileCheck %s
; CHECK:    .section  .debug_info
; CHECK:    .section  .apple_names
; CHECK:    .section  .apple_types

; RUN: sed -e 's/"Dwarf Version"/"CodeView"/' %s \
; RUN:     | llc -mtriple=i686-pc-win32 -filetype=asm -O0 \
; RUN:     | FileCheck -check-prefix=WIN32 %s
; WIN32:    .section .debug$S,"dr"

; RUN: llc -mtriple=i686-pc-win32 -filetype=null -O0 < %s

; generated from:
; clang -g -S -emit-llvm test.c -o test.ll
; int main()
; {
; 	return 0;
; }

define i32 @main() #0 !dbg !4 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval
  ret i32 0, !dbg !10
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.4 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "test.c", directory: "C:\5CProjects")
!2 = !{}
!4 = distinct !DISubprogram(name: "main", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !0, scopeLine: 2, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "test.c", directory: "C:CProjects")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 3}
!10 = !DILocation(line: 3, scope: !4)
!11 = !{i32 1, !"Debug Info Version", i32 3}
