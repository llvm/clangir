target triple = "wasm32-unknown-unknown"

; Will collide: local (internal linkage) with global (external) linkage
@colliding_global1 = internal default global i32 1, align 4
; Will collide: global with local
@colliding_global2 = default global i32 1, align 4
; Will collide: local with local
@colliding_global3 = internal default global i32 1, align 4

; Will collide: local with global
define internal i32 @colliding_func1() {
entry:
  ret i32 2
}
; Will collide: global with local
define i32 @colliding_func2() {
entry:
  ret i32 2
}
; Will collide: local with local
define internal i32 @colliding_func3() {
entry:
  ret i32 2
}


define ptr @get_global1A() {
entry:
  ret ptr @colliding_global1
}
define ptr @get_global2A() {
entry:
  ret ptr @colliding_global2
}
define ptr @get_global3A() {
entry:
  ret ptr @colliding_global3
}

define ptr @get_func1A() {
entry:
  ret ptr @colliding_func1
}
define ptr @get_func2A() {
entry:
  ret ptr @colliding_func2
}
define ptr @get_func3A() {
entry:
  ret ptr @colliding_func3
}
