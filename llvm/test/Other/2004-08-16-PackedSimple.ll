; RUN: llvm-as < %s | llvm-dis

@foo = external global <4 x float>              ; <ptr> [#uses=1]
@bar = external global <4 x float>              ; <ptr> [#uses=1]

define void @main() {
        %t0 = load <4 x float>, ptr @foo            ; <<4 x float>> [#uses=3]
        %t2 = fadd <4 x float> %t0, %t0          ; <<4 x float>> [#uses=1]
        %t3 = select i1 false, <4 x float> %t0, <4 x float> %t2         ; <<4 x float>> [#uses=1]
        store <4 x float> %t3, ptr @bar
        ret void
}

