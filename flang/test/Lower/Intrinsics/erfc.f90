! RUN: bbc -emit-fir %s -o - --math-runtime=fast | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %s -o - | FileCheck --check-prefixes=ALL %s
! RUN: bbc -emit-fir %s -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %s -o - | FileCheck --check-prefixes=ALL %s
! RUN: bbc -emit-fir %s -o - --math-runtime=precise | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %s -o - | FileCheck --check-prefixes=ALL %s

function test_real4(x)
  real :: x, test_real4
  test_real4 = erfc(x)
end function

! ALL-LABEL: @_QPtest_real4
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @erfcf({{%[A-Za-z0-9._]+}}) {{.*}}: (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = erfc(x)
end function

! ALL-LABEL: @_QPtest_real8
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @erfc({{%[A-Za-z0-9._]+}}) {{.*}}: (f64) -> f64
