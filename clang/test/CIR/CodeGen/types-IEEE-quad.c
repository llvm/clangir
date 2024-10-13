// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CHECK %s

long double i = 0;
long double t2(long double i2) {
    return i2 + i ; 
}

// CHECK: cir.global external @i = #cir.fp<0.000000e+00> : !cir.long_double<!cir.f128> {alignment = 16 : i64} loc({{.*}})
// CHECK: cir.func @t2(%arg0: !cir.long_double<!cir.f128> loc({{.*}})) -> !cir.long_double<!cir.f128>
// CHECK: %{{[0-9]+}} = cir.alloca !cir.long_double<!cir.f128>, !cir.ptr<!cir.long_double<!cir.f128>>, ["i2", init] {alignment = 16 : i64}
// CHECK: %{{[0-9]+}} = cir.alloca !cir.long_double<!cir.f128>, !cir.ptr<!cir.long_double<!cir.f128>>, ["__retval"] {alignment = 16 : i64}    
// CHECK: cir.store %arg0, %{{[0-9]+}} : !cir.long_double<!cir.f128>, !cir.ptr<!cir.long_double<!cir.f128>>
// CHECK: %{{[0-9]+}} = cir.load %{{[0-9]+}} : !cir.ptr<!cir.long_double<!cir.f128>>, !cir.long_double<!cir.f128>
// CHECK: %{{[0-9]+}} = cir.get_global @i : !cir.ptr<!cir.long_double<!cir.f128>>
// CHECK: %{{[0-9]+}} = cir.load %{{[0-9]+}} : !cir.ptr<!cir.long_double<!cir.f128>>, !cir.long_double<!cir.f128>
// CHECK: cir.store %{{[0-9]+}}, %{{[0-9]+}} : !cir.long_double<!cir.f128>, !cir.ptr<!cir.long_double<!cir.f128>>
// CHECK: %{{[0-9]+}} = cir.load %{{[0-9]+}} : !cir.ptr<!cir.long_double<!cir.f128>>, !cir.long_double<!cir.f128>
// CHECK: cir.return %{{[0-9]+}} : !cir.long_double<!cir.f128>

