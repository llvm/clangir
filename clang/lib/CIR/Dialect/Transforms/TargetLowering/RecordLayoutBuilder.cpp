//=== RecordLayoutBuilder.cpp - Helper class for building record layouts ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/AST/CGRecordLayoutBuilder.cpp. The
// queries are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//
