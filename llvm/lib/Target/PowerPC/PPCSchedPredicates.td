//===--- PPCSchedPredicates.td - PowerPC Scheduling Preds -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Automatically generated file, do not edit!
//
// This file defines scheduling predicate definitions that are used by the
// PowerPC subtargets.
//===----------------------------------------------------------------------===//
// Identify instructions that write BF pipelines with 7 cycles.
def P10W_BF_7C_Pred : MCSchedPredicate<
      CheckOpcode<[FADD,
                   FADDS,
                   FADDS_rec,
                   FADD_rec,
                   FCFID,
                   FCFIDS,
                   FCFIDS_rec,
                   FCFIDU,
                   FCFIDUS,
                   FCFIDUS_rec,
                   FCFIDU_rec,
                   FCFID_rec,
                   FCTID,
                   FCTIDU,
                   FCTIDUZ,
                   FCTIDUZ_rec,
                   FCTIDU_rec,
                   FCTIDZ,
                   FCTIDZ_rec,
                   FCTID_rec,
                   FCTIW,
                   FCTIWU,
                   FCTIWUZ,
                   FCTIWUZ_rec,
                   FCTIWU_rec,
                   FCTIWZ,
                   FCTIWZ_rec,
                   FCTIW_rec,
                   FMADD,
                   FMADDS,
                   FMADDS_rec,
                   FMADD_rec,
                   FMSUB,
                   FMSUBS,
                   FMSUBS_rec,
                   FMSUB_rec,
                   FMUL,
                   FMULS,
                   FMULS_rec,
                   FMUL_rec,
                   FNMADD,
                   FNMADDS,
                   FNMADDS_rec,
                   FNMADD_rec,
                   FNMSUB,
                   FNMSUBS,
                   FNMSUBS_rec,
                   FNMSUB_rec,
                   FRE,
                   FRES,
                   FRES_rec,
                   FRE_rec,
                   FRIMD, FRIMS,
                   FRIMD_rec, FRIMS_rec,
                   FRIND, FRINS,
                   FRIND_rec, FRINS_rec,
                   FRIPD, FRIPS,
                   FRIPD_rec, FRIPS_rec,
                   FRIZD, FRIZS,
                   FRIZD_rec, FRIZS_rec,
                   FRSP,
                   FRSP_rec,
                   FRSQRTE,
                   FRSQRTES,
                   FRSQRTES_rec,
                   FRSQRTE_rec,
                   FSELD, FSELS,
                   FSELD_rec, FSELS_rec,
                   FSUB,
                   FSUBS,
                   FSUBS_rec,
                   FSUB_rec,
                   VADDFP,
                   VCFSX, VCFSX_0,
                   VCFUX, VCFUX_0,
                   VCTSXS, VCTSXS_0,
                   VCTUXS, VCTUXS_0,
                   VEXPTEFP,
                   VEXPTEFP,
                   VLOGEFP,
                   VMADDFP,
                   VNMSUBFP,
                   VREFP,
                   VRFIM,
                   VRFIN,
                   VRFIP,
                   VRFIZ,
                   VRSQRTEFP,
                   VSUBFP,
                   XSADDDP,
                   XSADDSP,
                   XSCVDPHP,
                   XSCVDPSP,
                   XSCVDPSPN,
                   XSCVDPSXDS, XSCVDPSXDSs,
                   XSCVDPSXWS, XSCVDPSXWSs,
                   XSCVDPUXDS, XSCVDPUXDSs,
                   XSCVDPUXWS, XSCVDPUXWSs,
                   XSCVSPDP,
                   XSCVSXDDP,
                   XSCVSXDSP,
                   XSCVUXDDP,
                   XSCVUXDSP,
                   XSMADDADP,
                   XSMADDASP,
                   XSMADDMDP,
                   XSMADDMSP,
                   XSMSUBADP,
                   XSMSUBASP,
                   XSMSUBMDP,
                   XSMSUBMSP,
                   XSMULDP,
                   XSMULSP,
                   XSNMADDADP,
                   XSNMADDASP,
                   XSNMADDMDP,
                   XSNMADDMSP,
                   XSNMSUBADP,
                   XSNMSUBASP,
                   XSNMSUBMDP,
                   XSNMSUBMSP,
                   XSRDPI,
                   XSRDPIC,
                   XSRDPIM,
                   XSRDPIP,
                   XSRDPIZ,
                   XSREDP,
                   XSRESP,
                   XSRSP,
                   XSRSQRTEDP,
                   XSRSQRTESP,
                   XSSUBDP,
                   XSSUBSP,
                   XVADDDP,
                   XVADDSP,
                   XVCVDPSP,
                   XVCVDPSXDS,
                   XVCVDPSXWS,
                   XVCVDPUXDS,
                   XVCVDPUXWS,
                   XVCVSPBF16,
                   XVCVSPDP,
                   XVCVSPHP,
                   XVCVSPSXDS,
                   XVCVSPSXWS,
                   XVCVSPUXDS,
                   XVCVSPUXWS,
                   XVCVSXDDP,
                   XVCVSXDSP,
                   XVCVSXWDP,
                   XVCVSXWSP,
                   XVCVUXDDP,
                   XVCVUXDSP,
                   XVCVUXWDP,
                   XVCVUXWSP,
                   XVMADDADP,
                   XVMADDASP,
                   XVMADDMDP,
                   XVMADDMSP,
                   XVMSUBADP,
                   XVMSUBASP,
                   XVMSUBMDP,
                   XVMSUBMSP,
                   XVMULDP,
                   XVMULSP,
                   XVNMADDADP,
                   XVNMADDASP,
                   XVNMADDMDP,
                   XVNMADDMSP,
                   XVNMSUBADP,
                   XVNMSUBASP,
                   XVNMSUBMDP,
                   XVNMSUBMSP,
                   XVRDPI,
                   XVRDPIC,
                   XVRDPIM,
                   XVRDPIP,
                   XVRDPIZ,
                   XVREDP,
                   XVRESP,
                   XVRSPI,
                   XVRSPIC,
                   XVRSPIM,
                   XVRSPIP,
                   XVRSPIZ,
                   XVRSQRTEDP,
                   XVRSQRTESP,
                   XVSUBDP,
                   XVSUBSP]>
>;

// Identify instructions that write CY pipelines with 7 cycles.
def P10W_CY_7C_Pred : MCSchedPredicate<
      CheckOpcode<[CFUGED,
                   CNTLZDM,
                   CNTTZDM,
                   PDEPD,
                   PEXTD,
                   VCFUGED,
                   VCIPHER,
                   VCIPHERLAST,
                   VCLZDM,
                   VCTZDM,
                   VGNB,
                   VNCIPHER,
                   VNCIPHERLAST,
                   VPDEPD,
                   VPEXTD,
                   VPMSUMB,
                   VPMSUMD,
                   VPMSUMH,
                   VPMSUMW,
                   VSBOX]>
>;

// Identify instructions that write MM pipelines with 10 cycles.
def P10W_MM_10C_Pred : MCSchedPredicate<
      CheckOpcode<[PMXVBF16GER2,
                   PMXVBF16GER2NN,
                   PMXVBF16GER2NP,
                   PMXVBF16GER2PN,
                   PMXVBF16GER2PP,
                   PMXVF16GER2,
                   PMXVF16GER2NN,
                   PMXVF16GER2NP,
                   PMXVF16GER2PN,
                   PMXVF16GER2PP,
                   PMXVF32GER,
                   PMXVF32GERNN,
                   PMXVF32GERNP,
                   PMXVF32GERPN,
                   PMXVF32GERPP,
                   PMXVF64GER,
                   PMXVF64GERNN,
                   PMXVF64GERNP,
                   PMXVF64GERPN,
                   PMXVF64GERPP,
                   PMXVI16GER2,
                   PMXVI16GER2PP,
                   PMXVI16GER2S,
                   PMXVI16GER2SPP,
                   PMXVI4GER8,
                   PMXVI4GER8PP,
                   PMXVI8GER4,
                   PMXVI8GER4PP,
                   PMXVI8GER4SPP,
                   XVBF16GER2,
                   XVBF16GER2NN,
                   XVBF16GER2NP,
                   XVBF16GER2PN,
                   XVBF16GER2PP,
                   XVF16GER2,
                   XVF16GER2NN,
                   XVF16GER2NP,
                   XVF16GER2PN,
                   XVF16GER2PP,
                   XVF32GER,
                   XVF32GERNN,
                   XVF32GERNP,
                   XVF32GERPN,
                   XVF32GERPP,
                   XVF64GER,
                   XVF64GERNN,
                   XVF64GERNP,
                   XVF64GERPN,
                   XVF64GERPP,
                   XVI16GER2,
                   XVI16GER2PP,
                   XVI16GER2S,
                   XVI16GER2SPP,
                   XVI4GER8,
                   XVI4GER8PP,
                   XVI8GER4,
                   XVI8GER4PP,
                   XVI8GER4SPP,
                   XXMFACC,
                   XXMFACC,
                   XXMTACC,
                   XXSETACCZ]>
>;

