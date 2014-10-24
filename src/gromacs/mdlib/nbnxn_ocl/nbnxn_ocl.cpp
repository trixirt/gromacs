/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
#include "config.h"

#include <stdlib.h>
#include <assert.h>

#if defined(_MSVC)
#include <limits>
#endif

//#include <cuda.h>

#include "types/simple.h"
#include "gromacs/mdlib/nbnxn_pairlist.h"
#include "gromacs/mdlib/nb_verlet.h"
#include "types/force_flags.h"
#include "../nbnxn_consts.h"
#include "types/hw_info.h"

#ifdef TMPI_ATOMICS
#include "thread_mpi/atomic.h"
#endif

//#include "nbnxn_cuda_types.h"

#include <CL/opencl.h>

#include "nbnxn_ocl_types.h"
//#include "../../gmxlib/cuda_tools/cudautils.cuh"
//#include "nbnxn_cuda.h"
//#include "../nbnxn_cuda/nbnxn_cuda.h"

#include "nbnxn_ocl.h"
//#include "gromacs/mdlib/nbnxn_cuda/nbnxn_cuda_data_mgmt.h"
#include "gromacs/mdlib/nbnxn_ocl/nbnxn_ocl_data_mgmt.h"

#include "gromacs/pbcutil/ishift.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"

#if defined TEXOBJ_SUPPORTED && __CUDA_ARCH__ >= 300
#define USE_TEXOBJ
#endif

/////*! Texture reference for LJ C6/C12 parameters; bound to cu_nbparam_t.nbfp */
////texture<float, 1, cudaReadModeElementType> nbfp_texref;
////
/////*! Texture reference for LJ-PME parameters; bound to cu_nbparam_t.nbfp_comb */
////texture<float, 1, cudaReadModeElementType> nbfp_comb_texref;
////
/////*! Texture reference for Ewald coulomb force table; bound to cu_nbparam_t.coulomb_tab */
////texture<float, 1, cudaReadModeElementType> coulomb_tab_texref;

/* Convenience defines */
#define NCL_PER_SUPERCL         (NBNXN_GPU_NCLUSTER_PER_SUPERCLUSTER)
#define CL_SIZE                 (NBNXN_GPU_CLUSTER_SIZE)

/***** The kernels come here *****/
//#include "nbnxn_cuda_kernel_utils.cuh"

/* Top-level kernel generation: will generate through multiple inclusion the
 * following flavors for all kernels:
 * - force-only output;
 * - force and energy output;
 * - force-only with pair list pruning;
 * - force and energy output with pair list pruning.
 */
/** Force only **/
//#include "nbnxn_cuda_kernels.cuh"
//#include "nbnxn_ocl_kernels.clh"

/** Force & energy **/
#define CALC_ENERGIES
//#include "nbnxn_cuda_kernels.cuh"
//#include "nbnxn_ocl_kernels.clh"
#undef CALC_ENERGIES

/*** Pair-list pruning kernels ***/
/** Force only **/
#define PRUNE_NBL
//#include "nbnxn_cuda_kernels.cuh"
//#include "nbnxn_ocl_kernels.clh"
/** Force & energy **/
#define CALC_ENERGIES
//#include "nbnxn_cuda_kernels.cuh"
//#include "nbnxn_ocl_kernels.clh"
#undef CALC_ENERGIES
#undef PRUNE_NBL

/////*! Nonbonded kernel function pointer type */
////typedef void (*nbnxn_cu_kfunc_ptr_t)(const cu_atomdata_t,
////                                     const cu_nbparam_t,
////                                     const cu_plist_t,
////                                     bool);

/*********************************/

/* XXX always/never run the energy/pruning kernels -- only for benchmarking purposes */
static bool always_ener  = (getenv("GMX_GPU_ALWAYS_ENER") != NULL);
static bool never_ener   = (getenv("GMX_GPU_NEVER_ENER") != NULL);
static bool always_prune = (getenv("GMX_GPU_ALWAYS_PRUNE") != NULL);

/* Uncomment this define to enable kernel debugging */
//#define DEBUG_OCL

/* Specifies which kernel run to debug */
#define DEBUG_RUN_STEP 2

/* Bit-pattern used for polling-based GPU synchronization. It is used as a float
 * and corresponds to having the exponent set to the maximum (127 -- single
 * precision) and the mantissa to 0.
 */
static unsigned int poll_wait_pattern = (0x7FU << 23);

/////*! Returns the number of blocks to be used for the nonbonded GPU kernel. */
////static inline int calc_nb_kernel_nblock(int nwork_units, cuda_dev_info_t *dinfo)
////{
////    int max_grid_x_size;
////
////    assert(dinfo);
////
////    max_grid_x_size = dinfo->prop.maxGridSize[0];
////
////    /* do we exceed the grid x dimension limit? */
////    if (nwork_units > max_grid_x_size)
////    {
////        gmx_fatal(FARGS, "Watch out, the input system is too large to simulate!\n"
////                  "The number of nonbonded work units (=number of super-clusters) exceeds the"
////                  "maximum grid size in x dimension (%d > %d)!", nwork_units, max_grid_x_size);
////    }
////
////    return nwork_units;
////}


/* Constant arrays listing all kernel function names. */

/*! Pointers to the non-bonded kernel names organized in 2-dim arrays by:
 *  electrostatics and VDW type.
 *
 *  Note that the row- and column-order of function pointers has to match the
 *  order of corresponding enumerated electrostatics and vdw types, resp.,
 *  defined in nbnxn_cuda_types.h.
 */

/*! Force-only kernel function names. */
static const char* nb_kfunc_noener_noprune_ptr[eelOclNR][evdwOclNR] =
{
    { "nbnxn_kernel_ElecCut_VdwLJ_F_opencl",            "nbnxn_kernel_ElecCut_VdwLJFsw_F_opencl",            "nbnxn_kernel_ElecCut_VdwLJPsw_F_opencl",            "nbnxn_kernel_ElecCut_VdwLJEwCombGeom_F_opencl",            "nbnxn_kernel_ElecCut_VdwLJEwCombLB_F_opencl"            },
    { "nbnxn_kernel_ElecRF_VdwLJ_F_opencl",             "nbnxn_kernel_ElecRF_VdwLJFsw_F_opencl",             "nbnxn_kernel_ElecRF_VdwLJPsw_F_opencl",             "nbnxn_kernel_ElecRF_VdwLJEwCombGeom_F_opencl",             "nbnxn_kernel_ElecRF_VdwLJEwCombLB_F_opencl"             },
    { "nbnxn_kernel_ElecEwQSTab_VdwLJ_F_opencl",        "nbnxn_kernel_ElecEwQSTab_VdwLJFsw_F_opencl",        "nbnxn_kernel_ElecEwQSTab_VdwLJPsw_F_opencl",        "nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_F_opencl",        "nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_F_opencl"        },
    { "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_F_opencl", "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_F_opencl", "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_F_opencl", "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_F_opencl", "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_F_opencl" },
    { "nbnxn_kernel_ElecEw_VdwLJ_F_opencl",             "nbnxn_kernel_ElecEw_VdwLJFsw_F_opencl",             "nbnxn_kernel_ElecEw_VdwLJPsw_F_opencl",             "nbnxn_kernel_ElecEw_VdwLJEwCombGeom_F_opencl",             "nbnxn_kernel_ElecEw_VdwLJEwCombLB_F_opencl"             },
    { "nbnxn_kernel_ElecEwTwinCut_VdwLJ_F_opencl",      "nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_F_opencl",      "nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_F_opencl",      "nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_F_opencl",      "nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_F_opencl"      }
};

/*! Force + energy kernel function pointers. */
static const char* nb_kfunc_ener_noprune_ptr[eelOclNR][evdwOclNR] =
{
    { "nbnxn_kernel_ElecCut_VdwLJ_VF_opencl",            "nbnxn_kernel_ElecCut_VdwLJFsw_VF_opencl",            "nbnxn_kernel_ElecCut_VdwLJPsw_VF_opencl",            "nbnxn_kernel_ElecCut_VdwLJEwCombGeom_VF_opencl",            "nbnxn_kernel_ElecCut_VdwLJEwCombLB_VF_opencl"              },
    { "nbnxn_kernel_ElecRF_VdwLJ_VF_opencl",             "nbnxn_kernel_ElecRF_VdwLJFsw_VF_opencl",             "nbnxn_kernel_ElecRF_VdwLJPsw_VF_opencl",             "nbnxn_kernel_ElecRF_VdwLJEwCombGeom_VF_opencl",             "nbnxn_kernel_ElecRF_VdwLJEwCombLB_VF_opencl"               },
    { "nbnxn_kernel_ElecEwQSTab_VdwLJ_VF_opencl",        "nbnxn_kernel_ElecEwQSTab_VdwLJFsw_VF_opencl",        "nbnxn_kernel_ElecEwQSTab_VdwLJPsw_VF_opencl",        "nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_VF_opencl",        "nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_VF_opencl"          },
    { "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_VF_opencl", "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_VF_opencl", "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_VF_opencl", "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_VF_opencl", "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_VF_opencl"     },
    { "nbnxn_kernel_ElecEw_VdwLJ_VF_opencl",             "nbnxn_kernel_ElecEw_VdwLJFsw_VF_opencl",             "nbnxn_kernel_ElecEw_VdwLJPsw_VF_opencl",             "nbnxn_kernel_ElecEw_VdwLJEwCombGeom_VF_opencl",             "nbnxn_kernel_ElecEw_VdwLJEwCombLB_VF_opencl"               },
    { "nbnxn_kernel_ElecEwTwinCut_VdwLJ_VF_opencl",      "nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_VF_opencl",      "nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_VF_opencl",      "nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_VF_opencl",      "nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_VF_opencl"        }
};

/*! Force + pruning kernel function pointers. */
static const char* nb_kfunc_noener_prune_ptr[eelOclNR][evdwOclNR] =
{
    { "nbnxn_kernel_ElecCut_VdwLJ_F_prune_opencl",             "nbnxn_kernel_ElecCut_VdwLJFsw_F_prune_opencl",            "nbnxn_kernel_ElecCut_VdwLJPsw_F_prune_opencl",            "nbnxn_kernel_ElecCut_VdwLJEwCombGeom_F_prune_opencl",            "nbnxn_kernel_ElecCut_VdwLJEwCombLB_F_prune_opencl"            },
    { "nbnxn_kernel_ElecRF_VdwLJ_F_prune_opencl",              "nbnxn_kernel_ElecRF_VdwLJFsw_F_prune_opencl",             "nbnxn_kernel_ElecRF_VdwLJPsw_F_prune_opencl",             "nbnxn_kernel_ElecRF_VdwLJEwCombGeom_F_prune_opencl",             "nbnxn_kernel_ElecRF_VdwLJEwCombLB_F_prune_opencl"             },
    { "nbnxn_kernel_ElecEwQSTab_VdwLJ_F_prune_opencl",         "nbnxn_kernel_ElecEwQSTab_VdwLJFsw_F_prune_opencl",        "nbnxn_kernel_ElecEwQSTab_VdwLJPsw_F_prune_opencl",        "nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_F_prune_opencl",        "nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_F_prune_opencl"        },
    { "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_F_prune_opencl",  "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_F_prune_opencl", "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_F_prune_opencl", "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_F_prune_opencl", "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_F_prune_opencl" },
    { "nbnxn_kernel_ElecEw_VdwLJ_F_prune_opencl",              "nbnxn_kernel_ElecEw_VdwLJFsw_F_prune_opencl",             "nbnxn_kernel_ElecEw_VdwLJPsw_F_prune_opencl",             "nbnxn_kernel_ElecEw_VdwLJEwCombGeom_F_prune_opencl",             "nbnxn_kernel_ElecEw_VdwLJEwCombLB_F_prune_opencl"             },
    { "nbnxn_kernel_ElecEwTwinCut_VdwLJ_F_prune_opencl",       "nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_F_prune_opencl",      "nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_F_prune_opencl",      "nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_F_prune_opencl",      "nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_F_prune_opencl"      }
};

/*! Force + energy + pruning kernel function pointers. */
static const char* nb_kfunc_ener_prune_ptr[eelOclNR][evdwOclNR] =
{
    { "nbnxn_kernel_ElecCut_VdwLJ_VF_prune_opencl",            "nbnxn_kernel_ElecCut_VdwLJFsw_VF_prune_opencl",            "nbnxn_kernel_ElecCut_VdwLJPsw_VF_prune_opencl",            "nbnxn_kernel_ElecCut_VdwLJEwCombGeom_VF_prune_opencl",            "nbnxn_kernel_ElecCut_VdwLJEwCombLB_VF_prune_opencl"            },
    { "nbnxn_kernel_ElecRF_VdwLJ_VF_prune_opencl",             "nbnxn_kernel_ElecRF_VdwLJFsw_VF_prune_opencl",             "nbnxn_kernel_ElecRF_VdwLJPsw_VF_prune_opencl",             "nbnxn_kernel_ElecRF_VdwLJEwCombGeom_VF_prune_opencl",             "nbnxn_kernel_ElecRF_VdwLJEwCombLB_VF_prune_opencl"             },
    { "nbnxn_kernel_ElecEwQSTab_VdwLJ_VF_prune_opencl",        "nbnxn_kernel_ElecEwQSTab_VdwLJFsw_VF_prune_opencl",        "nbnxn_kernel_ElecEwQSTab_VdwLJPsw_VF_prune_opencl",        "nbnxn_kernel_ElecEwQSTab_VdwLJEwCombGeom_VF_prune_opencl",        "nbnxn_kernel_ElecEwQSTab_VdwLJEwCombLB_VF_prune_opencl"        },
    { "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJ_VF_prune_opencl", "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJFsw_VF_prune_opencl", "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJPsw_VF_prune_opencl", "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombGeom_VF_prune_opencl", "nbnxn_kernel_ElecEwQSTabTwinCut_VdwLJEwCombLB_VF_prune_opencl" },
    { "nbnxn_kernel_ElecEw_VdwLJ_VF_prune_opencl",             "nbnxn_kernel_ElecEw_VdwLJFsw_VF_prune_opencl",             "nbnxn_kernel_ElecEw_VdwLJPsw_VF_prune_opencl",             "nbnxn_kernel_ElecEw_VdwLJEwCombGeom_VF_prune_opencl",             "nbnxn_kernel_ElecEw_VdwLJEwCombLB_VF_prune_opencl"             },
    { "nbnxn_kernel_ElecEwTwinCut_VdwLJ_VF_prune_opencl",      "nbnxn_kernel_ElecEwTwinCut_VdwLJFsw_VF_prune_opencl",      "nbnxn_kernel_ElecEwTwinCut_VdwLJPsw_VF_prune_opencl",      "nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombGeom_VF_prune_opencl",      "nbnxn_kernel_ElecEwTwinCut_VdwLJEwCombLB_VF_prune_opencl"      }
};

/////*! Return a pointer to the kernel version to be executed at the current step. */
////static inline nbnxn_cu_kfunc_ptr_t select_nbnxn_kernel(int  eeltype,
////                                                       int  evdwtype,
////                                                       bool bDoEne,
////                                                       bool bDoPrune)
////{
////    nbnxn_cu_kfunc_ptr_t res;
////
////    assert(eeltype < eelCuNR);
////    assert(evdwtype < eelCuNR);
////
////    if (bDoEne)
////    {
////        if (bDoPrune)
////        {
////            res = nb_kfunc_ener_prune_ptr[eeltype][evdwtype];
////        }
////        else
////        {
////            res = nb_kfunc_ener_noprune_ptr[eeltype][evdwtype];
////        }
////    }
////    else
////    {
////        if (bDoPrune)
////        {
////            res = nb_kfunc_noener_prune_ptr[eeltype][evdwtype];
////        }
////        else
////        {
////            res = nb_kfunc_noener_noprune_ptr[eeltype][evdwtype];
////        }
////    }
////
////    return res;
////}
////
/////*! Calculates the amount of shared memory required by the CUDA kernel in use. */
////static inline int calc_shmem_required()
////{
////    int shmem;
////
////    /* size of shmem (force-buffers/xq/atom type preloading) */
////    /* NOTE: with the default kernel on sm3.0 we need shmem only for pre-loading */
////    /* i-atom x+q in shared memory */
////    shmem  = NCL_PER_SUPERCL * CL_SIZE * sizeof(float4);
////    /* cj in shared memory, for both warps separately */
////    shmem += 2 * NBNXN_GPU_JGROUP_SIZE * sizeof(int);
////#ifdef IATYPE_SHMEM
////    /* i-atom types in shared memory */
////    shmem += NCL_PER_SUPERCL * CL_SIZE * sizeof(int);
////#endif
////#if __CUDA_ARCH__ < 300
////    /* force reduction buffers in shared memory */
////    shmem += CL_SIZE * CL_SIZE * 3 * sizeof(float);
////#endif
////
////    return shmem;
////}

/*! Return a pointer to the kernel version to be executed at the current step. */
static inline cl_kernel select_nbnxn_kernel(nbnxn_opencl_ptr_t ocl_nb,
                                            int  eeltype,
                                                       int  evdwtype,
                                                       bool bDoEne,
                                                       bool bDoPrune)
{
    const char* kernel_name_to_run;
    cl_kernel *kernel_ptr;
    cl_int cl_error;

    assert(eeltype < eelOclNR);
    assert(evdwtype < eelOclNR);

    if (bDoEne)
    {
        if (bDoPrune)
        {
            kernel_name_to_run = nb_kfunc_ener_prune_ptr[eeltype][evdwtype];
            kernel_ptr = &(ocl_nb->kernel_ener_prune_ptr[eeltype][evdwtype]);
        }
        else
        {
            kernel_name_to_run = nb_kfunc_ener_noprune_ptr[eeltype][evdwtype];
            kernel_ptr = &(ocl_nb->kernel_ener_noprune_ptr[eeltype][evdwtype]);
        }
    }
    else
    {
        if (bDoPrune)
        {
            kernel_name_to_run = nb_kfunc_noener_prune_ptr[eeltype][evdwtype];
            kernel_ptr = &(ocl_nb->kernel_noener_prune_ptr[eeltype][evdwtype]);
        }
        else
        {
            kernel_name_to_run = nb_kfunc_noener_noprune_ptr[eeltype][evdwtype];
            kernel_ptr = &(ocl_nb->kernel_noener_noprune_ptr[eeltype][evdwtype]);
        }
    }
#ifndef NDEBUG
    printf("Selected kernel: %s\n",kernel_name_to_run);
#endif

    if (NULL == kernel_ptr[0])
    {
        *kernel_ptr = clCreateKernel(ocl_nb->dev_info->program, kernel_name_to_run, &cl_error);
        assert(cl_error == CL_SUCCESS);
    }
    // TO DO: handle errors

    return *kernel_ptr;
}

/*! Returns the number of blocks to be used for the nonbonded GPU kernel. */
static inline int calc_nb_kernel_nblock(int nwork_units, ocl_gpu_info_t *dinfo)
{
    int max_grid_x_size;

    assert(dinfo);

    // TO DO: fix for OpenCL implementation
    //max_grid_x_size = dinfo->prop.maxGridSize[0];

    ///* do we exceed the grid x dimension limit? */
    //if (nwork_units > max_grid_x_size)
    //{
    //    gmx_fatal(FARGS, "Watch out, the input system is too large to simulate!\n"
    //              "The number of nonbonded work units (=number of super-clusters) exceeds the"
    //              "maximum grid size in x dimension (%d > %d)!", nwork_units, max_grid_x_size);
    //}

    return nwork_units;
}

/*! Calculates the amount of shared memory required by the CUDA kernel in use. */
static inline int calc_shmem_required()
{
    int shmem;

    /* size of shmem (force-buffers/xq/atom type preloading) */
    /* NOTE: with the default kernel on sm3.0 we need shmem only for pre-loading */
    /* i-atom x+q in shared memory */
    //shmem  = NCL_PER_SUPERCL * CL_SIZE * sizeof(float4);
    shmem  = NCL_PER_SUPERCL * CL_SIZE * sizeof(float) * 4; /* xqib */
    /* cj in shared memory, for both warps separately */
    shmem += 2 * NBNXN_GPU_JGROUP_SIZE * sizeof(int);       /* cjs  */
#ifdef IATYPE_SHMEM // CUDA ARCH >= 300
    /* i-atom types in shared memory */
    #pragma error "Should not be defined"
    shmem += NCL_PER_SUPERCL * CL_SIZE * sizeof(int);       /* atib */
#endif
/* #if __CUDA_ARCH__ < 300 */
    /* force reduction buffers in shared memory */
    shmem += CL_SIZE * CL_SIZE * 3 * sizeof(float);         /* f_buf */
/* #endif */
    /* Warp vote. In fact it must be * number of warps in block.. */
    shmem += sizeof(cl_uint) * 2; /* warp_any */
    return shmem;
}


static void fillin_ocl_structures(cl_atomdata_t *adat, cl_nbparam_t *nbp, cl_plist_t *plist,
                                  cl_atomdata_params_t *atomdata_params, cl_nbparam_params_t *nbparams_params)
{
    atomdata_params->natoms = adat->natoms;
    atomdata_params->natoms_local = adat->natoms_local;
    atomdata_params->ntypes = adat->ntypes;
    atomdata_params->nalloc = adat->nalloc;
    atomdata_params->bShiftVecUploaded = adat->bShiftVecUploaded;

    nbparams_params->coulomb_tab_scale = nbp->coulomb_tab_scale;
    nbparams_params->coulomb_tab_size = nbp->coulomb_tab_size;
    nbparams_params->c_rf = nbp->c_rf;
    nbparams_params->dispersion_shift = nbp->dispersion_shift;
    nbparams_params->eeltype = nbp->eeltype;
    nbparams_params->epsfac = nbp->epsfac;
    nbparams_params->ewaldcoeff_lj = nbp->ewaldcoeff_lj;
    nbparams_params->ewald_beta = nbp->ewald_beta;
    nbparams_params->rcoulomb_sq = nbp->rcoulomb_sq;
    nbparams_params->repulsion_shift = nbp->repulsion_shift;
    nbparams_params->rlist_sq = nbp->rlist_sq;
    nbparams_params->rvdw_sq = nbp->rvdw_sq;
    nbparams_params->rvdw_switch = nbp->rvdw_switch;
    nbparams_params->sh_ewald = nbp->sh_ewald;
    nbparams_params->sh_lj_ewald = nbp->sh_lj_ewald;
    nbparams_params->two_k_rf = nbp->two_k_rf;
    nbparams_params->vdwtype = nbp->vdwtype;
    nbparams_params->vdw_switch = nbp->vdw_switch;

}

/* Waits for the commands associated with the input event to finish.
 * Then it releases the event and sets it to 0.
 * Don't use this function when more than one wait will be issued for the event.
 */
void wait_ocl_event(cl_event *ocl_event)
{
    cl_int cl_error;

    /* Blocking wait for the event */
    cl_error = clWaitForEvents(1, ocl_event);
    assert(CL_SUCCESS == cl_error);

    /* Release event and reset it to 0 */
    cl_error = clReleaseEvent(*ocl_event);
    assert(CL_SUCCESS == cl_error);
    *ocl_event = 0;
}

/* Returns the duration in miliseconds for the command associated with the event.
 * It then releases the event and sets it to 0.
 * The function returns 0.0 if the input event, *ocl_event, is 0.
 * Don't use this function when more than one wait will be issued for the event.
 */
double ocl_event_elapsed_ms(cl_event *ocl_event)
{
    cl_int cl_error;
    cl_ulong start_ns, end_ns;
    double elapsed_ms;

    elapsed_ms = 0.0;
    assert(NULL != ocl_event);    

    if (*ocl_event)
    {
        cl_error = clGetEventProfilingInfo(*ocl_event, CL_PROFILING_COMMAND_START,
            sizeof(cl_ulong), &start_ns, NULL);
        assert(CL_SUCCESS == cl_error);

        cl_error = clGetEventProfilingInfo(*ocl_event, CL_PROFILING_COMMAND_END,
            sizeof(cl_ulong), &end_ns, NULL);
        assert(CL_SUCCESS == cl_error);

        clReleaseEvent(*ocl_event);
        *ocl_event = 0;

        elapsed_ms = (end_ns - start_ns) / 1000000.0;
    }

    return elapsed_ms;
}

/*! As we execute nonbonded workload in separate streams, before launching
   the kernel we need to make sure that he following operations have completed:
   - atomdata allocation and related H2D transfers (every nstlist step);
   - pair list H2D transfer (every nstlist step);
   - shift vector H2D transfer (every nstlist step);
   - force (+shift force and energy) output clearing (every step).

   These operations are issued in the local stream at the beginning of the step
   and therefore always complete before the local kernel launch. The non-local
   kernel is launched after the local on the same device/context, so this is
   inherently scheduled after the operations in the local stream (including the
   above "misc_ops").
   However, for the sake of having a future-proof implementation, we use the
   misc_ops_done event to record the point in time when the above  operations
   are finished and synchronize with this event in the non-local stream.
 */
void nbnxn_ocl_launch_kernel(nbnxn_opencl_ptr_t        ocl_nb,
                              const nbnxn_atomdata_t *nbatom,
                              int                     flags,
                              int                     iloc)
{
    cl_int cl_error;
    //cudaError_t          stat;
    int                  adat_begin, adat_len; /* local/nonlocal offset and length used for xq and f */
    /* CUDA kernel launch-related stuff */
    int                  shmem, nblock;
    //dim3                 dim_block, dim_grid;
    size_t                 dim_block[3], dim_grid[3];
    //nbnxn_cu_kfunc_ptr_t nb_kernel = NULL; /* fn pointer to the nonbonded kernel */
    cl_kernel nb_kernel = NULL; /* fn pointer to the nonbonded kernel */

    cl_atomdata_t       *adat    = ocl_nb->atdat;
    cl_nbparam_t        *nbp     = ocl_nb->nbparam;
    cl_plist_t          *plist   = ocl_nb->plist[iloc];
    cl_timers_t         *t       = ocl_nb->timers;
    cl_command_queue     stream  = ocl_nb->stream[iloc];

    bool                 bCalcEner   = flags & GMX_FORCE_VIRIAL;
    bool                 bCalcFshift = flags & GMX_FORCE_VIRIAL;
    bool                 bDoTime     = ocl_nb->bDoTime;
    cl_uint                  arg_no;

    cl_atomdata_params_t atomdata_params;
    cl_nbparam_params_t nbparams_params;
    cl_plist_params_t plist_params;
#ifdef DEBUG_OCL
        float* debug_buffer_h;
        size_t debug_buffer_size;
#endif

    /* turn energy calculation always on/off (for debugging/testing only) */
    bCalcEner = (bCalcEner || always_ener) && !never_ener;

    /* don't launch the kernel if there is no work to do */
    if (plist->nsci == 0)
    {
        return;
    }

    /* calculate the atom data index range based on locality */
    if (LOCAL_I(iloc))
    {
        adat_begin  = 0;
        adat_len    = adat->natoms_local;
    }
    else
    {
        adat_begin  = adat->natoms_local;
        adat_len    = adat->natoms - adat->natoms_local;
    }

    /* When we get here all misc operations issues in the local stream are done,
       so we record that in the local stream and wait for it in the nonlocal one. */
    if (ocl_nb->bUseTwoStreams)
    {
        if (iloc == eintLocal)
        {
            cl_error = clEnqueueMarker(stream, &(ocl_nb->misc_ops_done));
            assert(CL_SUCCESS == cl_error);
        }
        else
        {
            wait_ocl_event(&(ocl_nb->misc_ops_done));
        }
    }

    /* beginning of timed HtoD section */

    /* HtoD x, q */
    ocl_copy_H2D_async(adat->xq, nbatom->x + adat_begin * 4, adat_begin,
        adat_len * sizeof(float) * 4, stream, bDoTime ? (&(t->nb_h2d[iloc])) : NULL);

    /* beginning of timed nonbonded calculation section */

    /* get the pointer to the kernel flavor we need to use */
    nb_kernel = select_nbnxn_kernel(ocl_nb,
                                    nbp->eeltype,
                                    nbp->vdwtype,
                                    bCalcEner,
                                    plist->bDoPrune || always_prune);

    /* kernel launch config */
    nblock    = calc_nb_kernel_nblock(plist->nsci, ocl_nb->dev_info);
    //dim_block = dim3(CL_SIZE, CL_SIZE, 1);
    dim_block[0] = CL_SIZE;
    dim_block[1] = CL_SIZE;
    dim_block[2] = 1;

    //dim_grid  = dim3(nblock, 1, 1);
    dim_grid[0] = nblock * dim_block[0];
    dim_grid[1] = 1 * dim_block[1];
    dim_grid[2]= 1 * dim_block[2];

    shmem     = calc_shmem_required();

#ifdef DEBUG_OCL
    {
        static int run_step = 1;

        if (DEBUG_RUN_STEP == run_step)
        {
            debug_buffer_size = dim_grid[0] * dim_grid[1] * dim_grid[2] * sizeof(float);
            debug_buffer_h = (float*)calloc(1, debug_buffer_size);
            assert(NULL != debug_buffer_h);

            if (NULL == ocl_nb->debug_buffer)
            {
                ocl_nb->debug_buffer = clCreateBuffer(ocl_nb->dev_info->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                    debug_buffer_size, debug_buffer_h, &cl_error);

                assert(CL_SUCCESS == cl_error);
            }
         }

        run_step++;
    }
#endif
    if (debug)
    {
        fprintf(debug, "GPU launch configuration:\n\tLocal work size: %dx%dx%d\n\t"
                "Global work size : %dx%d\n\t#Super-clusters/clusters: %d/%d (%d)\n",
                dim_block[0], dim_block[1], dim_block[2],
                dim_grid[0], dim_grid[1], plist->nsci*NCL_PER_SUPERCL,
                NCL_PER_SUPERCL, plist->na_c);
    }

    fillin_ocl_structures(adat, nbp, plist, &atomdata_params, &nbparams_params);

    arg_no = 0;
    cl_error = clSetKernelArg(nb_kernel, arg_no++, sizeof(atomdata_params), &(atomdata_params));
    cl_error |= clSetKernelArg(nb_kernel, arg_no++, sizeof(nbparams_params), &(nbparams_params));
    cl_error |= clSetKernelArg(nb_kernel, arg_no++, sizeof(cl_mem), &(adat->xq));
    cl_error |= clSetKernelArg(nb_kernel, arg_no++, sizeof(cl_mem), &(adat->f));
    cl_error |= clSetKernelArg(nb_kernel, arg_no++, sizeof(cl_mem), &(adat->e_lj));
    cl_error |= clSetKernelArg(nb_kernel, arg_no++, sizeof(cl_mem), &(adat->e_el));
    cl_error |= clSetKernelArg(nb_kernel, arg_no++, sizeof(cl_mem), &(adat->fshift));
    cl_error |= clSetKernelArg(nb_kernel, arg_no++, sizeof(cl_mem), &(adat->atom_types));
    cl_error |= clSetKernelArg(nb_kernel, arg_no++, sizeof(cl_mem), &(adat->shift_vec));
    cl_error |= clSetKernelArg(nb_kernel, arg_no++, sizeof(cl_mem), &(nbp->nbfp_climg2d));
    cl_error |= clSetKernelArg(nb_kernel, arg_no++, sizeof(cl_mem), &(nbp->nbfp_comb_climg2d));
    cl_error |= clSetKernelArg(nb_kernel, arg_no++, sizeof(cl_mem), &(nbp->coulomb_tab_climg2d));
    cl_error |= clSetKernelArg(nb_kernel, arg_no++, sizeof(cl_mem), &(plist->sci));
    cl_error |= clSetKernelArg(nb_kernel, arg_no++, sizeof(cl_mem), &(plist->cj4));
    cl_error |= clSetKernelArg(nb_kernel, arg_no++, sizeof(cl_mem), &(plist->excl));
    cl_error |= clSetKernelArg(nb_kernel, arg_no++, sizeof(int), &bCalcFshift);
    cl_error |= clSetKernelArg(nb_kernel, arg_no++, shmem, NULL);
    cl_error |= clSetKernelArg(nb_kernel, arg_no++, sizeof(cl_mem), &(ocl_nb->debug_buffer));

    assert(cl_error == CL_SUCCESS);

    if(cl_error)
        printf("ClERROR! %d\n",cl_error);

    cl_error = clEnqueueNDRangeKernel(stream, nb_kernel, 3, NULL, dim_grid, dim_block, 0, NULL, bDoTime ? &(t->nb_k[iloc]) : NULL);
    assert(cl_error == CL_SUCCESS);

#ifdef DEBUG_OCL
    {
        static int run_step = 1;

        if (DEBUG_RUN_STEP == run_step)
        {
            FILE *pf;
            char file_name[256] = {0};

            ocl_copy_D2H_async(debug_buffer_h, ocl_nb->debug_buffer, 0,
                debug_buffer_size, stream, NULL);

            // Make sure all data has been transfered back from device
            clFinish(stream);

            printf("\nWriting debug_buffer to debug_buffer_ocl.txt...");

            sprintf(file_name, "debug_buffer_ocl_%d.txt", DEBUG_RUN_STEP);
            pf = fopen(file_name, "wt");
            assert(pf != NULL);

            fprintf(pf,"%20s", "");
            for (int j = 0; j < dim_grid[0]; j++)
            {
                char label[20];
                sprintf(label, "(wIdx=%2d thIdx=%2d)", j / dim_block[0], j % dim_block[0]);
                fprintf(pf, "%20s", label);
            }

            for (int i = 0; i < dim_grid[1]; i++)
            {
                char label[20];
                sprintf(label, "(wIdy=%2d thIdy=%2d)", i / dim_block[1], i % dim_block[1]);
                fprintf(pf, "\n%20s", label);

                for (int j = 0; j < dim_grid[0]; j++)
                    fprintf(pf, "%20.5f", debug_buffer_h[i * dim_grid[0] + j]);

                //fprintf(pf, "\n");
            }

            fclose(pf);

            printf(" done.\n");


            free(debug_buffer_h);
            debug_buffer_h = NULL;
        }

        run_step++;
    }
#endif
}

void dump_compare_results_cj4(nbnxn_cj4_t* results, int cnt, char* out_file, char* ref_file)
{
    FILE *pf;

    pf = fopen(out_file, "wt");
    assert(pf != NULL);

    fprintf(pf, "%20s%20s%20s%20s%20s%20s%20s%20s\n",
        "cj[0]", "cj[1]", "cj[2]", "cj[3]",
        "imei[0].excl_ind", "imei[0].imask",
        "imei[1].excl_ind", "imei[1].imask");

    for (int index = 0; index < cnt; index++)
    {
        fprintf(pf, "%20d%20d%20d%20d%20d%20u%20d%20u\n",
            results[index].cj[0], results[index].cj[1], results[index].cj[2], results[index].cj[3],
            results[index].imei[0].excl_ind, results[index].imei[0].imask,
            results[index].imei[1].excl_ind, results[index].imei[1].imask);
    }

    fclose(pf);

    printf("\nWrote results to %s", out_file);

    pf = fopen(ref_file, "rt");
    if (pf)
    {
        char c;
        int diff = 0;
        printf("\n%s file found. Comparing results...", ref_file);

        /* Skip the first line */
        c = 0;
        while (c != '\n') fscanf(pf, "%c", &c);

        for (int index = 0; index < cnt; index++)
        {
            int ref_val;

            for (int j = 0; j < 4; j++)
            {
                fscanf(pf, "%d", &ref_val);
                if (ref_val != results[index].cj[j])
                {
                    printf("\nDifference for cj[%d] at index %d computed value = %d reference value = %d",
                        j, index, results[index].cj[j], ref_val);

                    diff++;
                }
            }

            for (int j = 0; j < 2; j++)
            {
                fscanf(pf, "%d", &ref_val);
                if (ref_val != results[index].imei[j].excl_ind)
                {
                    printf("\nDifference for imei[%d].excl_ind at index %d computed value = %d reference value = %d",
                        j, index, results[index].imei[j].excl_ind, ref_val);

                    diff++;
                }

                fscanf(pf, "%u", &ref_val);
                if (ref_val != results[index].imei[j].imask)
                {
                    printf("\nDifference for imei[%d].imask at index %d computed value = %u reference value = %u",
                        j, index, results[index].imei[j].imask, ref_val);

                    diff++;
                }

            }
        }

        printf("\nFinished comparing results. Total number of differences: %d", diff);
        fclose(pf);
    }
    else
        printf("\n%s file not found. No comparison performed.", ref_file);
}

void dump_compare_results_f(float* results, int cnt, char* out_file, char* ref_file)
{
    FILE *pf;
    float cmp_eps = 0.001f;

    pf = fopen(out_file, "wt");
    assert(pf != NULL);

    for (int index = 0; index < cnt; index++)
    {
        fprintf(pf, "%15.5f\n", results[index]);
    }

    fclose(pf);

    printf("\nWrote results to %s", out_file);

    pf = fopen(ref_file, "rt");
    if (pf)
    {
        int diff = 0;
        printf("\n%s file found. Comparing results...", ref_file);
        for (int index = 0; index < cnt; index++)
        {
            float ref_val;
            fscanf(pf, "%f", &ref_val);
            if (((ref_val - results[index]) > cmp_eps) ||
                ((ref_val - results[index]) < -cmp_eps))
            {
                printf("\nDifference at index %d computed value = %15.5f reference value = %15.5f",
                    index, results[index], ref_val);

                diff++;
            }
        }

        printf("\nFinished comparing results. Total number of differences: %d", diff);
        fclose(pf);
    }
    else
        printf("\n%s file not found. No comparison performed.", ref_file);
}

void nbnxn_ocl_launch_cpyback(nbnxn_opencl_ptr_t        ocl_nb,
                               const nbnxn_atomdata_t *nbatom,
                               int                     flags,
                               int                     aloc)
{    
    cl_int      cl_error;
    int         adat_begin, adat_len, adat_end; /* local/nonlocal offset and length used for xq and f */
    int         iloc = -1;

    /* determine interaction locality from atom locality */
    if (LOCAL_A(aloc))
    {
        iloc = eintLocal;
    }
    else if (NONLOCAL_A(aloc))
    {
        iloc = eintNonlocal;
    }
    else
    {
        char stmp[STRLEN];
        sprintf(stmp, "Invalid atom locality passed (%d); valid here is only "
                "local (%d) or nonlocal (%d)", aloc, eatLocal, eatNonlocal);
                
        gmx_incons(stmp);
    }

    cl_atomdata_t   *adat    = ocl_nb->atdat;
    cl_timers_t     *t       = ocl_nb->timers;
    bool             bDoTime = ocl_nb->bDoTime;
    //cudaStream_t     stream  = ocl_nb->stream[iloc];
    cl_command_queue stream  = ocl_nb->stream[iloc];

    bool             bCalcEner   = flags & GMX_FORCE_VIRIAL;
    bool             bCalcFshift = flags & GMX_FORCE_VIRIAL;

    /* don't launch copy-back if there was no work to do */
    if (ocl_nb->plist[iloc]->nsci == 0)
    {
        return;
    }

    /* calculate the atom data index range based on locality */
    if (LOCAL_A(aloc))
    {
        adat_begin  = 0;
        adat_len    = adat->natoms_local;
        adat_end    = ocl_nb->atdat->natoms_local;
    }
    else
    {
        adat_begin  = adat->natoms_local;
        adat_len    = adat->natoms - adat->natoms_local;
        adat_end    = ocl_nb->atdat->natoms;
    }

    /* beginning of timed D2H section */

    if (!ocl_nb->bUseStreamSync)
    {
        /* For safety reasons set a few (5%) forces to NaN. This way even if the
           polling "hack" fails with some future NVIDIA driver we'll get a crash. */
        for (int i = adat_begin; i < 3*adat_end + 2; i += adat_len/20)
        {
#ifdef NAN
            nbatom->out[0].f[i] = NAN;
#else
#  ifdef _MSVC
            if (numeric_limits<float>::has_quiet_NaN)
            {
                nbatom->out[0].f[i] = numeric_limits<float>::quiet_NaN();
            }
            else
#  endif
            {
                nbatom->out[0].f[i] = GMX_REAL_MAX;
            }
#endif
        }

        /* Set the last four bytes of the force array to a bit pattern
           which can't be the result of the force calculation:
           max exponent (127) and zero mantissa. */
        *(unsigned int*)&nbatom->out[0].f[adat_end*3 - 1] = poll_wait_pattern;
    }

    /* With DD the local D2H transfer can only start after the non-local
       has been launched. */
    if (iloc == eintLocal && ocl_nb->bUseTwoStreams)
    {
        wait_ocl_event(&(ocl_nb->nonlocal_done));        
    }

    /* DtoH f */    
    ocl_copy_D2H_async(nbatom->out[0].f + adat_begin * 3, adat->f, adat_begin,
                      (adat_len)*sizeof(float) * 3, stream, bDoTime ? &(t->nb_d2h_f[iloc]) : NULL);

    /* After the non-local D2H is launched the nonlocal_done event can be
       recorded which signals that the local D2H can proceed. This event is not
       placed after the non-local kernel because we first need the non-local
       data back first. */
    if (iloc == eintNonlocal)
    {        
        cl_error = clEnqueueMarker(stream, &(ocl_nb->nonlocal_done));
        assert(CL_SUCCESS == cl_error);
    }

    /* only transfer energies in the local stream */
    if (LOCAL_I(iloc))
    {
        /* DtoH fshift */
        if (bCalcFshift)
        {            
             // TO DO: review fshift data type and how its size is computed
            ocl_copy_D2H_async(ocl_nb->nbst.fshift, adat->fshift, 0,
                              3 * SHIFTS * sizeof(float), stream, bDoTime ? &(t->nb_d2h_fshift[iloc]) : NULL);
        }

        /* DtoH energies */
        if (bCalcEner)
        {            
            ocl_copy_D2H_async(ocl_nb->nbst.e_lj, adat->e_lj, 0,
                              sizeof(float), stream, bDoTime ? &(t->nb_d2h_e_lj[iloc]) : NULL);
            
            ocl_copy_D2H_async(ocl_nb->nbst.e_el, adat->e_el, 0,
                              sizeof(float), stream, bDoTime ? &(t->nb_d2h_e_el[iloc]) : NULL);
        }
    }

/* Uncomment this define to enable cj4 debugging for the first kernel run */
//#define DEBUG_DUMP_CJ4_OCL
#ifdef DEBUG_DUMP_CJ4_OCL
    {
        static int run_step = 1;

        if (DEBUG_RUN_STEP == run_step)
        {
            nbnxn_cj4_t *temp_cj4;
            int cnt;
            size_t size;
            char ocl_file_name[256] = {0};
            char cuda_file_name[256] = {0};

            cnt = ocl_nb->plist[0]->ncj4;
            size = cnt * sizeof(nbnxn_cj4_t);
            temp_cj4 = (nbnxn_cj4_t*)malloc(size);

            ocl_copy_D2H_async(temp_cj4, ocl_nb->plist[0]->cj4, 0,
                size, stream, NULL);

            // Make sure all data has been transfered back from device
            clFinish(stream);

            sprintf(ocl_file_name, "ocl_cj4_%d.txt", DEBUG_RUN_STEP);
            sprintf(cuda_file_name, "cuda_cj4_%d.txt", DEBUG_RUN_STEP);
            dump_compare_results_cj4(temp_cj4, cnt, ocl_file_name, cuda_file_name);

            free(temp_cj4);
        }

        run_step++;
    }
#endif

/* Uncomment this define to enable f debugging for the first kernel run */
//#define DEBUG_DUMP_F_OCL
#ifdef DEBUG_DUMP_F_OCL
    {
        static int run_step = 1;

        if (DEBUG_RUN_STEP == run_step)
        {
            char ocl_file_name[256] = {0};
            char cuda_file_name[256] = {0};

            // Make sure all data has been transfered back from device
            clFinish(stream);

            sprintf(ocl_file_name, "ocl_f_%d.txt", DEBUG_RUN_STEP);
            sprintf(cuda_file_name, "cuda_f_%d.txt", DEBUG_RUN_STEP);

            dump_compare_results_f(nbatom->out[0].f + adat_begin * 3, (adat_len) * 3,
                ocl_file_name, cuda_file_name);
        }

        run_step++;
    }
#endif

/* Uncomment this define to enable fshift debugging for the first kernel run */
//#define DEBUG_DUMP_FSHIFT_OCL
#ifdef DEBUG_DUMP_FSHIFT_OCL
    {
        static int run_step = 1;

        if (DEBUG_RUN_STEP == run_step)
        {
            char ocl_file_name[256] = {0};
            char cuda_file_name[256] = {0};

            // Make sure all data has been transfered back from device
            clFinish(stream);
            
            sprintf(ocl_file_name, "ocl_fshift_%d.txt", DEBUG_RUN_STEP);
            sprintf(cuda_file_name, "cuda_fshift_%d.txt", DEBUG_RUN_STEP);

            dump_compare_results_f(ocl_nb->nbst.fshift, SHIFTS * 3,
                ocl_file_name, cuda_file_name);
        }

        run_step++;
    }
#endif
}

/* Atomic compare-exchange operation on unsigned values. It is used in
 * polling wait for the GPU.
 */
static inline bool atomic_cas(volatile unsigned int *ptr,
                              unsigned int           oldval,
                              unsigned int           newval)
{
    assert(ptr);

#ifdef TMPI_ATOMICS
    return tMPI_Atomic_cas((tMPI_Atomic_t *)ptr, oldval, newval);
#else
    gmx_incons("Atomic operations not available, atomic_cas() should not have been called!");
    return true;
#endif
}

void nbnxn_ocl_wait_gpu(nbnxn_opencl_ptr_t cu_nb,
                         const nbnxn_atomdata_t *nbatom,
                         int flags, int aloc,
                         real *e_lj, real *e_el, rvec *fshift)
{
    /* NOTE:  only implemented for single-precision at this time */
    cl_int                 cl_error;
    int                    i, adat_end, iloc = -1;	
    volatile unsigned int *poll_word;

    /* determine interaction locality from atom locality */
    if (LOCAL_A(aloc))
    {
        iloc = eintLocal;
    }
    else if (NONLOCAL_A(aloc))
    {
        iloc = eintNonlocal;
    }
    else
    {
        char stmp[STRLEN];
        sprintf(stmp, "Invalid atom locality passed (%d); valid here is only "
                "local (%d) or nonlocal (%d)", aloc, eatLocal, eatNonlocal);
        gmx_incons(stmp);
    }

    cl_plist_t      *plist    = cu_nb->plist[iloc];
    cl_timers_t     *timers   = cu_nb->timers;
    wallclock_gpu_t *timings  = cu_nb->timings;
    cl_nb_staging    nbst     = cu_nb->nbst;

	bool             bCalcEner   = flags & GMX_FORCE_VIRIAL;
	bool             bCalcFshift = flags & GMX_FORCE_VIRIAL;

    /* turn energy calculation always on/off (for debugging/testing only) */
    bCalcEner = (bCalcEner || always_ener) && !never_ener;

    /* don't launch wait/update timers & counters if there was no work to do

       NOTE: if timing with multiple GPUs (streams) becomes possible, the
       counters could end up being inconsistent due to not being incremented
       on some of the nodes! */
    if (cu_nb->plist[iloc]->nsci == 0)
    {
        return;
    }

    /* calculate the atom data index range based on locality */
    if (LOCAL_A(aloc))
    {
        adat_end = cu_nb->atdat->natoms_local;
    }
    else
    {
        adat_end = cu_nb->atdat->natoms;
    }

    if (cu_nb->bUseStreamSync)
    {
        /* Actual sync point. Waits for everything to be finished in the command queue. TODO: Find out if a more fine grained solution is needed */
        cl_error = clFinish(cu_nb->stream[iloc]);
        assert(CL_SUCCESS == cl_error);
    }
    else
    {
        /* Busy-wait until we get the signal pattern set in last byte
         * of the l/nl float vector. This pattern corresponds to a floating
         * point number which can't be the result of the force calculation
         * (maximum, 127 exponent and 0 mantissa).
         * The polling uses atomic compare-exchange.
         */
        poll_word = (volatile unsigned int*)&nbatom->out[0].f[adat_end*3 - 1];
        while (atomic_cas(poll_word, poll_wait_pattern, poll_wait_pattern))
        {
        }
    }

    /* timing data accumulation */
    if (cu_nb->bDoTime)
    {
        /* only increase counter once (at local F wait) */
        if (LOCAL_I(iloc))
        {
            timings->nb_c++;
            timings->ktime[plist->bDoPrune ? 1 : 0][bCalcEner ? 1 : 0].c += 1;
        }

        /* kernel timings */

        timings->ktime[plist->bDoPrune ? 1 : 0][bCalcEner ? 1 : 0].t +=
            ocl_event_elapsed_ms(timers->nb_k + iloc);

        /* X/q H2D and F D2H timings */
        timings->nb_h2d_t += ocl_event_elapsed_ms(timers->nb_h2d        + iloc);            
        timings->nb_d2h_t += ocl_event_elapsed_ms(timers->nb_d2h_f      + iloc);
        timings->nb_d2h_t += ocl_event_elapsed_ms(timers->nb_d2h_fshift + iloc);
        timings->nb_d2h_t += ocl_event_elapsed_ms(timers->nb_d2h_e_el   + iloc);
        timings->nb_d2h_t += ocl_event_elapsed_ms(timers->nb_d2h_e_lj   + iloc);

        /* only count atdat and pair-list H2D at pair-search step */
        if (plist->bDoPrune)
        {
            /* atdat transfer timing (add only once, at local F wait) */
            if (LOCAL_A(aloc))
            {
                timings->pl_h2d_c++;
                timings->pl_h2d_t += ocl_event_elapsed_ms(&(timers->atdat));
            }

            timings->pl_h2d_t +=
                ocl_event_elapsed_ms(timers->pl_h2d_sci     + iloc) +
                ocl_event_elapsed_ms(timers->pl_h2d_cj4     + iloc) +
                ocl_event_elapsed_ms(timers->pl_h2d_excl    + iloc);

        }
    }

	/* add up energies and shift forces (only once at local F wait) */
	if (LOCAL_I(iloc))
	{
	    if (bCalcEner)
	    {
	        *e_lj += *nbst.e_lj;
	        *e_el += *nbst.e_el;
	    }

	    if (bCalcFshift)
	    {
	        for (i = 0; i < SHIFTS; i++)
	        {
	            fshift[i][0] += nbst.fshift[i*3];
				fshift[i][1] += nbst.fshift[i*3+1];
	            fshift[i][2] += nbst.fshift[i*3+2 ];
	        }
	    }
	}

	/* turn off pruning (doesn't matter if this is pair-search step or not) */
	plist->bDoPrune = false;

}
////void nbnxn_cuda_wait_gpu(nbnxn_cuda_ptr_t cu_nb,
////                         const nbnxn_atomdata_t *nbatom,
////                         int flags, int aloc,
////                         real *e_lj, real *e_el, rvec *fshift)
////{
////    /* NOTE:  only implemented for single-precision at this time */
////    cudaError_t            stat;
////    int                    i, adat_end, iloc = -1;
////    volatile unsigned int *poll_word;
////
////    /* determine interaction locality from atom locality */
////    if (LOCAL_A(aloc))
////    {
////        iloc = eintLocal;
////    }
////    else if (NONLOCAL_A(aloc))
////    {
////        iloc = eintNonlocal;
////    }
////    else
////    {
////        char stmp[STRLEN];
////        sprintf(stmp, "Invalid atom locality passed (%d); valid here is only "
////                "local (%d) or nonlocal (%d)", aloc, eatLocal, eatNonlocal);
////        gmx_incons(stmp);
////    }
////
////    cu_plist_t      *plist    = cu_nb->plist[iloc];
////    cu_timers_t     *timers   = cu_nb->timers;
////    wallclock_gpu_t *timings  = cu_nb->timings;
////    nb_staging       nbst     = cu_nb->nbst;
////
////    bool             bCalcEner   = flags & GMX_FORCE_VIRIAL;
////    bool             bCalcFshift = flags & GMX_FORCE_VIRIAL;
////
////    /* turn energy calculation always on/off (for debugging/testing only) */
////    bCalcEner = (bCalcEner || always_ener) && !never_ener;
////
////    /* don't launch wait/update timers & counters if there was no work to do
////
////       NOTE: if timing with multiple GPUs (streams) becomes possible, the
////       counters could end up being inconsistent due to not being incremented
////       on some of the nodes! */
////    if (cu_nb->plist[iloc]->nsci == 0)
////    {
////        return;
////    }
////
////    /* calculate the atom data index range based on locality */
////    if (LOCAL_A(aloc))
////    {
////        adat_end = cu_nb->atdat->natoms_local;
////    }
////    else
////    {
////        adat_end = cu_nb->atdat->natoms;
////    }
////
////    if (cu_nb->bUseStreamSync)
////    {
////        stat = cudaStreamSynchronize(cu_nb->stream[iloc]);
////        CU_RET_ERR(stat, "cudaStreamSynchronize failed in cu_blockwait_nb");
////    }
////    else
////    {
////        /* Busy-wait until we get the signal pattern set in last byte
////         * of the l/nl float vector. This pattern corresponds to a floating
////         * point number which can't be the result of the force calculation
////         * (maximum, 127 exponent and 0 mantissa).
////         * The polling uses atomic compare-exchange.
////         */
////        poll_word = (volatile unsigned int*)&nbatom->out[0].f[adat_end*3 - 1];
////        while (atomic_cas(poll_word, poll_wait_pattern, poll_wait_pattern))
////        {
////        }
////    }
////
////    /* timing data accumulation */
////    if (cu_nb->bDoTime)
////    {
////        /* only increase counter once (at local F wait) */
////        if (LOCAL_I(iloc))
////        {
////            timings->nb_c++;
////            timings->ktime[plist->bDoPrune ? 1 : 0][bCalcEner ? 1 : 0].c += 1;
////        }
////
////        /* kernel timings */
////        timings->ktime[plist->bDoPrune ? 1 : 0][bCalcEner ? 1 : 0].t +=
////            cu_event_elapsed(timers->start_nb_k[iloc], timers->stop_nb_k[iloc]);
////
////        /* X/q H2D and F D2H timings */
////        timings->nb_h2d_t += cu_event_elapsed(timers->start_nb_h2d[iloc],
////                                              timers->stop_nb_h2d[iloc]);
////        timings->nb_d2h_t += cu_event_elapsed(timers->start_nb_d2h[iloc],
////                                              timers->stop_nb_d2h[iloc]);
////
////        /* only count atdat and pair-list H2D at pair-search step */
////        if (plist->bDoPrune)
////        {
////            /* atdat transfer timing (add only once, at local F wait) */
////            if (LOCAL_A(aloc))
////            {
////                timings->pl_h2d_c++;
////                timings->pl_h2d_t += cu_event_elapsed(timers->start_atdat,
////                                                      timers->stop_atdat);
////            }
////
////            timings->pl_h2d_t += cu_event_elapsed(timers->start_pl_h2d[iloc],
////                                                  timers->stop_pl_h2d[iloc]);
////        }
////    }
////
////    /* add up energies and shift forces (only once at local F wait) */
////    if (LOCAL_I(iloc))
////    {
////        if (bCalcEner)
////        {
////            *e_lj += *nbst.e_lj;
////            *e_el += *nbst.e_el;
////        }
////
////        if (bCalcFshift)
////        {
////            for (i = 0; i < SHIFTS; i++)
////            {
////                fshift[i][0] += nbst.fshift[i].x;
////                fshift[i][1] += nbst.fshift[i].y;
////                fshift[i][2] += nbst.fshift[i].z;
////            }
////        }
////    }
////
////    /* turn off pruning (doesn't matter if this is pair-search step or not) */
////    plist->bDoPrune = false;
////}

/////*! Return the reference to the nbfp texture. */
////const struct texture<float, 1, cudaReadModeElementType> &nbnxn_cuda_get_nbfp_texref()
////{
////    return nbfp_texref;
////}
////
/////*! Return the reference to the nbfp_comb texture. */
////const struct texture<float, 1, cudaReadModeElementType> &nbnxn_cuda_get_nbfp_comb_texref()
////{
////    return nbfp_comb_texref;
////}
////
/////*! Return the reference to the coulomb_tab. */
////const struct texture<float, 1, cudaReadModeElementType> &nbnxn_cuda_get_coulomb_tab_texref()
////{
////    return coulomb_tab_texref;
////}

/////*! Set up the cache configuration for the non-bonded kernels,
//// */
////void nbnxn_cuda_set_cacheconfig(cuda_dev_info_t *devinfo)
////{
////    cudaError_t stat;
////
////    for (int i = 0; i < eelCuNR; i++)
////    {
////        for (int j = 0; j < evdwCuNR; j++)
////        {
////            if (devinfo->prop.major >= 3)
////            {
////                /* Default kernel on sm 3.x 48/16 kB Shared/L1 */
////                cudaFuncSetCacheConfig(nb_kfunc_ener_prune_ptr[i][j], cudaFuncCachePreferShared);
////                cudaFuncSetCacheConfig(nb_kfunc_ener_noprune_ptr[i][j], cudaFuncCachePreferShared);
////                cudaFuncSetCacheConfig(nb_kfunc_noener_prune_ptr[i][j], cudaFuncCachePreferShared);
////                stat = cudaFuncSetCacheConfig(nb_kfunc_noener_noprune_ptr[i][j], cudaFuncCachePreferShared);
////            }
////            else
////            {
////                /* On Fermi prefer L1 gives 2% higher performance */
////                /* Default kernel on sm_2.x 16/48 kB Shared/L1 */
////                cudaFuncSetCacheConfig(nb_kfunc_ener_prune_ptr[i][j], cudaFuncCachePreferL1);
////                cudaFuncSetCacheConfig(nb_kfunc_ener_noprune_ptr[i][j], cudaFuncCachePreferL1);
////                cudaFuncSetCacheConfig(nb_kfunc_noener_prune_ptr[i][j], cudaFuncCachePreferL1);
////                stat = cudaFuncSetCacheConfig(nb_kfunc_noener_noprune_ptr[i][j], cudaFuncCachePreferL1);
////            }
////            CU_RET_ERR(stat, "cudaFuncSetCacheConfig failed");
////        }
////    }
////}
