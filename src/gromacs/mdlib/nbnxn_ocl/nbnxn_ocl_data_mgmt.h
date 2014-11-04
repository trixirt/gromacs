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

#if !defined(NBNXN_OCL_DATA_MGMT_H) && defined(GMX_USE_OPENCL)
#define NBNXN_OCL_DATA_MGMT_H

#include "types/simple.h"
#include "types/interaction_const.h"
//#include "types/nbnxn_cuda_types_ext.h"
#include "types/nbnxn_ocl_types_ext.h"
#include "types/hw_info.h"


#if defined(GMX_GPU) && defined(GMX_USE_OPENCL)
#define FUNC_TERM ;
#define FUNC_QUALIFIER
#define FUNC_TERM_P ;
#else
#define FUNC_TERM {}
#define FUNC_QUALIFIER static;
#define FUNC_TERM_P {return NULL}
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct nonbonded_verlet_group_t;
struct nbnxn_pairlist_t;
struct nbnxn_atomdata_t;

FUNC_QUALIFIER
void nbnxn_ocl_convert_gmx_to_gpu_flavors(
    const int gmx_eeltype,
    const int gmx_vdwtype,
    const int gmx_vdw_modifier,
    const int gmx_ljpme_comb_rule,
    int *gpu_eeltype,
    int *gpu_vdwtype) FUNC_TERM

/** Initializes the data structures related to CUDA nonbonded calculations. */
FUNC_QUALIFIER
void nbnxn_ocl_init(FILE gmx_unused                 *fplog,
                     nbnxn_opencl_ptr_t gmx_unused     *p_cu_nb,
                     const gmx_gpu_info_t gmx_unused *gpu_info,
                     const gmx_gpu_opt_t gmx_unused  *gpu_opt,
                     int gmx_unused                   my_gpu_index,
                     /* true of both local and non-local are don on GPU */
                     gmx_bool gmx_unused              bLocalAndNonlocal) FUNC_TERM

/** Initializes simulation constant data. */
FUNC_QUALIFIER
void nbnxn_ocl_init_const(nbnxn_opencl_ptr_t               gmx_unused         cu_nb,
                           const interaction_const_t      gmx_unused        *ic,
                           const struct nonbonded_verlet_group_t gmx_unused *nbv_group) FUNC_TERM
                           
/** Initializes pair-list data for GPU, called at every pair search step. */
FUNC_QUALIFIER
void nbnxn_ocl_init_pairlist(nbnxn_opencl_ptr_t       gmx_unused         cu_nb,
                              const struct nbnxn_pairlist_t gmx_unused *h_nblist,
                              int                    gmx_unused         iloc) FUNC_TERM

/** Initializes atom-data on the GPU, called at every pair search step. */
FUNC_QUALIFIER
void nbnxn_ocl_init_atomdata(const nbnxn_opencl_ptr_t       gmx_unused   cu_nb,
                              const struct nbnxn_atomdata_t gmx_unused *atomdata) FUNC_TERM

/*! \brief Update parameters during PP-PME load balancing. */
FUNC_QUALIFIER
void nbnxn_ocl_pme_loadbal_update_param(const struct nonbonded_verlet_t gmx_unused *nbv,
                                         const interaction_const_t gmx_unused       *ic) FUNC_TERM

/** Uploads shift vector to the GPU if the box is dynamic (otherwise just returns). */
FUNC_QUALIFIER
void nbnxn_ocl_upload_shiftvec(nbnxn_opencl_ptr_t       gmx_unused         cu_nb,
                                const struct nbnxn_atomdata_t gmx_unused *nbatom) FUNC_TERM

/** Clears GPU outputs: nonbonded force, shift force and energy. */
FUNC_QUALIFIER
void nbnxn_ocl_clear_outputs(
	nbnxn_opencl_ptr_t ocl_nb,
    int gmx_unused      flags) FUNC_TERM

/** Frees all GPU resources used for the nonbonded calculations. */
FUNC_QUALIFIER
void nbnxn_ocl_free(nbnxn_opencl_ptr_t gmx_unused  cu_nb) FUNC_TERM

/** Returns the GPU timings structure or NULL if GPU is not used or timing is off. */
FUNC_QUALIFIER
wallclock_gpu_t * nbnxn_ocl_get_timings(nbnxn_opencl_ptr_t gmx_unused cu_nb) FUNC_TERM_P

/** Resets nonbonded GPU timings. */
FUNC_QUALIFIER
void nbnxn_ocl_reset_timings(struct nonbonded_verlet_t gmx_unused *nbv) FUNC_TERM

///////** Calculates the minimum size of proximity lists to improve SM load balance
////// *  with CUDA non-bonded kernels. */
//////FUNC_QUALIFIER
//////int nbnxn_cuda_min_ci_balanced(nbnxn_cuda_ptr_t gmx_unused cu_nb)
//////#ifdef GMX_GPU
//////;
//////#else
//////{
//////    return -1;
//////}
//////#endif
//////


/** Calculates the minimum size of proximity lists to improve SM load balance
 *  with CUDA non-bonded kernels. */
FUNC_QUALIFIER
int nbnxn_ocl_min_ci_balanced(nbnxn_opencl_ptr_t gmx_unused ocl_nb)
#if defined(GMX_GPU) && defined(GMX_USE_OPENCL)
;
#else
{
    return -1;
}
#endif

/** Returns if analytical Ewald CUDA kernels are used. */
FUNC_QUALIFIER
gmx_bool nbnxn_ocl_is_kernel_ewald_analytical(const nbnxn_opencl_ptr_t gmx_unused ocl_nb)
#if defined(GMX_GPU) && defined(GMX_USE_OPENCL)
;
#else
{
    return FALSE;
}
#endif

FUNC_QUALIFIER
int ocl_copy_H2D_async(cl_mem d_dest, void * h_src, size_t offset, size_t bytes, cl_command_queue command_queue, cl_event *copy_event);

FUNC_QUALIFIER
int ocl_copy_D2H_async(void * h_dest, cl_mem d_src, size_t offset, size_t bytes, cl_command_queue command_queue, cl_event *copy_event);

#ifdef __cplusplus
}
#endif

#undef FUNC_TERM
#undef FUNC_QUALIFIER

#endif /* NBNXN_CUDA_DATA_MGMT_H */
