/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2010, The GROMACS development team.
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

#ifndef GMX_GMXLIB_GPU_UTILS_GPU_UTILS_H
#define GMX_GMXLIB_GPU_UTILS_GPU_UTILS_H

#include "config.h"

#include "gromacs/gmxlib/gpu_utils/gpu_macros.h"
#include "gromacs/legacyheaders/types/hw_info.h"
#include "gromacs/legacyheaders/types/simple.h"

#ifndef GMX_GPU
typedef int ocl_gpu_id_t;
typedef int ocl_vendor_id_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct gmx_gpu_info_t;

GPU_FUNC_QUALIFIER
int detect_gpus(struct gmx_gpu_info_t gmx_unused *gpu_info, char gmx_unused *err_str) GPU_FUNC_TERM_WITH_RETURN(-1)

GPU_FUNC_QUALIFIER
void pick_compatible_gpus(const struct gmx_gpu_info_t gmx_unused *gpu_info,
                          gmx_gpu_opt_t gmx_unused        *gpu_opt) GPU_FUNC_TERM

GPU_FUNC_QUALIFIER
gmx_bool check_selected_gpus(int gmx_unused                  *checkres,
                             const struct gmx_gpu_info_t gmx_unused *gpu_info,
                             gmx_gpu_opt_t gmx_unused        *gpu_opt) GPU_FUNC_TERM_WITH_RETURN(-1)

GPU_FUNC_QUALIFIER
void free_gpu_info(const struct gmx_gpu_info_t gmx_unused *gpu_info) GPU_FUNC_TERM

GPU_FUNC_QUALIFIER
gmx_bool init_gpu(int gmx_unused mygpu, char gmx_unused *result_str,
                  const struct gmx_gpu_info_t gmx_unused *gpu_info,
                  const gmx_gpu_opt_t gmx_unused *gpu_opt) GPU_FUNC_TERM_WITH_RETURN(-1)

CUDA_FUNC_QUALIFIER
gmx_bool free_cuda_gpu(char gmx_unused *result_str) CUDA_FUNC_TERM_WITH_RETURN(-1)

/*! \brief Returns the device ID of the GPU currently in use.*/
CUDA_FUNC_QUALIFIER
int get_current_cuda_gpu_device_id(void) CUDA_FUNC_TERM_WITH_RETURN(-1)

CUDA_FUNC_QUALIFIER
int get_cuda_gpu_device_id(const struct gmx_gpu_info_t gmx_unused *gpu_info,
                           const gmx_gpu_opt_t gmx_unused  *gpu_opt,
                           int gmx_unused                   index) CUDA_FUNC_TERM_WITH_RETURN(-1)

OPENCL_FUNC_QUALIFIER
char* get_ocl_gpu_device_name(const gmx_gpu_info_t gmx_unused *gpu_info,
                              const gmx_gpu_opt_t gmx_unused *gpu_opt,
                              int gmx_unused                  idx) OPENCL_FUNC_TERM_WITH_RETURN(NULL)

GPU_FUNC_QUALIFIER
void get_gpu_device_info_string(char gmx_unused *s,
                                const struct gmx_gpu_info_t gmx_unused *gpu_info,
                                int gmx_unused index) GPU_FUNC_TERM

GPU_FUNC_QUALIFIER
size_t sizeof_gpu_dev_info(void) GPU_FUNC_TERM_WITH_RETURN(0)

/*! \brief Function that should return a pointer *ptr to memory
 * of size nbytes.
 *
 * Error handling should be done within this function.
 */
typedef void gmx_host_alloc_t (void **ptr, size_t nbytes);

/*! \brief Function that should free the memory pointed to by *ptr.
 *
 * NULL should not be passed to this function.
 */
typedef void gmx_host_free_t (void *ptr);

/*! \brief Set allocation functions used by the GPU host */
void gpu_set_host_malloc_and_free(bool               bUseGpuKernels,
                                  gmx_host_alloc_t **nb_alloc,
                                  gmx_host_free_t  **nb_free);

#ifdef __cplusplus
}
#endif

#endif
