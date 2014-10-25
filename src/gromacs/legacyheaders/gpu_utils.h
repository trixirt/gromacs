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

#ifndef _GPU_UTILS_H_
#define _GPU_UTILS_H_

#include "config.h"

#include "types/simple.h"
#include "types/hw_info.h"

/* TODO This needs to be fixed
 *      Use universal generic interface functions for all OpenCL/CUDA ops here
 *      Handle specifics inside the functions-> CUDA/OpenCL path
 */
/* Careful here, OCL functions are in gpu_utils.c while CUDA in gpu_utils.cu. so
    ocl functions will not have access to any cuda functions in here */
#if defined(GMX_GPU) && !defined(GMX_USE_OPENCL)
#define FUNC_TERM_INT ;
#define FUNC_TERM_SIZE_T ;
#define FUNC_TERM_VOID ;
#define FUNC_QUALIFIER
#define FUNC_TERM_INT_OPENCL {return -1;}
#define FUNC_TERM_VOID_OPENCL {}
#define FUNC_QUALIFIER_OPENCL static
typedef int ocl_gpu_id_t;
typedef int ocl_vendor_id_t;
#elif defined(GMX_GPU) && defined(GMX_USE_OPENCL)
#define FUNC_TERM_INT {return -1; }
#define FUNC_TERM_SIZE_T {return 0; }
#define FUNC_TERM_VOID {}
#define FUNC_QUALIFIER static
#define FUNC_TERM_INT_OPENCL ;
#define FUNC_TERM_VOID_OPENCL ;
#define FUNC_QUALIFIER_OPENCL 
#else
#define FUNC_TERM_INT {return -1; }
#define FUNC_TERM_SIZE_T {return 0; }
#define FUNC_TERM_VOID {}
#define FUNC_QUALIFIER static
#define FUNC_TERM_INT_OPENCL {return -1;}
#define FUNC_TERM_VOID_OPENCL {}
#define FUNC_QUALIFIER_OPENCL static
typedef int ocl_gpu_id_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

FUNC_QUALIFIER
int detect_cuda_gpus(gmx_gpu_info_t gmx_unused *gpu_info, char gmx_unused *err_str) FUNC_TERM_INT

FUNC_QUALIFIER_OPENCL
int detect_ocl_gpus(gmx_gpu_info_t gmx_unused *gpu_info, char gmx_unused *err_str) FUNC_TERM_INT_OPENCL

FUNC_QUALIFIER
void pick_compatible_cuda_gpus(const gmx_gpu_info_t gmx_unused *gpu_info,
                          gmx_gpu_opt_t gmx_unused        *gpu_opt) FUNC_TERM_VOID

FUNC_QUALIFIER_OPENCL
void pick_compatible_ocl_gpus(const gmx_gpu_info_t gmx_unused *gpu_info,
                          gmx_gpu_opt_t gmx_unused        *gpu_opt) FUNC_TERM_VOID_OPENCL

FUNC_QUALIFIER
gmx_bool check_selected_cuda_gpus(int gmx_unused                  *checkres,
                                  const gmx_gpu_info_t gmx_unused *gpu_info,
                                  gmx_gpu_opt_t gmx_unused        *gpu_opt) FUNC_TERM_INT

FUNC_QUALIFIER
void free_cuda_gpu_info(const gmx_gpu_info_t gmx_unused *gpu_info) FUNC_TERM_VOID

FUNC_QUALIFIER_OPENCL
void free_ocl_gpu_info(const gmx_gpu_info_t gmx_unused *gpu_info) FUNC_TERM_VOID_OPENCL

FUNC_QUALIFIER
gmx_bool init_cuda_gpu(int gmx_unused mygpu, char gmx_unused *result_str,
                  const gmx_gpu_info_t gmx_unused *gpu_info,
                  const gmx_gpu_opt_t gmx_unused *gpu_opt) FUNC_TERM_INT

FUNC_QUALIFIER_OPENCL
gmx_bool init_ocl_gpu(int gmx_unused mygpu, char gmx_unused *result_str,
                  const gmx_gpu_info_t gmx_unused *gpu_info,
                  const gmx_gpu_opt_t gmx_unused *gpu_opt,
                  const int gmx_unused eeltype,
                  const int gmx_unused vdwtype,
                  const gmx_bool gmx_unused bOclDoFastGen
                     ) FUNC_TERM_INT_OPENCL

FUNC_QUALIFIER
gmx_bool free_gpu(char gmx_unused *result_str) FUNC_TERM_INT

/*! \brief Returns the device ID of the GPU currently in use.*/
FUNC_QUALIFIER
int get_current_gpu_device_id(void) FUNC_TERM_INT

FUNC_QUALIFIER
int get_cuda_gpu_device_id(const gmx_gpu_info_t gmx_unused *gpu_info,
                      const gmx_gpu_opt_t gmx_unused  *gpu_opt,
                      int gmx_unused                   index) FUNC_TERM_INT

FUNC_QUALIFIER_OPENCL
ocl_gpu_id_t get_ocl_gpu_device_id(const gmx_gpu_info_t gmx_unused *gpu_info,
                      const gmx_gpu_opt_t gmx_unused  *gpu_opt,
                      int gmx_unused                   index) FUNC_TERM_INT_OPENCL

FUNC_QUALIFIER
void get_cuda_gpu_device_info_string(char gmx_unused *s, const gmx_gpu_info_t gmx_unused *gpu_info, int gmx_unused index) FUNC_TERM_VOID

FUNC_QUALIFIER_OPENCL
void get_ocl_gpu_device_info_string(char gmx_unused *s, const gmx_gpu_info_t gmx_unused *gpu_info, int gmx_unused index) FUNC_TERM_VOID_OPENCL

FUNC_QUALIFIER_OPENCL
ocl_vendor_id_t get_vendor_id(char *vendor_name) FUNC_TERM_INT_OPENCL

FUNC_QUALIFIER
size_t sizeof_cuda_dev_info(void) FUNC_TERM_SIZE_T

FUNC_QUALIFIER_OPENCL
void ocl_pmalloc(void **h_ptr, size_t nbytes) FUNC_TERM_VOID_OPENCL

FUNC_QUALIFIER_OPENCL
void ocl_pfree(void *h_ptr) FUNC_TERM_VOID_OPENCL

#if defined(GMX_GPU) && defined(GMX_USE_OPENCL) && !defined(NDEBUG)
/* Debugger callable function that prints the name of a kernel function pointer */
cl_int dbg_ocl_kernel_name(const cl_kernel kernel);
cl_int dbg_ocl_kernel_name_address(void* kernel);
#endif

#ifdef __cplusplus
}
#endif

#undef FUNC_TERM_INT
#undef FUNC_TERM_VOID
#undef FUNC_QUALIFIER

#undef FUNC_TERM_INT_OPENCL
#undef FUNC_TERM_VOID_OPENCL
#undef FUNC_QUALIFIER_OPENCL

#endif /* _GPU_UTILS_H_ */
