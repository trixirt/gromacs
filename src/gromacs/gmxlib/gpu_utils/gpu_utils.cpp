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

/** \file gpu_utils.cpp
 *  \brief Detection and initialization for OpenCL devices.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <memory.h>
#include <string.h>

#include "gromacs/legacyheaders/types/hw_info.h"

#include "gromacs/gmxlib/ocl_tools/oclutils.h"

#include "gromacs/legacyheaders/types/enums.h"
#include "gromacs/legacyheaders/gpu_utils.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"

#include "ocl_compiler.hpp"

#define CALLOCLFUNC_LOGERROR(func, err_str, retval) { \
        cl_int opencl_ret = func; \
        if (CL_SUCCESS != opencl_ret) \
        { \
            sprintf(err_str, "OpenCL error %d", opencl_ret); \
            retval = -1; \
        } \
        else{ \
            retval = 0; } \
}


/*! \brief Helper function that checks whether a given GPU status indicates compatible GPU.
 *
 * \param[in] stat  GPU status.
 * \returns         true if the provided status is egpuCompatible, otherwise false.
 */
static bool is_compatible_ocl_gpu(int stat)
{
    return (stat == egpuCompatible);
}

/*! \brief Returns true if the gpu characterized by the device properties is
 *  supported by the native gpu acceleration.
 * \returns             true if the GPU properties passed indicate a compatible
 *                      GPU, otherwise false.
 */
static int is_gmx_supported_ocl_gpu_id(gpu_info_ptr_t ocl_gpu_device)
{
    /* Only AMD and NVIDIA GPUs are supported for now */
    if ((_OCL_VENDOR_NVIDIA_ == ocl_gpu_device->vendor_e) ||
        (_OCL_VENDOR_AMD_ == ocl_gpu_device->vendor_e))
    {
        return egpuCompatible;
    }

    return egpuIncompatible;
}

/*! \brief Returns an ocl_vendor_id_t value corresponding to the input OpenCL vendor name.
 *
 *  \param[in] vendor_name String with OpenCL vendor name.
 *  \returns               ocl_vendor_id_t value for the input vendor_name
 */
ocl_vendor_id_t get_vendor_id(char *vendor_name)
{
    if (vendor_name)
    {
        if (strstr(vendor_name, "NVIDIA"))
        {
            return _OCL_VENDOR_NVIDIA_;
        }
        else
        if (strstr(vendor_name, "AMD") ||
            strstr(vendor_name, "Advanced Micro Devices"))
        {
            return _OCL_VENDOR_AMD_;
        }
        else
        if (strstr(vendor_name, "Intel"))
        {
            return _OCL_VENDOR_INTEL_;
        }
    }
    return _OCL_VENDOR_UNKNOWN_;
}


/*! \brief Detect all OpenCL GPUs in the system.
 *
 *  Will detect all OpenCL GPUs supported by the device driver in use. Also
 *  check for the compatibility of each and fill the gpu_info->ocl_dev array
 *  with the required information on each the device.
 *  If GMX_OCL_FORCE_CPU environment variable is defined, OpenCL CPU devices will also be detected.
 *
 *  \param[in] gpu_info    Pointer to structure holding GPU information.
 *  \param[out] err_str    The error message of any CUDA API error that caused
 *                         the detection to fail (if there was any). The memory
 *                         the pointer points to should be managed externally.
 *  \returns               non-zero if the detection encountered a failure, zero otherwise.
 */
int detect_ocl_gpus(gmx_gpu_info_t *gpu_info, char *err_str)
{
    int             retval;
    cl_uint         ocl_platform_count;
    cl_platform_id *ocl_platform_ids;
    cl_device_type  req_dev_type = CL_DEVICE_TYPE_GPU;

    retval           = 0;
    ocl_platform_ids = NULL;

    if (getenv("GMX_OCL_FORCE_CPU") != NULL)
    {
        req_dev_type = CL_DEVICE_TYPE_CPU;
    }

    while (1)
    {
        CALLOCLFUNC_LOGERROR(clGetPlatformIDs(0, NULL, &ocl_platform_count), err_str, retval)
        if (0 != retval)
        {
            break;
        }

        if (1 > ocl_platform_count)
        {
            break;
        }

        snew(ocl_platform_ids, ocl_platform_count);

        CALLOCLFUNC_LOGERROR(clGetPlatformIDs(ocl_platform_count, ocl_platform_ids, NULL), err_str, retval)
        if (0 != retval)
        {
            break;
        }

        for (unsigned int i = 0; i < ocl_platform_count; i++)
        {
            cl_uint ocl_device_count;

            CALLOCLFUNC_LOGERROR(clGetDeviceIDs(ocl_platform_ids[i], req_dev_type, 0, NULL, &ocl_device_count), err_str, retval)
            if (0 != retval)
            {
                continue;
            }

            if (1 <= ocl_device_count)
            {
                gpu_info->n_dev += ocl_device_count;
            }
        }

        if (1 > gpu_info->n_dev)
        {
            break;
        }

        snew(gpu_info->gpu_dev, gpu_info->n_dev);

        {
            int           device_index;
            cl_device_id *ocl_device_ids;

            snew(ocl_device_ids, gpu_info->n_dev);
            device_index = 0;

            for (unsigned int i = 0; i < ocl_platform_count; i++)
            {
                cl_uint ocl_device_count;

                CALLOCLFUNC_LOGERROR(clGetDeviceIDs(ocl_platform_ids[i], req_dev_type, gpu_info->n_dev, ocl_device_ids, &ocl_device_count), err_str, retval)
                if (0 != retval)
                {
                    continue;
                }

                if (1 > ocl_device_count)
                {
                    break;
                }

                for (unsigned int j = 0; j < ocl_device_count; j++)
                {
                    gpu_info->gpu_dev[device_index].ocl_gpu_id.ocl_platform_id = ocl_platform_ids[i];
                    gpu_info->gpu_dev[device_index].ocl_gpu_id.ocl_device_id   = ocl_device_ids[j];

                    gpu_info->gpu_dev[device_index].device_name[0] = 0;
                    clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_NAME, sizeof(gpu_info->gpu_dev[device_index].device_name), gpu_info->gpu_dev[device_index].device_name, NULL);

                    gpu_info->gpu_dev[device_index].device_version[0] = 0;
                    clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_VERSION, sizeof(gpu_info->gpu_dev[device_index].device_version), gpu_info->gpu_dev[device_index].device_version, NULL);

                    gpu_info->gpu_dev[device_index].device_vendor[0] = 0;
                    clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_VENDOR, sizeof(gpu_info->gpu_dev[device_index].device_vendor), gpu_info->gpu_dev[device_index].device_vendor, NULL);

                    gpu_info->gpu_dev[device_index].compute_units = 0;
                    clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(gpu_info->gpu_dev[device_index].compute_units), &(gpu_info->gpu_dev[device_index].compute_units), NULL);

                    gpu_info->gpu_dev[device_index].vendor_e = get_vendor_id(gpu_info->gpu_dev[device_index].device_vendor);

                    gpu_info->gpu_dev[device_index].stat = is_gmx_supported_ocl_gpu_id(gpu_info->gpu_dev + device_index);

                    if (egpuCompatible == gpu_info->gpu_dev[device_index].stat)
                    {
                        gpu_info->n_dev_compatible++;
                    }

                    device_index++;
                }
            }

            gpu_info->n_dev = device_index;

            /* Dummy sort of devices -  AMD first, then NVIDIA, then Intel */
            // TODO: Sort devices based on performance.
            if (0 < gpu_info->n_dev)
            {
                int last = -1;
                for (int i = 0; i < gpu_info->n_dev; i++)
                {
                    if (_OCL_VENDOR_AMD_ == gpu_info->gpu_dev[i].vendor_e)
                    {
                        if ((last + 1) < i)
                        {
                            gpu_info_t ocl_gpu_info;
                            ocl_gpu_info = gpu_info->gpu_dev[i];
                            last++;
                            gpu_info->gpu_dev[i]    = gpu_info->gpu_dev[last];
                            gpu_info->gpu_dev[last] = ocl_gpu_info;
                        }
                    }
                }

                /* if more than 1 device left to be sorted */
                if ((gpu_info->n_dev - 1 - last) > 1)
                {
                    for (int i = 0; i < gpu_info->n_dev; i++)
                    {
                        if (_OCL_VENDOR_NVIDIA_ == gpu_info->gpu_dev[i].vendor_e)
                        {
                            if ((last + 1) < i)
                            {
                                gpu_info_t ocl_gpu_info;
                                ocl_gpu_info = gpu_info->gpu_dev[i];
                                last++;
                                gpu_info->gpu_dev[i] = gpu_info->gpu_dev[last];
                                gpu_info->gpu_dev[last] = ocl_gpu_info;
                            }
                        }
                    }
                }
            }

            sfree(ocl_device_ids);
        }

        break;
    }

    sfree(ocl_platform_ids);

    return retval;
}

/*! \brief Frees the ocl_dev array of \gpu_info.
 *
 * \param[in]    gpu_info    pointer to structure holding GPU information
 */
void free_ocl_gpu_info(const gmx_gpu_info_t gmx_unused *gpu_info)
{
    if (gpu_info)
    {
        for (int i = 0; i < gpu_info->n_dev; i++)
        {
            cl_int cl_error;

            if (gpu_info->gpu_dev[i].context)
            {
                cl_error                     = clReleaseContext(gpu_info->gpu_dev[i].context);
                gpu_info->gpu_dev[i].context = NULL;
                assert(CL_SUCCESS == cl_error);
            }

            if (gpu_info->gpu_dev[i].program)
            {
                cl_error                     = clReleaseProgram(gpu_info->gpu_dev[i].program);
                gpu_info->gpu_dev[i].program = NULL;
                assert(CL_SUCCESS == cl_error);
            }
        }

        sfree(gpu_info->gpu_dev);
    }
}

/*! \brief Select the OpenCL GPUs compatible with the native GROMACS acceleration.
 *
 * This function selects the compatible gpus and initializes
 * gpu_info->dev_use and gpu_info->n_dev_use.
 *
 * Given the list of OpenCL GPUs available in the system check each device in
 * gpu_info->ocl_dev and place the indices of the compatible GPUs into
 * dev_use with this marking the respective GPUs as "available for use."
 * Note that \detect_ocl_gpus must have been called before.
 *
 * \param[in]     gpu_info    Pointer to structure holding GPU information
 * \param[in,out] gpu_opt     Pointer to structure holding GPU options
 */
void pick_compatible_ocl_gpus(const gmx_gpu_info_t *gpu_info,
                              gmx_gpu_opt_t        *gpu_opt)
{
    int  i, ncompat;
    int *compat;

    assert(gpu_info);
    /* ocl_dev/nocl_dev have to be either NULL/0 or not (NULL/0) */
    assert((gpu_info->n_dev != 0 ? 0 : 1) ^ (gpu_info->n_dev == NULL ? 0 : 1));

    snew(compat, gpu_info->n_dev);
    ncompat = 0;
    for (i = 0; i < gpu_info->n_dev; i++)
    {
        if (is_compatible_ocl_gpu(gpu_info->gpu_dev[i].stat))
        {
            ncompat++;
            compat[ncompat - 1] = i;
        }
    }

    gpu_opt->n_dev_use = ncompat;
    snew(gpu_opt->dev_use, ncompat);
    memcpy(gpu_opt->dev_use, compat, ncompat*sizeof(*compat));
    sfree(compat);
}

/*! \brief Check the existence/compatibility of a set of OpnCL GPUs specified by their device IDs.
 *
 * Given the a list of gpu->n_dev_use GPU device IDs stored in
 * gpu_opt->dev_use check the existence and compatibility
 * of the respective GPUs. Also provide the caller with an array containing
 * the result of checks in \checkres.
 *
 * \param[out]  checkres    Check result for each ID passed in \requested_devs
 * \param[in]   gpu_info    Pointer to structure holding GPU information
 * \param[out]  gpu_opt     Pointer to structure holding GPU options
 * \returns                 TRUE if each of the requested GPUs are compatible
 */
gmx_bool check_selected_ocl_gpus(int                  *checkres,
                                 const gmx_gpu_info_t *gpu_info,
                                 gmx_gpu_opt_t        *gpu_opt)
{
    int  i, id;
    bool bAllOk;

    assert(checkres);
    assert(gpu_info);
    assert(gpu_opt->n_dev_use >= 0);

    if (gpu_opt->n_dev_use == 0)
    {
        return TRUE;
    }

    assert(gpu_opt->dev_use);

    /* we will assume that all GPUs requested are valid IDs,
       otherwise we'll bail anyways */

    bAllOk = true;
    for (i = 0; i < gpu_opt->n_dev_use; i++)
    {
        id = gpu_opt->dev_use[i];

        /* devices are stored in increasing order of IDs in ocl_dev */
        gpu_opt->dev_use[i] = id;

        checkres[i] = (id >= gpu_info->n_dev) ?
            egpuNonexistent : gpu_info->gpu_dev[id].stat;

        bAllOk = bAllOk && is_compatible_ocl_gpu(checkres[i]);
    }

    return bAllOk;
}

/*! \brief Formats and returns a device information string for a given OpenCL GPU.
 *
 * Given an index *directly* into the array of available GPUs (ocl_dev)
 * returns a formatted info string for the respective GPU which includes
 * name, vendor, device version, number of compute units and detection status.
 *
 * \param[out]  s           Pointer to output string (has to be allocated externally)
 * \param[in]   gpu_info    Pointer to structure holding GPU information
 * \param[in]   index       An index *directly* into the array of available GPUs
 */
void get_ocl_gpu_device_info_string(char gmx_unused *s, const gmx_gpu_info_t gmx_unused *gpu_info, int gmx_unused index)
{
    assert(s);
    assert(gpu_info);

    if (index < 0 && index >= gpu_info->n_dev)
    {
        return;
    }

    gpu_info_t  *dinfo = &gpu_info->gpu_dev[index];

    bool             bGpuExists =
        dinfo->stat == egpuCompatible ||
        dinfo->stat == egpuIncompatible;

    if (!bGpuExists)
    {
        sprintf(s, "#%d: %s, stat: %s",
                index, "N/A",
                gpu_detect_res_str[dinfo->stat]);
    }
    else
    {
        sprintf(s, "#%d: name: %s, vendor: %s device version: %s, comp. units: %d, stat: %s",
                index, dinfo->device_name, dinfo->device_vendor,
                dinfo->device_version,
                dinfo->compute_units, gpu_detect_res_str[dinfo->stat]);
    }
}

/*! \brief Initializes the OpenCL GPU with the given index.
 *
 * Initializes the OpenCL context for the OpenCL GPU with the given index and also
 * compiles the OpenCL kernels.
 *
 * \param[in]  mygpu          Index of the GPU to initialize
 * \param[out] result_str     The message related to the error that occurred
 *                            during the initialization (if there was any).
 * \param[in] gpu_info        GPU info of all detected devices in the system.
 * \param[in] gpu_opt         Options for using the GPUs in gpu_info
 * \param[in] eeltype         Type of electrostatics kernels that will be launched on this device. Ignored if bOclDoFastGen is false.
 * \param[in] vdwtype         Type of Vdw kernels that will be launched on this device. Ignored if bOclDoFastGen is false.
 * \param[in] vdw_modifier    Vdw interaction modifier. Ignored if bOclDoFastGen is false.
 * \param[in] ljpme_comb_rule LJ-PME combination rule. Ignored if bOclDoFastGen is false.
 * \param[in] bOclDoFastGen   If true, only the requested kernels are compiled, significantly reducing
 * the total compilatin time. If false, all OpenCL kernels are compiled.
 * \returns                   true if no error occurs during initialization.
 */
gmx_bool init_ocl_gpu(int gmx_unused                   mygpu,
                      char gmx_unused                 *result_str,
                      const gmx_gpu_info_t gmx_unused *gpu_info,
                      const gmx_gpu_opt_t gmx_unused  *gpu_opt,
                      const int gmx_unused             eeltype,
                      const int gmx_unused             vdwtype,
                      const int gmx_unused             vdw_modifier,
                      const int gmx_unused             ljpme_comb_rule,
                      const gmx_bool gmx_unused        bOclDoFastGen
                      )
{
    cl_context_properties context_properties[3];
    gpu_info_ptr_t        selected_ocl_gpu;
    cl_platform_id        platform_id;
    cl_device_id          device_id;
    cl_context            context;
    cl_program            program;
    cl_int                cl_error;

    gmx_algo_family_t     gmx_algo_family;

    gmx_algo_family.eeltype         = eeltype;
    gmx_algo_family.vdwtype         = vdwtype;
    gmx_algo_family.vdw_modifier    = vdw_modifier;
    gmx_algo_family.ljpme_comb_rule = ljpme_comb_rule;

    int retval;

    assert(gpu_info);
    assert(result_str);

    retval        = -1;
    result_str[0] = 0;

    if (mygpu < 0 || mygpu >= gpu_opt->n_dev_use)
    {
        char        sbuf[STRLEN];
        sprintf(sbuf, "Trying to initialize an inexistent GPU: "
                "there are %d %s-selected GPU(s), but #%d was requested.",
                gpu_opt->n_dev_use, gpu_opt->bUserSet ? "user" : "auto", mygpu);
        gmx_incons(sbuf);
    }

    while (1)
    {
        selected_ocl_gpu = gpu_info->gpu_dev + gpu_opt->dev_use[mygpu];
        platform_id      = selected_ocl_gpu->ocl_gpu_id.ocl_platform_id;
        device_id        = selected_ocl_gpu->ocl_gpu_id.ocl_device_id;

        context_properties[0] = CL_CONTEXT_PLATFORM;
        context_properties[1] = (cl_context_properties)platform_id;
        context_properties[2] = 0;

        context = clCreateContext(context_properties, 1, &device_id, NULL, NULL, &cl_error);
        CALLOCLFUNC_LOGERROR(cl_error, result_str, retval)
        if (0 != retval)
        {
            break;
        }

        cl_error =
            ocl_compile_program(_default_source_,
                                _auto_vendor_kernels_,
                                &gmx_algo_family,
                                bOclDoFastGen,
                                result_str,
                                context,
                                device_id,
                                selected_ocl_gpu->vendor_e,
                                &program
                                );
        if (cl_error != CL_SUCCESS)
        {
            retval = -1;
            break;
        }

        retval = 0;
        break;
    }

    if (0 == retval)
    {
        selected_ocl_gpu->context = context;
        selected_ocl_gpu->program = program;
    }

    return (0 == retval);
}

/*! \brief Returns an identifier for the OpenCL GPU with a given index into the array of used GPUs.
 *
 * Getter function which, given an index into the array of GPUs in use
 * (dev_use) -- typically a tMPI/MPI rank --, returns an identifier of the
 * respective OpenCL GPU.
 *
 * \param[in]    gpu_info   Pointer to structure holding GPU information
 * \param[in]    gpu_opt    Pointer to structure holding GPU options
 * \param[in]    idx        Index into the array of used GPUs
 * \returns                 ocl_gpu_id_t data structure identifying the requested OpenCL GPU
 */
ocl_gpu_id_t get_ocl_gpu_device_id(const gmx_gpu_info_t *gpu_info,
                                   const gmx_gpu_opt_t  *gpu_opt,
                                   int                   idx)
{
    assert(gpu_info);
    assert(gpu_opt);
    assert(idx >= 0 && idx < gpu_opt->n_dev_use);

    return gpu_info->gpu_dev[gpu_opt->dev_use[idx]].ocl_gpu_id;
}

/*! \brief Returns the name for the OpenCL GPU with a given index into the array of used GPUs.
 *
 * Getter function which, given an index into the array of GPUs in use
 * (dev_use) -- typically a tMPI/MPI rank --, returns the device name for the
 * respective OpenCL GPU.
 *
 * \param[in]    gpu_info   Pointer to structure holding GPU information
 * \param[in]    gpu_opt    Pointer to structure holding GPU options
 * \param[in]    idx        Index into the array of used GPUs
 * \returns                 A string with the name of the requested OpenCL GPU
 */
char* get_ocl_gpu_device_name(const gmx_gpu_info_t *gpu_info,
                              const gmx_gpu_opt_t  *gpu_opt,
                              int                   idx)
{
    assert(gpu_info);
    assert(gpu_opt);
    assert(idx >= 0 && idx < gpu_opt->n_dev_use);

    return gpu_info->gpu_dev[gpu_opt->dev_use[idx]].device_name;
}

/*! \brief Returns the size of the ocl_gpu_info_t struct.
 * \returns                 size in bytes of ocl_gpu_info_t
 */
size_t sizeof_ocl_dev_info(void)
{
    return sizeof(gpu_info_t);
}

/*! \brief Allocates nbytes of host memory. Use ocl_free to free memory allocated with this function.
 *
 *  \todo
 *  This function should allocate page-locked memory to help reduce D2H and H2D
 *  transfer times, similar with pmalloc from pmalloc_cuda.cu.
 *
 * \param[in,out]    h_ptr   Pointer where to store the address of the newly allocated buffer.
 * \param[in]        nbytes  Size in bytes of the buffer to be allocated.
 */
void ocl_pmalloc(void **h_ptr, size_t nbytes)
{
    char        strbuf[STRLEN];

#ifndef NDEBUG
    printf("Warning, pmalloc in OpenCL is doing a normal alloc instead of page-locked alloc\n");
#endif

    if (nbytes == 0)
    {
        *h_ptr = NULL;
        return;
    }

    *h_ptr = calloc(1, nbytes);

    if (!*h_ptr)
    {
        sprintf(strbuf, "calloc of size %d bytes failed", (int)nbytes);
        gmx_fatal(FARGS, "%s: %s\n", __FUNCTION__, strbuf);
    }
}

/*! \brief Frees memory allocated with ocl_pmalloc.
 *
 * \param[in]    h_ptr   Buffer allocated with ocl_pmalloc that needs to be freed.
 */
void ocl_pfree(void *h_ptr)
{

#ifndef NDEBUG
    printf("Warning, pfree in OpenCL is not deallocating page-locked memory\n");
#endif

    if (h_ptr)
    {
        free(h_ptr);
        h_ptr = NULL;
    }
    return;
}

/*! \brief Prints the name of a kernel function pointer.
 *
 * \param[in]    kernel   OpenCL kernel
 * \returns               CL_SUCCESS if the operation was successful, an OpenCL error otherwise.
 */
cl_int dbg_ocl_kernel_name(const cl_kernel kernel)
{
    cl_int cl_error = CL_SUCCESS;
    char   kernel_name[256];
    cl_error = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME,
                               sizeof(kernel_name), &kernel_name, NULL);
    if (cl_error)
    {
        printf("No kernel found!\n");
    }
    else
    {
        printf("%s\n", kernel_name);
    }
    return cl_error;
}

/*! \brief Prints the name of a kernel function pointer.
 *
 * \param[in]    kernel   OpenCL kernel
 * \returns               CL_SUCCESS if the operation was successful, an OpenCL error otherwise.
 */
cl_int dbg_ocl_kernel_name_address(void* kernel)
{
    cl_int cl_error = CL_SUCCESS;
    char   kernel_name[256];
    cl_error = clGetKernelInfo((cl_kernel)kernel, CL_KERNEL_FUNCTION_NAME,
                               sizeof(kernel_name), &kernel_name, NULL);
    if (cl_error)
    {
        printf("No kernel found!\n");
    }
    else
    {
        printf("%s\n", kernel_name);
    }
    return cl_error;
}
