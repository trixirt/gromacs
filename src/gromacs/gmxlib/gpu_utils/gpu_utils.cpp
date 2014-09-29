
/*! @file */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <memory.h>
#include <string.h>

#include "types/hw_info.h"
#include "gpu_utils.h"
#include "gromacs/utility/smalloc.h"

#include <CL/opencl.h>

#include "ocl_compiler.hpp"

#define CALLOCLFUNC_LOGERROR(func, err_str, retval) {\
    cl_int opencl_ret = func;\
    if (CL_SUCCESS != opencl_ret)\
    {\
        sprintf(err_str, "OpenCL error %d", opencl_ret);\
        retval = -1;\
    }\
    else\
        retval = 0;\
    }


static int is_gmx_supported_ocl_gpu_id()
{
    // TO DO: add code for this function
    return egpuCompatible;
}

int detect_ocl_gpus(gmx_gpu_info_t *gpu_info, char *err_str)
{
    int retval;    
    cl_uint ocl_platform_count;
    cl_platform_id *ocl_platform_ids;    
    cl_device_type req_dev_type = CL_DEVICE_TYPE_GPU;
    
    retval = 0;
    ocl_platform_ids = NULL;

    if(getenv("OCL_FORCE_CPU")!=NULL)
        req_dev_type = CL_DEVICE_TYPE_CPU;
    
    while (1)
    {
        CALLOCLFUNC_LOGERROR(clGetPlatformIDs(0, NULL, &ocl_platform_count), err_str, retval)
        if (0 != retval)
            break;

        if (1 > ocl_platform_count)
            break;

        snew(ocl_platform_ids, ocl_platform_count);        

        CALLOCLFUNC_LOGERROR(clGetPlatformIDs(ocl_platform_count, ocl_platform_ids, NULL), err_str, retval)
        if (0 != retval)
            break;

        for (unsigned int i = 0; i < ocl_platform_count; i++)
        {            
            cl_uint ocl_device_count;

            CALLOCLFUNC_LOGERROR(clGetDeviceIDs(ocl_platform_ids[i], req_dev_type, 0, NULL, &ocl_device_count), err_str, retval)
            if (0 != retval)
                break;

            if (1 <= ocl_device_count)
                gpu_info->nocl_dev += ocl_device_count;
        }

        if (1 > gpu_info->nocl_dev)
            break;

        snew(gpu_info->ocl_dev, gpu_info->nocl_dev);

        {
            int device_index;
            cl_device_id *ocl_device_ids;

            snew(ocl_device_ids, gpu_info->nocl_dev);
            device_index = 0;

            for (unsigned int i = 0; i < ocl_platform_count; i++)
            {            
                cl_uint ocl_device_count;

                CALLOCLFUNC_LOGERROR(clGetDeviceIDs(ocl_platform_ids[i], req_dev_type, gpu_info->nocl_dev, ocl_device_ids, &ocl_device_count), err_str, retval)
                if (0 != retval)
                    break;

                if (1 > ocl_device_count)
                    break;
                    
                for (unsigned int j = 0; j < ocl_device_count; j++)
                {
                    gpu_info->ocl_dev[device_index].ocl_gpu_id.ocl_platform_id = ocl_platform_ids[i];
                    gpu_info->ocl_dev[device_index].ocl_gpu_id.ocl_device_id = ocl_device_ids[j];
                    
                    gpu_info->ocl_dev[device_index].device_name[0] = 0;
                    clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_NAME, sizeof(gpu_info->ocl_dev[device_index].device_name), gpu_info->ocl_dev[device_index].device_name, NULL);

                    gpu_info->ocl_dev[device_index].device_version[0] = 0;
                    clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_VERSION, sizeof(gpu_info->ocl_dev[device_index].device_version), gpu_info->ocl_dev[device_index].device_version, NULL);

                    gpu_info->ocl_dev[device_index].device_vendor[0] = 0;
                    clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_VENDOR, sizeof(gpu_info->ocl_dev[device_index].device_vendor), gpu_info->ocl_dev[device_index].device_vendor, NULL);

                    gpu_info->ocl_dev[device_index].compute_units = 0;
                    clGetDeviceInfo(ocl_device_ids[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(gpu_info->ocl_dev[device_index].compute_units), &(gpu_info->ocl_dev[device_index].compute_units), NULL);

                    gpu_info->ocl_dev[device_index].stat = is_gmx_supported_ocl_gpu_id();

                    if (egpuCompatible == gpu_info->ocl_dev[device_index].stat)
                        gpu_info->nocl_dev_compatible++;

                    device_index++;
                }
            }

            gpu_info->nocl_dev = device_index;

            // Dummy sort of devices -  to be improved
            // Nvidia first, then AMD, then Intel
            if (0 < gpu_info->nocl_dev)
            {
                int last = -1;
                for (int i = 0; i < gpu_info->nocl_dev; i++)
                    if (strstr(gpu_info->ocl_dev[i].device_vendor, "NVIDIA"))
                        if ((last + 1) < i)
                        {
                            ocl_gpu_info_t ocl_gpu_info;
                            ocl_gpu_info = gpu_info->ocl_dev[i];                            
                            last++;
                            gpu_info->ocl_dev[i] = gpu_info->ocl_dev[last];
                            gpu_info->ocl_dev[last] = ocl_gpu_info;
                        }

                // if more than 1 device left to be sorted
                if ((gpu_info->nocl_dev - 1 - last) > 1)                
                    for (int i = 0; i < gpu_info->nocl_dev; i++)
                        if (strstr(gpu_info->ocl_dev[i].device_vendor, "AMD"))
                            if ((last + 1) < i)
                            {
                                ocl_gpu_info_t ocl_gpu_info;
                                ocl_gpu_info = gpu_info->ocl_dev[i];                            
                                last++;
                                gpu_info->ocl_dev[i] = gpu_info->ocl_dev[last];
                                gpu_info->ocl_dev[last] = ocl_gpu_info;
                            }
            }

            sfree(ocl_device_ids);
        }


        break;
    }
    
    sfree(ocl_platform_ids);    

    return retval;
}


void free_ocl_gpu_info(const gmx_gpu_info_t gmx_unused *gpu_info)
{
    if (gpu_info)
        sfree(gpu_info->ocl_dev);
}


void pick_compatible_ocl_gpus(const gmx_gpu_info_t *gpu_info,
                          gmx_gpu_opt_t        *gpu_opt)
{
    int  i, ncompat;
    int *compat;

    assert(gpu_info);
    /* cuda_dev/ncuda_dev have to be either NULL/0 or not (NULL/0) */
    assert((gpu_info->nocl_dev != 0 ? 0 : 1) ^ (gpu_info->nocl_dev == NULL ? 0 : 1));

    snew(compat, gpu_info->nocl_dev);
    ncompat = 0;
    for (i = 0; i < gpu_info->nocl_dev; i++)
    {
        //if (is_compatible_gpu(gpu_info->cuda_dev[i].stat))
        if (egpuCompatible == gpu_info->ocl_dev[i].stat)
        {
            ncompat++;
            compat[ncompat - 1] = i;
        }
    }

    gpu_opt->nocl_dev_use = ncompat;    
    snew(gpu_opt->ocl_dev_use, ncompat);
    memcpy(gpu_opt->ocl_dev_use, compat, ncompat*sizeof(*compat));
    sfree(compat);
}


void get_ocl_gpu_device_info_string(char gmx_unused *s, const gmx_gpu_info_t gmx_unused *gpu_info, int gmx_unused index)
{
    assert(s);
    assert(gpu_info);

    if (index < 0 && index >= gpu_info->nocl_dev)
    {
        return;
    }

    
    ocl_gpu_info_t *dinfo = &gpu_info->ocl_dev[index];

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

gmx_bool init_ocl_gpu(int gmx_unused mygpu, char gmx_unused *result_str,
                  const gmx_gpu_info_t gmx_unused *gpu_info,
                  const gmx_gpu_opt_t gmx_unused *gpu_opt)
{
    
    cl_context_properties context_properties[3];
    ocl_gpu_info_ptr_t selected_ocl_gpu;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_program program;
    cl_int cl_error;
    cl_command_queue command_queue;
    cl_uint num_kernels;
    cl_kernel *kernels;

    int retval;

    assert(gpu_info);
    assert(result_str);

    retval = -1;
    result_str[0] = 0;

    while (1)
    {        
        
        //gpuid = gpu_info->cuda_dev[gpu_opt->cuda_dev_use[mygpu]].id;
        selected_ocl_gpu = gpu_info->ocl_dev + gpu_opt->ocl_dev_use[mygpu];
        platform_id = selected_ocl_gpu->ocl_gpu_id.ocl_platform_id;
        device_id = selected_ocl_gpu->ocl_gpu_id.ocl_device_id;
        
        context_properties[0] = CL_CONTEXT_PLATFORM;
        context_properties[1] = (cl_context_properties)platform_id;
        context_properties[2] = 0;
        
        context = clCreateContext(context_properties, 1, &device_id, NULL, NULL, &cl_error);
        CALLOCLFUNC_LOGERROR(cl_error, result_str, retval)
        if (0 != retval)
            break;

        command_queue = clCreateCommandQueue(context, device_id, 0, &cl_error);
        CALLOCLFUNC_LOGERROR(cl_error, result_str, retval)
        if (0 != retval)
            break;

        cl_error = 
            ocl_compile_program(_default_kernel_source_,
                                result_str,
                                context,
                                device_id,
                                selected_ocl_gpu->device_vendor,
                                &program
                               );          
        if(cl_error != CL_SUCCESS) {
            retval=-1; 
            break;                      
        }
        
        cl_kernel aux_kernel;
        
        aux_kernel = clCreateKernel(program,"memset_f",&cl_error);
        assert(cl_error == CL_SUCCESS);       
        selected_ocl_gpu->auxiliary_kernels[selected_ocl_gpu->_aux_kernel_memset_f_] = aux_kernel;          
        
        aux_kernel = clCreateKernel(program,"memset_f2",&cl_error);
        assert(cl_error == CL_SUCCESS);       
        selected_ocl_gpu->auxiliary_kernels[selected_ocl_gpu->_aux_kernel_memset_f2_] = aux_kernel;           
        
        aux_kernel = clCreateKernel(program,"memset_f3",&cl_error);
        assert(cl_error == CL_SUCCESS);        
        selected_ocl_gpu->auxiliary_kernels[selected_ocl_gpu->_aux_kernel_memset_f3_] = aux_kernel;        
        
        aux_kernel = clCreateKernel(program,"zero_e_fshift",&cl_error);
        assert(cl_error == CL_SUCCESS);        
        selected_ocl_gpu->auxiliary_kernels[selected_ocl_gpu->_aux_kernel_zero_e_fshift_] = aux_kernel;                       
        
        //{
        //    //cl_kernel k = clCreateKernel(program, "nbnxn_kernel_ElecCut_VdwLJ_F_prune_opencl", &cl_error);
        //    char kernel_name[256];
        //    cl_int num_args;

        //    cl_error = clCreateKernelsInProgram(program, 0, NULL, &num_kernels);

        //    kernels = (cl_kernel*)malloc(num_kernels * sizeof(cl_kernel));
        //    cl_error = clCreateKernelsInProgram(program, num_kernels, kernels, NULL);

        //    for (cl_uint i = 0; i < num_kernels; i++)
        //    {                
        //        cl_error = clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME,
        //            sizeof(kernel_name), &kernel_name, NULL);

        //        cl_error = clGetKernelInfo(kernels[i], CL_KERNEL_NUM_ARGS,
        //            sizeof(num_args), &num_args, NULL);
        //    }

        //    //free(kernels);
        //}

        retval = 0;
        break;
    }

    if (0 == retval)
    {
        selected_ocl_gpu->context = context;
        selected_ocl_gpu->command_queue = command_queue;
        selected_ocl_gpu->program = program;
        //selected_ocl_gpu->num_kernels = num_kernels;
        //selected_ocl_gpu->kernels = kernels;
    }

    return (0 == retval);
}


/*! \brief Returns the device ID of the GPU with a given index into the array of used GPUs.
 *
 * Getter function which, given an index into the array of GPUs in use
 * (cuda_dev_use) -- typically a tMPI/MPI rank --, returns the device ID of the
 * respective CUDA GPU.
 *
 * \param[in]    gpu_info   pointer to structure holding GPU information
 * \param[in]    gpu_opt    pointer to structure holding GPU options
 * \param[in]    idx        index into the array of used GPUs
 * \returns                 device ID of the requested GPU
 */
ocl_gpu_id_t get_ocl_gpu_device_id(const gmx_gpu_info_t *gpu_info,
                      const gmx_gpu_opt_t  *gpu_opt,
                      int                   idx)
{
    assert(gpu_info);
    assert(gpu_opt);
    assert(idx >= 0 && idx < gpu_opt->ncuda_dev_use);

    return gpu_info->ocl_dev[gpu_opt->ocl_dev_use[idx]].ocl_gpu_id;
    //return gpu_info->cuda_dev[gpu_opt->cuda_dev_use[idx]].id;
}

/* Debugger callable function that prints the name of a kernel function pointer */
cl_int dbg_ocl_kernel_name(const cl_kernel kernel)
{     
    cl_int cl_error = CL_SUCCESS;    
    char kernel_name[256];
    cl_error = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME,
                            sizeof(kernel_name), &kernel_name, NULL);                       
    if(cl_error)
    {
        printf("No kernel found!\n",kernel);
    }else{
        printf("%s\n",kernel_name);
    }
    return cl_error;    
}

/* Debugger callable function that prints the name of a kernel function pointer */
cl_int dbg_ocl_kernel_name_address(void* kernel)
{     
    cl_int cl_error = CL_SUCCESS;    
    char kernel_name[256];
    cl_error = clGetKernelInfo((cl_kernel)kernel, CL_KERNEL_FUNCTION_NAME,
                            sizeof(kernel_name), &kernel_name, NULL);                       
    if(cl_error)
    {
        printf("No kernel found!\n",kernel);
    }else{
        printf("%s\n",kernel_name);
    }
    return cl_error;    
}