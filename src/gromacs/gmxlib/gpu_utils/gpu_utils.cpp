#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <memory.h>
#include <string.h>

#include "types/hw_info.h"
#include "gpu_utils.h"
#include "gromacs/utility/smalloc.h"

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

    retval = 0;
    ocl_platform_ids = NULL;

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

            CALLOCLFUNC_LOGERROR(clGetDeviceIDs(ocl_platform_ids[i], CL_DEVICE_TYPE_GPU, 0, NULL, &ocl_device_count), err_str, retval)
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

                CALLOCLFUNC_LOGERROR(clGetDeviceIDs(ocl_platform_ids[i], CL_DEVICE_TYPE_GPU, gpu_info->nocl_dev, ocl_device_ids, &ocl_device_count), err_str, retval)
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

//#define OCL_FILE_PATH "C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib\\nbnxn_ocl\\nbnxn_ocl_kernels.cl"
//#define OCL_FILE_PATH "C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\gmxlib\\ocl_tools\\vectype_ops.clh"
//#define OCL_FILE_PATH "C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib\\nbnxn_ocl\\nbnxn_ocl_kernel_nvidia.clh"

#if !defined(OCL_INSTALL_DIR_NAME)
#pragma error "OCL_INSTALL_DIR_NAME has not been defined"
#endif

#if defined(__linux__) || (defined(__APPLE__)&&defined(__MACH__))
#define SEPARATOR '/'
#elif defined(_WIN32)
#define SEPARATOR '\\'
#endif 

typedef enum{ 
    _generic_main_kernel_source_ = 0     
} kernel_source_index_t;

/* TODO to structure .. */
static const char*      kernel_sources[]         = {"nbnxn_ocl_kernel_nvidia.clh"};
static const size_t     kernel_sources_lengths[] = {28};
static const int        n_kernel_sources         = 1;

/* TODO comments .. */
static size_t  get_ocl_kernel_source_path_length(kernel_source_index_t kernel_src)
{
    char * filepath=NULL;
    if( (filepath = getenv("OCL_FILE_PATH")) != NULL) return strlen(filepath + 1);
    else
    {
        /* Note we add 1 for the separator and 1 for the termination null char in the resulting string */
        return( kernel_sources_lengths[kernel_src] + strlen(OCL_INSTALL_DIR_NAME) + 2);    
    }
}

/* TODO comments .. */
static void get_ocl_kernel_source_path(
    char * ocl_source_path,
    kernel_source_index_t kernel_src, 
    size_t path_length
)
{
    char *filepath = NULL;   

    assert(path_length != 0);
    assert(ocl_source_path != NULL);        
    
    if( (filepath = getenv("OCL_FILE_PATH")) != NULL)
    {
        FILE *file_ok = NULL;        
        
        //Try to open the file to check that it exists
        file_ok = fopen(filepath,"rb");
        if( file_ok )
        {
            fclose(file_ok);
            strncpy(ocl_source_path, filepath, strlen(filepath));
            ocl_source_path[strlen(filepath)] = '\0';
        }else
        {
            printf("Warning, you seem to have misconfigured the OCL_FILE_PATH environent variable: %s\n",
                filepath);
        }
    }else
    {
        size_t chars_copied = 0;
        strncpy(ocl_source_path, OCL_INSTALL_DIR_NAME, strlen(OCL_INSTALL_DIR_NAME));  
        chars_copied += strlen(OCL_INSTALL_DIR_NAME);
        
        ocl_source_path[chars_copied++] = SEPARATOR;
        
        strncpy(&ocl_source_path[chars_copied], 
                kernel_sources[kernel_src], 
                kernel_sources_lengths[kernel_src]);
        chars_copied += kernel_sources_lengths[kernel_src];
        
        ocl_source_path[chars_copied++] = '\0';
        
        assert(chars_copied == path_length);
        
    }
}

#if defined(__linux__) || (defined(__APPLE__)&&defined(__MACH__))
#undef SEPARATOR
#elif defined(_WIN32)
#undef SEPARATOR
#endif 

char* load_ocl_source(const char* filename, size_t* p_source_length)
{    
    FILE* filestream = NULL;
    char* ocl_source;
    size_t source_length;
    
    source_length = 0;
    filestream = fopen(filename, "rb");
    if(!filestream) 
        return NULL;

    fseek(filestream, 0, SEEK_END); 
    source_length = ftell(filestream);
    fseek(filestream, 0, SEEK_SET); 
    
    ocl_source = (char*)malloc(source_length + 1);     
    if (fread(ocl_source, source_length, 1, filestream) != 1)
    {
        fclose(filestream);
        free(ocl_source);
        return 0;
    }
    
    fclose(filestream);
    ocl_source[source_length] = '\0';

    *p_source_length = source_length;
    return ocl_source;
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

    char* ocl_source;
    char* ocl_source_path;
    
    size_t ocl_source_length;
    size_t path_length;

    assert(gpu_info);
    assert(result_str);

    retval = -1;
    ocl_source = NULL;
    ocl_source_length = 0;
    result_str[0] = 0;

    while (1)
    {
    
        path_length = get_ocl_kernel_source_path_length(_generic_main_kernel_source_);
        if(path_length) ocl_source_path = (char*)alloca(path_length);
    
        get_ocl_kernel_source_path(ocl_source_path, _generic_main_kernel_source_, path_length);                       
        
        ocl_source = load_ocl_source(ocl_source_path, &ocl_source_length);
                
        if (!ocl_source)
        {            
            sprintf(result_str, "Error loading OpenCL code %s",ocl_source_path);
            break;
        }        
        
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

        program = clCreateProgramWithSource(context, 1, (const char**)(&ocl_source), &ocl_source_length, &cl_error);
        CALLOCLFUNC_LOGERROR(cl_error, result_str, retval)
        if (0 != retval)
            break;

        command_queue = clCreateCommandQueue(context, device_id, 0, &cl_error);
        CALLOCLFUNC_LOGERROR(cl_error, result_str, retval)
        if (0 != retval)
            break;

        // Build the program
        {		
            size_t build_log_size       = 0;
            cl_int build_status         = CL_SUCCESS;
            //TODO Extend the compiler to automatically append needed flags
            //-x clc++ for AMD platform to allow function overloading
            const char * build_options  = "-Iopencl -x clc++";
            
            //cl_error = clBuildProgram(program, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
            //cl_error = clBuildProgram(program, 0, NULL, "-I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib\\nbnxn_ocl\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\pbcutil\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\legacyheaders -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib", NULL, NULL);        
            //cl_error = clBuildProgram(program, 0, NULL, "-I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib\\nbnxn_ocl\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\pbcutil\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\legacyheaders\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib\\", NULL, NULL);        
            build_status = clBuildProgram(program, 0, NULL, build_options, NULL, NULL);        
            
            // Do not fail now if the compilation fails. Dump the LOG and then fail.
            CALLOCLFUNC_LOGERROR(build_status, result_str, retval);
            
            // Get log string size
            cl_error = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
            //CALLOCLFUNC_LOGERROR(cl_error, result_str, retval);
            
            if (build_log_size && (cl_error == CL_SUCCESS) )
            {
                char *build_log = NULL;
                
                // Allocate memory to fit the build log - it can be very large in case of errors
                build_log = (char*)malloc(build_log_size);
                if (build_log)
                {
                    cl_error = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, NULL);
                    
                    // Free buildLog buffer
                    free(build_log);
                }
            }
            if (0 != retval)
                break;		    
        }

        {
            //cl_kernel k = clCreateKernel(program, "nbnxn_kernel_ElecCut_VdwLJ_F_prune_opencl", &cl_error);
            char kernel_name[256];
            cl_int num_args;
            
            cl_error = clCreateKernelsInProgram(program, 0, NULL, &num_kernels);

            kernels = (cl_kernel*)malloc(num_kernels * sizeof(cl_kernel));
            cl_error = clCreateKernelsInProgram(program, num_kernels, kernels, NULL);

            for (cl_uint i = 0; i < num_kernels; i++)
            {                
                cl_error = clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME,
                    sizeof(kernel_name), &kernel_name, NULL);

                cl_error = clGetKernelInfo(kernels[i], CL_KERNEL_NUM_ARGS,
                    sizeof(num_args), &num_args, NULL);
            }

            //free(kernels);
        }

        retval = 0;
        break;
    }

    if (0 == retval)
    {
        selected_ocl_gpu->context = context;
        selected_ocl_gpu->command_queue = command_queue;
        selected_ocl_gpu->num_kernels = num_kernels;
        selected_ocl_gpu->kernels = kernels;
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