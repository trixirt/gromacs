
/*! @file */

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

/* This path is defined by CMake and it depends on the install prefix option.
   The opencl kernels are installed in bin/opencl.*/
#if !defined(OCL_INSTALL_DIR_NAME)
#pragma error "OCL_INSTALL_DIR_NAME has not been defined"
#endif

#if defined(__linux__) || (defined(__APPLE__)&&defined(__MACH__))
#define SEPARATOR '/'
#elif defined(_WIN32)
#define SEPARATOR '\\'
#endif 

typedef enum{
    _invalid_option_          =0,
    _amd_cpp_                   ,
    _nvdia_verbose_ptxas_       ,
    _generic_cl11_        ,
    _generic_cl12_        ,
    _generic_fast_relaxed_math_ ,
    _generic_debug_compilation_ ,
    _include_install_opencl_dir_,
    _include_source_opencl_dirs_,
    _num_build_options_
} build_options_index_t;

#define BUILD_INC_PATH(PATH) "-I"PATH

static const char* build_options_list[] = {
    "",
    "-x clc++",
    "-cl-nv-verbose",
    "-cl-std=CL1.1",
    "-cl-std=CL1.2",
    "-cl-fast-relaxed-math",
    "-cl-opt-disable",
    "-I"OCL_INSTALL_DIR_NAME,
    "-I../../src/gromacs/gmxlib/ocl_tools -I../../src/gromacs/mdlib/nbnxn_ocl -I../../src/gromacs/pbcutil"
};
    

static const char* get_ocl_build_option(build_options_index_t build_option_id)
{
    if(build_option_id<_num_build_options_)
        return build_options_list[build_option_id];
    else
        return build_options_list[_invalid_option_];
}

static size_t get_ocl_build_option_length(build_options_index_t build_option_id)
{
    if(build_option_id<_num_build_options_)
        return strlen(build_options_list[build_option_id]);
    else
        return strlen(build_options_list[_invalid_option_]);
}

static size_t
create_ocl_build_options_length(ocl_gpu_info_ptr_t ocl_gpu,
                         const char * custom_build_options_prepend,
                         const char * custom_build_options_append)
{
    size_t build_options_length = 0;
    size_t whitespace = 1;
    
    if(custom_build_options_prepend)
        build_options_length += strlen(custom_build_options_prepend)+whitespace;
    
    if (!strcmp(ocl_gpu->device_vendor,"Advanced Micro Devices, Inc.") )
        build_options_length += get_ocl_build_option_length(_amd_cpp_)+whitespace;        

    build_options_length += get_ocl_build_option_length(_include_install_opencl_dir_)+whitespace;
    build_options_length += get_ocl_build_option_length(_include_source_opencl_dirs_)+whitespace;    
    
    if(custom_build_options_append)
        build_options_length += strlen(custom_build_options_append)+whitespace;    
    
    return build_options_length+1;
}

static void 
create_ocl_build_options(char * build_options_string,
                         size_t build_options_length,
                         ocl_gpu_info_ptr_t ocl_gpu,
                         const char * custom_build_options_prepend,
                         const char * custom_build_options_append)
{
    //cl_device_type device_type;
    //clGetDeviceInfo(ocl_gpu->ocl_gpu_id.ocl_device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);  
    size_t char_added=0;
    
    if(custom_build_options_prepend)
    {
        strncpy( build_options_string+char_added, 
                 custom_build_options_prepend, 
                 strlen(custom_build_options_prepend));
        
        char_added += strlen(custom_build_options_prepend);
        build_options_string[char_added++] =' ';
    }    
    
    if (!strcmp(ocl_gpu->device_vendor,"Advanced Micro Devices, Inc.") )
    {
        strncpy( build_options_string+char_added, 
                 get_ocl_build_option(_amd_cpp_),
                 get_ocl_build_option_length(_amd_cpp_) );
        
        char_added += get_ocl_build_option_length(_amd_cpp_);        
        build_options_string[char_added++]=' ';
    }
    
    strncpy( build_options_string+char_added,
             get_ocl_build_option(_include_install_opencl_dir_),
             get_ocl_build_option_length(_include_install_opencl_dir_)
    );
    char_added += get_ocl_build_option_length(_include_install_opencl_dir_);
    build_options_string[char_added++]=' ';
    
    strncpy( build_options_string+char_added,
             get_ocl_build_option(_include_source_opencl_dirs_),
             get_ocl_build_option_length(_include_source_opencl_dirs_)
    );
    char_added += get_ocl_build_option_length(_include_source_opencl_dirs_);
    build_options_string[char_added++]=' ';    
    
    if(custom_build_options_append)
    {
        strncpy( build_options_string+char_added, 
                 custom_build_options_append, 
                 strlen(custom_build_options_append) );
        
        char_added += strlen(custom_build_options_append);         
        build_options_string[char_added++]=' ';        
    }
    
    build_options_string[char_added++] = '\0';
    
    assert(char_added == build_options_length);
    
}

typedef enum{ 
    _generic_main_kernel_ = 0,
    _num_kernels_
} kernel_filename_index_t;

/* TODO to structure .. */
static const char*      kernel_filenames[]         = {"nbnxn_ocl_kernel_nvidia.clh"};

/* TODO comments .. */
static size_t  get_ocl_kernel_source_file_info(kernel_filename_index_t kernel_src_id)
{
    char * kernel_filename=NULL;
    if( (kernel_filename = getenv("OCL_FILE_PATH")) != NULL) return (strlen(kernel_filename) + 1);
    else
    {
        /* Note we add 1 for the separator and 1 for the termination null char in the resulting string */
        return( strlen(kernel_filenames[kernel_src_id]) + strlen(OCL_INSTALL_DIR_NAME) + 2);    
    }
}

/* TODO comments .. */
static void get_ocl_kernel_source_path(
    char * ocl_kernel_filename,
    kernel_filename_index_t kernel_src_id, 
    size_t kernel_filename_len
)
{
    char *filename = NULL;   

    assert(kernel_filename_len != 0);
    assert(ocl_kernel_filename != NULL);        
    
    if( (filename = getenv("OCL_FILE_PATH")) != NULL)
    {
        FILE *file_ok = NULL;        
        
        //Try to open the file to check that it exists
        file_ok = fopen(filename,"rb");
        if( file_ok )
        {
            fclose(file_ok);
            strncpy(ocl_kernel_filename, filename, strlen(filename));
            ocl_kernel_filename[strlen(filename)] = '\0';
        }else
        {
            printf("Warning, you seem to have misconfigured the OCL_FILE_PATH environent variable: %s\n",
                filename);
        }
    }else
    {
        size_t chars_copied = 0;
        strncpy(ocl_kernel_filename, OCL_INSTALL_DIR_NAME, strlen(OCL_INSTALL_DIR_NAME));  
        chars_copied += strlen(OCL_INSTALL_DIR_NAME);
        
        ocl_kernel_filename[chars_copied++] = SEPARATOR;
        
        strncpy(&ocl_kernel_filename[chars_copied], 
                kernel_filenames[kernel_src_id], 
                strlen(kernel_filenames[kernel_src_id]) );
        chars_copied += strlen(kernel_filenames[kernel_src_id]);
        
        ocl_kernel_filename[chars_copied++] = '\0';
        
        assert(chars_copied == kernel_filename_len);
        
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

void handle_ocl_build_log(const char*   build_log,
                          const char*   build_options_string,
                          cl_int        build_status,
                          kernel_filename_index_t kernel_filename_id)
{
    bool dumpFile  = false;
    bool dumpStdErr= false;
#ifdef DNDEBUG
    if(build_status != CL_SUCCESS) dumpFile = true;
#else
    dumpFile = true;
    if(build_status != CL_SUCCESS) dumpStdErr = true;
#endif          
    
    if(dumpFile || dumpStdErr)
    {
        FILE * build_log_file       = NULL;   
        const char *fail_header     = "Compilation of source file failed! \n";
        const char *success_header  = "Compilation of source file was successful! \n";
        const char *log_header      = "--------------LOG START---------------\n";
        const char *log_footer      = "---------------LOG END----------------\n"; 
        char status_suffix[10];
        char *build_info;
        
        build_info = (char*)malloc(32 + strlen(build_options_string) );
        sprintf(build_info,"-- Used build options: %s\n", build_options_string);
    
        if(dumpFile)
        {
            strncpy(status_suffix, (build_status==CL_SUCCESS)?"SUCCEEDED":"FAILED",10);        
    
            char *log_fname = (char*)malloc(strlen(kernel_filenames[kernel_filename_id]) 
                                     + strlen(status_suffix) + 2
                                   );
            
            sprintf(log_fname,"%s.%s",kernel_filenames[kernel_filename_id],status_suffix);       
            build_log_file = fopen(log_fname,"w");       
            free(log_fname);
        }
    
        size_t complete_message_size = 0;
        char * complete_message;

    
        complete_message_size =  (build_status == CL_SUCCESS)?strlen(success_header):strlen(fail_header);
        complete_message_size += strlen(build_info) + strlen(log_header) + strlen(log_footer);
        complete_message_size += strlen(build_log);
        complete_message_size += 1; //null termination
        complete_message = (char*)malloc(complete_message_size);         
    
        sprintf(complete_message,"%s%s%s%s%s",
            (build_status == CL_SUCCESS)?success_header:fail_header,
            build_info,
            log_header,            
            build_log,
            log_footer);
    
        if(dumpFile)
        {
            if(build_log_file)
                fprintf(build_log_file, "%s" , complete_message);

            fclose(build_log_file);
        }
        if(dumpStdErr)
        {
            if(build_status != CL_SUCCESS)
                fprintf(stderr, "%s", complete_message);
        }     
        free(complete_message);
        free(build_info);
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

    char* ocl_source;
    char* kernel_filename;
    
    size_t ocl_source_length;
    size_t kernel_filename_len;

    assert(gpu_info);
    assert(result_str);

    retval = -1;
    ocl_source = NULL;
    ocl_source_length = 0;
    result_str[0] = 0;

    while (1)
    {
    
        kernel_filename_len = get_ocl_kernel_source_file_info(_generic_main_kernel_);
        if(kernel_filename_len) kernel_filename = (char*)alloca(kernel_filename_len);
    
        get_ocl_kernel_source_path(kernel_filename, _generic_main_kernel_, kernel_filename_len);                       
        
        ocl_source = load_ocl_source(kernel_filename, &kernel_filename_len);
                
        if (!ocl_source)
        {            
            sprintf(result_str, "Error loading OpenCL code %s",kernel_filename);
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
            
            // Anca, how to add manually the includes:
            // const char * custom_build_options_prepend = 
            //      "-I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib\\nbnxn_ocl\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\pbcutil\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\legacyheaders\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib\\"
            // 
            
            size_t build_options_length = 
                create_ocl_build_options_length(selected_ocl_gpu,NULL,NULL);
             
            char * build_options_string = (char *)alloca(build_options_length);
            
            create_ocl_build_options(build_options_string,
                                     build_options_length,
                                     selected_ocl_gpu,NULL,NULL);
          
            size_t build_log_size       = 0;
            cl_int build_status         = CL_SUCCESS;
            
            //cl_error = clBuildProgram(program, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
            //cl_error = clBuildProgram(program, 0, NULL, "-I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib\\nbnxn_ocl\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\pbcutil\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\legacyheaders -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib", NULL, NULL);        
            //cl_error = clBuildProgram(program, 0, NULL, "-I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib\\nbnxn_ocl\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\pbcutil\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\legacyheaders\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib\\", NULL, NULL);        
            build_status = clBuildProgram(program, 0, NULL, build_options_string, NULL, NULL);        
            
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
                    
                    if(!cl_error)
                    {
                        handle_ocl_build_log(build_log,                                          
                                             build_options_string,
                                             build_status,
                                             _generic_main_kernel_
                                            );
                    }
                    
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