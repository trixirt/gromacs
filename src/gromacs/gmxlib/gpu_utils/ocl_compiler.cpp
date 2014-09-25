

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "config.h"

#include <CL/opencl.h>

#include "ocl_compiler.hpp"
    
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

const char* build_options_list[] = {
    "",
    "-x clc++",
    "-cl-nv-verbose",
    "-cl-std=CL1.1",
    "-cl-std=CL1.2",
    "-cl-fast-relaxed-math",
    "-cl-opt-disable",
    "-g",
    "-I"OCL_INSTALL_DIR_NAME,
    "-I../../src/gromacs/gmxlib/ocl_tools -I../../src/gromacs/mdlib/nbnxn_ocl -I../../src/gromacs/pbcutil"
};

static const char*      kernel_filenames[]         = {"nbnxn_ocl_kernels.cl"};

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
create_ocl_build_options_length(
    char* build_device_vendor,
    const char * custom_build_options_prepend,
    const char * custom_build_options_append)
{
    size_t build_options_length = 0;
    size_t whitespace = 1;
    
    assert(build_device_vendor != NULL);
    
    if(custom_build_options_prepend)
        build_options_length += 
            strlen(custom_build_options_prepend)+whitespace;
       
    /*if ( !strcmp(
            build_device_vendor,"Advanced Micro Devices, Inc." ) 
       )
        build_options_length += get_ocl_build_option_length(_amd_cpp_)+whitespace;        */

    build_options_length += 
        get_ocl_build_option_length(_generic_noopt_compilation_)+whitespace;    
    
    build_options_length += 
        get_ocl_build_option_length(_generic_debug_symbols_)+whitespace;         
        
    build_options_length += 
        get_ocl_build_option_length(_include_install_opencl_dir_)+whitespace;
        
    build_options_length += 
        get_ocl_build_option_length(_include_source_opencl_dirs_)+whitespace;    
    
    if(custom_build_options_append)
        build_options_length += 
            strlen(custom_build_options_append)+whitespace;    
    
    return build_options_length+1;
}

static void 
create_ocl_build_options(char * build_options_string,
                         size_t build_options_length,
                         char * build_device_vendor,
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
    
    strncpy( build_options_string+char_added, 
             get_ocl_build_option(_generic_noopt_compilation_),
             get_ocl_build_option_length(_generic_noopt_compilation_) );
        
    char_added += get_ocl_build_option_length(_generic_noopt_compilation_);        
    build_options_string[char_added++]=' ';    
    
    strncpy( build_options_string+char_added, 
             get_ocl_build_option(_generic_debug_symbols_),
             get_ocl_build_option_length(_generic_debug_symbols_) );
        
    char_added += get_ocl_build_option_length(_generic_debug_symbols_);        
    build_options_string[char_added++]=' ';        
    
    /*if (!strcmp(build_device_vendor,"Advanced Micro Devices, Inc.") )
    {
        strncpy( build_options_string+char_added, 
                 get_ocl_build_option(_amd_cpp_),
                 get_ocl_build_option_length(_amd_cpp_) );
        
        char_added += get_ocl_build_option_length(_amd_cpp_);        
        build_options_string[char_added++]=' ';
    }*/
    
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

/* TODO comments .. */
static size_t  get_ocl_kernel_source_file_info(kernel_source_index_t kernel_src_id)
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
    kernel_source_index_t kernel_src_id, 
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
                          kernel_source_index_t kernel_filename_id)
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
        char *log_fname;
        
        build_info = (char*)malloc(32 + strlen(build_options_string) );
        sprintf(build_info,"-- Used build options: %s\n", build_options_string);
    
        if(dumpFile)
        {
            strncpy(status_suffix, (build_status==CL_SUCCESS)?"SUCCEEDED":"FAILED",10);        
    
            log_fname = (char*)malloc(strlen(kernel_filenames[kernel_filename_id]) 
                                     + strlen(status_suffix) + 2
                                   );
            
            sprintf(log_fname,"%s.%s",kernel_filenames[kernel_filename_id],status_suffix);       
            build_log_file = fopen(log_fname,"w");       
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
            
            printf("The OpenCL compilation log has been saved in \"%s\"\n",log_fname);
            
            free(log_fname);
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

cl_int 
ocl_compile_program(
    kernel_source_index_t kernel_source_file,
    char                * result_str,
    cl_context            context,
    cl_device_id          device_id,
    char *                ocl_device_vendor,
    cl_program          * p_program
)
{
    cl_int cl_error     = CL_SUCCESS;
    
    char* ocl_source        = NULL;
    char* kernel_filename   = NULL;
    
    size_t ocl_source_length    = 0;
    size_t kernel_filename_len  = 0;
        
    kernel_filename_len = get_ocl_kernel_source_file_info(kernel_source_file);
    if(kernel_filename_len) kernel_filename = (char*)malloc(kernel_filename_len);
    
    get_ocl_kernel_source_path(kernel_filename, kernel_source_file, kernel_filename_len);                       
    
    ocl_source = load_ocl_source(kernel_filename, &kernel_filename_len);
        
    if (!ocl_source)
    {            
        sprintf(result_str, "Error loading OpenCL code %s",kernel_filename);
        return CL_BUILD_PROGRAM_FAILURE;
    }        
    
    free(kernel_filename);    
    
    *p_program = clCreateProgramWithSource(context, 1, (const char**)(&ocl_source), &ocl_source_length, &cl_error);
    //CALLOCLFUNC_LOGERROR(cl_error, result_str, retval)
    //if (0 != retval)
    //    break;    
    
    // Build the program
    cl_int build_status         = CL_SUCCESS;    
    {
        
        // Anca, how to add manually the includes:
        // const char * custom_build_options_prepend = 
        //      "-I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib\\nbnxn_ocl\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\pbcutil\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\legacyheaders\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib\\"
        // 
        
        size_t build_options_length = 
        create_ocl_build_options_length(ocl_device_vendor,NULL,NULL);
        
        char * build_options_string = (char *)malloc(build_options_length);
        
        create_ocl_build_options(build_options_string,
                                 build_options_length,
                                 ocl_device_vendor,NULL,NULL);
        
        size_t build_log_size       = 0;
        
        //cl_error = clBuildProgram(program, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
        //cl_error = clBuildProgram(program, 0, NULL, "-I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib\\nbnxn_ocl\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\pbcutil\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\legacyheaders -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib", NULL, NULL);        
        //cl_error = clBuildProgram(program, 0, NULL, "-I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib\\nbnxn_ocl\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\pbcutil\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\legacyheaders\\ -I C:\\Anca\\SC\\gromacs\\gromacs\\src\\gromacs\\mdlib\\", NULL, NULL);        
        build_status = clBuildProgram(*p_program, 0, NULL, build_options_string, NULL, NULL);        
        
        // Do not fail now if the compilation fails. Dump the LOG and then fail.
        //CALLOCLFUNC_LOGERROR(build_status, result_str, retval);
        
        // Get log string size
        cl_error = clGetProgramBuildInfo(*p_program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
        //CALLOCLFUNC_LOGERROR(cl_error, result_str, retval);
        
        if (build_log_size && (cl_error == CL_SUCCESS) )
        {
            char *build_log = NULL;
            
            // Allocate memory to fit the build log - it can be very large in case of errors
            build_log = (char*)malloc(build_log_size);
            if (build_log)
            {
                cl_error = clGetProgramBuildInfo(*p_program, device_id, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, NULL);                    
                
                if(!cl_error)
                {
                    handle_ocl_build_log(build_log,                                          
                                         build_options_string,
                                         build_status,
                                         kernel_source_file
                    );
                }
                
                // Free buildLog buffer
                free(build_log);
            }
        }
        free(build_options_string);
    }

    return build_status | cl_error;
        
}