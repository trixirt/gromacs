

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "config.h"

#include <CL/opencl.h>

#include "ocl_compiler.hpp"
#include "../mdlib/nbnxn_ocl/nbnxn_ocl_types.h"

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
    "-I../../src/gromacs/gmxlib/ocl_tools           -I../../src/gromacs/mdlib/nbnxn_ocl            -I../../src/gromacs/pbcutil            -I../../src/gromacs/mdlib \
    -I../../../gromacs/src/gromacs/gmxlib/ocl_tools -I../../../gromacs/src/gromacs/mdlib/nbnxn_ocl -I../../../gromacs/src/gromacs/pbcutil -I../../../gromacs/src/gromacs/mdlib"
};

/* Available sources */
static const char * kernel_filenames[]         = {"nbnxn_ocl_kernels.cl"};

/* Defines to enable specific kernels based on vendor */
static const char * kernel_vendor_spec_definitions[] = {"-D_WARPLESS_SOURCE_", "-D_NVIDIA_SOURCE_", "-D_AMD_SOURCE_"};

typedef enum eelOcl  eelOcl_t;
static const char * kernel_electrostatic_family_definitions[] =
    {"-DEL_CUTOFF -D_EELNAME=_ElecCut",
     "-DEL_RF -D_EELNAME=_ElecRF",
     "-DEL_EWALD_ANA -D_EELNAME=_ElecEw",
     "-DEL_EWALD_ANA -DLJ_CUTOFF_CHECK -D_EELNAME=_ElecEwTwinCut",
     "-DEL_EWALD_TAB -D_EELNAME=_ElecEwQSTab",
     "-DEL_EWALD_TAB -DLJ_CUTOFF_CHECK -D_EELNAME=_ElecEwQSTabTwinCut"};

typedef enum evdwOcl evdwOcl_t;
static const char * kernel_VdW_family_definitions[] =
    {"-D_VDWNAME=_VdwLJ",
     "-DLJ_EWALD_COMB_GEOM -D_VDWNAME=_VdwLJEwCombGeom",
     "-DLJ_EWALD_COMB_LB -D_VDWNAME=_VdwLJEwCombLB",
     "-DLJ_FORCE_SWITCH -D_VDWNAME=_VdwLJFsw",
     "-DLJ_POT_SWITCH -D_VDWNAME=_VdwLJPsw"};


/**
 * \brief Get the string of a build option of the specific id
 * \param  build_option_id  The option id as defines in the header
 * \return String containing the actual build option string for the compiler
 */
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
    ocl_vendor_id_t vendor_id,
    const char *    custom_build_options_prepend,
    const char *    custom_build_options_append)
{
    size_t build_options_length = 0;
    size_t whitespace = 1;

    assert(vendor_id <= _OCL_VENDOR_UNKNOWN_);

    if(custom_build_options_prepend)
        build_options_length +=
            strlen(custom_build_options_prepend)+whitespace;

    if ( (vendor_id == _OCL_VENDOR_AMD_) && getenv("OCL_DEBUG") && getenv("OCL_FORCE_CPU") )
    {
        build_options_length += get_ocl_build_option_length(_generic_debug_symbols_)+whitespace;
    }

    if(getenv("OCL_NOOPT"))
    {
        build_options_length +=
            get_ocl_build_option_length(_generic_noopt_compilation_)+whitespace;
    }
    if(getenv("OCL_FASTMATH"))
    {
       build_options_length +=
            get_ocl_build_option_length(_generic_fast_relaxed_math_)+whitespace      ;
    }

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
create_ocl_build_options(char *             build_options_string,
                         size_t             build_options_length,
                         ocl_vendor_id_t    build_device_vendor_id,
                         const char *       custom_build_options_prepend,
                         const char *       custom_build_options_append)
{
    size_t char_added=0;

    if(custom_build_options_prepend)
    {
        strncpy( build_options_string+char_added,
                 custom_build_options_prepend,
                 strlen(custom_build_options_prepend));

        char_added += strlen(custom_build_options_prepend);
        build_options_string[char_added++] =' ';
    }

    if(getenv("OCL_NOOPT") )
    {
        strncpy( build_options_string+char_added,
                get_ocl_build_option(_generic_noopt_compilation_),
                get_ocl_build_option_length(_generic_noopt_compilation_) );

        char_added += get_ocl_build_option_length(_generic_noopt_compilation_);
        build_options_string[char_added++]=' ';

    }

    if(getenv("OCL_FASTMATH") )
    {
        strncpy( build_options_string+char_added,
                get_ocl_build_option(_generic_fast_relaxed_math_),
                get_ocl_build_option_length(_generic_fast_relaxed_math_) );

        char_added += get_ocl_build_option_length(_generic_fast_relaxed_math_);
        build_options_string[char_added++]=' ';
    }

    if ( ( build_device_vendor_id == _OCL_VENDOR_AMD_ ) && getenv("OCL_DEBUG") && getenv("OCL_FORCE_CPU"))
    {
        strncpy( build_options_string+char_added,
                get_ocl_build_option(_generic_debug_symbols_),
                get_ocl_build_option_length(_generic_debug_symbols_) );

        char_added += get_ocl_build_option_length(_generic_debug_symbols_);
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
#ifdef NDEBUG
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

/**
 *  \brief Get the warp size reported by device
 *  This is platform implementation dependant and seems to only work on the Nvidia and Amd platforms!
 *  Nvidia reports 32, Amd for GPU 64. Ignore the rest
 *  \param  context   Current OpenCL context
 *  \param  device_id OpenCL device with the context
 *  \return cl_int value of the warp size
 */
static cl_int ocl_get_warp_size(cl_context context, cl_device_id device_id)
{
    cl_int cl_error = CL_SUCCESS;
    size_t warp_size = 0;
    const char *dummy_kernel="__kernel void test(__global int* test){test[get_local_id(0)] = 0;}";

    cl_program program =
        clCreateProgramWithSource(context, 1, (const char**)&dummy_kernel, NULL, &cl_error);

    cl_error =
        clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program,"test",&cl_error);

    cl_error = clGetKernelWorkGroupInfo(kernel,device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                    sizeof(size_t), &warp_size, NULL);

    clReleaseKernel(kernel);
    clReleaseProgram(program);

    assert(warp_size!=0);
    assert(cl_error==CL_SUCCESS);
    return warp_size;

}

/**
 * \brief Automatically select vendor-specific kernel from vendor id
 * \param ocl_vendor_id_t Vendor id enumerator (amd,nvidia,intel,unknown)
 * \return Vendor-specific kernel version
 */
static kernel_vendor_spec_t ocl_autoselect_kernel_from_vendor(ocl_vendor_id_t vendor_id)
{
    kernel_vendor_spec_t kernel_vendor;
    printf("Selecting kernel source automatically\n");
    switch(vendor_id)
    {
        case _OCL_VENDOR_AMD_:
            kernel_vendor = _amd_vendor_kernels_;
            printf("Selecting kernel for AMD\n");
            break;
        case _OCL_VENDOR_NVIDIA_:
            kernel_vendor = _nvidia_vendor_kernels_;
            printf("Selecting kernel for Nvidia\n");
            break;
        default:
            kernel_vendor = _generic_vendor_kernels_;
            printf("Selecting generic kernel\n");
            break;
    }
    return kernel_vendor;
}

/**
 * \brief Returns the compiler define string needed to activate vendor-specific kernels
 * \param kernel_spec Kernel vendor specification
 * \return String with the define for the spec
 */
static const char * ocl_get_vendor_specific_define(kernel_vendor_spec_t kernel_spec)
{
    assert(kernel_spec < _auto_vendor_kernels_ );
    printf("Setting up kernel vendor spec definitions:  %s \n",kernel_vendor_spec_definitions[kernel_spec]);
    return kernel_vendor_spec_definitions[kernel_spec];
}

/**
 * \brief Populates algo_defines with the compiler defines required to avoid all flavor generation
 * For example if flavor eelOclRF with evdwOclFSWITCH, the output will be such that the corresponding
 * kernel flavor is generated:
 * -D_OCL_FASTGEN_ (will replace flavor generator kernels.clh with the fastgen one kernels_fastgen.clh)
 * -DEL_RF             (The eelOclRF flavor)
 * -D_EELNAME=_ElecRF  (The first part of the generated kernel name )
 * -DLJ_EWALD_COMB_GEOM (The evdwOclFSWITCH flavor)
 * -D_VDWNAME=_VdwLJEwCombGeom (The second part of the generated kernel name )
 * prune/energy are still generated as originally. It is only the the flavor-level that has changed, so that
 * only the required flavor for the simulation is compiled.
 * \param p_kernel_algo_family Pointer to algo_family structure (eel,vdw)
 * \param p_algo_defines       String to populate with the defines
 */
static void ocl_get_fastgen_define(kernel_algo_family_t * p_kernel_algo_family, char * p_algo_defines)
{
    printf("Setting up kernel fastgen definitions: ");
    sprintf(p_algo_defines,"-D_OCL_FASTGEN_ %s %s ",
            kernel_electrostatic_family_definitions[p_kernel_algo_family->eeltype],
            kernel_VdW_family_definitions[p_kernel_algo_family->vdwtype]
    );
    printf(" %s \n",p_algo_defines);
}

cl_int
ocl_compile_program(
    kernel_source_index_t       kernel_source_file,
    kernel_vendor_spec_t        kernel_vendor_spec,
    kernel_algo_family_t *      p_kernel_algo_family,
    int                         DoFastGen,
    char *                      result_str,
    cl_context                  context,
    cl_device_id                device_id,
    ocl_vendor_id_t             ocl_device_vendor,
    cl_program *                p_program
)
{
    cl_int cl_error     = CL_SUCCESS;
    cl_int warp_size    = 0;

    warp_size = ocl_get_warp_size(context,device_id);

    char* ocl_source        = NULL;
    char* kernel_filename   = NULL;

    size_t ocl_source_length    = 0;
    size_t kernel_filename_len  = 0;

    if ( kernel_vendor_spec == _auto_vendor_kernels_)
        kernel_vendor_spec = ocl_autoselect_kernel_from_vendor(ocl_device_vendor);

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

    // Build the program
    cl_int build_status         = CL_SUCCESS;
    {
        char custom_build_options_prepend[512] = {0};

        const char * kernel_vendor_spec_define =
            ocl_get_vendor_specific_define(kernel_vendor_spec);

        char kernel_fastgen_define[128] = {0};
        if(DoFastGen)
            ocl_get_fastgen_define(p_kernel_algo_family, kernel_fastgen_define);

        sprintf(custom_build_options_prepend, "-DWARP_SIZE_TEST=%d %s %s", warp_size, kernel_vendor_spec_define, kernel_fastgen_define);

        size_t build_options_length =
                create_ocl_build_options_length(ocl_device_vendor,custom_build_options_prepend,NULL);

        char * build_options_string = (char *)malloc(build_options_length);

        create_ocl_build_options(build_options_string,
                                 build_options_length,
                                 ocl_device_vendor,
                                 custom_build_options_prepend,NULL);

        size_t build_log_size       = 0;
        
        build_status = clBuildProgram(*p_program, 0, NULL, build_options_string, NULL, NULL);

        // Get log string size
        cl_error = clGetProgramBuildInfo(*p_program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);

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

