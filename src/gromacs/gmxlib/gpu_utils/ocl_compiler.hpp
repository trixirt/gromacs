

#ifndef OCL_COMPILER_H
#define OCL_COMPILER_H
#include <CL/cl_platform.h>

typedef enum{ 
    _default_kernel_source_file_ = 0,
    _num_kernels_
} kernel_source_index_t;

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

cl_int
ocl_compile_program(
    kernel_source_index_t kernel_source_file,
    char                 *result_str,
    cl_context            context,
    cl_device_id          device_id,
    char *                ocl_device_vendor,
    cl_program            program
);

#endif /* OCL_COMPILER_H */