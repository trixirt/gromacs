

#ifndef OCL_COMPILER_H
#define OCL_COMPILER_H

#include <CL/opencl.h>
#include "../legacyheaders/types/hw_info.h"

typedef enum{
    _generic_vendor_kernels_ = 0, /* Standard (warp-less) source file with generated methods/energy/prune */
    _nvidia_vendor_kernels_     , /* Nvidia source file with generated methods/energy/prune */
    _amd_vendor_kernels_        , /* AMD source file with generated methods/energy/prune */
    _auto_vendor_kernels_         /* Compiler will select source based on vendor id*/
} kernel_vendor_spec_t;

typedef enum{
    _default_source_ = 0  /* The default top-level source  */
} kernel_source_index_t;

typedef enum{
    _invalid_option_          =0,
    _amd_cpp_                   ,
    _nvdia_verbose_ptxas_       ,
    _generic_cl11_        ,
    _generic_cl12_        ,
    _generic_fast_relaxed_math_ ,
    _generic_noopt_compilation_ ,
    _generic_debug_symbols_,
    _include_install_opencl_dir_,
    _include_source_opencl_dirs_,
    _num_build_options_
} build_options_index_t;

cl_int
ocl_compile_program(
    kernel_source_index_t kernel_source_file,
    kernel_vendor_spec_t  kernel_vendor_spec,
    char                * result_str,
    cl_context            context,
    cl_device_id          device_id,
    ocl_vendor_id_t       ocl_device_vendor,
    cl_program          * p_program
);

#endif /* OCL_COMPILER_H */
