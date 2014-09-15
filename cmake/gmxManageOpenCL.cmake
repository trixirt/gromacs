#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2012,2013,2014, by the GROMACS development team, led by
# Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
# and including many others, as listed in the AUTHORS file in the
# top-level source directory and at http://www.gromacs.org.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# http://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at http://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out http://www.gromacs.org.

# If the user did not set GMX_OPENCL we'll consider this option to be
# in "auto" mode meaning that we will:
# - search for CUDA and set GMX_GPU=ON we find it
# - check whether GPUs are present
# - if CUDA is not found but GPUs were detected issue a warning
#if (NOT DEFINED GMX_GPU)
#    set(GMX_GPU_AUTO TRUE CACHE INTERNAL "GPU acceleration will be selected automatically")
#endif()
#option(GMX_GPU "Enable GPU acceleration" OFF)

option(GMX_USE_OPENCL "Enable OpenCL accelerators" OFF)
option(GMX_OPENCL_FORCE_LOCAL_HEADERS "Use the OpenCL headers redistributed with Gromacs" OFF)
option(GMX_OPENCL_FORCE_CL11_API "Try this if you are having compilations issues with OpenCL enabled" OFF)
if(GMX_USE_OPENCL AND GMX_DOUBLE)
    message(FATAL_ERROR "OpenCL not available in double precision - Yet!")
endif()

if(NOT GMX_OPENMP)
	message(WARNING "To use OpenCL GPU acceleration efficiently, mdrun requires OpenMP multi-threading. Without OpenMP a single CPU core can be used with a GPU which is not optimal. Note that with MPI multiple processes can be forced to use a single GPU, but this is typically inefficient. You need to set both C and C++ compilers that support OpenMP (CC and CXX environment variables, respectively) when using GPUs.")
endif()

# detect OpenCL devices in the build host machine
if (GMX_USE_OPENCL AND NOT GMX_OPENCL_DETECTION_DONE)
    include(gmxDetectGpu)
    gmx_detect_OpenCL()
endif()

message(STATUS "gmx_detect_OpenCL set GMX_DETECT_OPENCL_AVAILABLE: " ${GMX_DETECT_OPENCL_AVAILABLE})

#Now configure necessary paths
if (GMX_USE_OPENCL AND GMX_DETECT_OPENCL_AVAILABLE)
    message(STATUS "Configuring OpenCL")
    #Where can OpenCL headers be? and with what priority?
    #1: In system
    #2: In paths indicated by environtment variables
    #3: In standard installation paths (e.g. /opt/AMDAPP, /usr/local/cuda etc..
    #4: In Gromacs    
    if(GMX_OPENCL_FORCE_LOCAL_HEADERS)
        set(OPENCL_INCLUDE_DIRS ../src)
    else()    
        find_path(OPENCL_INCLUDE_DIRS NAMES CL/opencl.h CL/cl.h CL/cl_platform.h CL/cl_ext.h 
        PATHS ../src /usr/local/cuda/include /opt/AMDAPP/include /opt/intel/opencl*/include
        ${CUDA_INC_PATH} ${AMDAPPSDKROOT}/include ${INTELOCLSDKROOT}/include
        )  
    endif()
    
    if(GMX_OPENCL_FORCE_CL11_API)
        set(OPENCL_DEFINITIONS "-DCL_USE_DEPRECATED_OPENCL_1_1_APIS")
    endif(GMX_OPENCL_FORCE_CL11_API)
    
    set(OPENCL_DEFINITIONS "${OPENCL_DEFINITIONS} -Wno-comments")
    
    find_library(OPENCL_LIBRARIES OpenCL)
    
    message(STATUS "OpenCL lib: " ${OPENCL_LIBRARIES} ", PATH: " ${OPENCL_INCLUDE_DIRS} ", DEFINITIONS: " ${OPENCL_DEFINITIONS})
    set(OPENCL_FOUND TRUE)
    
    add_definitions(${OPENCL_DEFINITIONS})
    include_directories(${OPENCL_INCLUDE_DIRS})
    
endif(GMX_USE_OPENCL AND GMX_DETECT_OPENCL_AVAILABLE)

