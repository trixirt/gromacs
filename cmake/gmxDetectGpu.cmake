#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2012, by the GROMACS development team, led by
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

# The gmx_detect_gpu() macro aims to detect GPUs available in the build machine
# and provide the number, names, and compute-capabilities of these devices.
#
# The current version is limited to checking the availability of NVIDIA GPUs
# without compute-capability information.
#
# The current detection relies on the following checks in the order of listing:
# - output of nvidia-smi (if available);
# - presence and content of of /proc/driver/nvidia/gpus/*/information (Linux)
# - output of lspci (Linux)
#
# If any of the checks succeeds in finding devices, consecutive checks will not
# be carried out. Additionally, when lspci is used and a device with unknown
# PCI ID is encountered, lspci tries to check the online PCI ID database. If
# this is not possible or the device is simply not recognized, no device names
# will be available.
#
# The following advanced variables are defined:
# - GMX_DETECT_GPU_AVAILABLE - TRUE if any GPUs were detected, otherwise FALSE
# - GMX_DETECT_GPU_COUNT     - # of GPUs detected
# - GMX_DETECT_GPU_INFO      - list of information strings of the detected GPUs
#
# NOTE: The proper solution is to detect hardware compatible with the native
# GPU acceleration. However, this requires checking the compute capability
# of the device which is not possible with the current checks and requires
# interfacing with the CUDA driver API.
#

# check whether the number of GPUs machetes the number of elements in the GPU info list
macro(check_num_gpu_info NGPU GPU_INFO)
    list(LENGTH ${GPU_INFO} _len)
    if (NOT NGPU EQUAL _len)
        list(APPEND ${GMX_DETECT_GPU_INFO} "NOTE: information about some GPU(s) missing!")
    endif()
endmacro()

macro(gmx_detect_gpu)

    if (NOT DEFINED GMX_DETECT_GPU_COUNT OR NOT DEFINED GMX_DETECT_GPU_INFO)

        set(GMX_DETECT_GPU_COUNT 0)
        set(GMX_DETECT_GPU_INFO  "")

        message(STATUS "Looking for NVIDIA GPUs present in the system")

        # nvidia-smi-based detection.
        # Requires the nvidia-smi tool to be installed and available in the path
        # or in one of the default search locations
        if (NOT DEFINED GMX_DETECT_GPU_COUNT_NVIDIA_SMI)
            # try to find the nvidia-smi binary
            # TODO add location hints
            find_program(_nvidia_smi "nvidia-smi")
            if (_nvidia_smi)
                set(GMX_DETECT_GPU_COUNT_NVIDIA_SMI 0)
                # execute nvidia-smi -L to get a short list of GPUs available
                exec_program(${_nvidia_smi_path} ARGS -L
                    OUTPUT_VARIABLE _nvidia_smi_out
                    RETURN_VALUE    _nvidia_smi_ret)
                # process the stdout of nvidia-smi
                if (_nvidia_smi_ret EQUAL 0)
                    # convert string with newlines to list of strings
                    string(REGEX REPLACE "\n" ";" _nvidia_smi_out "${_nvidia_smi_out}")
                    foreach(_line ${_nvidia_smi_out})
                        if (_line MATCHES "^GPU [0-9]+:")
                            math(EXPR GMX_DETECT_GPU_COUNT_NVIDIA_SMI "${GMX_DETECT_GPU_COUNT_NVIDIA_SMI}+1")
                            # the UUID is not very useful for the user, remove it
                            string(REGEX REPLACE " \\(UUID:.*\\)" "" _gpu_info "${_line}")
                            if (NOT _gpu_info STREQUAL "")
                                list(APPEND GMX_DETECT_GPU_INFO "${_gpu_info}")
                            endif()
                        endif()
                    endforeach()

                    check_num_gpu_info(${GMX_DETECT_GPU_COUNT_NVIDIA_SMI} GMX_DETECT_GPU_INFO)
                    set(GMX_DETECT_GPU_COUNT ${GMX_DETECT_GPU_COUNT_NVIDIA_SMI})
                endif()
            endif()

            unset(_nvidia_smi CACHE)
            unset(_nvidia_smi_ret)
            unset(_nvidia_smi_out)
            unset(_gpu_name)
            unset(_line)
        endif()

        if (UNIX AND NOT (APPLE OR CYGWIN))
            # /proc/driver/nvidia/gpus/*/information-based detection.
            # Requires the NVDIA closed source driver to be installed and loaded
            if (NOT DEFINED GMX_DETECT_GPU_COUNT_PROC AND GMX_DETECT_GPU_COUNT EQUAL 0)
                set(GMX_DETECT_GPU_COUNT_PROC 0)
                file(GLOB _proc_nv_gpu_info "/proc/driver/nvidia/gpus/*/information")
                foreach (_file ${_proc_nv_gpu_info})
                    math(EXPR GMX_DETECT_GPU_COUNT_PROC "${GMX_DETECT_GPU_COUNT_PROC}+1")
                    # assemble information strings similar to the nvidia-smi output
                    # GPU ID = directory name on /proc/driver/nvidia/gpus/
                    string(REGEX REPLACE "/proc/driver/nvidia/gpus.*([0-9]+).*information" "\\1" _gpu_id ${_file})
                    # GPU name
                    file(STRINGS ${_file} _gpu_name LIMIT_COUNT 1 REGEX "^Model:.*" NO_HEX_CONVERSION)
                    string(REGEX REPLACE "^Model:[ \t]*(.*)" "\\1" _gpu_name "${_gpu_name}")
                    if (NOT _gpu_id STREQUAL "" AND NOT _gpu_name STREQUAL "")
                        list(APPEND GMX_DETECT_GPU_INFO "GPU ${_gpu_id}: ${_gpu_name}")
                    endif()
                endforeach()

                check_num_gpu_info(${GMX_DETECT_GPU_COUNT_PROC} GMX_DETECT_GPU_INFO)
                set(GMX_DETECT_GPU_COUNT ${GMX_DETECT_GPU_COUNT_PROC})

                unset(_proc_nv_gpu_info)
                unset(_gpu_name)
                unset(_gpu_id)
                unset(_file)
            endif()

            # lspci-based detection (does not provide GPU information).
            # Requires lspci and for GPU names to be fetched from the central
            # PCI ID db if not available locally.
            if (NOT DEFINED GMX_DETECT_GPU_COUNT_LSPCI AND GMX_DETECT_GPU_COUNT EQUAL 0)
                set(GMX_DETECT_GPU_COUNT_LSPCI 0)
                exec_program(lspci ARGS -q
                    OUTPUT_VARIABLE _lspci_out
                    RETURN_VALUE    _lspci_ret)
                # prehaps -q is not supported, try running without
                if (NOT RETURN_VALUE EQUAL 0)
                    exec_program(lspci
                        OUTPUT_VARIABLE _lspci_out
                        RETURN_VALUE    _lspci_ret)
                endif()
                if (_lspci_ret EQUAL 0)
                    # convert string with newlines to list of strings
                    STRING(REGEX REPLACE ";" "\\\\;" _lspci_out "${_lspci_out}")
                    string(REGEX REPLACE "\n" ";" _lspci_out "${_lspci_out}")
                    foreach(_line ${_lspci_out})
                        string(TOUPPER "${_line}" _line_upper)
                        if (_line_upper MATCHES ".*VGA.*NVIDIA.*" OR _line_upper MATCHES ".*3D.*NVIDIA.*")
                            math(EXPR GMX_DETECT_GPU_COUNT_LSPCI "${GMX_DETECT_GPU_COUNT_LSPCI}+1")
                            # Try to parse out the device name which should be
                            # included in the lspci -q output between []-s
                            string(REGEX REPLACE ".*\\[(.*)\\].*" "\\1" _gpu_name "${_line}")
                            if (NOT _gpu_name EQUAL "")
                                list(APPEND GMX_DETECT_GPU_INFO "${_gpu_name}")
                            endif()
                        endif()
                    endforeach()

                    check_num_gpu_info(${GMX_DETECT_GPU_COUNT_LSPCI} GMX_DETECT_GPU_INFO)
                    set(GMX_DETECT_GPU_COUNT ${GMX_DETECT_GPU_COUNT_LSPCI})
                endif()

                unset(_lspci_ret)
                unset(_lspci_out)
                unset(_gpu_name)
                unset(_line)
                unset(_line_upper)
            endif()
        endif()

        if (GMX_DETECT_GPU_COUNT GREATER 0)
            set(GMX_DETECT_GPU_AVAILABLE YES)
        else()
            set(GMX_DETECT_GPU_AVAILABLE NO)
        endif()
        set(GMX_DETECT_GPU_AVAILABLE ${GMX_DETECT_GPU_AVAILABLE} CACHE BOOL "Whether any NVIDIA GPU was detected" FORCE)

        set(GMX_DETECT_GPU_COUNT ${GMX_DETECT_GPU_COUNT}
            CACHE STRING "Number of NVIDIA GPUs detected")
        set(GMX_DETECT_GPU_INFO ${GMX_DETECT_GPU_INFO}
            CACHE STRING "basic information on the detected NVIDIA GPUs")

        set(GMX_DETECT_GPU_COUNT_NVIDIA_SMI ${GMX_DETECT_GPU_COUNT_NVIDIA_SMI}
            CACHE INTERNAL "Number of NVIDIA GPUs detected using nvidia-smi")
        set(GMX_DETECT_GPU_COUNT_PROC ${GMX_DETECT_GPU_COUNT_PROC}
            CACHE INTERNAL "Number of NVIDIA GPUs detected in /proc/driver/nvidia/gpus")
        set(GMX_DETECT_GPU_COUNT_LSPCI ${GMX_DETECT_GPU_COUNT_LSPCI}
            CACHE INTERNAL "Number of NVIDIA GPUs detected using lspci")

        mark_as_advanced(GMX_DETECT_GPU_AVAILABLE
                         GMX_DETECT_GPU_COUNT
                         GMX_DETECT_GPU_INFO)

        if (GMX_DETECT_GPU_AVAILABLE)
            message(STATUS "Number of NVIDIA GPUs detected: ${GMX_DETECT_GPU_COUNT} ")
        else()
            message(STATUS "Could not detect NVIDIA GPUs")
        endif()

    endif (NOT DEFINED GMX_DETECT_GPU_COUNT OR NOT DEFINED GMX_DETECT_GPU_INFO)
endmacro(gmx_detect_gpu)

macro(gmx_detect_OpenCL_linux)
    message(STATUS "Looking for working OpenCL configurations in the system via ICD loader")
    #find_package(OpenCL REQUIRED)
    
    #First look for libOpenCL
    find_library(cl_libopencl "libOpenCL.so")
    if(cl_libopencl)
        message(STATUS "libOpenCL ok: " ${cl_libopencl})
    else()
        message(FATAL_ERROR "No libOpenCL")
    endif()    
    
    #Now look for icds
    find_file(GMX_DETECT_OPENCL_LINUX_ICD_AMD "amdocl64.icd" "/etc/OpenCL/vendors")
    if(GMX_DETECT_OPENCL_LINUX_ICD_AMD)
        message(STATUS "ICD OK: ${GMX_DETECT_OPENCL_LINUX_ICD_AMD}")
        list(APPEND GMX_DETECT_OPENCL_LINUX_ICD_LIST "amdocl64.icd")
        list(APPEND GMX_DETECT_OPENCL_LINUX_ICD_LIST_PRINT "AMD")
    endif(GMX_DETECT_OPENCL_LINUX_ICD_AMD)    
    
    find_file(GMX_OPENCL_ICD_INTEL "intel64.icd" "/etc/OpenCL/vendors")
    if(GMX_DETECT_OPENCL_LINUX_ICD_INTEL)
        message(STATUS "ICD OK: ${GMX_DETECT_OPENCL_LINUX_ICD_INTEL}")
        list(APPEND GMX_DETECT_OPENCL_LINUX_ICD_LIST "intel64.icd")  
        list(APPEND GMX_DETECT_OPENCL_LINUX_ICD_LIST_PRINT "Intel")        
    endif(GMX_DETECT_OPENCL_LINUX_ICD_INTEL)     
    
    find_file(GMX_DETECT_OPENCL_LINUX_ICD_NVIDIA "nvidia.icd" "/etc/OpenCL/vendors")
    if(GMX_DETECT_OPENCL_LINUX_ICD_NVIDIA)
        message(STATUS "ICD OK: ${GMX_DETECT_OPENCL_LINUX_ICD_NVIDIA}")
        list(APPEND GMX_DETECT_OPENCL_LINUX_ICD_LIST "nvidia.icd")   
        list(APPEND GMX_DETECT_OPENCL_LINUX_ICD_LIST_PRINT "nVidia")                
    endif(GMX_DETECT_OPENCL_LINUX_ICD_NVIDIA)     
    
    list(LENGTH GMX_DETECT_OPENCL_LINUX_ICD_LIST_PRINT _num_cl_icds_found_)
    if( _num_cl_icds_found_ )
        message(STATUS " OpenCL Platforms detected (" ${_num_cl_icds_found_} ") : " ${GMX_DETECT_OPENCL_LINUX_ICD_LIST_PRINT})
    else()
        message(FATAL_ERROR "No ICD config found")
    endif()    

    set(GMX_DETECT_OPENCL_LINUX_AVAILABLE TRUE)
    set(GMX_DETECT_OPENCL_LINUX_WITH_ICD  TRUE)
    
    unset(_num_cl_icds_found_)
endmacro(gmx_detect_OpenCL_linux)

macro(gmx_detect_OpenCL)
    if (UNIX AND NOT (APPLE OR CYGWIN))
        gmx_detect_OpenCL_linux()
    endif()    
    if(GMX_DETECT_OPENCL_LINUX_AVAILABLE)
        set(GMX_DETECT_OPENCL_AVAILABLE TRUE)
    endif(GMX_DETECT_OPENCL_LINUX_AVAILABLE)
endmacro(gmx_detect_OpenCL)

# - Try to find OpenCL
# This module tries to find an OpenCL implementation on your system. Currently
# it supports searching system locations or detecting environment variables
# for the following implementations:
#  AMD Advanced Parallel Processing SDK
#  NVIDIA CUDA Toolkit
#  Intel OpenCL SDK
#  Generic system installed version
#  Custom location
#
# To set manually the paths, define these environment or CMake variables:
#  OPENCL_ROOT         - Root path containing include/CL/cl.h
#
# Once done this will define
#  GMX_DETECT_OPENCL_AVAILABLE              - System has an OpenCL library
#  OPENCL_INCLUDE_DIRS       - The OpenCL include directories needed
#  OPENCL_LIBRARIES          - Link libraries needed for OpenCL
#  OPENCL_VERSION_STRING     - Version of OpenCL that was found
#  OPENCL_HAS_CXX            - Whether or not C++ bindings are available
#  OPENCL_CXX_VERSION_STRING - Version of the C++ bindings if available
#  OPENCL_CXX_DEFINITIONS    - Compiler defines needed for the C++ bindings
#                             (May be nexessary if C++ bindings are of a
#                              different version than the C API; i.e OpenCL 1.2
#                              but with C++ bindings for 1.1)
macro(gmx_find_OpenCL)
if(NOT GMX_DETECT_OPENCL_AVAILABLE)
  include(CheckTypeSize)
  CHECK_TYPE_SIZE("void*" SIZEOF_VOID_P)

  # User specified OpenCL location
  if(OPENCL_ROOT)
    message(STATUS "OpenCL: Searching in custom location")    
    set(CMAKE_FIND_ROOT_PATH ${OPENCL_ROOT})
	set(_CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH})
    set(_OPENCL_ROOT_OPTS "ONLY_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH")
  elseif(NOT "$ENV{OPENCL_ROOT}" STREQUAL "")
    message(STATUS "OpenCL: Searching in custom location")
    set(_CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH})
    set(CMAKE_FIND_ROOT_PATH $ENV{OPENCL_ROOT})
    set(_OPENCL_ROOT_OPTS "ONLY_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH")

  # AMD APP SDK
  elseif(NOT "$ENV{AMDAPPSDKROOT}" STREQUAL "")
    message(STATUS "OpenCL: Searching for AMD APP SDK")    
    set(CMAKE_FIND_ROOT_PATH $ENV{AMDAPPSDKROOT})
	set(_CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH})
    set(_OPENCL_ROOT_OPTS "ONLY_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH")
    if(SIZEOF_VOID_P EQUAL 4)
      #set(_OPENCL_LIB_OPTS "PATH_SUFFIXES x86")
	  set(_OPENCL_LIB_DIR_SUFFIX "\\x86")
    else()
      #set(_OPENCL_LIB_OPTS PATH_SUFFIX x86_64)
      #set(_OPENCL_LIB_OPTS "PATH_SUFFIXES x86_64")
	  set(_OPENCL_LIB_DIR_SUFFIX "\\x86_64")
    endif()

  # NVIDIA CUDA
  elseif(NOT "$ENV{CUDA_PATH}" STREQUAL "")
    message(STATUS "OpenCL: Searching for NVIDIA CUDA SDK")    
    set(CMAKE_FIND_ROOT_PATH $ENV{CUDA_PATH})
	set(_CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH})
    set(_OPENCL_ROOT_OPTS "ONLY_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH")
    if(WIN32)
      if(SIZEOF_VOID_P EQUAL 4)
        #set(_OPENCL_LIB_OPTS "PATH_SUFFIX Win32")
		set(_OPENCL_LIB_DIR_SUFFIX "\\Win32")
      else()
        #set(_OPENCL_LIB_OPTS PATH_SUFFIX Win64)
		#set(_OPENCL_LIB_OPTS "PATH_SUFFIX x64")		
		set(_OPENCL_LIB_DIR_SUFFIX "\\x64")		
      endif()
    else()
      if(SIZEOF_VOID_P EQUAL 4)
        set(_OPENCL_LIB_DIR_SUFFIX)
      else()
        set(_OPENCL_LIB_DIR_SUFFIX 64)
      endif()
    endif()	

  # Intel OpenCL SDK
  elseif(NOT "$ENV{INTELOCLSDKROOT}" STREQUAL "")
    message(STATUS "OpenCL: Searching for Intel OpenCL SDK")    
    set(CMAKE_FIND_ROOT_PATH $ENV{INTELOCLSDKROOT})
	set(_CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH})
    set(_OPENCL_ROOT_OPTS "ONLY_CMAKE_FIND_ROOT_PATH NO_DEFAULT_PATH")
    if(WIN32)
      if(SIZEOF_VOID_P EQUAL 4)
        #set(_OPENCL_LIB_OPTS "PATH_SUFFIX x86")
		set(_OPENCL_LIB_DIR_SUFFIX "\\x86")
      else()
        #set(_OPENCL_LIB_OPTS "PATH_SUFFIX x64")
		set(_OPENCL_LIB_DIR_SUFFIX "\\x64")
      endif()
    else()
      if(SIZEOF_VOID_P EQUAL 4)
        set(_OPENCL_LIB_DIR_SUFFIX)
      else()
        set(_OPENCL_LIB_DIR_SUFFIX 64)
      endif()
    endif()

  # System location
  else()
    message(STATUS "OpenCL: Searching in system location")
  endif()

  if(APPLE)
    set(_OPENCL_INCLUDE_BASE OpenCL)
  else()
    set(_OPENCL_INCLUDE_BASE CL)
  endif()
  
  # Find the headers
  find_path(OPENCL_INCLUDE_DIR ${_OPENCL_INCLUDE_BASE}/cl.h
    PATHS /include
    ${_OPENCL_ROOT_OPTS}
  )
  
  if(OPENCL_INCLUDE_DIR)
    # Interrogate the C header for version information
    set(CMAKE_REQUIRED_INCLUDES ${OPENCL_INCLUDE_DIR})

    include(CheckSymbolExists)
    foreach(_MINOR_VER 0 1 2 3)
      CHECK_SYMBOL_EXISTS(CL_VERSION_1_${_MINOR_VER} "CL/cl.h" _OPENCL_VER)
      if(_OPENCL_VER)
        set(OPENCL_VERSION_STRING "1.${_MINOR_VER}")
        unset(_OPENCL_VER CACHE)
      else()
        break()
      endif()
    endforeach()

    if(EXISTS ${OPENCL_INCLUDE_DIR}/${_OPENCL_INCLUDE_BASE}/cl.hpp)
      set(OPENCL_HAS_CXX TRUE)

      # Interrogate the C++ header for seperate version information
      file(STRINGS ${OPENCL_INCLUDE_DIR}/${_OPENCL_INCLUDE_BASE}/cl.hpp
        _OPENCL_VER REGEX "version 1\\.[0-3]"
      )
      string(REGEX MATCH "1\\.([0-9])" OPENCL_CXX_VERSION_STRING
        "${_OPENCL_VER}"
      )
      set(_MINOR_VER ${CMAKE_MATCH_1})
      if(OPENCL_CXX_VERSION_STRING VERSION_LESS OPENCL_VERSION_STRING)
        set(OPENCL_CXX_DEFINITIONS -DCL_USE_DEPRECATED_OPENCL_1_${_MINOR_VER}_APIS)
      endif()
    else()
      set(OPENCL_HAS_CXX FALSE)
    endif()

    unset(CMAKE_REQUIRED_INCLUDES)
  endif()
  
 
  # Find the library
  #find_library(OPENCL_LIBRARY OpenCL
#	PATHS /lib${_OPENCL_LIB_DIR_SUFFIX}
#    ${_OPENCL_LIB_OPTS}	
#	${_OPENCL_ROOT_OPTS}
#   ) 

  find_library(OPENCL_LIBRARY OpenCL
	PATHS /lib${_OPENCL_LIB_DIR_SUFFIX}    
	${_OPENCL_ROOT_OPTS}
   )  
  
  # Restore the original search paths
  set(CMAKE_FIND_ROOT_PATH ${_CMAKE_FIND_ROOT_PATH})

  #include(FindPackageHandleStandardArgs)
  # FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenCL
  #   REQUIRED_VARS OPENCL_INCLUDE_DIR OPENCL_LIBRARY
  #   VERSION_VAR OPENCL_VERSION_STRING
  # )
  set(GMX_DETECT_OPENCL_AVAILABLE FALSE)
  if(OPENCL_INCLUDE_DIR)
	if(OPENCL_LIBRARY)
		set(GMX_DETECT_OPENCL_AVAILABLE TRUE)
	endif()
  endif()	
  
  set(OPENCL_INCLUDE_DIRS ${OPENCL_INCLUDE_DIR})
  set(OPENCL_LIBRARIES ${OPENCL_LIBRARY})
  
endif()

message(STATUS "GMX_DETECT_OPENCL_AVAILABLE: " "${GMX_DETECT_OPENCL_AVAILABLE} ")
message(STATUS "OPENCL_INCLUDE_DIRS: " "${OPENCL_INCLUDE_DIRS} ")
message(STATUS "OPENCL_LIBRARIES: " "${OPENCL_LIBRARIES} ")

set(GMX_OPENCL_DETECTION_DONE TRUE)
endmacro()

