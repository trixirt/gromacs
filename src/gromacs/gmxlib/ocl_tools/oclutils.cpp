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

/** \file oclutils.cpp
*  \brief OpenCL equivalent of cudautils.cu
*/

#include "oclutils.h"
#include <assert.h>

/*! Launches synchronous or asynchronous host to device memory copy.
*
*  If copy_event is not NULL, on return it will contain an event object
*  identifying this particular host to device operation. The event can further
*  be used to queue a wait for this operation or to query profiling information.
*
*  OpenCL equivalent of cu_copy_H2D_generic.
*/
static int ocl_copy_H2D_generic(cl_mem d_dest, void* h_src,
    size_t offset, size_t bytes,
    bool bAsync /* = false*/,
    cl_command_queue command_queue,
    cl_event *copy_event)
{
    cl_int cl_error;

    if (d_dest == NULL || h_src == NULL || bytes == 0)
    {
        return -1;
    }

    if (bAsync)
    {
        cl_error = clEnqueueWriteBuffer(command_queue, d_dest, CL_FALSE, offset, bytes, h_src, 0, NULL, copy_event);
        assert(cl_error == CL_SUCCESS);
        // TODO: handle errors
    }
    else
    {
        cl_error = clEnqueueWriteBuffer(command_queue, d_dest, CL_TRUE, offset, bytes, h_src, 0, NULL, copy_event);
        assert(cl_error == CL_SUCCESS);
        // TODO: handle errors
    }

    return 0;
}

/*! Launches asynchronous host to device memory copy.
*
*  If copy_event is not NULL, on return it will contain an event object
*  identifying this particular host to device operation. The event can further
*  be used to queue a wait for this operation or to query profiling information.
*
*  OpenCL equivalent of cu_copy_H2D_async.
*/
int ocl_copy_H2D_async(cl_mem d_dest, void * h_src,
    size_t offset, size_t bytes,
    cl_command_queue command_queue,
    cl_event *copy_event)
{
    return ocl_copy_H2D_generic(d_dest, h_src, offset, bytes, true, command_queue, copy_event);
}

/*! Launches synchronous host to device memory copy.
*
*  OpenCL equivalent of cu_copy_H2D.
*/
int ocl_copy_H2D(cl_mem d_dest, void * h_src,
    size_t offset, size_t bytes,
    cl_command_queue command_queue)
{
    return ocl_copy_H2D_generic(d_dest, h_src, offset, bytes, false, command_queue, NULL);
}

/*! Launches synchronous or asynchronous device to host memory copy.
*
*  If copy_event is not NULL, on return it will contain an event object
*  identifying this particular device to host operation. The event can further
*  be used to queue a wait for this operation or to query profiling information.
*
*  OpenCL equivalent of cu_copy_D2H_generic.
*/
int ocl_copy_D2H_generic(void * h_dest, cl_mem d_src,
    size_t offset, size_t bytes,
    bool bAsync,
    cl_command_queue command_queue,
    cl_event *copy_event)
{
    cl_int cl_error;

    if (h_dest == NULL || d_src == NULL || bytes == 0)
    {
        return -1;
    }

    if (bAsync)
    {
        cl_error = clEnqueueReadBuffer(command_queue, d_src, CL_FALSE, offset, bytes, h_dest, 0, NULL, copy_event);
        assert(cl_error == CL_SUCCESS);
        // TODO: handle errors
    }
    else
    {
        cl_error = clEnqueueReadBuffer(command_queue, d_src, CL_TRUE, offset, bytes, h_dest, 0, NULL, copy_event);
        assert(cl_error == CL_SUCCESS);
        // TODO: handle errors
    }

    return 0;
}

/*! Launches asynchronous device to host memory copy.
*
*  If copy_event is not NULL, on return it will contain an event object
*  identifying this particular host to device operation. The event can further
*  be used to queue a wait for this operation or to query profiling information.
*
*  OpenCL equivalent of cu_copy_D2H_async.
*/
int ocl_copy_D2H_async(void * h_dest, cl_mem d_src,
    size_t offset, size_t bytes,
    cl_command_queue command_queue,
    cl_event *copy_event)
{
    return ocl_copy_D2H_generic(h_dest, d_src, offset, bytes, true, command_queue, copy_event);
}