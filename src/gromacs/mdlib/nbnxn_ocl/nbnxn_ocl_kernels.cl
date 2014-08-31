/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#define KERNEL_RADIUS 8

#define      ROWS_BLOCKDIM_X 16
#define      ROWS_BLOCKDIM_Y 4
#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8

#define    ROWS_RESULT_STEPS 4
#define      ROWS_HALO_STEPS 1
#define COLUMNS_RESULT_STEPS 4
#define   COLUMNS_HALO_STEPS 1




#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)



////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y, 1)))
void convolutionRowsOrig(
    __global float *d_Dst,
    __global float *d_Src,
    __constant float *c_Kernel,
    int imageW,
    int imageH,
    int pitch
){
    __local float l_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (get_group_id(0) * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + get_local_id(0);
    const int baseY = get_group_id(1) * ROWS_BLOCKDIM_Y + get_local_id(1);

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Load main data
    for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
        l_Data[get_local_id(1)][get_local_id(0) + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];

    //Load left halo
    for(int i = 0; i < ROWS_HALO_STEPS; i++)
        l_Data[get_local_id(1)][get_local_id(0) + i * ROWS_BLOCKDIM_X]  = (baseX + i * ROWS_BLOCKDIM_X >= 0) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;

    //Load right halo
    for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
        l_Data[get_local_id(1)][get_local_id(0) + i * ROWS_BLOCKDIM_X]  = (baseX + i * ROWS_BLOCKDIM_X < imageW) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;

    //Compute and store results
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){
        float sum = 0;

        for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
            sum += c_Kernel[KERNEL_RADIUS - j] * l_Data[get_local_id(1)][get_local_id(0) + i * ROWS_BLOCKDIM_X + j];

        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, 1)))
void convolutionColumns(
    __global float *d_Dst,
    __global float *d_Src,
    __constant float *c_Kernel,
    int imageW,
    int imageH,
    int pitch
){
    __local float l_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = get_group_id(0) * COLUMNS_BLOCKDIM_X + get_local_id(0);
    const int baseY = (get_group_id(1) * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + get_local_id(1);
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Load main data
    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
        l_Data[get_local_id(0)][get_local_id(1) + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];

    //Load upper halo
    for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
        l_Data[get_local_id(0)][get_local_id(1) + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y >= 0) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

    //Load lower halo
    for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
        l_Data[get_local_id(0)][get_local_id(1) + i * COLUMNS_BLOCKDIM_Y]  = (baseY + i * COLUMNS_BLOCKDIM_Y < imageH) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

    //Compute and store results
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
        float sum = 0;

        for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
            sum += c_Kernel[KERNEL_RADIUS - j] * l_Data[get_local_id(0)][get_local_id(1) + i * COLUMNS_BLOCKDIM_Y + j];

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y, 1)))
void convolutionRows(
    __global float *d_Dst,
    __global float *d_Src,
    __constant float *c_Kernel,
    int imageW,
    int imageH,
    int pitch
){
    __local float l_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (get_group_id(0) * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + get_local_id(0);
    const int baseY = get_group_id(1) * ROWS_BLOCKDIM_Y + get_local_id(1);

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Load main data
    for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
        l_Data[get_local_id(1)][get_local_id(0) + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];

    //Load left halo
    for(int i = 0; i < ROWS_HALO_STEPS; i++)
        l_Data[get_local_id(1)][get_local_id(0) + i * ROWS_BLOCKDIM_X]  = (baseX + i * ROWS_BLOCKDIM_X >= 0) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;

    //Load right halo
    for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
        l_Data[get_local_id(1)][get_local_id(0) + i * ROWS_BLOCKDIM_X]  = (baseX + i * ROWS_BLOCKDIM_X < imageW) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;

    //Compute and store results
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){
        float sum = 0;

        for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
            sum += c_Kernel[KERNEL_RADIUS - j] * l_Data[get_local_id(1)][get_local_id(0) + i * ROWS_BLOCKDIM_X + j];

        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}


__kernel
void convolutionColumns_simplu(
    __global float *d_Dst,
    __global float *d_Src,
    __constant float *c_Kernel,
    int imageW,
    int imageH,
    int pitch
){
	int i, j;
	int line = get_global_id(1);
	int column = get_global_id(0);

	float sum;

	for (i = -8; i <= 8; i++)
		for (j = -8; j <=8; j++)
			if ((line + i) >= 0)
				if ((line + i) < imageH)
					if ((column + j) >= 0)
						if ((column + j) < imageW)
							sum += d_Src[(line + i) * imageW + column + j] * c_Kernel[i];

	d_Dst[line * imageW + column] = sum;
}

__kernel
void convolutionColumnstex(
    __global float *d_Dst,
    __global float *d_Src,
    __constant float *c_Kernel,
    int imageW,
    int imageH,
    int pitch,
	sampler_t texSampler,
	read_only image2d_t t2D
){
	int i, j;
	int line = get_global_id(1);
	int column = get_global_id(0);

	float sum;

	sampler_t smp = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

	for (i = -8; i <= 8; i++)
		for (j = -8; j <=8; j++)
			//if ((line + i) >= 0)
			//	if ((line + i) < imageH)
			//		if ((column + j) >= 0)
			//			if ((column + j) < imageW)
						{
							float v = read_imagef(t2D, smp, (int2)(column + j,	line + i)).x;
							sum += v * c_Kernel[i];
							//d_Src[(line + i) * imageW + column + j] * c_Kernel[i];
						}

	d_Dst[line * imageW + column] = sum;
}

__kernel
void convolutionColumns_shmem(
    __global float *d_Dst,
    __global float *d_Src,
    __constant float *c_Kernel,
    int imageW,
    int imageH,
    int pitch
){
	__local float neigh[COLUMNS_BLOCKDIM_Y+16][COLUMNS_BLOCKDIM_X+16];
	int i, j;
	int line = get_global_id(1);
	int column = get_global_id(0);

	float sum;

	neigh[get_local_id(1) + 8][get_local_id(0) + 8] = d_Src[line * imageW + column];
	if (get_local_id(1) < 8)
	{
		if (line >= 8)
			neigh[get_local_id(1)][get_local_id(0) + 8] = d_Src[(line - 8) * imageW + column];
		else
			neigh[get_local_id(1)][get_local_id(0) + 8] = 0;
	}

	if (get_local_id(1) >= get_local_size(1) - 8)
		if ((line + 8) < imageH)
			neigh[get_local_id(1) + 16][get_local_id(0) + 8] = d_Src[(line + 8) * imageW + column];
		else
			neigh[get_local_id(1) + 16][get_local_id(0) + 8] = 0;

	if (get_local_id(0) < 8)
		if (column >= 8)
			neigh[get_local_id(1) + 8][get_local_id(0)] = d_Src[line * imageW + column - 8];
		else
			neigh[get_local_id(1) + 8][get_local_id(0)] = 0;

	if (get_local_id(0) >= get_local_size(0) - 8)
		if ((column + 8) < imageW)
			neigh[get_local_id(1) + 8][get_local_id(0) + 16] = d_Src[line * imageW + column + 8];
		else
			neigh[get_local_id(1) + 8][get_local_id(0) + 16] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (i = -8; i <= 8; i++)
		for (j = -8; j <=8; j++)
			//if ((line + i) >= 0)
			//	if ((line + i) < imageH)
			//		if ((column + j) >= 0)
			//			if ((column + j) < imageW)
							sum += neigh[get_local_id(1) + i + 8][get_local_id(0) + j + 8] * c_Kernel[i];

	d_Dst[line * imageW + column] = sum;

}
////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, 1)))
void convolutionColumns11(
    __global float *d_Dst,
    __global float *d_Src,
    __constant float *c_Kernel,
    int imageW,
    int imageH,
    int pitch
){
    __local float l_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = get_group_id(0) * COLUMNS_BLOCKDIM_X + get_local_id(0);
    const int baseY = (get_group_id(1) * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + get_local_id(1);
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Load main data
    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
        l_Data[get_local_id(0)][get_local_id(1) + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];

    //Load upper halo
    for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
        l_Data[get_local_id(0)][get_local_id(1) + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y >= 0) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

    //Load lower halo
    for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
        l_Data[get_local_id(0)][get_local_id(1) + i * COLUMNS_BLOCKDIM_Y]  = (baseY + i * COLUMNS_BLOCKDIM_Y < imageH) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

    //Compute and store results
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
        float sum = 0;

        for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
            sum += c_Kernel[KERNEL_RADIUS - j] * l_Data[get_local_id(0)][get_local_id(1) + i * COLUMNS_BLOCKDIM_Y + j];

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }
}