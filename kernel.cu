/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

__global__ void reduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    
	__shared__ float pSum[2 * BLOCK_SIZE];
	
	int i = 2 * blockIdx.x * blockDim.x;
	int tx = threadIdx.x;
	
	pSum[tx] = (tx < size) ? in[i + tx] : 0;
	pSum[blockDim.x + tx] = ((blockDim.x + tx) < size) ? in[i + blockDim.x + tx] : 0;
	
	for(int offset = blockDim.x; offset >= 1; offset >>= 1) {
		__syncthreads();
		if(tx < offset)
			pSum[tx] += pSum[tx + offset];
	}
	
	if(tx == 0) {
		out[blockIdx.x + tx] = pSum[tx];
	}
}