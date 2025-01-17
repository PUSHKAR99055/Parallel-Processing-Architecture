#include <iostream>
#include <sys/time.h>
#include "sssp_common.h"

void matrixInit(float *a, float *b, float *c)
{
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {   
            a[i * MATRIX_SIZE + j] = rand() / (float)1147654321;
            b[i * MATRIX_SIZE + j] = rand() / (float)1147654321;
            c[i * MATRIX_SIZE + j] = (float)0;
        }
    }
}

__device__ void warpReduce(volatile matElement *newSharedB, int threadId)
{
    if(newSharedB[threadId].value > newSharedB[threadId + 32].value){
        newSharedB[threadId].value = newSharedB[threadId + 32].value;
        newSharedB[threadId].row = newSharedB[threadId + 32].row;
        newSharedB[threadId].col = newSharedB[threadId + 32].col;
    }

    if(newSharedB[threadId].value > newSharedB[threadId + 16].value){
        newSharedB[threadId].value = newSharedB[threadId + 16].value;
        newSharedB[threadId].row = newSharedB[threadId + 16].row;
        newSharedB[threadId].col = newSharedB[threadId + 16].col;
    }

    if(newSharedB[threadId].value > newSharedB[threadId + 8].value){
        newSharedB[threadId].value = newSharedB[threadId + 8].value;
        newSharedB[threadId].row = newSharedB[threadId + 8].row;
        newSharedB[threadId].col = newSharedB[threadId + 8].col;
    }

    if(newSharedB[threadId].value > newSharedB[threadId + 4].value){
        newSharedB[threadId].value = newSharedB[threadId + 4].value;
        newSharedB[threadId].row = newSharedB[threadId + 4].row;
        newSharedB[threadId].col = newSharedB[threadId + 4].col;
    }

    if(newSharedB[threadId].value > newSharedB[threadId + 2].value){
        newSharedB[threadId].value = newSharedB[threadId + 2].value;
        newSharedB[threadId].row = newSharedB[threadId + 2].row;
        newSharedB[threadId].col = newSharedB[threadId + 2].col;
    }

    if(newSharedB[threadId].value > newSharedB[threadId + 1].value){
        newSharedB[threadId].value = newSharedB[threadId + 1].value;
        newSharedB[threadId].row = newSharedB[threadId + 1].row;
        newSharedB[threadId].col = newSharedB[threadId + 1].col;
    }
}

__device__ void minBlockReduce(matElement *newSharedB, int threadId)
{
    for (unsigned int stride = (BLOCK_DIM * BLOCK_DIM)/2; stride > 32; stride >>= 1)
    {
        if(threadId < stride)
        {
            if(newSharedB[threadId].value > newSharedB[threadId + stride].value){
                newSharedB[threadId] = newSharedB[threadId + stride];
            }
        }
        __syncthreads();
    }
    if(threadId < 32) warpReduce(newSharedB, threadId);
}

__global__ void find2Min(int16_t firstMinRow, int16_t firstMinCol, float *c, matElement *d_minValueFromEachBlock)
{
    int16_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int16_t col = blockIdx.x * blockDim.x + threadIdx.x;

    int16_t threadId = threadIdx.y * BLOCK_DIM + threadIdx.x;                       //thread id within each block only

    __shared__ matElement sharedC[BLOCK_DIM * BLOCK_DIM];

    if(row == 0 && col == 0) c[firstMinRow * MATRIX_SIZE + firstMinCol] = __FLT_MAX__;
    
    __syncthreads();

    sharedC[threadId].value = c[row * MATRIX_SIZE + col];
    sharedC[threadId].row = row;
    sharedC[threadId].col = col;
    __syncthreads();

    minBlockReduce(sharedC, threadId);
    if(threadId == 0){   
        d_minValueFromEachBlock[blockIdx.y * gridDim.x + blockIdx.x].value = sharedC[0].value;
        d_minValueFromEachBlock[blockIdx.y * gridDim.x + blockIdx.x].row = sharedC[0].row;
        d_minValueFromEachBlock[blockIdx.y * gridDim.x + blockIdx.x].col = sharedC[0].col;
    }

    // if(row == 0 && col ==0) c[firstMinRow * MATRIX_SIZE + firstMinCol] = tempVal;                   //replace the first min val with the original since we replaced it with FLT_MAX for finding second min
}

__global__ void tiledMatrixMultiply(float *a, float *b, float *c, matElement *d_minValueFromEachBlock)
{
    int16_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int16_t col = blockIdx.x * blockDim.x + threadIdx.x;

    int16_t threadId = threadIdx.y * BLOCK_DIM + threadIdx.x;

    __shared__ float sharedA[BLOCK_DIM * BLOCK_DIM];
    __shared__ float sharedB[BLOCK_DIM * BLOCK_DIM * sizeof(matElement)];                    

    float temp = 0;

    for (int i = 0; i < MATRIX_SIZE / TILE_SZE; i++)
    {
        sharedA[threadId] = a[row * MATRIX_SIZE + (i * TILE_SZE + threadIdx.x)];                 //index into the global a with the global row (since we are tiling across x dimention of a) and each thread's tile 
        sharedB[threadId] = b[(i * TILE_SZE + threadIdx.y) * MATRIX_SIZE + col];                 //index into the global b with each thread's tile idexes (since we are tiling across y dimention of b) and globale column 
        __syncthreads();                                                                         //make sure all values of the sub-matrices are loaded by thre threads before proceding

        for (int j = 0; j < TILE_SZE; j++)
        {
            temp += sharedA[threadIdx.y * TILE_SZE + j] * sharedB[j * TILE_SZE + threadIdx.x];
        }

        __syncthreads();                                                                         //make sure all sub-matrix calculation is done by threads before advancing to the next sub-matricies

    }
    matElement *newSharedB = (matElement*) sharedB;                                              //reuse shared mem for finding min element

    newSharedB[threadId].value = temp;
    newSharedB[threadId].row = row;
    newSharedB[threadId].col = col;
    __syncthreads();
    
    c[row * MATRIX_SIZE + col] = temp;

    minBlockReduce(newSharedB, threadId);
    if(threadId == 0){   
        d_minValueFromEachBlock[blockIdx.y * gridDim.x + blockIdx.x].value = newSharedB[0].value;
        d_minValueFromEachBlock[blockIdx.y * gridDim.x + blockIdx.x].row = newSharedB[0].row;
        d_minValueFromEachBlock[blockIdx.y * gridDim.x + blockIdx.x].col = newSharedB[0].col;
    }
}

extern float* computeMatrixMult(matElement *minElement)
{
    struct timeval start_time, end_time;
    double exec_time;
    minElement[0].value = __FLT_MAX__;
    minElement[1].value = __FLT_MAX__;

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    matElement *h_minValueFromEachBlock;
    matElement *d_minValueFromEachBlock;

    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    h_minValueFromEachBlock = (matElement*)malloc((MATRIX_SIZE / BLOCK_DIM) * (MATRIX_SIZE / BLOCK_DIM) * sizeof(matElement));

    CHECK(cudaMallocHost(&d_a, size));
    CHECK(cudaMallocHost(&d_b, size));
    CHECK(cudaMallocHost(&d_c, size));
    CHECK(cudaMallocHost(&d_minValueFromEachBlock, (MATRIX_SIZE / BLOCK_DIM) * (MATRIX_SIZE / BLOCK_DIM) * sizeof(matElement)));

    matrixInit(h_a, h_b, h_c);

    dim3 blockPerGrid(MATRIX_SIZE / BLOCK_DIM , MATRIX_SIZE / BLOCK_DIM);
    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);

    
    gettimeofday(&start_time, NULL);

    CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    

    tiledMatrixMultiply<<<blockPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, d_minValueFromEachBlock);

    CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_minValueFromEachBlock, d_minValueFromEachBlock, (MATRIX_SIZE / BLOCK_DIM) * (MATRIX_SIZE / BLOCK_DIM) * sizeof(matElement), cudaMemcpyDeviceToHost));
    for (int i = 0; i < (MATRIX_SIZE / BLOCK_DIM) * (MATRIX_SIZE / BLOCK_DIM); i++)
    {
        if(h_minValueFromEachBlock[i].value < minElement[0].value)
        {
            minElement[0].value = h_minValueFromEachBlock[i].value;           
            minElement[0].row = h_minValueFromEachBlock[i].row;
            minElement[0].col = h_minValueFromEachBlock[i].col;
        }
    }

    find2Min<<<blockPerGrid, threadsPerBlock>>>(minElement[0].row, minElement[0].col, d_c, d_minValueFromEachBlock);

    CHECK(cudaMemcpy(h_minValueFromEachBlock, d_minValueFromEachBlock, (MATRIX_SIZE / BLOCK_DIM) * (MATRIX_SIZE / BLOCK_DIM) * sizeof(matElement), cudaMemcpyDeviceToHost));
    
    d_c[minElement[0].row * MATRIX_SIZE + minElement[0].col] = minElement[0].value;
    
    for (int i = 0; i < (MATRIX_SIZE / BLOCK_DIM) * (MATRIX_SIZE / BLOCK_DIM); i++)
    {
        if(h_minValueFromEachBlock[i].value < minElement[1].value)
        {
            minElement[1].value = h_minValueFromEachBlock[i].value;           
            minElement[1].row = h_minValueFromEachBlock[i].row;
            minElement[1].col = h_minValueFromEachBlock[i].col;
        }
    }
    gettimeofday(&end_time, NULL);

    free(h_a);
    free(h_b);

    cudaFree(d_a);
    cudaFree(d_b);

    exec_time = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec)/(double)1000000;

    std::cout<<"Execution time - "<<exec_time<<std::endl;
    
    std::cout<<"Matrix size - "<<MATRIX_SIZE<<std::endl;

    std::cout<<"Min value 1 (val, row, col) - ("<<minElement[0].value<<", "<<minElement[0].row<<", "<<minElement[0].col<<")"<<std::endl;

    std::cout<<"Min value 2 (val, row, col) - ("<<minElement[1].value<<", "<<minElement[1].row<<", "<<minElement[1].col<<")"<<std::endl;

    return d_c;

}