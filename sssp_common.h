// #define __COMMON__
// #ifdef __COMMON__

#define MATRIX_SIZE 1024
#define BLOCK_DIM 32                
#define TILE_SZE BLOCK_DIM          //Tile size is same as block dimension. defined for better code understandability
#define TEST 0

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if(error != cudaSuccess)\
    {\
        std::cout<<"Error: "<<__FILE__<<":"<<__LINE__<<std::endl;\
        std::cout<<"Code: "<<error<<", reason: "<<cudaGetErrorString(error)<<std::endl;\
        exit(1);\
    }\
}

typedef struct
{
    float value;
    int16_t row, col;

} matElement;

typedef struct
{
    float value;
    int pathIndex;
} pathElement;

// #endif