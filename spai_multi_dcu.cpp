#include "hip/hip_runtime.h"
#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <memory.h>
#include <string>
#include <vector>
#include "common/dataType.h"
#include "common/read.h"
#include "common/init.h"
#include "common/assemble.h"
#include "common/cuFormatConversion.h"
#include "bicgstab/bicgstab_solver.h"

using namespace std;

//==============================================================================
// 多DCU管理结构
//==============================================================================

struct MultiDCU_Context {
    int numDCUs;                    // DCU总数
    int *deviceIds;                 // 设备ID数组
    CSC_Matrix **devCSC_A;         // 每个DCU上的矩阵A副本
    CSC_Matrix **devCSC_M_local;   // 每个DCU上的局部预条件子
    int *colStart;                  // 每个DCU负责的起始列
    int *colEnd;                    // 每个DCU负责的结束列
    hipStream_t *streams;           // 每个DCU的流
};

#define CHECK_HIP_ERROR(call) { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        printf("HIP Error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_KERNEL_ERROR(stage, device_id) do { \
    hipError_t err = hipGetLastError(); \
    if (err != hipSuccess) { \
        printf("Kernel '%s' failed on DCU %d: %s\n", stage, device_id, hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

//==============================================================================
// Kernel函数（从单DCU版本复制）
//==============================================================================

template<unsigned int N2SIZE>
__global__ void cuComputeN2MAXwithSparityofA(int *aPtr, int nCol, int *n2max) {
    __shared__ int n2_s[N2SIZE];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    int tid = threadIdx.x;

    int col, col_s, col_e, tV;
    int i;

    int value = 0;
    for (col = gid; col < nCol; col += offset) {
        col_s = aPtr[col];
        col_e = aPtr[col + 1];
        tV = col_e - col_s;
        if (value < tV) value = tV;
    }

    n2_s[tid] = value;
    __syncthreads();

    i = N2SIZE / 2;
    while (tid < i) {
        if (n2_s[tid] < n2_s[tid + i]) n2_s[tid] = n2_s[tid + i];
        i = i / 2;
        __syncthreads();
    }

    if (tid == 0) n2max[blockIdx.x] = n2_s[0];
}

template<unsigned int N1SIZE>
__global__ void cuComputeN1MAX(int *n1, int nCol, int *n1max) {

    __shared__ int n1_s[N1SIZE];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x * gridDim.x;
    int tid = threadIdx.x;

    int col, value;
    int i;

    value = 0;

    for (col = gid; col < nCol; col += offset) {
        if (value < n1[col]) value = n1[col];

    }//for col

    n1_s[tid] = value;

    __syncthreads();

    i = N1SIZE / 2;
    while (tid < i) {
        if (n1_s[tid] < n1_s[tid + i]) n1_s[tid] = n1_s[tid + i];
        i = i / 2;
        __syncthreads();
    }

    if (tid == 0) n1max[blockIdx.x] = n1_s[0];

}


template<unsigned int WarpSize, unsigned int ISIZE>
__global__ void cuComputeN1withSparityofA_SpMMv1(int *aPtr, int *aIndex, int *mPtr, int *mIndex,
                                                 int *I, int n1max, int nCol, int *n1, int *atomic) {
    __shared__ int I_s[ISIZE];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp
    int tid = threadIdx.x / WarpSize;

    int col, col_s, col_e;
    int fcol, fcol_s, fcol_e;
    int othercol, ocol, ocol_s, ocol_e;
    int bV, bV1, bV2;
    int j, j1, j2, tN, flag, idx;

    int seg = ISIZE / (blockDim.x / WarpSize);
    int TSEG = tid * seg;

    for (col = warp_id; col < nCol; col += offset) {
        //read the columns
        col_s = mPtr[col];
        col_e = mPtr[col + 1];
        bV = col_e - col_s;

        //for the first column, all data are written into I
        fcol = mIndex[col_s];
        fcol_s = aPtr[fcol];
        fcol_e = aPtr[fcol + 1];
        bV1 = fcol_e - fcol_s;

        atomic[col] = bV1;

        for (j = lane; j < bV1; j += WarpSize) {
            I_s[TSEG + j] = aIndex[j + fcol_s];
        }//for j

        __syncthreads();


        //for other columns
        for (othercol = 1; othercol < bV; othercol++) {
            ocol = mIndex[col_s + othercol];
            ocol_s = aPtr[ocol];
            ocol_e = aPtr[ocol + 1];

            bV2 = ocol_e - ocol_s;
            //if the element in other column does not exist in I, append it to I
            tN = atomic[col];
            for (j = lane; j < bV2; j += WarpSize) {
                idx = aIndex[j + ocol_s];
                flag = -1;
                for (j1 = 0; j1 < tN; j1++) {
                    if (idx == I_s[TSEG + j1]) {
                        flag = 1;
                        break;
                    }
                }
                if (flag == -1) {
                    j2 = atomicAdd(&atomic[col], 1);
                    if (j2 >= seg) {
                        printf("exceed the maximum shared size for col = %d\n", col);
                        break;
                    }
                    I_s[TSEG + j2] = idx;
                }

            }//for j
            __syncthreads();


        }//for othercol

        n1[col] = atomic[col];

    }//end for col
}

template<unsigned int WarpSize>
__global__ void cuComputeN1withSparityofA2_SpMMv1(int *aPtr, int *aIndex, int *mPtr, int *mIndex,
                                                  int *I, int n1max, int nCol, int *n1, int *atomic) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp


    int col, col_s, col_e;
    int fcol, fcol_s, fcol_e;
    int othercol, ocol, ocol_s, ocol_e;
    int bV, bV1, bV2, tN, flag;
    int j, j1, j2, idx;

    for (col = warp_id; col < nCol; col += offset) {
        //read the columns
        col_s = mPtr[col];
        col_e = mPtr[col + 1];
        bV = col_e - col_s;

        //for the first column, all data are written into I
        fcol = mIndex[col_s];
        fcol_s = aPtr[fcol];
        fcol_e = aPtr[fcol + 1];
        bV1 = fcol_e - fcol_s;

        atomic[col] = bV1;

        for (j = lane; j < bV1; j += WarpSize) {
            I[col * n1max + j] = aIndex[j + fcol_s];
        }//for j

        __syncthreads();

        //for other columns
        for (othercol = 1; othercol < bV; othercol++) {
            ocol = mIndex[col_s + othercol];
            ocol_s = aPtr[ocol];
            ocol_e = aPtr[ocol + 1];

            bV2 = ocol_e - ocol_s;
            //if the element in other column does not exist in I, append it to I
            tN = atomic[col];

            for (j = lane; j < bV2; j += WarpSize) {
                idx = aIndex[j + ocol_s];
                flag = -1;
                for (j1 = 0; j1 < tN; j1++) {
                    if (idx == I[col * n1max + j1]) {
                        flag = 1;
                        break;
                    }
                }
                if (flag == -1) {
                    j2 = atomicAdd(&atomic[col], 1);
                    if (j2 >= n1max) {
                        printf("exceed the maximum shared size for col = %d\n", col);
                        break;
                    }
                    I[col * n1max + j2] = idx;
                }

            }//for j 0:bv2

            __syncthreads();

        }//for othercol

        n1[col] = atomic[col];

    }//end for col
}

template<unsigned int WarpSize>
__global__ void computeJ_SSPAIv10(int *MPtr, int *MIndex,
                                  int nCol, int *J, int *jPTR, int N2MAX) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);//index of threads in warp

    int col, col_s, col_e, j, bV;

    for (col = warp_id; col < nCol; col += offset) {
        col_s = MPtr[col];
        col_e = MPtr[col + 1];
        bV = col_e - col_s;
        jPTR[col] = bV;
        for (j = lane; j < bV; j += WarpSize) {
            J[col * N2MAX + j] = MIndex[j + col_s];
        }
    }//for col

}

template<unsigned int WarpSize>
__global__ void computeJ2_SSPAIv10(int *MPtr, int *MIndex,
                                   int nCol, int *J, int *jPTR, int N2MAX, int sK) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);//index of threads in warp

    int col, col_s, col_e, j, bV;
    int colK;

    for (col = warp_id; col < nCol; col += offset) {
        colK = col + sK;
        col_s = MPtr[colK];
        col_e = MPtr[colK + 1];
        bV = col_e - col_s;
        jPTR[col] = bV;
        for (j = lane; j < bV; j += WarpSize) {
            J[col * N2MAX + j] = MIndex[j + col_s];
        }
    }//for col

}

template<unsigned int WarpSize>
__global__ void computeI_iter_Symbol_SpMMv1(int *APtr, int *AIndex, int nCol,
                                            int *I, int *iPTR, int N1MAX, int *J, int *jPTR, int N2MAX, int *atomic) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp
    //int tid = threadIdx.x / WarpSize;

    int col, fcol, fcol_s, fcol_e, bV, bV1, jN, tN;
    int ocol, othercol, othercol_s, othercol_e, j, j1, j2;

    int JBnd;
    int IBnd;

    //int seg = SIZE_I_SHARED/CounterSize;

    for (col = warp_id; col < nCol; col += offset) {
        JBnd = col * N2MAX;
        IBnd = col * N1MAX;

        //read the columns
        fcol = J[JBnd];
        fcol_s = APtr[fcol];
        fcol_e = APtr[fcol + 1];
        bV = fcol_e - fcol_s;

        atomic[col] = bV;

        //for the first column, all data are written into I
        for (j = lane; j < bV; j += WarpSize) {
            I[IBnd + j] = AIndex[j + fcol_s];
        }//for j

        __syncthreads();

        //for other columns
        jN = jPTR[col];
        for (othercol = 1; othercol < jN; othercol++) {
            ocol = J[JBnd + othercol];
            othercol_s = APtr[ocol];
            othercol_e = APtr[ocol + 1];

            bV1 = othercol_e - othercol_s;
            //if the element in other column does not exist in I, append it to I
            tN = atomic[col];
            for (j = lane; j < bV1; j += WarpSize) {
                int idx = AIndex[j + othercol_s];
                int flag = 1;
                for (j1 = 0; j1 < tN; j1++) {
                    if (idx == I[IBnd + j1]) {
                        flag = -1;
                        break;
                    }
                }

                if (flag == 1) {
                    j2 = atomicAdd(&atomic[col], 1);
                    I[IBnd + j2] = idx;
                }
            }//for j


            __syncthreads();

        }//for othercol

        iPTR[col] = atomic[col];

        tN = iPTR[col];
        //ODD_EVEN
        for (int i = 1; i <= tN; i++) {
            if (i % 2 == 1) {
                for (int j1 = lane; j1 < tN; j1 += WarpSize) {
                    if ((j1 * 2 + 1) < tN && (I[IBnd + j1 * 2] > I[IBnd + j1 * 2 + 1])) {
                        bV1 = I[IBnd + j1 * 2];
                        I[IBnd + j1 * 2] = I[IBnd + j1 * 2 + 1];
                        I[IBnd + j1 * 2 + 1] = bV1;
                    }
                }
                __syncthreads();
            } else {
                for (int j2 = lane; j2 < tN; j2 += WarpSize) {
                    if ((j2 * 2 + 2) < tN && (I[IBnd + j2 * 2 + 1] > I[IBnd + j2 * 2 + 2])) {
                        bV1 = I[IBnd + j2 * 2 + 2];
                        I[IBnd + j2 * 2 + 2] = I[IBnd + j2 * 2 + 1];
                        I[IBnd + j2 * 2 + 1] = bV1;
                    }
                }
                __syncthreads();
            }
            if (WarpSize > 32) {
                __syncthreads();
            }
        }

    }//for col
}

template<unsigned int WarpSize, unsigned int CounterSize, unsigned int SIZE_I_SHARED>
__global__ void computeI_SSPAIv10(int *APtr, int *AIndex, int nCol,
                                  int *I, int *iPTR, int N1MAX, int *J, int *jPTR, int N2MAX) {

    __shared__ int I_s[SIZE_I_SHARED];
    __shared__ int counter_s[CounterSize];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp
    int tid = threadIdx.x / WarpSize;

    int col, fcol, fcol_s, fcol_e, bV, bV1;
    int ocol, othercol, othercol_s, othercol_e, j, j1;

    int seg = SIZE_I_SHARED / CounterSize;

    for (col = warp_id; col < nCol; col += offset) {
        //read the columns
        fcol = J[col * N2MAX];
        fcol_s = APtr[fcol];
        fcol_e = APtr[fcol + 1];
        bV = fcol_e - fcol_s;

        counter_s[tid] = bV;

        __syncthreads();

        //for the first column, all data are written into I
        for (j = lane; j < bV; j += WarpSize) {
            I[col * N1MAX + j] = AIndex[j + fcol_s];
        }//for j

        __syncthreads();

        //for other columns
        for (othercol = 1; othercol < jPTR[col]; othercol++) {
            ocol = J[col * N2MAX + othercol];
            othercol_s = APtr[ocol];
            othercol_e = APtr[ocol + 1];

            bV1 = othercol_e - othercol_s;
            //if the element in other column does not exist in I, append it to I
            for (j = lane; j < bV1; j += WarpSize) {
                I_s[tid * seg + j] = AIndex[j + othercol_s];
                for (j1 = 0; j1 < counter_s[tid]; j1++) {
                    if (I_s[tid * seg + j] == I[col * N1MAX + j1]) {
                        I_s[tid * seg + j] = -1;
                        break;
                    }
                }
            }//for j

            __syncthreads();

            if (lane == 0) {
                for (j = 0; j < bV1; j++) {
                    if (I_s[tid * seg + j] != -1) {
                        I[col * N1MAX + counter_s[tid]] = I_s[tid * seg + j];
                        //counter++;
                        counter_s[tid] += 1;
                    }
                }
            }

            __syncthreads();
        }//for othercol

        iPTR[col] = counter_s[tid];


        //ODD_EVEN
        //if(lane==0){
        for (int i = 1; i <= counter_s[tid]; i++) {
            if (i % 2 == 1) {
                for (int j1 = lane; j1 < counter_s[tid]; j1 += WarpSize) {
                    if (((j1 + 1) % 2 == 1) && (I[col * N1MAX + j1] > I[col * N1MAX + j1 + 1]) &&
                        (j1 + 1) < counter_s[tid]) {
                        bV1 = I[col * N1MAX + j1];
                        I[col * N1MAX + j1] = I[col * N1MAX + j1 + 1];
                        I[col * N1MAX + j1 + 1] = bV1;
                    }
                }
                __syncthreads();
            } else {
                for (int j2 = lane; j2 < counter_s[tid]; j2 += WarpSize) {
                    if (((j2 + 1) % 2 == 0) && (I[col * N1MAX + j2] > I[col * N1MAX + j2 + 1]) &&
                        (j2 + 1) < counter_s[tid]) {
                        bV1 = I[col * N1MAX + j2];
                        I[col * N1MAX + j2] = I[col * N1MAX + j2 + 1];
                        I[col * N1MAX + j2 + 1] = bV1;
                    }
                }
                __syncthreads();
            }
        }

        // }
    }//for col
}

template<unsigned int WarpSize, unsigned int SIZE_I_SHARED>
__global__ void computeIShared_Symbol_SpMMv1(int *APtr, int *AIndex, int nCol,
                                             int *I, int *iPTR, int N1MAX, int *J, int *jPTR, int N2MAX, int *atomic) {

    __shared__ int I_s[SIZE_I_SHARED];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp
    int tid = threadIdx.x / WarpSize;

    int col, fcol, fcol_s, fcol_e, bV, bV1, jN, tN;
    int ocol, othercol, othercol_s, othercol_e, j, j1, j2;

    int seg = SIZE_I_SHARED / (blockDim.x / WarpSize);
    int TSEG = seg * tid;
    int JBnd;
    int IBnd;

    for (col = warp_id; col < nCol; col += offset) {

        JBnd = col * N2MAX;
        IBnd = col * N1MAX;

        //read the columns
        fcol = J[JBnd];
        fcol_s = APtr[fcol];
        fcol_e = APtr[fcol + 1];
        bV = fcol_e - fcol_s;

        atomic[col] = bV;

        //for the first column, all data are written into I
        for (j = lane; j < bV; j += WarpSize) {
            I_s[TSEG + j] = AIndex[j + fcol_s];
        }//for j

        __syncthreads();


        jN = jPTR[col];
        //for other columns
        for (othercol = 1; othercol < jN; othercol++) {
            ocol = J[JBnd + othercol];
            othercol_s = APtr[ocol];
            othercol_e = APtr[ocol + 1];

            bV1 = othercol_e - othercol_s;
            //if the element in other column does not exist in I, append it to I
            tN = atomic[col];
            for (j = lane; j < bV1; j += WarpSize) {
                int idx = AIndex[j + othercol_s];
                int flag = 1;
                for (j1 = 0; j1 < tN; j1++) {
                    if (idx == I_s[TSEG + j1]) {
                        flag = -1;
                        break;
                    }
                }

                if (flag == 1) {
                    j2 = atomicAdd(&atomic[col], 1);
                    I_s[TSEG + j2] = idx;
                }
            }//for j

            __syncthreads();


        }//for othercol

        iPTR[col] = atomic[col];

        tN = iPTR[col];
        //ODD_EVEN
        for (int i = 1; i <= tN; i++) {
            if (i % 2 == 1) {
                for (int j1 = lane; j1 < tN; j1 += WarpSize) {
                    if ((j1 * 2 + 1) < tN && (I_s[TSEG + j1 * 2] > I_s[TSEG + j1 * 2 + 1])) {
                        bV1 = I_s[TSEG + j1 * 2];
                        I_s[TSEG + j1 * 2] = I_s[TSEG + j1 * 2 + 1];
                        I_s[TSEG + j1 * 2 + 1] = bV1;
                    }
                }
                __syncthreads();
            } else {
                for (int j2 = lane; j2 < tN; j2 += WarpSize) {
                    if ((j2 * 2 + 2) < tN && (I_s[TSEG + j2 * 2 + 1] > I_s[TSEG + j2 * 2 + 2])) {
                        bV1 = I_s[TSEG + j2 * 2 + 2];
                        I_s[TSEG + j2 * 2 + 2] = I_s[TSEG + j2 * 2 + 1];
                        I_s[TSEG + j2 * 2 + 1] = bV1;
                    }
                }
                __syncthreads();
            }
            __syncthreads();
        }


        //copy shared memory to I
        for (j = lane; j < tN; j += WarpSize) {
            I[IBnd + j] = I_s[TSEG + j];
        }

    }//for col
}

template<unsigned int WarpSize, unsigned int SIZE_I_SHARED>
__global__ void computeIShared_iter_Symbol_SpMMv1(int *APtr, int *AIndex, int nCol,
                                                  int *I, int *iPTR, int N1MAX, int *J, int *jPTR, int N2MAX,
                                                  int *atomic) {

    __shared__ int I_s[SIZE_I_SHARED];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp
    int tid = threadIdx.x / WarpSize;

    int col, fcol, fcol_s, fcol_e, bV, bV1, jN, tN;
    int ocol, othercol, othercol_s, othercol_e, j, j1, j2;

    int seg = SIZE_I_SHARED / (blockDim.x / WarpSize);
    int TSEG = seg * tid;
    int JBnd;
    int IBnd;

    for (col = warp_id; col < nCol; col += offset) {

        JBnd = col * N2MAX;
        IBnd = col * N1MAX;

        //read the columns
        fcol = J[JBnd];
        fcol_s = APtr[fcol];
        fcol_e = APtr[fcol + 1];
        bV = fcol_e - fcol_s;

        atomic[col] = bV;

        //for the first column, all data are written into I
        for (j = lane; j < bV; j += WarpSize) {
            I_s[TSEG + j] = AIndex[j + fcol_s];
        }//for j

        __syncthreads();


        jN = jPTR[col];
        //for other columns
        for (othercol = 1; othercol < jN; othercol++) {
            ocol = J[JBnd + othercol];
            othercol_s = APtr[ocol];
            othercol_e = APtr[ocol + 1];

            bV1 = othercol_e - othercol_s;
            //if the element in other column does not exist in I, append it to I
            tN = atomic[col];
            for (j = lane; j < bV1; j += WarpSize) {
                int idx = AIndex[j + othercol_s];
                int flag = 1;
                for (j1 = 0; j1 < tN; j1++) {
                    if (idx == I_s[TSEG + j1]) {
                        flag = -1;
                        break;
                    }
                }

                if (flag == 1) {
                    j2 = atomicAdd(&atomic[col], 1);
                    I_s[TSEG + j2] = idx;
                }
            }//for j

            __syncthreads();


        }//for othercol

        iPTR[col] = atomic[col];

        tN = iPTR[col];
        //ODD_EVEN
        for (int i = 1; i <= tN; i++) {
            if (i % 2 == 1) {
                for (int j1 = lane; j1 < tN; j1 += WarpSize) {
                    if ((j1 * 2 + 1) < tN && (I_s[TSEG + j1 * 2] > I_s[TSEG + j1 * 2 + 1])) {
                        bV1 = I_s[TSEG + j1 * 2];
                        I_s[TSEG + j1 * 2] = I_s[TSEG + j1 * 2 + 1];
                        I_s[TSEG + j1 * 2 + 1] = bV1;
                    }
                }
                __syncthreads();
            } else {
                for (int j2 = lane; j2 < tN; j2 += WarpSize) {
                    if ((j2 * 2 + 2) < tN && (I_s[TSEG + j2 * 2 + 1] > I_s[TSEG + j2 * 2 + 2])) {
                        bV1 = I_s[TSEG + j2 * 2 + 2];
                        I_s[TSEG + j2 * 2 + 2] = I_s[TSEG + j2 * 2 + 1];
                        I_s[TSEG + j2 * 2 + 1] = bV1;
                    }
                }
                __syncthreads();
            }
            if (WarpSize > 32) {
                __syncthreads();
            }
        }

        //copy shared memory to I
        for (j = lane; j < tN; j += WarpSize) {
            I[IBnd + j] = I_s[TSEG + j];
        }

    }//for col
}

template<unsigned int WarpSize>
__global__ void computeI_Symbol_SpMMv1(int *APtr, int *AIndex, int nCol,
                                       int *I, int *iPTR, int N1MAX, int *J, int *jPTR, int N2MAX, int *atomic) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp
    //int tid = threadIdx.x / WarpSize;

    int col, fcol, fcol_s, fcol_e, bV, bV1, jN, tN;
    int ocol, othercol, othercol_s, othercol_e, j, j1, j2;

    int JBnd;
    int IBnd;

    //int seg = SIZE_I_SHARED/CounterSize;

    for (col = warp_id; col < nCol; col += offset) {
        JBnd = col * N2MAX;
        IBnd = col * N1MAX;

        //read the columns
        fcol = J[JBnd];
        fcol_s = APtr[fcol];
        fcol_e = APtr[fcol + 1];
        bV = fcol_e - fcol_s;

        atomic[col] = bV;

        //for the first column, all data are written into I
        for (j = lane; j < bV; j += WarpSize) {
            I[IBnd + j] = AIndex[j + fcol_s];
        }//for j

        __syncthreads();

        //for other columns
        jN = jPTR[col];
        for (othercol = 1; othercol < jN; othercol++) {
            ocol = J[JBnd + othercol];
            othercol_s = APtr[ocol];
            othercol_e = APtr[ocol + 1];

            bV1 = othercol_e - othercol_s;
            //if the element in other column does not exist in I, append it to I
            tN = atomic[col];
            for (j = lane; j < bV1; j += WarpSize) {
                int idx = AIndex[j + othercol_s];
                int flag = 1;
                for (j1 = 0; j1 < tN; j1++) {
                    if (idx == I[IBnd + j1]) {
                        flag = -1;
                        break;
                    }
                }

                if (flag == 1) {
                    j2 = atomicAdd(&atomic[col], 1);
                    I[IBnd + j2] = idx;
                }
            }//for j


            __syncthreads();

        }//for othercol

        iPTR[col] = atomic[col];

        tN = iPTR[col];
        //ODD_EVEN
        for (int i = 1; i <= tN; i++) {
            if (i % 2 == 1) {
                for (int j1 = lane; j1 < tN; j1 += WarpSize) {
                    if ((j1 * 2 + 1) < tN && (I[IBnd + j1 * 2] > I[IBnd + j1 * 2 + 1])) {
                        bV1 = I[IBnd + j1 * 2];
                        I[IBnd + j1 * 2] = I[IBnd + j1 * 2 + 1];
                        I[IBnd + j1 * 2 + 1] = bV1;
                    }
                }
                __syncthreads();
            } else {
                for (int j2 = lane; j2 < tN; j2 += WarpSize) {
                    if ((j2 * 2 + 2) < tN && (I[IBnd + j2 * 2 + 1] > I[IBnd + j2 * 2 + 2])) {
                        bV1 = I[IBnd + j2 * 2 + 2];
                        I[IBnd + j2 * 2 + 2] = I[IBnd + j2 * 2 + 1];
                        I[IBnd + j2 * 2 + 1] = bV1;
                    }
                }
                __syncthreads();
            }
            __syncthreads();
        }

    }//for col
}

template<unsigned int WarpSize>
__global__ void OE_ParallelSort(int nCol, int *I, int *iPTR, int N1MAX) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp

    int col, i, j1, j2, bV1, SZ;


    for (col = warp_id; col < nCol; col += offset) {
        SZ = iPTR[col];
        //ODD_EVEN
        for (i = 1; i <= SZ; i++) {
            if (i % 2 == 1) {
                for (j1 = lane; j1 < SZ; j1 += WarpSize) {
                    if (((j1 + 1) % 2 == 1) && (I[col * N1MAX + j1] > I[col * N1MAX + j1 + 1]) && (j1 + 1) < SZ) {
                        bV1 = I[col * N1MAX + j1];
                        I[col * N1MAX + j1] = I[col * N1MAX + j1 + 1];
                        I[col * N1MAX + j1 + 1] = bV1;
                    }
                }
                __syncthreads();
            } else {
                for (j2 = lane; j2 < SZ; j2 += WarpSize) {
                    if (((j2 + 1) % 2 == 0) && (I[col * N1MAX + j2] > I[col * N1MAX + j2 + 1]) && (j2 + 1) < SZ) {
                        bV1 = I[col * N1MAX + j2];
                        I[col * N1MAX + j2] = I[col * N1MAX + j2 + 1];
                        I[col * N1MAX + j2 + 1] = bV1;
                    }
                }
                __syncthreads();
            }
        }
    }//for col
}

template<unsigned int WarpSize, unsigned int CounterSize>
__global__ void computeI1_SSPAIv10(int *APtr, int *AIndex, int nCol,
                                   int *I, int *iPTR, int N1MAX, int *J, int *jPTR, int N2MAX) {

    __shared__ int counter_s[CounterSize];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp
    int tid = threadIdx.x / WarpSize;

    int col, fcol, fcol_s, fcol_e, bV, bV1;
    int ocol, othercol, othercol_s, othercol_e, j, j1;

    //int seg = SIZE_I_SHARED/CounterSize;

    for (col = warp_id; col < nCol; col += offset) {
        //read the columns
        fcol = J[col * N2MAX];
        fcol_s = APtr[fcol];
        fcol_e = APtr[fcol + 1];
        bV = fcol_e - fcol_s;

        counter_s[tid] = bV;

        __syncthreads();

        //for the first column, all data are written into I
        for (j = lane; j < bV; j += WarpSize) {
            I[col * N1MAX + j] = AIndex[j + fcol_s];
        }//for j

        __syncthreads();


        //for other columns
        for (othercol = 1; othercol < jPTR[col]; othercol++) {
            ocol = J[col * N2MAX + othercol];
            othercol_s = APtr[ocol];
            othercol_e = APtr[ocol + 1];

            bV1 = othercol_e - othercol_s;
            //if the element in other column does not exist in I, append it to I
            int counter = 0;
            for (j = 0; j < bV1; j += 1) {
                int idx = AIndex[j + othercol_s];
                int flag = 1;
                for (j1 = 0; j1 < counter_s[tid]; j1++) {
                    if (idx == I[col * N1MAX + j1]) {
                        flag = -1;
                        break;
                    }
                }

                if (flag == 1) {
                    I[col * N1MAX + counter_s[tid] + counter] = idx;
                    counter++;
                }
            }//for j

            counter_s[tid] += counter;
            __syncthreads();

        }//for othercol

        iPTR[col] = counter_s[tid];


        //}
    }
}

template<unsigned int WarpSize>
__global__ void ComputeTildeACSR_SSPAIv10(double *A1, double *AData, int *APtr, int *AIndex, int nCol,
                                          int *I, int *iPTR, int *J, int *jPTR, int N1MAX, int N2MAX) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);//index of threads in warp

    int col, i, irow, j, jcol, jcol_b, jcol_e, jcol1;
    int AM, AN;
    double idata;

    for (col = warp_id; col < nCol; col += offset) {
        AM = iPTR[col];
        AN = jPTR[col];
        for (i = 0; i < AM; i++) {
            irow = I[col * N1MAX + i];
            for (j = lane; j < AN; j += WarpSize) {
                jcol = J[col * N2MAX + j];
                jcol_b = APtr[jcol];
                jcol_e = APtr[jcol + 1];

                idata = 0.0;
                for (jcol1 = jcol_b; jcol1 < jcol_e; jcol1++) {
                    if (AIndex[jcol1] == irow) {
                        idata = AData[jcol1];
                        break;
                    }
                }

                A1[col * N1MAX * N2MAX + i * N2MAX + j] = idata;

            }//for j

            __syncthreads();
        }//for i

    }//for col
}

template<unsigned int WarpSize>
__global__ void ComputeTildeE_SSPAIv10(int *E, int *I, int *iPTR, int N1MAX, int nCol) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);//index of threads in warp

    int col, i, AM;

    for (col = warp_id; col < nCol; col += offset) {
        E[col] = -1;
        AM = iPTR[col];
        for (i = lane; i < AM; i += WarpSize) {
            if (I[col * N1MAX + i] == col) {
                E[col] = i;
                break;
            }
        }//for i
    }//for col
}

template<unsigned int WarpSize>
__global__ void ComputeTildeE2_SSPAIv10(int *E, int *I, int *iPTR, int N1MAX, int nCol, int sK) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);//index of threads in warp

    int col, i, AM, colK;

    for (col = warp_id; col < nCol; col += offset) {
        colK = sK + col;
        E[col] = -1;
        AM = iPTR[col];
        for (i = lane; i < AM; i += WarpSize) {
            if (I[col * N1MAX + i] == colK) {
                E[col] = i;
                break;
            }
        }//for i
    }//for col
}

template<unsigned int WarpSize>
__global__ void Sol_SSPAIv10(double *Q, double *R, double *X, int *E, int *jPTR, int N1MAX, int N2MAX, int nCol) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);//index of threads in warp

    int col, AN, i, j;

    //compute Q^TE
    for (col = warp_id; col < nCol; col += offset) {
        AN = jPTR[col];
        if (E[col] == -1) {
            for (i = lane; i < AN; i += WarpSize) {
                X[col * N2MAX + i] = 0.0;
            }//for i
        } else {
            for (i = lane; i < AN; i += WarpSize) {
                X[col * N2MAX + i] = Q[col * N1MAX * N2MAX + E[col] * N2MAX + i];
            }//for i
        }

        __syncthreads();

        //solving the upper triangular system
        for (i = AN - 1; i >= 0; i--) {
            if (lane == 0) { X[col * N2MAX + i] /= R[col * N2MAX * N2MAX + i * N2MAX + i]; }

            __syncthreads();

            for (j = lane; j < i; j += WarpSize) {
                X[col * N2MAX + j] -= R[col * N2MAX * N2MAX + j * N2MAX + i] * X[col * N2MAX + i];//csr
                //X[col*N2MAX+j] -= R[col*N2MAX*N2MAX+i*N2MAX+j] * X[col*N2MAX+i]; //csc
            }// for j
            __syncthreads();
        }//for i

    }//for col

}

template<unsigned int WarpSize, unsigned int SIZE_R_SHARED>
__global__ void QR_RShared_SSPAIv10(double *Q, double *R, int *iPTR, int *jPTR, int N1MAX, int N2MAX, int nCol) {
    __shared__ double R_s[SIZE_R_SHARED];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);//index of threads in warp

    int tid = threadIdx.x / WarpSize;

    int col, AM, AN;
    int i, j, k;
    double rii, tR;

    int segR = SIZE_R_SHARED * WarpSize / blockDim.x;

    for (col = warp_id; col < nCol; col += offset) {
        //for each col corresponding to a tilde of A
        AM = iPTR[col];
        AN = jPTR[col];

        for (i = 0; i < AN; i++) {

            /*Compute R in parallel an put them into shared memory*/
            for (j = lane + i; j < AN; j += WarpSize) {
                tR = 0.0;
                for (k = 0; k < AM; k++) {
                    tR += Q[col * N1MAX * N2MAX + i + k * N2MAX] * Q[col * N1MAX * N2MAX + j + k * N2MAX];
                }//for k

                R_s[tid * segR + j - i] = tR;
            }//for j

            __syncthreads();

            rii = sqrt(R_s[tid * segR]);
            //normalize column i of Q
            for (j = lane; j < AM; j += WarpSize) {
                Q[col * N1MAX * N2MAX + j * N2MAX + i] /= rii;
            }//for j

            __syncthreads();

            //compute projection factors
            for (j = lane + i; j < AN; j += WarpSize) {
                R_s[tid * segR + j - i] /= rii;
                R[col * N2MAX * N2MAX + i * N2MAX + j] = R_s[tid * segR + j - i];
            }//for j

            __syncthreads();

            for (j = lane + i + 1; j < AN; j += WarpSize) {
                for (k = 0; k < AM; k++) {
                    Q[col * N1MAX * N2MAX + k * N2MAX + j] -=
                            R_s[tid * segR + j - i] * Q[col * N1MAX * N2MAX + k * N2MAX + i];
                }
            }

            __syncthreads();

        }//for i

    }//for col

}

template<unsigned int WarpSize>
__global__ void modifyData4_SSPAIv10(double *mData, int *mPtr,
                                     double *X, int *jPTR, int n2max, int nCol) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp

    int col, j;
    int col_s;

    for (col = warp_id; col < nCol; col += offset) {
        col_s = mPtr[col];
        for (j = lane; j < jPTR[col]; j += WarpSize) {
            mData[col_s + j] = X[col * n2max + j];
        }
    }
}

template<unsigned int WarpSize>
__global__ void modifyIndexAndData_SSPAIv10(double *mData, int *mIndex, int *mPtr,
                                            double *X, int *J, int *jPTR, int n2max, int nCol) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp

    int col, j;
    int col_s;

    for (col = warp_id; col < nCol; col += offset) {
        col_s = mPtr[col];
        for (j = lane; j < jPTR[col]; j += WarpSize) {
            mData[col_s + j] = X[col * n2max + j];
            mIndex[col_s + j] = J[col * n2max + j];
        }
    }
}

template<unsigned int WarpSize>
__global__ void modifyData24_SSPAIv10(double *mData, int *mPtr,
                                      double *X, int *jPTR, int n2max, int nCol, int sK) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp

    int col, j;
    int col_s, colK;

    for (col = warp_id; col < nCol; col += offset) {
        colK = col + sK;
        col_s = mPtr[colK];
        for (j = lane; j < jPTR[col]; j += WarpSize) {
            mData[col_s + j] = X[col * n2max + j];
        }
    }
}

template<unsigned int WarpSize>
__global__ void modifyIndexAndData2_SSPAIv10(double *mData, int *mIndex, int *mPtr,
                                             double *X, int *J, int *jPTR, int n2max, int nCol, int sK) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp

    int col, j;
    int col_s, colK;

    for (col = warp_id; col < nCol; col += offset) {
        colK = col + sK;
        col_s = mPtr[colK];
        for (j = lane; j < jPTR[col]; j += WarpSize) {
            mData[col_s + j] = X[col * n2max + j];
            mIndex[col_s + j] = J[col * n2max + j];
        }
    }
}

template<unsigned int WarpSize>
__global__ void modifyIndexAndData3_SSPAIv10(double *mData, int *mIndex, int *mPtr,
                                             double *mTmpData, int *mTmpIndex, int *mTmpPtr, int *aPtr, int nCol) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
    int offset = blockDim.x / WarpSize * gridDim.x;
    int warp_id = gid / WarpSize; //global warp index
    int lane = gid & (WarpSize - 1);// index of threads in warp

    int col, j;
    int col_s, col_s1;

    for (col = warp_id; col < nCol; col += offset) {
        col_s = mPtr[col];
        col_s1 = aPtr[col];
        for (j = lane; j < mTmpPtr[col]; j += WarpSize) {
            mData[col_s + j] = mTmpData[col_s1 + j];
            mIndex[col_s + j] = mTmpIndex[col_s1 + j];
        }
    }
}

//==============================================================================
// 打印多DCU性能统计
//==============================================================================

void printPerformanceStats(MultiDCU_Context *ctx, float totalTime) {
    printf("\n========================================\n");
    printf("性能统计\n");
    printf("========================================\n");
    printf("DCU数量: %d\n", ctx->numDCUs);
    printf("总计算时间: %8.4f ms\n", totalTime);

    if (ctx->numDCUs > 0) {
        printf("平均每DCU时间: %8.4f ms\n", totalTime / ctx->numDCUs);
    }

    // 打印各DCU的列分布
    printf("\n列分布:\n");
    for (int i = 0; i < ctx->numDCUs; i++) {
        int cols = ctx->colEnd[i] - ctx->colStart[i];
        int nnz = (ctx->devCSC_M_local[i]) ? ctx->devCSC_M_local[i]->nonzeroes : 0;
        printf("  DCU %d: 列 [%d, %d) = %d 列, %d 非零元\n",
               i, ctx->colStart[i], ctx->colEnd[i], cols, nnz);
    }
    printf("========================================\n\n");
}

//==============================================================================
// 启用DCU之间的P2P内存访问（可选优化）
//==============================================================================

void enableP2P(MultiDCU_Context *ctx) {
    printf("尝试启用DCU之间的P2P内存访问...\n");

    for (int i = 0; i < ctx->numDCUs; i++) {
        for (int j = 0; j < ctx->numDCUs; j++) {
            if (i != j) {
                CHECK_HIP_ERROR(hipSetDevice(i));
                int canAccessPeer = 0;
                hipError_t err = hipDeviceCanAccessPeer(&canAccessPeer, i, j);

                if (err == hipSuccess && canAccessPeer) {
                    err = hipDeviceEnablePeerAccess(j, 0);
                    if (err == hipSuccess) {
                        printf("  ✓ DCU %d -> DCU %d P2P访问已启用\n", i, j);
                    } else if (err != hipErrorPeerAccessAlreadyEnabled) {
                        printf("  ✗ DCU %d -> DCU %d P2P访问启用失败\n", i, j);
                    }
                } else {
                    printf("  - DCU %d 和 DCU %d 不支持P2P访问\n", i, j);
                }
            }
        }
    }
    printf("\n");
}

//==============================================================================
// 多DCU初始化函数
//==============================================================================

MultiDCU_Context* initMultiDCU(int requestedDCUs = -1) {
    MultiDCU_Context *ctx = (MultiDCU_Context*)malloc(sizeof(MultiDCU_Context));

    // 查询可用DCU数量
    int availableDCUs = 0;
    CHECK_HIP_ERROR(hipGetDeviceCount(&availableDCUs));

    if (availableDCUs == 0) {
        printf("错误: 未检测到DCU设备\n");
        exit(EXIT_FAILURE);
    }

    // 确定使用的DCU数量
    if (requestedDCUs <= 0 || requestedDCUs > availableDCUs) {
        ctx->numDCUs = availableDCUs;
    } else {
        ctx->numDCUs = requestedDCUs;
    }

    printf("========================================\n");
    printf("多DCU配置\n");
    printf("========================================\n");
    printf("可用DCU数量: %d\n", availableDCUs);
    printf("使用DCU数量: %d\n", ctx->numDCUs);
    printf("========================================\n");

    // 分配设备ID数组
    ctx->deviceIds = (int*)malloc(sizeof(int) * ctx->numDCUs);
    ctx->devCSC_A = (CSC_Matrix**)malloc(sizeof(CSC_Matrix*) * ctx->numDCUs);
    ctx->devCSC_M_local = (CSC_Matrix**)malloc(sizeof(CSC_Matrix*) * ctx->numDCUs);
    ctx->colStart = (int*)malloc(sizeof(int) * ctx->numDCUs);
    ctx->colEnd = (int*)malloc(sizeof(int) * ctx->numDCUs);
    ctx->streams = (hipStream_t*)malloc(sizeof(hipStream_t) * ctx->numDCUs);

    // 初始化每个DCU
    for (int i = 0; i < ctx->numDCUs; i++) {
        ctx->deviceIds[i] = i;
        CHECK_HIP_ERROR(hipSetDevice(i));

        // 查询设备属性
        hipDeviceProp_t prop;
        CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, i));
        printf("DCU %d: %s (计算能力 %d.%d, Wavefront大小 %d)\n",
               i, prop.name, prop.major, prop.minor, prop.warpSize);

        // 创建流
        CHECK_HIP_ERROR(hipStreamCreate(&ctx->streams[i]));
    }
    printf("========================================\n\n");

    // 尝试启用P2P内存访问以优化多DCU间的数据传输
    enableP2P(ctx);

    return ctx;
}

//==============================================================================
// 列分割函数：将矩阵列分配给各个DCU
//==============================================================================

void distributeColumns(MultiDCU_Context *ctx, int totalCols) {
    if (!ctx || ctx->numDCUs <= 0) {
        printf("错误: 无效的上下文\n");
        return;
    }

    if (totalCols <= 0) {
        printf("错误: 总列数必须大于0\n");
        return;
    }

    int colsPerDCU = totalCols / ctx->numDCUs;
    int remainder = totalCols % ctx->numDCUs;

    printf("========================================\n");
    printf("列分配策略\n");
    printf("========================================\n");
    printf("总列数: %d\n", totalCols);
    printf("DCU数量: %d\n", ctx->numDCUs);
    printf("每DCU基础列数: %d\n", colsPerDCU);
    printf("余数列: %d\n", remainder);
    printf("----------------------------------------\n");

    int currentCol = 0;
    for (int i = 0; i < ctx->numDCUs; i++) {
        ctx->colStart[i] = currentCol;
        // 前面的DCU多分配一列（如果有余数）
        int extraCol = (i < remainder) ? 1 : 0;
        int colsForThisDCU = colsPerDCU + extraCol;
        ctx->colEnd[i] = currentCol + colsForThisDCU;
        currentCol = ctx->colEnd[i];

        printf("DCU %d: 列 [%6d, %6d) = %6d 列 (%.1f%%)\n",
               i, ctx->colStart[i], ctx->colEnd[i], colsForThisDCU,
               100.0 * colsForThisDCU / totalCols);
    }

    // 验证分配
    if (currentCol != totalCols) {
        printf("\n警告: 列分配不完整！分配了 %d 列，期望 %d 列\n",
               currentCol, totalCols);
    }

    printf("========================================\n\n");
}

//==============================================================================
// 验证矩阵结构的完整性
//==============================================================================

bool validateMatrix(CSC_Matrix *mat, const char *name) {
    if (!mat) {
        printf("错误: 矩阵 %s 为空指针\n", name);
        return false;
    }

    if (mat->n <= 0 || mat->nCol <= 0 || mat->nRow <= 0) {
        printf("错误: 矩阵 %s 维度无效 (n=%d, nCol=%d, nRow=%d)\n",
               name, mat->n, mat->nCol, mat->nRow);
        return false;
    }

    if (mat->nonzeroes < 0) {
        printf("错误: 矩阵 %s 非零元数量无效 (%d)\n", name, mat->nonzeroes);
        return false;
    }

    if (mat->nonzeroes > 0) {
        if (!mat->mPtr || !mat->mIndex || !mat->mData) {
            printf("错误: 矩阵 %s 数据指针为空\n", name);
            return false;
        }
    }

    return true;
}

//==============================================================================
// 复制矩阵A到所有DCU（只读共享）
//==============================================================================

void replicateMatrixA(MultiDCU_Context *ctx, CSC_Matrix *CSC_A) {
    if (!ctx || !CSC_A) {
        printf("错误: 无效的上下文或矩阵指针\n");
        return;
    }

    // 验证矩阵
    if (!validateMatrix(CSC_A, "CSC_A (replicateMatrixA)")) {
        printf("错误: 矩阵验证失败，无法复制到DCU\n");
        exit(EXIT_FAILURE);
    }

    printf("========================================\n");
    printf("复制矩阵A到所有DCU\n");
    printf("========================================\n");
    printf("矩阵维度: %d x %d\n", CSC_A->n, CSC_A->n);
    printf("非零元数: %d\n", CSC_A->nonzeroes);
    printf("稀疏度: %.2f%%\n",
           100.0 * CSC_A->nonzeroes / ((double)CSC_A->n * CSC_A->n));
    printf("----------------------------------------\n");

    for (int i = 0; i < ctx->numDCUs; i++) {
        CHECK_HIP_ERROR(hipSetDevice(i));

        ctx->devCSC_A[i] = (CSC_Matrix*)malloc(sizeof(CSC_Matrix));
        if (!ctx->devCSC_A[i]) {
            printf("错误: DCU %d - 无法分配主机内存\n", i);
            exit(EXIT_FAILURE);
        }

        ctx->devCSC_A[i]->n = CSC_A->n;
        ctx->devCSC_A[i]->nonzeroes = CSC_A->nonzeroes;
        ctx->devCSC_A[i]->nCol = CSC_A->n;
        ctx->devCSC_A[i]->nRow = CSC_A->n;

        // 在设备上分配内存
        CHECK_HIP_ERROR(hipMalloc((void**)&ctx->devCSC_A[i]->mPtr,
                                   sizeof(int) * (CSC_A->nCol + 1)));
        CHECK_HIP_ERROR(hipMalloc((void**)&ctx->devCSC_A[i]->mIndex,
                                   sizeof(int) * CSC_A->nonzeroes));
        CHECK_HIP_ERROR(hipMalloc((void**)&ctx->devCSC_A[i]->mData,
                                   sizeof(double) * CSC_A->nonzeroes));

        // 异步复制数据（使用对应的流）
        CHECK_HIP_ERROR(hipMemcpyAsync(ctx->devCSC_A[i]->mData, CSC_A->mData,
                                       CSC_A->nonzeroes * sizeof(double),
                                       hipMemcpyHostToDevice, ctx->streams[i]));
        CHECK_HIP_ERROR(hipMemcpyAsync(ctx->devCSC_A[i]->mIndex, CSC_A->mIndex,
                                       CSC_A->nonzeroes * sizeof(int),
                                       hipMemcpyHostToDevice, ctx->streams[i]));
        CHECK_HIP_ERROR(hipMemcpyAsync(ctx->devCSC_A[i]->mPtr, CSC_A->mPtr,
                                       (CSC_A->nCol + 1) * sizeof(int),
                                       hipMemcpyHostToDevice, ctx->streams[i]));

        printf("  DCU %d: 矩阵A复制中... (%d 非零元)\n", i, CSC_A->nonzeroes);
    }

    // 同步所有DCU
    for (int i = 0; i < ctx->numDCUs; i++) {
        CHECK_HIP_ERROR(hipSetDevice(i));
        CHECK_HIP_ERROR(hipStreamSynchronize(ctx->streams[i]));
    }

    printf("矩阵A复制完成\n\n");
}

//==============================================================================
// 列范围SPAI计算函数（每个DCU调用此函数处理自己负责的列）
//==============================================================================

float StaticSPAIv20_ColumnRange(CSC_Matrix *devA, CSC_Matrix *devM,
                                 int colStart, int colEnd, int deviceId,
                                 hipStream_t stream) {
    if (!devA || !devM) return 0.0f;
    if (colStart < 0 || colEnd > devA->nCol || colStart >= colEnd) return 0.0f;

    CHECK_HIP_ERROR(hipSetDevice(deviceId));
    int localCols = colEnd - colStart;

    printf("DCU %d: Pre-GSSPAI processing [%d,%d)...\n", deviceId, colStart, colEnd);

    hipEvent_t start, stop;
    float elapsedTime, preTime = 0.0, computeTime = 0.0, postTime = 0.0;
    const int threadsPerBlock = 256;
    int blocksPerGrid = 15 * 8 * 4;

    printf("DCU %d: Compute n2max\n", deviceId);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    int *n2 = (int*)malloc(sizeof(int) * blocksPerGrid);
    int *dev_n2;
    CHECK_HIP_ERROR(hipMalloc((void**)&dev_n2, sizeof(int) * blocksPerGrid));

    cuComputeN2MAXwithSparityofA<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(
        devA->mPtr + colStart, localCols, dev_n2);

    CHECK_HIP_ERROR(hipMemcpy(n2, dev_n2, blocksPerGrid * sizeof(int), hipMemcpyDeviceToHost));
    int n2max = 0;
    for (int i = 0; i < blocksPerGrid; i++) if (n2max < n2[i]) n2max = n2[i];
    hipFree(dev_n2);
    free(n2);

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsedTime, start, stop);
    preTime += elapsedTime;
    hipEventDestroy(start);
    hipEventDestroy(stop);

    printf("DCU %d: n2max=%d\n", deviceId, n2max);
    printf("DCU %d: Compute n1max\n", deviceId);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    int WarpSize = (n2max <= 2) ? 2 : (n2max <= 4) ? 4 : (n2max <= 8) ? 8 : (n2max <= 16) ? 16 : 32;
    blocksPerGrid = (localCols - 1) / (threadsPerBlock / WarpSize) + 1;

    int *dev_n1, *dev_tI, *dev_atomic;
    CHECK_HIP_ERROR(hipMalloc((void**)&dev_n1, sizeof(int) * localCols));
    int ssize = min(n2max * 8, localCols);
    CHECK_HIP_ERROR(hipMalloc((void**)&dev_tI, sizeof(int) * localCols * ssize));
    CHECK_HIP_ERROR(hipMalloc((void**)&dev_atomic, sizeof(int) * localCols));

    if (n2max <= 2) {
        if (ssize <= 16) cuComputeN1withSparityofA_SpMMv1<2, 2048><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
        else if (ssize <= 32) cuComputeN1withSparityofA_SpMMv1<2, 4096><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
        else if (ssize <= 64) cuComputeN1withSparityofA_SpMMv1<2, 8192><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
        else cuComputeN1withSparityofA2_SpMMv1<2><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
    } else if (n2max <= 4) {
        if (ssize <= 32) cuComputeN1withSparityofA_SpMMv1<4, 2048><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
        else if (ssize <= 64) cuComputeN1withSparityofA_SpMMv1<4, 4096><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
        else if (ssize <= 128) cuComputeN1withSparityofA_SpMMv1<4, 8192><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
        else cuComputeN1withSparityofA2_SpMMv1<4><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
    } else if (n2max <= 8) {
        if (ssize <= 64) cuComputeN1withSparityofA_SpMMv1<8, 2048><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
        else if (ssize <= 128) cuComputeN1withSparityofA_SpMMv1<8, 4096><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
        else if (ssize <= 256) cuComputeN1withSparityofA_SpMMv1<8, 8192><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
        else cuComputeN1withSparityofA2_SpMMv1<8><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
    } else if (n2max <= 16) {
        if (ssize <= 128) cuComputeN1withSparityofA_SpMMv1<16, 2048><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
        else if (ssize <= 256) cuComputeN1withSparityofA_SpMMv1<16, 4096><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
        else if (ssize <= 512) cuComputeN1withSparityofA_SpMMv1<16, 8192><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
        else cuComputeN1withSparityofA2_SpMMv1<16><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
    } else {
        if (ssize <= 256) cuComputeN1withSparityofA_SpMMv1<32, 2048><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
        else if (ssize <= 512) cuComputeN1withSparityofA_SpMMv1<32, 4096><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
        else if (ssize <= 1024) cuComputeN1withSparityofA_SpMMv1<32, 8192><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
        else cuComputeN1withSparityofA2_SpMMv1<32><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr + colStart, devA->mIndex, devA->mPtr + colStart, devA->mIndex, dev_tI, ssize, localCols, dev_n1, dev_atomic);
    }
    hipFree(dev_tI);

    blocksPerGrid = 15 * 8 * 4;
    int *n1 = (int*)malloc(sizeof(int) * blocksPerGrid);
    int *dev_n1max;
    CHECK_HIP_ERROR(hipMalloc((void**)&dev_n1max, sizeof(int) * blocksPerGrid));

    cuComputeN1MAX<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(dev_n1, localCols, dev_n1max);

    CHECK_HIP_ERROR(hipMemcpy(n1, dev_n1max, blocksPerGrid * sizeof(int), hipMemcpyDeviceToHost));
    int n1max = 0;
    for (int i = 0; i < blocksPerGrid; i++) if (n1max < n1[i]) n1max = n1[i];
    hipFree(dev_n1);
    hipFree(dev_n1max);
    free(n1);

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsedTime, start, stop);
    preTime += elapsedTime;
    hipEventDestroy(start);
    hipEventDestroy(stop);

    printf("DCU %d: n1max=%d\n", deviceId, n1max);

    WarpSize = 32;
    blocksPerGrid = (localCols - 1) / (256 / WarpSize) + 1;

    printf("DCU %d: Compute-GSSPAI processing...\n", deviceId);

    int compareSize = 1024 * 1024 * 100 * 4 * 3;
    long realSize = (long)localCols * n1max * n2max;
    int itertions = (realSize - 1) / compareSize + 1;

    if (itertions != 1) {
        const int nCols = (localCols - 1) / itertions + 1;

        double *dev_tildeA, *dev_R, *dev_X;
        int *dev_J, *dev_jPTR, *dev_I, *dev_iPTR, *dev_E;

        CHECK_HIP_ERROR(hipMalloc((void**)&dev_J, sizeof(int) * nCols * n2max));
        CHECK_HIP_ERROR(hipMalloc((void**)&dev_jPTR, sizeof(int) * nCols));
        CHECK_HIP_ERROR(hipMalloc((void**)&dev_I, sizeof(int) * nCols * n1max));
        CHECK_HIP_ERROR(hipMalloc((void**)&dev_iPTR, sizeof(int) * nCols));
        CHECK_HIP_ERROR(hipMalloc((void**)&dev_tildeA, sizeof(double) * nCols * n1max * n2max));
        CHECK_HIP_ERROR(hipMalloc((void**)&dev_R, sizeof(double) * nCols * n2max * n2max));
        CHECK_HIP_ERROR(hipMalloc((void**)&dev_X, sizeof(double) * nCols * n2max));
        CHECK_HIP_ERROR(hipMalloc((void**)&dev_E, sizeof(int) * nCols));

        double *dev_mTmpData;
        CHECK_HIP_ERROR(hipMalloc((void**)&dev_mTmpData, sizeof(double) * devA->nonzeroes));

        for (int iter = 0; iter < itertions; iter++) {
            int sK = colStart + iter * nCols;
            int eGrid = nCols;
            if (iter == (itertions - 1)) eGrid = localCols - (itertions - 1) * nCols;

            blocksPerGrid = (eGrid - 1) / (threadsPerBlock / WarpSize) + 1;

            if (n2max <= 2) computeJ2_SSPAIv10<2><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
            else if (n2max <= 4) computeJ2_SSPAIv10<4><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
            else if (n2max <= 8) computeJ2_SSPAIv10<8><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
            else if (n2max <= 16) computeJ2_SSPAIv10<16><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
            else if (n2max <= 32) computeJ2_SSPAIv10<32><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
            else if (n2max <= 64) computeJ2_SSPAIv10<64><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
            else if (n2max <= 128) computeJ2_SSPAIv10<128><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
            else computeJ2_SSPAIv10<256><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);

            if (n2max <= 2) {
                if (n1max <= 8) computeIShared_iter_Symbol_SpMMv1<2, 1024><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 16) computeIShared_iter_Symbol_SpMMv1<2, 2048><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 32) computeIShared_iter_Symbol_SpMMv1<2, 4096><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 64) computeIShared_iter_Symbol_SpMMv1<2, 8192><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else computeI_iter_Symbol_SpMMv1<2><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n2max <= 4) {
                if (n1max <= 8) computeIShared_iter_Symbol_SpMMv1<4, 512><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 16) computeIShared_iter_Symbol_SpMMv1<4, 1024><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 32) computeIShared_iter_Symbol_SpMMv1<4, 2048><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 64) computeIShared_iter_Symbol_SpMMv1<4, 4096><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 128) computeIShared_iter_Symbol_SpMMv1<4, 8192><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else computeI_iter_Symbol_SpMMv1<4><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n2max <= 8) {
                if (n1max <= 8) computeIShared_iter_Symbol_SpMMv1<8, 256><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 16) computeIShared_iter_Symbol_SpMMv1<8, 512><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 32) computeIShared_iter_Symbol_SpMMv1<8, 1024><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 64) computeIShared_iter_Symbol_SpMMv1<8, 2048><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 128) computeIShared_iter_Symbol_SpMMv1<8, 4096><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 256) computeIShared_iter_Symbol_SpMMv1<8, 8192><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else computeI_iter_Symbol_SpMMv1<8><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n2max <= 16) {
                if (n1max <= 16) computeIShared_iter_Symbol_SpMMv1<16, 256><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 32) computeIShared_iter_Symbol_SpMMv1<16, 512><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 64) computeIShared_iter_Symbol_SpMMv1<16, 1024><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 128) computeIShared_iter_Symbol_SpMMv1<16, 2048><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 256) computeIShared_iter_Symbol_SpMMv1<16, 4096><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 512) computeIShared_iter_Symbol_SpMMv1<16, 8192><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else computeI_iter_Symbol_SpMMv1<16><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n2max <= 32) {
                if (n1max <= 32) computeIShared_iter_Symbol_SpMMv1<32, 256><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 64) computeIShared_iter_Symbol_SpMMv1<32, 512><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 128) computeIShared_iter_Symbol_SpMMv1<32, 1024><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 256) computeIShared_iter_Symbol_SpMMv1<32, 2048><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 512) computeIShared_iter_Symbol_SpMMv1<32, 4096><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 1024) computeIShared_iter_Symbol_SpMMv1<32, 8192><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else computeI_iter_Symbol_SpMMv1<32><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n2max <= 64) {
                if (n1max <= 64) computeIShared_iter_Symbol_SpMMv1<64, 256><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 128) computeIShared_iter_Symbol_SpMMv1<64, 512><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 256) computeIShared_iter_Symbol_SpMMv1<64, 1024><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 512) computeIShared_iter_Symbol_SpMMv1<64, 2048><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 1024) computeIShared_iter_Symbol_SpMMv1<64, 4096><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 2048) computeIShared_iter_Symbol_SpMMv1<64, 8192><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else computeI_iter_Symbol_SpMMv1<64><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else {
                if (n1max <= 128) computeIShared_iter_Symbol_SpMMv1<128, 256><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 256) computeIShared_iter_Symbol_SpMMv1<128, 512><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 512) computeIShared_iter_Symbol_SpMMv1<128, 1024><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 1024) computeIShared_iter_Symbol_SpMMv1<128, 2048><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 2048) computeIShared_iter_Symbol_SpMMv1<128, 4096><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else if (n1max <= 4096) computeIShared_iter_Symbol_SpMMv1<128, 8192><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                else computeI_iter_Symbol_SpMMv1<128><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            }

            if (n2max <= 2) ComputeTildeACSR_SSPAIv10<2><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
            else if (n2max <= 4) ComputeTildeACSR_SSPAIv10<4><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
            else if (n2max <= 8) ComputeTildeACSR_SSPAIv10<8><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
            else if (n2max <= 16) ComputeTildeACSR_SSPAIv10<16><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
            else if (n2max <= 32) ComputeTildeACSR_SSPAIv10<32><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
            else if (n2max <= 64) ComputeTildeACSR_SSPAIv10<64><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
            else if (n2max <= 128) ComputeTildeACSR_SSPAIv10<128><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
            else ComputeTildeACSR_SSPAIv10<256><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);

            if (n2max <= 2) QR_RShared_SSPAIv10<2, 256><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
            else if (n2max <= 4) QR_RShared_SSPAIv10<4, 256><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
            else if (n2max <= 8) QR_RShared_SSPAIv10<8, 256><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
            else if (n2max <= 16) QR_RShared_SSPAIv10<16, 256><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
            else if (n2max <= 32) QR_RShared_SSPAIv10<32, 256><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
            else if (n2max <= 64) QR_RShared_SSPAIv10<64, 256><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
            else if (n2max <= 128) {
                QR_RShared_SSPAIv10<128, 256><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
            } else {
                if (n2max <= 256) QR_RShared_SSPAIv10<256, 256><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
                else if (n2max <= 512) QR_RShared_SSPAIv10<256, 512><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
                else if (n2max <= 1024) QR_RShared_SSPAIv10<256, 1024><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
                else if (n2max <= 2048) QR_RShared_SSPAIv10<256, 2048><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
            }

            if (n2max <= 2) ComputeTildeE2_SSPAIv10<2><<<blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
            else if (n2max <= 4) ComputeTildeE2_SSPAIv10<4><<<blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
            else if (n2max <= 8) ComputeTildeE2_SSPAIv10<8><<<blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
            else if (n2max <= 16) ComputeTildeE2_SSPAIv10<16><<<blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
            else if (n2max <= 32) ComputeTildeE2_SSPAIv10<32><<<blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
            else if (n2max <= 64) ComputeTildeE2_SSPAIv10<64><<<blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
            else if (n2max <= 128) ComputeTildeE2_SSPAIv10<128><<<blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
            else ComputeTildeE2_SSPAIv10<256><<<blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);

            if (n2max <= 2) Sol_SSPAIv10<2><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X, dev_E, dev_jPTR, n1max, n2max, eGrid);
            else if (n2max <= 4) Sol_SSPAIv10<4><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X, dev_E, dev_jPTR, n1max, n2max, eGrid);
            else if (n2max <= 8) Sol_SSPAIv10<8><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X, dev_E, dev_jPTR, n1max, n2max, eGrid);
            else if (n2max <= 16) Sol_SSPAIv10<16><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X, dev_E, dev_jPTR, n1max, n2max, eGrid);
            else if (n2max <= 32) Sol_SSPAIv10<32><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X, dev_E, dev_jPTR, n1max, n2max, eGrid);
            else if (n2max <= 64) Sol_SSPAIv10<64><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X, dev_E, dev_jPTR, n1max, n2max, eGrid);
            else if (n2max <= 128) Sol_SSPAIv10<128><<<blocksPerBlock, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X, dev_E, dev_jPTR, n1max, n2max, eGrid);
            else Sol_SSPAIv10<256><<<blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X, dev_E, dev_jPTR, n1max, n2max, eGrid);

            if (n2max <= 2) modifyIndexAndData2_SSPAIv10<2><<<blocksPerGrid, threadsPerBlock>>>(dev_mTmpData, devA->mIndex, devA->mPtr, dev_X, dev_J, dev_jPTR, n2max, eGrid, sK);
            else if (n2max <= 4) modifyIndexAndData2_SSPAIv10<4><<<blocksPerGrid, threadsPerBlock>>>(dev_mTmpData, devA->mIndex, devA->mPtr, dev_X, dev_J, dev_jPTR, n2max, eGrid, sK);
            else if (n2max <= 8) modifyIndexAndData2_SSPAIv10<8><<<blocksPerGrid, threadsPerBlock>>>(dev_mTmpData, devA->mIndex, devA->mPtr, dev_X, dev_J, dev_jPTR, n2max, eGrid, sK);
            else if (n2max <= 16) modifyIndexAndData2_SSPAIv10<16><<<blocksPerGrid, threadsPerBlock>>>(dev_mTmpData, devA->mIndex, devA->mPtr, dev_X, dev_J, dev_jPTR, n2max, eGrid, sK);
            else modifyIndexAndData2_SSPAIv10<32><<<blocksPerGrid, threadsPerBlock>>>(dev_mTmpData, devA->mIndex, devA->mPtr, dev_X, dev_J, dev_jPTR, n2max, eGrid, sK);
        }

        hipFree(dev_atomic);
        hipFree(dev_J);
        hipFree(dev_jPTR);
        hipFree(dev_I);
        hipFree(dev_iPTR);
        hipFree(dev_tildeA);
        hipFree(dev_R);
        hipFree(dev_E);
        hipFree(dev_X);

        int *h_colPtr = (int*)malloc((localCols + 1) * sizeof(int));
        CHECK_HIP_ERROR(hipMemcpy(h_colPtr, devA->mPtr + colStart, (localCols + 1) * sizeof(int), hipMemcpyDeviceToHost));

        int localNonzeros = h_colPtr[localCols] - h_colPtr[0];
        int dataOffset = h_colPtr[0];

        devM->nCol = localCols;
        devM->nRow = devA->nRow;
        devM->n = devA->n;
        devM->nonzeroes = localNonzeros;

        CHECK_HIP_ERROR(hipMalloc((void**)&devM->mPtr, sizeof(int) * (localCols + 1)));
        for (int i = 0; i <= localCols; i++) h_colPtr[i] -= dataOffset;
        CHECK_HIP_ERROR(hipMemcpy(devM->mPtr, h_colPtr, (localCols + 1) * sizeof(int), hipMemcpyHostToDevice));
        free(h_colPtr);

        CHECK_HIP_ERROR(hipMalloc((void**)&devM->mIndex, sizeof(int) * localNonzeros));
        CHECK_HIP_ERROR(hipMemcpy(devM->mIndex, devA->mIndex + dataOffset, localNonzeros * sizeof(int), hipMemcpyDeviceToDevice));
        CHECK_HIP_ERROR(hipMalloc((void**)&devM->mData, sizeof(double) * localNonzeros));
        CHECK_HIP_ERROR(hipMemcpy(devM->mData, dev_mTmpData + dataOffset, localNonzeros * sizeof(double), hipMemcpyDeviceToDevice));

        hipFree(dev_mTmpData);
    }

    printf("DCU %d: Time - Pre: %.2f ms, Compute: %.2f ms, Post: %.2f ms, Total: %.2f ms\n",
           deviceId, preTime, computeTime, postTime, preTime + computeTime + postTime);

    return preTime + computeTime + postTime;
}

//==============================================================================
// 多DCU并行计算SPAI（主函数）
// 每个DCU独立计算自己负责的列
//==============================================================================

float StaticSPAIv20_MultiDCU(MultiDCU_Context *ctx, CSC_Matrix *devCSC_M_global) {
    float totalTime = 0.0;

    printf("========================================\n");
    printf("多DCU静态SPAI预条件子计算\n");
    printf("========================================\n");

    // 分配列给各个DCU
    distributeColumns(ctx, ctx->devCSC_A[0]->nCol);

    // 检查OpenMP线程数
    omp_set_num_threads(ctx->numDCUs);
    printf("使用 %d 个OpenMP线程进行并行计算\n\n", ctx->numDCUs);

    #pragma omp parallel num_threads(ctx->numDCUs)
    {
        int dcuId = omp_get_thread_num();

        // 验证线程ID有效性
        if (dcuId >= 0 && dcuId < ctx->numDCUs) {
            CHECK_HIP_ERROR(hipSetDevice(dcuId));

            printf("DCU %d: 线程 %d 开始计算列 [%d, %d)\n",
                   dcuId, dcuId, ctx->colStart[dcuId], ctx->colEnd[dcuId]);

            // 创建局部矩阵结构（只包含该DCU负责的列）
            CSC_Matrix *devA_local = ctx->devCSC_A[dcuId];
            ctx->devCSC_M_local[dcuId] = (CSC_Matrix*)malloc(sizeof(CSC_Matrix));
            CSC_Matrix *devM_local = ctx->devCSC_M_local[dcuId];

            // 调用列范围SPAI计算函数
            float localTime = StaticSPAIv20_ColumnRange(
                devA_local,
                devM_local,
                ctx->colStart[dcuId],
                ctx->colEnd[dcuId],
                dcuId,
                ctx->streams[dcuId]
            );

            #pragma omp critical
            {
                totalTime += localTime;
                printf("DCU %d: 计算耗时 %8.4f ms\n", dcuId, localTime);
            }
        } else {
            printf("错误: 无效的线程ID %d (预期范围: 0-%d)\n", dcuId, ctx->numDCUs - 1);
        }

        #pragma omp barrier  // 等待所有DCU完成
    }

    printf("========================================\n");
    printf("所有DCU计算完成\n");
    printf("========================================\n\n");

    return totalTime;
}

//==============================================================================
// 聚合结果：将各DCU的局部预条件子合并成全局矩阵
//==============================================================================

void aggregateResults(MultiDCU_Context *ctx, CSC_Matrix *devCSC_M_global) {
    printf("聚合多DCU计算结果...\n");

    // 验证输入
    if (!ctx || !devCSC_M_global || ctx->numDCUs <= 0) {
        printf("错误: 无效的上下文或全局矩阵指针\n");
        return;
    }

    // 计算全局矩阵的总非零元个数
    int totalNonzeros = 0;
    for (int i = 0; i < ctx->numDCUs; i++) {
        if (ctx->devCSC_M_local[i]) {
            totalNonzeros += ctx->devCSC_M_local[i]->nonzeroes;
        }
    }

    printf("  总非零元数: %d\n", totalNonzeros);

    devCSC_M_global->nonzeroes = totalNonzeros;
    devCSC_M_global->nCol = ctx->devCSC_A[0]->nCol;
    devCSC_M_global->nRow = ctx->devCSC_A[0]->nRow;
    devCSC_M_global->n = ctx->devCSC_A[0]->n;

    // 在第一个DCU上分配全局矩阵
    CHECK_HIP_ERROR(hipSetDevice(0));

    // 分配列指针数组（总是需要）
    CHECK_HIP_ERROR(hipMalloc((void**)&devCSC_M_global->mPtr,
                               sizeof(int) * (devCSC_M_global->nCol + 1)));

    // 只有当有非零元时才分配数据数组
    if (totalNonzeros > 0) {
        CHECK_HIP_ERROR(hipMalloc((void**)&devCSC_M_global->mIndex,
                                   sizeof(int) * totalNonzeros));
        CHECK_HIP_ERROR(hipMalloc((void**)&devCSC_M_global->mData,
                                   sizeof(double) * totalNonzeros));
    } else {
        devCSC_M_global->mIndex = NULL;
        devCSC_M_global->mData = NULL;
        printf("  警告: 没有非零元，跳过数据数组分配\n");
    }

    // 1. 在主机上分配临时数组收集列指针
    int totalCols = ctx->devCSC_A[0]->nCol;
    int *h_globalPtr = (int*)malloc(sizeof(int) * (totalCols + 1));
    h_globalPtr[0] = 0;

    // 2. 从各DCU收集局部列指针并计算全局偏移
    int currentOffset = 0;
    for (int dcuId = 0; dcuId < ctx->numDCUs; dcuId++) {
        CHECK_HIP_ERROR(hipSetDevice(dcuId));
        CSC_Matrix *localM = ctx->devCSC_M_local[dcuId];
        int localCols = ctx->colEnd[dcuId] - ctx->colStart[dcuId];

        if (localM->mPtr) {
            // 从设备复制局部列指针到主机
            int *h_localPtr = (int*)malloc(sizeof(int) * (localCols + 1));
            CHECK_HIP_ERROR(hipMemcpy(h_localPtr, localM->mPtr,
                                       (localCols + 1) * sizeof(int),
                                       hipMemcpyDeviceToHost));

            // 将局部指针合并到全局指针，加上偏移
            for (int i = 0; i <= localCols; i++) {
                h_globalPtr[ctx->colStart[dcuId] + i] = h_localPtr[i] + currentOffset;
            }

            currentOffset = h_globalPtr[ctx->colStart[dcuId] + localCols];
            free(h_localPtr);
        }
    }

    // 3. 复制全局列指针到设备
    CHECK_HIP_ERROR(hipSetDevice(0));
    CHECK_HIP_ERROR(hipMemcpy(devCSC_M_global->mPtr, h_globalPtr,
                               (totalCols + 1) * sizeof(int),
                               hipMemcpyHostToDevice));

    // 4. 从各DCU复制数据和索引
    for (int dcuId = 0; dcuId < ctx->numDCUs; dcuId++) {
        CSC_Matrix *localM = ctx->devCSC_M_local[dcuId];

        // 计算该DCU的起始偏移
        int startNnz = h_globalPtr[ctx->colStart[dcuId]];
        int numNnz = localM->nonzeroes;

        if (numNnz > 0 && localM->mIndex && localM->mData) {
            if (dcuId == 0) {
                // DCU 0: 直接在同一设备上复制
                CHECK_HIP_ERROR(hipSetDevice(0));
                CHECK_HIP_ERROR(hipMemcpy(devCSC_M_global->mIndex + startNnz,
                                           localM->mIndex,
                                           numNnz * sizeof(int),
                                           hipMemcpyDeviceToDevice));
                CHECK_HIP_ERROR(hipMemcpy(devCSC_M_global->mData + startNnz,
                                           localM->mData,
                                           numNnz * sizeof(double),
                                           hipMemcpyDeviceToDevice));
            } else {
                // 其他DCU: 通过主机中转复制
                CHECK_HIP_ERROR(hipSetDevice(dcuId));
                int *h_tempIndex = (int*)malloc(numNnz * sizeof(int));
                double *h_tempData = (double*)malloc(numNnz * sizeof(double));

                CHECK_HIP_ERROR(hipMemcpy(h_tempIndex, localM->mIndex,
                                           numNnz * sizeof(int),
                                           hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(hipMemcpy(h_tempData, localM->mData,
                                           numNnz * sizeof(double),
                                           hipMemcpyDeviceToHost));

                // 复制到全局矩阵
                CHECK_HIP_ERROR(hipSetDevice(0));
                CHECK_HIP_ERROR(hipMemcpy(devCSC_M_global->mIndex + startNnz,
                                           h_tempIndex,
                                           numNnz * sizeof(int),
                                           hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(devCSC_M_global->mData + startNnz,
                                           h_tempData,
                                           numNnz * sizeof(double),
                                           hipMemcpyHostToDevice));

                free(h_tempIndex);
                free(h_tempData);
            }

            printf("  DCU %d: 复制了 %d 个非零元 (偏移 %d)\n",
                   dcuId, numNnz, startNnz);
        }
    }

    // 验证聚合结果
    printf("\n验证全局矩阵:\n");
    printf("  总列数: %d\n", totalCols);
    printf("  总非零元: %d\n", totalNonzeros);
    printf("  列指针范围: [0, %d]\n", h_globalPtr[totalCols]);

    if (h_globalPtr[totalCols] != totalNonzeros) {
        printf("  警告: 列指针末尾值 (%d) 与总非零元 (%d) 不匹配\n",
               h_globalPtr[totalCols], totalNonzeros);
    }

    free(h_globalPtr);
    printf("结果聚合完成\n\n");
}

//==============================================================================
// 清理多DCU资源
//==============================================================================

void cleanupMultiDCU(MultiDCU_Context *ctx) {
    if (!ctx) return;

    printf("清理多DCU资源...\n");

    for (int i = 0; i < ctx->numDCUs; i++) {
        CHECK_HIP_ERROR(hipSetDevice(i));

        // 释放矩阵A
        if (ctx->devCSC_A && ctx->devCSC_A[i]) {
            if (ctx->devCSC_A[i]->mPtr) hipFree(ctx->devCSC_A[i]->mPtr);
            if (ctx->devCSC_A[i]->mIndex) hipFree(ctx->devCSC_A[i]->mIndex);
            if (ctx->devCSC_A[i]->mData) hipFree(ctx->devCSC_A[i]->mData);
            free(ctx->devCSC_A[i]);
        }

        // 释放局部矩阵M
        if (ctx->devCSC_M_local && ctx->devCSC_M_local[i]) {
            if (ctx->devCSC_M_local[i]->mPtr)
                hipFree(ctx->devCSC_M_local[i]->mPtr);
            if (ctx->devCSC_M_local[i]->mIndex)
                hipFree(ctx->devCSC_M_local[i]->mIndex);
            if (ctx->devCSC_M_local[i]->mData)
                hipFree(ctx->devCSC_M_local[i]->mData);
            free(ctx->devCSC_M_local[i]);
        }

        // 销毁流
        if (ctx->streams) {
            hipStreamDestroy(ctx->streams[i]);
        }

        printf("  DCU %d 资源已释放\n", i);
    }

    // 释放主机端数组
    if (ctx->deviceIds) free(ctx->deviceIds);
    if (ctx->devCSC_A) free(ctx->devCSC_A);
    if (ctx->devCSC_M_local) free(ctx->devCSC_M_local);
    if (ctx->colStart) free(ctx->colStart);
    if (ctx->colEnd) free(ctx->colEnd);
    if (ctx->streams) free(ctx->streams);

    free(ctx);
    printf("多DCU资源清理完成\n\n");
}

//==============================================================================
// 主函数
//==============================================================================

int main(int argc, char **argv) {
    char filename[256];

    cout << "========================================" << endl;
    cout << "多DCU SPAI预条件子求解器" << endl;
    cout << "========================================" << endl;
    cout << "输入矩阵文件名 (例如: circuit_2.mtx):" << endl;
    cin >> filename;

    // 读取矩阵
    CSC_Matrix *CSC_A = (CSC_Matrix*)malloc(sizeof(CSC_Matrix));
    if (!CSC_A) {
        printf("错误: 无法分配内存\n");
        return EXIT_FAILURE;
    }

    printf("\n正在读取矩阵文件: %s\n", filename);
    readMatrixToCSC(filename, CSC_A);

    // 验证矩阵
    if (!validateMatrix(CSC_A, "CSC_A")) {
        free(CSC_A);
        return EXIT_FAILURE;
    }

    printf("矩阵信息:\n");
    printf("  维度: %d x %d\n", CSC_A->n, CSC_A->n);
    printf("  非零元: %d\n", CSC_A->nonzeroes);
    printf("  稀疏度: %.2f%%\n\n",
           100.0 * CSC_A->nonzeroes / ((double)CSC_A->n * CSC_A->n));

    // 初始化多DCU环境
    int requestedDCUs = (argc > 1) ? atoi(argv[1]) : -1;  // 从命令行参数指定DCU数量
    MultiDCU_Context *ctx = initMultiDCU(requestedDCUs);

    // 复制矩阵A到所有DCU
    hipEvent_t start, stop;
    float elapsedTime;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    replicateMatrixA(ctx, CSC_A);

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsedTime, start, stop);
    printf("矩阵复制时间: %8.4f ms\n\n", elapsedTime);

    // 多DCU并行计算SPAI预条件子
    CSC_Matrix *devCSC_M_global = (CSC_Matrix*)malloc(sizeof(CSC_Matrix));
    float preconditioningTime = StaticSPAIv20_MultiDCU(ctx, devCSC_M_global);

    // 聚合结果
    aggregateResults(ctx, devCSC_M_global);

    // 打印性能统计
    printPerformanceStats(ctx, preconditioningTime);

    // TODO: 后续的求解过程（BiCGSTAB）
    // 注意：求解器可以在单个DCU上运行，或者也实现多DCU版本
    printf("提示: BiCGSTAB求解器尚未集成\n");
    printf("      当前仅完成预条件子计算阶段\n\n");

    // 清理资源
    cleanupMultiDCU(ctx);

    // 释放全局预条件子矩阵
    if (devCSC_M_global) {
        CHECK_HIP_ERROR(hipSetDevice(0));
        if (devCSC_M_global->mPtr) hipFree(devCSC_M_global->mPtr);
        if (devCSC_M_global->mIndex) hipFree(devCSC_M_global->mIndex);
        if (devCSC_M_global->mData) hipFree(devCSC_M_global->mData);
        free(devCSC_M_global);
    }

    // 释放主机端矩阵A
    if (CSC_A) {
        if (CSC_A->mPtr) free(CSC_A->mPtr);
        if (CSC_A->mIndex) free(CSC_A->mIndex);
        if (CSC_A->mData) free(CSC_A->mData);
        free(CSC_A);
    }

    hipEventDestroy(start);
    hipEventDestroy(stop);

    printf("========================================\n");
    printf("程序执行完成\n");
    printf("========================================\n");

    return EXIT_SUCCESS;
}
