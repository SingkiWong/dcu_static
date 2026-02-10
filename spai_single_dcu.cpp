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
#ifndef BICGSTAB_SUMMATION_BLOCK_SIZE
#define BICGSTAB_SUMMATION_BLOCK_SIZE 256
#endif
#ifndef BICGSTAB_THREADS_PER_BLOCK
#define BICGSTAB_THREADS_PER_BLOCK 256
#endif
#include "bicgstab/bicgstab_solver.h"

using namespace std;

#define CHECK_KERNEL_ERROR(stage) do { \
    hipError_t err = hipGetLastError(); \
    if (err != hipSuccess) { \
        printf("Kernel '%s' failed: %s\n", stage, hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_HIP_ERROR(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        printf("HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

template<unsigned int N2SIZE>
__global__ void cuComputeN2MAXwithSparityofA(int *aPtr, int nCol, int *n2max) {

    __shared__ int n2_s[N2SIZE];
    int gid = blockIdx.x * blockDim.x + threadIdx.x; //global index
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

    }//for col

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

float StaticSPAIv20(CSC_Matrix *devA, CSC_Matrix *devM) {

    /*+++++++++++++++++Pre-GSSPAI-Adaptive++++++++++++++++++++++++*/
    printf("Pre-GSSPAI is processing..........................\n");
    hipEvent_t start, stop;
    float elapsedTime;
    float preTime = 0.0;


    printf("-------------------Compute n2max\n");
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    const int threadsPerBlock = 256;
    int blocksPerGrid = 15 * 8 * 4;

    int *n2 = (int *) malloc(sizeof(int) * blocksPerGrid);
    int *dev_n2;
    hipMalloc((void **) &dev_n2, sizeof(int) * blocksPerGrid);

    cuComputeN2MAXwithSparityofA<threadsPerBlock><<< blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->n, dev_n2);

    hipMemcpyAsync(n2, dev_n2, blocksPerGrid * sizeof(int), hipMemcpyDeviceToHost, 0);
    hipDeviceSynchronize();
    int n2max = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        if (n2max < n2[i]) n2max = n2[i];
    }

    hipFree(dev_n2);
    free(n2);

    //int n2max = computeN2MAX(CSC_A);

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsedTime, start, stop);
    printf("Time = : %8.4f ms \n ", elapsedTime);
    preTime += elapsedTime;
    hipEventDestroy(start);
    hipEventDestroy(stop);

    printf("n2max=%d\n", n2max);

    printf("-------------------Compute n1max\n");
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    int WarpSize;

    if (n2max <= 2) {
        WarpSize = 2;
    } else if (n2max > 2 && n2max <= 4) {
        WarpSize = 4;
    } else if (n2max > 4 && n2max <= 8) {
        WarpSize = 8;
    } else if (n2max > 8 && n2max <= 16) {
        WarpSize = 16;
    } else {
        WarpSize = 32;
    }

    blocksPerGrid = (devA->n - 1) / (threadsPerBlock / WarpSize) + 1;


    int *dev_n1;
    hipMalloc((void **) &dev_n1, sizeof(int) * devA->nCol);
    int *dev_tI;
    int ssize = n2max * 8;
    ssize = min(ssize, devA->nCol);

    hipMalloc((void **) &dev_tI, sizeof(int) * devA->nCol * ssize);

    int *dev_atomic;
    hipMalloc((void **) &dev_atomic, sizeof(int) * devA->nCol);
    //printf("n2max =%d,  ssize = %d\n", n2max, ssize);
    //exit(0);

    if (n2max <= 2) {
        if (ssize <= 16) {
            printf("warp <= %d, number <= %d\n", 2, 16);
            cuComputeN1withSparityofA_SpMMv1<2, 2048><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->mPtr, devA->mIndex, dev_tI, ssize, devA->nCol, dev_n1, dev_atomic);
        } else if (ssize > 16 && ssize <= 32) {
            printf("warp <= %d,  number1 > %d, number2 <= %d\n", 2, 16, 32);
            cuComputeN1withSparityofA_SpMMv1<2, 4096><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->mPtr, devA->mIndex, dev_tI, ssize, devA->nCol, dev_n1, dev_atomic);
        } else if (ssize > 32 && ssize <= 64) {
            printf("warp <= %d,  number1 > %d, number2 <= %d\n", 2, 32, 64);
            cuComputeN1withSparityofA_SpMMv1<2, 8192><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->mPtr, devA->mIndex, dev_tI, ssize, devA->nCol, dev_n1, dev_atomic);
        } else {
            printf("warp <= %d no shared\n", 2);
            cuComputeN1withSparityofA2_SpMMv1<2><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex,
                                                                                     devA->mPtr, devA->mIndex, dev_tI,
                                                                                     ssize, devA->nCol, dev_n1,
                                                                                     dev_atomic);
        }
    } else if (n2max > 2 && n2max <= 4) {
        if (ssize <= 32) {
            printf("warp > %d, warp <= %d,  number <= %d\n", 2, 4, 32);
            cuComputeN1withSparityofA_SpMMv1<4, 2048><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->mPtr, devA->mIndex, dev_tI, ssize, devA->nCol, dev_n1, dev_atomic);
        } else if (ssize > 32 && ssize <= 64) {
            printf("warp > %d, warp <= %d,  number1 > %d, number2 <= %d\n", 2, 4, 32, 64);
            cuComputeN1withSparityofA_SpMMv1<4, 4096><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->mPtr, devA->mIndex, dev_tI, ssize, devA->nCol, dev_n1, dev_atomic);
        } else if (ssize > 64 && ssize <= 128) {
            printf("warp > %d, warp <= %d,  number1 > %d, number2 <= %d\n", 2, 4, 64, 128);
            cuComputeN1withSparityofA_SpMMv1<4, 8192><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->mPtr, devA->mIndex, dev_tI, ssize, devA->nCol, dev_n1, dev_atomic);
        } else {
            printf("warp > %d, warp <= %d no shared\n", 2, 4);
            cuComputeN1withSparityofA2_SpMMv1<4><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex,
                                                                                     devA->mPtr, devA->mIndex, dev_tI,
                                                                                     ssize, devA->nCol, dev_n1,
                                                                                     dev_atomic);
        }
    } else if (n2max > 4 && n2max <= 8) {
        if (ssize <= 64) {
            printf("+++++warp > %d, warp <= %d,  number <= %d\n", 4, 8, 64);
            cuComputeN1withSparityofA_SpMMv1<8, 2048><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->mPtr, devA->mIndex, dev_tI, ssize, devA->nCol, dev_n1, dev_atomic);
        } else if (ssize > 64 && ssize <= 128) {
            printf("warp > %d, warp <= %d,  number1 > %d, number2 <= %d\n", 4, 8, 64, 128);
            cuComputeN1withSparityofA_SpMMv1<8, 4096><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->mPtr, devA->mIndex, dev_tI, ssize, devA->nCol, dev_n1, dev_atomic);
        } else if (ssize > 128 && ssize <= 256) {
            printf("warp > %d, warp <= %d,  number1 > %d, number2 <= %d\n", 4, 8, 128, 256);
            cuComputeN1withSparityofA_SpMMv1<8, 8192><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->mPtr, devA->mIndex, dev_tI, ssize, devA->nCol, dev_n1, dev_atomic);
        } else {
            printf("warp > %d, warp <= %d no shared\n", 4, 8);
            cuComputeN1withSparityofA2_SpMMv1<8><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex,
                                                                                     devA->mPtr, devA->mIndex, dev_tI,
                                                                                     ssize, devA->nCol, dev_n1,
                                                                                     dev_atomic);
        }
    } else if (n2max > 8 && n2max <= 16) {
        if (ssize <= 128) {
            printf("+++++warp > %d, warp <= %d,  number <= %d\n", 8, 16, 128);
            cuComputeN1withSparityofA_SpMMv1<16, 2048><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->mPtr, devA->mIndex, dev_tI, ssize, devA->nCol, dev_n1, dev_atomic);
        } else if (ssize > 128 && ssize <= 256) {
            printf("warp > %d, warp <= %d,  number1 > %d, number2 <= %d\n", 8, 16, 128, 256);
            cuComputeN1withSparityofA_SpMMv1<16, 4096><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->mPtr, devA->mIndex, dev_tI, ssize, devA->nCol, dev_n1, dev_atomic);
        } else if (ssize > 256 && ssize <= 512) {
            printf("warp > %d, warp <= %d,  number1 > %d, number2 <= %d\n", 8, 16, 256, 512);
            cuComputeN1withSparityofA_SpMMv1<16, 8192><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->mPtr, devA->mIndex, dev_tI, ssize, devA->nCol, dev_n1, dev_atomic);
        } else {
            printf("warp > %d, warp <= %d no SHared\n", 8, 16);
            cuComputeN1withSparityofA2_SpMMv1<16><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex,
                                                                                      devA->mPtr, devA->mIndex, dev_tI,
                                                                                      ssize, devA->nCol, dev_n1,
                                                                                      dev_atomic);
        }
    } else {
        if (ssize <= 256) {
            printf("+++++warp >= %d, number <= %d\n", 32, 256);
            cuComputeN1withSparityofA_SpMMv1<32, 2048><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->mPtr, devA->mIndex, dev_tI, ssize, devA->nCol, dev_n1, dev_atomic);
        } else if (ssize > 256 && ssize <= 512) {
            printf("+++++warp >= %d, number1 > %d, number2 <= %d\n", 32, 256, 512);
            cuComputeN1withSparityofA_SpMMv1<32, 4096><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->mPtr, devA->mIndex, dev_tI, ssize, devA->nCol, dev_n1, dev_atomic);
        } else if (ssize > 512 && ssize <= 1024) {
            printf("+++++warp >= %d, number1 > %d, number2 <= %d\n", 32, 512, 1024);
            cuComputeN1withSparityofA_SpMMv1<32, 8192><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->mPtr, devA->mIndex, dev_tI, ssize, devA->nCol, dev_n1, dev_atomic);
        } else {
            printf("+++++warp >= %d no Shared\n", 32);
            cuComputeN1withSparityofA2_SpMMv1<32><<<blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex,
                                                                                      devA->mPtr, devA->mIndex, dev_tI,
                                                                                      ssize, devA->nCol, dev_n1,
                                                                                      dev_atomic);

        }
    }
    hipFree(dev_tI);

    blocksPerGrid = 15 * 8 * 4;
    int *n1 = (int *) malloc(sizeof(int) * blocksPerGrid);
    int *dev_n1max;
    hipMalloc((void **) &dev_n1max, sizeof(int) * blocksPerGrid);

    cuComputeN1MAX<threadsPerBlock><<< blocksPerGrid, threadsPerBlock>>>(dev_n1, devA->n, dev_n1max);

    hipMemcpyAsync(n1, dev_n1max, blocksPerGrid * sizeof(int), hipMemcpyDeviceToHost, 0);
    hipDeviceSynchronize();

    int n1max = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        if (n1max < n1[i]) n1max = n1[i];
    }

    hipFree(dev_n1);
    hipFree(dev_n1max);
    free(n1);

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsedTime, start, stop);

    printf("Time = : %8.4f ms \n ", elapsedTime);
    preTime += elapsedTime;

    hipEventDestroy(start);
    hipEventDestroy(stop);

    printf("n1max=%d\n", n1max);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    // 统一使用32，兼容NVIDIA和AMD CDNA架构
    WarpSize = 32;
    blocksPerGrid = (devA->nCol - 1) / (256 / WarpSize) + 1;

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsedTime, start, stop);

    printf("Time = : %8.4f ms \n ", elapsedTime);
    preTime += elapsedTime;

    hipEventDestroy(start);
    hipEventDestroy(stop);

    printf("blocksPerGrid = %d\n", blocksPerGrid);

    printf("\n");
    printf("preTime = : %12.4f ms \n ", preTime);

    /*+++++++++++++++++Compute-GSPAI-Adaptive++++++++++++++++++++++++*/
    printf("Compute-GSSPAI is processing......................\n");

    float computeTime = 0.0;
    float postTime = 0.0;
    //exit(0);
    /*********************************************************************************
    **                                                                               *
    ** Judge whether the memroy is exceeded?                                         *
    **                                                                               *
    **********************************************************************************/

    int compareSize = 1024 * 1024 * 100 * 4 * 3;
    long realSize = (long) devA->nCol * n1max * n2max;

    int itertions = (realSize - 1) / compareSize + 1;
    //int itertions = 2;

    printf("compareSize=%d, realSize=%ld, iterations =%d\n", compareSize, realSize, itertions);

    //exit(0);
    if (itertions != 1) {
        const int nCols = (devA->nCol - 1) / itertions + 1;

        printf("------Allocate GPU global memroy\n");

        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start, 0);

        double *dev_tildeA, *dev_R;

        int *dev_J, *dev_jPTR;
        int *dev_I, *dev_iPTR;
        int *dev_E;
        double *dev_X;

        hipMalloc((void **) &dev_J, sizeof(int) * nCols * n2max);
        hipMalloc((void **) &dev_jPTR, sizeof(int) * nCols);

        hipMalloc((void **) &dev_I, sizeof(int) * nCols * n1max);
        hipMalloc((void **) &dev_iPTR, sizeof(int) * nCols);

        hipMalloc((void **) &dev_tildeA, sizeof(double) * nCols * n1max * n2max);
        hipMalloc((void **) &dev_R, sizeof(double) * nCols * n2max * n2max);
        hipMalloc((void **) &dev_X, sizeof(double) * nCols * n2max);
        hipMalloc((void **) &dev_E, sizeof(int) * nCols);

        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsedTime, start, stop);

        printf("Time = : %8.4f ms \n ", elapsedTime);
        computeTime += elapsedTime;

        hipEventDestroy(start);
        hipEventDestroy(stop);

//	    int *dev_mTmpPtr, *dev_mTmpIndex;
        double *dev_mTmpData;
//
//	    hipMalloc((void**)&dev_mTmpPtr, sizeof(int) * (devA->nCol + 1)) ;
//      hipMalloc((void**)&dev_mTmpIndex, sizeof(int) * devA->nonzeroes ) ;
        hipMalloc((void **) &dev_mTmpData, sizeof(double) * devA->nonzeroes);

        for (int iter = 0; iter < itertions; iter++) {

            int sK = iter * nCols;
            int eGrid = nCols;
            if (iter == (itertions - 1)) eGrid = devA->nCol - (itertions - 1) * nCols;

            printf("iter = %d, eGrid=%d, sK=%d\n", iter, eGrid, sK);
            printf("warpSize =%d\n", WarpSize);

            blocksPerGrid = (eGrid - 1) / (threadsPerBlock / WarpSize) + 1;

            printf("---------------------find jIndex\n");

            CHECK_HIP_ERROR(hipMemset(dev_jPTR, 0, sizeof(int) * nCols));
            CHECK_HIP_ERROR(hipMemset(dev_iPTR, 0, sizeof(int) * nCols));
            CHECK_HIP_ERROR(hipMemset(dev_E, 0, sizeof(int) * nCols));

            hipEventCreate(&start);
            hipEventCreate(&stop);
            hipEventRecord(start, 0);

            if (n2max <= 2) {
                computeJ2_SSPAIv10<2><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
            } else if (n2max > 2 && n2max <= 4) {
                computeJ2_SSPAIv10<4><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
            } else if (n2max > 4 && n2max <= 8) {
                computeJ2_SSPAIv10<8><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
            } else if (n2max > 8 && n2max <= 16) {
                computeJ2_SSPAIv10<16><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
            } else if (n2max > 16 && n2max <= 32) {
                computeJ2_SSPAIv10<32><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
            } else if (n2max > 32 && n2max <= 64) {
                computeJ2_SSPAIv10<64><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
            } else if (n2max > 64 && n2max <= 128) {
                computeJ2_SSPAIv10<128><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
            } else {
                computeJ2_SSPAIv10<256><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_J, dev_jPTR, n2max, sK);
            }
            CHECK_KERNEL_ERROR("computeJ2_SSPAIv10");

            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&elapsedTime, start, stop);

            printf("Time = : %8.4f ms \n ", elapsedTime);
            computeTime += elapsedTime;

            hipEventDestroy(start);
            hipEventDestroy(stop);

            printf("---------------------find iIndex\n");

            hipEventCreate(&start);
            hipEventCreate(&stop);
            hipEventRecord(start, 0);


            if (n2max <= 2) {
                if (n1max <= 8) {
                    computeIShared_iter_Symbol_SpMMv1<2, 1024><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 8 && n1max <= 16) {
                    computeIShared_iter_Symbol_SpMMv1<2, 2048><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 16 && n1max <= 32) {
                    computeIShared_iter_Symbol_SpMMv1<2, 4096><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 32 && n1max <= 64) {
                    computeIShared_iter_Symbol_SpMMv1<2, 8192><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else {
                    computeI_iter_Symbol_SpMMv1<2><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                }
            } else if (n2max > 2 && n2max <= 4) {
                if (n1max <= 8) {
                    computeIShared_iter_Symbol_SpMMv1<4, 512><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 8 && n1max <= 16) {
                    computeIShared_iter_Symbol_SpMMv1<4, 1024><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 16 && n1max <= 32) {
                    computeIShared_iter_Symbol_SpMMv1<4, 2048><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 32 && n1max <= 64) {
                    computeIShared_iter_Symbol_SpMMv1<4, 4096><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 64 && n1max <= 128) {
                    computeIShared_iter_Symbol_SpMMv1<4, 8192><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else {
                    computeI_iter_Symbol_SpMMv1<4><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                }
            } else if (n2max > 4 && n2max <= 8) {
                if (n1max <= 8) {
                    computeIShared_iter_Symbol_SpMMv1<8, 256><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 8 && n1max <= 16) {
                    computeIShared_iter_Symbol_SpMMv1<8, 512><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 16 && n1max <= 32) {
                    computeIShared_iter_Symbol_SpMMv1<8, 1024><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 32 && n1max <= 64) {
                    computeIShared_iter_Symbol_SpMMv1<8, 2048><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 64 && n1max <= 128) {
                    computeIShared_iter_Symbol_SpMMv1<8, 4096><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 128 && n1max <= 256) {
                    computeIShared_iter_Symbol_SpMMv1<8, 8192><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else {
                    computeI_iter_Symbol_SpMMv1<8><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                }
            } else if (n2max > 8 && n2max <= 16) {
                if (n1max <= 16) {
                    computeIShared_iter_Symbol_SpMMv1<16, 256><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 16 && n1max <= 32) {
                    computeIShared_iter_Symbol_SpMMv1<16, 512><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 32 && n1max <= 64) {
                    computeIShared_iter_Symbol_SpMMv1<16, 1024><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 64 && n1max <= 128) {
                    computeIShared_iter_Symbol_SpMMv1<16, 2048><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 128 && n1max <= 256) {
                    computeIShared_iter_Symbol_SpMMv1<16, 4096><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 256 && n1max <= 512) {
                    computeIShared_iter_Symbol_SpMMv1<16, 8192><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else {
                    computeI_iter_Symbol_SpMMv1<16><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                }
            } else if (n2max > 16 && n2max <= 32) {
                if (n1max <= 32) {
                    computeIShared_iter_Symbol_SpMMv1<32, 256><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 32 && n1max <= 64) {
                    computeIShared_iter_Symbol_SpMMv1<32, 512><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 64 && n1max <= 128) {
                    computeIShared_iter_Symbol_SpMMv1<32, 1024><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 128 && n1max <= 256) {
                    computeIShared_iter_Symbol_SpMMv1<32, 2048><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 256 && n1max <= 512) {
                    computeIShared_iter_Symbol_SpMMv1<32, 4096><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 512 && n1max <= 1024) {
                    computeIShared_iter_Symbol_SpMMv1<32, 8192><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else {
                    computeI_iter_Symbol_SpMMv1<32><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                }
            } else if (n2max > 32 && n2max <= 64) {
                if (n1max <= 64) {
                    computeIShared_iter_Symbol_SpMMv1<64, 256><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 64 && n1max <= 128) {
                    computeIShared_iter_Symbol_SpMMv1<64, 512><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 128 && n1max <= 256) {
                    computeIShared_iter_Symbol_SpMMv1<64, 1024><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 256 && n1max <= 512) {
                    computeIShared_iter_Symbol_SpMMv1<64, 2048><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 512 && n1max <= 1024) {
                    computeIShared_iter_Symbol_SpMMv1<64, 4096><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 1024 && n1max <= 2048) {
                    computeIShared_iter_Symbol_SpMMv1<64, 8192><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else {
                    computeI_iter_Symbol_SpMMv1<64><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                }
            } else if (n2max > 64 && n2max <= 128) {
                if (n1max <= 128) {
                    computeIShared_iter_Symbol_SpMMv1<128, 256><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 128 && n1max <= 256) {
                    computeIShared_iter_Symbol_SpMMv1<128, 512><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 256 && n1max <= 512) {
                    computeIShared_iter_Symbol_SpMMv1<128, 1024><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 512 && n1max <= 1024) {
                    computeIShared_iter_Symbol_SpMMv1<128, 2048><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 1024 && n1max <= 2048) {
                    computeIShared_iter_Symbol_SpMMv1<128, 4096><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 2048 && n1max <= 4096) {
                    computeIShared_iter_Symbol_SpMMv1<128, 8192><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else {
                    computeI_iter_Symbol_SpMMv1<128><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                }
            } else if (n2max > 128 && n2max <= 256) {
                if (n1max <= 256) {
                    computeIShared_iter_Symbol_SpMMv1<256, 256><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 256 && n1max <= 512) {
                    computeIShared_iter_Symbol_SpMMv1<256, 512><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 512 && n1max <= 1024) {
                    computeIShared_iter_Symbol_SpMMv1<256, 1024><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 1024 && n1max <= 2048) {
                    computeIShared_iter_Symbol_SpMMv1<256, 2048><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 2048 && n1max <= 4096) {
                    computeIShared_iter_Symbol_SpMMv1<256, 4096><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 4096 && n1max <= 8192) {
                    computeIShared_iter_Symbol_SpMMv1<256, 8192><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else {
                    computeI_iter_Symbol_SpMMv1<256><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                }
            } else {
                if (n1max <= 512) {
                    computeIShared_iter_Symbol_SpMMv1<256, 512><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 512 && n1max <= 1024) {
                    computeIShared_iter_Symbol_SpMMv1<256, 1024><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 1024 && n1max <= 2048) {
                    computeIShared_iter_Symbol_SpMMv1<256, 2048><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 2048 && n1max <= 4096) {
                    computeIShared_iter_Symbol_SpMMv1<256, 4096><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else if (n1max > 4096 && n1max <= 8192) {
                    computeIShared_iter_Symbol_SpMMv1<256, 8192><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                } else {
                    computeI_iter_Symbol_SpMMv1<256><<<
                    blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, eGrid, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
                }
            }

            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&elapsedTime, start, stop);

            printf("Time = : %8.4f ms \n ", elapsedTime);
            computeTime += elapsedTime;

            hipEventDestroy(start);
            hipEventDestroy(stop);


            printf("-----------------find tilde of A\n");

            hipEventCreate(&start);
            hipEventCreate(&stop);
            hipEventRecord(start, 0);

            if (n2max <= 2) {
                ComputeTildeACSR_SSPAIv10<2><<<
                blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr,
                        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
            } else if (n2max > 2 && n2max <= 4) {
                ComputeTildeACSR_SSPAIv10<4><<<
                blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr,
                        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
            } else if (n2max > 4 && n2max <= 8) {
                ComputeTildeACSR_SSPAIv10<8><<<
                blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr,
                        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
            } else if (n2max > 8 && n2max <= 16) {
                ComputeTildeACSR_SSPAIv10<16><<<
                blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr,
                        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
            } else if (n2max > 16 && n2max <= 32) {
                ComputeTildeACSR_SSPAIv10<32><<<
                blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr,
                        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
            } else if (n2max > 32 && n2max <= 64) {
                ComputeTildeACSR_SSPAIv10<64><<<
                blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr,
                        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
            } else if (n2max > 64 && n2max <= 128) {
                ComputeTildeACSR_SSPAIv10<128><<<
                blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr,
                        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
            } else {
                ComputeTildeACSR_SSPAIv10<256><<<
                blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr,
                        devA->mIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
            }

//		    int blocksPerGrid2 = (eGrid - 1)/(threadsPerBlock/32) + 1;
//		    ComputeTildeACSR_SSPAIv10<32><< <blocksPerGrid2, threadsPerBlock>> >(dev_tildeA, dev_aData, dev_aPtr,
//		        dev_aIndex, eGrid, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);

            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&elapsedTime, start, stop);

            printf("Time = : %8.4f ms \n ", elapsedTime);
            computeTime += elapsedTime;

            hipEventDestroy(start);
            hipEventDestroy(stop);

            printf("++++find QR\n");

            hipEventCreate(&start);
            hipEventCreate(&stop);
            hipEventRecord(start, 0);

            if (n2max <= 2) {
                QR_RShared_SSPAIv10<2, 256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                        dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
            } else if (n2max > 2 && n2max <= 4) {
                QR_RShared_SSPAIv10<4, 256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                        dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
            } else if (n2max > 4 && n2max <= 8) {
                QR_RShared_SSPAIv10<8, 256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                        dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
            } else if (n2max > 8 && n2max <= 16) {
                QR_RShared_SSPAIv10<16, 256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                        dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
            } else if (n2max > 16 && n2max <= 32) {
                QR_RShared_SSPAIv10<32, 256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                        dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
            } else if (n2max > 32 && n2max <= 64) {
                QR_RShared_SSPAIv10<64, 256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                        dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
            } else if (n2max > 64 && n2max <= 128) {
                QR_RShared_SSPAIv10<128, 256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                        dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
            } else {
                if (n2max > 128 && n2max <= 256) {
                    QR_RShared_SSPAIv10<256, 256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                            dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
                } else if (n2max > 256 && n2max <= 512) {
                    QR_RShared_SSPAIv10<256, 512><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                            dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
                } else if (n2max > 512 && n2max <= 1024) {
                    QR_RShared_SSPAIv10<256, 1024><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                            dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
                } else if (n2max > 1024 && n2max <= 2048) {
                    QR_RShared_SSPAIv10<256, 2048><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                            dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);
                } else {
                    printf("Sorry, exceed the maximum shared memory in the QR decomposition\n");
                    exit(0);
                }
            }

//        blocksPerGrid = (eGrid - 1)/(threadsPerBlock/32) + 1;
//        QR_RShared_SSPAIv10<32, 2048><< <blocksPerGrid, threadsPerBlock>> >(dev_tildeA,
//		         dev_R, dev_iPTR, dev_jPTR, n1max, n2max, eGrid);

            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&elapsedTime, start, stop);

            printf("Time = : %8.4f ms \n ", elapsedTime);
            computeTime += elapsedTime;

            hipEventDestroy(start);
            hipEventDestroy(stop);

            printf("--------------Compute tilde of E\n");

            hipEventCreate(&start);
            hipEventCreate(&stop);
            hipEventRecord(start, 0);

            if (n2max <= 2) {
                ComputeTildeE2_SSPAIv10<2><<<
                blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
            } else if (n2max > 2 && n2max <= 4) {
                ComputeTildeE2_SSPAIv10<4><<<
                blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
            } else if (n2max > 4 && n2max <= 8) {
                ComputeTildeE2_SSPAIv10<8><<<
                blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
            } else if (n2max > 8 && n2max <= 16) {
                ComputeTildeE2_SSPAIv10<16><<<
                blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
            } else if (n2max > 16 && n2max <= 32) {
                ComputeTildeE2_SSPAIv10<32><<<
                blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
            } else if (n2max > 32 && n2max <= 64) {
                ComputeTildeE2_SSPAIv10<64><<<
                blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
            } else if (n2max > 64 && n2max <= 128) {
                ComputeTildeE2_SSPAIv10<128><<<
                blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
            } else {
                ComputeTildeE2_SSPAIv10<256><<<
                blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
            }

//        blocksPerGrid = (eGrid - 1)/(threadsPerBlock/32) + 1;
//			  ComputeTildeE2_SSPAIv10<32><< <blocksPerGrid, threadsPerBlock>> >(dev_E, dev_I, dev_iPTR, n1max, eGrid, sK);
            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&elapsedTime, start, stop);

            printf("Time = : %8.4f ms \n ", elapsedTime);
            computeTime += elapsedTime;

            hipEventDestroy(start);
            hipEventDestroy(stop);

            printf("-------------------------Solve X\n");

            hipEventCreate(&start);
            hipEventCreate(&stop);
            hipEventRecord(start, 0);

            if (n2max <= 2) {
                Sol_SSPAIv10<2><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X,
                        dev_E, dev_jPTR, n1max, n2max, eGrid);
            } else if (n2max > 2 && n2max <= 4) {
                Sol_SSPAIv10<4><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X,
                        dev_E, dev_jPTR, n1max, n2max, eGrid);
            } else if (n2max > 4 && n2max <= 8) {
                Sol_SSPAIv10<8><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X,
                        dev_E, dev_jPTR, n1max, n2max, eGrid);
            } else if (n2max > 8 && n2max <= 16) {
                Sol_SSPAIv10<16><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X,
                        dev_E, dev_jPTR, n1max, n2max, eGrid);
            } else if (n2max > 16 && n2max <= 32) {
                Sol_SSPAIv10<32><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X,
                        dev_E, dev_jPTR, n1max, n2max, eGrid);
            } else if (n2max > 32 && n2max <= 64) {
                Sol_SSPAIv10<64><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X,
                        dev_E, dev_jPTR, n1max, n2max, eGrid);
            } else if (n2max > 64 && n2max <= 128) {
                Sol_SSPAIv10<128><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X,
                        dev_E, dev_jPTR, n1max, n2max, eGrid);
            } else {
                Sol_SSPAIv10<256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X,
                        dev_E, dev_jPTR, n1max, n2max, eGrid);
            }

            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&elapsedTime, start, stop);

            printf("Time = : %8.4f ms \n ", elapsedTime);
            computeTime += elapsedTime;

            hipEventDestroy(start);
            hipEventDestroy(stop);

            printf("\n");
            printf("computeTime = : %12.4f ms \n", computeTime);

            /*---------Store arrays templately------------------*/

            //hipMemcpyAsync( dev_mTmpPtr + sK, dev_jPTR, eGrid * sizeof( int ), hipMemcpyDeviceToDevice, 0 ) ;

            printf("-------------------------Store arrays temply For postTime\n");

            hipEventCreate(&start);
            hipEventCreate(&stop);
            hipEventRecord(start, 0);

            int WarpSize1;
            if (n2max <= 2) {
                WarpSize1 = 2;
            } else if (n2max > 2 && n2max <= 4) {
                WarpSize1 = 4;
            } else if (n2max > 4 && n2max <= 8) {
                WarpSize1 = 8;
            } else if (n2max > 8 && n2max <= 16) {
                WarpSize1 = 16;
            } else {
                WarpSize1 = 32;
            }

            blocksPerGrid = (eGrid - 1) / (threadsPerBlock / WarpSize1) + 1;

            if (n2max <= 2) {
                modifyData24_SSPAIv10<2><<< blocksPerGrid, threadsPerBlock>>>(dev_mTmpData,
                        devA->mPtr, dev_X, dev_jPTR, n2max, eGrid, sK);
            } else if (n2max > 2 && n2max <= 4) {
                modifyData24_SSPAIv10<4><<< blocksPerGrid, threadsPerBlock>>>(dev_mTmpData,
                        devA->mPtr, dev_X, dev_jPTR, n2max, eGrid, sK);
            } else if (n2max > 4 && n2max <= 8) {
                modifyData24_SSPAIv10<8><<< blocksPerGrid, threadsPerBlock>>>(dev_mTmpData,
                        devA->mPtr, dev_X, dev_jPTR, n2max, eGrid, sK);
            } else if (n2max > 8 && n2max <= 16) {
                modifyData24_SSPAIv10<16><<< blocksPerGrid, threadsPerBlock>>>(dev_mTmpData,
                        devA->mPtr, dev_X, dev_jPTR, n2max, eGrid, sK);
            } else {
                modifyData24_SSPAIv10<32><<< blocksPerGrid, threadsPerBlock>>>(dev_mTmpData,
                        devA->mPtr, dev_X, dev_jPTR, n2max, eGrid, sK);
            }

            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&elapsedTime, start, stop);

            printf("Time = : %8.4f ms \n ", elapsedTime);
            postTime += elapsedTime;

            hipEventDestroy(start);
            hipEventDestroy(stop);

        }
        printf("Post-GSSPAI is processing.........................\n");
        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start, 0);

        hipFree(dev_J);
        hipFree(dev_I);
        hipFree(dev_jPTR);
        hipFree(dev_iPTR);

        hipFree(dev_tildeA);
        hipFree(dev_R);
        hipFree(dev_E);
        hipFree(dev_X);



        devM->n = devA->n;
        devM->nCol = devA->nCol;
        devM->nRow = devA->nRow;
        devM->nonzeroes = devA->nonzeroes;

        hipMalloc((void **) &devM->mIndex, sizeof(int) * devM->nonzeroes);
        hipMalloc((void **) &devM->mData, sizeof(double) * devM->nonzeroes);
        hipMalloc((void **) &devM->mPtr, sizeof(int) * (devM->nCol + 1));

        hipMemcpy(devM->mIndex, devA->mIndex, sizeof(int) * devM->nonzeroes, hipMemcpyDeviceToDevice);
        hipMemcpy(devM->mPtr, devA->mPtr, sizeof(int) * (devM->nCol + 1), hipMemcpyDeviceToDevice);
        hipMemcpy(devM->mData, dev_mTmpData, sizeof(double) * devM->nonzeroes, hipMemcpyDeviceToDevice);

        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsedTime, start, stop);

        printf("Time = : %8.4f ms \n ", elapsedTime);
        postTime += elapsedTime;

        hipEventDestroy(start);
        hipEventDestroy(stop);

        printf("postTime = : %12.4f ms \n ", postTime);

//      exit(0);
        printf("*************************************************\n");
        printf("TotalPreTime = : %12.4f ms \n", preTime);
        printf("TotalComputeTime = : %12.4f ms \n", computeTime);
        printf("TotalPostTime = : %12.4f ms \n ", postTime);

        printf("TotalTime = : %12.4f ms \n ", preTime + computeTime + postTime);
        printf("\n");

        
        hipFree(dev_mTmpData);



    } else {
        printf("------Allocate GPU global memroy\n");

        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start, 0);

        double *dev_tildeA, *dev_R;

        int *dev_J, *dev_jPTR;
        int *dev_I, *dev_iPTR;
        int *dev_E;
        double *dev_X;

        hipMalloc((void **) &dev_J, sizeof(int) * devA->nCol * n2max);
        hipMalloc((void **) &dev_jPTR, sizeof(int) * devA->nCol);

        hipMalloc((void **) &dev_I, sizeof(int) * devA->nCol * n1max);
        hipMalloc((void **) &dev_iPTR, sizeof(int) * devA->nCol);

        hipMalloc((void **) &dev_tildeA, sizeof(double) * devA->nCol * n1max * n2max);
        hipMalloc((void **) &dev_R, sizeof(double) * devA->nCol * n2max * n2max);
        hipMalloc((void **) &dev_X, sizeof(double) * devA->nCol * n2max);
        hipMalloc((void **) &dev_E, sizeof(int) * devA->nCol);

        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsedTime, start, stop);

        printf("Time = : %8.4f ms \n ", elapsedTime);
        computeTime += elapsedTime;

        hipEventDestroy(start);
        hipEventDestroy(stop);

        CHECK_HIP_ERROR(hipMemset(dev_jPTR, 0, sizeof(int) * devA->nCol));
        CHECK_HIP_ERROR(hipMemset(dev_iPTR, 0, sizeof(int) * devA->nCol));
        CHECK_HIP_ERROR(hipMemset(dev_E, 0, sizeof(int) * devA->nCol));

        printf("---------------------find jIndex\n");

        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start, 0);
        if (n2max <= 2) {
            computeJ_SSPAIv10<2><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->nCol, dev_J, dev_jPTR, n2max);
        } else if (n2max > 2 && n2max <= 4) {
            computeJ_SSPAIv10<4><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->nCol, dev_J, dev_jPTR, n2max);
        } else if (n2max > 4 && n2max <= 8) {
            computeJ_SSPAIv10<8><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->nCol, dev_J, dev_jPTR, n2max);
        } else if (n2max > 8 && n2max <= 16) {
            computeJ_SSPAIv10<16><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->nCol, dev_J, dev_jPTR, n2max);
        } else if (n2max > 16 && n2max <= 32) {
            computeJ_SSPAIv10<32><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->nCol, dev_J, dev_jPTR, n2max);
        } else if (n2max > 32 && n2max <= 64) {
            computeJ_SSPAIv10<64><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->nCol, dev_J, dev_jPTR, n2max);
        } else if (n2max > 64 && n2max <= 128) {
            computeJ_SSPAIv10<128><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->nCol, dev_J, dev_jPTR, n2max);
        } else {
            computeJ_SSPAIv10<256><<<
            blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devA->nCol, dev_J, dev_jPTR, n2max);
        }

        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsedTime, start, stop);

        printf("Time = : %8.4f ms \n ", elapsedTime);
        computeTime += elapsedTime;

        hipEventDestroy(start);
        hipEventDestroy(stop);

        printf("---------------------find iIndex\n");

        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start, 0);


        devM->nCol = devA->nCol;
        if (n2max <= 2) {
            if (n1max <= 8) {
                computeIShared_Symbol_SpMMv1<2, 1024><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 8 && n1max <= 16) {
                computeIShared_Symbol_SpMMv1<2, 2048><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 16 && n1max <= 32) {
                computeIShared_Symbol_SpMMv1<2, 4096><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 32 && n1max <= 64) {
                computeIShared_Symbol_SpMMv1<2, 8192><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else {
                computeI_Symbol_SpMMv1<2><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            }
        } else if (n2max > 2 && n2max <= 4) {
            if (n1max <= 8) {
                computeIShared_Symbol_SpMMv1<4, 512><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 8 && n1max <= 16) {
                computeIShared_Symbol_SpMMv1<4, 1024><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 16 && n1max <= 32) {
                computeIShared_Symbol_SpMMv1<4, 2048><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 32 && n1max <= 64) {
                computeIShared_Symbol_SpMMv1<4, 4096><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 64 && n1max <= 128) {
                computeIShared_Symbol_SpMMv1<4, 8192><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else {
                computeI_Symbol_SpMMv1<4><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            }
        } else if (n2max > 4 && n2max <= 8) {
            if (n1max <= 8) {
                computeIShared_Symbol_SpMMv1<8, 256><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 8 && n1max <= 16) {
                computeIShared_Symbol_SpMMv1<8, 512><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 16 && n1max <= 32) {
                printf("n2max = %d, n1max = %d\n", n2max, n1max);
                computeIShared_Symbol_SpMMv1<8, 1024><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 32 && n1max <= 64) {
                computeIShared_Symbol_SpMMv1<8, 2048><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 64 && n1max <= 128) {
                computeIShared_Symbol_SpMMv1<8, 4096><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 128 && n1max <= 256) {
                computeIShared_Symbol_SpMMv1<8, 8192><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else {
                computeI_Symbol_SpMMv1<8><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            }
        } else if (n2max > 8 && n2max <= 16) {
            if (n1max <= 16) {
                computeIShared_Symbol_SpMMv1<16, 256><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 16 && n1max <= 32) {
                computeIShared_Symbol_SpMMv1<16, 512><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 32 && n1max <= 64) {
                computeIShared_Symbol_SpMMv1<16, 1024><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 64 && n1max <= 128) {
                computeIShared_Symbol_SpMMv1<16, 2048><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 128 && n1max <= 256) {
                computeIShared_Symbol_SpMMv1<16, 4096><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 256 && n1max <= 512) {
                computeIShared_Symbol_SpMMv1<16, 8192><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else {
                computeI_Symbol_SpMMv1<16><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            }
        } else if (n2max > 16 && n2max <= 32) {
            if (n1max <= 32) {
                computeIShared_Symbol_SpMMv1<32, 256><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 32 && n1max <= 64) {
                computeIShared_Symbol_SpMMv1<32, 512><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 64 && n1max <= 128) {
                computeIShared_Symbol_SpMMv1<32, 1024><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 128 && n1max <= 256) {
                computeIShared_Symbol_SpMMv1<32, 2048><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 256 && n1max <= 512) {
                computeIShared_Symbol_SpMMv1<32, 4096><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 512 && n1max <= 1024) {
                computeIShared_Symbol_SpMMv1<32, 8192><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else {
                computeI_Symbol_SpMMv1<32><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            }
        } else if (n2max > 32 && n2max <= 64) {
            if (n1max <= 64) {
                computeIShared_Symbol_SpMMv1<64, 256><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 64 && n1max <= 128) {
                computeIShared_Symbol_SpMMv1<64, 512><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 128 && n1max <= 256) {
                computeIShared_Symbol_SpMMv1<64, 1024><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 256 && n1max <= 512) {
                computeIShared_Symbol_SpMMv1<64, 2048><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 512 && n1max <= 1024) {
                computeIShared_Symbol_SpMMv1<64, 4096><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 1024 && n1max <= 2048) {
                computeIShared_Symbol_SpMMv1<64, 8192><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else {
                computeI_Symbol_SpMMv1<64><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            }
        } else if (n2max > 64 && n2max <= 128) {
            if (n1max <= 128) {
                computeIShared_Symbol_SpMMv1<128, 256><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 128 && n1max <= 256) {
                computeIShared_Symbol_SpMMv1<128, 512><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 256 && n1max <= 512) {
                computeIShared_Symbol_SpMMv1<128, 1024><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 512 && n1max <= 1024) {
                computeIShared_Symbol_SpMMv1<128, 2048><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 1024 && n1max <= 2048) {
                computeIShared_Symbol_SpMMv1<128, 4096><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 2048 && n1max <= 4096) {
                computeIShared_Symbol_SpMMv1<128, 8192><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else {
                computeI_Symbol_SpMMv1<32><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            }
        } else if (n2max > 128 && n2max <= 256) {
            if (n1max <= 256) {
                computeIShared_Symbol_SpMMv1<256, 256><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 256 && n1max <= 512) {
                computeIShared_Symbol_SpMMv1<256, 512><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 512 && n1max <= 1024) {
                computeIShared_Symbol_SpMMv1<256, 1024><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 1024 && n1max <= 2048) {
                computeIShared_Symbol_SpMMv1<256, 2048><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 2048 && n1max <= 4096) {
                computeIShared_Symbol_SpMMv1<256, 4096><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 4096 && n1max <= 8192) {
                computeIShared_Symbol_SpMMv1<256, 8192><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else {
                computeI_Symbol_SpMMv1<256><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            }
        } else {
            if (n1max <= 512) {
                computeIShared_Symbol_SpMMv1<256, 512><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 512 && n1max <= 1024) {
                computeIShared_Symbol_SpMMv1<256, 1024><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 1024 && n1max <= 2048) {
                computeIShared_Symbol_SpMMv1<256, 2048><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 2048 && n1max <= 4096) {
                computeIShared_Symbol_SpMMv1<256, 4096><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else if (n1max > 4096 && n1max <= 8192) {
                computeIShared_Symbol_SpMMv1<256, 8192><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            } else {
                computeI_Symbol_SpMMv1<256><<<
                blocksPerGrid, threadsPerBlock>>>(devA->mPtr, devA->mIndex, devM->nCol, dev_I, dev_iPTR, n1max, dev_J, dev_jPTR, n2max, dev_atomic);
            }
        }

        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsedTime, start, stop);

        printf("Time = : %8.4f ms \n ", elapsedTime);
        computeTime += elapsedTime;

        hipEventDestroy(start);
        hipEventDestroy(stop);

        //exit(0);

        printf("-----------------find tilde of A\n");

        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start, 0);
        if (n2max <= 2) {
            ComputeTildeACSR_SSPAIv10<2><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr,
                    devA->mIndex, devA->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
        } else if (n2max > 2 && n2max <= 4) {
            ComputeTildeACSR_SSPAIv10<4><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr,
                    devA->mIndex, devA->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
        } else if (n2max > 4 && n2max <= 8) {
            ComputeTildeACSR_SSPAIv10<8><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr,
                    devA->mIndex, devA->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
        } else if (n2max > 8 && n2max <= 16) {
            ComputeTildeACSR_SSPAIv10<16><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr,
                    devA->mIndex, devA->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
        } else if (n2max > 16 && n2max <= 32) {
            ComputeTildeACSR_SSPAIv10<32><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr,
                    devA->mIndex, devA->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
        } else if (n2max > 32 && n2max <= 64) {
            ComputeTildeACSR_SSPAIv10<64><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr,
                    devA->mIndex, devA->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
        } else if (n2max > 64 && n2max <= 128) {
            ComputeTildeACSR_SSPAIv10<128><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr,
                    devA->mIndex, devA->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
        } else {
            ComputeTildeACSR_SSPAIv10<256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, devA->mData, devA->mPtr,
                    devA->mIndex, devA->nCol, dev_I, dev_iPTR, dev_J, dev_jPTR, n1max, n2max);
        }

        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsedTime, start, stop);

        printf("Time = : %8.4f ms \n ", elapsedTime);
        computeTime += elapsedTime;

        hipEventDestroy(start);
        hipEventDestroy(stop);

        printf("-----------------------find QR\n");

        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start, 0);

        if (n2max <= 2) {
            QR_RShared_SSPAIv10<2, 256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                    dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devA->nCol);
        } else if (n2max > 2 && n2max <= 4) {
            QR_RShared_SSPAIv10<4, 256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                    dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devA->nCol);
        } else if (n2max > 4 && n2max <= 8) {
            QR_RShared_SSPAIv10<8, 256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                    dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devA->nCol);
        } else if (n2max > 8 && n2max <= 16) {
            QR_RShared_SSPAIv10<16, 256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                    dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devA->nCol);
        } else if (n2max > 16 && n2max <= 32) {
            QR_RShared_SSPAIv10<32, 256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                    dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devA->nCol);
        } else if (n2max > 32 && n2max <= 64) {
            QR_RShared_SSPAIv10<64, 256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                    dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devA->nCol);
        } else if (n2max > 64 && n2max <= 128) {
            QR_RShared_SSPAIv10<128, 256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                    dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devA->nCol);
        } else {
            if (n2max > 128 && n2max <= 256) {
                QR_RShared_SSPAIv10<256, 256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                        dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devA->nCol);
            } else if (n2max > 256 && n2max <= 512) {
                QR_RShared_SSPAIv10<256, 512><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                        dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devA->nCol);
            } else if (n2max > 512 && n2max <= 1024) {
                QR_RShared_SSPAIv10<256, 1024><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                        dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devA->nCol);
            } else if (n2max > 1024 && n2max <= 2048) {
                QR_RShared_SSPAIv10<256, 2048><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA,
                        dev_R, dev_iPTR, dev_jPTR, n1max, n2max, devA->nCol);
            } else {
                printf("Sorry, exceed the maximum shared memory in the QR decomposition\n");
                exit(0);
            }
        }
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsedTime, start, stop);

        printf("Time = : %8.4f ms \n ", elapsedTime);
        computeTime += elapsedTime;

        hipEventDestroy(start);
        hipEventDestroy(stop);

        printf("--------------Compute tilde of E\n");

        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start, 0);

        if (n2max <= 2) {
            ComputeTildeE_SSPAIv10<2><<<
            blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, devA->nCol);
        } else if (n2max > 2 && n2max <= 4) {
            ComputeTildeE_SSPAIv10<4><<<
            blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, devA->nCol);
        } else if (n2max > 4 && n2max <= 8) {
            ComputeTildeE_SSPAIv10<8><<<
            blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, devA->nCol);
        } else if (n2max > 8 && n2max <= 16) {
            ComputeTildeE_SSPAIv10<16><<<
            blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, devA->nCol);
        } else if (n2max > 16 && n2max <= 32) {
            ComputeTildeE_SSPAIv10<32><<<
            blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, devA->nCol);
        } else if (n2max > 32 && n2max <= 64) {
            ComputeTildeE_SSPAIv10<64><<<
            blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, devA->nCol);
        } else if (n2max > 64 && n2max <= 128) {
            ComputeTildeE_SSPAIv10<128><<<
            blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, devA->nCol);
        } else {
            ComputeTildeE_SSPAIv10<256><<<
            blocksPerGrid, threadsPerBlock>>>(dev_E, dev_I, dev_iPTR, n1max, devA->nCol);
        }

        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsedTime, start, stop);

        printf("Time = : %8.4f ms \n ", elapsedTime);
        computeTime += elapsedTime;

        hipEventDestroy(start);
        hipEventDestroy(stop);

        printf("-------------------------Solve X\n");

        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start, 0);

        if (n2max <= 2) {
            Sol_SSPAIv10<2><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X,
                    dev_E, dev_jPTR, n1max, n2max, devA->nCol);
        } else if (n2max > 2 && n2max <= 4) {
            Sol_SSPAIv10<4><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X,
                    dev_E, dev_jPTR, n1max, n2max, devA->nCol);
        } else if (n2max > 4 && n2max <= 8) {
            Sol_SSPAIv10<8><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X,
                    dev_E, dev_jPTR, n1max, n2max, devA->nCol);
        } else if (n2max > 8 && n2max <= 16) {
            Sol_SSPAIv10<16><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X,
                    dev_E, dev_jPTR, n1max, n2max, devA->nCol);
        } else if (n2max > 16 && n2max <= 32) {
            Sol_SSPAIv10<32><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X,
                    dev_E, dev_jPTR, n1max, n2max, devA->nCol);
        } else if (n2max > 32 && n2max <= 64) {
            Sol_SSPAIv10<64><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X,
                    dev_E, dev_jPTR, n1max, n2max, devA->nCol);
        } else if (n2max > 64 && n2max <= 128) {
            Sol_SSPAIv10<128><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X,
                    dev_E, dev_jPTR, n1max, n2max, devA->nCol);
        } else {
            Sol_SSPAIv10<256><<< blocksPerGrid, threadsPerBlock>>>(dev_tildeA, dev_R, dev_X,
                    dev_E, dev_jPTR, n1max, n2max, devA->nCol);
        }


        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsedTime, start, stop);

        printf("Time = : %8.4f ms \n ", elapsedTime);
        computeTime += elapsedTime;

        hipEventDestroy(start);
        hipEventDestroy(stop);

        printf("\n");
        printf("computeTime = : %12.4f ms \n", computeTime);

        /*+++++++++++++++++Post-GSPAI-Adaptive++++++++++++++++++++++++*/

        printf("Post-GSSPAI is processing.........................\n");
        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start, 0);

        hipFree(dev_J);
        hipFree(dev_I);
        //hipFree(dev_jPTR);
        hipFree(dev_iPTR);

        hipFree(dev_tildeA);
        hipFree(dev_R);
        hipFree(dev_E);
        //hipFree(dev_X);

        devM->nCol = devA->n;
        devM->nRow = devA->n;
        devM->n = devA->n;
        devM->nonzeroes = devA->nonzeroes;

        /*-----assemble mPtr*/
        hipMalloc((void **) &devM->mPtr, sizeof(int) * (devM->nCol + 1));
        hipMemcpy(devM->mPtr, devA->mPtr, sizeof(int) * (devM->nCol + 1), hipMemcpyDeviceToDevice);

        /*-------assemble mIndex and mData*/
        hipMalloc((void **) &devM->mIndex, sizeof(int) * devM->nonzeroes);
        hipMalloc((void **) &devM->mData, sizeof(double) * devM->nonzeroes);

        hipMemcpy(devM->mIndex, devA->mIndex, sizeof(int) * devM->nonzeroes, hipMemcpyDeviceToDevice);

        if (n2max <= 2) {
            WarpSize = 2;
        } else if (n2max > 2 && n2max <= 4) {
            WarpSize = 4;
        } else if (n2max > 4 && n2max <= 8) {
            WarpSize = 8;
        } else if (n2max > 8 && n2max <= 16) {
            WarpSize = 16;
        } else {
            WarpSize = 32;
        }

        blocksPerGrid = (devM->nCol - 1) / (threadsPerBlock / WarpSize) + 1;

        if (n2max <= 2) {
            modifyData4_SSPAIv10<2><<< blocksPerGrid, threadsPerBlock>>>(devM->mData,
                    devM->mPtr, dev_X, dev_jPTR, n2max, devM->nCol);
        } else if (n2max > 2 && n2max <= 4) {
            modifyData4_SSPAIv10<4><<< blocksPerGrid, threadsPerBlock>>>(devM->mData,
                    devM->mPtr, dev_X, dev_jPTR, n2max, devM->nCol);
        } else if (n2max > 4 && n2max <= 8) {
            modifyData4_SSPAIv10<8><<< blocksPerGrid, threadsPerBlock>>>(devM->mData,
                    devM->mPtr, dev_X, dev_jPTR, n2max, devM->nCol);
        } else if (n2max > 8 && n2max <= 16) {
            modifyData4_SSPAIv10<16><<< blocksPerGrid, threadsPerBlock>>>(devM->mData,
                    devM->mPtr, dev_X, dev_jPTR, n2max, devM->nCol);
        } else {
            modifyData4_SSPAIv10<32><<< blocksPerGrid, threadsPerBlock>>>(devM->mData,
                    devM->mPtr, dev_X, dev_jPTR, n2max, devM->nCol);
        }


        hipFree(dev_jPTR);
        hipFree(dev_X);

        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsedTime, start, stop);

        printf("Time = : %8.4f ms \n ", elapsedTime);
        postTime += elapsedTime;

        hipEventDestroy(start);
        hipEventDestroy(stop);

        printf("postTime = : %12.4f ms \n ", postTime);

        //exit(0);

        printf("******************The preconditioner time*************************\n");
        printf("TotalPreTime = : %12.4f ms \n", preTime);
        printf("TotalComputeTime = : %12.4f ms \n", computeTime);
        printf("TotalPostTime = : %12.4f ms \n ", postTime);

        printf("TotalTime = : %12.4f ms \n ", preTime + computeTime + postTime);
        printf("\n");

    
    }

    return preTime + computeTime + postTime;
}

/*
 *Test for Reading the matrix from the file that comes from the SuiteSparse Matrix Collection
 */
int main(int argc, char **argv) {
    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);
    if (deviceCount < 1) {
        printf("错误: 未检测到DCU设备\n");
        return -1;
    }
    printf("检测到 %d 张 DCU，使用第一张DCU运行\n", deviceCount);
    hipSetDevice(0);  // 明确使用第一张DCU

    // 查询并显示设备信息
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("使用DCU设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("Wavefront大小: %d\n", prop.warpSize);

    char filename[50];

    cout << "Input the matrix filename:" << endl;
    cin >> filename;

    CSC_Matrix *CSC_A;
    CSC_A = (CSC_Matrix *) malloc(sizeof(CSC_Matrix));
    readMatrixToCSC(filename, CSC_A);


    hipEvent_t start, stop;
    float preconditioningTime = 0, elapsedTime;

    printf("--------Transfer A values to GPU\n");
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    CSC_Matrix *devCSC_A;
    devCSC_A = (CSC_Matrix *) malloc(sizeof(CSC_Matrix));
    devCSC_A->n = CSC_A->n;
    devCSC_A->nonzeroes = CSC_A->nonzeroes;
    devCSC_A->nCol = CSC_A->n;
    devCSC_A->nRow = CSC_A->n;

    CSC_Matrix *devCSC_M;
    devCSC_M = (CSC_Matrix *) malloc(sizeof(CSC_Matrix));

    hipMalloc((void **) &devCSC_A->mPtr, sizeof(int) * (CSC_A->nCol + 1));
    hipMalloc((void **) &devCSC_A->mIndex, sizeof(int) * CSC_A->nonzeroes);
    hipMalloc((void **) &devCSC_A->mData, sizeof(double) * CSC_A->nonzeroes);

    hipMemcpyAsync(devCSC_A->mData, CSC_A->mData, CSC_A->nonzeroes * sizeof(double), hipMemcpyHostToDevice, 0);
    hipMemcpyAsync(devCSC_A->mIndex, CSC_A->mIndex, CSC_A->nonzeroes * sizeof(int), hipMemcpyHostToDevice, 0);
    hipMemcpyAsync(devCSC_A->mPtr, CSC_A->mPtr, (CSC_A->nCol + 1) * sizeof(int), hipMemcpyHostToDevice, 0);


    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsedTime, start, stop);

    printf("elapsedTime = : %8.4f ms \n ", elapsedTime);
    preconditioningTime += elapsedTime;

    hipEventDestroy(start);
    hipEventDestroy(stop);

    preconditioningTime += StaticSPAIv20(devCSC_A, devCSC_M);



    //transfer CSC_A to CSR_A and CSC_M to CSR_M
    printf("++++++++++++transfer CSC_A to CSR_A and CSC_M to CSR_M\n");
    float solvingTime = 0;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    CSR_Matrix *devCSR_A;
    devCSR_A = (CSR_Matrix *) malloc(sizeof(CSR_Matrix));

    CSR_Matrix *devCSR_M;
    devCSR_M = (CSR_Matrix *) malloc(sizeof(CSR_Matrix));

    devCSR_A->n = devCSC_A->n;
    devCSR_A->nonzeroes = devCSC_A->nonzeroes;
    devCSR_A->nCol = devCSC_A->n;
    devCSR_A->nRow = devCSC_A->n;
    hipMalloc((void **) &devCSR_A->mPtr, sizeof(int) * (devCSR_A->nRow + 1));
    hipMalloc((void **) &devCSR_A->mIndex, sizeof(int) * devCSR_A->nonzeroes);
    hipMalloc((void **) &devCSR_A->mData, sizeof(double) * devCSR_A->nonzeroes);

    devCSR_M->n = devCSC_M->n;
    devCSR_M->nonzeroes = devCSC_M->nonzeroes;
    devCSR_M->nCol = devCSC_M->n;
    devCSR_M->nRow = devCSC_M->n;
    hipMalloc((void **) &devCSR_M->mPtr, sizeof(int) * (devCSR_M->nRow + 1));
    hipMalloc((void **) &devCSR_M->mIndex, sizeof(int) * devCSR_M->nonzeroes);
    hipMalloc((void **) &devCSR_M->mData, sizeof(double) * devCSR_M->nonzeroes);


    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsedTime, start, stop);

    printf("elapsedTime = : %8.4f ms \n ", elapsedTime);
    solvingTime += elapsedTime;

    hipEventDestroy(start);
    hipEventDestroy(stop);

    cuCSC2CSR(devCSC_A->nRow, devCSC_A->nCol, devCSC_A->nonzeroes, devCSC_A->mData, devCSC_A->mIndex, devCSC_A->mPtr,
              devCSR_A->mData, devCSR_A->mIndex, devCSR_A->mPtr);

    cuCSC2CSR(devCSC_M->nRow, devCSC_M->nCol, devCSC_M->nonzeroes, devCSC_M->mData, devCSC_M->mIndex, devCSC_M->mPtr,
              devCSR_M->mData, devCSR_M->mIndex, devCSR_M->mPtr);

    int MAX_ITER = 10000;
    double TOL = 1e-7;
    double *b, *x;
    b = (double *) malloc(sizeof(double) * devCSR_A->n);
    x = (double *) malloc(sizeof(double) * devCSR_A->n);

    for (int j = 0; j < devCSR_A->n; j++) {
        b[j] = 1.0;
        x[j] = 1.0;
    }


    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    double *dev_x, *dev_b;

    //----------allocate the memeory for x and b and compy values from host to device
    hipMalloc((void **) &dev_x, sizeof(double) * devCSR_A->n);
    hipMalloc((void **) &dev_b, sizeof(double) * devCSR_A->n);

    hipMemcpyAsync(dev_x, x, devCSR_A->n * sizeof(double), hipMemcpyHostToDevice, 0);
    hipMemcpyAsync(dev_b, b, devCSR_A->n * sizeof(double), hipMemcpyHostToDevice, 0);

    cublas2_pbicgstabv2(devCSR_A, devCSR_M, dev_b, dev_x, TOL, MAX_ITER);

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsedTime, start, stop);

    printf("elapsedTime = : %8.4f ms \n ", elapsedTime);
    solvingTime += elapsedTime;

    hipEventDestroy(start);
    hipEventDestroy(stop);

    free(b);
    free(x);

    hipFree(devCSC_A->mData);
    hipFree(devCSC_A->mIndex);
    hipFree(devCSC_A->mPtr);
    hipFree(devCSC_M->mData);
    hipFree(devCSC_M->mIndex);
    hipFree(devCSC_M->mPtr);

    hipFree(devCSR_A->mData);
    hipFree(devCSR_A->mIndex);
    hipFree(devCSR_A->mPtr);
    hipFree(devCSR_M->mData);
    hipFree(devCSR_M->mIndex);
    hipFree(devCSR_M->mPtr);

    hipFree(dev_b);
    hipFree(dev_x);

    free(CSC_A);
    free(devCSC_A);
    free(devCSR_A);
    free(devCSC_M);
    free(devCSR_M);

    //-------------++++++++++++++++++++++++++++++++++++++++++++++++++
    printf("-----------------++++++++++++++++++++++++++++++++++++++++\n");
    printf("preconditioningTime = : %12.4f ms \n ", preconditioningTime);
    printf("solvingTime = : %12.4f ms \n ", solvingTime);
    printf("\n");

    printf("totalTime = : %12.4f ms \n ", preconditioningTime + solvingTime);

    return 0;
}



/*多dcu基本工作
int num_devices;
hipGetDeviceCount(&num_devices);*/
