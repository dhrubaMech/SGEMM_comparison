#include <stdio.h>
#include <chrono>
#include <iostream>
#include <fstream>

#include <cuda.h>

#include "helperFunctions.h"

using namespace std;

#define TILESIZE 32
#define WPT 4            // work per thread
#define RTS TILESIZE/WPT  // reduced tile size

#define WIDTH 2	  // need to keep this multiple of TILESIZE
#if WIDTH == 1
    typedef float floatX;
#elif WIDTH == 2
    typedef float2 floatX;
#elif WIDTH == 4
    typedef float4 floatX;
#else 
    #error "WIDTH value";
#endif

// ts = 32, TSMxTSK 16x64
 
#define TSM  64
#define TSK  32
#define TSN  TSM
#define WPTN WPT
#define WPTM WPTN

#define RTSM TSM/WPTM
#define RTSN TSN/WPTN
#define TotElement TSK*TSM
#define TotThreads RTSM*RTSN
#define LPT  (TotElement + TotThreads - 1)/TotThreads


#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

__global__ void MatMulCUDAnaive(const float* dA, const float* dB, float* dC, const int M, const int N, const int K){
    int row = threadIdx.y + blockIdx.y * blockDim.y ;
    int col = threadIdx.x + blockIdx.x * blockDim.x ;

    if (row < M && col < N){
        float summ = 0.0f;
        for (int k = 0 ; k < K ; k++){
            summ += dA[row*K + k] * dB[k*N + col];
        }
        dC[row*N + col] = summ;
    }
}


__global__ void MatMulCUDAtiling(const float* dA, const float* dB, float* dC, const int M, const int N, const int K){
    // allocating sub arryas within shared Memory
    __shared__ float Asub[TILESIZE][TILESIZE];
    __shared__ float Bsub[TILESIZE][TILESIZE];

    // indexing within the tile
    const int ty = threadIdx.y ;
    const int tx = threadIdx.x ;
    
    const int globalRow = TILESIZE*blockIdx.y + ty;
    const int globalCol = TILESIZE*blockIdx.x + tx;

    float summ = 0.0f;
    for (int t = 0 ; t < (K+TILESIZE-1)/TILESIZE ; t++){
        const int Acol_idx = TILESIZE*t + tx ; // A row idx will be same as global row idx
        const int Brow_idx = TILESIZE*t + ty ; // B col idx will be same as global col idx

        if (globalRow < M && Acol_idx < K){
            Asub[ty][tx] = dA[globalRow*K + Acol_idx];
        }
        else{
            Asub[ty][tx] = 0.0f;
        }

        if (Brow_idx < K && globalCol < N){
            Bsub[ty][tx] = dB[Brow_idx*N + globalCol];
	    // Bsub[tx][ty] = dB[Brow_idx*N + globalCol];
        }
        else{
            Bsub[ty][tx] = 0.0f;
	    // Bsub[tx][ty] = 0.0f;
        }
        __syncthreads();

        for (int k = 0 ; k < TILESIZE ; k++){
            summ += Asub[ty][k] * Bsub[k][tx];
	    // summ += Asub[ty][k] * Bsub[tx][k];
        }
        __syncthreads();
    }

    if ((globalRow < M) && (globalCol < N)){
        dC[globalRow*N + globalCol] = summ;
    }
}

__global__ void MatMulCUDAmoreWPT(const float* dA, const float* dB, float* dC, const int M, const int N, const int K){
    __shared__ float Asub[TILESIZE][TILESIZE];
    __shared__ float Bsub[TILESIZE][TILESIZE];
    
    const int tx = threadIdx.x ;  // this will go 0 -> RTS
    const int ty = threadIdx.y ;  // this will go 0 -> TILESIZE

    const int globalCol = TILESIZE*blockIdx.x + tx;
    const int globalRow = TILESIZE*blockIdx.y + ty;

    float summ[WPT];
    for (int w = 0 ; w < WPT ; w++){
        summ[w] = 0.0f;
    }

    for (int t = 0 ; t < (K + TILESIZE - 1)/TILESIZE ; t++){
        for (int w = 0 ; w < WPT ; w++){
            int Acol_idx = t*TILESIZE + tx;
            int Brow_idx = t*TILESIZE + ty;

            if (globalRow < M && (Acol_idx + w*RTS) < N){
                Asub[ty][tx + w*RTS] = dA[globalRow*K + (Acol_idx + w*RTS)];
            }
            else{
                Asub[ty][tx + w*RTS] = 0.0f ;
            }

            if (Brow_idx < K && (globalCol + w*RTS) < N){
                Bsub[ty][tx + w*RTS] = dB[Brow_idx*N + (globalCol + w*RTS)];
            }
            else{
                Bsub[ty][tx + w*RTS] = 0.0f ;
            }
        }
        __syncthreads();

        for (int k = 0 ; k < TILESIZE ; k++){
            for (int w = 0 ; w < WPT ; w++){
                summ[w] += Asub[ty][k] * Bsub[k][tx + w*RTS];
            }
        }
        __syncthreads();
    }

    for (int w = 0 ; w < WPT ; w++){
        if (globalRow < M && (globalCol + w*RTS) < N){
            dC[globalRow*N + (globalCol + w*RTS)] = summ[w] ;
        }
    }
}


// NOTE : assuming that matrix sizes are divisible of WIDTH.
__global__ void MatMulCUDAvectorsV2(const floatX* dA, const floatX* dB, floatX* dC, const int M, const int N, const int K){
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int globalRow = TILESIZE         * blockIdx.y + ty;
    const int globalCol = (TILESIZE/WIDTH) * blockIdx.x + tx;

    __shared__ floatX Asub[TILESIZE][TILESIZE/WIDTH];
    __shared__ floatX Bsub[TILESIZE][TILESIZE/WIDTH];
    
    #if WIDTH == 1
        floatX summ = 0.0f;
    #elif WIDTH == 2
        floatX summ = make_float2(0.0f, 0.0f);
    #elif WIDTH == 4
        floatX summ = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #endif

    const int NTiles = (K + TILESIZE - 1) / TILESIZE;
    for (int t = 0 ; t < NTiles ; t++){
        int ArowIdx = globalRow;
        int AcolIdx = t*(TILESIZE/WIDTH) + tx;

        int BrowIdx = t*TILESIZE + ty;  
        int BcolIdx = globalCol;

        if (ArowIdx < M && AcolIdx < K/WIDTH)
            Asub[ty][tx] = dA[ArowIdx * (K/WIDTH) + AcolIdx];
        else
	    #if WIDTH == 2
                Asub[ty][tx] = make_float2(0.0f, 0.0f);
	    #endif

        if (BrowIdx < K && BcolIdx < N/WIDTH)
            Bsub[ty][tx] = dB[BrowIdx * (N/WIDTH) + BcolIdx];
        else
	    #if WIDTH == 2
                Bsub[ty][tx] = make_float2(0.0f, 0.0f);
	    #endif
        __syncthreads();

        floatX vecA, vecB;
        float valA;
        for (int k = 0 ; k < (TILESIZE+WIDTH-1)/WIDTH ; k++){
            vecA = Asub[ty][k];
            for (int w = 0 ; w < WIDTH ; w++){
                // this only works for float2, need to add for float and float4 [will do next]		    
                #if WIDTH == 2
		    valA = (w == 0) ? vecA.x : vecA.y;
                    vecB = Bsub[k * WIDTH + w][tx];
                    summ.x += valA * vecB.x;
                    summ.y += valA * vecB.y;
                #endif
            }
        }
        __syncthreads();
    }
    
    if (globalRow < M && globalCol < N/WIDTH){
      dC[globalRow*(N/WIDTH) + globalCol] = summ;

      // to test if every element is getting modified //   
      // dC[globalRow * (N/WIDTH) + globalCol].x = WIDTH*(globalRow * N/WIDTH + globalCol) + 0;
      // dC[globalRow * (N/WIDTH) + globalCol].y = WIDTH*(globalRow * N/WIDTH + globalCol) + 1;
    }
}

__global__ void MatMulCUDArectTiles(const float* dA, const float* dB, float* dC, const int M, const int N, const int K){
    __shared__ float Asub[TSM][TSK];
    __shared__ float Bsub[TSK][TSN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int globalRow = TSM*blockIdx.y + ty;
    const int globalCol = TSN*blockIdx.x + tx;

    float summ = 0.0f;

    const int numTiles = (K + TSK - 1) / TSK;
    for (int t = 0 ; t < numTiles ; t++){
        for (int i = tx; i < TSK; i += blockDim.x){
            int col = TSK * t + i;
            Asub[ty][i] = (globalRow < M && col < K) ? dA[globalRow * K + col] : 0.0f;
        }

        for (int i = ty; i < TSK; i += blockDim.y){
            int row = TSK * t + i;
            Bsub[i][tx] = (row < K && globalCol < N) ? dB[row * N + globalCol] : 0.0f;
        }
        __syncthreads();

        for (int k = 0 ; k < TSK ; k++){
            summ += Asub[ty][k] * Bsub[k][tx];
        }
        __syncthreads();
    }
    if ((globalRow < M) && (globalCol < N)){
        dC[globalRow*N + globalCol] = summ;
        // dC[globalRow*N + globalCol] = globalRow*N + globalCol;
    }
}


__global__ void MatMulCUDArectTilesWPTcol(const float* dA, const float* dB, float* dC, const int M, const int N, const int K){
    __shared__ float Asub[TSM][TSK];
    __shared__ float Bsub[TSK][TSN];

    const int tx = threadIdx.x;    // 0 -> TSN/WPTN
    const int ty = threadIdx.y;    // 0 -> TSM

    const int globalRow = TSM*blockIdx.y + ty;
    const int globalCol = TSN*blockIdx.x + tx * WPTN;

    float summ[WPTN] = {0.0f};

    const int numTiles = (K + TSK -1)/TSK;
    for (int t = 0 ; t < numTiles ; t++){
        for (int i = tx ; i < TSK ; i += blockDim.x){
            int ArowIdx = globalRow;
            int AcolIdx = TSK*t + i;
            Asub[ty][i] = (ArowIdx < M && AcolIdx < K) ? dA[ArowIdx*K + AcolIdx] : 0.0f;
        }

        for (int i = ty ; i < TSK ; i += blockDim.y){
            int BrowIdx = TSK*t + i;
            for (int w = 0; w < WPTN; w++) {
                int BcolIdx = globalCol + w;
                Bsub[i][tx*WPTN + w] = (BrowIdx < K && BcolIdx < N) ? dB[BrowIdx*N + BcolIdx] : 0.0f;
		// Bsub[i][tx + w*blockDim.x] = (BrowIdx < K && BcolIdx < N) ? dB[BrowIdx*N + BcolIdx] : 0.0f;
            }
        }
	__syncthreads();

	for (int k = 0 ; k < TSK ; k++){
	    float temp = Asub[ty][k];
	    for (int w = 0 ; w < WPTN ; w++){
		summ[w] += temp * Bsub[k][tx*WPTN + w];
		// summ[w] += temp * Bsub[k][tx + w*blockDim.x];
	    }
	}
        __syncthreads();
    }

    for (int w = 0 ; w < WPTN ; w++){
        if (globalRow < M && (globalCol+w < N)){
            dC[globalRow*N + globalCol + w] = summ[w];
            // dC[globalRow*N + globalCol+w] = globalRow*N + globalCol+w;
        }
    }
}

__global__ void MatMulCUDArectTilesWPTrow(const float* dA, const float* dB, float* dC, const int M, const int N, const int K){
    __shared__ float Asub[TSM][TSK];
    __shared__ float Bsub[TSK][TSN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int globalRow = TSM*blockIdx.y + ty * WPTN;
    const int globalCol = TSN*blockIdx.x + tx;

    float summ[WPTN] = {0.0f};

    const int numTiles = (K + TSK -1)/TSK;
    for (int t = 0 ; t < numTiles ; t++){
        for (int w = 0 ; w < WPTN ; w++){
            for (int i = tx ; i < TSK ; i += blockDim.x){
                int ArowIdx = globalRow + w;
                int AcolIdx = TSK*t + i;
                Asub[ty*WPTN + w][i] = (ArowIdx < M && AcolIdx < K) ? dA[ArowIdx*K + AcolIdx] : 0.0f;
            }
        }
        for (int i = ty ; i < TSK ; i += blockDim.y){
            int BrowIdx = TSK*t + i;
            int BcolIdx = globalCol;
            Bsub[i][tx] = (BrowIdx < K && BcolIdx < N) ? dB[BrowIdx*N + BcolIdx] : 0.0f;
        }
        __syncthreads();

        for (int w = 0 ; w < WPTN ; w++){
            for (int k = 0 ; k < TSK ; k++){
                summ[w] += Asub[ty*WPTN + w][k] * Bsub[k][tx];
            }
        }
        __syncthreads();
    }
    for (int w = 0 ; w < WPTN ; w++){
        if ((globalRow+w < M) && (globalCol < N)){
            dC[(globalRow+w)*N + globalCol] = summ[w];
            // dC[(globalRow+w)*N + globalCol] = (globalRow+w)*N + globalCol;
        }
    }
}


__global__ void MatMulCUDA2Dregisters(const float* dA, const float* dB, float* dC, const int M, const int N, const int K){
    __shared__ float Asub[TSM][TSK];
    __shared__ float Bsub[TSK][TSM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int globalCol = TSN*blockIdx.x + tx * WPTN;
    const int globalRow = TSM*blockIdx.y + ty * WPTM;

    float summ[WPTM][WPTN];
    for (int wm = 0 ; wm < WPTM ; wm++){
        for (int wn = 0 ; wn < WPTN ; wn++){
	    summ[wm][wn] = 0.0f;
	}
    }

    // allocating space in register for moving memory from shared to register
    float AsubReg, BsubReg[WPTN];

    const int numTiles = (K + TSK -1)/TSK;
    for (int t = 0 ; t < numTiles ; t++){
	for (int wm = 0 ; wm < WPTM ; wm++){
	    for (int i = tx ; i < TSK ; i += blockDim.x){
		int AcolIdx = t*TSK + i;
		int ArowIdx = globalRow + wm;
		Asub[ty*WPTM + wm][i] = (ArowIdx < M) && (AcolIdx < K) ? dA[ArowIdx*K + AcolIdx] : 0.0f;
	    }
	}
	for (int wn = 0 ; wn < WPTN ; wn++){
	    for (int i = ty ; i < TSK ; i += blockDim.y){
	    	int BcolIdx = globalCol + wn;
		int BrowIdx = t*TSK + i;
		Bsub[i][tx*WPTN + wn] = (BrowIdx < K) && (BcolIdx < N) ? dB[BrowIdx*N + BcolIdx] : 0.0f;
	    }
	}
	__syncthreads();

	for (int k = 0 ; k < TSK ; k++){
	    for (int wn = 0 ; wn < WPTN ; wn++){
		BsubReg[wn] = Bsub[k][tx*WPTN + wn];
	    }

	    for (int wm = 0 ; wm < WPTM ; wm++){
		AsubReg = Asub[ty*WPTM + wm][k];
	    	for (int wn = 0 ; wn < WPTN ; wn++){
		    summ[wm][wn] += AsubReg * BsubReg[wn];
		}
	    }
	}
	__syncthreads();
    }
    for (int wm = 0 ; wm < WPTM ; wm++){
        for (int wn = 0 ; wn < WPTN ; wn++){
	    if ((globalRow + wm) < M && (globalCol + wn) < N){
	    	dC[(globalRow + wm)*N + globalCol + wn] = summ[wm][wn];
            	// dC[(globalRow + wm)*N + globalCol + wn] = (globalRow + wm)*N + globalCol + wn;
	    }
	}
    }
}


__global__ void MatMulCUDA2DregisterVector(const floatX* dA, const floatX* dB, float* dC, const int M, const int N, const int K){
    __shared__ float Asub[TSM][TSK];
    __shared__ float Bsub[TSK][TSN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int globalCol = TSN*blockIdx.x + tx * WPTN;
    const int globalRow = TSM*blockIdx.y + ty * WPTM;

    float summ[WPTM][WPTN];
    for (int wm = 0 ; wm < WPTM ; wm++){
        for (int wn = 0 ; wn < WPTN ; wn++){
	    summ[wm][wn] = 0.0f;
	}
    }

    float Areg,Breg[WPTN];

    int numTiles = (K + TSK -1)/TSK;
    for (int t = 0 ; t < numTiles ; t++){
    	for (int i = tx*WIDTH ; i < TSK ; i += blockDim.x*WIDTH){
	    int AcolIdx = t*TSK + i;
	    for (int wm = 0 ; wm < WPTM ; wm++){
		int ArowIdx = globalRow + wm;
                #if WIDTH == 1
	            Asub[ty*WPTM + wm][i] = (ArowIdx < M) && (AcolIdx < K) ? dA[ArowIdx*K + AcolIdx] : 0.0f;
	        #elif WIDTH == 2
		    Asub[ty*WPTM + wm][i]   = (ArowIdx < M) && (AcolIdx+WIDTH-1 < K) ? dA[ArowIdx*K/2 + AcolIdx/2].x : 0.0f;
		    Asub[ty*WPTM + wm][i+1] = (ArowIdx < M) && (AcolIdx+WIDTH-1 < K) ? dA[ArowIdx*K/2 + AcolIdx/2].y : 0.0f;
                #elif WIDTH == 4
		    Asub[ty*WPTM + wm][i]   = (ArowIdx < M) && (AcolIdx+WIDTH-1 < K) ? dA[ArowIdx*K/4 + AcolIdx/4].x : 0.0f;
		    Asub[ty*WPTM + wm][i+1] = (ArowIdx < M) && (AcolIdx+WIDTH-1 < K) ? dA[ArowIdx*K/4 + AcolIdx/4].y : 0.0f;
		    Asub[ty*WPTM + wm][i+2] = (ArowIdx < M) && (AcolIdx+WIDTH-1 < K) ? dA[ArowIdx*K/4 + AcolIdx/4].z : 0.0f;
		    Asub[ty*WPTM + wm][i+3] = (ArowIdx < M) && (AcolIdx+WIDTH-1 < K) ? dA[ArowIdx*K/4 + AcolIdx/4].w : 0.0f;
                #endif
	    }
	}
        for (int i = ty ; i < TSK ; i += blockDim.y){
	    int BrowIdx = t*TSK + i;
	    for (int wn = 0 ; wn < WPTN ; wn += WIDTH){
                int BcolIdx = globalCol + wn;
                #if WIDTH == 1
		    Bsub[i][tx*WPTN] = (BrowIdx < K) && (BcolIdx < N) ? dB[BrowIdx*N + BcolIdx] : 0.0f;
                #elif WIDTH == 2
		    Bsub[i][tx*WPTN + wn]     = (BrowIdx < K) && (BcolIdx+WIDTH-1 < N) ? dB[BrowIdx*N/2 + BcolIdx/2].x : 0.0f;
                    Bsub[i][tx*WPTN + wn + 1] = (BrowIdx < K) && (BcolIdx+WIDTH-1 < N) ? dB[BrowIdx*N/2 + BcolIdx/2].y : 0.0f;
                #elif WIDTH == 4
		    Bsub[i][tx*WPTN + wn    ] = (BrowIdx < K) && (BcolIdx+WIDTH-1 < N) ? dB[BrowIdx*N/4 + BcolIdx/4].x : 0.0f;
		    Bsub[i][tx*WPTN + wn + 1] = (BrowIdx < K) && (BcolIdx+WIDTH-1 < N) ? dB[BrowIdx*N/4 + BcolIdx/4].y : 0.0f;
		    Bsub[i][tx*WPTN + wn + 2] = (BrowIdx < K) && (BcolIdx+WIDTH-1 < N) ? dB[BrowIdx*N/4 + BcolIdx/4].z : 0.0f;
		    Bsub[i][tx*WPTN + wn + 3] = (BrowIdx < K) && (BcolIdx+WIDTH-1 < N) ? dB[BrowIdx*N/4 + BcolIdx/4].w : 0.0f;
                #endif
            }
        }
	__syncthreads();

	for (int k = 0 ; k < TSK ; k++){
	    for (int wn = 0 ; wn < WPTN ; wn++){
		Breg[wn] = Bsub[k][tx*WPTN + wn];
	    }
	
	    for (int wm = 0 ; wm < WPTM ; wm++){
		Areg = Asub[ty*WPTM + wm][k];
	        for (int wn = 0 ; wn < WPTN ; wn++){
		    summ[wm][wn] += Areg * Breg[wn];
		}
	    }
	}
	__syncthreads();
    }
    for (int wm = 0 ; wm < WPTM ; wm++){
	for (int wn = 0 ; wn < WPTN ; wn++){
	    if ((globalRow + wm < M) && (globalCol + wn < N)){
	        dC[(globalRow + wm)*N + (globalCol + wn)] = summ[wm][wn];
                // dC[(globalRow + wm)*N + (globalCol + wn)] = (globalRow + wm)*N + (globalCol + wn);
	    }
	}
    }
}

#define Mx 10
#define Kx 10
__global__ void copy(floatX *dA, float *B, const int M, const int K){
    
    // __shared__ float Asub[Mx][Kx];
    __shared__ float Bsub[Mx][Kx];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    /*
    if (ty == 0){
        for (int wm = 0 ; wm < M ; wm++){
	    int ArowIdx = ty + wm;
	    int AcolIdx = tx;

            //Asub[ty+wm][tx] = 1;

	    Asub[ty+wm][tx*WIDTH]     = (ArowIdx < M) && (AcolIdx < K/2) ? dA[ArowIdx*K/2 + AcolIdx].x : 0.0f;
	    Asub[ty+wm][tx*WIDTH + 1] = (ArowIdx < M) && (AcolIdx < K/2) ? dA[ArowIdx*K/2 + AcolIdx].y : 0.0f;
        }
    }
    */
    
    // for (int wn = 0 ; wn < K/WIDTH ; wn++){
    	int BrowIdx = ty;
	int BcolIdx = tx;

	if (tx*WIDTH + 1 < K){
	Bsub[ty][tx*WIDTH]     = (BrowIdx < M) && (BcolIdx < K/WIDTH) ? dA[BrowIdx*K/WIDTH + BcolIdx].x : 0.0f;
	Bsub[ty][tx*WIDTH + 1] = (BrowIdx < M) && (BcolIdx < K/WIDTH) ? dA[BrowIdx*K/WIDTH + BcolIdx].y : 0.0f;
	}
    // }

    __syncthreads();

    if (tx==0 && ty == 0){
	for (int m = 0 ; m < M ; m++){
	    for (int k = 0 ; k < K ; k++){
	        // B[m*K + k] = Asub[m][k];
		B[m*K + k] = Bsub[m][k];
	    }
	}
    }
}

void checkSubCopying(const float *A, const int M, const int K){

    floatX *dA; float *dB;

    cudaMalloc((void**) &dA, M*K * sizeof(float));
    cudaMalloc((void**) &dB, M*K * sizeof(float));

    cudaMemcpy(dA, A, M*K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks(M,K);
    copy<<<1,blocks>>>(dA,dB,M,K);
    cudaDeviceSynchronize();

    float *B = new float[M*K];
    cudaMemcpy(B, dB, M*K * sizeof(float), cudaMemcpyDeviceToHost);

    show2DMat(A,M,K);
    show2DMat(B,M,K); 

    checkDiff(A,B,M,K,"A to B copy");

    delete[] B;
    cudaFree(dA); cudaFree(dB);
}

int main(){

    const int repeat = 11; 

    const bool CPU_naive    = true ;
    const bool OpenMP_naive = false ;
    const bool CUDA_naive   = false ;
    const bool CUDA_Tiling  = false ;
    const bool CUDA_TilingWithWPT    = false ;
    const bool CUDA_TilingWithVector = false ;
    const bool CUDA_RectTiling       = false ;
    const bool CUDA_RectTilingWPT    = false ; 
    const bool CUDA_2DRegister       = false ;
    const bool CUDA_2DRegisterVector = true ;

    printf("\nTS %d | TSM %d | TSK %d | TSN %d | WPT %d\n",TILESIZE,TSM,TSK,TSN,WPT);

    const int M = 1000, N = 1000, K = 1000;

    float *A = new float[M*K];
    float *B = new float[K*N];

    fillRandomMatrix(A,M,K); fillRandomMatrix(B,K,N);
    // show2DMat(A,M,K);
    // show2DMat(B,N,K);

    // checkSubCopying(A,M,K); exit(1);

    float *C1 = new float[M*N];
    if (CPU_naive){
	// ofstream file("KernelTimings/CPUnaive_N"+to_string(repeat)+".csv");
	for (int r = 0 ; r < 1 ; r++){
	    auto s0 = std::chrono::steady_clock::now();
   	    simpleMatMul(A,B,C1,M,N,K);
   	    auto e0 = std::chrono::steady_clock::now();
    	    std::chrono::duration<double, std::micro> T0 = e0 - s0;
    	    printf("Time taken by Naive MatMul ~ %f microsecs\n\n",T0.count());
	    // file << T0.count() << ((r != repeat-1) ? "," : "");
	}
	// file.close();
    }
    
    // show2DMat(C1,r1,c2);
    
    float *C2 = new float[M*N];
    if (OpenMP_naive){
        auto s1 = std::chrono::steady_clock::now();
        MatMulOpenMp(A,B,C2,M,N,K);
        auto e1 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::micro> T1 = e1 - s1;
        printf("Time taken by OpenMP MatMul ~ %f microsecs\n",T1.count());
        checkDiff(C1,C2,M,N,"CPU naive & OpenMP naive");
    }

    // for CUDA //

    float *C3 = new float[M*N];

    float *dA, *dB, *dC;
    cudaMalloc((void**) &dA, M*K * sizeof(float));
    cudaMalloc((void**) &dB, K*N * sizeof(float));
    cudaMalloc((void**) &dC, M*N * sizeof(float));

    cudaMemcpy(dA, A, M*K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, K*N * sizeof(float), cudaMemcpyHostToDevice);

    // NAIVE //
    if (CUDA_naive){
	dim3 block(TILESIZE,TILESIZE);
    	dim3  grid((N + TILESIZE -1)/TILESIZE , (M + TILESIZE -1)/TILESIZE) ;

	ofstream file("KernelTimings/CUDAnaive_N"+to_string(repeat)+".csv");
	for (int r = 0 ; r < repeat ; r++){
	
	    CHECK_CUDA(cudaMemset(dC, 0.0f, M*N * sizeof(float)));
	    memset(C3, 0.0f, M*N * sizeof(float));

            auto s2 = std::chrono::steady_clock::now();
            MatMulCUDAnaive<<<grid,block>>>(dA,dB,dC,M,N,K);
            CHECK_CUDA(cudaDeviceSynchronize());
            auto e2 = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::micro> T2 = e2 - s2;
            printf("Time taken by naive CUDA MatMul ~ %f microsecs\n",T2.count());
	    file << T2.count() << ((r != repeat-1) ? "," : "");


            CHECK_CUDA(cudaMemcpy(C3, dC, M*N * sizeof(float), cudaMemcpyDeviceToHost));
            checkDiff(C1,C3,M,N,"CPU naive & GPU naive");
            // show2DMat(C3,M,N);
    	}
	file.close();
    }

        
    // TILING //
    if (CUDA_Tiling){
	dim3 block(TILESIZE,TILESIZE);
        dim3  grid((N + TILESIZE -1)/TILESIZE , (M + TILESIZE -1)/TILESIZE) ;

	ofstream file("KernelTimings/CUDAtiling_N"+to_string(repeat)+".csv");
        for (int r = 0 ; r < repeat ; r++){

	    cudaMemset(dC, 0.0f, M*N * sizeof(float));
	    memset(C3, 0.0f, M*N * sizeof(float));

            auto s3 = std::chrono::steady_clock::now();
            MatMulCUDAtiling<<<grid,block>>>(dA,dB,dC,M,N,K);
            cudaDeviceSynchronize();
            auto e3 = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::micro> T3 = e3 - s3;
            printf("Time taken by tiling CUDA MatMul ~ %f microsecs\n",T3.count());
	    file << T3.count() << ((r != repeat-1) ? "," : "");
        
            cudaMemcpy(C3, dC, M*N * sizeof(float), cudaMemcpyDeviceToHost);
            checkDiff(C1,C3,M,N,"CPU naive & GPU tiling");
            // show2DMat(C3,M,N);
	}
	file.close();
    }

    
    // More work per thread // 
    if (CUDA_TilingWithWPT){
	dim3 block(RTS,TILESIZE);
    	dim3  grid((N + TILESIZE -1)/TILESIZE , (M + TILESIZE -1)/TILESIZE) ;

	ofstream file("KernelTimings/CUDAtilingWPT_N"+to_string(repeat)+".csv");
        for (int r = 0 ; r < repeat ; r++){

	    cudaMemset(dC, 0.0f, M*N * sizeof(float));
	    memset(C3, 0.0f, M*N * sizeof(float));
        
	    auto s4 = std::chrono::steady_clock::now();
            MatMulCUDAmoreWPT<<<grid,block>>>(dA,dB,dC,M,N,K);
            cudaDeviceSynchronize();
            auto e4 = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::micro> T4 = e4 - s4;
            printf("Time taken by tiling+WPT CUDA MatMul ~ %f microsecs\n",T4.count());
	    file << T4.count() << ((r != repeat-1) ? "," : "");
        
            cudaMemcpy(C3, dC, M*N * sizeof(float), cudaMemcpyDeviceToHost);
            checkDiff(C1,C3,M,N,"CPU Naive & GPU Tiling + WPT");
            // show2DMat(C3,M,N);
	}
	file.close();
    }


    // CUDA vectors // 
    // [Note : assuming assuming total number of elements in the matrix is divisible by WIDTH or codes fails]
    floatX *dA_f, *dB_f, *dC_f ;
    cudaMalloc((void**) &dA_f, M*(K/WIDTH) * sizeof(floatX)) ;
    cudaMalloc((void**) &dB_f, K*(N/WIDTH) * sizeof(floatX)) ;
    cudaMalloc((void**) &dC_f, M*(N/WIDTH) * sizeof(floatX)) ;

    if (CUDA_TilingWithVector){
	cudaMemcpy(dA_f, A, M*K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dB_f, B, K*N * sizeof(float), cudaMemcpyHostToDevice);

	// float *Bt = TransposeMatrix(B,K,N);
	// cudaMemcpy(dB_f, Bt, K*N * sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemset(dC, 0.0f, M*N * sizeof(float));

	dim3 block(TILESIZE/WIDTH,TILESIZE);
	dim3  grid(((N/WIDTH)+(TILESIZE/WIDTH)-1)/(TILESIZE/WIDTH), (M+TILESIZE-1)/TILESIZE);

	ofstream file("KernelTimings/CUDAtilingVect_N"+to_string(repeat)+".csv");
        for (int r = 0 ; r < repeat ; r++){

	    cudaMemset(dC_f, 0.0f, M*N * sizeof(float));
	    memset(C3, 0.0f, M*N * sizeof(float));

	    auto s5 = std::chrono::steady_clock::now();
            MatMulCUDAvectorsV2<<<grid,block>>>(dA_f,dB_f,dC_f,M,N,K);
            cudaDeviceSynchronize();
	    auto e5 = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::micro> T5 = e5 - s5;
            printf("Time taken by tiling+vectors CUDA MatMul ~ %f microsecs\n",T5.count());
	    file << T5.count() << ((r != repeat-1) ? "," : "");
        
            cudaMemcpy(C3, dC_f, M*N * sizeof(float), cudaMemcpyDeviceToHost);
            checkDiff(C1,C3,M,N,"CPU Naive & GPU Tiling + vectors");
            // show2DMat(C3,M,N);
	}
	file.close();
    }

    if (CUDA_RectTiling){
	cudaMemset(dC, 0.0f, M*N * sizeof(float));
	memset(C3, 0.0f, M*N * sizeof(float));
	
	dim3 block(TSN,TSM);
	dim3  grid((N+TSN-1)/TSN,(M+TSM-1)/TSM);

	auto s6 = std::chrono::steady_clock::now();
	MatMulCUDArectTiles<<<grid,block>>>(dA,dB,dC,M,N,K);
	cudaDeviceSynchronize();
	auto e6 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::micro> T6 = e6 - s6;
        printf("Time taken by rectangular tiling CUDA MatMul ~ %f microsecs\n",T6.count());

	cudaMemcpy(C3, dC, M*N * sizeof(float), cudaMemcpyDeviceToHost);
        checkDiff(C1,C3,M,N,"CPU Naive & GPU rectangular Tiling");
        // show2DMat(C3,M,N); 
    }

    if (CUDA_RectTilingWPT){
	dim3 block(TSN/WPTN,TSM);
	dim3  grid((N+TSN-1)/TSN,(M+TSM-1)/TSM);
	// dim3 block(TSN,TSM/WPTN);
        // dim3  grid((N+TSN-1)/TSN,(M+TSM-1)/TSM);

	ofstream file("KernelTimings/CUDARectilingWPT_N"+to_string(repeat)+".csv");
        for (int r = 0 ; r < repeat ; r++){

	    cudaMemset(dC, 0.0f, M*N * sizeof(float));
            memset(C3, 0.0f, M*N * sizeof(float));

            auto s7 = std::chrono::steady_clock::now();
            MatMulCUDArectTilesWPTcol<<<grid,block>>>(dA,dB,dC,M,N,K);
            // MatMulCUDArectTilesWPTrow<<<grid,block>>>(dA,dB,dC,M,N,K);
	    cudaDeviceSynchronize();
            auto e7= std::chrono::steady_clock::now();
            std::chrono::duration<double, std::micro> T7 = e7 - s7;
            printf("Time taken by rectangular tiling CUDA MatMul ~ %f microsecs\n",T7.count());
	    file << T7.count() << ((r != repeat-1) ? "," : "");

            cudaMemcpy(C3, dC, M*N * sizeof(float), cudaMemcpyDeviceToHost);
            checkDiff(C1,C3,M,N,"CPU Naive & GPU rectangular Tiling");
            // show2DMat(C3,M,N);
	}
	file.close();
    }

    if (CUDA_2DRegister){
        dim3 block(TSN/WPTN,TSM/WPTM);
        dim3  grid((N+TSN-1)/TSN,(M+TSM-1)/TSM);

	// ofstream file("KernelTimings/CUDA2Dregister_N"+to_string(repeat)+".csv");
	ofstream file("EffectTilesize/CUDA2Dregister_N"+to_string(repeat)+"_TSM"+to_string(TSM)+"_TSN"+to_string(TSN)+"_TSK"+to_string(TSK)+"_WPT"+to_string(WPT)+".csv");
        for (int r = 0 ; r < repeat ; r++){

	    cudaMemset(dC, 0.0f, M*N * sizeof(float));
	    memset(C3, 0.0f, M*N * sizeof(float));

            auto s8 = std::chrono::steady_clock::now();
            MatMulCUDA2Dregisters<<<grid,block>>>(dA,dB,dC,M,N,K);
            cudaDeviceSynchronize();
            auto e8= std::chrono::steady_clock::now();
            std::chrono::duration<double, std::micro> T8 = e8 - s8;
            printf("Time taken by 2D register tiling CUDA MatMul ~ %f microsecs\n",T8.count());
	    file << T8.count() << ((r != repeat-1) ? "," : "");

            cudaMemcpy(C3, dC, M*N * sizeof(float), cudaMemcpyDeviceToHost);
            checkDiff(C1,C3,M,N,"CPU Naive & GPU 2D register Tiling");
            // show2DMat(C3,M,N);
	}
	file.close();
    }

    if (CUDA_2DRegisterVector){
	cudaMemcpy(dA_f, A, M*K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dB_f, B, K*N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(TSN/WPTN,TSM/WPTM);
        dim3  grid((N+TSN-1)/TSN,(M+TSM-1)/TSM);

        ofstream file("KernelTimings/CUDA2DregisterVector_N"+to_string(repeat)+".csv");
	// ofstream file("EffectTilesize/CUDA2DregisterVectors_N"+to_string(repeat)+"_TSM"+to_string(TSM)+"_TSN"+to_string(TSN)+"_TSK"+to_string(TSK)+"_WPT"+to_string(WPT)+"_WIDTH"+to_string(WIDTH)+".csv");
        for (int r = 0 ; r < repeat ; r++){

            cudaMemset(dC, 0.0f, M*N * sizeof(float));
            memset(C3, 0.0f, M*N * sizeof(float));

            auto s9 = std::chrono::steady_clock::now();
            MatMulCUDA2DregisterVector<<<grid,block>>>(dA_f,dB_f,dC,M,N,K);
            cudaDeviceSynchronize();
            auto e9= std::chrono::steady_clock::now();
            std::chrono::duration<double, std::micro> T9 = e9 - s9;
            printf("Time taken by 2D register tiling + Vectors CUDA MatMul ~ %f microsecs\n",T9.count());
            file << T9.count() << ((r != repeat-1) ? "," : "");

            cudaMemcpy(C3, dC, M*N * sizeof(float), cudaMemcpyDeviceToHost);
            checkDiff(C1,C3,M,N,"CPU Naive & GPU 2D register + Vectors");
            // show2DMat(C3,M,N);
        }
        file.close();
    }






    
    delete[] A ; delete[] B ;
    delete[] C1; delete[] C2; 
    delete[] C3; 

    cudaFree(dA) ; cudaFree(dB) ; cudaFree(dC);
    cudaFree(dA_f) ; cudaFree(dB_f) ; cudaFree(dC_f);

    return 0;
}
