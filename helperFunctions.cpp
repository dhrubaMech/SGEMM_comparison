#include <stdio.h>
#include <random>
#include <iostream>

#include "helperFunctions.h"

using namespace std;

void fillRandomMatrix(float* a, const int rows, const int cols){
    static random_device rd;
    static mt19937 prng(rd());
    static uniform_real_distribution<float> dist(0.0f,1.0f);

    for (int i = 0 ; i < rows ; i++){
        for (int j = 0 ; j < cols ; j++){
            // a[i][j] = dist(prng);
            a[i*cols+j] = dist(prng);
        }
    }
}

void show2DMat(const float* a, const int rows, const int cols){
    for (int i = 0 ; i < rows ; i++){
        printf("| ");
        for (int j = 0 ; j < cols ; j++){
            // printf("%f ",a[i][j]);
            printf("%f ",a[i*cols + j]);
        }
        printf("|\n");
    }
    printf("\n");
}

void checkDiff(const float* C1, const float* C2, const int r1, const int c1, const char* comp){
    float error = 0.0f;
    for (int i = 0 ; i < r1 ; i++){
        for (int j = 0 ; j < c1 ; j++){
            // error += abs(C1[i*c1 + j] - C2[i*c1 + j]);
	    error += fabs(C1[i*c1 + j] - C2[i*c1 + j]);
        }
    }
    error /= (r1 * c1);
    printf("Average Error b/w %s : %f\n\n",comp,error);
}

void simpleMatMul(const float* A, const float* B, float* C, const int M, const int N, const int K){
    for (int i = 0 ; i < M ; i++){
        for (int j = 0 ; j < N ; j++){
            float summ = 0.0f;
            for (int k = 0 ; k < K ; k++){
                summ += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = summ;
        }
    }
}

void MatMulOpenMp(const float* A, const float* B, float* C, const int M, const int N, const int K){
    #pragma omp for
    for (int i = 0 ; i < M ; i++){
        for (int j = 0 ; j < N ; j++){
            float summ = 0.0f;
            for (int k = 0 ; k < K ; k++){
                summ += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = summ;
        }
    }
}

float* TransposeMat(const float* A, const int M, const int N){
    float *A_t = new float[N*M];
    for (int m = 0 ; m < M ; m++){
	for (int n = 0 ; n < N ; n++){
	    A_t[n*M + m] = A[m*N + n];
	}
    }
    return A_t;
}

