#ifndef HELPERFUNCTION_H

#define HELPERFUNCTION_H

void fillRandomMatrix(float* a, const int rows, const int cols);

void show2DMat(const float* a, const int rows, const int cols);

void checkDiff(const float* C1, const float* C2, const int r1, const int c1, const char* comp);

void simpleMatMul(const float* A, const float* B, float* C, const int M, const int N, const int K);

void MatMulOpenMp(const float* A, const float* B, float* C, const int M, const int N, const int K);

float* TransposeMat(const float* A, const int M, const int K);

#endif
