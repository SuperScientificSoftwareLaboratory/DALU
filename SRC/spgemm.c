#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "f2c.h"
#include <math.h>
void print_dense(double *A, int m, int n)
{
    printf("dense\n");
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%lf\t", A[j * m + i]);
        }
        printf("\n");
    }
}
void print_csr(int m, int n, int *csrRowPtr, int *csrColIdx, double *csrVal)
{
    printf("csr\n");
    printf("rowPtr\n");
    for (int i = 0; i <= m; i++)
    {
        printf("%d\t", csrRowPtr[i]);
    }
    printf("\n");
    printf("colIdx\n");
    for (int i = 0; i < m; i++)
    {
        for (int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++)
        {
            printf("%d\t", csrColIdx[j]);
        }
        printf("\n");
    }
    printf("val\n");
    for (int i = 0; i < m; i++)
    {
        for (int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++)
        {
            printf("%lf\t", csrVal[j]);
        }
        printf("\n");
    }
}
void print_csc(int m, int n, int *cscColPtr, int *cscRowIdx, double *cscVal)
{
    printf("csc\n");
    printf("colPtr\n");
    for (int i = 0; i <= n; i++)
    {
        printf("%d\t", cscColPtr[i]);
    }
    printf("\n");
    printf("rowIdx\n");
    for (int i = 0; i < n; i++)
    {
        for (int j = cscColPtr[i]; j < cscColPtr[i + 1]; j++)
        {
            printf("%d\t", cscRowIdx[j]);
        }
        printf("\n");
    }
    printf("val\n");
    for (int i = 0; i < n; i++)
    {
        for (int j = cscColPtr[i]; j < cscColPtr[i + 1]; j++)
        {
            printf("%lf\t", cscVal[j]);
        }
        printf("\n");
    }
}

void dns2csr_step1(int m, int n, int ld, int *csrRowPtrA, double *A)
{
    csrRowPtrA[0] = 0;
    for (int mi = 0; mi < m; mi++)
    {
        int cnt = 0;
        for (int ni = 0; ni < n; ni++)
        {
            cnt = fabs(A[ni * ld + mi]) > 1e-15 ? cnt + 1 : cnt;
        }
        csrRowPtrA[mi + 1] = csrRowPtrA[mi] + cnt;
    }
}

void dns2csr_step2(int m, int n, int ld,
                   int *csrRowPtrA, int *csrColIdxA, double *csrValA,
                   double *A)
{
    for (int mi = 0; mi < m; mi++)
    {
        int cnt = 0;
        for (int ni = 0; ni < n; ni++)
        {
            double val = A[ni * ld + mi];
            if (fabs(A[ni * ld + mi]) > 1e-15)
            {
                csrColIdxA[csrRowPtrA[mi] + cnt] = ni;
                csrValA[csrRowPtrA[mi] + cnt] = val;
                cnt++;
            }
        }
    }
}
void dns2csc_step1(int m, int n, int ld, int *cscColPtrA, double *A)
{
    cscColPtrA[0] = 0;
    for (int i = 0; i < n; i++)
    {
        int cnt = 0;
        double *temp_A = A + i * ld;
        for (int j = 0; j < m; j++)
        {
            cnt = fabs(temp_A[j]) > 1e-15 ? cnt + 1 : cnt;
        }
        cscColPtrA[i + 1] = cscColPtrA[i] + cnt;
    }
}
void dns2csc_step2(int m, int n, int ld,
                   int *cscColPtrA, int *cscRowIdxA, double *cscValA,
                   double *A)
{
    for (int i = 0; i < n; i++)
    {
        int cnt = 0;
        double *temp_A = A + i * ld;
        int colPtr = cscColPtrA[i];
        for (int j = 0; j < m; j++)
        {
            if (fabs(temp_A[j]) > 1e-15)
            {
                cscRowIdxA[colPtr + cnt] = j;
                cscValA[colPtr + cnt] = temp_A[j];
                cnt++;
            }
        }
    }
}
int spmm_csc_msp(int m, int k, int n, int lda, int ldb, double *A, double *B, double *C)
{
    memset(C, 0, sizeof(double) * m * n);

    int *cscColPtrB = (int *)malloc(sizeof(int) * (n + 1));
    dns2csc_step1(k, n, ldb, cscColPtrB, B);
    int nnzB = cscColPtrB[n];
    int *cscRowIdxB = (int *)malloc(sizeof(int) * nnzB);
    double *cscValB = (double *)malloc(sizeof(double) * nnzB);
    dns2csc_step2(k, n, ldb, cscColPtrB, cscRowIdxB, cscValB, B);

    for (int i = 0; i < n; i++)
    {
        for (int j = cscColPtrB[i]; j < cscColPtrB[i + 1]; j++)
        {
            double value_B = cscValB[j];
            double *value_C = C + i * m;
            double *value_A = A + cscRowIdxB[j] * m;
            for (int l = 0; l < m; l++)
            {
                value_C[l] += value_B * value_A[l];
            }
        }
    }
    free(cscColPtrB);
    free(cscRowIdxB);
    free(cscValB);
}
int spmm_csc_spm(int m, int k, int n, int lda, int ldb, double *A, double *B, double *C)
{
    memset(C, 0, sizeof(double) * m * n);
    int *cscColPtrA = (int *)malloc(sizeof(int) * (k + 1));
    dns2csc_step1(m, k, lda, cscColPtrA, A);
    int nnzA = cscColPtrA[k];
    int *cscRowIdxA = (int *)malloc(sizeof(int) * nnzA);
    double *cscValA = (double *)malloc(sizeof(double) * nnzA);
    dns2csc_step2(m, k, lda, cscColPtrA, cscRowIdxA, cscValA, A);

    // print_dense(A,m,k);
    // print_csc(m,k,cscColPtrA, cscRowIdxA, cscValA);
    // print_dense(B,k,n);

    for (int i = 0; i < k; i++)
    {
        for (int j = cscColPtrA[i]; j < cscColPtrA[i + 1]; j++)
        {
            int row_index_A = cscRowIdxA[j];
            double value_A = cscValA[j];
            // double *value_temp_B = B + row_index_A * k;
            for (int l = 0; l < n; l++)
            {
                C[l * m + row_index_A] += value_A * B[l * k + i];
            }
            // double *value_C = C +
        }
    }
    // print_dense(C,m,n);
    // exit(0);
    free(cscColPtrA);
    free(cscRowIdxA);
    free(cscValA);
}
int spgemm_csc(int m, int k, int n, int lda, int ldb, double *A, double *B, double *C)
{
    memset(C, 0, sizeof(double) * m * n);

    int *cscColPtrA = (int *)malloc(sizeof(int) * (k + 1));
    dns2csc_step1(m, k, lda, cscColPtrA, A);
    int nnzA = cscColPtrA[k];
    int *cscRowIdxA = (int *)malloc(sizeof(int) * nnzA);
    double *cscValA = (double *)malloc(sizeof(double) * nnzA);
    dns2csc_step2(m, k, lda, cscColPtrA, cscRowIdxA, cscValA, A);

    int *cscColPtrB = (int *)malloc(sizeof(int) * (n + 1));
    dns2csc_step1(k, n, ldb, cscColPtrB, B);
    int nnzB = cscColPtrB[n];
    int *cscRowIdxB = (int *)malloc(sizeof(int) * nnzB);
    double *cscValB = (double *)malloc(sizeof(double) * nnzB);
    dns2csc_step2(k, n, ldb, cscColPtrB, cscRowIdxB, cscValB, B);
    for (int i = 0; i < n; i++)
    {
        for (int j = cscColPtrB[i]; j < cscColPtrB[i + 1]; j++)
        {
            int row_index_B = cscRowIdxB[j];
            double value_B = cscValB[j];
            double *value_C = C + i * m;
            for (int l = cscColPtrA[row_index_B]; l < cscColPtrA[row_index_B + 1]; l++)
            {
                value_C[cscRowIdxA[l]] += cscValA[l] * value_B;
            }
        }
    }
    free(cscColPtrA);
    free(cscRowIdxA);
    free(cscValA);
    free(cscColPtrB);
    free(cscRowIdxB);
    free(cscValB);
    return 0;
}
int spgemm_csr(int m, int k, int n, int lda, int ldb, double *A, double *B, double *C)
{
    memset(C, 0, sizeof(double) * m * n);
    // convert A to sparse
    int *csrRowPtrA = (int *)malloc(sizeof(int) * (m + 1));
    // dns2csr_step1(m, k, csrRowPtrA, A);
    dns2csr_step1(m, k, lda, csrRowPtrA, A);
    int nnzA = csrRowPtrA[m]; // xiugai*********************************
    int *csrColIdxA = (int *)malloc(sizeof(int) * nnzA);
    double *csrValA = (double *)malloc(sizeof(double) * nnzA);
    // dns2csr_step2(m, k, csrRowPtrA, csrColIdxA, csrValA, A);
    dns2csr_step2(m, k, lda, csrRowPtrA, csrColIdxA, csrValA, A);

    // convert B to sparse
    int *csrRowPtrB = (int *)malloc(sizeof(int) * (k + 1));
    dns2csr_step1(k, n, ldb, csrRowPtrB, B);
    int nnzB = csrRowPtrB[k];
    int *csrColIdxB = (int *)malloc(sizeof(int) * nnzB);
    double *csrValB = (double *)malloc(sizeof(double) * nnzB);
    dns2csr_step2(k, n, ldb, csrRowPtrB, csrColIdxB, csrValB, B);

    // do spgemm (two input matrices are sparse, the output is dense)
    for (int mi = 0; mi < m; mi++)
    {
        for (int posa = csrRowPtrA[mi]; posa < csrRowPtrA[mi + 1]; posa++)
        {
            int colidx = csrColIdxA[posa];
            double val = csrValA[posa];
            for (int posb = csrRowPtrB[colidx]; posb < csrRowPtrB[colidx + 1]; posb++)
            {
                C[mi + csrColIdxB[posb] * m] += val * csrValB[posb];
            }
        }
    }
    free(csrRowPtrA);
    free(csrColIdxA);
    free(csrValA);
    free(csrRowPtrB);
    free(csrColIdxB);
    free(csrValB);
    return 0;
}
int spmm(int m, int k, int n, int lda, int ldb, double *A, double *B, double *C)
{
    memset(C, 0, sizeof(double) * m * n);
    // convert B to sparse
    int *csrRowPtrB = (int *)malloc(sizeof(int) * (k + 1));
    dns2csr_step1(k, n, ldb, csrRowPtrB, B);
    int nnzB = csrRowPtrB[k];
    int *csrColIdxB = (int *)malloc(sizeof(int) * nnzB);
    double *csrValB = (double *)malloc(sizeof(double) * nnzB);
    dns2csr_step2(k, n, ldb, csrRowPtrB, csrColIdxB, csrValB, B);
    //  A*B (A is dense matrix ,B is CSR sparse format matrix)
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
            for (int l = csrRowPtrB[j]; l < csrRowPtrB[j + 1]; l++)
            {
                C[csrColIdxB[l] * m + i] += A[j * lda + i] * csrValB[l];
            }
    }
    free(csrRowPtrB);
    free(csrColIdxB);
    free(csrValB);
    return 0;
}
int spmm_csr_msp(int m, int k, int n, int lda, int ldb, double *A, double *B, double *C)
{
    memset(C, 0, sizeof(double) * m * n);
    // convert B to sparse
    int *csrRowPtrB = (int *)malloc(sizeof(int) * (k + 1));
    dns2csr_step1(k, n, ldb, csrRowPtrB, B);
    int nnzB = csrRowPtrB[k];
    int *csrColIdxB = (int *)malloc(sizeof(int) * nnzB);
    double *csrValB = (double *)malloc(sizeof(double) * nnzB);
    dns2csr_step2(k, n, ldb, csrRowPtrB, csrColIdxB, csrValB, B);
    //  A*B (A is dense matrix ,B is CSR sparse format matrix)
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
            for (int l = csrRowPtrB[j]; l < csrRowPtrB[j + 1]; l++)
            {
                C[csrColIdxB[l] * m + i] += A[j * lda + i] * csrValB[l];
            }
    }
    free(csrRowPtrB);
    free(csrColIdxB);
    free(csrValB);
    return 0;
}

int spmm_csr_spm(int m, int k, int n, int lda, int ldb, double *A, double *B, double *C)
{

    // print_dense(A,m,k);
    // print_dense(B,k,n);
    // print_dense(C,m,n);
    memset(C, 0, sizeof(double) * m * n);
    // convert A to sparse
    int *csrRowPtrA = (int *)malloc(sizeof(int) * (m + 1));
    dns2csr_step1(m, k, lda, csrRowPtrA, A);
    int nnzA = csrRowPtrA[m];
    int *csrColIdxA = (int *)malloc(sizeof(int) * nnzA);
    double *csrValA = (double *)malloc(sizeof(double) * nnzA);
    dns2csr_step2(m, k, lda, csrRowPtrA, csrColIdxA, csrValA, A);

    // print_csr(m,k,csrRowPtrA,csrColIdxA,csrValA);
    // A*B (A is CSR sparse format matrix ,B is dense matrix)
    for (int i = 0; i < m; i++)
    {
        int row_begin = csrRowPtrA[i];
        int row_end = csrRowPtrA[i + 1];
        for (int l = 0; l < n; l++)
        {
            // double value = 0;
            for (int j = row_begin; j < row_end; j++)
            {
                C[m * l + i] += csrValA[j] * B[l * k + csrColIdxA[j]];
            }
        }
    }
    // print_dense(C,m,n);
    // exit(0);
    free(csrRowPtrA);
    free(csrColIdxA);
    free(csrValA);
    return 0;
}
int spgemm(int m, int k, int n, int lda, int ldb, double *A, double *B, double *C)
{
    // 0-2: csr
    // 0: sp*sp
    // 1: m*sp
    // 2: sp*m
    // 3-5: csc
    // 3: sp*sp
    // 4: m*sp
    // 5: sp*m
    //  printf("%d\t%d\t%d\t%d\t%d\n",m,k,n,lda,ldb);
    int gemm_selection = 4;
    switch (gemm_selection)
    {
    case 0:
        spgemm_csr(m, k, n, lda, ldb, A, B, C);
        break;
    case 1:
        spmm_csr_msp(m, k, n, lda, ldb, A, B, C);
        break;
    case 2:
        spmm_csr_spm(m, k, n, lda, ldb, A, B, C);
        break;
    case 3:
        spgemm_csc(m, k, n, lda, ldb, A, B, C);
        break;
    case 4:
        spmm_csc_msp(m, k, n, lda, ldb, A, B, C); // 选这个
        break;
    case 5:
        spmm_csc_spm(m, k, n, lda, ldb, A, B, C);
        break;
    default:
        exit(0);
    }
}