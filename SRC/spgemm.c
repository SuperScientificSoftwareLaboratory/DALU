#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "f2c.h"
#include  <math.h>


void dns2csr_step1(int m, int n,int ld, int *csrRowPtrA, double *A)
{
    csrRowPtrA[0] = 0;
    for (int mi = 0; mi < m; mi++)
    {
        int cnt = 0;
        for (int ni = 0; ni < n; ni++)
        {
            cnt = fabs(A[ni * ld + mi] )> 1e-15 ? cnt + 1 : cnt;
        }
        csrRowPtrA[mi+1] = csrRowPtrA[mi] + cnt;
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

int spgemm(int m, int k, int n,int lda,int ldb, double *A,double *B,double *C)
{
       // num_spgemm++;
       // printf("num_spgemm=%d\n",num_spgemm);
    memset(C, 0, sizeof(double) * m * n);

    // convert A to sparse
    int *csrRowPtrA = (int *)malloc(sizeof(int) * (m+1));
    //dns2csr_step1(m, k, csrRowPtrA, A);
    dns2csr_step1(m, k, lda,csrRowPtrA, A);
    int nnzA = csrRowPtrA[m];//xiugai*********************************
    int *csrColIdxA = (int *)malloc(sizeof(int) * nnzA);
    double *csrValA = (double *)malloc(sizeof(double) * nnzA);
    //dns2csr_step2(m, k, csrRowPtrA, csrColIdxA, csrValA, A);
    dns2csr_step2(m, k, lda,csrRowPtrA, csrColIdxA, csrValA, A);



    // convert B to sparse
    int *csrRowPtrB = (int *)malloc(sizeof(int) * (k+1));
    dns2csr_step1(k, n, ldb,csrRowPtrB, B);
    int nnzB = csrRowPtrB[k];
    int *csrColIdxB = (int *)malloc(sizeof(int) * nnzB);
    double *csrValB = (double *)malloc(sizeof(double) * nnzB);
    dns2csr_step2(k, n, ldb,csrRowPtrB, csrColIdxB, csrValB, B);


    // do spgemm (two input matrices are sparse, the output is dense)

    for (int mi = 0; mi < m; mi++)
    {
        for (int posa = csrRowPtrA[mi]; posa < csrRowPtrA[mi+1]; posa++)
        {
            int colidx = csrColIdxA[posa];
            double val = csrValA[posa];
            for (int posb = csrRowPtrB[colidx]; posb < csrRowPtrB[colidx+1]; posb++)
            {
                C[mi  + csrColIdxB[posb]*m] += val * csrValB[posb];
		
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


/*
void dns2csr_step1(int m, int n, int *csrRowPtrA, double *A)
{
    csrRowPtrA[0] = 0;
    for (int mi = 0; mi < m; mi++)
    {
        int cnt = 0;
        for (int ni = 0; ni < n; ni++)
        {
            cnt = fabs(A[ni * m + mi] )> 1e-15 ? cnt + 1 : cnt;
        }
        csrRowPtrA[mi+1] = csrRowPtrA[mi] + cnt;
    }
}

void dns2csr_step2(int m, int n, 
                   int *csrRowPtrA, int *csrColIdxA, double *csrValA, 
                   double *A)
{
    for (int mi = 0; mi < m; mi++)
    {
        int cnt = 0;
        for (int ni = 0; ni < n; ni++)
        {
            double val = A[ni * m + mi];
            if (fabs(A[ni * m + mi]) >1e-15)
            {
                csrColIdxA[csrRowPtrA[mi] + cnt] = ni;
                csrValA[csrRowPtrA[mi] + cnt] = val;
                cnt++;
            }
        }
    }
}

int spgemm(int m, int k, int n, double *A,double *B,double *C)
{
    memset(C, 0, sizeof(double) * m * n);

    // convert A to sparse
    int *csrRowPtrA = (int *)malloc(sizeof(int) * (m+1));
    dns2csr_step1(m, k, csrRowPtrA, A);
    int nnzA = csrRowPtrA[m];
    int *csrColIdxA = (int *)malloc(sizeof(int) * nnzA);
    double *csrValA = (double *)malloc(sizeof(double) * nnzA);
    dns2csr_step2(m, k, csrRowPtrA, csrColIdxA, csrValA, A);


    // convert B to sparse
    int *csrRowPtrB = (int *)malloc(sizeof(int) * (k+1));
    dns2csr_step1(k, n, csrRowPtrB, B);
    int nnzB = csrRowPtrB[k];
    int *csrColIdxB = (int *)malloc(sizeof(int) * nnzB);
    double *csrValB = (double *)malloc(sizeof(double) * nnzB);
    dns2csr_step2(k, n, csrRowPtrB, csrColIdxB, csrValB, B);



    // do spgemm (two input matrices are sparse, the output is dense)
    for (int mi = 0; mi < m; mi++)
    {
        for (int posa = csrRowPtrA[mi]; posa < csrRowPtrA[mi+1]; posa++)
        {
            int colidx = csrColIdxA[posa];
            double val = csrValA[posa];
            for (int posb = csrRowPtrB[colidx]; posb < csrRowPtrB[colidx+1]; posb++)
            {
                C[mi  + csrColIdxB[posb]* m] += val * csrValB[posb];
		
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

*/


