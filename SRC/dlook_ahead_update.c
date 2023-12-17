/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/************************************************************************/
/*! @file
 * \brief Look-ahead update of the Schur complement.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 5.4) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 1, 2014
 *
 * Modified:
 *  September 18, 2017
 *  June 1, 2018  add parallel AWPM pivoting; add back arrive_at_ublock()
 *
 */

#include <assert.h>  /* assertion doesn't work if NDEBUG is defined */

iukp = iukp0; /* point to the first block in index[] */
rukp = rukp0; /* point to the start of nzval[] */
j = jj0 = 0;  /* After the j-loop, jj0 points to the first block in U
                 outside look-ahead window. */

#if 0
for (jj = 0; jj < nub; ++jj) assert(perm_u[jj] == jj); /* Sherry */
#endif

#ifdef ISORT
while (j < nub && iperm_u[j] <= k0 + num_look_aheads)
#else
while (j < nub && perm_u[2 * j] <= k0 + num_look_aheads)
#endif
{
    double zero = 0.0;

#if 1
    /* Search is needed because a permutation perm_u is involved for j  */
    /* Search along the row for the pointers {iukp, rukp} pointing to
     * block U(k,j).
     * j    -- current block in look-ahead window, initialized to 0 on entry
     * iukp -- point to the start of index[] metadata
     * rukp -- point to the start of nzval[] array
     * jb   -- block number of block U(k,j), update destination column
     */
    arrive_at_ublock(
		     j, &iukp, &rukp, &jb, &ljb, &nsupc,
         	     iukp0, rukp0, usub, perm_u, xsup, grid
		    );
#else
    jb = usub[iukp];
    ljb = LBj (jb, grid);     /* Local block number of U(k,j). */
    nsupc = SuperSize(jb);
    iukp += UB_DESCRIPTOR; /* Start fstnz of block U(k,j). */
#endif

    j++;
    jj0++;
    jj = iukp;

    while (usub[jj] == klst) ++jj; /* Skip zero segments */

    ldu = klst - usub[jj++];
    ncols = 1;

    /* This loop computes ldu. */
    for (; jj < iukp + nsupc; ++jj) { /* for each column jj in block U(k,j) */
        segsize = klst - usub[jj];
        if (segsize) {
            ++ncols;
            if (segsize > ldu)  ldu = segsize;
        }
    }
#if ( DEBUGlevel>=3 )
    ++num_update;
#endif

#if ( DEBUGlevel>=3 )
    printf ("(%d) k=%d,jb=%d,ldu=%d,ncols=%d,nsupc=%d\n",
	    iam, k, jb, ldu, ncols, nsupc);
    ++num_copy;
#endif

    /* Now copy one block U(k,j) to bigU for GEMM, padding zeros up to ldu. */
    tempu = bigU; /* Copy one block U(k,j) to bigU for GEMM */
    for (jj = iukp; jj < iukp + nsupc; ++jj) {
        segsize = klst - usub[jj];
        if (segsize) {
            lead_zero = ldu - segsize;
            for (i = 0; i < lead_zero; ++i) tempu[i] = zero;
            tempu += lead_zero;
            for (i = 0; i < segsize; ++i) {
                tempu[i] = uval[rukp + i];
            }
            rukp += segsize;
            tempu += segsize;
        }
    }
    tempu = bigU; /* set back to the beginning of the buffer */

    nbrow = lsub[1]; /* number of row subscripts in L(:,k) */
    if (myrow == krow) nbrow = lsub[1] - lsub[3]; /* skip diagonal block for those rows. */
    // double ttx =SuperLU_timer_();

    int current_b = 0; /* Each thread starts searching from first block.
                          This records the moving search target.           */
    lptr = lptr0; /* point to the start of index[] in supernode L(:,k) */
    luptr = luptr0;

#ifdef _OPENMP
    /* Sherry -- examine all the shared variables ??
       'firstprivate' ensures that the private variables are initialized
       to the values before entering the loop.  */
#pragma omp parallel for \
    firstprivate(lptr,luptr,ib,current_b) private(lb) \
    default(shared) schedule(dynamic)
#endif
    for (lb = 0; lb < nlb; lb++) { /* Loop through each block in L(:,k) */
        int temp_nbrow; /* automatic variable is private */

        /* Search for the L block that my thread will work on.
           No need to search from 0, can continue at the point where
           it is left from last iteration.
           Note: Blocks may not be sorted in L. Different thread picks up
	   different lb.   */
        for (; current_b < lb; ++current_b) {
            temp_nbrow = lsub[lptr + 1];    /* Number of full rows. */
            lptr += LB_DESCRIPTOR;  /* Skip descriptor. */
            lptr += temp_nbrow;   /* move to next block */
            luptr += temp_nbrow;  /* move to next block */
        }

#ifdef _OPENMP
        int_t thread_id = omp_get_thread_num ();
#else
        int_t thread_id = 0;
#endif
        double * tempv = bigV + ldt*ldt*thread_id;

        int *indirect_thread  = indirect + ldt * thread_id;
        int *indirect2_thread = indirect2 + ldt * thread_id;
        ib = lsub[lptr];        /* block number of L(i,k) */
        temp_nbrow = lsub[lptr + 1];    /* Number of full rows. */
	/* assert (temp_nbrow <= nbrow); */

        lptr += LB_DESCRIPTOR;  /* Skip descriptor. */

	/*if (thread_id == 0) tt_start = SuperLU_timer_();*/

        /* calling gemm */
	stat->ops[FACT] += 2.0 * (flops_t)temp_nbrow * ldu * ncols;

//update by liwenhao______________________________________________________________start
int res_syy;
//计算AB稀疏度：
int numA1=0;
int numB1=0;
int numC1=0;
int a_dai1=0;
int b_dai1=0;
float sp_a1,sp_b1;
double avg_rowA1 = 0;
double avg_colA1 = 0;
double avg_rowB1 = 0;
double avg_colB1 = 0;
double stand_rowA1=0;
double stand_rowB1=0;
double stand_colA1=0;
double stand_colB1=0;


if(temp_nbrow*ldu*ncols<GEMM_THRESHOLD_NUM){
	//当m*n*k小于阈值时，将res_syy设为1
	res_syy=1;
}else{
    
//计算A的非0元个数
    for(int i=0;i<temp_nbrow;i++){
        for(int j=0;j<ldu;j++){
            if(fabs(lusup[luptr + (knsupc - ldu) * nsupr+nsupr*j+i])> 1e-15){
                numA1++;
            }
        }
    }
//计算B的非0元个数
    for(int i=0;i<ldu;i++){
        for(int j=0;j<ncols;j++){
		    if(fabs(lusup[luptr + (knsupc - ldu) * nsupr+nsupr*j+i])> 1e-15){
                numB1++;
            }
        }
    }
//A 带宽
    int a_start1=0;
    int a_end1=0;
    for (int l = 0; l < ldu*temp_nbrow; l++) {
        if(fabs(lusup[luptr + (knsupc - ldu) * nsupr+l])> 1e-15){
            a_start1=l;
            break ;
        }    
	}
    for (int l = 0; l < ldu*temp_nbrow; l++) {
        if(fabs(lusup[luptr + (knsupc - ldu) * nsupr+l])> 1e-15){
            a_end1=l;
        }
	}
    a_dai1=a_end1-a_start1;
//B 带宽
    int b_start1=0;
    int b_end1=0;
    for (int l = 0; l < ldu*ncols; l++) {
        if(fabs(tempu[l])>1e-15){
            b_start1=l;
            break ;
        }
	}
    for (int l = 0; l < ldu*ncols; l++) {
        if(fabs(tempu[l])>1e-15)
            b_end1=l;
	}
    int b_dai1=b_end1-b_start1;	
//-------------------------
//计算A、B的稀疏度
    int lenga1=ldu*temp_nbrow;
    sp_a1=(float) numA1/lenga1;
    int lengb1=ncols*ldu;
    sp_b1=(float) numB1/lengb1;
    int lengc1=temp_nbrow*ncols;


    avg_rowA1=(double)numA1/temp_nbrow;
    avg_colA1=(double)numA1/ldu;
    avg_rowB1=(double)numB1/ldu;
    avg_colB1=(double)numB1/ncols;

//A standard deviation of non-zeros row 
    int row_noA1[temp_nbrow];
    for(int i=0;i<temp_nbrow;i++){
        row_noA1[i]=0;
        for(int j=0;j<ldu;j++){
            if(fabs(lusup[luptr + (knsupc - ldu) * nsupr+nsupr*j+i])> 1e-15){
                row_noA1[i]++;
            }
        }
    }
    double tmpA1=0;
    for(int i=0;i<temp_nbrow;i++){
        double t=row_noA1[i]-avg_rowA1;
        tmpA1=tmpA1 + t*t;  
    }
   
    stand_rowA1=tmpA1/temp_nbrow;

	//B standard deviation of non-zeros row
    int row_noB1[ldu];
    for(int i=0;i<ldu;i++){
        row_noB1[i]=0;
        for(int j=0;j<ncols;j++){
		    if(fabs(tempu[ldu*j+i])>1e-15){
                row_noB1[i]++;
            }
        }
    }

    double tmpB1=0;
    for(int i=0;i<ldu;i++){
        double t=row_noB1[i]-avg_rowB1;
        tmpB1=tmpB1 + t*t;
    }
   
    stand_rowB1=tmpB1/ldu;

	//A standard deviation of non-zeros col
    
    int col_noA1[ldu];
    for(int i=0;i<ldu;i++){
        col_noA1[i]=0;
        for(int j=0;j<temp_nbrow;j++){
		    if(fabs(lusup[luptr + (knsupc - ldu) * nsupr+nsupr*i+j])> 1e-15){
                col_noA1[i]++;
            }
        }
    }
    stand_colA1=0;
    double tmpA11=0;
    for(int i=0;i<ldu;i++){
        double t=col_noA1[i]-avg_colA1;
        tmpA11=tmpA11 + t*t; 
    }
    stand_colA1=tmpA11/ldu;
//B standard deviation of non-zeros col
    
    int col_noB1[ncols];
    for(int i=0;i<ncols;i++){
        col_noB1[i]=0;
        for(int j=0;j<ldu;j++){
		    if(fabs(tempu[ldu*i+j])>1e-15){
                col_noB1[i]++;
            }
        }
    }
    stand_colB1=0;
    double tmpB11=0;
    for(int i=0;i<ncols;i++){
        double t=col_noB1[i]-avg_colB1;
        tmpB11=tmpB11 + t*t;    
    }

    stand_colB1=tmpB11/ncols;
	    // printf("%lf,%d,%lf\n",tmpB11,ncols,stand_colB1);

//#include "syy.c"
	
	double test_vectors[1][15] = {
    {temp_nbrow,ldu,ncols, numA1,numB1,sp_a1,sp_b1,a_dai1,b_dai1,
 	avg_rowA1,avg_rowB1,avg_colA1,avg_colB1,stand_colA1,stand_colB1},
    };

		// printf("%d,%d,%d,%d,%d,%lf,%lf,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf\n"
		// ,temp_nbrow,ldu,ncols, numA1,numB1,sp_a1,sp_b1,a_dai1,b_dai1,avg_rowA1,avg_rowB1,avg_colA1,avg_colB1,stand_colA1,stand_colB1);

	// double test_vectors[1][10] = {
    // {ldu,sp_a1,sp_b1,a_dai1,b_dai1,
 	// avg_rowA1,avg_rowB1,avg_colA1,avg_colB1,stand_colA1},
    // };

    double **vectors;     
    vectors = (double **)malloc(1*sizeof(double *)); 
    for(int i=0; i<1; i++) {
        vectors[i] = test_vectors[i];
    }
    
    int *res = predict(tree, vectors, 1);
    
	res_syy = res[0];
	//printf("res=%d\n",res_syy);
    }

    double look_ahead_tempt =SuperLU_timer_();
    if ( res_syy == 1 ){
        #if defined (USE_VENDOR_BLAS)
                dgemm_("N", "N", &temp_nbrow, &ncols, &ldu, &alpha,
                        &lusup[luptr + (knsupc - ldu) * nsupr], &nsupr,
                        tempu, &ldu, &beta, tempv, &temp_nbrow, 1, 1);
        #else
                dgemm_("N", "N", &temp_nbrow, &ncols, &ldu, &alpha,
                        &lusup[luptr + (knsupc - ldu) * nsupr], &nsupr,
                        tempu, &ldu, &beta, tempv, &temp_nbrow );
        #endif
    }else{
        spgemm(temp_nbrow, ldu,ncols,nsupr,ldu, &lusup[luptr + (knsupc - ldu) * nsupr],tempu, tempv);
    }

    look_ahead_gemm+=SuperLU_timer_()-look_ahead_tempt;


//update by liwenhao______________________________________________________________end
#if 0
	if (thread_id == 0) {
	    tt_end = SuperLU_timer_();
	    LookAheadGEMMTimer += tt_end - tt_start;
	    tt_start = tt_end;
	}
#endif
        /* Now scattering the output. */
        if (ib < jb) {    /* A(i,j) is in U. */
            dscatter_u (ib, jb,
                       nsupc, iukp, xsup,
                       klst, temp_nbrow,
                       lptr, temp_nbrow, lsub,
                       usub, tempv, Ufstnz_br_ptr, Unzval_br_ptr, grid);
        } else {          /* A(i,j) is in L. */
            dscatter_l (ib, ljb, nsupc, iukp, xsup, klst, temp_nbrow, lptr,
                       temp_nbrow, usub, lsub, tempv,
                       indirect_thread, indirect2_thread,
                       Lrowind_bc_ptr, Lnzval_bc_ptr, grid);
        }

        ++current_b;         /* Move to next block. */
        lptr += temp_nbrow;
        luptr += temp_nbrow;

#if 0
	if (thread_id == 0) {
	    tt_end = SuperLU_timer_();
	    LookAheadScatterTimer += tt_end - tt_start;
	}
#endif
    } /* end parallel for lb = 0, nlb ... all blocks in L(:,k) */

    iukp += nsupc; /* Mov to block U(k,j+1) */

    /* =========================================== *
     * == factorize L(:,j) and send if possible == *
     * =========================================== */
    kk = jb; /* destination column that is just updated */
    kcol = PCOL (kk, grid);
#ifdef ISORT
    kk0 = iperm_u[j - 1];
#else
    kk0 = perm_u[2 * (j - 1)];
#endif
    look_id = kk0 % (1 + num_look_aheads);

    if (look_ahead[kk] == k0 && kcol == mycol) {
        /* current column is the last dependency */
        look_id = kk0 % (1 + num_look_aheads);

        /* Factor diagonal and subdiagonal blocks and test for exact
           singularity.  */
        factored[kk] = 0;

        double tt1 = SuperLU_timer_();

        PDGSTRF2(options, kk0, kk, thresh, Glu_persist, grid, Llu,
                  U_diag_blk_send_req, tag_ub, stat, info);

        pdgstrf2_timer += SuperLU_timer_() - tt1;

        /* stat->time7 += SuperLU_timer_() - ttt1; */

        /* Multicasts numeric values of L(:,kk) to process rows. */
        send_req = send_reqs[look_id];
        msgcnt = msgcnts[look_id];

        lk = LBj (kk, grid);    /* Local block number. */
        lsub1 = Lrowind_bc_ptr[lk];
        lusup1 = Lnzval_bc_ptr[lk];
        if (lsub1) {
            msgcnt[0] = lsub1[1] + BC_HEADER + lsub1[0] * LB_DESCRIPTOR;
            msgcnt[1] = lsub1[1] * SuperSize (kk);
        } else {
            msgcnt[0] = 0;
            msgcnt[1] = 0;
        }

        scp = &grid->rscp;      /* The scope of process row. */
        for (pj = 0; pj < Pc; ++pj) {
            if (ToSendR[lk][pj] != EMPTY) {
#if ( PROFlevel>=1 )
                TIC (t1);
#endif
                MPI_Isend (lsub1, msgcnt[0], mpi_int_t, pj,
                           SLU_MPI_TAG (0, kk0) /* (4*kk0)%tag_ub */ ,
                           scp->comm, &send_req[pj]);
                MPI_Isend (lusup1, msgcnt[1], MPI_DOUBLE, pj,
                           SLU_MPI_TAG (1, kk0) /* (4*kk0+1)%tag_ub */ ,
                           scp->comm, &send_req[pj + Pc]);
#if ( PROFlevel>=1 )
                TOC (t2, t1);
                stat->utime[COMM] += t2;
                msg_cnt += 2;
                msg_vol += msgcnt[0] * iword + msgcnt[1] * dword;
#endif
#if ( DEBUGlevel>=2 )
                printf ("[%d] -2- Send L(:,%4d): #lsub %4d, #lusup %4d to Pj %2d, tags %d:%d \n",
                        iam, kk, msgcnt[0], msgcnt[1], pj,
			SLU_MPI_TAG(0,kk0), SLU_MPI_TAG(1,kk0));
#endif
            }  /* end if ( ToSendR[lk][pj] != EMPTY ) */
        } /* end for pj ... */
    } /* end if( look_ahead[kk] == k0 && kcol == mycol ) */
} /* end while j < nub and perm_u[j] <k0+NUM_LOOK_AHEAD */

