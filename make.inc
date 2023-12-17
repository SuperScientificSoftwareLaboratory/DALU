############################################################################
#
#  Program:         SuperLU_DIST
#
#  Module:          make.inc
#
#  Purpose:         Top-level Definitions
#
#  Creation date:   March 1, 2016	version 5.0.0
#
#  Modified:	    October 13, 2017    version 5.2.1
#		    February 20, 2021   version 7.0.0
#		    October 5, 2021     version 7.1.0
#
############################################################################
#
#  The name of the libraries to be created/linked to
#
SuperLUroot = /usr/local
#DSUPERLULIB = $(SuperLUroot)/SRC/libsuperlu_dist.a
#DSUPERLULIB = $(SuperLUroot)/lib/libsuperlu_dist.a
DSUPERLULIB = /usr/local/lib/libsuperlu_dist.a
INCLUDEDIR  = $(SuperLUroot)/include

XSDK_INDEX_SIZE = 
SLU_HAVE_LAPACK = FALSE
HAVE_PARMETIS   = 
HAVE_COMBBLAS   = 
HAVE_CUDA       = 
HAVE_HIP        = 

XSDK_ENABLE_Fortran = ON
ifeq ($(XSDK_ENABLE_Fortran),ON)
#  DFORTRANLIB = $(SuperLUroot)/lib/libsuperlu_dist_fortran.a
  DFORTRANLIB = /usr/local/lib/libsuperlu_dist_fortran.a
  LIBS = $(DFORTRANLIB) $(DSUPERLULIB) /home/wtc/DALU/DALU/CBLAS/libblas.a -lm
  LIBS +=  -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lgfortran -lm -lgcc_s -lgcc -lquadmath -lm -lgcc_s -lgcc -lpthread -lc -lgcc_s -lgcc
else
  LIBS = $(DSUPERLULIB) /home/wtc/DALU/DALU/CBLAS/libblas.a -lm
endif

LIBS	 += 
LIBS	 += 
LIBS 	 += 
LIBS 	 += -lgcc_s -lgcc -lc -lpthread -lgcc_s -lgcc -lmpi 
# LIBS     += 

CUDALIBS =  
LIBS     += $(CUDALIBS)

#
#  The archiver and the flag(s) to use when building archive (library)
#  If your system has no ranlib, set RANLIB = echo.
#
ARCH         = /usr/bin/ar
ARCHFLAGS    = cr
RANLIB       = /usr/bin/ranlib

CC           = /usr/bin/mpicc
CFLAGS 	     = -O3 -DNDEBUG -DUSE_VENDOR_BLAS -fopenmp   
##-fPIC
#CFLAGS      += -D
# CFLAGS     += 
CXX          = /usr/bin/mpicxx
CXXFLAGS     = -O3 -DNDEBUG -fopenmp  
NVCC	     = 
NVCCFLAGS    = -DUSE_VENDOR_BLAS 

NOOPTS       = -O0
FORTRAN	     = /usr/bin/mpif77
FFLAGS	     =   

LOADER       = /usr/bin/mpicxx
LOADOPTS     =      
