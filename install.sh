DALU_HOME=$(pwd)
export FILE_NAME="build"
if [ ! -d $FILE_NAME ];then
  mkdir $FILE_NAME
fi
cd $FILE_NAME
rm -rf *
cmake .. -DTPL_ENABLE_PARMETISLIB=False \
-DTPL_BLAS_LIBRARIES="${DALU_HOME}/CBLAS/libblas.a" \
-DCMAKE_C_COMPILER=mpicc \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_Fortran_COMPILER=mpif77 \
