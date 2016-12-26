#ifndef CAFFE_UTIL_FIXED_MATH_FUNCTION_HPP_
#define CAFFE_UTIL_FIXED_MATH_FUNCTION_HPP_
//#include <sg14/fixed_point>
//typedef sg14::fixed_point<int32_t,-20> myfp;
//matrix multiplication
#include "sg14/fixed_point"
typedef sg14::fixed_point<int32_t, -20> myfp;

// typedef enum {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
// typedef enum {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
// typedef enum {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
// typedef enum {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
// typedef enum {CblasLeft=141, CblasRight=142} CBLAS_SIDE;

namespace fix_point{
  void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const myfp alpha, const myfp *A,
                 const int lda, const myfp *B, const int ldb,
                 const myfp beta, myfp *C, const int ldc);

}
#endif
