#ifndef CAFFE_UTIL_FIXED_MATH_FUNCTION_HPP_
#define CAFFE_UTIL_FIXED_MATH_FUNCTION_HPP_
//#include <sg14/fixed_point>
//typedef sg14::fixed_point<int32_t,-20> myfp;
//matrix multiplication
typedef enum {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
typedef enum {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
typedef enum {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum {CblasLeft=141, CblasRight=142} CBLAS_SIDE;

namespace fix_point{
  template<typename Dtype>
  void fix_gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const Dtype alpha, const Dtype *A,
                 const int lda, const Dtype *B, const int ldb,
                 const Dtype beta, myfp *C, const int ldc);
}
#endif
