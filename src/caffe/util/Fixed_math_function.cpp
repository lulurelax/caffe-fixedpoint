#include "caffe/util/Fixed_math_function.hpp"
#include "sg14/fixed_point"
typedef sg14::fixed_point<int32_t, -20> myfp;
namespace fixed_point{
  template<typename Dtype>
  void fix_gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const Dtype alpha, const Dtype *A,
                 const int lda, const Dtype *B, const int ldb,
                 const Dtype beta, Dtype *C, const int ldc){
                   
                 }
  template void fix_gemm<myfp>(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const Dtype alpha, const Dtype *A,
                 const int lda, const Dtype *B, const int ldb,
                 const Dtype beta, Dtype *C, const int ldc);
}
