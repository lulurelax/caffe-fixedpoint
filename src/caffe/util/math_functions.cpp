#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
//#include "sg14/fixed_point"
#include <limits>
#include <iostream>
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
//typedef sg14::fixed_point<int32_t,
int log_sign=0;
namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  //LOG(INFO)<<"time flies.";
  //std::cout<<"row: "<<M<<" col: "<<N<<" vec: "<<K<<std::endl;
  std::cout<<"in: "<<std::endl;
  std::cout<<M<<"and"<<K<<"and"<<N<<"and"<<alpha<<"and"<<beta<<std::endl;
//   if(log_sign<1){
//
  for(int i=0;i<10;i++){
    if(i%100==0) std::cout<<std::endl;
    std::cout<<A[i]<<" ";
  }
  for(int i=0;i<10;i++){
    if(i%100==0) std::cout<<std::endl;
    std::cout<<B[i]<<" ";
  }
//   log_sign++;
// }
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
  std::cout<<"out: "<<std::endl;
   for(int i=0;i<10;i++)
     std::cout<<C[i]<<" ";
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  std::cout<<"row: "<<M<<" col: "<<N<<" vec: "<<K<<std::endl;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}
template<>
void caffe_cpu_gemm<myfp>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const myfp alpha, const myfp* A, const myfp* B, const myfp beta,
    myfp* C) {
      float* fa=(float*)malloc(M*K*sizeof(float));
      float* fb=(float*)malloc(N*K*sizeof(float));
      float* fc=(float*)malloc(M*N*sizeof(float));
      for(int i=0;i<M*K;i++){
        fa[i]=float(A[i]);
      }
      for(int i=0;i<N*K;i++){
        fb[i]=float(B[i]);
      }
      for(int i=0;i<N*M;i++){
        fc[i]=float(C[i]);
      }
      float fbeta, falpha;
      fbeta=float(beta);falpha=float(alpha);

      caffe::caffe_cpu_gemm<float>(TransA, TransB, M, N, K, falpha, fa, fb, fbeta, fc);

      for(int i=0;i<M*N;i++){
        C[i]=fc[i];
      }
      free(fa);
      free(fb);
      free(fc);
  // int lda = (TransA == CblasNoTrans) ? K : M;
  // int ldb = (TransB == CblasNoTrans) ? N : K;
  // std::cout<<"row: "<<M<<" col: "<<N<<" vec: "<<K<<std::endl;
  // cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
  //     ldb, beta, C, N);
}


template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}
template <>
void caffe_cpu_gemv<myfp>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const myfp alpha, const myfp* A, const myfp* x,
    const myfp beta, myfp* y) {
      int xlen,ylen;
      float falpha=float(alpha),fbeta=float(beta);
      if(TransA == CblasTrans){xlen=M;ylen=N; }
      else if ( TransA == CblasConjTrans ){xlen=N;ylen=M;}
      else if ( TransA == CblasNoTrans ){xlen=N;ylen=M;}
      else
      {
        // cblas_xerbla(2, "cblas_sgemm",
        // "Illegal TransA setting, %d\n", TransA);
        // CBLAS_CallFromC = 0;
        // RowMajorStrg = 0;
        return;
      }
      float* fa=(float*)malloc(M*N*sizeof(float));
      float* fx=(float*)malloc(xlen*sizeof(float));
      float* fy=(float*)malloc(ylen*sizeof(float));
      for(int i=0;i<N*M;i++){
        fa[i]=float(A[i]);
      }
      for(int i=0;i<xlen;i++){
        fx[i]=float(x[i]);
      }
      for(int i=0;i<ylen;i++){
        fy[i]=float(y[i]);
      }
      caffe_cpu_gemv<float>(TransA, M, N, falpha, fa, fx, fbeta, fy);
      for(int i=0;i<ylen;i++){
        y[i]=fy[i];
      }
      free(fa);
      free(fx);
      free(fy);
  // cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<myfp>(const int N, const myfp alpha, const myfp* X,
    myfp* Y) {
      float falpha=float(alpha);
      float* fx=(float*)malloc(N*sizeof(float));
      //float* fx=(float*)malloc(xlen*sizeof(float));
      float* fy=(float*)malloc(N*sizeof(float));
      for(int i=0;i<N;i++){
        fx[i]=float(X[i]);
      }
      for(int i=0;i<N;i++){
        fy[i]=float(Y[i]);
      }
      cblas_saxpy(N, falpha, fx, 1, fy, 1);
      for(int i=0;i<N;i++){
        Y[i]=fy[i];
      }
      free(fx);
      free(fy);
    }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    //memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    for(int i = 0; i < N; ++i){
      Y[i]=0;
    }
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);
template void caffe_set<myfp>(const int N, const myfp alpha, myfp* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}
template<> void caffe_add_scalar(const int N, const myfp alpha, myfp* Y){
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
      NO_GPU;
      /*
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;

#endif*/
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);
template void caffe_copy<myfp>(const int N, const myfp* X, myfp* Y);


template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}
template <>
void caffe_scal<myfp>(const int N, const myfp alpha, myfp *X) {
  float* fx=(float*)malloc(N*sizeof(float));
  for(int i=0;i<N;i++){
    fx[i]=float(X[i]);
  }
  float falpha=float(alpha);

  cblas_sscal(N, falpha, fx, 1);
  for(int i=0;i<N;i++){
    X[i]=fx[i];
  }
  free(fx);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}
template <>
void caffe_cpu_axpby<myfp>(const int N, const myfp alpha, const myfp* X,
                            const myfp beta, myfp* Y) {
  float* fx=(float*)malloc(N*sizeof(float));
  float* fy=(float*)malloc(N*sizeof(float));
  float falpha=float(alpha),fbeta=float(beta);
  for(int i=0;i<N;i++){
      fx[i]=float(X[i]);
      fy[i]=float(Y[i]);
  }
  cblas_saxpby(N, falpha, fx, 1, fbeta, fy, 1);
  for(int i=0;i<N;i++){
    Y[i]=fy[i];
  }
  free(fx);
  free(fy);
}


template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}
template <>
void caffe_add<myfp>(const int n, const myfp* a, const myfp* b,
    myfp* y) {
      float* fa=(float*)malloc(n*sizeof(float));
      float* fb=(float*)malloc(n*sizeof(float));
      float* fy=(float*)malloc(n*sizeof(float));
      for(int i=0;i<n;i++){
        fa[i]=float(a[i]);
        fb[i]=float(b[i]);
      }
  vsAdd(n, fa, fb, fy);
  for(int i=0;i<n;i++){
    y[i]=fy[i];
    // fb[i]=b[i];
  }
  free(fa);
  free(fb);
  free(fy);
}

template <>
void caffe_sub<myfp>(const int n, const myfp* a, const myfp* b,
    myfp* y) {
  // vsSub(n, a, b, y);
  float* fa=(float*)malloc(n*sizeof(float));
  float* fb=(float*)malloc(n*sizeof(float));
  float* fy=(float*)malloc(n*sizeof(float));
  for(int i=0;i<n;i++){
    fa[i]=float(a[i]);
    fb[i]=float(b[i]);
  }
vsSub(n, fa, fb, fy);
for(int i=0;i<n;i++){
y[i]=fy[i];
// fb[i]=b[i];
}
free(fa);
free(fb);
free(fy);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}
template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}
template <>
void caffe_mul<myfp>(const int n, const myfp* a, const myfp* b,
    myfp* y) {
      float* fa=(float*)malloc(n*sizeof(float));
      float* fb=(float*)malloc(n*sizeof(float));
      float* fy=(float*)malloc(n*sizeof(float));
      for(int i=0;i<n;i++){
        fa[i]=float(a[i]);
        fb[i]=float(b[i]);
      }
    vsMul(n, fa, fb, fy);
    for(int i=0;i<n;i++){
    y[i]=fy[i];
    // fb[i]=b[i];
    }
    free(fa);
    free(fb);
    free(fy);
  // vsMul(n, a, b, y);
}
template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}
template <>
void caffe_div<myfp>(const int n, const myfp* a, const myfp* b,
    myfp* y) {
      float* fa=(float*)malloc(n*sizeof(float));
      float* fb=(float*)malloc(n*sizeof(float));
      float* fy=(float*)malloc(n*sizeof(float));
      for(int i=0;i<n;i++){
        fa[i]=float(a[i]);
        fb[i]=float(b[i]);
      }
    vsDiv(n, fa, fb, fy);
    for(int i=0;i<n;i++){
    y[i]=fy[i];
    // fb[i]=b[i];
    }
    free(fa);
    free(fb);
    free(fy);
  // vsDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}
template <>
void caffe_powx<myfp>(const int n, const myfp* a, const myfp b,
    myfp* y) {
      float* fa=(float*)malloc(n*sizeof(float));
      //float* fb=(float*)malloc(n*sizeof(float));
      float* fy=(float*)malloc(n*sizeof(float));
      for(int i=0;i<n;i++){
        fa[i]=float(a[i]);
        fy[i]=float(y[i]);
      }
      float fb=float(b);
    vsPowx(n, fa, fb, fy);
    for(int i=0;i<n;i++){
    y[i]=fy[i];
    // fb[i]=b[i];
    }
    free(fa);
    // free(fb);
    free(fy);
  // vsPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}
template <>
void caffe_sqr<myfp>(const int n, const myfp* a, myfp* y) {
  float* fa=(float*)malloc(n*sizeof(float));
  // float* fb=(float*)malloc(n*sizeof(float));
  float* fy=(float*)malloc(n*sizeof(float));
  for(int i=0;i<n;i++){
    fa[i]=float(a[i]);
    // fb[i]=b[i];
  }
vsSqr(n, fa, fy);
for(int i=0;i<n;i++){
y[i]=fy[i];
// fb[i]=b[i];
}
free(fa);
// free(fb);
free(fy);
  // vsSqr(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}
template <>
void caffe_exp<myfp>(const int n, const myfp* a, myfp* y) {
  float* fa=(float*)malloc(n*sizeof(float));
  // float* fb=(float*)malloc(n*sizeof(float));
  float* fy=(float*)malloc(n*sizeof(float));
  for(int i=0;i<n;i++){
    fa[i]=float(a[i]);
    // fb[i]=b[i];
  }
vsExp(n, fa, fy);
for(int i=0;i<n;i++){
y[i]=fy[i];
// fb[i]=b[i];
}
free(fa);
// free(fb);
free(fy);
  // vsExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}
template <>
void caffe_log<myfp>(const int n, const myfp* a, myfp* y) {
  float* fa=(float*)malloc(n*sizeof(float));
  // float* fb=(float*)malloc(n*sizeof(float));
  float* fy=(float*)malloc(n*sizeof(float));
  for(int i=0;i<n;i++){
    fa[i]=float(a[i]);
    // fb[i]=b[i];
  }
vsLn(n, fa, fy);
for(int i=0;i<n;i++){
y[i]=fy[i];
// fb[i]=b[i];
}
free(fa);
// free(fb);
free(fy);
  // vsLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}
template <>
void caffe_abs<myfp>(const int n, const myfp* a, myfp* y) {
  float* fa=(float*)malloc(n*sizeof(float));
  // float* fb=(float*)malloc(n*sizeof(float));
  float* fy=(float*)malloc(n*sizeof(float));
  for(int i=0;i<n;i++){
    fa[i]=float(a[i]);
    // fb[i]=b[i];
  }
vsAbs(n, fa, fy);
for(int i=0;i<n;i++){
y[i]=fy[i];
// fb[i]=b[i];
}
free(fa);
// free(fb);
free(fy);
    // vsAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {

  return Dtype(boost::math::nextafter<float>(
      float(b), std::numeric_limits<float>::max()));
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template
myfp caffe_nextafter(const myfp b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
//  float* fr=(float*)malloc(n*sizeof(float));
  float fa=a,fb=b;
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(fa, fb);
  boost::uniform_real<float> random_distribution(fa, caffe_nextafter<float>(fb));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<float> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = Dtype(variate_generator());
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);
template
void caffe_rng_uniform<myfp>(const int n, const myfp a, const myfp b,
                                                              myfp* r);
template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  float fsigma=sigma,fa=a;
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(fsigma, 0);
  boost::normal_distribution<float> random_distribution(fa, fsigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<float> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = Dtype(variate_generator());
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);
template
void caffe_rng_gaussian<myfp>(const int n, const myfp mu,
                                const myfp sigma, myfp* r);
template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  float fp=p;
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(fp, 0);
  CHECK_LE(fp, 1);
  boost::bernoulli_distribution<float> random_distribution(fp);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<float> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);
template
void caffe_rng_bernoulli<myfp>(const int n, const myfp p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  float fp=p;
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(fp, 0);
  CHECK_LE(fp, 1);
  boost::bernoulli_distribution<float> random_distribution(fp);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<float> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);
template
void caffe_rng_bernoulli<myfp>(const int n, const myfp p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}
template <>
myfp caffe_cpu_strided_dot<myfp>(const int n, const myfp* x,
    const int incx, const myfp* y, const int incy) {
      float* fx=(float*)malloc(n*sizeof(float));
      float* fy=(float*)malloc(n*sizeof(float));
      for(int i=0;i<n;i++){
        fx[i]=float(x[i]);
        fy[i]=float(y[i]);
      }
      float returnval=cblas_sdot(n, fx, incx, fy, incy);
      free(fx);
      free(fy);
  return myfp(returnval);

}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);
template
myfp caffe_cpu_dot<myfp>(const int n, const myfp* x, const myfp* y);


template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}
template <>
myfp caffe_cpu_asum<myfp>(const int n, const myfp* x) {
  float* fx=(float*)malloc(n*sizeof(float));
  // float* fy=(float*)malloc(n*sizeof(float));
  for(int i=0;i<n;i++){
    fx[i]=float(x[i]);
    // fy[i]=y[i];
  }
  float reval=cblas_sasum(n, fx, 1);
  free(fx);
  return myfp(reval);
}
template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}
template <>
void caffe_cpu_scale<myfp>(const int n, const myfp alpha, const myfp *x,
                             myfp* y) {
   float* fx=(float*)malloc(n*sizeof(float));
   float* fy=(float*)malloc(n*sizeof(float));
   for(int i=0;i<n;i++){
     fx[i]=float(x[i]);
     fy[i]=float(y[i]);
   }
   float falpha=float(alpha);
  cblas_scopy(n, fx, 1, fy, 1);
  cblas_sscal(n, falpha, fy, 1);
  for(int i=0;i<n;i++){
    // fx[i]=x[i];
    y[i]=fy[i];
  }
  free(fx);
  free(fy);
}
}  // namespace caffe
