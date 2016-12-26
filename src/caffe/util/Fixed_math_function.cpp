#include "caffe/util/Fixed_math_function.hpp"
#include "sg14/fixed_point"
#include <cblas.h>
typedef sg14::fixed_point<int32_t, -20> myfp;
// typedef F77_CHAR char*;
// typedef F77_INT long;
namespace fixed_point{
  void cblas_sgemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,const CBLAS_TRANSPOSE TransB, const int M, const int N,
    const int K, const myfp alpha, const myfp  *A,
    const int lda, const myfp  *B, const int ldb,
    const myfp beta, myfp  *C, const int ldc){
      char TA, TB;
      #ifdef F77_CHAR
      F77_CHAR F77_TA, F77_TB;
      #else
      #define F77_TA &TA
      #define F77_TB &TB
      #endif

      #ifdef F77_INT
      F77_INT F77_M=M, F77_N=N, F77_K=K, F77_lda=lda, F77_ldb=ldb;
      F77_INT F77_ldc=ldc;
      #else
      #define F77_M M
      #define F77_N N
      #define F77_K K
      #define F77_lda lda
      #define F77_ldb ldb
      #define F77_ldc ldc
      #endif

      extern int CBLAS_CallFromC;
      extern int RowMajorStrg;
      RowMajorStrg = 0;
      CBLAS_CallFromC = 1;
      if( layout == CblasColMajor )
      {
        if(TransA == CblasTrans) TA='T';
        else if ( TransA == CblasConjTrans ) TA='C';
        else if ( TransA == CblasNoTrans )   TA='N';
        else
        {
          cblas_xerbla(2, "cblas_sgemm",
          "Illegal TransA setting, %d\n", TransA);
          CBLAS_CallFromC = 0;
          RowMajorStrg = 0;
          return;
        }

        if(TransB == CblasTrans) TB='T';
        else if ( TransB == CblasConjTrans ) TB='C';
        else if ( TransB == CblasNoTrans )   TB='N';
        else
        {
          cblas_xerbla(3, "cblas_sgemm",
          "Illegal TransB setting, %d\n", TransB);
          CBLAS_CallFromC = 0;
          RowMajorStrg = 0;
          return;
        }

        #ifdef F77_CHAR
        F77_TA = C2F_CHAR(&TA);
        F77_TB = C2F_CHAR(&TB);
        #endif

        F77_sgemm(F77_TA, F77_TB, &F77_M, &F77_N, &F77_K, &alpha, A, &F77_lda, B, &F77_ldb, &beta, C, &F77_ldc);
      } else if (layout == CblasRowMajor)
      {
        RowMajorStrg = 1;
        if(TransA == CblasTrans) TB='T';
        else if ( TransA == CblasConjTrans ) TB='C';
        else if ( TransA == CblasNoTrans )   TB='N';
        else
        {
          cblas_xerbla(2, "cblas_sgemm",
          "Illegal TransA setting, %d\n", TransA);
          CBLAS_CallFromC = 0;
          RowMajorStrg = 0;
          return;
        }
        if(TransB == CblasTrans) TA='T';
        else if ( TransB == CblasConjTrans ) TA='C';
        else if ( TransB == CblasNoTrans )   TA='N';
        else
        {
          cblas_xerbla(2, "cblas_sgemm",
          "Illegal TransA setting, %d\n", TransA);
          CBLAS_CallFromC = 0;
          RowMajorStrg = 0;
          return;
        }
        #ifdef F77_CHAR
        F77_TA = C2F_CHAR(&TA);
        F77_TB = C2F_CHAR(&TB);
        #endif

        F77_sgemm(F77_TA, F77_TB, &F77_N, &F77_M, &F77_K, &alpha, B, &F77_ldb, A, &F77_lda, &beta, C, &F77_ldc);
      } else
      cblas_xerbla(1, "cblas_sgemm",
      "Illegal layout setting, %d\n", layout);
      CBLAS_CallFromC = 0;
      RowMajorStrg = 0;
    }

  void cblas_sgemv(const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const myfp alpha, const myfp  *A, const int lda,
    const myfp  *X, const int incX, const myfp beta,
    myfp  *Y, const int incY){
         char TA;
      #ifdef F77_CHAR
         F77_CHAR F77_TA;
      #else
         #define F77_TA &TA
      #endif
      #ifdef F77_INT
         F77_INT F77_M=M, F77_N=N, F77_lda=lda, F77_incX=incX, F77_incY=incY;
      #else
         #define F77_M M
         #define F77_N N
         #define F77_lda lda
         #define F77_incX incX
         #define F77_incY incY
      #endif

         extern int CBLAS_CallFromC;
         extern int RowMajorStrg;
         RowMajorStrg = 0;

         CBLAS_CallFromC = 1;
         if (layout == CblasColMajor)
         {
            if (TransA == CblasNoTrans) TA = 'N';
            else if (TransA == CblasTrans) TA = 'T';
            else if (TransA == CblasConjTrans) TA = 'C';
            else
            {
               cblas_xerbla(2, "cblas_sgemv","Illegal TransA setting, %d\n", TransA);
               CBLAS_CallFromC = 0;
               RowMajorStrg = 0;
            }
            #ifdef F77_CHAR
               F77_TA = C2F_CHAR(&TA);
            #endif
            F77_sgemv(F77_TA, &F77_M, &F77_N, &alpha, A, &F77_lda, X, &F77_incX,
                      &beta, Y, &F77_incY);
         }
         else if (layout == CblasRowMajor)
         {
            RowMajorStrg = 1;
            if (TransA == CblasNoTrans) TA = 'T';
            else if (TransA == CblasTrans) TA = 'N';
            else if (TransA == CblasConjTrans) TA = 'N';
            else
            {
               cblas_xerbla(2, "cblas_sgemv", "Illegal TransA setting, %d\n", TransA);
               CBLAS_CallFromC = 0;
               RowMajorStrg = 0;
               return;
            }
            #ifdef F77_CHAR
               F77_TA = C2F_CHAR(&TA);
            #endif
            F77_sgemv(F77_TA, &F77_N, &F77_M, &alpha, A, &F77_lda, X,
                      &F77_incX, &beta, Y, &F77_incY);
         }
         else cblas_xerbla(1, "cblas_sgemv", "Illegal layout setting, %d\n", layout);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
    }

  void F77_sgemm(int *layout, char *transpa, char *transpb, int *m, int *n,
      int *k, myfp *alpha, myfp *a, int *lda, myfp *b, int *ldb,
      myfp *beta, myfp *c, int *ldc ) {

        myfp *A, *B, *C;
        int i,j,LDA, LDB, LDC;
        CBLAS_TRANSPOSE transa, transb;

        get_transpose_type(transpa, &transa);
        get_transpose_type(transpb, &transb);

        if (*layout == TEST_ROW_MJR) {
          if (transa == CblasNoTrans) {
            LDA = *k+1;
            A = (myfp *)malloc( (*m)*LDA*sizeof( myfp ) );
            for( i=0; i<*m; i++ )
            for( j=0; j<*k; j++ )
            A[i*LDA+j]=a[j*(*lda)+i];
          }
          else {
            LDA = *m+1;
            A   = ( myfp* )malloc( LDA*(*k)*sizeof( myfp ) );
            for( i=0; i<*k; i++ )
            for( j=0; j<*m; j++ )
            A[i*LDA+j]=a[j*(*lda)+i];
          }
          if (transb == CblasNoTrans) {
            LDB = *n+1;
            B   = ( myfp* )malloc( (*k)*LDB*sizeof( myfp ) );
            for( i=0; i<*k; i++ )
            for( j=0; j<*n; j++ )
            B[i*LDB+j]=b[j*(*ldb)+i];
          }
          else {
            LDB = *k+1;
            B   = ( myfp* )malloc( LDB*(*n)*sizeof( myfp ) );
            for( i=0; i<*n; i++ )
            for( j=0; j<*k; j++ )
            B[i*LDB+j]=b[j*(*ldb)+i];
          }
          LDC = *n+1;
          C   = ( myfp* )malloc( (*m)*LDC*sizeof( myfp ) );
          for( j=0; j<*n; j++ )
          for( i=0; i<*m; i++ )
          C[i*LDC+j]=c[j*(*ldc)+i];
          cblas_sgemm( CblasRowMajor, transa, transb, *m, *n, *k, *alpha, A, LDA,
            B, LDB, *beta, C, LDC );
            for( j=0; j<*n; j++ )
            for( i=0; i<*m; i++ )
            c[j*(*ldc)+i]=C[i*LDC+j];
            free(A);
            free(B);
            free(C);
          }
          else if (*layout == TEST_COL_MJR)
          cblas_sgemm( CblasColMajor, transa, transb, *m, *n, *k, *alpha, a, *lda,
            b, *ldb, *beta, c, *ldc );
            else
            cblas_sgemm( UNDEFINED, transa, transb, *m, *n, *k, *alpha, a, *lda,
              b, *ldb, *beta, c, *ldc );
            }

          }
  void F77_sgemv(int *layout, char *transp, int *m, int *n, myfp *alpha,
          	       myfp *a, int *lda, myfp *x, int *incx, myfp *beta,
          	       myfp *y, int *incy ) {

            myfp *A;
            int i,j,LDA;
            CBLAS_TRANSPOSE trans;

            get_transpose_type(transp, &trans);
            if (*layout == TEST_ROW_MJR) {
               LDA = *n+1;
               A   = ( myfp* )malloc( (*m)*LDA*sizeof( myfp ) );
               for( i=0; i<*m; i++ )
                  for( j=0; j<*n; j++ )
                     A[ LDA*i+j ]=a[ (*lda)*j+i ];
               cblas_sgemv( CblasRowMajor, trans,
          		  *m, *n, *alpha, A, LDA, x, *incx, *beta, y, *incy );
               free(A);
            }
            else if (*layout == TEST_COL_MJR)
               cblas_sgemv( CblasColMajor, trans,
          		  *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy );
            else
               cblas_sgemv( UNDEFINED, trans,
          		  *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy );
  }

  void cblas_saxpy( const int N, const myfp alpha, const myfp *X,
    const int incX, myfp *Y, const int incY){
    #ifdef F77_INT
       F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
    #else
       #define F77_N N
       #define F77_incX incX
       #define F77_incY incY
    #endif
       F77_saxpy( &F77_N, &alpha, X, &F77_incX, Y, &F77_incY);
  }
  void F77_saxpy(const int *N, const myfp *alpha, const myfp *X,
                      const int *incX, myfp *Y, const int *incY){
     cblas_saxpy(*N, *alpha, X, *incX, Y, *incY);
     return;
  }
  myfp cblas_sdot( const int N, const myfp *X,
    const int incX, const myfp *Y, const int incY){
       myfp dot=0;
    #ifdef F77_INT
       F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
    #else
       #define F77_N N
       #define F77_incX incX
       #define F77_incY incY
    #endif
       //F77_sdot_sub( &F77_N, X, &F77_incX, Y, &F77_incY, &dot)
       for(int i=0;i<N,i++){
         dot+=X[i]*Y[i];
       }
       return dot;
  }

  myfp cblas_sasum( const int N, const myfp *X, const int incX)
  {
     myfp asum;
  #ifdef F77_INT
     F77_INT F77_N=N, F77_incX=incX;
  #else
     #define F77_N N
     #define F77_incX incX
  #endif
     F77_sasum_sub( &F77_N, X, &F77_incX, &asum);
     return asum;
  }
  
  void cblas_scopy( const int N, const myfp *X,
                        const int incX, myfp *Y, const int incY)
  {
  #ifdef F77_INT
     F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
  #else
     #define F77_N N
     #define F77_incX incX
     #define F77_incY incY
  #endif
    for(int i=0;i<N;i++){
      Y[i]=X[i];
    }
      //A   = ( myfp* )malloc( LDA*(*k)*sizeof( myfp ) );
     //F77_scopy( &F77_N, X, &F77_incX, Y, &F77_incY);
  }

  void cblas_sscal( const int N, const myfp alpha, myfp *X,
                         const int incX)
  {
    #ifdef F77_INT
       F77_INT F77_N=N, F77_incX=incX;
    #else
       #define F77_N N
       #define F77_incX incX
    #endif
    for(int i=0;i<N;i++){
      X[i]=X[i]*alpha;
    }
      // F77_sscal( &F77_N, &alpha, X, &F77_incX);
  }
