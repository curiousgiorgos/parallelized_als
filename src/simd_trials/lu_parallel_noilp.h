// Parallel implementation of LU decomposition
// Using a single FMA 

#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <regex>
#include <tuple> 
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <cmath>
#include <immintrin.h>
#include <chrono>


typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SparseMatrixType; // declares a column-major sparse matrix type of doulbe
typedef Eigen::Triplet<float> T; // declares Eigen Triple type to hold <row,column,value>
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixType;
typedef Eigen::TriangularView<MatrixType, Eigen::Upper> UpperMatrixType;

void lu_fnmadd(int j, int k, int n, MatrixType& A, MatrixType& L); 

MatrixType lu_decomposition(MatrixType& A) {
    A.transposeInPlace();
    int n = A.rows();
    
    //MatrixType I = MatrixType::Identity(n, n);
    //UpperMatrixType L = I.template triangularView<Eigen::Upper>();
    MatrixType L = MatrixType::Identity(n, n);

    for (int k = 0; k < n - 1; ++k) {
        Eigen::VectorXf vec = A.diagonal();
        int i = k+1;
        int cutoff = ((n - (i+1)) /8 ) * 8;
        
        __m256 r_den = _mm256_set1_ps(1/vec(k));

        for (int i = k + 1; i < cutoff; i+=8) {
            __m256 r_num = _mm256_loadu_ps (&A(k,i));
            __m256 r_div = _mm256_mul_ps (r_num, r_den);
            _mm256_storeu_ps(&L(k,i),r_div);
        }

        for (; i<n; i++) {
            L(k,i) = A(k,i) / vec(k);          
        }
    
        // Update U (stored in A)
        for (int j = k; j < n; ++j) {
            int i = k + 1;
            int cutoff =  ((n - (i + 1)) / 8) * 8;

            __m256 r_1 = _mm256_set1_ps(A(j, k));
            __m256 r_2;
            __m256 r_3; 

            for (; i < cutoff; i += 8) {
                r_2 = _mm256_loadu_ps( & A(j, i));
                r_3 = _mm256_loadu_ps( & L(k, i));

                r_2 = _mm256_fnmadd_ps(r_3, r_1, r_2);
                _mm256_storeu_ps( & A(j, i), r_2);
            }
            
            for (; i < n; i++) {
                A(j, i) = A(j, i) - L(k, i) * A(j, k);
            }
        }
    }

    A.transposeInPlace();
    L.transposeInPlace();
    //L = L.transpose();
    
    return L;
}


Eigen::VectorXf forward_sub(MatrixType L, Eigen::VectorXf b){
    
    //L.transposeInPlace();
    int n = L.cols();
    Eigen::VectorXf x = Eigen::VectorXf::Zero(n);
    //b.transposeInPlace();

    for (int i = 0; i < n; ++i){
        __m256 sum = _mm256_setzero_ps();
        int j{0};
        int unroll = (i/8)*8;
        
        for (j = 0; j < unroll; j+=8){
            __m256 r_x = _mm256_loadu_ps(&x(j));
            __m256 r_l = _mm256_loadu_ps(&L(i,j)); 
            sum = _mm256_fmadd_ps(r_x,r_l,sum); // accumulate the products in sum
        }

        // extract top and bottom half from 256
        const __m128 upper = _mm256_extractf128_ps(sum, 1); // [a,b,c,d]
        const __m128 lower = _mm256_castps256_ps128(sum); // [e,f,g,h]

        const __m128 sum_half = _mm_add_ps(upper, lower); // [a+e , b+f , c+g, d+h]
        const __m128 lo_half  = _mm_movehl_ps(sum_half, sum_half); // [c+g,d+h,-,-]


        const __m128 sum_quart = _mm_add_ps(sum_half, lo_half); //[a+e+c+g, b+f+d+h, -, -]
        const __m128 lo_quar = _mm_shuffle_ps(sum_quart, sum_quart, 0x1); // [b+f+d+h, - , - , -]

        const __m128 fsum = _mm_add_ps(sum_quart, lo_quar); // [a+b+c+d+e+f+g+h,-, -, - ] 
        float s = _mm_cvtss_f32(fsum); // the sum! 


        for(; j<i; j++){
            s += L(i,j) * x(j);
        }

        x(i) = (b(i) - s) / L(i,i);
    }
    return x;
}


Eigen::VectorXf backward_sub(MatrixType U, Eigen::VectorXf b){
   //U.transposeInPlace();
    int n = U.rows();
    Eigen::VectorXf x = Eigen::VectorXf::Zero(n);

    for (int i = n - 1; i >= 0; --i) {
        __m256 sum = _mm256_setzero_ps ();
        int j{i+1};
        int unroll = ((n - (j + 1))/8)*8;
        for (j = i + 1; j < unroll; j+=8) {
            __m256 r_x = _mm256_loadu_ps (&x(j));
            __m256 r_l = _mm256_loadu_ps (&U(i, j)); 
            sum = _mm256_fmadd_ps(r_x,r_l,sum);
        }

        // extract top and bottom half from 256
        const __m128 upper = _mm256_extractf128_ps(sum, 1); // [a,b,c,d]
        const __m128 lower = _mm256_castps256_ps128(sum); // [e,f,g,h]

        const __m128 sum_half = _mm_add_ps(upper, lower); // [a+e , b+f , c+g, d+h]
        const __m128 lo_half  = _mm_movehl_ps(sum_half, sum_half); // [c+g,d+h,-,-]

        const __m128 sum_quart = _mm_add_ps(sum_half, lo_half); //[a+e+c+g, b+f+d+h, -, -]
        const __m128 lo_quar = _mm_shuffle_ps(sum_quart, sum_quart, 0x1); // [b+f+d+h, - , - , -]

        const __m128 fsum = _mm_add_ps(sum_quart, lo_quar); // [a+b+c+d+e+f+g+h,-, -, - ] 
        float s = _mm_cvtss_f32(fsum); // the sum! 
        

        for(; j<n; j++){
            s += U(i,j) * x(j);
        }
        x(i) = (b(i) - s) / U(i,i);
    } 
  
    return x;
}



Eigen::VectorXf lu_solver(MatrixType& A, Eigen::VectorXf b ){
    
    MatrixType L = lu_decomposition(A); // returns L and modifies A
    Eigen::VectorXf y = forward_sub(L, b); // Ly = b
    Eigen::VectorXf x = backward_sub(A, y); // Ux = y
    return x;
}
