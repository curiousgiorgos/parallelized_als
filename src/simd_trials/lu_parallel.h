// Parallel implementation of LU decomposition
// Using FMAs and ILP
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

void lu_fnmadd(int j, int k, int n, MatrixType& A, MatrixType& L); 
void lu_fmadd_backward(int i, int n, MatrixType& U, Eigen::VectorXf& b, Eigen::VectorXf& x);
void lu_fmadd_forward(int i, MatrixType& L, Eigen::VectorXf& b, Eigen::VectorXf& x);

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
            lu_fnmadd(j, k, n, A, L);
        }
    }

    A.transposeInPlace();
    L.transposeInPlace();
    
    return L;
}


Eigen::VectorXf forward_sub(MatrixType L, Eigen::VectorXf b){
    
    int n = L.cols();
    Eigen::VectorXf x = Eigen::VectorXf::Zero(n);
    for (int i = 0; i < n; ++i){
        lu_fmadd_forward(i, L, b, x);

    }

    return x;
}


Eigen::VectorXf backward_sub(MatrixType U, Eigen::VectorXf b) {

    int n = U.rows();
    Eigen::VectorXf x = Eigen::VectorXf::Zero(n);
    for (int i = n - 1; i >= 0; --i) {
        lu_fmadd_backward(i, n, U, b, x);
    } 
  
    return x;
}



Eigen::VectorXf lu_solver(MatrixType& A, Eigen::VectorXf b ) {
    
    MatrixType L = lu_decomposition(A); // returns L and modifies A
    Eigen::VectorXf y = forward_sub(L, b); // Ly = b
    Eigen::VectorXf x = backward_sub(A, y); // Ux = y
    return x;
}


void lu_fnmadd(int j, int k, int n, MatrixType & A, MatrixType& L) {

    int i = k + 1;
    int cutoff = ((n - (i + 1)) / 40) * 40;

    __m256 r_1 = _mm256_set1_ps(A(j, k));
    __m256 r_2;
    __m256 r_3;
    __m256 r_4;
    __m256 r_5;
    __m256 r_6;
    __m256 r_7;
    __m256 r_8;
    __m256 r_9;
    __m256 r_10;
    __m256 r_11;
    
    for (; i < cutoff; i += 40) {
        r_2 = _mm256_loadu_ps( & A(j, i));
        r_3 = _mm256_loadu_ps( & L(k, i));
        r_4 = _mm256_loadu_ps( & A(j, i + 8));
        r_5 = _mm256_loadu_ps( & L(k, i + 8));
        r_6 = _mm256_loadu_ps( & A(j, i + 16));
        r_7 = _mm256_loadu_ps( & L(k, i + 16));
        r_8 = _mm256_loadu_ps( & A(j, i + 24));
        r_9 = _mm256_loadu_ps( & L(k, i + 24));
        r_10 = _mm256_loadu_ps( & A(j, i + 32));
        r_11 = _mm256_loadu_ps( & L(k, i + 32));

        r_2 = _mm256_fnmadd_ps(r_3, r_1, r_2);
        r_4 = _mm256_fnmadd_ps(r_5, r_1, r_4);
        r_6 = _mm256_fnmadd_ps(r_7, r_1, r_6);
        r_8 = _mm256_fnmadd_ps(r_9, r_1, r_8);
        r_10 = _mm256_fnmadd_ps(r_11, r_1, r_10);

        _mm256_storeu_ps( & A(j, i), r_2);
        _mm256_storeu_ps( & A(j, i + 8), r_4);
        _mm256_storeu_ps( & A(j, i + 16), r_6);
        _mm256_storeu_ps( & A(j, i + 24), r_8);
        _mm256_storeu_ps( & A(j, i + 32), r_10);
    }

    for (; i < n; i++) {
        A(j, i) = A(j, i) - L(k, i) * A(j, k);
    }

}

void lu_fmadd_backward(int i, int n, MatrixType& U, Eigen::VectorXf& b, Eigen::VectorXf& x) {

        int j = i+1;
        int unroll = ((n - (j + 1)) / 40) * 40;
    
        __m256 r_sum =  _mm256_setzero_ps ();
        __m256 r_2;
        __m256 r_3;
        __m256 r_4;
        __m256 r_5;
        __m256 r_6;
        __m256 r_7;
        __m256 r_8;
        __m256 r_9;
        __m256 r_10;
        __m256 r_11;


        for (j = i + 1; j < unroll; j+=40) {
             
            __m256 r_2 = _mm256_loadu_ps (&x(j));
            __m256 r_3 = _mm256_loadu_ps (&U(i, j)); 

            __m256 r_4 = _mm256_loadu_ps (&x(j + 8));
            __m256 r_5 = _mm256_loadu_ps (&U(i, j + 8)); 

            __m256 r_6 = _mm256_loadu_ps (&x(j + 16));
            __m256 r_7 = _mm256_loadu_ps (&U(i, j + 16)); 

            __m256 r_8 = _mm256_loadu_ps (&x(j + 24));
            __m256 r_9 = _mm256_loadu_ps (&U(i, j + 24));      

            __m256 r_10 = _mm256_loadu_ps (&x(j + 32));
            __m256 r_11 = _mm256_loadu_ps (&U(i, j + 32));                         

            r_2 =  _mm256_mul_ps(r_2 , r_3);
            r_4 =  _mm256_mul_ps(r_4 , r_5);
            r_6 =  _mm256_mul_ps(r_6 , r_7);
            r_8 =  _mm256_mul_ps(r_8 , r_9);
            r_10 =  _mm256_mul_ps(r_10 , r_11);
            
            r_2 =  _mm256_add_ps(r_2 , r_4);
            r_6 =  _mm256_add_ps(r_6 , r_8);

            r_2 = _mm256_add_ps(r_2 , r_10);

            r_2 = _mm256_add_ps(r_2 , r_6);

            r_sum = _mm256_add_ps(r_sum, r_2);
        }

        // extract top and bottom half from 256
        const __m128 upper = _mm256_extractf128_ps(r_sum, 1); // [a,b,c,d]
        const __m128 lower = _mm256_castps256_ps128(r_sum); // [e,f,g,h]

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


void lu_fmadd_forward(int i, MatrixType& L, Eigen::VectorXf& b, Eigen::VectorXf& x) {

        int j = 0;
        int unroll = (i / 40) * 40;
    
        __m256 r_sum =  _mm256_setzero_ps ();
        __m256 r_2;
        __m256 r_3;
        __m256 r_4;
        __m256 r_5;
        __m256 r_6;
        __m256 r_7;
        __m256 r_8;
        __m256 r_9;
        __m256 r_10;
        __m256 r_11;


        for (j = 0; j < unroll; j += 40) {
             
            __m256 r_2 = _mm256_loadu_ps (&x(j));
            __m256 r_3 = _mm256_loadu_ps (&L(i, j)); 

            __m256 r_4 = _mm256_loadu_ps (&x(j + 8));
            __m256 r_5 = _mm256_loadu_ps (&L(i, j + 8)); 

            __m256 r_6 = _mm256_loadu_ps (&x(j + 16));
            __m256 r_7 = _mm256_loadu_ps (&L(i, j + 16)); 

            __m256 r_8 = _mm256_loadu_ps (&x(j + 24));
            __m256 r_9 = _mm256_loadu_ps (&L(i, j + 24));      

            __m256 r_10 = _mm256_loadu_ps (&x(j + 32));
            __m256 r_11 = _mm256_loadu_ps (&L(i, j + 32));                         

            r_2 =  _mm256_mul_ps(r_2 , r_3);
            r_4 =  _mm256_mul_ps(r_4 , r_5);
            r_6 =  _mm256_mul_ps(r_6 , r_7);
            r_8 =  _mm256_mul_ps(r_8 , r_9);
            r_10 =  _mm256_mul_ps(r_10 , r_11);
            
            r_2 =  _mm256_add_ps(r_2 , r_4);
            r_6 =  _mm256_add_ps(r_6 , r_8);

            r_2 = _mm256_add_ps(r_2 , r_10);

            r_2 = _mm256_add_ps(r_2 , r_6);

            r_sum = _mm256_add_ps(r_sum, r_2);
        }

        // extract top and bottom half from 256
        const __m128 upper = _mm256_extractf128_ps(r_sum, 1); // [a,b,c,d]
        const __m128 lower = _mm256_castps256_ps128(r_sum); // [e,f,g,h]

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



/*
int main() {
    // MatrixType A = MatrixType::Random(3, 3);
    MatrixType A(4,4);
    A << 1, 2, -1, 1,
        -1, 1, 2, -1,
        2, -1, 2, 2,
        1, 1, -1, 2;

    std::cout << "A:\n" << A << '\n';
    std::cout << '\n';

    MatrixType L = lu_decomposition(A); 
    //returns L and modifies A

    Eigen::VectorXf  b(3);

    b << 6, 3, 14, 8;

    Eigen::VectorXf  y = forward_sub(L, b); // Ly = b
    Eigen::VectorXf  x = backward_sub(A, y); // Ux = y
    std::cout << "x:\n" << x << '\n';
    std::cout << '\n';
    std::cout << "U:\n" << A << '\n';
    std::cout << '\n';
    std::cout << "L:\n" << L << '\n';

    // sol 1, 2, 3, 4
    return 0;
}
*/