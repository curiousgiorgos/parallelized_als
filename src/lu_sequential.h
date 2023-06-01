// Sequential implementation of LU decomposition

#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <regex>
#include <tuple> 
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <cmath>
#include <chrono>

// define relevant data types 
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SparseMatrixType; // declares a column-major sparse matrix type of doulbe
typedef Eigen::Triplet<float> T; // declares Eigen Triple type to hold <row,column,value>
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixType;
typedef Eigen::VectorXf Vector;

// function that decomposoes A into L and U
// returns L and stores U into A 
MatrixType lu_decomposition(MatrixType& A) {
    int n = A.rows();
    MatrixType L = MatrixType::Identity(n, n);
    
    for (int k = 0; k < n - 1; ++k) {
        // Update L
        double diagonal = A(k,k);
        for (int i = k + 1; i < n; ++i) {
            L(i,k) = A(i,k) / diagonal;
        }

        // Update U (stored in A)
        for (int j = k; j < n; ++j) {
            double mul = A(k,j);
            for (int i = k + 1; i < n; ++i) {
                A(i,j) = A(i,j) - L(i,k) * mul;
            }
        }
    }

    return L;
}

// functions that implements forward subtitution 
Vector forward_sub(MatrixType L, Vector b){
    int n = L.rows();
    Vector x = Vector::Zero(n);
    
    // solve forward subtitution 
    for (int i = 0; i < n; ++i){
        float s = 0.0;
        for (int j = 0; j < i; ++j){
            s += L(i,j) * x(j);
        }
        x(i) = (b(i) - s) / L(i,i);
    }
    return x;
}

// functions that implements backward subtitution 
Vector backward_sub(MatrixType U, Vector b){
    int n = U.rows();
    Vector x = Vector::Zero(n);

    // solve backward subtitution 
    for (int i = n - 1; i >= 0; --i) {
        float s = 0.0;
        for (int j = i + 1; j < n; ++j) {
            s += U(i,j) * x(j);
        }
        x(i) = (b(i) - s) / U(i,i);
    }

    return x;
}

Vector lu_solver(MatrixType& A, Vector b ){
    MatrixType L = lu_decomposition(A); // returns L and modifies A
    Vector y = forward_sub(L, b); // Ly = b
    Vector x = backward_sub(A, y); // Ux = y
    return x;
}


// int main() {
//      // MatrixType A = MatrixType::Random(3, 3);
//      MatrixType A(3,3);
//      A << 1, 2, -2,
//           2, 1, -5,
//           1, -4, 1;
//      std::cout << "A:\n" << A << '\n';
//      std::cout << '\n';
//      MatrixType L = lu_decomposition(A); // returns L and modifies A
//      Vector b(3);
//      b << -15, -21, 18;
//      Vector y = forward_sub(L, b); // Ly = b
//      Vector x = backward_sub(A, y); // Ux = y
//      std::cout << "x:\n" << x << '\n';

//           std::cout << "U:\n" << A << '\n';

//                std::cout << "L:\n" << L << '\n';
//      return 0;
//}

