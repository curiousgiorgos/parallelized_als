//  Utility header 
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <regex>
#include <tuple> 
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <chrono>


typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SparseMatrixType; // declares a column-major sparse matrix type of doulbe
typedef Eigen::Triplet<float> T; // declares Eigen Triple type to hold <row,column,value>
//typedef Eigen::MatrixXf MatrixType; // declares a Eigen MatrixXf matrix
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixType;


typedef Eigen::SparseMatrix<float> SparseMatrixTypeCol; // declares a column-major sparse matrix type of doulbe

// 3000 - 6040
// 1838 - 3952

#define ROWS 6040
#define COLUMNS 3952

void loadData(SparseMatrixType * data, std::vector<T> * coefficients, SparseMatrixTypeCol * data_trans, std::vector<T> * coefficients_trans) {

    std::ifstream inFile;

    const std::string movielens1m = "../data/movielens_1M/ratings.dat";
    const std::string toydata = "../data/toydata.dat";

    const std::string path = movielens1m;

    inFile.open(path);

    if (!inFile.is_open()) 
    {
        std::cout << "Error opening data file\n";
        exit(EXIT_FAILURE);
    }

    std::string line;
    std::regex e("[0-9][0-9]*");
    std::vector <std::string> matches(3);

    float user;
    float movie;
    float rating;
    int count;

    while (getline(inFile, line)) 
    {
        auto words_begin = std::sregex_iterator(line.begin(), line.end(), e);
        auto words_end = std::sregex_iterator();

        count = 0;
        for (std::sregex_iterator i = words_begin; i != words_end; ++i)
         {
            std::smatch match = * i;
            std::string match_str = match.str();
            matches[count] = match_str;
            count += 1;
            if (count == 3) 
            {
                break;
            }
        }

        user = std::stoi(matches[0]) - 1;
        movie = std::stoi(matches[1]) - 1;
        rating = std::stof(matches[2]);

        coefficients -> push_back(T(user, movie, rating));
        //coefficients_trans -> push_back(T(movie, user, rating));
    }

    data -> setFromTriplets(coefficients -> begin(), coefficients -> end());

    data_trans -> setFromTriplets(coefficients -> begin(), coefficients -> end());

    inFile.close();
}

// helper functions that prints the predicted value of observed entries 
void printMatrixDiff(SparseMatrixType * data, MatrixType * W,  MatrixType * H) {

     for (int l = 0; l < data -> outerSize(); ++l) 
     {
         for (SparseMatrixType::InnerIterator it( * data, l); it; ++it) 
         {
            int row = it.row();
            int col = it.col();
            float res_val = (W -> row(row).array() * H -> col(col).array().transpose()).sum();
            std::cout << "Observed " << it.value() << " Predicted " <<  std::round(res_val * 10.0) / 10.0 << " Predicted 2 " << "\n";
         }
     }
 }

// helper functions that prints the predicted matrix 
void printMatrixPredictions(MatrixType* W, MatrixType* H) {

    MatrixType preds = *W * *H;
    for (int r = 0; r < preds.innerSize(); ++r) 
    {
        for (int c = 0; c < preds.outerSize(); ++c) 
        {
            std::cout << "User: " << r << " Movie: " << c << " Rating: " << std::round(preds.row(r)(c) * 10.0) / 10.0 << "\n";
        }
    }
}

// helper functions that prints matrices
void printMatrix(MatrixType * W) {
    for (int r = 0; r < W->innerSize(); ++r) 
    {
        for (int c = 0; c < W->outerSize(); ++c) 
        {
            std::cout << W->row(r)(c) << "\n";
        }
    }
}
