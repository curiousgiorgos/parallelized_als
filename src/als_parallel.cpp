// Parallel implementation of Alternating Least Squares 
// Uses vectorized LU decomposition defined in lu_parallel to solve the system of linear equations 

#include "utils.h"
#include "lu_parallel.h"

// Driver Code
int main(int argc, char * argv[]) {

    // Define A and A transpose
    SparseMatrixType * data = new SparseMatrixType(ROWS, COLUMNS);
    SparseMatrixTypeCol * data_trans = new SparseMatrixTypeCol(ROWS, COLUMNS);

    // triplet vectors <user,movie,rating>
    std::vector<T> * coeff = new std::vector<T>;

    // load matrices 
    loadData(data, coeff, data_trans);

    // rank used 
    int rank = 256;

    // Define W and H transposed 
    MatrixType W = MatrixType::Random(ROWS, rank);
    MatrixType H = MatrixType::Random(rank, COLUMNS);

    // lambda * identity matrix, used for regularization
    float lambda = 0.01;
    MatrixType lamI = lambda * MatrixType::Identity(rank, rank);

    // number of iterations to run the algorithm 
    int max_iter = 10;

    // rmse for performance and early stoppage 
    float prev_rmse = std::numeric_limits<float> ::infinity();
    float rmse;

    // timing function 
    auto t0 = std::chrono::steady_clock::now(); 

    for (int iter = 0; iter < max_iter; ++iter) 
    {   
        // Update H^T  
        #pragma omp parallel for schedule(dynamic,2) proc_bind(close)
        // iterate over columns the columns to A
        for (int i = 0; i < COLUMNS; i++) 
        { 
            // define helper matrices 
            MatrixType lhs;
            MatrixType rhs;
            MatrixType W_row(ROWS, rank);
            MatrixType data_vec(1, ROWS);
            int obs_count = 0 ;
            // create W sub and data sub iterate over the selected column 
            // as data_trans are stored COLUMN-Major this iterates over sequential data in memory
            for (SparseMatrixTypeCol::InnerIterator it( *data_trans, i); it; ++it)
            {
                // extract W entry for observed value 
                W_row.row(obs_count) = W.row(it.row());
                // add obverved rating in data vec
                data_vec(0, obs_count) = it.value();
                obs_count += 1; 
            }
            // resize matrices to observed dimensions 
            MatrixType resized  = W_row.block(0, 0, obs_count, rank);
            MatrixType resized_vec = data_vec.block(0, 0, 1, obs_count);

            // define right hand side and left hand side of linear equation
            rhs = resized.transpose() * resized_vec.transpose();
            lhs = resized.transpose() * resized + lamI;

            // https://stackoverflow.com/questions/43403273/alternating-least-squares-derivative
            //H.col(i) = lhs.partialPivLu().solve(rhs);

            // use lu_solver to solve the linear equation  
            H.col(i) = lu_solver(lhs, rhs);
        }

        // Update W 
        #pragma omp parallel for schedule(dynamic,2)  proc_bind(close)
        // iterate over the rows of A
        for (int i = 0; i < ROWS; i++) 
        {
            // define helper matrices 
            MatrixType lhs;
            MatrixType rhs;
            MatrixType  H_col(rank, COLUMNS);
            MatrixType data_vec(COLUMNS, 1);
            int obs_count = 0;

            // create H sub and data sub by iterating over selected row 
            // as data are stored ROW-Major this iterates over sequential data in memory 
            for (SparseMatrixType::InnerIterator it( *data, i); it; ++it) 
            {
                H_col.col(obs_count) = H.col(it.col());
                data_vec(obs_count, 0) = it.value();
                obs_count += 1;
            }

            MatrixType resized  = H_col.block(0, 0, rank, obs_count);
            MatrixType resized_vec = data_vec.block(0, 0, obs_count, 1);

            rhs = resized * resized_vec;
            lhs = resized * resized.transpose() + lamI;
            // https://stackoverflow.com/questions/43403273/alternating-least-squares-derivative
            //W.row(i) = lhs.partialPivLu().solve(rhs).transpose();
            W.row(i) = lu_solver(lhs,rhs).transpose();
        }

        // calculate rmse 
        rmse = 0;
        #pragma omp parallel for reduction(+:rmse)
        // iterate over the observed values in A and caclulate the prediction based on W and H transposed
        for (int l = 0; l < data -> outerSize(); ++l) {
            for (SparseMatrixType::InnerIterator it( * data, l); it; ++it)
            {
                int curr_row = it.row();
                int curr_col = it.col();
                float res_val = (W.row(curr_row).array() * H.col(curr_col).array().transpose()).sum();
                float diff = res_val - it.value();
                rmse += pow(diff, 2);
            }
        }
        rmse = sqrt(rmse);

        //std::cout << "Iteration: " << iter + 1 << " RMSE: " << rmse << "\n";

        // early stoppage if RMSE improvement < 0.01% 
        if (abs(rmse - prev_rmse) / prev_rmse < 0.001) {break;}
        // update rmse 
        prev_rmse = rmse;
    }
    // calculate elapsed time 
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<float> diff = t1 - t0;

    printf("Iterations completed in %.2lfs\n", diff.count());

    return 0;
}