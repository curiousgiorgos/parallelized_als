// Sequential implementation of Alternating Least Squares 
// LOOP SCHEDULING GUIDED CHUNKSIZE 16

#include "utils.h"
#include "lu_sequential.h"

// Driver Code
int main(int argc, char * argv[]) {


    // data Matrix
    SparseMatrixType * data = new SparseMatrixType(ROWS, COLUMNS);
    SparseMatrixTypeCol * data_trans = new SparseMatrixTypeCol(ROWS, COLUMNS);

    // list of non-zeros coeff
    std::vector<T> * coeff = new std::vector<T> ;
    // list of non-zeros coeff, transposed
    std::vector<T> * coeff_trans = new std::vector<T> ;

    // create matrices
    loadData(data, coeff, data_trans, coeff_trans);

    //std::cout << "Completed data load \n";

    int rank;
    rank = 256;

    //int nthreads = omp_get_num_threads();
    
    MatrixType W = MatrixType::Random(ROWS, rank);
    MatrixType H = MatrixType::Random(rank, COLUMNS);

    // lambda * identity matrix, used for regularization
    float lambda = 0.01;
    MatrixType lamI;
    lamI = lambda * MatrixType::Identity(rank, rank);

    int max_iter = 10;
    float prev_rmse = std::numeric_limits<float> ::infinity();
    float rmse;


    auto t0 = std::chrono::steady_clock::now(); 

    for (int iter = 0; iter < max_iter; ++iter) 
    {   
        // Update H^T, users  
        #pragma omp parallel for schedule(guided,16) proc_bind(close)
        for (int i = 0; i < COLUMNS; i++) 
        { 
            
            MatrixType lhs;
            MatrixType rhs;
            MatrixType W_row(ROWS, rank);
            MatrixType data_vec(1, ROWS);
            int obs_count = 0 ;
            for (SparseMatrixTypeCol::InnerIterator it( *data_trans, i); it; ++it)
            {
                
                W_row.row(obs_count) = W.row(it.row());
                data_vec(0, obs_count) = it.value();
                obs_count += 1; 
            }
            
            MatrixType resized  = W_row.block(0, 0, obs_count, rank);
            MatrixType resized_vec = data_vec.block(0, 0, 1, obs_count);

            rhs = resized.transpose() * resized_vec.transpose();
            lhs = resized.transpose() * resized + lamI;

            // https://stackoverflow.com/questions/43403273/alternating-least-squares-derivative
            //H.col(i) = lhs.partialPivLu().solve(rhs);
            H.col(i) = lu_solver(lhs, rhs);
        }

        // Update W 
        #pragma omp parallel for schedule(guided,16)  proc_bind(close)
        for (int i = 0; i < ROWS; i++) 
        {
            MatrixType lhs;
            MatrixType rhs;
            MatrixType  H_col(rank, COLUMNS);
            MatrixType data_vec(COLUMNS, 1);
            int obs_count = 0;

            // create H sub and data sub
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
        if (abs(rmse - prev_rmse) / prev_rmse < 0.001) {break;}

        prev_rmse = rmse;
    }

    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<float> diff = t1 - t0;
    printf("Guided 16: Iterations completed in %.2lfs\n", diff.count());
        

    // print predicted values 
    //printMatrixDiff(data, &W, &H);

    // prints predicted matrix
    //printMatrixPredictions(&W, &H);

    return 0;
}