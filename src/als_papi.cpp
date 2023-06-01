// Sequential implementation of Alternating Least Squares 

#include <iostream>
#include<limits>
#include "utils.h"
#include "papi.h"
#include <cstdlib>
#include <numeric>

#define L1_SIZE_KB 32
#define L2_SIZE_KB 256
#define L3_SIZE_KB 40960

// Driver Code
int main(int argc, char * argv[]) {

    // data Matrix
    SparseMatrixType * data = new SparseMatrixType(ROWS, COLUMNS);
    SparseMatrixType * data_trans = new SparseMatrixType(COLUMNS, ROWS);

    // list of non-zeros coeff
    std::vector<T> * coeff = new std::vector<T> ;
    // list of non-zeros coeff, transposed
    std::vector<T> * coeff_trans = new std::vector<T> ;

    // create matrices
    loadData(data, coeff, data_trans, coeff_trans);

    std::cout << "Completed data load \n";

    int rank;
    rank = 20;
    int n = rank;

    MatrixType W = MatrixType::Random(ROWS, rank);
    MatrixType H = MatrixType::Random(rank, COLUMNS);

    // lambda * identity matrix, used for regularization
    double lambda = 0.01;
    MatrixType lamI;
    lamI = lambda * MatrixType::Identity(rank, rank);

    int max_iter = 200;
    double prev_rmse = std::numeric_limits<double> ::infinity();
    double rmse;

    // supporing variables 
    int curr_row; 
    int curr_col; 
    int data_row; 
    int data_col; 

    MatrixType lhs;
    MatrixType rhs;

    // Initialize PAPI
    int event_set = PAPI_NULL;
    int events[4] = {PAPI_TOT_CYC,
                     PAPI_TOT_INS,
                     PAPI_LST_INS,
                     PAPI_L1_DCM};
    long long int counters[4];
    PAPI_library_init(PAPI_VER_CURRENT);
    PAPI_create_eventset(&event_set);
    PAPI_add_events(event_set, events, 4);

    // One row of H
    int i = 0;

    MatrixType Wcol(0, rank);
    MatrixType data_vec(1, 0);

    for (SparseMatrixType::InnerIterator it( * data, i); it; ++it)
    {
        curr_row = Wcol.rows();
        Wcol.conservativeResize(curr_row + 1, rank);
        Wcol.row(curr_row) = W.row(it.row());

        data_col = data_vec.cols();
        data_vec.conservativeResize(1, data_col + 1);
        data_vec(0, data_col) = it.value();
    }

    lhs = Wcol.transpose() * data_vec.transpose();
    rhs = Wcol.transpose() * Wcol + lamI;

    // start PAPI measurement
    PAPI_start(event_set);

    // assuming no overhead to call this timer (will pollute PAPI_TOT_CYC and
    // PAPI_TOT_INS slightly, neglected here)
    const long long int t0 = PAPI_get_real_nsec();
    
    // https://stackoverflow.com/questions/43403273/alternating-least-squares-derivative
    H.col(i) = rhs.partialPivLu().solve(lhs);

    // assuming no overhead to call this timer (will pollute PAPI_TOT_CYC and
    // PAPI_TOT_INS slightly, neglected here)
    const long long int t1 = PAPI_get_real_nsec();

    // stop PAPI and get counter values
    PAPI_stop(event_set, counters);

    // clang-format off
    const long long total_cycles = counters[0];       // cpu cycles
    const long long total_instructions = counters[1]; // any
    const long long total_load_stores = counters[2];  // number of such instructions
    const long long total_l1d_misses = counters[3];   // number of access request to cache line
    // clang-format on

    const size_t flops = (n * n * n - n)/3 + 2 * n * n - n;
    const size_t mem_ops = n * n + 2 * n;
    const double twall = (static_cast<double>(t1) - t0) * 1.0e-9; // seconds
    const double IPC = static_cast<double>(total_instructions) / total_cycles;
    const double OI =
        static_cast<double>(flops) / (total_load_stores * sizeof(double));
    const double OI_theory =
        static_cast<double>(flops) / (mem_ops * sizeof(double));
    const double float_perf = flops / twall * 1.0e-9; // Gflop/s
    // const double sum = std::accumulate(y.begin(), y.end(), 0.0);

    std::cout << "Total cycles:                 " << total_cycles << '\n';
    std::cout << "Total instructions:           " << total_instructions << '\n';
    std::cout << "Instructions per cycle (IPC): " << IPC << '\n';
    std::cout << "L1 cache size:                " << L1_SIZE_KB << " KB\n";
    std::cout << "L2 cache size:                " << L2_SIZE_KB << " KB\n";
    std::cout << "L3 cache size:                " << L3_SIZE_KB << " KB\n";
    std::cout << "Total problem size:           " << 2 * n * sizeof(double) / 1024
            << " KB\n";
    std::cout << "Total L1 data misses:         " << total_l1d_misses << '\n';
    std::cout << "Total load/store:             " << total_load_stores
            << " (expected: " << mem_ops << ")\n";
    std::cout << "Operational intensity:        " << std::scientific << OI
            << " (expected: " << OI_theory << ")\n";
    std::cout << "Performance [Gflop/s]:        " << float_perf << '\n';
    std::cout << "Wall-time   [micro-seconds]:  " << twall * 1.0e6 << '\n';
    
    return 0;
}