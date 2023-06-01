#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np




def plot_cum():

    sequential_time = 2277.34 #in seconds 

    parallel_times = [262.75,136.97,70.89,38.77,20.43,10.43]
    n_threads = [1,2,4,8,16,32]

    plt.figure(figsize=(8,6))

    plt.plot(n_threads, [sequential_time/p for p in parallel_times], "o-", lw=2, color='darkolivegreen', label='Observed Speedup')
    plt.plot(n_threads,[8*n for n in n_threads],color='k',lw=1.5,ls='--',label='Ideal')


    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.legend(loc='upper left')
    plt.title("Cumulative Observed Speedup")

    plt.savefig("cumulative_speedup.png", dpi=300)



def plot_parallel():

    sequential_time = 2277.43 #in seconds 
    parallel_times = [2277.43,1142.79,576.41,313.28,163.04,82.38]
    n_threads = [1,2,4,8,16,32]

    plt.figure(figsize=(8,6))

    plt.plot(n_threads, [sequential_time/p for p in parallel_times], "o-", lw=2, color='darkolivegreen', label='Observed Speedup')
    plt.plot(n_threads,[n for n in n_threads],color='k',lw=1.5,ls='--',label='Ideal')


    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.legend(loc='upper left')
    plt.title("Parallel Observed Speedup")

    plt.savefig("parallel_speedup.png", dpi=300)


def plot_schdules():

    sequential_time = 524.07 #in seconds 
    dynamic_16 = [550, 257.65,130.03,73.09,39.24,21.51]
    static = [545, 283.17,149.09,79.76,41.28,21.54]
    dynamic_2 = [550.87,247.84,136.44,73.78,38.36,19.68]
    dynamic_128 = [548 ,273.03,143.26, 81.98, 47.56,35.45]
    guided_16 = [551, 277.44, 142.63, 77.68,41.03, 21.31]

    n_threads = [1,2,4,8,16,32]

    plt.figure(figsize=(8,6))

    plt.plot(n_threads, [sequential_time/p for p in dynamic_16], "o-", lw=2, label='(dynamic, 16)')
    plt.plot(n_threads, [sequential_time/p for p in static], "o-", lw=2, label='(static)')
    plt.plot(n_threads, [sequential_time/p for p in dynamic_2], "o-", lw=2, label='(dynamic, 2)')
    plt.plot(n_threads, [sequential_time/p for p in dynamic_128], "o-", lw=2, label='(dynamic, 128)')
    plt.plot(n_threads, [sequential_time/p for p in guided_16], "o-", lw=2, label='(guided, 16)')

    plt.plot(n_threads,[n for n in n_threads],color='k',lw=1.5,ls='--',label='Ideal')


    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.legend(loc='upper left')
    plt.title("Parallel Observed Speedup")

    plt.savefig("schedule_speedup.png", dpi=300)

def plot_simd_ilp_speedup():

    openmp_times = [2277.43,1142.79,576.41,313.28,163.04,82.38]
    simd_ilp_times = [262.75,136.97,70.89,38.77,20.43,10.43]
    n_threads = [1,2,4,8,16,32]

    plt.figure(figsize=(8,6))

    plt.plot(n_threads, [openmp_times[p]/simd_ilp_times[p] for p in range(6)], "o-", lw=2, color='darkolivegreen', label='SIMD+ILP Contribution')
    plt.axhline(y=8,color='k',lw=1.5,ls='--',label='8x')
    plt.ylim(0,10)

    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.legend()
    plt.title("SIMD+ILP Contribution to Speedup")

    plt.savefig("simd_ilp_speedup.png", dpi=300)

if __name__=="__main__":
    plot_cum()
    plot_parallel()
    plot_simd_ilp_speedup()