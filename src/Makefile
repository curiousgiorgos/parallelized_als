# File       : Makefile
# Description: Compile the targets. Eigen library must be loaded on the cluster before running.
# Copyright 2023 Harvard University. All Rights Reserved.
CXX ?= g++
CXXFLAGS = -O3 -Wall -fopenmp
CXXFLAGS += -msse -msse2 -msse3 -DNDEBUG -mavx2 -mfma -march=native #-ffast-math # targets: SSE, SSE2 and SSE3

.PHONY: clean

main: als_sequential als_parallel

als_loopsched: als_parallel_dynamic16 als_parallel_dynamic32 als_parallel_dynamic64 als_parallel_dynamic128 als_parallel_guided als_parallel_static als_parallel_static2

als_simd: als_parallel_full als_parallel_nosimd als_parallel_singlefma

# MAIN 

als_sequential: als_sequential.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

als_parallel: als_parallel.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

als_eigen: als_eigen.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<


# LOOP SCHEDULING TESTS
als_parallel_dynamic16:  loop_scheduling_trials/als_parallel_dynamic16.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

als_parallel_dynamic32:  loop_scheduling_trials/als_parallel_dynamic32.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

als_parallel_dynamic64:  loop_scheduling_trials/als_parallel_dynamic64.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

als_parallel_dynamic128: loop_scheduling_trials/als_parallel_dynamic128.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

als_parallel_guided: loop_scheduling_trials/als_parallel_guided.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

als_parallel_static: loop_scheduling_trials/als_parallel_static.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

als_parallel_static2: loop_scheduling_trials/als_parallel_static2.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# SIMD TESTS
als_parallel_full: simd_trials/als_parallel_full.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

als_parallel_nosimd: simd_trials/als_parallel_nosimd.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

als_parallel_singlefma: simd_trials/als_parallel_singlefma.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# PAPI
als_papi: als_papi.cpp
	g++ -O3 -g -Wall -Wextra -Wpedantic -I$(SHARED_DATA)/local/include -L$(SHARED_DATA)/local/lib -o $@ $< -lpapi

clean:
	rm -f als_sequential als_parallel als_papi als_parallel_nosimd als_parallel_static als_parallel_dynamic16 als_parallel_dynamic128 als_parallel_guided als_parallel_dynamic32 als_parallel_dynamic64 als_parallel_static2 als_parallel_full als_parallel_nosimd als_parallel_singlefma als_eigen
