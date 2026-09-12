CXX = g++
CXXFLAGS = -std=c++20 -O3 -Wall -Wextra -pthread -Iinclude

.PHONY: all tests demo hft_bench test bench run clean

all: bin_tests bin_demo bin_hft_bench

bin_tests: tests/test_all_tracks.cpp
	$(CXX) $(CXXFLAGS) tests/test_all_tracks.cpp -o bin_tests

bin_demo: examples/main_master_demo.cpp
	$(CXX) $(CXXFLAGS) examples/main_master_demo.cpp -o bin_demo

bin_hft_bench: tests/test_hft_ultralow_benchmark.cpp
	$(CXX) $(CXXFLAGS) tests/test_hft_ultralow_benchmark.cpp -o bin_hft_bench

test: bin_tests
	./bin_tests

bench: bin_hft_bench
	./bin_hft_bench

run: bin_demo
	./bin_demo

clean:
	rm -f bin_tests bin_demo bin_hft_bench
