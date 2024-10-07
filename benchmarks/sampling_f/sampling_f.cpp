#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "rododendrs.hpp"

const size_t N_RUNS = 100;

typedef std::function<void(std::set<size_t>&, size_t, size_t)> run_function_t;

struct Test {
    std::string name;
    run_function_t f;
    size_t population_size;
    double sample_part;
};

Test tests[] = {
        {"sample_in", rododendrs::sample_in, 1000, 1.0},
        {"sample_in", rododendrs::sample_in, 1000, 0.9},
        {"sample_in", rododendrs::sample_in, 1000, 0.5},
        {"sample_in", rododendrs::sample_in, 1000, 0.2},
        {"sample_in", rododendrs::sample_in, 1000, 0.1},
        {"sample_in", rododendrs::sample_in, 1000, 0.01},
        {"sample_in", rododendrs::sample_in, 1000, 0.001},
        {"sample_out", rododendrs::sample_out, 1000, 1.0},
        {"sample_out", rododendrs::sample_out, 1000, 0.9},
        {"sample_out", rododendrs::sample_out, 1000, 0.5},
        {"sample_out", rododendrs::sample_out, 1000, 0.2},
        {"sample_out", rododendrs::sample_out, 1000, 0.1},
        {"sample_out", rododendrs::sample_out, 1000, 0.01},
        {"sample_out", rododendrs::sample_out, 1000, 0.001},
        {"sample_shuffle", rododendrs::sample_shuffle, 1000, 1.0},
        {"sample_shuffle", rododendrs::sample_shuffle, 1000, 0.9},
        {"sample_shuffle", rododendrs::sample_shuffle, 1000, 0.5},
        {"sample_shuffle", rododendrs::sample_shuffle, 1000, 0.2},
        {"sample_shuffle", rododendrs::sample_shuffle, 1000, 0.1},
        {"sample_shuffle", rododendrs::sample_shuffle, 1000, 0.01},
        {"sample_shuffle", rododendrs::sample_shuffle, 1000, 0.001},
        {"sample", rododendrs::sample, 1000, 1.0},
        {"sample", rododendrs::sample, 1000, 0.9},
        {"sample", rododendrs::sample, 1000, 0.5},
        {"sample", rododendrs::sample, 1000, 0.2},
        {"sample", rododendrs::sample, 1000, 0.1},
        {"sample", rododendrs::sample, 1000, 0.01},
        {"sample", rododendrs::sample, 1000, 0.001},
        {"sample_in", rododendrs::sample_in, 100000, 1.0},
        {"sample_in", rododendrs::sample_in, 100000, 0.9},
        {"sample_in", rododendrs::sample_in, 100000, 0.5},
        {"sample_in", rododendrs::sample_in, 100000, 0.2},
        {"sample_in", rododendrs::sample_in, 100000, 0.1},
        {"sample_in", rododendrs::sample_in, 100000, 0.01},
        {"sample_in", rododendrs::sample_in, 100000, 0.001},
        {"sample_out", rododendrs::sample_out, 100000, 1.0},
        {"sample_out", rododendrs::sample_out, 100000, 0.9},
        {"sample_out", rododendrs::sample_out, 100000, 0.5},
        {"sample_out", rododendrs::sample_out, 100000, 0.2},
        {"sample_out", rododendrs::sample_out, 100000, 0.1},
        {"sample_out", rododendrs::sample_out, 100000, 0.01},
        {"sample_out", rododendrs::sample_out, 100000, 0.001},
        {"sample_shuffle", rododendrs::sample_shuffle, 100000, 1.0},
        {"sample_shuffle", rododendrs::sample_shuffle, 100000, 0.9},
        {"sample_shuffle", rododendrs::sample_shuffle, 100000, 0.5},
        {"sample_shuffle", rododendrs::sample_shuffle, 100000, 0.2},
        {"sample_shuffle", rododendrs::sample_shuffle, 100000, 0.1},
        {"sample_shuffle", rododendrs::sample_shuffle, 100000, 0.01},
        {"sample_shuffle", rododendrs::sample_shuffle, 100000, 0.001},
        {"sample", rododendrs::sample, 100000, 1.0},
        {"sample", rododendrs::sample, 100000, 0.9},
        {"sample", rododendrs::sample, 100000, 0.5},
        {"sample", rododendrs::sample, 100000, 0.2},
        {"sample", rododendrs::sample, 100000, 0.1},
        {"sample", rododendrs::sample, 100000, 0.01},
        {"sample", rododendrs::sample, 100000, 0.001},
};

int main()
{
    std::cout << "name,population_size,samples_n,avg_runtime_ns" << std::endl;
    for (auto t : tests) {
        const size_t samples_n = (double)t.population_size * t.sample_part;

        size_t total_runtime_ns = 0;
        for (size_t i = 0; i < N_RUNS; i++) {
            std::set<size_t> samples_idx;

            auto start = std::chrono::steady_clock::now();
            t.f(samples_idx, t.population_size, samples_n);
            auto finish = std::chrono::steady_clock::now();

            total_runtime_ns +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                            finish - start)
                            .count();
        }
        const double avg_runtime_ns = total_runtime_ns / (double)N_RUNS;
        std::cout << t.name << ",";
        std::cout << t.population_size << ",";
        std::cout << samples_n << ",";
        std::cout << avg_runtime_ns << std::endl;
    }
    return 0;
}
