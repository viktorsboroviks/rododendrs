#include <chrono>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "rododendrs.hpp"

const size_t N_RUNS            = 100;
const std::string OUT_CSV_NAME = "sampling_f_benchmark.csv";

typedef std::function<void(
        const std::vector<double>&, std::set<size_t>&, size_t)>
        run_function_t;

void sample_in(const std::vector<double>& population,
               std::set<size_t>& samples_idx,
               size_t n)
{
    assert(n > 0);
    assert(population.size() >= n);
    assert(samples_idx.empty());

    for (size_t i = 0; i < n; i++) {
        const size_t prev_n_samples = samples_idx.size();
        do {
            const size_t sample_i =
                    rododendrs::rnd_in_range(0, population.size());
            samples_idx.insert(sample_i);
        } while (prev_n_samples == samples_idx.size());
    }

    assert(samples_idx.size() == n);
}

void sample_out(const std::vector<double>& population,
                std::set<size_t>& samples_idx,
                size_t n)
{
    assert(n > 0);
    assert(population.size() >= n);
    assert(samples_idx.empty());

    for (size_t i = 0; i < population.size(); i++) {
        samples_idx.insert(i);
    }
    assert(samples_idx.size() == population.size());

    for (size_t i = 0; i < population.size() - n; i++) {
        const size_t prev_n_samples = samples_idx.size();
        do {
            const size_t sample_i =
                    rododendrs::rnd_in_range(0, population.size());
            samples_idx.erase(sample_i);
        } while (prev_n_samples == samples_idx.size());
    }

    assert(samples_idx.size() == n);
}

struct Test {
    std::string name;
    run_function_t f;
    size_t population_size;
    double sample_part;
};

Test tests[] = {
        {"sample_in", sample_in, 1000, 1.0},
        {"sample_in", sample_in, 1000, 0.9},
        {"sample_in", sample_in, 1000, 0.5},
        {"sample_in", sample_in, 1000, 0.2},
        {"sample_in", sample_in, 1000, 0.1},
        {"sample_in", sample_in, 1000, 0.01},
        {"sample_in", sample_in, 1000, 0.001},
        {"sample_out", sample_out, 1000, 1.0},
        {"sample_out", sample_out, 1000, 0.9},
        {"sample_out", sample_out, 1000, 0.5},
        {"sample_out", sample_out, 1000, 0.2},
        {"sample_out", sample_out, 1000, 0.1},
        {"sample_out", sample_out, 1000, 0.01},
        {"sample_out", sample_out, 1000, 0.001},
        {"sample_in", sample_in, 100000, 1.0},
        {"sample_in", sample_in, 100000, 0.9},
        {"sample_in", sample_in, 100000, 0.5},
        {"sample_in", sample_in, 100000, 0.2},
        {"sample_in", sample_in, 100000, 0.1},
        {"sample_in", sample_in, 100000, 0.01},
        {"sample_in", sample_in, 100000, 0.001},
        {"sample_out", sample_out, 100000, 1.0},
        {"sample_out", sample_out, 100000, 0.9},
        {"sample_out", sample_out, 100000, 0.5},
        {"sample_out", sample_out, 100000, 0.2},
        {"sample_out", sample_out, 100000, 0.1},
        {"sample_out", sample_out, 100000, 0.01},
        {"sample_out", sample_out, 100000, 0.001},
};

int main()
{
    std::cout << "name,population_size,samples_n,avg_runtime_ns" << std::endl;
    for (auto t : tests) {
        // init
        std::vector<double> population_var;
        for (size_t j = 0; j < t.population_size; j++) {
            population_var.push_back(rododendrs::rnd01());
        }
        const std::vector<double> population = population_var;
        const size_t samples_n = (double)population.size() * t.sample_part;

        size_t total_runtime_ns = 0;
        for (size_t i = 0; i < N_RUNS; i++) {
            std::set<size_t> samples_idx;

            auto start = std::chrono::steady_clock::now();
            t.f(population, samples_idx, samples_n);
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
