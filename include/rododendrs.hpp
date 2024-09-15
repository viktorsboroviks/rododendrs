#pragma once

#include <cassert>
#include <cmath>
#include <deque>
#include <mutex>
#include <random>
#include <set>
#include <vector>

namespace rododendrs {

class Random {
    // thread-safe implementation of rnd01() borrowed from
    // https://github.com/Arash-codedev/openGA/blob/master/README.md
    // assuming those people knew what they were doing
private:
    std::mutex mtx_rand;
    std::mt19937_64 rng;
    std::uniform_real_distribution<double> unif_dist;

public:
    Random()
    {
        // initialize the random number generator with time-dependent seed
        uint64_t timeSeed = std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count();
        std::seed_seq ss{uint32_t(timeSeed & 0xffffffff),
                         uint32_t(timeSeed >> 32)};
        rng.seed(ss);
        std::uniform_real_distribution<double> unif(0, 1);
    }

    double rnd01()
    {
        // prevent data race between threads
        std::lock_guard<std::mutex> lock(mtx_rand);
        return unif_dist(rng);
    }
};

// global variable that holds the random state
inline Random _g_random{};

double rnd01()
{
    return _g_random.rnd01();
}

double rnd_in_range(double min, double max)
{
    if (min == max) {
        return min;
    }
    assert(min < max);
    const double retval = (_g_random.rnd01() * (max - min)) + min;
    assert(retval >= min);
    assert(retval <= max);
    return retval;
}

// relative root mean squared error
template <template <typename...> typename P, template <typename...> typename C>
double rrmse(const P<double>& predicted, const C<double>& correct)
{
    assert(!predicted.empty());
    assert(predicted.size() == correct.size());
    double sum_top = 0;
    double sum_bot = 0;
    for (size_t i = 0; i < predicted.size(); i++) {
        sum_top += std::pow(predicted[i] - correct[i], 2);
        sum_bot += std::pow(correct[i], 2);
    }
    return std::sqrt((sum_top / (double)predicted.size()) / sum_bot);
}

template <template <typename...> typename T>
double mean(const T<double>& data)
{
    assert(!data.empty());
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

template <template <typename...> typename T>
double variance(const T<double>& data)
{
    assert(!data.empty());
    const double x_m = mean<T>(data);
    auto lambda      = [x_m](double accum, double x) {
        return accum + std::pow(x - x_m, 2.0);
    };
    return std::accumulate(data.begin(), data.end(), 0.0, lambda) /
           data.size();
}

template <template <typename...> typename T>
double sd(const T<double>& data)
{
    assert(!data.empty());
    return std::sqrt(variance<T>(data));
}

// returns pdf value of x in normal distribution
template <typename T>
T normal_distribution(T mean, T sd, T x)
{
    static const T pi = std::acos(T(-1));
    static const T m  = T(1) / std::sqrt(T(2) * pi);
    return m *
           std::exp(-std::pow(x - mean, T(2)) / (T(2) * std::pow(sd, T(2)))) /
           sd;
}

// Arturs Zoldners pdf overlap function
template <typename T>
constexpr T az_pdf_overlap(
        T mean1, T sd1, T n_samples1, T mean2, T sd2, T n_samples2)
{
    const T S   = 4.;  // integrate over range of 5 sigmas
    const int P = 2 * 1024;

    const T ww = n_samples1 + n_samples2;
    const T v1 = n_samples1 / ww;
    const T v2 = n_samples2 / ww;

    const T x_min = std::min(mean1 - S * sd1, mean2 - S * sd2);
    const T x_max = std::max(mean1 + S * sd1, mean2 + S * sd2);
    const T dx    = x_max - x_min;

    T ret = 0;

    // #pragma omp simd reductions(+: ret)
    for (int i = 0; i <= P; ++i) {
        const T x  = (x_min * (P - i) + x_max * (i)) / P;
        const T p1 = normal_distribution(mean1, sd1, x);
        const T p2 = normal_distribution(mean2, sd2, x);
        ret += (p1 * p2) / (v1 * p1 + v2 * p2);
    }

    return ret * dx / T(P);
}

template <typename TContainer>
void sample_in(const TContainer& population,
               std::set<size_t>& samples_idx,
               size_t n)
{
    assert(n > 0);
    assert(population.size() >= n);

#ifndef NDEBUG
    const size_t initial_n_samples = samples_idx.size();
#endif

    for (size_t i = 0; i < n; i++) {
        const size_t prev_n_samples = samples_idx.size();
        do {
            const size_t sample_i =
                    rododendrs::rnd_in_range(0, population.size());
            samples_idx.insert(sample_i);
        } while (prev_n_samples == samples_idx.size());
    }

#ifndef NDEBUG
    assert(samples_idx.size() == initial_n_samples + n);
#endif
}

template <typename TContainer>
void sample_out(const TContainer& population,
                std::set<size_t>& samples_idx,
                size_t n)
{
    assert(n > 0);
    assert(population.size() >= n);

#ifndef NDEBUG
    const size_t initial_n_samples = samples_idx.size();
#endif

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

#ifndef NDEBUG
    assert(samples_idx.size() == initial_n_samples + n);
#endif
}

template <typename TContainer>
void sample(const TContainer& population,
            std::set<size_t>& samples_idx,
            size_t n)
{
    // see 'benchmarks/sampling_f' for how this ratio was found
    const double SAMPLE_RATIO_IN = 0.6;

    const double sample_ratio = (double)n / (double)population.size();
    assert(sample_ratio > 0);
    assert(sample_ratio <= 1.0);

    if (sample_ratio <= SAMPLE_RATIO_IN) {
        return sample_in<TContainer>(population, samples_idx, n);
    }
    return sample_out<TContainer>(population, samples_idx, n);
}

}  // namespace rododendrs
