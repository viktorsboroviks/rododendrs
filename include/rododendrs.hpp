#pragma once

#include <cassert>
#include <cmath>
#include <deque>
#include <mutex>
#include <random>
#include <set>
#include <vector>

namespace rododendrs {

// can be used as a rng in std::shuffle and similar calls
inline auto rng = std::mt19937{std::random_device{}()};

double rnd_in_range(double min, double max)
{
    if (min == max) {
        return min;
    }
    std::uniform_real_distribution<double> unif_dist(min, max);
    const double retval = unif_dist(rng);
    assert(retval >= min);
    assert(retval <= max);
    return retval;
}

size_t rnd_in_range(size_t min, size_t max)
{
    if (min == max) {
        return min;
    }
    std::uniform_int_distribution<size_t> unif_dist(min, max);
    const double retval = unif_dist(rng);
    assert(retval >= min);
    assert(retval <= max);
    return retval;
}

double rnd01()
{
    return rnd_in_range(0.0, 1.0);
}

double rnd_norm(double mean, double sd, double min, double max)
{
    if (min == max) {
        return min;
    }
    std::normal_distribution<double> norm_dist(mean, sd);
    const double retval = std::clamp(norm_dist(rng), min, max);
    assert(retval >= min);
    assert(retval <= max);
    return retval;
}

// normal distribution with mean 0 and sd 1/3 in range [-1, 1]
double rnd01_norm()
{
    return rnd_norm(0.0, 1.0 / 3.0, -1.0, 1.0);
}

// r-squared, coefficient of determination
template <template <typename...> typename P, template <typename...> typename C>
double r2(const P<double>& predicted, const C<double>& correct)
{
    assert(!predicted.empty());
    assert(predicted.size() == correct.size());
    const double mean_correct =
            std::accumulate(correct.begin(), correct.end(), 0.0) /
            correct.size();
    double ssr = 0;
    double sst = 0;
    for (size_t i = 0; i < predicted.size(); i++) {
        ssr += std::pow(correct[i] - predicted[i], 2);
        sst += std::pow(correct[i] - mean_correct, 2);
    }
    return 1.0 - (ssr / sst);
}

// mean squared error
// correct data is also called observed data
template <template <typename...> typename P, template <typename...> typename C>
double mse(const P<double>& predicted, const C<double>& correct)
{
    assert(!predicted.empty());
    assert(predicted.size() == correct.size());
    double sum = 0;
    for (size_t i = 0; i < predicted.size(); i++) {
        sum += std::pow(predicted[i] - correct[i], 2);
    }
    return sum / static_cast<double>(predicted.size());
}

// root mean squared error
// correct data is also called observed data
template <template <typename...> typename P, template <typename...> typename C>
double rmse(const P<double>& predicted, const C<double>& correct)
{
    return std::sqrt(mse<P, C>(predicted, correct));
}

// relative root mean squared error
// correct data is also called observed data
template <template <typename...> typename P, template <typename...> typename C>
double rrmse(const P<double>& predicted, const C<double>& correct)
{
    assert(!predicted.empty());
    assert(predicted.size() == correct.size());
    double sum_correct = 0;
    for (size_t i = 0; i < correct.size(); i++) {
        sum_correct += std::pow(correct[i], 2);
    }
    return rmse<P, C>(predicted, correct) / sum_correct;
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

void sample_in(std::set<size_t>& samples_idx, size_t population_size, size_t n)
{
    assert(n > 0);
    assert(population_size >= n);

#ifndef NDEBUG
    const size_t initial_n_samples = samples_idx.size();
#endif

    for (size_t i = 0; i < n; i++) {
        const size_t prev_n_samples = samples_idx.size();
        do {
            const size_t sample_i =
                    rododendrs::rnd_in_range(0, population_size);
            samples_idx.insert(sample_i);
        } while (prev_n_samples == samples_idx.size());
    }

#ifndef NDEBUG
    assert(samples_idx.size() == initial_n_samples + n);
#endif
}

void sample_out(std::set<size_t>& samples_idx,
                size_t population_size,
                size_t n)
{
    assert(n > 0);
    assert(population_size >= n);

#ifndef NDEBUG
    const size_t initial_n_samples = samples_idx.size();
#endif

    std::set<size_t> new_samples_idx;
    for (size_t i = 0; i < population_size; i++) {
        new_samples_idx.insert(i);
    }
    assert(new_samples_idx.size() == population_size);

    for (size_t i = 0; i < population_size - n; i++) {
        const size_t prev_n_samples = new_samples_idx.size();
        do {
            const size_t sample_i =
                    rododendrs::rnd_in_range(0, population_size);
            new_samples_idx.erase(sample_i);
        } while (prev_n_samples == new_samples_idx.size());
    }

    assert(new_samples_idx.size() == n);
    samples_idx.insert(new_samples_idx.begin(), new_samples_idx.end());

#ifndef NDEBUG
    assert(samples_idx.size() == initial_n_samples + n);
#endif
}

void sample_shuffle(std::set<size_t>& samples_idx,
                    size_t population_size,
                    size_t n)
{
    assert(n > 0);
    assert(population_size >= n);
    assert(samples_idx.empty());

    std::vector<size_t> idx;
    for (size_t i = 0; i < population_size; i++) {
        idx.push_back(i);
    }

    // Create a random number generator
    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(idx.begin(), idx.end(), g);

    for (size_t i = 0; i < n; i++) {
        samples_idx.insert(idx[i]);
    }

    assert(samples_idx.size() == n);
}

void sample(std::set<size_t>& samples_idx, size_t population_size, size_t n)
{
    // see 'benchmarks/sampling_f' for how this ratio was found
    const double SAMPLE_RATIO_IN      = 0.45;
    const double SAMPLE_RATIO_SHUFFLE = 0.9;

    const double sample_ratio =
            static_cast<double>(n) / static_cast<double>(population_size);
    assert(sample_ratio > 0);
    assert(sample_ratio <= 1.0);

    if (sample_ratio < SAMPLE_RATIO_IN) {
        return sample_in(samples_idx, population_size, n);
    }
    else if (sample_ratio < SAMPLE_RATIO_SHUFFLE) {
        return sample_shuffle(samples_idx, population_size, n);
    }
    return sample_out(samples_idx, population_size, n);
}

// Welford's online algorithms
// ref: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

double welford_mean(double prev_mean, double value, size_t n)
{
    assert(n > 0);
    return prev_mean + (value - prev_mean) / static_cast<double>(n);
}

double welford_m2(double m2_prev, double prev_mean, double value, size_t n)
{
    assert(n > 0);
    if (n == 1) {
        return 0;
    }

    const double mean = welford_mean(prev_mean, value, n);
    const double m2   = m2_prev + (value - prev_mean) * (value - mean);
    return m2;
}

double welford_variance(double m2, size_t n)
{
    assert(n > 0);
    return m2 / static_cast<double>(n - 1);
}

double confidence_interval_margin(double sd, size_t n, double z)
{
    assert(n > 0);
    return std::abs(z * sd / std::sqrt(static_cast<double>(n)));
}

std::pair<double, double> confidence_interval(double mean,
                                              double sd,
                                              size_t n,
                                              double z = 1.96)
{
    assert(n > 0);
    const double margin = confidence_interval_margin(sd, n, z);
    return std::make_pair(mean - margin, mean + margin);
}

// z values
// ref:
// https://www.researchgate.net/figure/Critical-z-values-used-in-the-calculation-of-confidence-intervals_tbl1_320742650
const double Z_CI_50PCT = 0.67449;
const double Z_CI_75PCT = 1.15035;
const double Z_CI_90PCT = 1.64485;
const double Z_CI_95PCT = 1.95996;
const double Z_CI_99PCT = 2.57583;

class SMA {
private:
    size_t _period;
    double _sum = 0;
    std::deque<double> _values;

public:
    SMA(size_t period) :
        _period(period)
    {
        assert(_period > 0);
    }

    size_t size() const
    {
        assert(_values.size() <= _period);
        return _values.size();
    }

    size_t period() const
    {
        return _period;
    }

    bool ready() const
    {
        return size() == period();
    }

    void insert(double value)
    {
        _values.push_back(value);

        if (_values.size() == _period) {
            assert(_sum == 0);
            for (const auto& value : _values) {
                _sum += value;
            }
        }

        if (_values.size() > _period) {
            _sum -= _values.front();
            _values.pop_front();
            _sum += value;
        }
    }

    double get() const
    {
        if (_values.size() < _period) {
            return 0;
        }
        assert(_values.size() == _period);
        return _sum / static_cast<double>(_period);
    }
};

}  // namespace rododendrs
