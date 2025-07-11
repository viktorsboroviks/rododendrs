#pragma once

#define G_RNG_LOCK
// #define POPULATION_LOCK
// #define RDDR_DEBUG

#include <cassert>
#include <cmath>
#include <deque>
#include <limits>
#include <random>
#include <set>
#include <vector>
#if defined(G_RNG_LOCK) || defined(POPULATION_LOCK)
#include <mutex>
#endif

#ifdef RDDR_DEBUG
#define debug_rddr(x)                                                   \
    std::cout << "debug (" << __FILE__ << ":" << __LINE__ << "): " << x \
              << std::endl;
#else
#define debug_rddr(x) while (0)
#endif

namespace rododendrs {

// can be used as a rng in std::shuffle and similar calls
inline auto rng = std::mt19937{std::random_device{}()};

#ifdef G_RNG_LOCK
std::mutex g_rng_mutex;
#endif

template <typename T>
struct Range {
    T min;
    T max;
    T step_min;
    T step_max;
};

// warning!
// ai generated code: check that value is Range<any>
// see iestaade for example usage
template <typename T>
struct is_range : std::false_type {};
template <typename U>
struct is_range<Range<U>> : std::true_type {};
template <typename T>
inline constexpr bool is_range_v = is_range<T>::value;

// warning!
// ai generated code: from Range<T> get type T
// see iestaade for example usage
template <typename T>
struct range_value_type;
template <typename U>
struct range_value_type<Range<U>> {
    using type = U;
};
template <typename T>
using range_value_type_t = typename range_value_type<T>::type;

template <typename T>
T rnd_in_range(T min, T max)
{
    if (min == max) {
        return min;
    }
#ifdef G_RNG_LOCK
    std::lock_guard<std::mutex> lock(g_rng_mutex);
#endif
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> unif_dist(min, max);
        const T retval = unif_dist(rng);
        assert(retval >= min);
        assert(retval <= max);
        return retval;
    }
    else {
        std::uniform_real_distribution<T> unif_dist(min, max);
        const T retval = unif_dist(rng);
        assert(retval >= min);
        assert(retval <= max);
        return retval;
    }
}

template <typename T>
T rnd_in_range(const Range<T>& range)
{
    return rnd_in_range<T>(range.min, range.max);
}

double rnd01()
{
    return rnd_in_range<double>(0.0, 1.0);
}

// changes val by a random step in defined range
template <typename T>
T rnd_step(T val, T step_min, T step_max, T min, T max)
{
    assert(step_min <= step_max);
    assert(min <= max);
    T step = rododendrs::rnd_in_range<T>(step_min, step_max);
    if (rododendrs::rnd01() < 0.5) {
        step *= -1;
    }

    return std::clamp(val + step, min, max);
}

template <typename T>
T rnd_step(T val, const Range<T>& range)
{
    return rnd_step(val, range.step_min, range.step_max, range.min, range.max);
}

double rnd_norm(double mean, double sd, double min, double max)
{
    if (min == max) {
        return min;
    }
#ifdef G_RNG_LOCK
    std::lock_guard<std::mutex> lock(g_rng_mutex);
#endif
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
                    rododendrs::rnd_in_range<size_t>(0, population_size);
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
                    rododendrs::rnd_in_range<size_t>(0, population_size);
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

    const double _mean = welford_mean(prev_mean, value, n);
    const double m2    = m2_prev + (value - prev_mean) * (value - _mean);
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

struct ConfidenceInterval {
    double lower;  // cppcheck-suppress unusedStructMember
    double upper;  // cppcheck-suppress unusedStructMember
};

ConfidenceInterval confidence_interval(double mean,
                                       double sd,
                                       size_t n,
                                       double z = 1.96)
{
    assert(n > 0);
    const double margin = confidence_interval_margin(sd, n, z);
    return ConfidenceInterval{mean - margin, mean + margin};
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
    explicit SMA(size_t period) :
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

            _sum = std::accumulate(_values.begin(), _values.end(), 0.0);
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

struct CDF {
    std::vector<double> unique_values;
    std::vector<double> p;
};

template <typename T>
struct Stats {
    T min  = 0;
    T max  = 0;
    T mean = 0;
    T n    = 0;

    void reset()
    {
        min  = 0;
        max  = 0;
        mean = 0;
        n    = 0;
    }

    void add(T value)
    {
        n++;

        if (n == 1) {
            min = max = mean = value;
            return;
        }

        min  = std::min(min, value);
        max  = std::max(max, value);
        mean = welford_mean(mean, value, n);
    }
};

class Population {
private:
#ifdef POPULATION_LOCK
    std::mutex _mutex;
#endif

    size_t _max_size = 0;
    std::vector<double> _sorted_indices;

    Stats<double> _stats;

public:
    std::vector<double> values;

    void reserve(size_t max_size)
    {
        _max_size = max_size;

        values.reserve(_max_size);
        _sorted_indices.reserve(_max_size);

        assert(values.empty());
        assert(_sorted_indices.empty());
    }

    explicit Population(size_t max_size)
    {
        reserve(max_size);
    }

    Population() :
        Population(0)
    {
    }

#ifdef POPULATION_LOCK
    // custom copy constructor
    // cppcheck-suppress missingMemberCopy
    Population(const Population& other) :
        _sorted_indices(other._sorted_indices),
        _stats(other._stats),
        values(other.values)
    {
    }
#endif

#ifdef POPULATION_LOCK
    // custom copy assignment operator
    Population& operator=(const Population& other)
    {
        if (this != &other) {
            // Lock both mutexes to ensure thread safety during the copy
            std::lock_guard<std::mutex> lock(_mutex);

            values          = other.values;
            _sorted_indices = other._sorted_indices;
            _stats          = other._stats;
            _max_size       = other._max_size;

            // _mutex is not copied; it remains unique to this object
        }
        return *this;
    }
#endif

    void reset()
    {
#ifdef POPULATION_LOCK
        std::lock_guard<std::mutex> lock(_mutex);
#endif
        values.clear();
        values.reserve(_max_size);
        _sorted_indices.clear();
        _sorted_indices.reserve(_max_size);
        _stats.reset();

        assert(values.empty());
        assert(_sorted_indices.empty());
    }

    void insert(double value)
    {
#ifdef POPULATION_LOCK
        std::lock_guard<std::mutex> lock(_mutex);
#endif
        assert(values.size() == _sorted_indices.size());
        assert(values.size() < values.capacity());
        assert(_sorted_indices.size() < _sorted_indices.capacity());

        const size_t i_new = values.size();
        auto it            = std::lower_bound(_sorted_indices.begin(),
                                   _sorted_indices.end(),
                                   value,
                                   [&](size_t i, double new_value) {
                                       return values[i] < new_value;
                                   });

        _sorted_indices.insert(it, i_new);
        values.push_back(value);
        assert(values.size() > 0);
        assert(_sorted_indices.size() == values.size());

        _stats.add(value);
        assert(_stats.n == values.size());
    }

    size_t size() const
    {
        return values.size();
    }

    bool empty() const
    {
        return values.empty();
    }

    size_t capacity() const
    {
        return values.capacity();
    }

    double min() const
    {
        return _stats.min;
    }

    double max() const
    {
        return _stats.max;
    }

    double mean() const
    {
        return _stats.mean;
    }

    double median()
    {
#ifdef POPULATION_LOCK
        std::lock_guard<std::mutex> lock(_mutex);
#endif

        // ref: https://en.wikipedia.org/wiki/Median
        // odd number
        if (values.size() % 2 == 1) {
            const size_t med_i = (values.size() + 1) / 2;
            return values[_sorted_indices[med_i]];
        }

        // even number
        const size_t med_i1 = values.size() / 2;
        const size_t med_i2 = med_i1 + 1;
        return (values[_sorted_indices[med_i1]] +
                values[_sorted_indices[med_i2]]) /
               2.0;
    }

    CDF cdf()
    {
#ifdef POPULATION_LOCK
        std::lock_guard<std::mutex> lock(_mutex);
#endif
        assert(!_sorted_indices.empty());
        assert(values.size() == _sorted_indices.size());

        CDF cdf;
        cdf.unique_values.reserve(values.size());
        cdf.p.reserve(values.size());

        double p            = 0.0;
        const double p_diff = 1.0 / static_cast<double>(values.size());

#ifndef NDEBUG
        double prev_v = values[_sorted_indices[0]];
#endif
        for (size_t i = 0; i < values.size(); i++) {
            assert(i < _sorted_indices.size());
            const size_t i_sorted = _sorted_indices[i];
            assert(i_sorted < values.size());
            const double v = values[i_sorted];
#ifndef NDEBUG
            assert(v >= prev_v);
            prev_v = v;
#endif

            p += p_diff;
            assert(p >= 0);

            // if new value
            if (cdf.unique_values.empty() || cdf.unique_values.back() != v) {
                cdf.unique_values.push_back(v);
                cdf.p.push_back(p);
            }
            else {
                // same value
                // only update p
                cdf.p.back() = p;
            }
        }

        assert(cdf.unique_values.size() == cdf.p.size());
        assert(cdf.p.back() > 0);
        assert(1.0 - cdf.p.back() < 1.0e-3);
        return cdf;
    }
};

double kstest(const CDF& cdf_a, const CDF& cdf_b)
{
    assert(!cdf_a.p.empty());
    assert(!cdf_b.p.empty());
    assert(cdf_a.unique_values.size() == cdf_a.p.size());
    assert(cdf_b.unique_values.size() == cdf_b.p.size());

    double ret = 0.0;

    size_t ia_next = 0;
    size_t ib_next = 0;
    double pa      = 0.0;
    double pb      = 0.0;
    double va      = cdf_a.unique_values[0];
    double vb      = cdf_b.unique_values[0];

    // loop until both cdfs are exhausted
    while (ia_next < cdf_a.p.size() && ib_next < cdf_b.p.size()) {
        // select candidates
        const double va_candidate = ia_next < cdf_a.unique_values.size()
                                            ? cdf_a.unique_values[ia_next]
                                            : va;
        const double vb_candidate = ib_next < cdf_b.unique_values.size()
                                            ? cdf_b.unique_values[ib_next]
                                            : vb;
        assert(va_candidate >= va);
        assert(vb_candidate >= vb);

        // compare candidates
        // advance
        if (va_candidate < vb_candidate) {
            assert(pa <= cdf_a.p[ia_next]);
            va = va_candidate;
            pa = cdf_a.p[ia_next];
            ia_next++;
        }
        else {
            assert(pb <= cdf_b.p[ib_next]);
            va = vb_candidate;
            pb = cdf_b.p[ib_next];
            ib_next++;
        }

        // get kstest
        ret = std::max(ret, std::abs(pa - pb));
        assert(ret > 0.0);
        assert(ret <= 1.0);

        // abort if remaining points cannot produce higher kstest value
        const double max_ret = 1.0 - std::min(pa, pb);
        if (max_ret < ret) {
            break;
        }
    }
    return ret;
}

}  // namespace rododendrs
