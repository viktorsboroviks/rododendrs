#pragma once

#include <cassert>
#include <cmath>
#include <deque>
#include <mutex>
#include <random>

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

    Random(const Random& other)
    {
        // construct a new object on copy
        (void)other;
        Random();
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

// relative mean squared error
template <template <typename...> typename P, template <typename...> typename C>
double rmse(const P<double>& predicted, const C<double>& correct)
{
    assert(!predicted.empty());
    assert(predicted.size() == correct.size());
    double sum = 0;
    for (size_t i = 0; i < predicted.size(); i++) {
        sum += std::pow(predicted[i] - correct[i], 2) /
               std::pow(correct[i], 2);
    }
    sum = sum / (double)predicted.size();
    return std::sqrt(sum);
}

}  // namespace rododendrs
