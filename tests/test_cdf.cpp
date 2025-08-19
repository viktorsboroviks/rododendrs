#include <iostream>

#include "rododendrs.hpp"

// generate 2 random cdfs
// save to files:
// - cdf_a.csv
// - cdf_b.csv
// print kstest value
int main()
{
    const double v_min   = -10;
    const double v_max   = 10;
    const size_t n_min   = 1;
    const size_t n_max   = 5;
    const size_t len_min = 1;
    const size_t len_max = 3;

    rododendrs::CDF cdf_a;
    rododendrs::CDF cdf_b;

    // a
    const size_t len_a = rododendrs::rnd_in_range<size_t>(len_min, len_max);
    double v_prev      = v_min;
    for (size_t i = 0; i < len_a; i++) {
        double v;
        do {
            v = rododendrs::rnd_in_range<double>(v_min,
                                                 (v_max / len_a) * (i + 1));
        } while (v <= v_prev);
        assert(v > v_prev);
        v_prev         = v;
        const size_t n = rododendrs::rnd_in_range<size_t>(n_min, n_max);
        for (size_t j = 0; j < n; j++) {
            cdf_a.insert(v);
        }
    }

    // b
    const size_t len_b = rododendrs::rnd_in_range<size_t>(len_min, len_max);
    v_prev             = v_min;
    for (size_t i = 0; i < len_b; i++) {
        double v;
        do {
            v = rododendrs::rnd_in_range<double>(v_min,
                                                 (v_max / len_b) * (i + 1));
        } while (v <= v_prev);
        assert(v > v_prev);
        v_prev         = v;
        const size_t n = rododendrs::rnd_in_range<size_t>(n_min, n_max);
        for (size_t j = 0; j < n; j++) {
            cdf_b.insert(v);
        }
    }

    cdf_a.to_csv("output/cdf_a.csv");
    cdf_b.to_csv("output/cdf_b.csv");
    rododendrs::KstestCtx kctx(cdf_a, cdf_b);
    const double kstest = rododendrs::kstest(kctx);

    std::cout << kstest << std::endl;

    return 0;
}
