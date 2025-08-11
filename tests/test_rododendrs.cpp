#include <cassert>
#include <iostream>
#include <vector>

#include "rododendrs.hpp"

#ifdef NDEBUG
#error "asserts must be enabled for tests"
#endif

void test_cdf_sort()
{
    const size_t test_n     = 1000;
    const size_t data_n_min = 1;
    const size_t data_n_max = 10000;
    const double data_v_min = -1000;
    const double data_v_max = 1000;

    for (size_t test_i = 0; test_i < test_n; test_i++) {
        std::vector<double> data;
        const size_t data_size =
                rododendrs::rnd_in_range<size_t>(data_n_min, data_n_max);
        data.reserve(data_size);
        for (size_t data_i = 0; data_i < data_size; data_i++) {
            data.push_back(
                    rododendrs::rnd_in_range<double>(data_v_min, data_v_max));
        }

        // init
        rododendrs::CDF cdf;
        assert(cdf.empty());

        // insert data
        for (const double& v : data) {
            cdf.insert(v);
        }
        assert(!data.empty());
        assert(data.size() == data_size);

        // check sorted order
        double v_prev = data_v_min;
        for (const double& v : cdf.sorted_values) {
            assert(v >= v_prev);
            v_prev = v;
        }
    }

    std::cout << __FUNCTION__ << " done" << std::endl;
}

int main()
{
    test_cdf_sort();

    return 0;
}
