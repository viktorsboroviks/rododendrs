#include <cassert>
#include <iostream>
#include <vector>

#include "rododendrs.hpp"

#ifdef NDEBUG
#error "asserts must be enabled for tests"
#endif

void test_cdf()
{
    const size_t data_n_min = 1;
    const size_t data_n_max = 10000;
    const double data_v_min = -1000;
    const double data_v_max = 1000;
    const size_t remove_n   = 100;

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

    // check removal
    for (size_t remove_i = 0; remove_i < remove_n; remove_i++) {
        if (data_size < remove_i + 2) {
            break;
        }

        // do not remove last item so rm_i is always correct
        const size_t rm_i =
                rododendrs::rnd_in_range<size_t>(0, data_size - remove_i - 2);
        assert(cdf.size() == data_size - remove_i);
        const double rm_v = cdf.sorted_values[rm_i];
        cdf.remove(rm_v);
        assert(rm_v != cdf.sorted_values[rm_i]);
        assert(cdf.size() == data_size - remove_i - 1);

        // check sorted order
        v_prev = data_v_min;
        for (const double& v : cdf.sorted_values) {
            assert(v >= v_prev);
            v_prev = v;
        }
    }
}

void test_kstest_fail()
{
    // test empty cdf
    // expected assert to fail
    rododendrs::CDF cdf_a;
    rododendrs::CDF cdf_b;

    assert(cdf_a.empty());
    assert(cdf_b.empty());

    assert(rododendrs::kstest(cdf_a, cdf_b) == 0.0);
}

void test_kstest_same()
{
    // test cdf of same value v
    const double v_min   = -100;
    const double v_max   = 100;
    const size_t len_min = 1;
    const size_t len_max = 1000;

    rododendrs::CDF cdf_a;
    rododendrs::CDF cdf_b;
    assert(cdf_a.empty());
    assert(cdf_b.empty());

    const double v   = rododendrs::rnd_in_range<double>(v_min, v_max);
    const size_t len = rododendrs::rnd_in_range<size_t>(len_min, len_max);
    for (size_t i = 0; i < len; i++) {
        cdf_a.insert(v);
        cdf_b.insert(v);
    }
    assert(cdf_a.size() == len);
    assert(cdf_b.size() == len);

    const double kstest = rododendrs::kstest(cdf_a, cdf_b);
    assert(kstest == 0);
}

void test_kstest_not_same()
{
    // test cdf of same differnt values va, vb
    const double v_min   = -100;
    const double v_max   = 100;
    const size_t len_min = 1;
    const size_t len_max = 1000;

    rododendrs::CDF cdf_a;
    rododendrs::CDF cdf_b;
    assert(cdf_a.empty());
    assert(cdf_b.empty());

    const double va = rododendrs::rnd_in_range<double>(v_min, v_max);
    double vb;
    do {
        vb = rododendrs::rnd_in_range<double>(v_min, v_max);
    } while (va == vb);
    assert(va != vb);
    const size_t len = rododendrs::rnd_in_range<size_t>(len_min, len_max);
    for (size_t i = 0; i < len; i++) {
        cdf_a.insert(va);
        cdf_b.insert(vb);
    }
    assert(cdf_a.size() == len);
    assert(cdf_b.size() == len);

    const double kstest    = rododendrs::kstest(cdf_a, cdf_b);
    const double float_err = 1e-5;
    assert(kstest - 1.0 < float_err);
}

void test_kstest_match_rnd()
{
    // test cdf of matching random values
    const double v_min   = -100;
    const double v_max   = 100;
    const size_t len_min = 1;
    const size_t len_max = 1000;

    rododendrs::CDF cdf_a;
    rododendrs::CDF cdf_b;
    assert(cdf_a.empty());
    assert(cdf_b.empty());

    const size_t len = rododendrs::rnd_in_range<size_t>(len_min, len_max);
    for (size_t i = 0; i < len; i++) {
        const double v = rododendrs::rnd_in_range<double>(v_min, v_max);
        cdf_a.insert(v);
        cdf_b.insert(v);
    }
    assert(cdf_a.size() == len);
    assert(cdf_b.size() == len);

    const double kstest = rododendrs::kstest(cdf_a, cdf_b);
    assert(kstest == 0);
}

void test_kstest_match_rnd_n()
{
    // test cdf of matching random values with matching random occurence
    const double v_min   = -100;
    const double v_max   = 100;
    const size_t n_min   = 1;
    const size_t n_max   = 10;
    const size_t len_min = 1;
    const size_t len_max = 1000;

    rododendrs::CDF cdf_a;
    rododendrs::CDF cdf_b;
    assert(cdf_a.empty());
    assert(cdf_b.empty());

    const size_t len = rododendrs::rnd_in_range<size_t>(len_min, len_max);
    size_t total_len = 0;
    for (size_t i = 0; i < len; i++) {
        const double v = rododendrs::rnd_in_range<double>(v_min, v_max);
        const size_t n = rododendrs::rnd_in_range<size_t>(n_min, n_max);
        for (size_t j = 0; j < n; j++) {
            cdf_a.insert(v);
            cdf_b.insert(v);
            total_len++;
        }
    }
    assert(cdf_a.size() == total_len);
    assert(cdf_b.size() == total_len);

    const double kstest = rododendrs::kstest(cdf_a, cdf_b);
    assert(kstest == 0);
}

void test_kstest_match_rnd_n_len_diff()
{
    // test cdf of matching random values with matching random occurence
    const double v_min             = -100;
    const double v_max             = 100;
    const size_t n_min             = 1;
    const size_t n_max             = 100;
    const size_t len_min           = 1;
    const size_t len_max           = 10;
    const size_t len_diff_coef_min = 1;
    const size_t len_diff_coef_max = 10;

    rododendrs::CDF cdf_a;
    rododendrs::CDF cdf_b;
    assert(cdf_a.empty());
    assert(cdf_b.empty());

    const size_t len   = rododendrs::rnd_in_range<size_t>(len_min, len_max);
    size_t total_len_a = 0;
    size_t total_len_b = 0;
    for (size_t i = 0; i < len; i++) {
        const double v  = rododendrs::rnd_in_range<double>(v_min, v_max);
        const size_t na = rododendrs::rnd_in_range<size_t>(n_min, n_max);
        for (size_t j = 0; j < na; j++) {
            cdf_a.insert(v);
            total_len_a++;
        }
        const size_t len_diff_coef = rododendrs::rnd_in_range<size_t>(
                len_diff_coef_min, len_diff_coef_max);
        const size_t nb = na * len_diff_coef;
        for (size_t j = 0; j < nb; j++) {
            cdf_b.insert(v);
            total_len_b++;
        }
    }
    assert(cdf_a.size() == total_len_a);
    assert(cdf_b.size() == total_len_b);

    const double kstest    = rododendrs::kstest(cdf_a, cdf_b);
    const double float_err = 1e-5;
    assert(kstest - 0 < float_err);
}

int main()
{
    const size_t test_n = 100;

    for (size_t test_i = 0; test_i < test_n; test_i++) {
#if 0
        // debug
        std::cout << "test_i: " << test_i << std::endl;
#endif

        test_cdf();
#if 0
        test_kstest_fail();
#endif
        test_kstest_same();
        test_kstest_not_same();
        test_kstest_match_rnd();
        test_kstest_match_rnd_n();
        test_kstest_match_rnd_n_len_diff();
    }

    std::cout << "all tests passed" << std::endl;
    return 0;
}
