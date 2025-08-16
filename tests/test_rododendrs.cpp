#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "rododendrs.hpp"

#ifdef NDEBUG
#error "asserts must be enabled for tests"
#endif

void test_cdf()
{
    const size_t data_n_min = 1;
    const size_t data_n_max = 100;
    const double data_v_min = -10;
    const double data_v_max = 10;
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

void test_cdfctx()
{
    const size_t data_len_min = 1;
    const size_t data_len_max = 100;
    const double data_v_min   = -10;
    const double data_v_max   = 10;
    const size_t data_n_min   = 1;
    const size_t data_n_max   = 100;

    std::vector<double> data_v;
    std::vector<double> data_n;
    const size_t data_len =
            rododendrs::rnd_in_range<size_t>(data_len_min, data_len_max);
    data_v.reserve(data_len);
    data_n.reserve(data_len);
    for (size_t data_i = 0; data_i < data_len; data_i++) {
        data_v.push_back(
                rododendrs::rnd_in_range<double>(data_v_min, data_v_max));
        data_n.push_back(
                rododendrs::rnd_in_range<double>(data_n_min, data_n_max));
    }

    rododendrs::CDF cdf;
    assert(cdf.empty());

    for (size_t i = 0; i < data_len; i++) {
        for (size_t j = 0; j < data_n[i]; j++) {
            cdf.insert(data_v[i]);
        }
    }

    rododendrs::CdfCtx ctx(cdf);
    size_t cdf_i       = 0;
    size_t cdf_v_count = 0;
    while (!ctx.is_end()) {
        const double v = ctx.val();
        const auto it  = std::find(data_v.begin(), data_v.end(), v);
        assert(it != data_v.end());
        const size_t di = std::distance(data_v.begin(), it);
        cdf_i += data_n[di];
        assert(cdf_i <= cdf.size());

        cdf_v_count++;
        ctx.next();
    }
    assert(cdf_v_count == data_v.size());
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
    assert(rododendrs::approx_equal<double>(kstest, 0.0));
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

    const double kstest = rododendrs::kstest(cdf_a, cdf_b);
    assert(rododendrs::approx_equal<double>(kstest, 1.0));
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
    assert(rododendrs::approx_equal<double>(kstest, 0.0));
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
    assert(rododendrs::approx_equal<double>(kstest, 0.0));
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
    const size_t len_diff_coef = rododendrs::rnd_in_range<size_t>(
            len_diff_coef_min, len_diff_coef_max);
    for (size_t i = 0; i < len; i++) {
        const double v  = rododendrs::rnd_in_range<double>(v_min, v_max);
        const size_t na = rododendrs::rnd_in_range<size_t>(n_min, n_max);
        for (size_t j = 0; j < na; j++) {
            cdf_a.insert(v);
            total_len_a++;
        }
        const size_t nb = na * len_diff_coef;
        for (size_t j = 0; j < nb; j++) {
            cdf_b.insert(v);
            total_len_b++;
        }
    }
    assert(cdf_a.size() == total_len_a);
    assert(cdf_b.size() == total_len_b);

    const double kstest = rododendrs::kstest(cdf_a, cdf_b);
    assert(rododendrs::approx_equal<double>(kstest, 0.0));
}

void test_kstest_match_rnd_vals_nomatch_rnd_n()
{
    // test cdf of matching rnd values with not matching rnd n
    const double v_min   = -10;
    const double v_max   = 10;
    const size_t n_min   = 1;
    const size_t n_max   = 5;
    const size_t len_min = 1;
    const size_t len_max = 3;

    rododendrs::CDF cdf_a;
    rododendrs::CDF cdf_b;
    assert(cdf_a.empty());
    assert(cdf_b.empty());

    const size_t len   = rododendrs::rnd_in_range<size_t>(len_min, len_max);
    size_t total_len_a = 0;
    size_t total_len_b = 0;
    std::vector<double> ia_vnew = {0};
    std::vector<double> ib_vnew = {0};
    assert(ia_vnew.size() == 1);
    assert(ib_vnew.size() == 1);

    double v_prev = v_min;
    for (size_t i = 0; i < len; i++) {
        double v;
        do {
            v = rododendrs::rnd_in_range<double>(v_min, v_max);
        } while (v <= v_prev);
        assert(v > v_prev);
        v_prev          = v;
        const size_t na = rododendrs::rnd_in_range<size_t>(n_min, n_max);
        for (size_t j = 0; j < na; j++) {
            cdf_a.insert(v);
            total_len_a++;
        }

        size_t nb;
        do {
            nb = rododendrs::rnd_in_range<size_t>(n_min, n_max);
        } while (nb == na);
        assert(na != nb);
        for (size_t j = 0; j < nb; j++) {
            cdf_b.insert(v);
            total_len_b++;
        }

        ia_vnew.push_back(total_len_a);
        ib_vnew.push_back(total_len_b);
    }
    assert(cdf_a.size() == total_len_a);
    assert(cdf_b.size() == total_len_b);

    const double kstest = rododendrs::kstest(cdf_a, cdf_b);

    const double pa_step = 1.0 / static_cast<double>(total_len_a);
    const double pb_step = 1.0 / static_cast<double>(total_len_b);
    double pdiff_max     = 0;
    for (size_t i = 0; i < ia_vnew.size(); i++) {
        const double pa = pa_step * ia_vnew[i];
        const double pb = pb_step * ib_vnew[i];
        pdiff_max       = std::max(pdiff_max, std::abs(pa - pb));
    }

    assert(rododendrs::approx_equal<double>(kstest, pdiff_max));
}

void test_kstest_nomatch_rnd_vals_nomatch_rnd_n()
{
    // test cdf of not matching rnd values with not matching rnd n
    const double v_min   = -10;
    const double v_max   = 10;
    const size_t n_min   = 1;
    const size_t n_max   = 5;
    const size_t len_min = 1;
    const size_t len_max = 3;

    rododendrs::CDF cdf_a;
    rododendrs::CDF cdf_b;
    assert(cdf_a.empty());
    assert(cdf_b.empty());

    const size_t len_a = rododendrs::rnd_in_range<size_t>(len_min, len_max);
    const size_t len_b = rododendrs::rnd_in_range<size_t>(len_min, len_max);
    size_t total_len_a = 0;
    size_t total_len_b = 0;
    std::deque<double> a_vnew;
    std::deque<double> b_vnew;
    std::deque<double> ia_vnew;
    std::deque<double> ib_vnew;

    // init cdf_a
    double v_prev = v_min;
    for (size_t i = 0; i < len_a; i++) {
        double v;
        do {
            v = rododendrs::rnd_in_range<double>(v_min, v_max);
        } while (v <= v_prev);
        assert(v > v_prev);
        v_prev         = v;
        const size_t n = rododendrs::rnd_in_range<size_t>(n_min, n_max);
        for (size_t j = 0; j < n; j++) {
            cdf_a.insert(v);
            total_len_a++;
        }

        a_vnew.push_back(v);
        ia_vnew.push_back(total_len_a);
    }
    assert(cdf_a.size() == total_len_a);

    // init cdf_b
    v_prev = v_min;
    for (size_t i = 0; i < len_b; i++) {
        double v;
        do {
            v = rododendrs::rnd_in_range<double>(v_min, v_max);
        } while (v <= v_prev);
        assert(v > v_prev);
        v_prev         = v;
        const size_t n = rododendrs::rnd_in_range<size_t>(n_min, n_max);
        for (size_t j = 0; j < n; j++) {
            cdf_b.insert(v);
            total_len_b++;
        }

        b_vnew.push_back(v);
        ib_vnew.push_back(total_len_b);
    }
    assert(cdf_b.size() == total_len_b);

    // real kstest
    const double kstest = rododendrs::kstest(cdf_a, cdf_b);

    // test kstest
    const double pa_step = 1.0 / static_cast<double>(total_len_a);
    const double pb_step = 1.0 / static_cast<double>(total_len_b);
    double pdiff_max     = 0;
    double pa            = 0;
    double pb            = 0;
    while (true) {
        pdiff_max = std::max(pdiff_max, std::abs(pa - pb));

        // get next pa, pb
        bool update_a = false;
        bool update_b = false;
        if (!a_vnew.empty() && !b_vnew.empty() &&
            (a_vnew.front() == b_vnew.front())) {
            update_a = true;
            update_b = true;
        }
        else if (!a_vnew.empty() &&
                 (b_vnew.empty() || (a_vnew.front() < b_vnew.front()))) {
            update_a = true;
        }
        else if (!b_vnew.empty()) {
            update_b = true;
        }
        else {
            break;
        }

        if (update_a) {
            assert(!a_vnew.empty());
            pa = ia_vnew.front() * pa_step;
            a_vnew.pop_front();
            ia_vnew.pop_front();
        }
        if (update_b) {
            assert(!b_vnew.empty());
            pb = ib_vnew.front() * pb_step;
            b_vnew.pop_front();
            ib_vnew.pop_front();
        }
    }
    assert(rododendrs::approx_equal<double>(pa, 1.0));
    assert(rododendrs::approx_equal<double>(pb, 1.0));

    assert(rododendrs::approx_equal<double>(kstest, pdiff_max));
}

void test_kstest_save_restore()
{
    // compare saved/restored kstest with vanilla version
    const double v_min        = -10;
    const double v_max        = 10;
    const size_t n_min        = 1;
    const size_t n_max        = 5;
    const size_t len_min      = 1;
    const size_t len_max      = 3;

    rododendrs::CDF cdf_a;
    rododendrs::CDF cdf_b;

    // a
    const size_t len_a =
            rododendrs::rnd_in_range<size_t>(len_min, len_max);
    double v_prev = v_min;
    for (size_t i = 0; i < len_a; i++) {
        double v;
        do {
            v = rododendrs::rnd_in_range<double>(
                    v_min, (v_max / len_a) * (i + 1));
        } while (v <= v_prev);
        assert(v > v_prev);
        v_prev        = v;
        const size_t n = rododendrs::rnd_in_range<size_t>(n_min, n_max);
        for (size_t j = 0; j < n; j++) {
            cdf_a.insert(v);
        }
    }

    // b
    const size_t len_b =
            rododendrs::rnd_in_range<size_t>(len_min, len_max);
    v_prev = v_min;
    for (size_t i = 0; i < len_b; i++) {
        double v;
        do {
            v = rododendrs::rnd_in_range<double>(
                    v_min, (v_max / len_b) * (i + 1));
        } while (v <= v_prev);
        assert(v > v_prev);
        v_prev        = v;
        const size_t n = rododendrs::rnd_in_range<size_t>(n_min, n_max);
        for (size_t j = 0; j < n; j++) {
            cdf_b.insert(v);
        }
    }

    const double kstest_ref = rododendrs::kstest(cdf_a, cdf_b);
    rododendrs::KstestCtx kctx(cdf_a, cdf_b);

    const double kstest_first = rododendrs::kstest(kctx, cdf_a.size(), cdf_b.size());
    kctx.reset();
    const double kstest_reset = rododendrs::kstest(kctx, cdf_a.size(), cdf_b.size());
    kctx.restore();
    const double kstest_restore = rododendrs::kstest(kctx, cdf_a.size(), cdf_b.size());
    kctx.resize(cdf_a.size(), cdf_b.size());
    const double kstest_resize = rododendrs::kstest(kctx, cdf_a.size(), cdf_b.size());

    assert(rododendrs::approx_equal<double>(kstest_ref, kstest_first));
    assert(rododendrs::approx_equal<double>(kstest_ref, kstest_reset));
    assert(rododendrs::approx_equal<double>(kstest_ref, kstest_restore));
    assert(rododendrs::approx_equal<double>(kstest_ref, kstest_resize));
}

void test_kstest_nested_cdf()
{
    // nested kctx with interrupts at arbitrary place/in the middle of
    // unique_val stack
    const double v_min        = -10;
    const double v_max        = 10;
    const size_t n_min        = 1;
    const size_t n_max        = 5;
    const size_t len_min      = 1;
    const size_t len_max      = 3;
    const size_t n_nested_min = 1;
    const size_t n_nested_max = 10;

    std::vector<rododendrs::CDF> cdfs_a;
    std::vector<rododendrs::CDF> cdfs_b;
    rododendrs::CDF supercdf_a;
    rododendrs::CDF supercdf_b;

    // init data
    const size_t n_nested =
            rododendrs::rnd_in_range<size_t>(n_nested_min, n_nested_max);
    cdfs_a.resize(n_nested);
    cdfs_b.resize(n_nested);

    // create nested cdfs
    double va_prev = v_min;
    double vb_prev = v_min;
    for (size_t i_nest = 0; i_nest < n_nested; i_nest++) {
        // a
        const size_t len_a =
                rododendrs::rnd_in_range<size_t>(len_min, len_max);
        for (size_t i = 0; i < len_a; i++) {
            double v;
            do {
                v = rododendrs::rnd_in_range<double>(
                        v_min, (v_max / n_nested) * (i_nest + 1));
            } while (v <= va_prev);
            assert(v > va_prev);
            va_prev        = v;
            const size_t n = rododendrs::rnd_in_range<size_t>(n_min, n_max);
            for (size_t j = 0; j < n; j++) {
                supercdf_a.insert(v);
                for (size_t i_cdf = 0; i_cdf < (n_nested - i_nest); i_cdf++) {
                    cdfs_a[i_cdf].insert(v);
                }
            }
        }

        // b
        const size_t len_b =
                rododendrs::rnd_in_range<size_t>(len_min, len_max);
        for (size_t i = 0; i < len_b; i++) {
            double v;
            do {
                v = rododendrs::rnd_in_range<double>(
                        v_min, (v_max / n_nested) * (i_nest + 1));
            } while (v <= vb_prev);
            assert(v > vb_prev);
            vb_prev        = v;
            const size_t n = rododendrs::rnd_in_range<size_t>(n_min, n_max);
            for (size_t j = 0; j < n; j++) {
                supercdf_b.insert(v);
                for (size_t i_cdf = 0; i_cdf < (n_nested - i_nest); i_cdf++) {
                    cdfs_b[i_cdf].insert(v);
                }
            }
        }
    }

    // kstest check
    rododendrs::KstestCtx kctx(supercdf_a, supercdf_b);
    rododendrs::KstestCtx kctx_reset(supercdf_a, supercdf_b);
    size_t prev_cdf_size_a = 0;
    size_t prev_cdf_size_b = 0;
    for (size_t i_cdf = n_nested - 1; i_cdf > 0; i_cdf--) {
        std::cout << "i_cdf: " << i_cdf << std::endl;
        assert(cdfs_a[i_cdf].size() >= prev_cdf_size_a);
        assert(cdfs_b[i_cdf].size() >= prev_cdf_size_b);
        assert(!cdfs_a[i_cdf].empty());
        assert(!cdfs_b[i_cdf].empty());
        prev_cdf_size_a = cdfs_a[i_cdf].size();
        prev_cdf_size_b = cdfs_b[i_cdf].size();

        for (size_t i = 0; i < cdfs_a[i_cdf].size(); i++) {
            assert(cdfs_a[i_cdf].sorted_values[i] ==
                   supercdf_a.sorted_values[i]);
        }
        for (size_t i = 0; i < cdfs_b[i_cdf].size(); i++) {
            assert(cdfs_b[i_cdf].sorted_values[i] ==
                   supercdf_b.sorted_values[i]);
        }

        assert(kctx.a_next.i == kctx_reset.a_next.i);
        assert(kctx.b_next.i == kctx_reset.b_next.i);
        assert(kctx.a_next.i_vnew == kctx_reset.a_next.i_vnew);
        assert(kctx.b_next.i_vnew == kctx_reset.b_next.i_vnew);
        assert(kctx.a_next.i_max_pdiff == kctx_reset.a_next.i_max_pdiff);
        assert(kctx.b_next.i_max_pdiff == kctx_reset.b_next.i_max_pdiff);

        std::cout << "ctx - before kstest) " << std::endl;
        std::cout << "\ta: " << kctx.a_next.i_max_pdiff << "/" << kctx.a_next.i_end << std::endl;
        std::cout << "\tb: " << kctx.b_next.i_max_pdiff << "/" << kctx.b_next.i_end << std::endl;
        std::cout << kctx.max_pdiff << std::endl;
        const double kstest1 = rododendrs::kstest(
                kctx,
                cdfs_a[i_cdf].size(),
                cdfs_b[i_cdf].size());
        assert(kctx.a_next.i == kctx.a_next.i_end);
        assert(kctx.b_next.i == kctx.b_next.i_end);
        assert(kctx.max_pdiff == kstest1);

        std::cout << "ctx - after kstest) " << std::endl;
        std::cout << "\ta: " << kctx.a_next.i_max_pdiff << "/" << kctx.a_next.i_end << std::endl;
        std::cout << "\tb: " << kctx.b_next.i_max_pdiff << "/" << kctx.b_next.i_end << std::endl;
        std::cout << kctx.max_pdiff << std::endl;

        std::cout << "ref - before reset) " << std::endl;
        std::cout << "\ta: " << kctx_reset.a_next.i_max_pdiff << "/" << kctx_reset.a_next.i_end << std::endl;
        std::cout << "\tb: " << kctx_reset.b_next.i_max_pdiff << "/" << kctx_reset.b_next.i_end << std::endl;
        std::cout << kctx_reset.max_pdiff << std::endl;

        kctx_reset.reset();
        std::cout << "ref - after reset) " << std::endl;
        std::cout << "\ta: " << kctx_reset.a_next.i_max_pdiff << "/" << kctx_reset.a_next.i_end << std::endl;
        std::cout << "\tb: " << kctx_reset.b_next.i_max_pdiff << "/" << kctx_reset.b_next.i_end << std::endl;
        std::cout << kctx_reset.max_pdiff << std::endl;

        const double kstest2 = rododendrs::kstest(
                kctx_reset,
                cdfs_a[i_cdf].size(),
                cdfs_b[i_cdf].size());
        std::cout << "ref - after kstest) " << std::endl;
        std::cout << "\ta: " << kctx_reset.a_next.i_max_pdiff << "/" << kctx_reset.a_next.i_end << std::endl;
        std::cout << "\tb: " << kctx_reset.b_next.i_max_pdiff << "/" << kctx_reset.b_next.i_end << std::endl;
        std::cout << kctx_reset.max_pdiff << std::endl;
        assert(kctx.a_next.i_begin == kctx_reset.a_next.i_begin);
        assert(kctx.a_next.i_end == kctx_reset.a_next.i_end);
        assert(kctx.b_next.i_begin == kctx_reset.b_next.i_begin);
        assert(kctx.b_next.i_end == kctx_reset.b_next.i_end);
        assert(kctx_reset.a_next.i == kctx_reset.a_next.i_end);
        assert(kctx_reset.b_next.i == kctx_reset.b_next.i_end);
        assert(kctx_reset.max_pdiff == kstest2);

        assert(kctx.a_next.i == kctx_reset.a_next.i);
        assert(kctx.b_next.i == kctx_reset.b_next.i);
        assert(kctx.a_next.i_vnew == kctx_reset.a_next.i_vnew);
        assert(kctx.b_next.i_vnew == kctx_reset.b_next.i_vnew);
        assert(kctx.a_next.i_max_pdiff == kctx_reset.a_next.i_max_pdiff);
        assert(kctx.b_next.i_max_pdiff == kctx_reset.b_next.i_max_pdiff);

        const double kstest3 =
                rododendrs::kstest(cdfs_a[i_cdf], cdfs_b[i_cdf]);

        assert(rododendrs::approx_equal<double>(kstest1, kstest2));
        assert(rododendrs::approx_equal<double>(kstest2, kstest3));
    }
}

int main()
{
    const size_t test_n = 100;

    for (size_t test_i = 0; test_i < test_n; test_i++) {
        test_cdf();
        test_cdfctx();
#if 0
                        test_kstest_fail();
#endif
        test_kstest_same();
        test_kstest_not_same();
        test_kstest_match_rnd();
        test_kstest_match_rnd_n();
        test_kstest_match_rnd_n_len_diff();
        test_kstest_match_rnd_vals_nomatch_rnd_n();
        test_kstest_nomatch_rnd_vals_nomatch_rnd_n();
        test_kstest_save_restore();
        test_kstest_nested_cdf();
    }

    std::cout << "all tests passed" << std::endl;
    return 0;
}
