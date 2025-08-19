#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "rododendrs.hpp"

#ifdef NDEBUG
#error "asserts must be enabled for tests"
#endif

void test_kstest_files_partial()
{
    rododendrs::CDF supercdf_a("tests/data/supercdf_a.csv");
    rododendrs::CDF supercdf_b("tests/data/supercdf_b.csv");
    std::vector<rododendrs::CDF> cdfs_a;
    std::vector<rododendrs::CDF> cdfs_b;
    for (size_t i_cdf = 0; i_cdf <= 6; i_cdf++) {
        const std::string cdf_a_path =
                "tests/data/cdf_a" + std::to_string(i_cdf) + ".csv";
        const std::string cdf_b_path =
                "tests/data/cdf_b" + std::to_string(i_cdf) + ".csv";
        rododendrs::CDF cdf_a(cdf_a_path);
        rododendrs::CDF cdf_b(cdf_b_path);
        cdfs_a.push_back(cdf_a);
        cdfs_b.push_back(cdf_b);
    }

    const size_t n_nested = cdfs_a.size();
    std::ofstream f_dbg_ref("log_ref_part.txt");
    std::ofstream f_dbg_ctx("log_ctx_part.txt");

    // kstest check
    rododendrs::KstestCtx kctx(supercdf_a, supercdf_b);
    rododendrs::KstestCtx kctx_reset(supercdf_a, supercdf_b);
    size_t prev_cdf_len_a = 0;
    size_t prev_cdf_len_b = 0;

    kctx.max_pdiff           = 0.409;
    kctx.a_next.i_begin      = 0;
    kctx.a_next.i_end        = 11;
    kctx.a_next.i            = 11;
    kctx.a_next.i_vnew       = 10;
    kctx.a_next.i_max_pdiff  = 10;
    kctx.b_next.i_begin      = 0;
    kctx.b_next.i_end        = 26;
    kctx.b_next.i            = 26;
    kctx.b_next.i_vnew       = 25;
    kctx.b_next.i_max_pdiff  = 13;
    kctx.a_saved.i_begin     = 0;
    kctx.a_saved.i_end       = 11;
    kctx.a_saved.i           = 10;
    kctx.a_saved.i_vnew      = 10;
    kctx.a_saved.i_max_pdiff = 8;
    kctx.b_saved.i_begin     = 0;
    kctx.b_saved.i_end       = 26;
    kctx.b_saved.i           = 13;
    kctx.b_saved.i_vnew      = 13;
    kctx.b_saved.i_max_pdiff = 9;

    kctx_reset = kctx;

    for (size_t ni_cdf = 0; ni_cdf < n_nested; ni_cdf++) {
        const size_t i_cdf     = n_nested - 1 - ni_cdf;
        const size_t cdf_len_a = cdfs_a[i_cdf].size();
        const size_t cdf_len_b = cdfs_b[i_cdf].size();
        f_dbg_ref << "^^^^^^^^^^^^" << std::endl;
        f_dbg_ref << "i_cdf: " << i_cdf << std::endl;
        f_dbg_ref << "cdf_len_a: " << cdf_len_a << std::endl;
        f_dbg_ref << "cdf_len_b: " << cdf_len_b << std::endl;
        f_dbg_ref << "^^^^^^^^^^^^" << std::endl;
        f_dbg_ctx << "^^^^^^^^^^^^" << std::endl;
        f_dbg_ctx << "i_cdf: " << i_cdf << std::endl;
        f_dbg_ctx << "cdf_len_a: " << cdf_len_a << std::endl;
        f_dbg_ctx << "cdf_len_b: " << cdf_len_b << std::endl;
        f_dbg_ctx << "^^^^^^^^^^^^" << std::endl;
        assert(cdf_len_a >= prev_cdf_len_a);
        assert(cdf_len_b >= prev_cdf_len_b);
        assert(!cdfs_a[i_cdf].empty());
        assert(!cdfs_b[i_cdf].empty());
        prev_cdf_len_a = cdf_len_a;
        prev_cdf_len_b = cdf_len_b;

        for (size_t i = 0; i < cdf_len_a; i++) {
            assert(cdfs_a[i_cdf].sorted_values[i] ==
                   supercdf_a.sorted_values[i]);
        }
        for (size_t i = 0; i < cdf_len_b; i++) {
            assert(cdfs_b[i_cdf].sorted_values[i] ==
                   supercdf_b.sorted_values[i]);
        }

        assert(kctx.a_next.i == kctx_reset.a_next.i);
        assert(kctx.b_next.i == kctx_reset.b_next.i);
        assert(kctx.a_next.i_vnew == kctx_reset.a_next.i_vnew);
        assert(kctx.b_next.i_vnew == kctx_reset.b_next.i_vnew);
        assert(kctx.a_next.i_max_pdiff == kctx_reset.a_next.i_max_pdiff);
        assert(kctx.b_next.i_max_pdiff == kctx_reset.b_next.i_max_pdiff);

        rododendrs::pf_dbg = &f_dbg_ctx;
        kctx.resize(cdf_len_a, cdf_len_b);
        const double kstest1 = rododendrs::kstest(kctx, cdf_len_a, cdf_len_b);
        assert(kctx.a_next.i == kctx.a_next.i_end);
        assert(kctx.b_next.i == kctx.b_next.i_end);
        assert(kctx.max_pdiff == kstest1);

        rododendrs::pf_dbg = &f_dbg_ref;
        kctx_reset.reset();
        const double kstest2 =
                rododendrs::kstest(kctx_reset, cdf_len_a, cdf_len_b);
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

void test_kstest_files_full()
{
    rododendrs::CDF supercdf_a("tests/data/supercdf_a.csv");
    rododendrs::CDF supercdf_b("tests/data/supercdf_b.csv");
    std::vector<rododendrs::CDF> cdfs_a;
    std::vector<rododendrs::CDF> cdfs_b;
    size_t i_cdf = 0;
    while (true) {
        const std::string cdf_a_path =
                "tests/data/cdf_a" + std::to_string(i_cdf) + ".csv";
        const std::string cdf_b_path =
                "tests/data/cdf_b" + std::to_string(i_cdf) + ".csv";
        if (!std::filesystem::exists(cdf_a_path)) {
            break;
        }
        rododendrs::CDF cdf_a(cdf_a_path);
        rododendrs::CDF cdf_b(cdf_b_path);
        cdfs_a.push_back(cdf_a);
        cdfs_b.push_back(cdf_b);
        i_cdf++;
    }

    const size_t n_nested = cdfs_a.size();
    std::ofstream f_dbg_ref("log_ref.txt");
    std::ofstream f_dbg_ctx("log_ctx.txt");
    rododendrs::pf_dbg = &f_dbg_ref;

    // kstest check
    rododendrs::KstestCtx kctx(supercdf_a, supercdf_b);
    rododendrs::KstestCtx kctx_reset(supercdf_a, supercdf_b);
    size_t prev_cdf_len_a = 0;
    size_t prev_cdf_len_b = 0;

    for (size_t ni_cdf = 0; ni_cdf < n_nested; ni_cdf++) {
        i_cdf                  = n_nested - 1 - ni_cdf;
        const size_t cdf_len_a = cdfs_a[i_cdf].size();
        const size_t cdf_len_b = cdfs_b[i_cdf].size();
        f_dbg_ref << "^^^^^^^^^^^^" << std::endl;
        f_dbg_ref << "i_cdf: " << i_cdf << std::endl;
        f_dbg_ref << "cdf_len_a: " << cdf_len_a << std::endl;
        f_dbg_ref << "cdf_len_b: " << cdf_len_b << std::endl;
        f_dbg_ref << "^^^^^^^^^^^^" << std::endl;
        f_dbg_ctx << "^^^^^^^^^^^^" << std::endl;
        f_dbg_ctx << "i_cdf: " << i_cdf << std::endl;
        f_dbg_ctx << "cdf_len_a: " << cdf_len_a << std::endl;
        f_dbg_ctx << "cdf_len_b: " << cdf_len_b << std::endl;
        f_dbg_ctx << "^^^^^^^^^^^^" << std::endl;
        assert(cdf_len_a >= prev_cdf_len_a);
        assert(cdf_len_b >= prev_cdf_len_b);
        assert(!cdfs_a[i_cdf].empty());
        assert(!cdfs_b[i_cdf].empty());
        prev_cdf_len_a = cdf_len_a;
        prev_cdf_len_b = cdf_len_b;

        for (size_t i = 0; i < cdf_len_a; i++) {
            assert(cdfs_a[i_cdf].sorted_values[i] ==
                   supercdf_a.sorted_values[i]);
        }
        for (size_t i = 0; i < cdf_len_b; i++) {
            assert(cdfs_b[i_cdf].sorted_values[i] ==
                   supercdf_b.sorted_values[i]);
        }

        assert(kctx.a_next.i == kctx_reset.a_next.i);
        assert(kctx.b_next.i == kctx_reset.b_next.i);
        assert(kctx.a_next.i_vnew == kctx_reset.a_next.i_vnew);
        assert(kctx.b_next.i_vnew == kctx_reset.b_next.i_vnew);
        assert(kctx.a_next.i_max_pdiff == kctx_reset.a_next.i_max_pdiff);
        assert(kctx.b_next.i_max_pdiff == kctx_reset.b_next.i_max_pdiff);

        rododendrs::pf_dbg = &f_dbg_ctx;
        kctx.resize(cdf_len_a, cdf_len_b);
        const double kstest1 = rododendrs::kstest(kctx, cdf_len_a, cdf_len_b);
        assert(kctx.a_next.i == kctx.a_next.i_end);
        assert(kctx.b_next.i == kctx.b_next.i_end);
        assert(kctx.max_pdiff == kstest1);

        rododendrs::pf_dbg = &f_dbg_ref;
        kctx_reset.reset();
        const double kstest2 =
                rododendrs::kstest(kctx_reset, cdf_len_a, cdf_len_b);
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
    std::cout << "running kstest tests..." << std::endl;

    test_kstest_files_full();
    test_kstest_files_partial();

    std::cout << "all tests passed" << std::endl;
    return 0;
}
