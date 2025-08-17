import math
import subprocess
import pandas as pd
import scipy


def test_cdf(run_i: int):
    result = subprocess.run(
        "./test_cdf.o",
        shell=True,
        text=True,
        capture_output=True,
    )
    result_float = float(result.stdout.strip())

    pd_a = pd.read_csv("cdf_a.csv", index_col=0)
    pd_b = pd.read_csv("cdf_b.csv", index_col=0)
    pd_common = pd.merge(pd_a, pd_b, on="i", suffixes=("a", "b"))

    # calculate cdfs
    pd_common["cdf_a"] = pd_common["va"].rank(method="max") / len(pd_common)
    pd_common["cdf_b"] = pd_common["vb"].rank(method="max") / len(pd_common)

    result_scipy_full = scipy.stats.ks_2samp(pd_a, pd_b)
    result_scipy = result_scipy_full.statistic[0]
    is_close = math.isclose(result_float, result_scipy, abs_tol=1e-5)
    if not is_close:
        print(f"run_i: {run_i}")
        print(pd_a)
        print(pd_b)
        print(
            f"rododendrs::kstest: {result_float}, scipy.stats.ks_2samp: {result_scipy}"
        )
    assert is_close


print("running cdf tests...")
for i in range(100):
    test_cdf(i)

print("all tests passed")
