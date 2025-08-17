import pandas as pd
import plotly.graph_objects as go
import scipy

pd_a = pd.read_csv("a_next.csv", index_col=0)
pd_b = pd.read_csv("b_next.csv", index_col=0)
pd_common = pd.merge(pd_a, pd_b, on="i", suffixes=("a", "b"))
print(pd_common)

# calculate cdfs
pd_common["cdf_a"] = pd_common["va"].rank(method="max") / len(pd_common)
pd_common["cdf_b"] = pd_common["vb"].rank(method="max") / len(pd_common)

ks_stat, p_value = scipy.stats.ks_2samp(pd_common["va"], pd_common["vb"])
print(f"K-S Statistic: {ks_stat}, P-value: {p_value}")

# plot
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=pd_common["va"],
        y=pd_common["cdf_a"],
        mode="lines",
        line_shape="hv",
        name="CDF of a",
    )
)
fig.add_trace(
    go.Scatter(
        x=pd_common["vb"],
        y=pd_common["cdf_b"],
        mode="lines",
        line_shape="hv",
        name="CDF of b",
    )
)
fig.update_layout(
    title="CDFs",
    xaxis_title="Values",
    yaxis_title="CDF",
    legend_title="Legend",
    template="plotly_white",
)
fig.show()
