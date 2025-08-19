import pandas as pd
import plotly.graph_objects as go
import scipy

df_a = pd.read_csv("a_next.csv", index_col=0)
df_b = pd.read_csv("b_next.csv", index_col=0)

result_scipy_full = scipy.stats.ks_2samp(df_a, df_b)
print(result_scipy_full)
result_scipy = result_scipy_full.statistic[0]
result_scipy_loc = result_scipy_full.statistic_location[0]
print(f"scipy_kstest: {result_scipy}, loc: {result_scipy_loc}")

# plot
df_a["p"] = (df_a.index + 1) * (1.0 / float(len(df_a)))
df_b["p"] = (df_b.index + 1) * (1.0 / float(len(df_b)))

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df_a["v"],
        y=df_a["p"],
        mode="lines",
        line_shape="hv",
        name="CDF of a",
    )
)
fig.add_trace(
    go.Scatter(
        x=df_b["v"],
        y=df_b["p"],
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
