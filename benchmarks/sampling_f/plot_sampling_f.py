import argparse
import pandas as pd
import vplot

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-csv", help="path to a .csv input file")
parser.add_argument("--output-file", help="path to an output file")
args = parser.parse_args()

df = pd.read_csv(args.input_csv)
df["runtime_per_sample_ns"] = df["avg_runtime_ns"] / df["samples_n"]
df["samples_part"] = df["samples_n"] / df["population_size"]

subplots = []

col = 1
row = 1

for population_size in set(df["population_size"]):
    color = vplot.Color.RED
    traces = []
    for name in set(df["name"]):
        traces += [
            vplot.Scatter(
                x=df.loc[
                    (df["name"] == name) & (df["population_size"] == population_size)
                ]["samples_part"],
                y=df.loc[
                    (df["name"] == name) & (df["population_size"] == population_size)
                ]["avg_runtime_ns"],
                color=color.value,
                mode="lines+markers",
                marker_symbol=vplot.MarkerSymbol.CIRCLE,
                name=f"{name}, pop={population_size}",
                showlegend=True,
            )
        ]
        color = color.next()

    subplots += [
        vplot.Subplot(
            col=col,
            row=row,
            x_title="samples_part",
            y_title="avg_runtime_ns",
            traces=traces,
        )
    ]

    row += 1

vplot.PlotlyPlot(
    subplots=subplots,
).to_file(args.output_file)
