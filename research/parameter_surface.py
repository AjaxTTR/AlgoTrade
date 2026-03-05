"""Visualize optimizer results as a parameter surface heatmap."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_surface(results, param_x="orb_minutes", param_y="tp_atr_multiple", metric="sharpe"):
    pivot = results.pivot_table(
        values=metric,
        index=param_x,
        columns=param_y,
        aggfunc="mean",
    )

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
    )

    plt.title(f"{metric.upper()} Parameter Surface")
    plt.xlabel(param_y)
    plt.ylabel(param_x)

    plt.show()


if __name__ == "__main__":
    results = pd.read_csv("research/results.csv")

    plot_surface(
        results,
        param_x="orb_minutes",
        param_y="tp_atr_multiple",
        metric="sharpe",
    )
