### Plots

import os
import json
import pickle
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_incumbent(traj, out_dir):
    losses = traj["losses"]
    x = np.arange(1, len(losses) + 1)

    plt.figure()
    plt.plot(x, losses)
    plt.xlabel("Trajectory step")
    plt.ylabel("Best-so-far validation loss")
    plt.title("BOHB Incumbent (Best Loss vs Step)")
    out = os.path.join(out_dir, "plot_incumbent_best_loss.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

def plot_trials_scatter(df, out_dir):
    df2 = df.dropna(subset=["loss"]).copy()
    # sort for nicer x-axis
    df2 = df2.reset_index(drop=True)
    x = np.arange(len(df2))

    plt.figure()
    sc = plt.scatter(x, df2["loss"], c=df2["budget"])
    plt.xlabel("Trial index")
    plt.ylabel("Validation loss")
    plt.title("All Trials: Loss vs Trial Index (colored by budget)")
    plt.colorbar(sc, label="Budget")
    out = os.path.join(out_dir, "plot_trials_loss_scatter.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

def plot_loss_by_budget(df, out_dir):
    df2 = df.dropna(subset=["loss"]).copy()
    budgets = sorted(df2["budget"].unique())
    data = [df2[df2["budget"] == b]["loss"].values for b in budgets]

    plt.figure()
    plt.boxplot(data, labels=[str(b) for b in budgets])
    plt.xlabel("Budget")
    plt.ylabel("Validation loss")
    plt.title("Loss Distribution by Budget")
    out = os.path.join(out_dir, "plot_loss_by_budget_boxplot.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

def plot_hp_vs_loss(df, hp_name, out_dir, logx=False):
    col = f"hp_{hp_name}"
    if col not in df.columns:
        print(f"Skip: {hp_name} not found in CSV")
        return
    df2 = df.dropna(subset=["loss", col]).copy()

    plt.figure()
    if df2[col].dtype == object:
        df2.boxplot(column="loss", by=col)
        plt.suptitle("")
    else:
        plt.scatter(df2[col], df2["loss"])
    plt.xlabel(hp_name)
    plt.ylabel("Validation loss")
    title = f"{hp_name} vs Loss"
    plt.title(title)
    if logx:
        plt.xscale("log")
    out = os.path.join(out_dir, f"plot_hp_{hp_name}_vs_loss.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

def main(saved_dir):
    out_dir = os.path.join(saved_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # load trajectory + runs csv
    traj_path = os.path.join(saved_dir, "bohb_incumbent_trajectory.json")
    csv_path = os.path.join(saved_dir, "bohb_runs.csv")

    with open(traj_path, "r") as f:
        traj = json.load(f)

    df = pd.read_csv(csv_path)

    plot_incumbent(traj, out_dir)
    plot_trials_scatter(df, out_dir)
    plot_loss_by_budget(df, out_dir)

    # hyperparam vs loss plots (BOHB paper style)
    plot_hp_vs_loss(df, "learning_rate", out_dir, logx=True)
    plot_hp_vs_loss(df, "hidden_dim", out_dir, logx=False)
    plot_hp_vs_loss(df, "n_hidden_layers", out_dir, logx=False)
    plot_hp_vs_loss(df, "optimizer", out_dir, logx=False)
    plot_hp_vs_loss(df, "activation", out_dir, logx=False)

    print("\n✅ All plots saved in:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_dir", type=str, required=True, help="Path to bohb_saved_* folder")
    args = parser.parse_args()
    main(args.saved_dir)