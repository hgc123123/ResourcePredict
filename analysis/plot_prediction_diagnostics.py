#!/usr/bin/env python3
"""Render scientific figures from computed aggregates only (SVG, PDF, PNG)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", required=True, type=Path)
    a = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"serif", "font.size":9, "axes.labelsize":9,
                         "svg.fonttype":"none", "pdf.fonttype":42,
                         "axes.spines.top":False, "axes.spines.right":False})
    out = a.analysis_dir / "figures"
    out.mkdir(exist_ok=True)
    def save(fig, name):
        fig.tight_layout()
        for ext in ("svg", "pdf", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    daily = pd.read_csv(a.analysis_dir / "daily_comparisons_per_seed.csv")
    refs = list(daily.reference.unique())
    colors = ["#155A8A", "#B45A18"]
    fig, axs = plt.subplots(2,1,figsize=(7,3.8),sharex=True)
    for ax, resource in zip(axs, ("memory", "vram")):
        for ref, color in zip(refs, colors):
            block = daily[(daily.resource==resource)&(daily.reference==ref)].copy()
            block["date"] = pd.to_datetime(block.day)
            group = block.groupby("date").delta_mae
            mean, low, high = group.mean(), group.min(), group.max()
            ax.plot(mean.index, mean.values, color=color, linewidth=.9, label=ref)
            ax.fill_between(mean.index,low,high,color=color,alpha=.13)
        ax.axhline(0,color="black",linestyle="--",linewidth=.6)
        ax.set_ylabel(resource.upper()+"\nMAE difference")
    axs[0].legend(frameon=False,fontsize=8)
    axs[-1].set_xlabel("Submission day (UTC); band is the range across training seeds")
    save(fig,"paired_daily_error")
    metric = pd.read_csv(a.analysis_dir / "stratified_metrics_summary.csv")
    group = metric[metric.scope=="history_subgroup"]
    names = list(group["group"].unique())
    methods = ["Cascade full", "Cascade no_code", "Hierarchical Same-Script Last-1"]
    fig,axs=plt.subplots(1,2,figsize=(7,3))
    for ax,resource in zip(axs,("memory","vram")):
        for k,method in enumerate(methods):
            data=group[(group.resource==resource)&(group.method==method)].set_index("group").reindex(names)
            ax.bar(np.arange(len(names))+(k-1)*.24,data.mae_mean,width=.24,
                   yerr=data.mae_sd.fillna(0),capsize=2,label=method)
        ax.set_xticks(np.arange(len(names)),[str(n).replace("_","\n") for n in names],fontsize=7)
        ax.set_ylabel(resource.upper()+" MAE")
    axs[0].legend(frameon=False,fontsize=6)
    save(fig,"history_subgroups")
    frozen=pd.read_csv(a.analysis_dir / "frozen_accuracy_summary.csv")
    routing=pd.read_csv(a.analysis_dir / "cache_routing_summary.csv")
    routing["capacity_entries"]=routing.capacity_entries.astype(str)
    fig,axs=plt.subplots(1,2,figsize=(7,2.8))
    route=routing[(routing.variant=="full")&(routing.period=="test")&(routing.capacity_entries=="unbounded")]
    for ax,resource in zip(axs,("memory","vram")):
        data=frozen[(frozen.variant=="full")&(frozen.period=="test")&(frozen.resource==resource)]
        data=data[data.operation.str.startswith("threshold_")].merge(route[["operation","encoder_forward_rate_mean"]],on="operation")
        data=data.sort_values("encoder_forward_rate_mean")
        ax.errorbar(data.encoder_forward_rate_mean*100,data.mae_mean,yerr=data.mae_sd.fillna(0),marker="o",capsize=2,linewidth=.9)
        ax.set_xlabel("Jobs requiring encoder forward (%)")
        ax.set_ylabel(resource.upper()+" MAE")
    save(fig,"frozen_threshold_tradeoff")
    fig,ax=plt.subplots(figsize=(4,2.6))
    for op in ("full","no_bypass_frozen","unconditional_exact_bypass"):
        data=routing[(routing.variant=="full")&(routing.period=="test")&(routing.operation==op)&(routing.capacity_entries!="unbounded")].copy()
        data["capacity"]=pd.to_numeric(data.capacity_entries)
        data=data.sort_values("capacity")
        ax.errorbar(data.capacity,data.encoder_forward_rate_mean*100,yerr=data.encoder_forward_rate_sd.fillna(0)*100,marker="o",capsize=2,label=op)
    ax.set_xlabel("LRU capacity (entries)")
    ax.set_ylabel("Jobs requiring encoder forward (%)")
    ax.legend(frameon=False,fontsize=7)
    save(fig,"bounded_cache")
    print(f"Figures written to {out}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
