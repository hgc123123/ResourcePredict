"""Data-driven vector plots for the evaluation results.

Plots use only aggregate CSVs. No archived numerical values or private paths are
embedded in the drawing code. Figure data are saved beside SVG/PDF/300-dpi PNG.
"""
from __future__ import annotations

import math


METHODS = [
    ("Cascade full", "Full (exact bypass)", "#165B86", "o"),
    ("Cascade no_code", "No-Code (adaptive)", "#AD4627", "s"),
    ("Hierarchical Same-Script Last-1", "Hierarchical Last-1", "#267052", "^"),
    ("Standalone full", "Qwen standalone", "#70448D", "D"),
]


def make_figures(a, read, out, missing):
    import numpy as np
    import pandas as pd
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("Install requirements-analysis.txt for vector plots, or use --no-figures for aggregate tables.") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    out.mkdir(parents=True, exist_ok=True)
    analysis = a.study_root / "train_tail/analysis"
    results = []

    def save(fig, name, data, scope):
        fig.tight_layout()
        files = []
        for ext in ("svg", "pdf", "png"):
            filename = f"{name}.{ext}"
            fig.savefig(out / filename, dpi=300, bbox_inches="tight")
            files.append(filename)
        plt.close(fig)
        filename = name + "_data.csv"
        data.to_csv(out / filename, index=False)
        files.append(filename)
        results.append({"name": name, "files": files, "scope": scope})

    def errorbar(ax, x, mean, sd, **kw):
        ax.errorbar(x, mean, yerr=np.nan_to_num(np.asarray(sd, float), nan=0.0),
                    capsize=2, linewidth=.9, markersize=3.5, **kw)

    daily = read(analysis / "daily_comparisons_per_seed.csv", "Fig. 2 / daily paired VRAM MAE")
    if daily is not None and not daily.empty:
        daily = daily.loc[daily.resource.eq("vram")].copy()
        daily["date"] = pd.to_datetime(daily.day, utc=True)
        refs = [("Hierarchical Same-Script Last-1", "Full minus Hierarchical", "#165B86"),
                ("Cascade no_code", "Full minus No-Code", "#AD4627")]
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
        for ax, (ref, label, color) in zip(axes, refs):
            block = daily.loc[daily.reference.eq(ref)]
            group = block.groupby("date").delta_mae.agg(["mean", "std"])
            sd = group["std"].fillna(0)
            ax.plot(group.index, group["mean"], color=color, linewidth=.8, marker="o", markersize=1.5)
            ax.fill_between(group.index, group["mean"]-sd, group["mean"]+sd, alpha=.16, color=color)
            ax.axhline(0, color="#666666", linestyle="--", linewidth=.6)
            ax.set_xlabel(label + "\nTest submission day (UTC)")
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axes[0].set_ylabel("Daily VRAM MAE difference")
        save(fig, "fig2_paired_daily_vram", daily.drop(columns="date"),
             "Five-seed mean +/- sample SD on each nonempty day; negative favors Full. Daily counts differ.")

    subgroup = read(analysis / "stratified_metrics_summary.csv", "Fig. 3 / history subgroups")
    if subgroup is not None and not subgroup.empty:
        subgroup = subgroup.loc[subgroup.scope.eq("history_subgroup") & subgroup.method.isin([v[0] for v in METHODS])]
        groups = [("same_last1", "Same Last-1"), ("older_same_script", "Older exact script"),
                  ("no_same_script_history", "No exact script"), ("cold_start", "Cold start")]
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharex=True)
        for ax, res in zip(axes, ("memory", "vram")):
            part = subgroup.loc[subgroup.resource.eq(res)]
            labels = []
            for key, label in groups:
                n = part.loc[part.group.eq(key), "n_mean"]
                labels.append(label + (f"\n(n={int(n.iloc[0]):,})" if len(n) else ""))
            for index, (method, label, color, marker) in enumerate(METHODS):
                data = part.loc[part.method.eq(method)].set_index("group").reindex([x[0] for x in groups])
                yy = np.arange(len(groups))+(index-1.5)*.14
                ax.errorbar(data.mae_mean, yy, xerr=data.mae_sd.fillna(0), fmt=marker,
                            color=color, markersize=3, capsize=2, label=label)
            ax.set_yticks(np.arange(len(groups)), labels, fontsize=7.5)
            ax.invert_yaxis()
            ax.set_xlabel(res.capitalize() + " MAE")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .84))
        save(fig, "fig3_history_subgroups", subgroup,
             "Target-independent groups; means +/- sample SD over training seeds. Small cold-start groups limit rankings.")

    coverage = read(analysis / "token_coverage.csv", "Fig. 4 / tokenizer coverage")
    if coverage is not None and not coverage.empty:
        selected = coverage.loc[coverage.scope.isin(["overall", "test"])].copy()
        selected["fully_covered_rate"] = 1-selected.truncated_rate
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
        for scope, color, marker in (("overall", "#165B86", "o"), ("test", "#AD4627", "s")):
            block = selected.loc[selected.scope.eq(scope)].sort_values("max_tokens")
            x = np.arange(len(block))
            axes[0].plot(x, 100*block.truncated_rate, marker=marker, markersize=4, color=color, label=scope.capitalize())
            axes[1].plot(x, 100*block.mean_retained_token_fraction, marker=marker, markersize=4, color=color)
            for xx, row in zip(x, block.itertuples()):
                axes[0].annotate(f"{row.truncated_n:,}", (xx, 100*row.truncated_rate),
                                 xytext=(3, 4 if scope == "test" else -10), textcoords="offset points", fontsize=7, color=color)
        budgets = sorted(selected.max_tokens.unique())
        for ax in axes:
            ax.set_xticks(range(len(budgets)), [f"{int(v):,}"+("*" if v == 4096 else "") for v in budgets])
            ax.set_xlabel("Token budget (* trained configuration)")
            if 4096 in budgets:
                ax.axvspan(budgets.index(4096)-.14, budgets.index(4096)+.14, color="#EEEEEE", zorder=-2)
        axes[0].set_ylabel("Truncated scripts (%)")
        axes[1].set_ylabel("Mean retained-token fraction (%)")
        axes[0].legend(frameon=False)
        save(fig, "fig4_context_coverage", selected,
             "Tokenizer-only alternatives at 2048/8192 tokens; no accuracy equivalence or optimality claim.")

    accuracy = read(analysis / "frozen_accuracy_summary.csv", "Fig. 5 / frozen classifier thresholds")
    routing = read(analysis / "cache_routing_summary.csv", "Fig. 5 / bounded LRU replay")
    if accuracy is not None and routing is not None and not accuracy.empty and not routing.empty:
        aa = accuracy.loc[accuracy.variant.eq("full") & accuracy.period.eq("test") & accuracy.resource.eq("vram")]
        cc = routing.loc[routing.variant.eq("full") & routing.period.eq("test")].copy()
        cc["capacity_entries"] = cc.capacity_entries.astype(str)
        threshold = aa.loc[aa.operation.str.startswith("threshold_")].merge(
            cc.loc[cc.capacity_entries.eq("unbounded"), ["operation", "encoder_forward_rate_mean", "encoder_forward_rate_sd"]], on="operation")
        threshold["threshold"] = threshold.operation.str.replace("threshold_", "", regex=False).astype(float)
        threshold = threshold.sort_values("threshold")
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
        errorbar(axes[0], threshold.encoder_forward_rate_mean*100, threshold.mae_mean, threshold.mae_sd,
                 marker="o", color="#165B86")
        for row in threshold.itertuples():
            if row.threshold in (0, .5, .95, 1.01):
                axes[0].annotate(f"T={row.threshold:g}", (row.encoder_forward_rate_mean*100, row.mae_mean),
                                 xytext=(3, 5), textcoords="offset points", fontsize=7)
        axes[0].set_xlabel("Encoder forwards after reuse (%)")
        axes[0].set_ylabel("VRAM MAE (classifier control)")
        capacities = sorted(set(cc.capacity_entries)-{"unbounded"}, key=float)+["unbounded"]
        for op, label, color, marker in (("no_bypass_frozen", "No bypass", "#AD4627", "s"),
                                         ("full", "Classifier-equipped reference", "#165B86", "o")):
            block = cc.loc[cc.operation.eq(op)].set_index("capacity_entries").reindex(capacities)
            errorbar(axes[1], range(len(capacities)), block.encoder_forward_rate_mean*100,
                     block.encoder_forward_rate_sd*100, marker=marker, color=color, label=label)
        axes[1].set_xticks(range(len(capacities)), ["infinity" if c == "unbounded" else f"{int(float(c)):,}" for c in capacities], rotation=35)
        axes[1].set_xlabel("LRU capacity (entries)")
        axes[1].set_ylabel("Encoder forwards after reuse (%)")
        axes[1].legend(frameon=False, fontsize=6.5)
        save(fig, "fig5_frozen_routing_lru", pd.concat([threshold.assign(panel="threshold"), cc.assign(panel="LRU")], ignore_index=True),
             "Frozen classifier control and offline LRU sensitivity, not measured latency or test-selected policy; sample-SD whiskers.")

    window_parts = []
    for fraction in (600, 800, 1000):
        path = a.windows_root / f"prefix_{fraction:04d}/study/train_tail/analysis/test_metrics_summary.csv"
        frame = read(path, f"Fig. 6 / adaptive-routing prefix {fraction/1000:g}")
        if frame is not None:
            window_parts.append(frame.assign(prefix_fraction=fraction/1000))
    if window_parts:
        windows = pd.concat(window_parts, ignore_index=True)
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
        for ax, res in zip(axes, ("memory", "vram")):
            for method, label, color, marker in METHODS:
                part = windows.loc[windows.resource.eq(res) & windows.method.eq(method)].sort_values("prefix_fraction")
                if part.empty:
                    continue
                label = "Adaptive Full" if method == "Cascade full" else label
                ax.plot(part.prefix_fraction*100, part.mae_mean, marker=marker, color=color,
                        markersize=4, linewidth=.9, label=label)
            ax.set_xlabel("Retained-submission prefix (%)")
            ax.set_ylabel(res.capitalize() + " MAE")
            ax.set_xticks([60, 80, 100])
        axes[0].legend(frameon=False, fontsize=6.5)
        save(fig, "fig6_adaptive_temporal_windows", windows,
             "Seed-42 independent adaptive-routing temporal retrainings; no seed error bars. Not primary fixed exact-bypass retrainings.")

    hardware = read(a.hardware_root / "capacity_metrics_summary.csv", "Fig. 7 / capacity strata")
    if hardware is not None and not hardware.empty:
        selected = hardware.loc[hardware.scope.eq("capacity_band")].copy()
        order = ["up_to_16384", "16384_to_32768", "32768_to_65536", "65536_to_86016", "over_86016", "unmapped_node", "cpu_only"]
        labels = {"up_to_16384": "<=16,384", "16384_to_32768": "(16,384,\n32,768]", "32768_to_65536": "(32,768,\n65,536]",
                  "65536_to_86016": "(65,536,\n86,016]", "over_86016": ">86,016", "unmapped_node": "No map", "cpu_only": "CPU only"}
        names = [("Full Cascade", "Full", "#165B86", "o"), ("No-Code Cascade", "No-Code", "#AD4627", "s"),
                 ("Hierarchical Same-Script Last-1", "Hierarchical", "#267052", "^")]
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.1))
        for ax, res in zip(axes, ("memory", "vram")):
            block = selected.loc[selected.resource.eq(res)]
            groups = [g for g in order if g in set(block.group)]
            ticks = []
            for g in groups:
                n = block.loc[block.group.eq(g), "n"]
                ticks.append(labels.get(g, g)+f"\nn={int(n.iloc[0]):,}")
            for idx, (method, label, color, marker) in enumerate(names):
                part = block.loc[block.method.eq(method)].set_index("group").reindex(groups)
                errorbar(ax, np.arange(len(groups))+(idx-1)*.18, part.mae_mean, part.mae_sample_std,
                         color=color, marker=marker, linestyle="none", label=label)
            ax.set_xticks(range(len(groups)), ticks, fontsize=6.3)
            ax.set_ylabel(res.capitalize() + " MAE")
            ax.set_xlabel("Per-GPU registered capacity")
        axes[0].legend(frameon=False, ncol=3, fontsize=6.5)
        save(fig, "fig7_hardware_capacity", selected,
             "Posthoc frozen-primary-prediction strata; mean +/- sample SD, n counts each job once; unmatched nodes remain in overall metrics.")

    margins = read(analysis / "margin_diagnostics_summary.csv", "Fig. 8 / additive-margin shortfall")
    if margins is not None and not margins.empty:
        margins = margins.loc[margins.subset.eq("all_valid") & margins.method.isin(["Cascade full", "Hierarchical Same-Script Last-1"])]
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
        for ax, res in zip(axes, ("memory", "vram")):
            for method, label, color, marker in (METHODS[0], METHODS[2]):
                part = margins.loc[margins.resource.eq(res) & margins.method.eq(method)].sort_values("margin")
                errorbar(ax, part.margin, part.below_sampled_peak_rate_mean*100,
                         part.below_sampled_peak_rate_sd*100, marker=marker, color=color, label=label)
            ax.set_xlabel("Fixed additive normalized margin")
            ax.set_ylabel(res.capitalize() + " sampled-peak shortfall (%)")
            ax.set_xticks([0, .05, .10, .20])
        axes[0].legend(frameon=False, fontsize=6.7)
        save(fig, "fig8_margin_shortfall", margins,
             "Five-seed means +/- sample SD at prespecified margins; not allocation savings, OOM probabilities, or safety guarantees.")

    online = read(a.online_root / "summary/summary.csv", "Fig. 9 / measured serial prediction path")
    if online is not None and not online.empty:
        cases = [("exact_bypass_lru_1024", "Primary exact bypass / 1,024"),
                 ("full_lru_0", "Classifier reference / no cache"),
                 ("full_lru_128", "Classifier reference / 128"),
                 ("full_lru_1024", "Classifier reference / 1,024"),
                 ("no_bypass_lru_1024", "No bypass / 1,024"),
                 ("no_post_lru_1024", "No post / 1,024"),
                 ("no_trust_lru_1024", "No trust / 1,024")]
        cases = [(case, label) for case, label in cases if case in set(online.case)]
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=True)
        y = np.arange(len(cases))
        def values(metric, field="mean"):
            indexed = online.loc[online.metric.eq(metric)].set_index("case")
            if any(case not in indexed.index for case, _ in cases):
                raise ValueError(f"Online summary lacks {metric} for an expected case")
            return np.array([indexed.loc[case, field] for case, _ in cases], dtype=float)
        axes[0, 0].errorbar(values("all_requests.total_ms.mean"), y,
                             xerr=values("all_requests.total_ms.mean", "sd"), fmt="o", color="#165B86", capsize=2)
        axes[0, 0].set_xlabel("Mean latency (ms), +/- sample SD")
        for metric, label, color, marker in (("p95", "P95", "#165B86", "s"), ("p99", "P99", "#AD4627", "D")):
            axes[0, 1].plot(values(f"all_requests.total_ms.{metric}"), y, marker=marker, linestyle="none", color=color, label=label)
        axes[0, 1].set_xlabel("Mean within-run percentile (ms)")
        axes[0, 1].legend(frameon=False)
        for resource, color, marker in (("memory", "#165B86", "o"), ("vram", "#AD4627", "^")):
            axes[1, 0].plot(values(f"live_accuracy.{resource}.mae"), y, marker=marker,
                            linestyle="none", color=color, label=resource.capitalize())
        axes[1, 0].set_xlabel("Live batch-one MAE")
        axes[1, 0].legend(frameon=False)
        count = values("all_requests.encoder_forward_n")
        rate = values("all_requests.encoder_forward_rate")*100
        axes[1, 1].plot(count, y, "D", color="#267052", markersize=4)
        for yy, nn, rr in zip(y, count, rate):
            axes[1, 1].annotate(f" {nn:,.0f} / {rr:.2f}%", (nn, yy), xytext=(3, 3), textcoords="offset points", fontsize=7)
        axes[1, 1].set_xlabel("Encoder-forward count / rate")
        for ax in axes.flat:
            ax.set_yticks(y, [label for _, label in cases], fontsize=7)
            ax.grid(axis="x", alpha=.18)
        axes[0, 0].invert_yaxis()
        for ax in axes[:, 1]:
            ax.tick_params(labelleft=False)
        save(fig, "fig9_measured_online_path", online.loc[online.case.isin([v[0] for v in cases])],
             "Three technical repeats averaged within seed, then means/sample SD across seeds. Serial in-process path, not deployed scheduling; live accuracy differs from offline batching.")
    return results
