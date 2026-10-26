"""Main figures 2–5, rendered from compact numerical result tables."""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import t as student_t

from .style import (ACC_ORDER, COLORS, DISPLAY, MONTHS, WARM_MODELS,
                    label_panel, plt, save_figure, set_style)
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable


def values(rows, column):
    return np.array([float(row[column]) if row[column] != "" else np.nan
                     for row in rows])


def by_model(rows, model, key="lead"):
    return sorted([row for row in rows if row["model"] == model],
                  key=lambda row: int(row[key]) if key == "lead" else row[key])


def skill_comparison(rows):
    rows = [row for row in rows if 1 <= int(row["lead"]) <= 8]
    set_style(10)
    fig, ax = plt.subplots(figsize=(11.5, 8.3))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.10, top=0.78)
    handles = {}
    for name in ACC_ORDER:
        group = by_model(rows, name)
        if not group:
            continue
        width = 2.7 if name in ("NCM", "MMarkov-4", "MMarkov-3") else 1.65
        color = "#000000" if name == "MMarkov-3" else COLORS[name]
        line, = ax.plot(values(group, "lead"), values(group, "ACC"),
                        color=color, linewidth=width,
                        linestyle="--" if name == "MMarkov-4" else "-",
                        label=name, zorder=6 if name == "NCM" else 3)
        handles[name] = line
    ax.set(xlim=(0.5, 8.5), ylim=(0.5, 1), xlabel="Prediction lead (months)",
           ylabel="3-month running-mean Niño3.4 ACC")
    ax.set_xticks(range(1, 9))
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ncm = by_model(rows, "NCM")
    short = values(ncm[:6], "ACC")
    for lead, (score, y) in enumerate(zip(short, np.arange(0.82, 0.55, -0.045)), 1):
        ax.text(0.75, y, f"L{lead} = {score:.2f}", fontsize=12, ha="left")
    ax.text(0.75, 0.55, f"Mean (L1–L6) = {short.mean():.2f}", fontsize=12)
    order = [name for name in ACC_ORDER if name in handles]
    fig.legend([handles[name] for name in order], order, ncol=4, mode="expand",
               loc="upper left", bbox_to_anchor=(0.08, 0.97, 0.91, 0),
               fontsize=10.5, framealpha=1, borderaxespad=0.3, borderpad=0.6,
               columnspacing=0.65, handlelength=2.2, handletextpad=0.35,
               labelspacing=0.45)
    return fig


def seasonal_skill(rows):
    set_style(14, True)
    leads = np.arange(1, 9)
    months = np.arange(1, 13)
    fields, counts = {}, {}
    for name in ("MMarkov-4", "NCM"):
        fields[name] = np.full((12, 8), np.nan)
        counts[name] = np.zeros((12, 8), dtype=int)
        for row in rows:
            if row["model"] == name:
                m, lead = int(row["initial_month"])-1, int(row["lead"])-1
                fields[name][m, lead] = float(row["ACC"])
                counts[name][m, lead] = int(row["n"])
        if not np.isfinite(fields[name]).all() or (counts[name] < 3).any():
            raise ValueError(f"Incomplete month–lead table for {name}")
    if not np.array_equal(counts["NCM"], counts["MMarkov-4"]):
        raise ValueError("NCM and MMarkov-4 heatmaps need matching sample counts")
    fig = plt.figure(figsize=(16.2, 12.012))
    upper = fig.add_gridspec(1, 3, left=0.075, right=0.985,
                            bottom=7.084/12.012, top=11.34/12.012, wspace=0.28)
    lower = fig.add_gridspec(1, 3, left=0.075, right=0.985,
                            bottom=1.456/12.012, top=4.90/12.012, wspace=0.28)
    colors = ["#fcf6c2", "#f8e398", "#f1c163", "#ea9b49",
              "#e3703f", "#d05138", "#b13937"]
    cmap = ListedColormap(colors)
    cmap.set_under("white")
    cmap.set_over("#ad2d32")
    levels = np.arange(60, 96, 5)/100
    norm = BoundaryNorm(levels, len(colors))
    delta_colors = ["#fcf6c2", "#f9e8a1", "#f3ca6e", "#eeaf4c",
                    "#e78846", "#e1643c", "#cd4d37", "#b13937"]
    delta_map = ListedColormap(delta_colors)
    delta_map.set_under("white")
    delta_map.set_over("#ad2d32")
    delta_levels = np.arange(0, 0.241, 0.04)
    delta_norm = BoundaryNorm(delta_levels, len(delta_colors))
    top_axes = []
    data = [fields["MMarkov-4"], fields["NCM"], fields["NCM"]-fields["MMarkov-4"]]
    titles = ["MMarkov-4", "NCM", "NCM − MMarkov-4"]
    for i, (array, title) in enumerate(zip(data, titles)):
        ax = fig.add_subplot(upper[0, i])
        difference = i == 2
        ax.contourf(leads, months, array,
                    levels=delta_levels if difference else levels,
                    cmap=delta_map if difference else cmap,
                    norm=delta_norm if difference else norm,
                    extend="both", antialiased=False, corner_mask=False)
        ax.set(xlim=(1, 8), ylim=(1, 12), title=title)
        ax.set_xticks(leads)
        ax.set_yticks(months, MONTHS)
        ax.tick_params(labelsize=12, width=1.7, length=5)
        ax.set_title(title, fontsize=17, pad=14)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.1)
        label_panel(fig, ax, "abc"[i], 17)
        top_axes.append(ax)
    top_axes[0].set_ylabel("Initial month", fontsize=18, labelpad=8)
    fig.text(0.53, 6.58/12.012, "Prediction lead (months)",
             ha="center", va="center", fontsize=18)
    cax = fig.add_axes([0.105, 5.9976/12.012, 0.525, 0.1568/12.012])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal", boundaries=levels, ticks=levels,
                      spacing="proportional", extend="both", drawedges=True,
                      extendfrac=0.10)
    cb.ax.set_xticklabels([f"{v:.2f}" for v in levels], fontweight="normal")
    cb.ax.tick_params(length=0, pad=7, labelsize=12)
    cb.set_label("ACC", fontsize=12, labelpad=3)
    dbar = fig.add_axes([0.735, 5.9976/12.012, 0.245, 0.1568/12.012])
    dc = fig.colorbar(ScalarMappable(norm=delta_norm, cmap=delta_map), cax=dbar,
                      orientation="horizontal", boundaries=delta_levels,
                      ticks=delta_levels, spacing="proportional", extend="both",
                      drawedges=True)
    dc.ax.set_xticklabels([f"{v:g}" for v in delta_levels], fontweight="normal")
    dc.ax.tick_params(length=0, pad=7, labelsize=12)
    dc.set_label("ΔACC", fontsize=12, labelpad=3)
    for i, month in enumerate((2, 3, 5)):
        ax = fig.add_subplot(lower[0, i])
        for name in ("NCM", "MMarkov-4"):
            ax.plot(leads, fields[name][month-1], color=COLORS[name],
                    linestyle="-" if name == "NCM" else (0, (5, 3)),
                    linewidth=2.2, marker="o", markersize=3.8,
                    markerfacecolor=COLORS[name] if name == "NCM" else "white",
                    markeredgewidth=1.0, label=name)
        n = counts["NCM"][month-1]
        critical_t = student_t.ppf(0.975, n-2)
        critical_r = critical_t/np.sqrt(critical_t**2+n-2)
        if critical_r.min() >= 0.4:
            ax.plot(leads, critical_r, color="#808080", ls=(0, (5, 3)),
                    lw=1.3, label="95% significance", zorder=1)
        target = (month+leads-1) % 12
        labels = [MONTHS[m] for m in target]
        if month == 5:
            labels[-1] = "Jan(+1)"
        ax.set(xlim=(1, 8.5), ylim=(0.4, 1), ylabel="ACC")
        ax.set_xticks(leads, labels)
        ax.set_yticks(np.arange(0.4, 1.01, 0.1))
        ax.set_title(f"{MONTHS[month-1]} initialization", fontsize=17, pad=14)
        ax.tick_params(labelsize=11, width=1.5, length=4.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.7)
        ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.015),
                  frameon=False, fontsize=11.5, handlelength=2.7,
                  borderaxespad=0.65)
        label_panel(fig, ax, "def"[i], 17)
    fig.text(0.53, 0.0354, "Forecast target month", ha="center",
             va="center", fontsize=18)
    return fig


def warm_events(event_rows, mae_rows):
    set_style(9)
    fig = plt.figure(figsize=(11.4, 8.1), facecolor="white")
    for year, bottom, letters in ((2015, 0.605, "ab"), (2023, 0.14, "cd")):
        ax = fig.add_axes([0.065, bottom, 0.58, 0.28])
        bx = fig.add_axes([0.72, bottom, 0.265, 0.28])
        rows = [row for row in event_rows if int(row["event"]) == year]
        models = [*WARM_MODELS[year], "MMarkov-4", "MMarkov-3", "ORAS5", "NCM"]
        months = sorted({row["target_month"] for row in rows})
        if len(months) != 6:
            raise ValueError(f"Event {year} needs six target months")
        handles, arrays = {}, {}
        for name in models:
            group = by_model(rows, name, "target_month")
            arrays[name] = values(group, "prediction_C")
            style = ("--" if name == "MMarkov-4" else
                     "-." if name == "MMarkov-3" else "-")
            width = 2.15 if name == "NCM" else 1.8 if name == "ORAS5" else 1.25
            line, = ax.plot(range(6), arrays[name], color=COLORS[name], ls=style,
                            lw=width,
                            marker="o", ms=4.8 if name == "NCM" else 3.3,
                            mec="white", mew=0.35,
                            zorder=6 if name == "NCM" else 5 if name == "ORAS5" else 3)
            handles[name] = line
        maximum = max(np.nanmax(a) for a in arrays.values())
        ymax = max(2.42, math.ceil((maximum+0.12)*5)/5)
        ymin = 0.7 if year == 2023 else 1.0
        ax.set(xlim=(-0.35, 5.35), ylim=(ymin, ymax),
               ylabel="Niño3.4 SST anomaly (°C)", xlabel="Verification month")
        ax.set_xticks(range(6), months, fontsize=8)
        ax.set_yticks(np.arange(np.ceil(ymin*2)/2, np.floor(ymax*2)/2+0.01, 0.5))
        offsets = (
            [(0, 8), (-16, 8), (0, -10), (0, -12), (0, 8), (0, 8)]
            if year == 2015 else
            [(0, -10), (0, -20), (0, 6), (0, 13), (0, -10), (10, 0)]
        )
        for i, (value, (dx, dy)) in enumerate(zip(arrays["NCM"], offsets)):
            ax.annotate(f"{value:.2f}", (i, value), xytext=(dx, dy),
                        textcoords="offset points", ha="center",
                        va="bottom" if dy > 0 else "top" if dy < 0 else "center",
                        fontsize=8.2, color=COLORS["NCM"], fontweight="bold", zorder=7,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25})
        ranked = sorted([row for row in mae_rows if int(row["event"]) == year],
                        key=lambda row: int(row["rank"]))
        names = [row["model"] for row in ranked]
        mae = values(ranked, "MAE_C")
        bx.bar(range(len(names)), mae, color=[COLORS[name] for name in names],
               width=0.78, edgecolor="white", linewidth=0.6)
        bx.set_xticks(range(len(names)), [DISPLAY.get(name, name) for name in names],
                      rotation=42, ha="right", fontsize=7.5)
        bx.set_ylim(0, mae.max()*1.27)
        bx.set_ylabel("mean MAE (°C)")
        title = "October–February MAE" if year == 2015 else "September–February MAE"
        bx.set_title(title, fontsize=9.5, pad=7)
        for i, value in enumerate(mae):
            bx.text(i, value+mae.max()*0.025, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=7.5)
        order = ["NCM", "MMarkov-4", "MMarkov-3", *WARM_MODELS[year], "ORAS5"]
        fig.legend([handles[name] for name in order],
                   [DISPLAY.get(name, name) for name in order],
                   ncol=4, loc="lower left", mode="expand",
                   bbox_to_anchor=(0.065, bottom+0.298, 0.58, 0.005),
                   borderaxespad=0, fontsize=7.4, framealpha=1,
                   edgecolor="#C7CBD0", columnspacing=0.55, handlelength=1.6,
                   handletextpad=0.3, borderpad=0.35, labelspacing=0.3)
        label_panel(fig, ax, letters[0])
        label_panel(fig, bx, letters[1])
    return fig


def forecast(rows):
    set_style(9)
    fig, ax = plt.subplots(figsize=(803/96, 472/96))
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.2, top=0.67)
    order = ["NCM", "MMarkov-4", "MMarkov-3", "CFSv2", "CanESM5",
             "NASA GEOS5v2", "NCAR CCSM4", "NCAR CESM1", "GEM5.2 NEMO"]
    months = sorted({row["target_month"] for row in rows})
    if len(months) != 6:
        raise ValueError("The 2026 figure needs six target months")
    handles = {}
    for name in order[3:]+order[1:3]+["NCM"]:
        group = by_model(rows, name, "target_month")
        y = values(group, "prediction_C")
        if y.shape != (6,) or not np.isfinite(y).all():
            raise ValueError(f"Incomplete 2026 trajectory for {name}")
        own = name == "NCM"
        color = "#000000" if name == "MMarkov-3" else COLORS[name]
        line, = ax.plot(range(6), y, color=color,
                        lw=2.15 if own else 1.65 if name in order[1:3] else 1.35,
                        ls="--" if name in order[1:3] else "-", marker="o",
                        ms=5 if own else 3.6, mec="white", mew=0.35,
                        zorder=5 if own else 4)
        handles[name] = line
        if own:
            for i, value in enumerate(y):
                right, below = i == 5, i in (0, 2, 3, 4)
                offset = (12, 0) if right else (0, -8 if below else 6)
                ax.annotate(f"{value:.2f}", (i, value), xytext=offset,
                            textcoords="offset points",
                            ha="left" if right else "center",
                            va="center" if right else "top" if below else "bottom",
                            fontsize=8.5, color=COLORS["NCM"],
                            fontweight="bold", zorder=6,
                            bbox={"facecolor": "white", "edgecolor": "none",
                                  "pad": 0.3})
    ax.set(xlim=(-0.35, 5.35), ylim=(1, 5.55), xlabel="Target month",
           ylabel="Niño3.4 SST anomaly (°C)")
    ax.set_xticks(range(6), months)
    fig.legend([handles[name] for name in order], order, ncol=9,
               loc="lower left", bbox_to_anchor=(0.075, 0.715, 0.895, 0),
               mode="expand", borderaxespad=0, fontsize=6.8,
               framealpha=1, edgecolor="#C7CBD0", columnspacing=0.5,
               handlelength=1.4, handletextpad=0.3, borderpad=0.4)
    return fig


def draw_all(tables, output, dpi=600):
    functions = [lambda: skill_comparison(tables["acc"]),
                 lambda: seasonal_skill(tables["heatmap"]),
                 lambda: warm_events(tables["events"], tables["event_mae"]),
                 lambda: forecast(tables["forecast2026"])]
    for number, draw in enumerate(functions, 2):
        save_figure(draw(), output, number, dpi)
