"""Reproduce Checkpoint 2 migration-barrier summaries and figures.

This script reads the existing processed CSV without modifying it. It performs
descriptive analysis only; no machine-learning model is fitted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/cathode_migration_barrier_leads.csv"
FIGURE_DIR = ROOT / "results/eda/figures"

ILLINI_BLUE = "#13294B"
ILLINI_ORANGE = "#FF5F05"
LIGHT_BLUE = "#6FA3CF"
GRAY = "#A5A8AA"
ROLE_COLORS = {
    "title_mentions_cathode": ILLINI_ORANGE,
    "unresolved": GRAY,
    "electrolyte_evidence": ILLINI_BLUE,
}

REQUIRED_COLUMNS = {
    "source_record_id",
    "mobile_ion",
    "material_formula",
    "crystal_class",
    "migration_barrier_eV",
    "source_bibtex_key",
    "cathode_role_evidence",
    "framework_key",
    "auxiliary_barrier_training_status",
    "direct_conductivity_training_eligible",
    "original_paper_review_required",
    "hop_length_A",
    "cell_volume_A3",
    "mobile_ion_number_density_cm3",
    "mobile_ion_atomic_fraction",
}


def load_data() -> pd.DataFrame:
    data = pd.read_csv(SOURCE)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data["source_record_id"].duplicated().any():
        raise ValueError("source_record_id is not unique")
    numeric = [
        "migration_barrier_eV",
        "hop_length_A",
        "cell_volume_A3",
        "mobile_ion_number_density_cm3",
        "mobile_ion_atomic_fraction",
    ]
    if data[numeric].isna().any().any():
        raise ValueError("Required numeric analysis fields contain missing values")
    return data


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=ILLINI_BLUE)
    axis.xaxis.label.set_color(ILLINI_BLUE)
    axis.yaxis.label.set_color(ILLINI_BLUE)
    axis.title.set_color(ILLINI_BLUE)


def save_figure(figure: plt.Figure, filename: str) -> None:
    figure.savefig(FIGURE_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(figure)


def figure_histogram(data: pd.DataFrame) -> None:
    values = data["migration_barrier_eV"]
    q1, q3 = values.quantile([0.25, 0.75])
    upper_fence = q3 + 1.5 * (q3 - q1)

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), layout="constrained")
    axes[0].hist(values, bins=np.linspace(0, 9, 37), color=ILLINI_BLUE, alpha=0.88)
    axes[0].axvline(values.median(), color=ILLINI_ORANGE, linewidth=2, label="Median")
    axes[0].axvline(upper_fence, color=GRAY, linewidth=1.6, linestyle="--", label="IQR upper fence")
    axes[0].set(
        xlabel="Migration barrier (eV)",
        ylabel="Migration paths",
        title="All 284 paths",
    )
    axes[0].legend(frameon=False, fontsize=8)

    central = values[values <= 2.6]
    axes[1].hist(central, bins=np.linspace(0, 2.6, 27), color=LIGHT_BLUE, alpha=0.95)
    axes[1].axvline(values.median(), color=ILLINI_ORANGE, linewidth=2)
    axes[1].set(
        xlabel="Migration barrier (eV)",
        ylabel="Migration paths",
        title=f"Central range (n={len(central)})",
    )
    for axis in axes:
        style_axis(axis)
    figure.suptitle(
        "Distribution of calculated migration barriers",
        color=ILLINI_BLUE,
        fontsize=14,
        fontweight="bold",
    )
    save_figure(figure, "01_barrier_histogram.png")


def figure_li_na(data: pd.DataFrame) -> None:
    rng = np.random.default_rng(404)
    ions = ["Li", "Na"]
    arrays = [data.loc[data["mobile_ion"].eq(ion), "migration_barrier_eV"] for ion in ions]

    figure, axis = plt.subplots(figsize=(6.8, 4.7), layout="constrained")
    boxes = axis.boxplot(
        arrays,
        tick_labels=ions,
        patch_artist=True,
        showmeans=True,
        widths=0.5,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": ILLINI_ORANGE},
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": ILLINI_BLUE},
        capprops={"color": ILLINI_BLUE},
        flierprops={"marker": "", "markersize": 0},
    )
    for patch, color in zip(boxes["boxes"], [ILLINI_BLUE, ILLINI_ORANGE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    for position, (ion, values, color) in enumerate(zip(ions, arrays, [ILLINI_BLUE, ILLINI_ORANGE]), start=1):
        jitter = rng.normal(position, 0.055, len(values))
        axis.scatter(jitter, values, s=16, alpha=0.38, color=color, edgecolor="none")
        axis.text(
            position,
            11.0,
            f"n={len(values)}\nmedian={values.median():.3f} eV",
            ha="center",
            va="bottom",
            fontsize=9,
            color=ILLINI_BLUE,
        )
    axis.set_yscale("log")
    axis.set(
        ylabel="Migration barrier (eV, log scale)",
        title="Li and Na barrier distributions",
    )
    axis.set_ylim(0.002, 15)
    style_axis(axis)
    save_figure(figure, "02_li_na_barrier_comparison.png")


def figure_frameworks(data: pd.DataFrame) -> None:
    working = data.assign(
        ion_framework=data["mobile_ion"] + " | " + data["framework_key"]
    )
    counts = working["ion_framework"].value_counts()
    selected = counts.head(12).index
    summary = (
        working.loc[working["ion_framework"].isin(selected)]
        .groupby("ion_framework")["migration_barrier_eV"]
        .agg(["count", "median"])
        .sort_values("median")
    )
    labels = summary.index.tolist()
    arrays = [working.loc[working["ion_framework"].eq(label), "migration_barrier_eV"] for label in labels]
    rng = np.random.default_rng(404)

    figure, axis = plt.subplots(figsize=(10.8, 6.2), layout="constrained")
    boxes = axis.boxplot(
        arrays,
        tick_labels=[f"{label} (n={summary.loc[label, 'count']})" for label in labels],
        orientation="horizontal",
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.8},
        whiskerprops={"color": ILLINI_BLUE},
        capprops={"color": ILLINI_BLUE},
    )
    for patch in boxes["boxes"]:
        patch.set_facecolor(LIGHT_BLUE)
        patch.set_alpha(0.9)
    for position, (label, values) in enumerate(zip(labels, arrays), start=1):
        jitter = rng.normal(position, 0.045, len(values))
        color = ILLINI_ORANGE if label.startswith("Na") else ILLINI_BLUE
        axis.scatter(values, jitter, s=18, alpha=0.58, color=color, edgecolor="none")
    axis.set_xscale("log")
    axis.set(
        xlabel="Migration barrier (eV, log scale)",
        ylabel="Ion and normalized host framework",
        title="Most frequently represented ion–framework groups",
    )
    style_axis(axis)
    save_figure(figure, "03_framework_barrier_distributions.png")


def figure_hop_length(data: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.2), sharey=True, layout="constrained")
    for axis, ion in zip(axes, ["Li", "Na"]):
        subset = data.loc[data["mobile_ion"].eq(ion)]
        for role, role_data in subset.groupby("cathode_role_evidence"):
            axis.scatter(
                role_data["hop_length_A"],
                role_data["migration_barrier_eV"],
                s=34,
                alpha=0.72,
                color=ROLE_COLORS[role],
                label=role.replace("_", " "),
                edgecolor="white",
                linewidth=0.35,
            )
        rho, _ = spearmanr(subset["hop_length_A"], subset["migration_barrier_eV"])
        axis.text(
            0.98,
            0.97,
            f"Spearman ρ={rho:.2f}\nn={len(subset)}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color=ILLINI_BLUE,
            fontsize=9,
        )
        axis.set_yscale("log")
        axis.set(
            xlabel="Hop length (Å)",
            title=ion,
        )
        style_axis(axis)
    axes[0].set_ylabel("Migration barrier (eV, log scale)")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, frameon=False, fontsize=8, loc="lower right")
    figure.suptitle(
        "Migration barrier versus calculated hop length",
        color=ILLINI_BLUE,
        fontsize=14,
        fontweight="bold",
    )
    save_figure(figure, "04_barrier_vs_hop_length.png")


def print_summary(data: pd.DataFrame) -> None:
    print(f"Rows / unique paths: {len(data)} / {data.source_record_id.nunique()}")
    print(f"Unique material formulas: {data.material_formula.nunique()}")
    print(f"Unique framework_key values: {data.framework_key.nunique()}")
    print(
        "Unique ion-framework groups:",
        data[["mobile_ion", "framework_key"]].drop_duplicates().shape[0],
    )
    print(f"Identifiable paper keys: {data.source_bibtex_key.nunique()}")
    print(f"Rows missing paper keys: {data.source_bibtex_key.isna().sum()}")
    print("\nRole classifications")
    print(data["cathode_role_evidence"].value_counts().to_string())
    print("\nTraining-status fields")
    for column in [
        "auxiliary_barrier_training_status",
        "direct_conductivity_training_eligible",
        "original_paper_review_required",
    ]:
        print(f"\n{column}\n{data[column].value_counts(dropna=False).to_string()}")
    print("\nBarrier summaries")
    summaries = []
    for label, subset in [("All", data), *list(data.groupby("mobile_ion"))]:
        summaries.append(
            {
                "group": label,
                "n": len(subset),
                "mean_eV": subset.migration_barrier_eV.mean(),
                "median_eV": subset.migration_barrier_eV.median(),
                "std_eV": subset.migration_barrier_eV.std(),
                "min_eV": subset.migration_barrier_eV.min(),
                "max_eV": subset.migration_barrier_eV.max(),
            }
        )
    print(pd.DataFrame(summaries).to_string(index=False))
    print("\nDescriptive Spearman correlations")
    rows = []
    features = [
        "hop_length_A",
        "cell_volume_A3",
        "mobile_ion_number_density_cm3",
        "mobile_ion_atomic_fraction",
    ]
    for label, subset in [("All", data), *list(data.groupby("mobile_ion"))]:
        for feature in features:
            rho, nominal_p = spearmanr(subset[feature], subset["migration_barrier_eV"])
            rows.append(
                {
                    "group": label,
                    "feature": feature,
                    "n": len(subset),
                    "rho": rho,
                    "nominal_p": nominal_p,
                }
            )
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    print_summary(data)
    figure_histogram(data)
    figure_li_na(data)
    figure_frameworks(data)
    figure_hop_length(data)
    print(f"\nSaved four figures to {FIGURE_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
