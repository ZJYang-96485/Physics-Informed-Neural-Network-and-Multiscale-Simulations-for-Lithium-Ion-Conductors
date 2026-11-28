"""Checkpoint 4: grouped XGBoost baseline for cathode migration barriers.

The script reads the existing migration-barrier CSV, selects the 71 provisional
cathode paths, constructs the predeclared feature matrix, evaluates a median
DummyRegressor and a conservative XGBoost model, and writes predictions and up
to three figures. Source data are never modified.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/cathode_migration_barrier_leads.csv"
OUTPUT_DIR = ROOT / "results/ml"
FIGURE_DIR = OUTPUT_DIR / "figures"
PREDICTION_PATH = OUTPUT_DIR / "checkpoint4_cross_validated_predictions.csv"

RANDOM_SEED = 404
ILLINI_BLUE = "#13294B"
ILLINI_ORANGE = "#FF5F05"
LIGHT_BLUE = "#6FA3CF"
GRAY = "#A5A8AA"

MODEL_CONFIG = {
    "objective": "reg:squarederror",
    "n_estimators": 120,
    "learning_rate": 0.03,
    "max_depth": 2,
    "min_child_weight": 3,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.10,
    "reg_lambda": 5.0,
    "gamma": 0.0,
    "tree_method": "hist",
    "importance_type": "gain",
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
    "verbosity": 0,
}


@dataclass
class UnionFind:
    parent: dict[str, str] = field(default_factory=dict)

    def find(self, node: str) -> str:
        self.parent.setdefault(node, node)
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def parse_framework_key(value: str) -> dict[str, float]:
    parsed = {}
    for item in value.split("|"):
        element, fraction = item.split(":")
        parsed[element] = float(fraction)
    return parsed


def add_connected_groups(data: pd.DataFrame) -> pd.DataFrame:
    union_find = UnionFind()
    framework_nodes = []
    for row in data.itertuples(index=False):
        if pd.isna(row.source_bibtex_key):
            raise ValueError("Dataset A unexpectedly contains a missing source paper key")
        framework_node = f"F:{row.mobile_ion}|{row.framework_key}"
        paper_node = f"P:{row.source_bibtex_key}"
        union_find.union(framework_node, paper_node)
        framework_nodes.append(framework_node)

    labeled = data.copy()
    labeled["component_root"] = [union_find.find(node) for node in framework_nodes]
    component_rows = []
    for root, group in labeled.groupby("component_root"):
        frameworks = sorted(
            (group["mobile_ion"] + " | " + group["framework_key"]).unique()
        )
        component_rows.append(
            {
                "component_root": root,
                "n_paths": len(group),
                "frameworks": "; ".join(frameworks),
            }
        )
    components = pd.DataFrame(component_rows).sort_values(
        ["n_paths", "frameworks"], ascending=[False, True]
    )
    components["connected_group"] = [
        f"A-C{number:02d}" for number in range(1, len(components) + 1)
    ]
    root_to_id = components.set_index("component_root")["connected_group"]
    labeled["connected_group"] = labeled["component_root"].map(root_to_id)
    return labeled.drop(columns="component_root")


def prepare_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    source = pd.read_csv(SOURCE)
    data = source.loc[
        source["auxiliary_barrier_training_status"].eq("provisional_cathode_role")
    ].copy()
    if len(data) != 71:
        raise ValueError(f"Expected 71 provisional cathode paths, found {len(data)}")
    if not data["cathode_role_evidence"].eq("title_mentions_cathode").all():
        raise ValueError("Dataset A contains an unexpected cathode-role classification")
    if data["source_record_id"].duplicated().any():
        raise ValueError("source_record_id is not unique")

    data = add_connected_groups(data).reset_index(drop=True)
    if data["connected_group"].nunique() != 11:
        raise ValueError("Expected 11 connected groups")

    parsed = data["framework_key"].map(parse_framework_key)
    elements = sorted({element for composition in parsed for element in composition})
    if len(elements) != 14:
        raise ValueError(f"Expected 14 Dataset A host elements, found {len(elements)}")

    features = pd.DataFrame(index=data.index)
    features["mobile_ion_Na"] = data["mobile_ion"].eq("Na").astype(float)
    features["hop_length_A"] = data["hop_length_A"].astype(float)
    features["mobile_ion_density_1e22_cm3"] = (
        data["mobile_ion_number_density_cm3"].astype(float) / 1.0e22
    )
    for element in elements:
        features[f"host_fraction_{element}"] = parsed.map(
            lambda composition, symbol=element: composition.get(symbol, 0.0)
        )

    target = data["migration_barrier_eV"].astype(float)
    if features.isna().any().any() or target.isna().any():
        raise ValueError("Prepared model data contain missing values")
    if not np.isfinite(features.to_numpy()).all() or not np.isfinite(target).all():
        raise ValueError("Prepared model data contain non-finite values")
    return data, features, target, features.columns.tolist()


def make_xgboost() -> XGBRegressor:
    return XGBRegressor(**MODEL_CONFIG)


def metric_row(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    result = {
        "mae_eV": float(mean_absolute_error(actual, predicted)),
        "rmse_eV": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": np.nan,
    }
    if len(actual) >= 2 and not np.isclose(np.var(actual), 0.0):
        result["r2"] = float(r2_score(actual, predicted))
    return result


def evaluate_three_fold(
    data: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = GroupKFold(n_splits=3)
    groups = data["connected_group"]
    predictions = pd.DataFrame(index=data.index)
    fold_rows = []
    importance_rows = []

    for fold_number, (train_index, test_index) in enumerate(
        splitter.split(features, target, groups), start=1
    ):
        train_groups = set(groups.iloc[train_index])
        test_groups = set(groups.iloc[test_index])
        if train_groups.intersection(test_groups):
            raise RuntimeError("Connected-group leakage detected in three-fold CV")

        x_train = features.iloc[train_index]
        x_test = features.iloc[test_index]
        y_train = target.iloc[train_index]
        y_test = target.iloc[test_index]

        model = make_xgboost()
        dummy = DummyRegressor(strategy="median")
        model.fit(x_train, y_train)
        dummy.fit(x_train, y_train)
        xgb_prediction = model.predict(x_test)
        dummy_prediction = dummy.predict(x_test)

        predictions.loc[test_index, "threefold_fold"] = fold_number
        predictions.loc[test_index, "xgb_pred_3fold_eV"] = xgb_prediction
        predictions.loc[test_index, "dummy_pred_3fold_eV"] = dummy_prediction

        xgb_metrics = metric_row(y_test.to_numpy(), xgb_prediction)
        dummy_metrics = metric_row(y_test.to_numpy(), dummy_prediction)
        fold_rows.append(
            {
                "fold": fold_number,
                "n_train": len(train_index),
                "n_test": len(test_index),
                "n_train_groups": len(train_groups),
                "n_test_groups": len(test_groups),
                "test_groups": ";".join(sorted(test_groups)),
                "test_Li": int(data.iloc[test_index]["mobile_ion"].eq("Li").sum()),
                "test_Na": int(data.iloc[test_index]["mobile_ion"].eq("Na").sum()),
                **{f"xgb_{key}": value for key, value in xgb_metrics.items()},
                **{f"dummy_{key}": value for key, value in dummy_metrics.items()},
            }
        )

        permutation = permutation_importance(
            model,
            x_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=30,
            random_state=RANDOM_SEED + fold_number,
            n_jobs=1,
        )
        for feature, mean_importance, repeat_std in zip(
            features.columns,
            permutation.importances_mean,
            permutation.importances_std,
        ):
            importance_rows.append(
                {
                    "fold": fold_number,
                    "n_test": len(test_index),
                    "feature": feature,
                    "increase_in_mae_eV": float(mean_importance),
                    "repeat_std_eV": float(repeat_std),
                }
            )

    required = ["xgb_pred_3fold_eV", "dummy_pred_3fold_eV", "threefold_fold"]
    if predictions[required].isna().any().any():
        raise RuntimeError("Three-fold CV did not predict every row exactly once")
    return predictions, pd.DataFrame(fold_rows), pd.DataFrame(importance_rows)


def evaluate_logo(
    data: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = LeaveOneGroupOut()
    groups = data["connected_group"]
    predictions = pd.DataFrame(index=data.index)
    fold_rows = []

    for train_index, test_index in splitter.split(features, target, groups):
        test_group_values = sorted(groups.iloc[test_index].unique())
        if len(test_group_values) != 1:
            raise RuntimeError("LOGO fold contains more than one connected group")
        test_group = test_group_values[0]
        if test_group in set(groups.iloc[train_index]):
            raise RuntimeError("Connected-group leakage detected in LOGO")

        model = make_xgboost()
        dummy = DummyRegressor(strategy="median")
        model.fit(features.iloc[train_index], target.iloc[train_index])
        dummy.fit(features.iloc[train_index], target.iloc[train_index])
        xgb_prediction = model.predict(features.iloc[test_index])
        dummy_prediction = dummy.predict(features.iloc[test_index])

        predictions.loc[test_index, "xgb_pred_logo_eV"] = xgb_prediction
        predictions.loc[test_index, "dummy_pred_logo_eV"] = dummy_prediction
        xgb_metrics = metric_row(target.iloc[test_index].to_numpy(), xgb_prediction)
        dummy_metrics = metric_row(target.iloc[test_index].to_numpy(), dummy_prediction)
        fold_rows.append(
            {
                "held_out_group": test_group,
                "n_train": len(train_index),
                "n_test": len(test_index),
                "test_Li": int(data.iloc[test_index]["mobile_ion"].eq("Li").sum()),
                "test_Na": int(data.iloc[test_index]["mobile_ion"].eq("Na").sum()),
                **{f"xgb_{key}": value for key, value in xgb_metrics.items()},
                **{f"dummy_{key}": value for key, value in dummy_metrics.items()},
            }
        )

    required = ["xgb_pred_logo_eV", "dummy_pred_logo_eV"]
    if predictions[required].isna().any().any():
        raise RuntimeError("LOGO did not predict every row exactly once")
    return predictions, pd.DataFrame(fold_rows).sort_values("held_out_group")


def aggregate_importance(importance_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, group in importance_rows.groupby("feature"):
        weights = group["n_test"].to_numpy(dtype=float)
        values = group["increase_in_mae_eV"].to_numpy(dtype=float)
        weighted_mean = float(np.average(values, weights=weights))
        weighted_std = float(
            np.sqrt(np.average((values - weighted_mean) ** 2, weights=weights))
        )
        rows.append(
            {
                "feature": feature,
                "weighted_mean_increase_in_mae_eV": weighted_mean,
                "across_fold_std_eV": weighted_std,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "weighted_mean_increase_in_mae_eV", ascending=False
    )


def summarize_subsets(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ion_rows = []
    for ion, group in predictions.groupby("mobile_ion"):
        for model in ["xgb", "dummy"]:
            actual = group["actual_barrier_eV"].to_numpy()
            predicted = group[f"{model}_pred_3fold_eV"].to_numpy()
            ion_rows.append(
                {
                    "mobile_ion": ion,
                    "model": model,
                    "n": len(group),
                    **metric_row(actual, predicted),
                    "mean_residual_pred_minus_actual_eV": float(
                        np.mean(predicted - actual)
                    ),
                }
            )

    group_rows = []
    for connected_group, group in predictions.groupby("connected_group"):
        actual = group["actual_barrier_eV"].to_numpy()
        xgb_prediction = group["xgb_pred_3fold_eV"].to_numpy()
        dummy_prediction = group["dummy_pred_3fold_eV"].to_numpy()
        group_rows.append(
            {
                "connected_group": connected_group,
                "n": len(group),
                "ions": ",".join(sorted(group["mobile_ion"].unique())),
                "actual_mean_eV": float(np.mean(actual)),
                "xgb_mae_eV": float(mean_absolute_error(actual, xgb_prediction)),
                "dummy_mae_eV": float(mean_absolute_error(actual, dummy_prediction)),
                "xgb_mean_residual_eV": float(np.mean(xgb_prediction - actual)),
            }
        )
    return pd.DataFrame(ion_rows), pd.DataFrame(group_rows).sort_values(
        "xgb_mae_eV", ascending=False
    )


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=ILLINI_BLUE)
    axis.xaxis.label.set_color(ILLINI_BLUE)
    axis.yaxis.label.set_color(ILLINI_BLUE)
    axis.title.set_color(ILLINI_BLUE)


def plot_actual_vs_predicted(predictions: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 5.6), layout="constrained")
    colors = {"Li": ILLINI_BLUE, "Na": ILLINI_ORANGE}
    for ion, group in predictions.groupby("mobile_ion"):
        axis.scatter(
            group["actual_barrier_eV"],
            group["xgb_pred_3fold_eV"],
            s=48,
            alpha=0.78,
            color=colors[ion],
            label=f"XGBoost {ion} (n={len(group)})",
            edgecolor="white",
            linewidth=0.5,
        )
    axis.scatter(
        predictions["actual_barrier_eV"],
        predictions["dummy_pred_3fold_eV"],
        s=28,
        alpha=0.38,
        color=GRAY,
        marker="x",
        label="Fold-median dummy",
    )
    dominant = predictions["connected_group"].eq("A-C01")
    axis.scatter(
        predictions.loc[dominant, "actual_barrier_eV"],
        predictions.loc[dominant, "xgb_pred_3fold_eV"],
        s=95,
        facecolor="none",
        edgecolor="black",
        linewidth=0.8,
        label="A-C01 outline",
    )
    lower = min(predictions["actual_barrier_eV"].min(), predictions["xgb_pred_3fold_eV"].min())
    upper = max(predictions["actual_barrier_eV"].max(), predictions["xgb_pred_3fold_eV"].max())
    padding = 0.06 * (upper - lower)
    axis.plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        linestyle="--",
        linewidth=1.2,
        color=GRAY,
    )
    axis.set(
        xlabel="Calculated migration barrier (eV)",
        ylabel="Cross-validated prediction (eV)",
        title="Three-fold connected-group validation",
    )
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    style_axis(axis)
    figure.savefig(
        FIGURE_DIR / "01_cross_validated_predicted_vs_actual.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_residuals(predictions: pd.DataFrame) -> None:
    xgb_residual = predictions["xgb_pred_3fold_eV"] - predictions["actual_barrier_eV"]
    dummy_residual = predictions["dummy_pred_3fold_eV"] - predictions["actual_barrier_eV"]
    bins = np.linspace(
        min(xgb_residual.min(), dummy_residual.min()) - 0.05,
        max(xgb_residual.max(), dummy_residual.max()) + 0.05,
        18,
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), layout="constrained")
    axes[0].hist(
        dummy_residual,
        bins=bins,
        color=GRAY,
        alpha=0.55,
        label="Dummy",
    )
    axes[0].hist(
        xgb_residual,
        bins=bins,
        color=ILLINI_BLUE,
        alpha=0.72,
        label="XGBoost",
    )
    axes[0].axvline(0, color=ILLINI_ORANGE, linewidth=1.8)
    axes[0].set(
        xlabel="Residual: prediction − calculation (eV)",
        ylabel="Paths",
        title="All three-fold predictions",
    )
    axes[0].legend(frameon=False)

    arrays = [
        xgb_residual[predictions["mobile_ion"].eq("Li")],
        xgb_residual[predictions["mobile_ion"].eq("Na")],
    ]
    boxes = axes[1].boxplot(
        arrays,
        tick_labels=["Li", "Na"],
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": ILLINI_ORANGE,
        },
        medianprops={"color": "white", "linewidth": 2},
    )
    for patch, color in zip(boxes["boxes"], [ILLINI_BLUE, ILLINI_ORANGE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    axes[1].axhline(0, color=GRAY, linewidth=1.3, linestyle="--")
    axes[1].set(
        xlabel="Mobile ion",
        ylabel="XGBoost residual (eV)",
        title="Residuals by ion",
    )
    for axis in axes:
        style_axis(axis)
    figure.suptitle(
        "Residual distributions",
        color=ILLINI_BLUE,
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(
        FIGURE_DIR / "02_residual_distribution.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_importance(importance: pd.DataFrame) -> None:
    selected = importance.head(12).sort_values(
        "weighted_mean_increase_in_mae_eV", ascending=True
    )
    labels = [
        label.replace("host_fraction_", "host fraction ")
        .replace("mobile_ion_density_1e22_cm3", "nominal ion density")
        .replace("mobile_ion_Na", "mobile ion = Na")
        .replace("hop_length_A", "hop length")
        .replace("_", " ")
        for label in selected["feature"]
    ]
    colors = [
        ILLINI_ORANGE if value > 0 else GRAY
        for value in selected["weighted_mean_increase_in_mae_eV"]
    ]

    figure, axis = plt.subplots(figsize=(8.0, 5.3), layout="constrained")
    axis.barh(
        labels,
        selected["weighted_mean_increase_in_mae_eV"],
        xerr=selected["across_fold_std_eV"],
        color=colors,
        alpha=0.88,
        error_kw={"ecolor": ILLINI_BLUE, "elinewidth": 1, "capsize": 2},
    )
    axis.axvline(0, color=ILLINI_BLUE, linewidth=1)
    axis.set(
        xlabel="Validation MAE increase when permuted (eV)",
        ylabel="Feature",
        title="Three-fold permutation importance",
    )
    style_axis(axis)
    figure.savefig(
        FIGURE_DIR / "03_permutation_importance.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def print_metrics(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    logo_metrics: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    print("Source SHA256:", hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    print("Package versions:", {"xgboost": xgboost.__version__, "sklearn": sklearn.__version__})
    print("Model configuration:")
    print(json.dumps(MODEL_CONFIG, indent=2, sort_keys=True))
    print("\nThree-fold aggregate")
    for model in ["xgb", "dummy"]:
        print(
            model,
            metric_row(
                predictions["actual_barrier_eV"].to_numpy(),
                predictions[f"{model}_pred_3fold_eV"].to_numpy(),
            ),
        )
    print("\nThree-fold metrics")
    print(fold_metrics.to_string(index=False))
    print("\nLOGO aggregate")
    for model in ["xgb", "dummy"]:
        print(
            model,
            metric_row(
                predictions["actual_barrier_eV"].to_numpy(),
                predictions[f"{model}_pred_logo_eV"].to_numpy(),
            ),
        )
    print("\nLOGO metrics")
    print(logo_metrics.to_string(index=False))

    ion_metrics, group_metrics = summarize_subsets(predictions)
    print("\nThree-fold metrics by ion")
    print(ion_metrics.to_string(index=False))
    print("\nThree-fold metrics by connected group")
    print(group_metrics.to_string(index=False))

    dominant = predictions["connected_group"].eq("A-C01")
    print("\nDominant-group influence")
    for label, mask in [("A-C01", dominant), ("without_A-C01", ~dominant)]:
        actual = predictions.loc[mask, "actual_barrier_eV"].to_numpy()
        print(label, "n=", int(mask.sum()))
        for model in ["xgb", "dummy"]:
            predicted = predictions.loc[mask, f"{model}_pred_3fold_eV"].to_numpy()
            print(model, metric_row(actual, predicted))
    print("\nPermutation importance")
    print(importance.to_string(index=False))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data, features, target, feature_names = prepare_dataset()
    print(
        f"Prepared {len(data)} records, {len(feature_names)} features, "
        f"{int(features.isna().sum().sum())} missing feature values, "
        f"{data.connected_group.nunique()} connected groups"
    )
    print("Features:", feature_names)

    threefold_predictions, fold_metrics, importance_rows = evaluate_three_fold(
        data, features, target
    )
    logo_predictions, logo_metrics = evaluate_logo(data, features, target)

    predictions = data[
        [
            "source_record_id",
            "mobile_ion",
            "material_formula",
            "framework_key",
            "crystal_class",
            "source_title",
            "source_bibtex_key",
            "source_dataset_doi",
            "connected_group",
            "hop_length_A",
            "mobile_ion_number_density_cm3",
            "migration_barrier_eV",
        ]
    ].copy()
    predictions = predictions.rename(
        columns={"migration_barrier_eV": "actual_barrier_eV"}
    )
    predictions = predictions.join(threefold_predictions).join(logo_predictions)
    predictions["xgb_residual_3fold_eV"] = (
        predictions["xgb_pred_3fold_eV"] - predictions["actual_barrier_eV"]
    )
    predictions["dummy_residual_3fold_eV"] = (
        predictions["dummy_pred_3fold_eV"] - predictions["actual_barrier_eV"]
    )
    predictions["xgb_residual_logo_eV"] = (
        predictions["xgb_pred_logo_eV"] - predictions["actual_barrier_eV"]
    )
    predictions["dummy_residual_logo_eV"] = (
        predictions["dummy_pred_logo_eV"] - predictions["actual_barrier_eV"]
    )
    predictions["checkpoint2_high_barrier_review"] = predictions[
        "actual_barrier_eV"
    ].gt(1.604375)
    predictions.to_csv(PREDICTION_PATH, index=False, float_format="%.8g")

    importance = aggregate_importance(importance_rows)
    plot_actual_vs_predicted(predictions)
    plot_residuals(predictions)
    plot_importance(importance)
    print_metrics(predictions, fold_metrics, logo_metrics, importance)
    print(f"\nSaved predictions to {PREDICTION_PATH.relative_to(ROOT)}")
    print(f"Saved three figures to {FIGURE_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
