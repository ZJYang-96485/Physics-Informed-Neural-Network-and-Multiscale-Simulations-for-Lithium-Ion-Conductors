#!/usr/bin/env python3
"""Checkpoint 8: lightweight UMA path/descriptor pilot without model training.

The locally acquired Figshare energy table contains seven scalar energies per
path but not the corresponding atomic structure images. This script therefore:

1. reconstructs the energy profile, forward barrier, KRA barrier, and
   maximum-energy image for five fixed Li/Na paths;
2. matches those paths to the authors' S1 descriptor table using ICSD ID and
   the rounded forward/KRA barrier pair; and
3. leaves every structure-dependent field missing when the structure archive
   is not locally available.

No missing value is imputed and no predictive model is trained.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENERGY = ROOT / "data/raw/computational_transport/UMA_NEB_energy.csv"
DEFAULT_SUPPLEMENT = (
    ROOT / "data/raw/computational_transport/ADMA-38-e75410-s004.xlsx"
)
DEFAULT_OUTPUT = ROOT / "results/ml/physics_descriptor_pilot.csv"
DEFAULT_FIGURE = ROOT / "results/ml/figures/checkpoint8_energy_profiles.png"

# Selected after an all-path audit. The set contains Li and Na paths in oxide,
# sulfide, fluoride, and bromide hosts, and includes peak images at 3, 4, and 5.
SELECTED_PATHS = [
    (39648, 19),   # LiTaSiO5, oxide
    (73276, 6),    # Li2MgBr4, bromide
    (59023, 13),   # Na2ZnSnS4, sulfide
    (260875, 22),  # Na3TaF8, fluoride
    (45802, 52),   # NaNbWO6, oxide
]

ANION_SYMBOLS = {
    "O", "S", "F", "Cl", "Br", "I", "N", "P", "Se", "Te", "As", "B", "C"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy", type=Path, default=DEFAULT_ENERGY)
    parser.add_argument("--supplement", type=Path, default=DEFAULT_SUPPLEMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def read_energy_profiles(path: Path) -> tuple[pd.DataFrame, dict[tuple[int, int], pd.DataFrame]]:
    raw = pd.read_csv(path)
    required = {"St_list", "UMA-energy"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Energy CSV is missing {sorted(required - set(raw.columns))}")

    parsed = raw["St_list"].str.extract(
        r"^(?P<structure_id>\d+)_(?P<migration_path_id>\d+)_(?P<image_index>\d+)_relax$"
    )
    if parsed.isna().any().any():
        bad = raw.loc[parsed.isna().any(axis=1), "St_list"].head().tolist()
        raise ValueError(f"Could not parse trajectory names: {bad}")

    parsed = parsed.astype(int)
    raw = pd.concat([raw, parsed], axis=1)
    profiles: dict[tuple[int, int], pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for key, group in raw.groupby(["structure_id", "migration_path_id"], sort=False):
        group = group.sort_values("image_index").reset_index(drop=True)
        if group["image_index"].tolist() != list(range(7)):
            raise ValueError(f"Path {key} does not contain images 0..6 exactly once")
        energies = group["UMA-energy"].to_numpy(dtype=float)
        ts_offset = int(np.argmax(energies))
        ts_image = int(group.loc[ts_offset, "image_index"])
        profiles[key] = group
        rows.append(
            {
                "structure_id": int(key[0]),
                "migration_path_id": int(key[1]),
                "image_count": len(group),
                "max_energy_image": ts_image,
                "energy_initial_eV": energies[0],
                "energy_final_eV": energies[-1],
                "energy_max_eV": energies[ts_offset],
                "forward_barrier_recomputed_eV": energies[ts_offset] - energies[0],
                "kra_barrier_recomputed_eV": (
                    energies[ts_offset] - (energies[0] + energies[-1]) / 2
                ),
                "endpoint_energy_asymmetry_eV": energies[-1] - energies[0],
                "energy_profile_relative_to_initial_eV": ";".join(
                    f"{value - energies[0]:.6f}" for value in energies
                ),
            }
        )
    return pd.DataFrame(rows), profiles


def read_supplement(path: Path) -> pd.DataFrame:
    parts = []
    for mobile_ion in ("Li", "Na"):
        frame = pd.read_excel(path, sheet_name=mobile_ion)
        frame["mobile_ion"] = mobile_ion
        frame["supplement_row_1based"] = np.arange(2, len(frame) + 2)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def host_anion_family(formula: str) -> str:
    symbols = re.findall(r"[A-Z][a-z]?", formula)
    present = [symbol for symbol in symbols if symbol in ANION_SYMBOLS]
    priority = ["O", "S", "Se", "F", "Cl", "Br", "I", "N"]
    selected = [symbol for symbol in priority if symbol in present]
    return "+".join(selected) if selected else "unresolved"


def match_source_descriptor(
    path_row: pd.Series,
    supplement: pd.DataFrame,
    tolerance_eV: float = 0.003,
) -> pd.Series:
    candidates = supplement.loc[
        supplement["ICSD"].eq(path_row["structure_id"])
    ].copy()
    if candidates.empty:
        raise ValueError(f"No supplement rows for ICSD {path_row['structure_id']}")

    candidates["match_error_eV"] = (
        (candidates["Ea"] - path_row["forward_barrier_recomputed_eV"]).abs()
        + (candidates["Ea_KRA"] - path_row["kra_barrier_recomputed_eV"]).abs()
    )
    candidates = candidates.sort_values("match_error_eV")
    best = candidates.iloc[0]
    if float(best["match_error_eV"]) > tolerance_eV:
        raise ValueError(
            f"No rounded barrier match for ICSD/path "
            f"{path_row['structure_id']}/{path_row['migration_path_id']}; "
            f"best error={best['match_error_eV']:.6f} eV"
        )
    return best


def build_pilot(energy_path: Path, supplement_path: Path) -> tuple[pd.DataFrame, dict]:
    all_paths, profiles = read_energy_profiles(energy_path)
    supplement = read_supplement(supplement_path)

    selected_rows = []
    for structure_id, migration_path_id in SELECTED_PATHS:
        match = all_paths.loc[
            all_paths["structure_id"].eq(structure_id)
            & all_paths["migration_path_id"].eq(migration_path_id)
        ]
        if len(match) != 1:
            raise ValueError(f"Expected one energy path for {(structure_id, migration_path_id)}")
        path_row = match.iloc[0]
        source = match_source_descriptor(path_row, supplement)

        record = path_row.to_dict()
        record.update(
            {
                "formula": source["Composition"],
                "mobile_ion_species": source["mobile_ion"],
                "host_anion_family": host_anion_family(str(source["Composition"])),
                "source_forward_barrier_eV": source["Ea"],
                "source_kra_barrier_eV": source["Ea_KRA"],
                "source_descriptor_match_error_eV": source["match_error_eV"],
                "supplement_sheet": source["mobile_ion"],
                "supplement_row_1based": int(source["supplement_row_1based"]),
                "geometric_midpoint_image": 3,
                "max_energy_is_geometric_midpoint": int(record["max_energy_image"]) == 3,
                # These are verified source descriptors evaluated at the UMA TS.
                "distance_contraction_total_A": source["distance_deviation-TS"],
                "distance_contraction_anion_A": source["distance_deviation_anion-TS"],
                "distance_contraction_cation_A": source["distance_deviation_cation-TS"],
                "bottleneck_radius_ts_A": source["bottleneck_radius-TS"],
                "coordination_kra_source": source["CN-KRA"],
                "voronoi_volume_kra_source_A3": source["Voronoi volume-KA"],
                # Structure-dependent values are deliberately missing because the
                # locally acquired source folder has energies but no image structures.
                "initial_mobile_ion_position_frac": pd.NA,
                "final_mobile_ion_position_frac": pd.NA,
                "hop_distance_A": np.nan,
                "initial_coordination": np.nan,
                "final_coordination": np.nan,
                "minimum_coordination_along_path": np.nan,
                "minimum_anion_distance_A": np.nan,
                "minimum_path_voronoi_volume_A3": np.nan,
                "atomic_order_consistency_verified": False,
                "structure_images_available_locally": False,
                "feature_availability_category": (
                    "post-migration explanatory for UMA barrier; "
                    "low-fidelity trajectory predictor only for a separate DFT target"
                ),
                "source_trajectory": f"{structure_id}_{migration_path_id}_0-6_relax",
                "fidelity_level": "UMA-sm-OMat24 MLIP CI-NEB",
                "missing_geometry_reason": (
                    "Figshare structure-image archive not locally acquired; "
                    "checkpoint prohibits additional downloads"
                ),
            }
        )
        selected_rows.append(record)

    output = pd.DataFrame(selected_rows)
    ordered_columns = [
        "structure_id",
        "formula",
        "mobile_ion_species",
        "host_anion_family",
        "migration_path_id",
        "source_trajectory",
        "fidelity_level",
        "image_count",
        "geometric_midpoint_image",
        "max_energy_image",
        "max_energy_is_geometric_midpoint",
        "energy_initial_eV",
        "energy_final_eV",
        "energy_max_eV",
        "forward_barrier_recomputed_eV",
        "kra_barrier_recomputed_eV",
        "source_forward_barrier_eV",
        "source_kra_barrier_eV",
        "endpoint_energy_asymmetry_eV",
        "energy_profile_relative_to_initial_eV",
        "initial_mobile_ion_position_frac",
        "final_mobile_ion_position_frac",
        "hop_distance_A",
        "initial_coordination",
        "final_coordination",
        "minimum_coordination_along_path",
        "minimum_anion_distance_A",
        "distance_contraction_total_A",
        "distance_contraction_anion_A",
        "distance_contraction_cation_A",
        "bottleneck_radius_ts_A",
        "coordination_kra_source",
        "voronoi_volume_kra_source_A3",
        "minimum_path_voronoi_volume_A3",
        "feature_availability_category",
        "atomic_order_consistency_verified",
        "structure_images_available_locally",
        "source_descriptor_match_error_eV",
        "supplement_sheet",
        "supplement_row_1based",
        "missing_geometry_reason",
    ]
    return output[ordered_columns], profiles


def plot_profiles(pilot: pd.DataFrame, profiles: dict, output_path: Path) -> None:
    colors = ["#13294B", "#E84A27", "#4B9CD3", "#FFB000", "#1D58A7"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for color, (_, row) in zip(colors, pilot.iterrows()):
        key = (int(row["structure_id"]), int(row["migration_path_id"]))
        profile = profiles[key].sort_values("image_index")
        y = profile["UMA-energy"] - profile["UMA-energy"].iloc[0]
        label = (
            f"{row['formula']} ({row['mobile_ion_species']}; "
            f"ICSD {row['structure_id']}, path {row['migration_path_id']})"
        )
        ax.plot(profile["image_index"], y, marker="o", linewidth=1.8, color=color, label=label)
        peak = int(row["max_energy_image"])
        ax.scatter([peak], [y.iloc[peak]], s=55, facecolors="white", edgecolors=color, zorder=4)
    ax.axvline(3, color="#7F7F7F", linestyle="--", linewidth=1.0, label="Geometric midpoint image")
    ax.set_xlabel("CI-NEB image index")
    ax.set_ylabel("Energy relative to initial image (eV)")
    ax.set_title("Selected UMA-MLIP migration energy profiles")
    ax.set_xticks(range(7))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.legend(frameon=False, fontsize=7.5, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    for path in (args.energy, args.supplement):
        if not path.exists():
            raise FileNotFoundError(path)

    pilot, profiles = build_pilot(args.energy, args.supplement)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pilot.to_csv(args.output, index=False, float_format="%.8g")
    plot_profiles(pilot, profiles, args.figure)

    print(f"Wrote {len(pilot)} paths to {args.output}")
    print(f"Wrote energy-profile figure to {args.figure}")
    print(
        "Maximum-energy image differs from midpoint for "
        f"{(~pilot['max_energy_is_geometric_midpoint']).sum()} of {len(pilot)} paths"
    )
    print("Missing structure-dependent fields were preserved as missing values")


if __name__ == "__main__":
    main()
