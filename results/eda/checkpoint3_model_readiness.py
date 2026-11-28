"""Reproduce the Checkpoint 3 population, feature, and grouping audit.

The script reads existing repository CSV files and prints descriptive audit
tables. It does not change source data or fit a machine-learning model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition, Element


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/cathode_migration_barrier_leads.csv"

TRANSITION_METALS = {element.symbol for element in Element if element.is_transition_metal}
ANION_ELEMENTS = {"O", "F", "S", "Se", "Cl", "Br", "I", "N"}


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


def connected_group_table(
    data: pd.DataFrame,
    prefix: str,
    combine_missing_sources: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    union_find = UnionFind()
    framework_nodes = []
    for row in data.itertuples(index=False):
        framework_node = f"F:{row.mobile_ion}|{row.framework_key}"
        if pd.notna(row.source_bibtex_key):
            paper_node = f"P:{row.source_bibtex_key}"
        elif combine_missing_sources:
            paper_node = "P:MISSING_SOURCE"
        else:
            paper_node = f"P:MISSING:{row.source_record_id}"
        union_find.union(framework_node, paper_node)
        framework_nodes.append(framework_node)

    labeled = data.copy()
    labeled["component_root"] = [union_find.find(node) for node in framework_nodes]
    components = []
    for root, group in labeled.groupby("component_root"):
        frameworks = sorted(
            (group["mobile_ion"] + " | " + group["framework_key"]).unique()
        )
        papers = sorted(group["source_bibtex_key"].dropna().unique())
        components.append(
            {
                "component_root": root,
                "n_paths": len(group),
                "ions": ", ".join(sorted(group["mobile_ion"].unique())),
                "n_formulas": group["material_formula"].nunique(),
                "frameworks": "; ".join(frameworks),
                "papers": "; ".join(papers) if papers else "missing",
            }
        )
    component_table = pd.DataFrame(components).sort_values(
        ["n_paths", "frameworks"], ascending=[False, True]
    )
    component_table["connected_group"] = [
        f"{prefix}-C{number:02d}" for number in range(1, len(component_table) + 1)
    ]
    root_to_id = component_table.set_index("component_root")["connected_group"]
    labeled["connected_group"] = labeled["component_root"].map(root_to_id)
    return component_table.drop(columns="component_root"), labeled


def feature_audit(data: pd.DataFrame, label: str) -> dict[str, object]:
    parsed_frameworks = data["framework_key"].map(parse_framework_key)
    return {
        "dataset": label,
        "n_paths": len(data),
        "mobile_ion_valid_pct": 100 * data["mobile_ion"].notna().mean(),
        "material_formula_valid_pct": 100 * data["material_formula"].notna().mean(),
        "formula_parse_valid_pct": 100
        * data["material_formula"].map(lambda value: bool(Composition(value))).mean(),
        "framework_parse_valid_pct": 100 * parsed_frameworks.map(bool).mean(),
        "transition_metal_derivable_pct": 100 * parsed_frameworks.map(bool).mean(),
        "transition_metal_present_pct": 100
        * parsed_frameworks.map(
            lambda composition: bool(set(composition).intersection(TRANSITION_METALS))
        ).mean(),
        "anion_element_derivable_pct": 100 * parsed_frameworks.map(bool).mean(),
        "anion_element_present_pct": 100
        * parsed_frameworks.map(
            lambda composition: bool(set(composition).intersection(ANION_ELEMENTS))
        ).mean(),
        "crystal_class_valid_pct": 100 * data["crystal_class"].notna().mean(),
        "cell_volume_valid_pct": 100 * data["cell_volume_A3"].notna().mean(),
        "hop_length_valid_pct": 100 * data["hop_length_A"].notna().mean(),
        "ion_density_valid_pct": 100
        * data["mobile_ion_number_density_cm3"].notna().mean(),
        "ion_atomic_fraction_valid_pct": 100
        * data["mobile_ion_atomic_fraction"].notna().mean(),
        "source_key_valid_pct": 100 * data["source_bibtex_key"].notna().mean(),
    }


def main() -> None:
    data = pd.read_csv(SOURCE)
    cathode = data.loc[
        data["auxiliary_barrier_training_status"].eq("provisional_cathode_role")
    ].copy()

    cathode_components, cathode_labeled = connected_group_table(
        cathode, prefix="A", combine_missing_sources=True
    )
    extended_components_separate, _ = connected_group_table(
        data, prefix="B", combine_missing_sources=False
    )
    extended_components_conservative, _ = connected_group_table(
        data, prefix="B", combine_missing_sources=True
    )

    framework_table = (
        cathode_labeled.groupby(["mobile_ion", "framework_key"], as_index=False)
        .agg(
            n_paths=("source_record_id", "size"),
            n_formulas=("material_formula", "nunique"),
            barrier_min_eV=("migration_barrier_eV", "min"),
            barrier_median_eV=("migration_barrier_eV", "median"),
            barrier_max_eV=("migration_barrier_eV", "max"),
            connected_group=("connected_group", "first"),
            papers=(
                "source_bibtex_key",
                lambda values: "; ".join(sorted(values.dropna().unique())),
            ),
        )
        .sort_values(["mobile_ion", "framework_key"])
    )

    print("Feature availability")
    print(
        pd.DataFrame(
            [feature_audit(cathode, "A: provisional cathode"), feature_audit(data, "B: extended")]
        ).to_string(index=False)
    )
    print("\nDataset A: 19 ion-framework groups")
    print(framework_table.to_string(index=False))
    print("\nDataset A: 11 framework/paper connected groups")
    print(cathode_components.to_string(index=False))
    print("\nDataset B connected-group counts")
    print(
        {
            "distinct placeholder for each missing paper": len(extended_components_separate),
            "all missing-paper records grouped conservatively": len(
                extended_components_conservative
            ),
        }
    )


if __name__ == "__main__":
    main()
