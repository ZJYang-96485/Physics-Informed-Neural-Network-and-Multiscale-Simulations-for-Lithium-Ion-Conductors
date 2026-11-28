"""Feature engineering for Materials Project lithium insertion-electrode records."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


BASE_COLUMNS = [
    "battery_id",
    "battery_formula",
    "framework_formula",
    "formula_charge",
    "formula_discharge",
    "chemsys",
    "working_ion",
    "average_voltage",
    "capacity_grav",
    "capacity_vol",
    "energy_grav",
    "energy_vol",
    "max_delta_volume",
    "max_voltage_step",
    "num_steps",
    "fracA_charge",
    "fracA_discharge",
    "stability_charge",
    "stability_discharge",
    "id_charge",
    "id_discharge",
    "warnings",
    "last_updated",
]


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _weighted_property(
    composition: Composition,
    attribute: str,
    fallback: str | None = None,
) -> float:
    values: list[float] = []
    weights: list[float] = []
    for element in composition.elements:
        value = getattr(element, attribute, None)
        if value is None and fallback:
            value = getattr(element, fallback, None)
        value = _finite_float(value)
        if np.isfinite(value):
            values.append(value)
            weights.append(composition.get_atomic_fraction(element))
    if not values:
        return np.nan
    weights_array = np.asarray(weights)
    return float(np.average(values, weights=weights_array / weights_array.sum()))


def _anion_family(symbols: set[str]) -> str:
    if {"O", "F"}.issubset(symbols):
        return "oxyfluoride"
    if "F" in symbols:
        return "fluoride"
    for element, label in [
        ("P", "phosphate"),
        ("S", "sulfate"),
        ("Si", "silicate"),
        ("B", "borate"),
    ]:
        if {"O", element}.issubset(symbols):
            return label
    if "O" in symbols:
        return "oxide"
    if "S" in symbols:
        return "sulfide"
    if "Se" in symbols:
        return "selenide"
    return "other"


def _composition_features(formula: Any) -> dict[str, Any]:
    try:
        composition = Composition(str(formula))
    except Exception:
        return {}

    symbols = {element.symbol for element in composition.elements}
    transition_metals = [
        element for element in composition.elements if element.is_transition_metal
    ]
    dominant_tm = max(
        transition_metals,
        key=lambda element: composition.get_atomic_fraction(element),
        default=None,
    )
    electronegativities = [
        _finite_float(element.X)
        for element in composition.elements
        if np.isfinite(_finite_float(element.X))
    ]

    features = {
        "framework_n_elements": len(composition.elements),
        "mean_atomic_number": _weighted_property(composition, "Z"),
        "mean_atomic_mass": _weighted_property(composition, "atomic_mass"),
        "mean_electronegativity": _weighted_property(composition, "X"),
        "mean_atomic_radius": _weighted_property(
            composition, "atomic_radius", "atomic_radius_calculated"
        ),
        "mean_periodic_row": _weighted_property(composition, "row"),
        "mean_periodic_group": _weighted_property(composition, "group"),
        "mean_mendeleev_number": _weighted_property(composition, "mendeleev_no"),
        "min_electronegativity": min(electronegativities, default=np.nan),
        "max_electronegativity": max(electronegativities, default=np.nan),
        "transition_metal_fraction": sum(
            composition.get_atomic_fraction(element)
            for element in transition_metals
        ),
        "dominant_transition_metal": dominant_tm.symbol if dominant_tm else "none",
        "transition_metal_row": (
            {4: "3d", 5: "4d", 6: "5d"}.get(dominant_tm.row, "other")
            if dominant_tm
            else "none"
        ),
        "mixed_transition_metal": int(len(transition_metals) > 1),
        "anion_family": _anion_family(symbols),
        "contains_polyanion": int(
            "O" in symbols and bool(symbols.intersection({"P", "S", "Si", "B"}))
        ),
    }
    for symbol, name in [
        ("O", "oxygen_fraction"),
        ("F", "fluorine_fraction"),
        ("P", "phosphorus_fraction"),
        ("S", "sulfur_fraction"),
    ]:
        features[name] = (
            composition.get_atomic_fraction(Element(symbol)) if symbol in symbols else 0.0
        )
    features["electronegativity_range"] = (
        features["max_electronegativity"] - features["min_electronegativity"]
    )
    return features


def _structure_features(value: Any) -> dict[str, Any]:
    try:
        structure = value if isinstance(value, Structure) else Structure.from_dict(value)
    except Exception:
        return {}
    if not structure:
        return {}

    features = {
        "host_density": _finite_float(structure.density),
        "host_volume": _finite_float(structure.volume),
        "host_volume_per_atom": _finite_float(structure.volume / len(structure)),
        "host_n_sites": len(structure),
        "crystal_system": "unknown",
        "space_group_number": np.nan,
    }
    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
        features["crystal_system"] = analyzer.get_crystal_system()
        features["space_group_number"] = analyzer.get_space_group_number()
    except Exception:
        pass
    return features


def build_feature_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Create the cached model table from raw insertion-electrode records."""
    raw = pd.DataFrame(records)
    available_base = [column for column in BASE_COLUMNS if column in raw.columns]
    composition = raw["framework_formula"].apply(_composition_features).apply(pd.Series)
    structure = raw["host_structure"].apply(_structure_features).apply(pd.Series)
    table = pd.concat(
        [
            raw[available_base].reset_index(drop=True),
            composition.reset_index(drop=True),
            structure.reset_index(drop=True),
        ],
        axis=1,
    )
    numeric = [
        "average_voltage",
        "capacity_grav",
        "capacity_vol",
        "energy_grav",
        "energy_vol",
        "max_delta_volume",
        "max_voltage_step",
        "num_steps",
        "fracA_charge",
        "fracA_discharge",
        "stability_charge",
        "stability_discharge",
    ]
    for column in numeric:
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    if {"id_charge", "id_discharge"}.issubset(table.columns):
        table = table.drop_duplicates(["id_charge", "id_discharge"])
    return table.reset_index(drop=True)
