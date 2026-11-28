#!/usr/bin/env python3
"""Lightweight, read-only audit of the LiTraj datasets used in Checkpoint 6.

The script deliberately avoids downloading either large LiTraj archive.  It reads
the already-local BVEL13k archive and, when ``--remote`` is supplied, uses HTTP
byte ranges to read only the ZIP directory, the two small index CSVs, and up to
three selected nebDFT2k trajectories.

No repository data file is modified and no model is trained.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

BVEL_URL = "https://bs3u.obs.ru-moscow-1.hc.sbercloud.ru/litraj/BVEL13k.zip"
BVSE_URL = "https://bs3u.obs.ru-moscow-1.hc.sbercloud.ru/litraj/nebBVSE122k.zip"
DFT_URL = "https://bs3u.obs.ru-moscow-1.hc.sbercloud.ru/litraj/nebDFT2k.zip"


class HTTPRangeReader(io.RawIOBase):
    """Seekable, block-cached reader for a remote file with HTTP Range support."""

    def __init__(self, url: str, block_size: int = 1 << 20):
        self.url = url
        self.block_size = block_size
        self.pos = 0
        self.cache: dict[int, bytes] = {}
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.length = int(response.headers["Content-Length"])
            self.accept_ranges = response.headers.get("Accept-Ranges", "").lower()
        if "bytes" not in self.accept_ranges:
            raise RuntimeError(f"Server does not advertise byte ranges: {url}")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self.pos + offset
        elif whence == io.SEEK_END:
            new_pos = self.length + offset
        else:
            raise ValueError(f"Unknown whence: {whence}")
        if new_pos < 0:
            raise ValueError("Negative seek position")
        self.pos = min(new_pos, self.length)
        return self.pos

    def _block(self, index: int) -> bytes:
        if index not in self.cache:
            start = index * self.block_size
            end = min(start + self.block_size, self.length) - 1
            request = urllib.request.Request(
                self.url, headers={"Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(
                        f"Expected HTTP 206 for range request, got {response.status}"
                    )
                self.cache[index] = response.read()
        return self.cache[index]

    def read(self, size: int = -1) -> bytes:
        if self.pos >= self.length:
            return b""
        if size is None or size < 0:
            size = self.length - self.pos
        size = min(size, self.length - self.pos)
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            block_index = self.pos // self.block_size
            block_offset = self.pos % self.block_size
            block = self._block(block_index)
            take = min(remaining, len(block) - block_offset)
            chunks.append(block[block_offset : block_offset + take])
            self.pos += take
            remaining -= take
        return b"".join(chunks)

    @property
    def fetched_bytes(self) -> int:
        return sum(len(value) for value in self.cache.values())


def locate_member(names: Iterable[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one member ending in {suffix!r}; found {matches}")
    return matches[0]


def read_remote_index(url: str, suffix: str) -> tuple[pd.DataFrame, dict, HTTPRangeReader]:
    reader = HTTPRangeReader(url)
    with zipfile.ZipFile(reader) as archive:
        names = archive.namelist()
        member = locate_member(names, suffix)
        frame = pd.read_csv(io.BytesIO(archive.read(member)))
        metadata = {
            "url": url,
            "archive_bytes": reader.length,
            "archive_members": len(names),
            "index_member": member,
            "index_uncompressed_bytes": archive.getinfo(member).file_size,
        }
    metadata["range_bytes_fetched"] = reader.fetched_bytes
    return frame, metadata, reader


def summarize_index(name: str, frame: pd.DataFrame, target_columns: list[str]) -> dict:
    result = {
        "dataset": name,
        "rows": int(len(frame)),
        "unique_material_id": int(frame["material_id"].nunique()),
        "unique_chemsys": int(frame["chemsys"].nunique()),
        "rows_without_li_in_chemsys": int(
            (~frame["chemsys"].astype(str).str.split("-").apply(lambda x: "Li" in x)).sum()
        ),
        "split_counts": frame["_split"].value_counts().sort_index().to_dict(),
    }
    if "edge_id" in frame:
        result["unique_edge_id"] = int(frame["edge_id"].nunique())
    result["targets"] = {}
    for column in target_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        result["targets"][column] = {
            "valid": int(values.notna().sum()),
            "minimum": float(values.min()),
            "median": float(values.median()),
            "maximum": float(values.max()),
        }
    return result


def framework_key(symbols: list[str], mobile: str = "Li") -> str | None:
    counts = Counter(symbol for symbol in symbols if symbol != mobile)
    total = sum(counts.values())
    if not total:
        return None
    return "|".join(
        f"{element}:{counts[element] / total:.3f}" for element in sorted(counts)
    )


def iter_extxyz_compositions(stream: BinaryIO) -> Iterable[tuple[str, list[str]]]:
    """Yield (material_id, symbols) without loading the BVEL structures into RAM."""

    while True:
        line = stream.readline()
        if not line:
            return
        line = line.strip()
        if not line:
            continue
        atom_count = int(line)
        comment = stream.readline().decode("utf-8", errors="replace")
        match = re.search(r'material_id=(?:"([^\"]+)"|(\S+))', comment)
        if not match:
            raise RuntimeError(f"No material_id in extxyz comment: {comment[:200]}")
        material_id = match.group(1) or match.group(2)
        symbols = []
        for _ in range(atom_count):
            symbols.append(stream.readline().split(maxsplit=1)[0].decode("ascii"))
        yield material_id, symbols


def audit_local_bvel() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.read_csv(PROCESSED / "litraj_bvel13k_index.csv")
    records: list[dict] = []
    with zipfile.ZipFile(RAW / "litraj_BVEL13k.zip") as archive:
        for split in ("train", "val", "test"):
            member = locate_member(archive.namelist(), f"_{split}.xyz")
            with archive.open(member) as stream:
                for material_id, symbols in iter_extxyz_compositions(stream):
                    counts = Counter(symbols)
                    records.append(
                        {
                            "material_id": material_id,
                            "formula": "".join(
                                f"{element}{count if count != 1 else ''}"
                                for element, count in sorted(counts.items())
                            ),
                            "framework_key": framework_key(symbols),
                            "atom_count": len(symbols),
                            "split_from_xyz": split,
                        }
                    )
    structures = pd.DataFrame(records)
    if len(structures) != len(index) or structures["material_id"].nunique() != len(index):
        raise RuntimeError("BVEL structure/index row count mismatch")
    return index, structures


def overlap_audit(
    bvel_index: pd.DataFrame,
    structures: pd.DataFrame,
    bvse: pd.DataFrame | None,
    dft: pd.DataFrame | None,
) -> dict:
    leads = pd.read_csv(PROCESSED / "cathode_migration_barrier_leads.csv")
    matches = pd.read_csv(PROCESSED / "cathode_barrier_mp_framework_matches.csv")
    pool = pd.read_csv(PROCESSED / "cathode_mp_design_pool_eda.csv")

    bvel_ids = set(bvel_index["material_id"].dropna().astype(str))
    known_mp_ids = set(
        pd.concat(
            [
                matches["id_charge"],
                matches["id_discharge"],
                pool["id_charge"],
                pool["id_discharge"],
            ],
            ignore_index=True,
        )
        .dropna()
        .astype(str)
    )
    provisional = leads[
        leads["auxiliary_barrier_training_status"].eq("provisional_cathode_role")
    ]
    all_host_keys = set(leads["framework_key"].dropna().astype(str))
    provisional_host_keys = set(provisional["framework_key"].dropna().astype(str))
    provisional_li_keys = set(
        provisional.loc[provisional["mobile_ion"].eq("Li"), "framework_key"]
        .dropna()
        .astype(str)
    )
    provisional_na_keys = set(
        provisional.loc[provisional["mobile_ion"].eq("Na"), "framework_key"]
        .dropna()
        .astype(str)
    )
    pool_li_keys = set(
        pool.loc[pool["mobile_ion"].eq("Li"), "framework_key"].dropna().astype(str)
    )
    pool_na_keys = set(
        pool.loc[pool["mobile_ion"].eq("Na"), "framework_key"].dropna().astype(str)
    )
    bvel_host_keys = set(structures["framework_key"].dropna().astype(str))

    result = {
        "exact_bvel_mp_id_overlap_with_current_mp_tables": len(bvel_ids & known_mp_ids),
        "bvel_structures_with_any_existing_barrier_host_composition": int(
            structures["framework_key"].isin(all_host_keys).sum()
        ),
        "bvel_structures_with_provisional_cathode_host_composition": int(
            structures["framework_key"].isin(provisional_host_keys).sum()
        ),
        "unique_bvel_host_compositions": len(bvel_host_keys),
        "shared_host_compositions_all_barriers": len(bvel_host_keys & all_host_keys),
        "shared_host_compositions_provisional_cathodes": len(
            bvel_host_keys & provisional_host_keys
        ),
        "shared_host_compositions_provisional_li_cathodes": len(
            bvel_host_keys & provisional_li_keys
        ),
        "bvel_structures_with_provisional_li_cathode_host_composition": int(
            structures["framework_key"].isin(provisional_li_keys).sum()
        ),
        "shared_host_compositions_provisional_na_cathodes": len(
            bvel_host_keys & provisional_na_keys
        ),
        "bvel_structures_with_provisional_na_cathode_host_composition": int(
            structures["framework_key"].isin(provisional_na_keys).sum()
        ),
        "bvel_host_compositions_in_mp_li_insertion_pool": len(
            bvel_host_keys & pool_li_keys
        ),
        "bvel_structures_with_mp_li_insertion_host_composition": int(
            structures["framework_key"].isin(pool_li_keys).sum()
        ),
        "additional_bvel_host_compositions_in_mp_li_insertion_pool": len(
            (bvel_host_keys & pool_li_keys) - provisional_li_keys
        ),
        "bvel_structures_in_those_additional_host_compositions": int(
            structures["framework_key"]
            .isin((bvel_host_keys & pool_li_keys) - provisional_li_keys)
            .sum()
        ),
        "bvel_host_compositions_in_mp_na_insertion_pool": len(
            bvel_host_keys & pool_na_keys
        ),
        "unseen_bvel_host_compositions_vs_provisional_cathodes": len(
            bvel_host_keys - provisional_host_keys
        ),
        "phase_verified_rows_in_current_mp_match_table": int(
            matches["phase_match_verified"].fillna(False).astype(bool).sum()
        ),
    }
    if bvse is not None:
        result["bvel_bvse_shared_material_ids"] = len(
            bvel_ids & set(bvse["material_id"].astype(str))
        )
    if dft is not None:
        dft_ids = set(dft["material_id"].astype(str))
        dft_structures = structures[structures["material_id"].isin(dft_ids)]
        dft_host_keys = set(dft_structures["framework_key"].dropna().astype(str))
        dft_with_host = dft.merge(
            dft_structures[["material_id", "framework_key"]],
            on="material_id",
            how="left",
            validate="many_to_one",
        )
        result["bvel_dft_shared_material_ids"] = len(
            bvel_ids & dft_ids
        )
        result["dft_structures_with_provisional_li_cathode_host_composition"] = int(
            dft_structures["framework_key"].isin(provisional_li_keys).sum()
        )
        result["dft_paths_with_provisional_li_cathode_host_composition"] = int(
            dft_with_host["framework_key"].isin(provisional_li_keys).sum()
        )
        result["dft_host_compositions_shared_with_provisional_li_cathodes"] = len(
            dft_host_keys & provisional_li_keys
        )
        result["dft_structures_with_mp_li_insertion_host_composition"] = int(
            dft_structures["framework_key"].isin(pool_li_keys).sum()
        )
        result["dft_paths_with_mp_li_insertion_host_composition"] = int(
            dft_with_host["framework_key"].isin(pool_li_keys).sum()
        )
        result["dft_host_compositions_in_mp_li_insertion_pool"] = len(
            dft_host_keys & pool_li_keys
        )
        result["additional_dft_host_compositions_in_mp_li_insertion_pool"] = len(
            (dft_host_keys & pool_li_keys) - provisional_li_keys
        )
        result["dft_structures_in_those_additional_host_compositions"] = int(
            dft_structures["framework_key"]
            .isin((dft_host_keys & pool_li_keys) - provisional_li_keys)
            .sum()
        )
        result["dft_paths_in_those_additional_host_compositions"] = int(
            dft_with_host["framework_key"]
            .isin((dft_host_keys & pool_li_keys) - provisional_li_keys)
            .sum()
        )
    if bvse is not None and dft is not None:
        result["dft_edges_found_in_bvse"] = len(
            set(dft["edge_id"].astype(str)) & set(bvse["edge_id"].astype(str))
        )
    return result


def periodic_voronoi_volume(atoms, mobile_index: int, probe_position: np.ndarray) -> float:
    """Geometric periodic Voronoi volume around a mobile-ion/probe position."""

    from scipy.spatial import ConvexHull, Voronoi

    cell = np.asarray(atoms.cell)
    base = np.delete(np.asarray(atoms.positions), mobile_index, axis=0)
    base = np.vstack([base, probe_position])
    probe_base_index = len(base) - 1
    points = []
    probe_index = None
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                offset = np.array([i, j, k]) @ cell
                start = len(points)
                points.extend(base + offset)
                if i == j == k == 0:
                    probe_index = start + probe_base_index
    voronoi = Voronoi(np.asarray(points))
    region = voronoi.regions[voronoi.point_region[probe_index]]
    if not region or -1 in region:
        return float("nan")
    return float(ConvexHull(voronoi.vertices[region]).volume)


def local_probe_descriptors(atoms, mobile_index: int, position: np.ndarray) -> dict:
    from ase.geometry import get_distances

    anions = {"O", "F", "S", "Se", "N", "Cl", "Br", "I"}
    symbols = np.asarray(atoms.get_chemical_symbols())
    mask_framework = np.arange(len(atoms)) != mobile_index
    framework_positions = np.asarray(atoms.positions)[mask_framework]
    framework_symbols = symbols[mask_framework]
    distances = get_distances(
        np.asarray(position)[None, :],
        framework_positions,
        cell=np.asarray(atoms.cell),
        pbc=atoms.pbc,
    )[1][0]
    anion_distances = distances[np.isin(framework_symbols, list(anions))]
    return {
        "nearest_framework_A": float(distances.min()),
        "nearest_anion_A": float(anion_distances.min())
        if len(anion_distances)
        else float("nan"),
        "anion_coordination_within_3A": int((anion_distances <= 3.0).sum()),
        "voronoi_volume_A3": periodic_voronoi_volume(atoms, mobile_index, position),
    }


def trajectory_proof_of_concept(
    archive: zipfile.ZipFile, dft_index: pd.DataFrame, sample_count: int
) -> list[dict]:
    from ase.geometry import find_mic
    from ase.io import read

    sample_rows = (
        dft_index.sort_values(["_split", "material_id", "edge_id"])
        .groupby("_split", sort=True, group_keys=False)
        .head(1)
        .head(sample_count)
    )
    names = archive.namelist()
    results = []
    for _, row in sample_rows.iterrows():
        relaxed_name = locate_member(names, f"/{row['edge_id']}_relaxed.xyz")
        text = archive.read(relaxed_name).decode("utf-8")
        images = read(io.StringIO(text), index=":", format="extxyz")
        first, final = images[0], images[-1]
        symbols = np.asarray(first.get_chemical_symbols())
        li_indices = np.where(symbols == "Li")[0]
        li_displacements = []
        for index in li_indices:
            vector, length = find_mic(
                final.positions[index] - first.positions[index], first.cell, pbc=True
            )
            li_displacements.append((float(length), int(index), vector))
        hop_length, mobile_index, hop_vector = max(li_displacements)

        energies = []
        for image in images:
            try:
                energies.append(float(image.get_potential_energy()))
            except Exception:
                energies.append(float(image.info.get("energy", np.nan)))
        energies_array = np.asarray(energies)
        ts_image_index = int(np.nanargmax(energies_array))
        midpoint = first.positions[mobile_index] + 0.5 * hop_vector

        descriptor_sets = {
            "initial": local_probe_descriptors(
                first, mobile_index, first.positions[mobile_index]
            ),
            "geometric_midpoint": local_probe_descriptors(first, mobile_index, midpoint),
            "relaxed_max_energy_image": local_probe_descriptors(
                images[ts_image_index],
                mobile_index,
                images[ts_image_index].positions[mobile_index],
            ),
            "final": local_probe_descriptors(
                final, mobile_index, final.positions[mobile_index]
            ),
        }
        results.append(
            {
                "material_id": row["material_id"],
                "edge_id": row["edge_id"],
                "split": row["_split"],
                "em_bvse_eV": float(row["em_bvse"]),
                "em_dft_eV": float(row["em_dft"]),
                "trajectory_images": len(images),
                "migrating_li_index": mobile_index,
                "endpoint_hop_length_A": hop_length,
                "max_energy_image_index": ts_image_index,
                "barrier_from_lower_endpoint_eV": float(
                    energies_array[ts_image_index]
                    - min(energies_array[0], energies_array[-1])
                ),
                "descriptors": descriptor_sets,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Read remote nebBVSE/nebDFT indexes using byte ranges.",
    )
    parser.add_argument(
        "--sample-paths",
        type=int,
        default=0,
        choices=range(0, 4),
        metavar="{0,1,2,3}",
        help="Read and featurize up to three real nebDFT trajectories.",
    )
    args = parser.parse_args()

    bvel, structures = audit_local_bvel()
    output: dict = {
        "BVEL13k": summarize_index("BVEL13k", bvel, ["E_1D", "E_2D", "E_3D"])
    }

    bvse = None
    dft = None
    if args.remote:
        bvse, bvse_meta, _ = read_remote_index(BVSE_URL, "/nebBVSE122k_index.csv")
        dft, dft_meta, dft_reader = read_remote_index(DFT_URL, "/nebDFT2k_index.csv")
        output["nebBVSE122k"] = summarize_index("nebBVSE122k", bvse, ["em"])
        output["nebBVSE122k"]["access"] = bvse_meta
        output["nebDFT2k"] = summarize_index(
            "nebDFT2k", dft, ["em_bvse", "em_dft"]
        )
        paired_error = dft["em_bvse"] - dft["em_dft"]
        output["nebDFT2k"]["paired_bvse_dft_audit"] = {
            "pearson_r": float(dft["em_bvse"].corr(dft["em_dft"], method="pearson")),
            "spearman_r": float(dft["em_bvse"].corr(dft["em_dft"], method="spearman")),
            "mean_bvse_minus_dft_eV": float(paired_error.mean()),
            "median_absolute_difference_eV": float(paired_error.abs().median()),
            "mean_absolute_difference_eV": float(paired_error.abs().mean()),
            "has_specific_TM_counts": dft["has_specific_TM"]
            .astype(str)
            .value_counts()
            .to_dict(),
        }
        output["nebDFT2k"]["access"] = dft_meta
        if args.sample_paths:
            with zipfile.ZipFile(dft_reader) as archive:
                output["trajectory_proof_of_concept"] = trajectory_proof_of_concept(
                    archive, dft, args.sample_paths
                )
            output["nebDFT2k"]["access"][
                "range_bytes_fetched_after_samples"
            ] = dft_reader.fetched_bytes

    output["overlap"] = overlap_audit(bvel, structures, bvse, dft)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
