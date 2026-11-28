# Checkpoint 9 — Structural Descriptor Extraction for XGBoost

## Result

The five selected UMA-sm-OMat24 CI-NEB paths were recovered without downloading the complete archive. The script uses HTTP byte ranges to extract exactly 35 named members (five paths × seven images; 238,091 uncompressed bytes), checks ZIP CRC-32 and records SHA-256 hashes in `data/raw/computational_transport/umaneb_selected_paths/manifest.json`. No model was trained and the Checkpoint 8 files were preserved.

Sources: [Saravanan et al., *Advanced Materials*](https://doi.org/10.1002/adma.75410) and its [original Figshare record](https://doi.org/10.6084/m9.figshare.32221506), file `umaneb_data.zip` (104,370,519 bytes; MD5 `e8645198263ca822ba280dc1fd215ec1`). The exact previously missing members were `{ICSD}_{path}_{image}_relax` for paths 39648/19, 73276/6, 59023/13, 260875/22, and 45802/52, with image indices 0–6.

Run with:

```bash
python results/ml/checkpoint9_structural_descriptor_extraction.py --download-selected
```

Subsequent local runs omit `--download-selected`.

## 1. Structure and trajectory verification

For every path, images 0–6 exist and correspond to the seven energy rows in `UMA_NEB_energy.csv`. Atomic species and ordering, periodic cells, and lattice matrices are consistent across each path. The migrating atom is the Li/Na atom with the largest endpoint minimum-image displacement. The energy maximum is image 5, 4, 3, 3, and 4 for ICSD 39648, 73276, 59023, 260875, and 45802, respectively.

The selected maximum is called the **highest-energy image**, not a rigorously verified transition state. The source describes CI-NEB convergence, but the downloaded subset contains no per-image force residuals with which to independently certify a converged first-order saddle point.

## 2. Geometry definitions

- Distances use the minimum-image convention in the full periodic lattice.
- Endpoint hop distance is the minimum-image distance between the tracked mobile-ion sites in images 0 and 6.
- Coordination uses a dependency-light radial first shell: neighbors are sorted within 5 Å, the shell stops at the first gap ≥0.15 Å, and the cutoff is bounded by the nearest distance +0.8 Å and 1.35 times the nearest distance. These counts are reproducible but are **not CrystalNN counts**.
- Anions are assigned from the known host chemistry: O, Br, S, F, and O for the five paths.
- Voronoi volume is the bounded cell around the mobile ion in a 3×3×3 periodic replication, evaluated with SciPy Voronoi and ConvexHull.
- The relaxed-image contraction analogue is
  \(\max_j[(d_{j,IS}+d_{j,FS})/2-d_{j,HEI}]\), using the nearest atom of each framework species. Anion and cation maxima are also retained.

The relaxed-image contraction is not numerically identical to the authors' published `distance_deviation-TS`: the published descriptor places the UMA-NEB maximum-energy mobile-ion position in the original unrelaxed framework. The archive supplies relaxed NEB images, so the output retains both quantities under different prefixes and does not mislabel the relaxed analogue as the published descriptor.

## 3. Extracted results

| ICSD/path | Ion/host | Hop (Å) | Initial → final CN | HEI CN | Minimum anion distance (Å) | Minimum path Voronoi volume (Å³) |
|---|---|---:|---:|---:|---:|---:|
| 39648/19 | Li/oxide | 3.432 | 4 → 4 | 3 | 1.887 | 11.293 |
| 73276/6 | Li/bromide | 3.595 | 6 → 6 | 3 | 2.419 | 21.021 |
| 59023/13 | Na/sulfide | 3.265 | 3 → 4 | 3 | 2.636 | 27.029 |
| 260875/22 | Na/fluoride | 3.672 | 8 → 5 | 6 | 2.132 | 12.300 |
| 45802/52 | Na/oxide | 3.180 | 5 → 5 | 3 | 2.348 | 25.291 |

The CSV also contains species-resolved endpoint and highest-energy-image environments, minimum coordination along each path, minimum mobile-ion-to-framework distance, highest-energy-image Voronoi volume, relaxed-image contraction components, and the published source descriptors.

## 4. Feature-availability classes

The CSV is an audit table, not one model matrix. Prefixes deliberately keep the three classes separate.

### A. Pre-migration (`pre_`)

Endpoint fractional positions, hop distance, initial/final coordination, coordination difference, and endpoint species-resolved environments require only an equilibrium structure and enumerated endpoints. They can be generated for an unseen candidate at low cost. Endpoint enumeration itself must be performed consistently.

### B. Low-fidelity (`low_fidelity_`)

Highest-energy-image coordination/environment, minimum path coordination, minimum anion/framework distance, path Voronoi volumes, and relaxed-image contraction require a separately generated UMA-MLIP path. They can be used to predict a distinct higher-fidelity DFT-NEB barrier for the same path, but only after path matching and fidelity-aware grouped validation.

### C. Post-target explanatory (`post_target_`)

The published unrelaxed-TS contraction, TS bottleneck radius, coordination-KRA, and Voronoi-volume-KRA were constructed using the same UMA pathway/target calculation. They explain that target and would leak information if used to predict its UMA barrier. They are retained for provenance and mechanistic comparison only.

## 5. Recommendation

The most promising future XGBoost descriptors are:

1. **Endpoint hop distance** — inexpensive, candidate-ready, and directly describes jump geometry; partly redundant with any existing hop-length column.
2. **Initial/final coordination plus coordination asymmetry** — inexpensive and candidate-ready; adds local topology absent from bulk composition, but the radial-shell definition should be benchmarked against CrystalNN before scaling.
3. **Minimum anion distance along an independently generated UMA path** — physically meaningful local bottleneck information; useful only as a low-fidelity feature for a separate DFT target.
4. **Minimum path Voronoi volume** — captures local free space and is inexpensive after a path exists; potentially correlated with minimum anion/framework distance.
5. **Species-resolved relaxed-image contraction** — separates anion and framework-cation constriction; mechanistically useful for DFT calibration, but vulnerable to leakage for the same UMA target.

For the next pre-migration XGBoost experiment, use endpoint hop distance, endpoint coordination values/difference, and compact species-resolved endpoint environment features. Do not include any `low_fidelity_` or `post_target_` column in that matrix. A later UMA→DFT calibration study may use the low-fidelity block, while the post-target block must remain explanatory.

## Completion checks

- 35/35 selected structure images recovered; full archive not downloaded.
- 5/5 paths have verified atom order, PBC, constant lattice, and energy-image correspondence.
- Output: 5 rows × 42 columns; no missing values in the reported geometry fields.
- Script compiles and executes successfully.
- No XGBoost, neural network, GNN, PINN, or CNN training was performed.
