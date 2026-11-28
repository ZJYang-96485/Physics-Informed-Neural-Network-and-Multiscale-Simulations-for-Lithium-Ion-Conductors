# Checkpoint 6 — LiTraj Dataset Audit and Feature Engineering Feasibility

## Scope and decision

This checkpoint inspected the official LiTraj documentation, the complete local BVEL13k archive, the remote nebBVSE122k and nebDFT2k index files, and three real DFT-relaxed migration trajectories. The remote ZIP files were read through byte ranges; neither large archive was downloaded. Existing source datasets were not changed and no model was trained.

**Decision:** prioritize **nebDFT2k** as a source of path-resolved DFT labels and geometry, but first verify which records are cathodes. Use **nebBVSE122k only as paired low-fidelity information** and **BVEL13k as structure/percolation context**, not as interchangeable DFT labels. Improve a grouped, path-aware XGBoost baseline before starting a GNN.

The audit is reproducible with:

```bash
# Local BVEL audit only
.venv/bin/python results/ml/checkpoint6_litraj_feature_audit.py

# Index-only remote audit; reads small ZIP ranges, not the full archives
.venv/bin/python results/ml/checkpoint6_litraj_feature_audit.py --remote

# Also verify descriptors on three DFT trajectories
.venv/bin/python results/ml/checkpoint6_litraj_feature_audit.py --remote --sample-paths 3
```

## 1. LiTraj dataset inventory

The official [LiTraj repository](https://github.com/AIRI-Institute/LiTraj) describes structures from Materials Project and extended-XYZ storage. The record and structure counts below were computed from the actual index files rather than copied from the paper.

| Dataset | Records | Independent MP structures | Ion | Target | Structure/path content | Actual storage inspected |
|---|---:|---:|---|---|---|---|
| BVEL13k | 12,807 structure records | 12,807 | Li only | `E_1D`, `E_2D`, `E_3D`: 1D–3D BVSE percolation barriers in eV | Complete crystal structure for each row; no selected hop, endpoints, or intermediate trajectory | Index CSV plus train/validation/test extended-XYZ files; local archive is about 11 MB compressed and 35 MB expanded |
| nebBVSE122k | 122,421 unique vacancy hops | 11,290 | Li only | `em`: BVSE-NEB migration barrier in eV | Supercell with dummy `X` at the geometric path centroid; `edge_id` encodes source site, target site, and periodic offset; no distributed relaxed DFT images | Index CSV plus three centroid-structure extended-XYZ files; 199,873,007-byte archive, about 0.9 GB expanded |
| nebDFT2k | 1,681 unique vacancy hops | 867 | Li only | Paired `em_bvse` and `em_dft` migration barriers in eV | Per-hop BVSE-preconditioned trajectory, DFT-NEB-relaxed trajectory, and centroid supercell; endpoints and intermediate images are available | Index CSV, 1,681 `_init.xyz`, 1,681 `_relaxed.xyz`, and centroid structures; 68,583,182-byte archive, about 0.7 GB expanded |

The verified split counts are:

| Dataset | Train | Validation | Test | Chemical systems |
|---|---:|---:|---:|---:|
| BVEL13k | 10,159 | 1,331 | 1,317 | 1,812 |
| nebBVSE122k | 96,849 | 12,405 | 13,167 | 1,644 |
| nebDFT2k | 1,220 | 241 | 220 | 277 |

All indexed chemical systems contain Li; none of these datasets expands Na coverage. The official documentation confirms that nebDFT2k contains DFT climbing-image NEB trajectories, while nebBVSE122k distributes midpoint-marked structures for vacancy hops ([LiTraj dataset documentation](https://github.com/AIRI-Institute/LiTraj/blob/main/README.md)). The accompanying paper explains the DFT, BVSE, path-generation, and split construction in more detail ([Dembitskiy et al., 2025](https://www.nature.com/articles/s41524-025-01571-z)).

### Target compatibility

- **nebDFT2k is directly relevant** to the current target because both use calculated path-level migration barriers. It has far more independent structures than the current 11 framework/paper connected groups, and it includes the geometry missing from the current table.
- **nebBVSE122k is a lower-fidelity path dataset.** Its `em` values must not be pooled with DFT barriers as if they came from the same method.
- **BVEL13k has a different target.** A 1D/2D/3D percolation threshold for an entire crystal is not an individual DFT hop barrier.
- None of the three targets is experimental ionic conductivity, diffusivity, effective electrode conductivity, voltage, capacity, or synthesizability.

The three LiTraj sets are nested rather than additive: all 11,290 nebBVSE structures and all 867 nebDFT structures occur in BVEL13k, and all 1,681 DFT edges also occur in nebBVSE122k. The DFT/BVSE copies of an edge are paired fidelity levels, not independent observations.

## 2. Overlap with the existing cathode data

### What can and cannot be matched

| Matching evidence | Result | Interpretation |
|---|---|---|
| Exact MP ID | 0 matches between BVEL IDs and the current MP cathode tables | LiTraj uses legacy numeric IDs such as `mp-1020056`, whereas the current MP export uses opaque IDs such as `mp-aaaab...`. This is an unresolved identifier-version mismatch, not proof that the physical structures differ. |
| Exact structure fingerprint | Not currently possible | `cathode_migration_barrier_leads.csv` has no coordinates, and the existing MP framework-match table contains chemistry-only links. |
| Exact path endpoints | Not currently possible across datasets | The current 71-path table does not retain source/target site indices or atomic trajectories. |
| Verified phase match | 0 of 58 rows in `cathode_barrier_mp_framework_matches.csv` | Every existing MP match is explicitly chemistry-level and `phase_match_verified = False`. |
| Normalized host composition | Feasible as a screening match | Formula/proportion agreement identifies related chemistry only; it cannot establish a duplicate structure or path. |
| Source paper | No exact path-level crosswalk | LiTraj supplies MP/path identifiers and the LiTraj calculation provenance, not the individual literature references represented in the current barrier-lead table. |

Therefore, **zero LiTraj records are confirmed duplicates of the existing 71 paths**, but zero confirmed duplicates should not be read as zero physical overlap. A reliable duplicate audit requires an old-to-current MP ID crosswalk or direct structure matching, followed by endpoint matching.

### Composition-level screening results

The local BVEL structures were converted to normalized non-Li host-composition keys using atom counts. This produced 4,063 distinct host-composition keys.

- 592 BVEL structures share all 11 host-composition keys in the current provisional **Li** cathode subset.
- 4,826 BVEL structures across 468 host-composition keys also occur in the local MP Li insertion-electrode design pool.
- Within nebDFT2k, 232 of 867 structures and 358 of 1,681 paths fall in 142 of those MP Li insertion host-composition groups.
- Eight DFT structures/eight paths share five host-composition keys already represented in the current provisional Li subset.
- The remaining **224 DFT structures and 350 DFT paths span 137 additional host-composition groups** found in the MP Li insertion pool but not represented by the current Li barrier groups.

This last population is the most useful **candidate expansion pool**. It is not yet a set of 224 verified independent cathode frameworks: composition matching can join different polymorphs, oxidation states, or states of charge, and an insertion-electrode host match does not prove that the LiTraj migration path represents the working cathode phase.

### Defensible estimate of new cathode coverage

- **Verified new cathode frameworks now:** 0. Cathode role and exact phase/path identity have not been reviewed.
- **High-priority DFT verification pool:** up to 224 structures, 350 paths, and 137 additional host-composition groups.
- **Broader low-fidelity screening pool:** 4,271 BVEL structures across 460 MP insertion host-composition groups beyond the current provisional Li barrier keys.

The 224-structure number is an upper screening bound, not a result that should be used as a training count. A later curation step must verify cathode role, map the exact MP structure, remove structural duplicates, identify state of charge, and group polymorphs or closely related frameworks before the number of genuinely independent frameworks is known.

## 3. Physically informed descriptor feasibility

The feature strategies in [Dembitskiy et al.](https://www.nature.com/articles/s41524-025-01571-z) and [Honrao et al.](https://www.nature.com/articles/s41598-021-94275-5) support adding transparent local geometry rather than more opaque composition categories. LiTraj's [feature-engineering notebook](https://github.com/AIRI-Institute/LiTraj/blob/main/notebooks/featurizers.ipynb) constructs edge descriptors from the initial site, final site, and midpoint. The `ions` package supplies the path/percolation objects used by LiTraj ([ions repository](https://github.com/dembart/ions)).

For lattice matrix \(\mathbf A\), fractional periodic offset \(\mathbf n\), and endpoint positions \(\mathbf r_i,\mathbf r_f\), a reproducible minimum-image hop is

\[
\Delta\mathbf r=\arg\min_{\mathbf n\in\mathbb Z^3}
\left\|\mathbf r_f-\mathbf r_i+\mathbf A\mathbf n\right\|,
\qquad h=\|\Delta\mathbf r\|,
\qquad \mathbf r_m=\mathbf r_i+\tfrac12\Delta\mathbf r.
\]

At a probe position \(\mathbf q\), useful local quantities include

\[
d_{\min}^{\rm anion}(\mathbf q)=
\min_{j\in\mathrm{anions},\mathbf n}
\left\|\mathbf r_j-\mathbf q+\mathbf A\mathbf n\right\|,
\quad
CN_R^{\rm anion}(\mathbf q)=
\sum_{j,\mathbf n}\mathbb 1[d_{j\mathbf n}(\mathbf q)\le R].
\]

The geometric bottleneck feature can be the minimum periodic Voronoi-cell volume over the endpoints and midpoint,

\[
V_{\min}=\min\{V_{\rm Vor}(\mathbf r_i),V_{\rm Vor}(\mathbf r_m),V_{\rm Vor}(\mathbf r_f)\}.
\]

When a relaxed NEB trajectory is present, descriptors at the maximum-energy image should be recorded separately. The geometric midpoint is an input geometry, **not automatically the transition state**.

| Descriptor | BVEL13k | nebBVSE122k | nebDFT2k | Current 71 paths | Recommendation |
|---|---|---|---|---|---|
| Hop length | No selected path | Reconstructable from `edge_id` endpoints/offset | Directly available from endpoints | Already available | Retain; verify a single minimum-image convention |
| Minimum anion–Li distance | Structure-level Li sites only | Feasible at initial/final/midpoint | Feasible at initial/final/midpoint and each image | Not extractable without coordinates | Add first |
| Initial/final anion coordination | Structure-level averages possible | Feasible after endpoint reconstruction | Feasible directly | Not extractable | Add first; report cutoff or Voronoi definition |
| Local anion chemistry | Structure-level only | Feasible from neighboring species | Feasible from neighboring species | Host fractions only | Add compact neighbor fractions, not a long sparse element list |
| Periodic Voronoi volume | Structure-level Li sites only | Feasible at endpoints and `X` midpoint | Feasible at endpoints, midpoint, and relaxed images | Not extractable | Add first after numerical validation |
| Local free-sphere diameter | No path-specific value | Feasible but depends on the ionic/covalent-radius convention | Feasible | Not extractable | Defer until the radius definition is fixed and sensitivity-tested |
| Neighbor oxidation states | Not stored | Not stored explicitly | Not stored explicitly | Not present | Defer; charge assignment can be ambiguous and must not be fabricated |
| Actual transition-state descriptors | No | No DFT-relaxed trajectory | Yes: use the maximum-energy relaxed NEB image | No | Use for scientific interpretation and retrospective models; do not require it for deployment unless a path calculation is available |

Oxidation states can sometimes be inferred from charge balance or bond valence, but that inference may be non-unique for mixed-valence cathodes. It should be versioned, audited, and accompanied by a failure flag rather than silently treated as measured information. Similarly, a free-sphere diameter requires an explicit radius table; nearest-anion distance and Voronoi volume are cleaner first additions.

### Three-trajectory proof of concept

Three real DFT-relaxed trajectories—one from each official split—were read directly from the remote ZIP without saving the archive. The code identified the migrating Li, reproduced each published DFT barrier as the maximum image energy minus the lower endpoint energy, and extracted periodic nearest-neighbor, 3 Å anion coordination, and Voronoi-volume descriptors.

| Edge | Images | Hop length (Å) | DFT barrier (eV) | Max-energy image | Midpoint / max-image Voronoi volume (Å³) | Midpoint / max-image nearest-anion distance (Å) |
|---|---:|---:|---:|---:|---:|---:|
| `mp-1020056_0_0_1_1_0` | 7 | 5.182 | 2.067 | 3 | 8.02 / 10.26 | 1.859 / 1.931 |
| `mp-1020015_0_14_1_0_0` | 7 | 5.502 | 0.566 | 2 | 14.47 / 13.01 | 1.916 / 1.895 |
| `mp-1020057_2_0_0_1_0` | 7 | 4.793 | 3.578 | 3 | 8.34 / 10.86 | 1.555 / 1.701 |

The midpoint and relaxed maximum-energy environment differ in all three examples. This confirms both that path-specific extraction is feasible and that midpoint descriptors must remain distinct from transition-state descriptors. The proof of concept validates the calculation mechanics; three examples do not establish a statistical relationship with the barrier.

## 4. Recommended initial feature set

Retain the existing transparent features:

1. Mobile-ion identity for a future combined Li/Na table, while recognizing that LiTraj adds only Li data.
2. Minimum-image hop length.
3. Compact host composition: selected transition-metal fractions and local/global anion fractions.
4. Nominal mobile-ion number density, clearly labeled as occupied-ion density rather than carrier density.

Add only the following first path descriptors:

1. Initial and final anion coordination numbers: minimum, maximum, and absolute change.
2. Minimum anion–mobile-ion distance across initial, midpoint, and final positions.
3. Minimum periodic Voronoi volume across those three positions.
4. Local neighboring-anion fractions at the midpoint.
5. A simple endpoint asymmetry measure, such as \(|CN_i-CN_f|\) or \(|V_i-V_f|\).

Use DFT maximum-energy-image descriptors as a separate retrospective ablation. They are scientifically useful but cannot be known for an arbitrary new path without already performing a trajectory calculation. Defer inferred oxidation states, radius-sensitive free-sphere diameters, and the full 204-column LiTraj descriptor block until the small set above is validated.

## 5. BVSE as auxiliary information

The 1,681 DFT paths have paired BVSE values. In the inspected index, BVSE and DFT barriers have Pearson \(r=0.815\) and Spearman \(r=0.665\), but their mean absolute difference is 0.604 eV, their median absolute difference is 0.325 eV, and BVSE is higher on average by 0.443 eV. BVSE therefore contains useful ranking information but is not a drop-in DFT label.

A practical physically informed baseline is delta learning:

\[
\widehat E_{\rm DFT}=E_{\rm BVSE}+f_{\rm XGB}(\text{local geometry, chemistry}),
\]

compared against a direct XGBoost model using the same grouped folds. This is valid only when BVSE can also be computed for deployment candidates. If a learned BVSE estimate is used instead, it must be generated out of fold; otherwise the stacked model leaks target-associated structure information.

BVEL13k's percolation thresholds should be tested only as structure-level auxiliary descriptors. They should not be concatenated with hop barriers as additional target rows.

## 6. Leakage controls

1. Keep every path from one `material_id` in one fold.
2. Build a stronger structural-family group after exact structures are available so near-duplicate polymorphs or derived supercells do not cross folds under different IDs.
3. Keep the BVSE and DFT versions of the same `edge_id` in the same fold.
4. Do not count BVEL, nebBVSE, and nebDFT copies of one structure as independent samples.
5. Preserve the official LiTraj train/validation/test split for benchmark comparisons; perform feature selection only within training data.
6. If LiTraj is joined to the literature table, group by exact structure/framework and source paper after the MP-ID crosswalk is established.
7. Report both path-weighted metrics and equal-weight metrics across held-out structures or framework families.

Random path splitting would place different hops from the same crystal on both sides and substantially overstate generalization.

## 7. Next XGBoost development strategy

### Immediate sequence

1. **Curate the 232 DFT structures with MP Li insertion host chemistry.** Establish the MP-ID crosswalk and verify cathode phase, state of charge, and path relevance. Start with the 224 structures/350 paths in the 137 composition groups not represented by the current Li subset.
2. **Create a path table only for verified records.** Retain `material_id`, `edge_id`, fidelity, split, and cathode-review fields as provenance/grouping metadata, never as model features.
3. **Generate the five-descriptor local feature block** listed above and run range, missingness, symmetry, and unit checks.
4. **Compare three grouped baselines:** median, direct XGBoost, and BVSE-delta XGBoost. Use material/structural-family groups and leave the official test split untouched.
5. **Keep Na separate for now.** LiTraj supplies no Na path geometry. The existing 33 provisional Na paths should not be presented as equally supported by the Li-only expansion.
6. **Proceed to a GNN only after curation.** A graph model still needs verified structures, an explicit path representation, and enough independent cathode families. Dembitskiy et al. also found that limited DFT data remained a major obstacle for learned models, supporting a descriptor baseline before added model complexity ([paper](https://www.nature.com/articles/s41524-025-01571-z)).

### Answers to the checkpoint questions

1. **Dataset to prioritize:** nebDFT2k, restricted to cathode-verified records; use nebBVSE122k as paired auxiliary data and BVEL13k for context.
2. **Potential new cathode frameworks:** none are verified yet. The defensible screening pool is 224 DFT structures/350 paths across 137 additional MP-insertion-related host-composition groups.
3. **Existing features to retain:** hop length, compact chemistry, mobile-ion identity, and nominal ion density.
4. **New descriptors to introduce first:** endpoint coordination, coordination change, minimum anion distance, minimum Voronoi volume, and midpoint local anion chemistry.
5. **Descriptors to defer:** inferred oxidation states, radius-sensitive free-sphere diameter, hundreds of correlated edge features, and transition-state-only inputs for a deployment model.
6. **Can BVSE help?** Yes, through a grouped direct-feature or delta-learning comparison, but not as a DFT-equivalent label.
7. **Leakage prevention:** group all paths and fidelity levels by structure/structural family; pair identical `edge_id` records; preserve the official test split.
8. **Improve XGBoost before GNN?** Yes. The next bottleneck is verified cathode/path data and local geometry, not model capacity.

## Final recommendation

LiTraj offers a meaningful route out of the 71-path toy regime, but its full record count cannot be claimed as cathode training data. The next checkpoint should verify the **224-structure, 350-path DFT candidate pool**, then construct the small local-geometry feature block and evaluate direct versus BVSE-delta XGBoost under structure-family grouping.

Do not start a GNN until that curated path table demonstrates enough independent cathode structures and a leakage-safe representation. Stop here for review before downloading a full LiTraj archive or training any model.
