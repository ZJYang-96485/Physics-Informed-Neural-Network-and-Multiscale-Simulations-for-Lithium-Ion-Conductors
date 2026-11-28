# Checkpoint 5 — XGBoost Model Diagnosis

## Scope

This diagnosis uses the saved Checkpoint 4 predictions in `results/ml/checkpoint4_cross_validated_predictions.csv`, the 71 provisional cathode rows in `data/processed/cathode_migration_barrier_leads.csv`, and the executed `XGBoost_Migration_Barrier_Analysis.ipynb`. No model was trained, no source record was changed, and no additional dataset was collected.

The diagnosis concerns calculated migration barriers for already enumerated paths. It does not evaluate experimental conductivity or arbitrary Materials Project candidates.

## Main finding

The model is limited by **11 independent framework/paper connected groups and an incomplete representation of the migration environment**, not by the number of boosting rounds. The 71 rows contain repeated paths and formulas from a few families; one connected group supplies 32 paths. Composition, ion identity, hop length, and nominal ion density do not describe coordination changes, bottleneck geometry, saddle-site chemistry, topology, vacancies, or state of charge.

XGBoost's pooled three-fold MAE is 0.273 eV versus 0.275 eV for the median baseline, a difference of only 0.002 eV. Its pooled leave-one-group-out MAE is worse than the baseline, 0.278 versus 0.273 eV. The saved results therefore do not demonstrate useful generalization to unfamiliar cathode framework/paper groups.

## 1. Group-level performance

### Three-fold comparison

| Fold | Held-out paths | Held-out connected groups | XGBoost MAE (eV) | Median MAE (eV) | Median − XGBoost (eV) |
|---:|---:|---|---:|---:|---:|
| 1 | 32 | A-C01 | 0.313 | 0.334 | +0.022 |
| 2 | 20 | A-C02, A-C04, A-C09, A-C10, A-C11 | 0.237 | 0.217 | −0.021 |
| 3 | 19 | A-C03, A-C05, A-C06, A-C07, A-C08 | 0.242 | 0.236 | −0.006 |
| **Pooled** | **71** | **11 groups** | **0.273** | **0.275** | **+0.002** |

XGBoost improves on the baseline only in Fold 1, which contains the dominant A-C01 group. It is worse in both folds formed from the other ten groups.

### Training and held-out barrier distributions

| Fold | Train mean / median (eV) | Test mean / median (eV) | Train range (eV) | Test range (eV) | Held-out barriers outside train range |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.514 / 0.461 | 0.643 / 0.529 | 0.110–1.363 | 0.156–2.308 | 4 above the training maximum |
| 2 | 0.615 / 0.541 | 0.463 / 0.405 | 0.156–2.308 | 0.110–0.980 | 1 below the training minimum |
| 3 | 0.574 / 0.444 | 0.568 / 0.590 | 0.110–2.308 | 0.207–1.363 | 0 |

Fold 1 has a broader and higher held-out distribution than its training population. Four held-out barriers exceed every training barrier, including the retained 2.308 eV record. XGBoost predictions contract toward the center of the training data and underpredict these high barriers.

### Error by connected group

| Group | Paths | Ion(s) | XGBoost MAE (eV) | Median MAE (eV) | Share of XGBoost absolute error |
|---|---:|---|---:|---:|---:|
| A-C09 | 2 | Na | 0.386 | 0.281 | 4.0% |
| A-C03 | 6 | Li | 0.384 | 0.079 | 11.9% |
| A-C10 | 2 | Na | 0.346 | 0.331 | 3.6% |
| A-C01 | 32 | Li, Na | 0.313 | 0.334 | 51.7% |
| A-C07 | 3 | Na | 0.273 | 0.448 | 4.2% |
| A-C06 | 3 | Li | 0.253 | 0.342 | 3.9% |
| A-C11 | 1 | Li | 0.234 | 0.139 | 1.2% |
| A-C02 | 10 | Li | 0.205 | 0.204 | 10.6% |
| A-C04 | 5 | Na | 0.201 | 0.186 | 5.2% |
| A-C08 | 2 | Na | 0.150 | 0.236 | 1.5% |
| A-C05 | 5 | Na | 0.083 | 0.233 | 2.2% |

A-C03 is the clearest failure relative to the baseline: its XGBoost MAE is 0.384 eV, compared with 0.079 eV for the median. A-C09 has the largest per-path error, but only two paths, so it does not dominate the pooled result.

The two O–P–V frameworks inside A-C01 account for 51.7% of total XGBoost absolute error:

- Na `O:0.714|P:0.143|V:0.143`: 14 paths, MAE 0.458 eV, and 33.1% of total XGBoost absolute error.
- Li `O:0.714|P:0.143|V:0.143`: 18 paths, MAE 0.200 eV, and 18.6% of total XGBoost absolute error.

The Na framework spans 0.263–2.308 eV, while the Li framework spans 0.156–1.199 eV. Composition alone cannot explain this large path-level variation.

### Influence of the dominant group

| Population | Paths | XGBoost MAE (eV) | Median MAE (eV) |
|---|---:|---:|---:|
| All paths | 71 | 0.273 | 0.275 |
| A-C01 only | 32 | 0.313 | 0.334 |
| All groups except A-C01 | 39 | 0.240 | 0.226 |
| Equal-weight average across 11 groups | — | 0.257 | 0.256 |

A-C01 contains 45.1% of the paths but 51.7% of XGBoost's absolute error. XGBoost's improvement inside A-C01 is large enough to conceal worse performance across the remaining paths. Once A-C01 is removed, XGBoost is worse than the median. An equal-weight group average also slightly favors the median.

### Li and Na behavior

| Ion | Paths | Ion–framework groups | XGBoost MAE (eV) | Median MAE (eV) | Mean XGBoost residual (eV) |
|---|---:|---:|---:|---:|---:|
| Li | 38 | 11 | 0.235 | 0.208 | +0.017 |
| Na | 33 | 8 | 0.315 | 0.352 | −0.130 |

Path counts are superficially balanced, but chemistry coverage is not. XGBoost improves on the median for Na while systematically underpredicting Na barriers; for Li it is worse than the median. These differences cannot be assigned to ion identity alone because Li and Na occupy different framework and paper distributions.

## 2. Feature diagnostics

### Variation and sparsity

| Feature block | Observed variation | Diagnostic |
|---|---|---|
| Mobile-ion identity | 38 Li and 33 Na paths | Balanced by paths, not by matched cathode families or papers. |
| Hop length | 2.705–8.950 Å; 70 unique values | High path-level variation, but length alone does not specify the bottleneck or coordination environment. |
| Nominal mobile-ion density | 0.020–3.213 × 10²² cm⁻³; 38 unique values | Nine of 19 ion–framework groups have no within-group density variation. It is occupied-ion density, not carrier or vacancy density. |
| Host fractions | 14 elements across 17 host signatures | Sparse family indicators. Al, C, Co, Cr, Ni, S, and Si occur in five or fewer paths. |

No model-input pair has absolute Spearman correlation at or above 0.70. The largest is F fraction versus S fraction at 0.691, caused by a small shared chemistry family. Other moderate relationships include O versus V (0.582), P versus V (0.569), Mn versus V (−0.513), and ion density versus Mn fraction (0.496).

The lack of a very high pairwise correlation does not mean the composition variables are independent. Their fractions sum to approximately one, and most element columns are sparse indicators of one or two families. Importance can move between related family indicators without representing a general physical effect.

### Can the features distinguish frameworks?

The composition and ion features reproduce the 19 ion–framework labels by construction, because `framework_key` itself is a normalized host composition. This does not mean they distinguish crystal structures:

- The 71 paths contain 34 material formulas but only 19 ion–framework composition groups.
- Nine of the 19 groups contain more than one formula; these groups contain 48 of 71 paths.
- The feature table has 70 unique full vectors, mainly because hop length is almost always unique, but it contains no explicit local environment.
- Two Na O–P–V rows have identical model inputs and the same material formula but reported barriers of 0.25 and 0.17 eV. The current feature space cannot distinguish them.

Held-out chemistry also leaves the training support frequently:

| Fold | Held-out rows outside the training range of at least one feature |
|---:|---:|
| 1 | 7 of 32 |
| 2 | 12 of 20 |
| 3 | 19 of 19 |

Fold 3 contains unseen or out-of-range B, Al, C, S, O, P, Fe, density, or related fraction values for every held-out row. A tree ensemble can partition the observed training space but cannot physically extrapolate to unseen local environments. This distribution shift is a stronger explanation than insufficient hyperparameter tuning.

### Importance stability

Saved three-fold permutation importance gives:

| Feature | Mean increase in validation MAE when permuted (eV) | Across-fold standard deviation (eV) |
|---|---:|---:|
| Hop length | 0.0313 | 0.0179 |
| O fraction | 0.0008 | 0.0036 |
| P fraction | 0.0007 | 0.0006 |
| Mn fraction | 0.0001 | 0.0002 |
| Other displayed elemental fractions | approximately 0 | approximately 0 |

Hop length is the only feature with a meaningful positive average, but its across-fold standard deviation is 57% of its mean. The O-fraction variation exceeds its average importance, and most elemental fractions have negligible importance. This is evidence of weak, fold-dependent predictive use, not physical causation.

### Important missing structural information

The model lacks descriptors that distinguish two paths with similar composition and length:

1. Coordination number and coordinating species at the initial, saddle, and final sites.
2. Bottleneck radius, face area, and nearest-neighbor distances along the path.
3. Change in coordination and local polyhedral distortion along the path.
4. Migration topology, path dimensionality, site connectivity, and path multiplicity.
5. Vacancy concentration, site occupancy, and mobile-ion state of charge.
6. Local electrostatic environment, oxidation-state context, and anion arrangement.
7. Exact structure, symmetry, lattice parameters, and volume per atom for the reported phase.
8. DFT protocol fields that affect cross-paper comparability, such as functional, (U), supercell, magnetic state, and relaxation/NEB settings.

Some global fields already exist in the source table, including cell volume, atomic fraction, crystal class, and displacement summaries. They may support later ablations, but they do not replace path-resolved local geometry and require consistent definitions before use.

## 3. Data limitations and acquisition priorities

### Priority 1: genuinely new cathode frameworks

The most valuable unit is a new, verified framework/paper connected group. More paths from an existing framework improve estimates of within-framework variability, but they do not provide evidence of generalization to a new chemistry or structure.

The next data pass should prioritize underrepresented cathode families and independent papers. The 159 unresolved records in the existing 284-path table are the first review pool because some may expand cathode coverage without collecting a new dataset. Exact phase, path, cathode role, and source must be verified before moving a record into the cathode population.

### Priority 2: balanced Li and Na family coverage

The target is not equal path counts. It is comparable coverage of independent Li and Na framework families, ideally including chemically related Li/Na analogues calculated with comparable methods. This would separate mobile-ion effects from chemistry and paper effects.

### Priority 3: local path descriptors

For each verified barrier, retain the initial site, saddle/path geometry, final site, hop length, local coordination, bottleneck geometry, neighboring elements, and vacancy/site occupancy. These descriptors directly address the failure mode seen in identical or compositionally similar input rows with different barriers.

### Priority 4: complete and matched structures

Link each barrier to the exact reported phase and state of charge, not merely a similar composition. Materials Project structures may supply useful global descriptors, but a match must be confirmed before using them as substitutes for the literature structure.

### Priority 5: additional verified barrier calculations

After new framework coverage improves, add multiple paths per framework to measure anisotropy and identify the rate-limiting path. Use explicit provenance and consistent path definitions. More repeated paths from A-C01 alone would increase row count while worsening the current imbalance.

## 4. Recommended development direction

| Option | Value now | Main limitation | Recommendation |
|---|---|---|---|
| Expand the DFT barrier dataset | Directly increases independent chemistry coverage and can reduce group imbalance | Requires paper-level phase and role verification | **Highest priority** |
| Improve structural features | Addresses the inability to distinguish local paths with similar composition | Requires exact structures and path mappings for both training and future candidates | **Begin alongside data curation** |
| Proceed to a GNN | Could learn local structure when enough matched structures and labels exist | Only 11 connected groups; structures are not yet consistently linked to path labels; ordinary crystal graphs do not automatically encode a migration path | **Wait** |

For the senior-design timeline, the most reasonable next step is a focused **data-and-structure linkage checkpoint**, not a more complex model. Review the existing extended dataset for additional cathode frameworks, link verified rows to exact structures and migration paths, and define a small set of local geometric descriptors that can be calculated consistently.

GNN development should wait until there are several dozen independent verified cathode framework/paper groups with matched structures and an explicit way to represent the migration path. This is a practical readiness gate, not a guarantee of model accuracy. Starting a GNN now would add model complexity while leaving the main limitations unchanged.

## Final recommendation

1. Expand the verified cathode population by **independent framework/paper groups**, beginning with review of the existing unresolved records.
2. Link each accepted barrier to its exact structure, state of charge, and path endpoints.
3. Add transparent local path descriptors before revisiting model choice.
4. Keep the current grouped XGBoost result as the baseline.
5. Defer GNN implementation until the dataset and structure/path representation pass the readiness gate above.
