# Checkpoint 3 — Modeling Dataset Preparation and Validation Strategy

## Scope and reproducibility

This checkpoint uses the existing `data/processed/cathode_migration_barrier_leads.csv` and the Checkpoint 2 findings. It defines modeling populations, audits feature availability, and establishes validation groups. No records are removed, no source dataset is modified, and no model is trained.

The population and grouping audit can be reproduced from the repository root with:

```bash
.venv/bin/python results/eda/checkpoint3_model_readiness.py
```

## 1. Initial training populations

### Dataset A: provisional cathode subset

Dataset A contains the **71 paths** for which:

```text
auxiliary_barrier_training_status = provisional_cathode_role
cathode_role_evidence = title_mentions_cathode
```

It contains 38 Li and 33 Na paths, 34 material formulas, 19 ion–framework groups, 14 identifiable source papers, and 11 framework/paper connected groups. All 71 records retain their original `source_record_id`, title, BibTeX key, dataset DOI, and source URL. All also retain `original_paper_review_required = True`.

**Advantages**

- It is the only current population with explicit cathode evidence.
- Li and Na are balanced at the path level.
- Every prioritized numeric feature is populated.
- The smaller scope makes the intended prediction task easier to explain.

**Limitations**

- Cathode evidence comes from the source title, not a verified phase/path audit.
- Only 19 ion–framework groups and 11 connected groups are available.
- One connected group contains 32 of 71 paths, or 45.1% of the dataset.
- Several frameworks have only one or two paths.
- The dataset supports path-level barrier prediction, not conductivity prediction.

### All 19 ion–framework groups in Dataset A

| Ion | `framework_key` | Paths | Formulas | Connected group | Source paper key(s) |
|---|---|---:|---:|---|---|
| Li | `B:0.200|Co:0.200|O:0.600` | 2 | 1 | A-C03 | `seo2011first` |
| Li | `B:0.200|Fe:0.200|O:0.600` | 2 | 1 | A-C03 | `seo2011first` |
| Li | `B:0.200|Mn:0.200|O:0.600` | 2 | 1 | A-C03 | `seo2011first` |
| Li | `Co:0.333|O:0.667` | 2 | 2 | A-C02 | `canepa2017odyssey` |
| Li | `Cr:0.333|O:0.667` | 2 | 2 | A-C02 | `canepa2017odyssey`; `li2021search` |
| Li | `F:0.143|Fe:0.143|O:0.571|S:0.143` | 3 | 1 | A-C06 | `mueller2011evaluation` |
| Li | `F:0.143|Mn:0.143|O:0.571|P:0.143` | 1 | 1 | A-C11 | `alfaruqi2020density` |
| Li | `Fe:0.167|O:0.667|Si:0.167` | 2 | 1 | A-C02 | `li2021search` |
| Li | `Mn:0.333|O:0.667` | 2 | 2 | A-C02 | `canepa2017odyssey` |
| Li | `Ni:0.333|O:0.667` | 2 | 2 | A-C02 | `canepa2017odyssey` |
| Li | `O:0.714|P:0.143|V:0.143` | 18 | 5 | A-C01 | `lin2017comparison` |
| Na | `Al:0.054|Mn:0.054|O:0.649|P:0.243` | 3 | 2 | A-C07 | `wang2021experimental` |
| Na | `C:0.100|Fe:0.100|O:0.700|P:0.100` | 2 | 1 | A-C08 | `huang2014detailed` |
| Na | `F:0.143|O:0.571|P:0.143|V:0.143` | 2 | 1 | A-C09 | `luong2021tavorite` |
| Na | `Fe:0.167|O:0.667|P:0.167` | 3 | 2 | A-C04 | `aksyonov2018understanding`; `essehli2020selective` |
| Na | `Mn:0.250|O:0.750` | 5 | 1 | A-C05 | `zheng2018na2mno3` |
| Na | `Mn:0.333|O:0.667` | 2 | 2 | A-C04 | `aksyonov2018understanding`; `ling2013phase` |
| Na | `O:0.700|P:0.200|V:0.100` | 2 | 1 | A-C10 | `torres2021enlisting` |
| Na | `O:0.714|P:0.143|V:0.143` | 14 | 5 | A-C01 | `lin2017comparison` |

### All 11 framework/paper connected groups in Dataset A

A connected group joins paths whenever they share either the same `(mobile_ion, framework_key)` or the same source-paper key. This relationship is transitive.

| Group | Paths | Ion(s) | Included ion–frameworks | Source paper key(s) |
|---|---:|---|---|---|
| A-C01 | 32 | Li, Na | Li and Na `O:0.714|P:0.143|V:0.143` | `lin2017comparison` |
| A-C02 | 10 | Li | Li Co–O, Cr–O, Fe–O–Si, Mn–O, and Ni–O frameworks | `canepa2017odyssey`; `li2021search` |
| A-C03 | 6 | Li | Li B–Co–O, B–Fe–O, and B–Mn–O frameworks | `seo2011first` |
| A-C04 | 5 | Na | Na Fe–O–P and Mn–O frameworks | `aksyonov2018understanding`; `essehli2020selective`; `ling2013phase` |
| A-C05 | 5 | Na | Na `Mn:0.250|O:0.750` | `zheng2018na2mno3` |
| A-C06 | 3 | Li | Li Fe–O–S–F | `mueller2011evaluation` |
| A-C07 | 3 | Na | Na Al–Mn–O–P | `wang2021experimental` |
| A-C08 | 2 | Na | Na C–Fe–O–P | `huang2014detailed` |
| A-C09 | 2 | Na | Na F–O–P–V | `luong2021tavorite` |
| A-C10 | 2 | Na | Na O–P–V | `torres2021enlisting` |
| A-C11 | 1 | Li | Li F–Mn–O–P | `alfaruqi2020density` |

### Dataset B: extended migration dataset

Dataset B contains all **284 paths** and preserves the original evidence classifications:

| Evidence classification | Paths | Intended use |
|---|---:|---|
| `title_mentions_cathode` | 71 | Primary provisional cathode population |
| `unresolved` | 159 | Retained literature-review and auxiliary population |
| `electrolyte_evidence` | 54 | Generic-barrier or domain-shift population; excluded from a cathode claim |

Dataset B has 175 Li and 109 Na paths, 169 formulas, 80 host framework keys, 111 ion–framework groups, and 56 identifiable paper keys. Six paths lack a paper key and title.

**Advantages**

- It gives broader chemistry and structure coverage.
- It has enough framework groups for a preliminary generic-barrier validation exercise.
- It can test whether conclusions from Dataset A depend on its narrow chemistry.

**Limitations**

- It is not a verified cathode dataset.
- Role classification is unresolved for 159 paths.
- Several high-barrier outliers and repeated framework families are concentrated within individual papers.
- Results from Dataset B cannot be presented as cathode performance.

No records should be silently transferred between A and B. Dataset A is a defined subset of B, and every Dataset B result must retain the three evidence classifications.

## 2. Feature-availability audit

All requested source columns are complete in both datasets. Composition parsing also succeeds for every row. “Available” below refers to a valid value in the current table; it does not guarantee availability during Materials Project screening.

| Feature | Dataset A valid | Dataset B valid | Available for a new MP candidate? | Leakage or deployment concern |
|---|---:|---:|---|---|
| Mobile-ion identity | 100% | 100% | Yes; present in the MP Li/Na electrode tables | No direct target leakage. Ion remains confounded with chemistry and paper coverage. |
| Chemical composition | 100% | 100% | Yes; MP provides charged, discharged, and framework formulas | Use normalized elemental fractions. An opaque formula or framework category would memorize families. |
| Transition-metal composition | 100% derivable; a transition metal is present in 100% | 100% derivable; a transition metal is present in 73.2% | Yes, from composition | Safe as elemental fractions under grouped validation. Oxidation state is not available and must not be inferred. |
| Anion-element chemistry | 100% derivable | 100% derivable | Yes, from composition | O/F/S/Se/halogen presence or fractions are available. Formal anion charge and local bonding are not. |
| `crystal_class` | 100% | 100% | Not in the same taxonomy | Manual labels may encode paper or material role. Dataset B includes the role-like class `ELECTROLYTES`. |
| Detailed lattice, symmetry, coordination, bottleneck descriptors | 0% | 0% | Some standard descriptors are obtainable from a full MP structure, but they have no matching values in this barrier table | Adding them only on the MP side would create a train/deployment schema mismatch. |
| `cell_volume_A3` | 100% | 100% | Conditional on retrieving the exact phase and cell | Cell volume depends on cell convention and repeats within frameworks. It is not target leakage but can encode source conventions. |
| `hop_length_A` | 100% | 100% | No, not from the current MP electrode summary alone | It requires an enumerated migration path. This is a deployment mismatch, not a conductivity label or direct barrier leak. |
| `mobile_ion_number_density_cm3` | 100% | 100% | Conditional on an exact structure and state of charge | Nominal occupied-ion density is not mobile-carrier density. Repeated structures require grouped validation. |
| `mobile_ion_atomic_fraction` | 100% | 100% | Yes, from composition | It is strongly redundant with ion number density and can destabilize feature-importance interpretation. |

The existing Li-only `li_insertion_electrodes_features.csv` confirms that composition, transition-metal identity, anion family, host volume, lattice parameters, crystal system, and space-group number can be calculated for its 2,774 cached Li MP structures. The current Li/Na MP summary has formulas and endpoint IDs for both ions, but it does not contain full structure descriptors or migration hop lengths. No equivalent raw Na structure cache is currently present.

### Recommended small initial feature set

For a **path-level** preliminary model, use:

1. `mobile_ion`, encoded as Li or Na.
2. `hop_length_A`.
3. Host elemental fractions parsed deterministically from `framework_key`. In Dataset A this creates fractions for the 14 observed host elements and carries transition-metal and anion-element chemistry without using opaque framework IDs.
4. `mobile_ion_number_density_cm3`.

This feature set predicts a barrier for an already enumerated migration path. It cannot score an arbitrary MP electrode document until a candidate path and hop length have been generated.

### Features reserved for ablation

- `mobile_ion_atomic_fraction`: compare against number density rather than assuming both add independent information. Their full-dataset Spearman correlation is 0.908.
- `cell_volume_A3`: test separately because it depends on cell representation and is partly absorbed by number density.
- `crystal_class`: test separately because Dataset A has 14 classes across only 71 paths, the taxonomy is not directly reproduced in MP, and Dataset B contains role-coded labels.

### Features excluded from model inputs

- `source_record_id`, `system_name`, source title, BibTeX key, source DOI, and source URL.
- `framework_key` or `material_formula` as opaque categorical strings.
- `cathode_role_evidence`, `auxiliary_barrier_training_status`, `direct_conductivity_training_eligible`, and `original_paper_review_required`.
- `computed_barrier_not_conductivity` and all review/status fields.
- `largest_mobile_displacement_fractional` and `displacement_margin_fractional` until their extraction role and candidate-time availability are explicitly defined.
- `migration_barrier_eV` anywhere except the target column.

Source and status columns must remain in the modeling table as provenance and grouping metadata, even though they are excluded from the feature matrix.

## 3. Validation strategy

| Split strategy | Dataset A groups | Dataset B groups | Leakage control | Assessment |
|---|---:|---:|---|---|
| Random path split | 71 rows | 284 rows | Poor | Related paths from the same framework and paper can appear in both train and test. Do not use for reported results. |
| Group by `(mobile_ion, framework_key)` | 19 | 111 | Better | Prevents the same ion–host framework from crossing folds, but the same paper and DFT protocol can remain on both sides. |
| Group by source paper | 14 | 56 identifiable papers plus 6 missing-paper rows | Better | Prevents direct paper overlap, but the same ion–framework can cross folds when studied by multiple papers. |
| Framework/paper connected group | **11** | **38 conservative groups** | Best available | Prevents both framework and paper connections from crossing folds. Recommended. |

Checkpoint 2 counted 41 Dataset B connected groups when every missing-paper record was assigned a separate placeholder. For validation, the six missing-paper rows should be kept in one conservative unknown-source group, leaving **38 groups**, until their sources are resolved.

### Recommended implementation

**Dataset A**

- Use three-fold validation on the 11 connected groups.
- Balance folds by path count and ion where possible, without splitting a connected group.
- Add leave-one-connected-group-out results as a sensitivity analysis.
- Do not reserve a permanent final holdout yet. With only 11 connected groups, doing so would leave too little independent information for model development.

The connected-group sizes are 32, 10, 6, 5, 5, 3, 3, 2, 2, 2, and 1 paths. Fold-level metrics will therefore be sensitive to which fold receives A-C01.

**Dataset B**

- Use five-fold validation over the 38 conservative connected groups.
- Keep the cathode, unresolved, and electrolyte classifications for stratified reporting.
- Apply the feature schema and hyperparameter choices defined from Dataset A without using Dataset B to inflate a cathode performance claim.

Group-balanced metrics should give each connected group equal weight in addition to reporting path-weighted error. Path-weighted error alone would allow A-C01 and other large groups to dominate the result.

## 4. Modeling-readiness decision

### 1. Which dataset should be used for the initial XGBoost model?

Use **Dataset A, the 71 provisional cathode paths**, as the primary exploratory population. Use Dataset B only as a separately labeled generic-barrier sensitivity analysis.

### 2. Which features should be included?

Use mobile-ion identity, hop length, normalized host elemental fractions from `framework_key`, and nominal mobile-ion number density.

### 3. Which features should be excluded?

Exclude identifiers, source fields, evidence and review flags, opaque formula/framework categories, and conductivity-related flags. Keep cell volume, atomic fraction, and crystal class out of the core model and test them only through predeclared ablations.

### 4. What is the recommended validation strategy?

Use framework/paper connected groups: three-fold connected-group validation plus leave-one-group-out sensitivity for Dataset A. Never report a random path split as the primary result.

### 5. Is the current dataset sufficient for an exploratory baseline?

**Yes, narrowly.** It is sufficient to test whether a compact XGBoost pipeline can learn path-level DFT barriers under strict grouped validation. It is not sufficient for a stable estimate of performance on unfamiliar cathode families or for selecting synthesis candidates.

### 6. Required limitations for preliminary results

- Cathode status is provisional and based on title evidence.
- All 71 paths still require original-paper review.
- There are only 11 independent framework/paper connected groups.
- One group supplies 45.1% of Dataset A.
- The feature table lacks local coordination, bottleneck size, oxidation states, defects, and verified state-of-charge descriptors.
- Hop length is unavailable for ordinary MP candidates without a path-generation step.
- Li and Na occupy different chemistry and paper distributions.
- Extreme barriers are retained and can strongly affect path-weighted error.
- The target is a calculated migration barrier, not diffusivity or ionic conductivity.

## Final recommendation

**Proceed with the 71-path provisional cathode dataset for an exploratory, path-level XGBoost baseline.** Use the 284-path dataset only as an auxiliary generic-barrier analysis with its role classifications preserved. The next model must use connected framework/paper groups and the compact feature set above.

Additional verified cathode paths are required before reporting dependable generalization or using the model for materials selection. If the intended deployment is direct ranking of arbitrary Materials Project candidates, the prediction target or workflow must later be revised to address the missing migration-path and hop-length information.
