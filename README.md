# Physics-Informed Neural Modeling and Multiscale Simulations for Li/Na Battery Cathodes

## Current cathode scope (November 2026)

The senior design target is **ion transport within practical intercalation cathodes**, starting with Li and Na. The moving ion is an explicit input; voltage, capacity, computed phase stability, volume change, and synthesis evidence are separate material-selection checks. The first 33 notebook sections document an earlier **solid-electrolyte** conductivity prototype and must not be interpreted as a validated cathode predictor or used to rank cathodes.

The cathode data audit and exploratory analysis are in notebook Sections 34–35. A [literature DFT-NEB dataset](https://doi.org/10.1038/s41597-025-06196-x) provides 619 downloadable migration-path records; coordinate changes identify 175 Li and 109 Na paths. Paper titles explicitly mention cathodes for 71 of these paths (38 Li, 33 Na), representing only 19 ion/framework groups from 14 papers. A [Materials Project insertion-electrode](https://docs.materialsproject.org/apps/explorer-apps/battery-explorer/tutorial) summary contains 2,774 Li and 416 Na computed electrode documents. The Li cache is from MP version 2026.04.13 and the Na query from 2026.09.28. The [source audit](data/processed/cathode_data_source_audit.csv), [source provenance](data/processed/cathode_source_provenance.csv), [barrier leads](data/processed/cathode_migration_barrier_leads.csv), [MP Li/Na electrode summary](data/processed/mp_li_na_insertion_electrode_summary.csv), and [composition-level joins](data/processed/cathode_barrier_mp_framework_matches.csv) preserve citations and match limits.

Section 35 screens an **exploratory pool of 979 MP electrode documents** using 2.5–4.5 V, at least 100 mAh/g, endpoint hull energies at most 0.10 eV/atom, and volume change at most 15%. The [pool table](data/processed/cathode_mp_design_pool_eda.csv) includes charged/discharged phase IDs, ion-content range, electrochemical properties, MP experimental-structure links, and explicit evidence flags. Of these 979 records, 48 discharged phases have an MP experimental-structure link; [Kononova et al.'s text-mined synthesis recipes](https://doi.org/10.1038/s41597-019-0224-1) add 406 formula/paper leads covering 112 MP records, with original paper DOIs in the [recipe table](data/processed/cathode_synthesis_recipe_evidence.csv). A recipe formula match does not verify a phase or synthesis route. The [EDA figure](figures/figure_17_cathode_transport_and_synthesis_eda.png) and [association table](data/processed/cathode_transport_eda_associations.csv) show the calculated transport descriptors and the limited independent sample size. The [data dictionary](data/processed/cathode_data_dictionary.csv) defines units and evidence levels.

We also checked the [Huang–Cole text-mined battery database](https://doi.org/10.1038/s41597-020-00602-2): 7,168 extracted conductivity records contain 616 exact formula-name matches to this MP pool, but only 15 distinct formulas and no verified ionic cathode conductivity labels. Conductivity type and phase are often unresolved, and at least one cathode paper has obvious range-parsing errors. The larger [Lian et al. computed Li/Na barrier archive](https://doi.org/10.57760/sciencedb.j00113.00321) is recorded as a future proxy lead, not counted as ingested data. **No direct experimental cathode conductivity labels are currently accepted.** The repository retains the earlier electrolyte prototype until a phase-verified cathode training set is finalized; the present tables are an EDA snapshot, not a validated predictor.

The next data gate is to assemble experimental **cathode** transport measurements with ion, composition/state of charge, temperature, measurement method, and original DOI. GITT chemical diffusivity, tracer diffusivity, migration barriers, and ionic conductivity remain different targets. The old electrolyte labels are excluded from direct cathode supervision; barrier data can support an auxiliary physics task after phase and role review.

## Earlier electrolyte prototype (archived working notes)

This senior-design project develops a physics-constrained neural network to predict room-temperature lithium-ion conductivity from composition and verified crystal structure. The practical deliverable is three to five reviewable material-design cards: experimentally indexed parent phases, controlled composition changes, stability and synthesis evidence, uncertainty, and proposed tests. Detailed multiscale simulations follow the four-week ML phase.

![Physics-informed ionic-conductivity model](figures/figure_01_physics_informed_conductivity_model.png)

## Current model decision

The existing two-branch neural network is a **diagnostic baseline**, not a material-selection model. Its structured training set has 159 verified conductivity–CIF pairs but only 68 connected material/publication groups. It accepts composition and six structure descriptors and returns a numerical room-temperature conductivity estimate in S/cm as well as a fast-conductor score. Section 31 demonstrates inference on the MP review pool and saves exploratory predictions in `data/processed/mp_review_pool_exploratory_conductivity_predictions.csv`; these are not validated material recommendations. The official test split has influenced earlier iterations, so future model choices use training-only group folds and require a new external-family holdout.

[McHaffie et al.](https://doi.org/10.1039/D5DD00052A) demonstrate occupancy-aware transfer learning for small experimental structure datasets, while [OBELiX](https://doi.org/10.1039/D5DD00441A) documents overlap among sources and overfitting of large graph networks. We therefore plan an occupancy-aware graph encoder pretrained on LiTraj's computed migration barriers for Materials Project structures, followed by a small experimental-conductivity classifier and regressor. A physics-based Arrhenius head requires verified multi-temperature labels; single-temperature measurements cannot identify a migration barrier separately from the prefactor. This plan will be tested against the current scratch network before any MP conductivity ranking.

We have now identified a concrete, larger pretraining task in [LiTraj](https://doi.org/10.1038/s41524-025-01571-z): **12,807 distinct Li-containing MP structures** with computed 1D/2D/3D percolation barriers. Its downloadable BVEL13k index is in `data/processed/litraj_bvel13k_index.csv`; full structures remain in the source archive and a local ignored cache. The three barriers can train a graph encoder with the physical ordering $0\le E_{1D}\le E_{2D}\le E_{3D}$ before fine-tuning on experimental conductivity. This does **not** turn computed barriers into measured conductivity labels. Its chemistry covers 40 of 43 chemical systems in our MP review pool, though old numeric MP IDs and current opaque IDs cannot be joined directly; see `data/processed/litraj_bvel13k_coverage_audit.csv`.

The [CaltechDATA public CSV](https://data.caltech.edu/records/23mvv-6gk43) contains 571 rows. After a conservative formula/DOI and extrapolation-flag audit against OBELiX, 169 rows (162 formulas) remain **provisional**; original measurement conditions and phase identities still need review. Their measured values, ICSD identifiers, source-paper DOIs, and dataset DOI are in `data/processed/caltech_provisional_external_labels.csv`. See [Figure 13](figures/figure_13_effective_label_scale.png) and `data/processed/caltech_obelix_overlap_summary.csv`. Materials Project adds unlabeled structures and feasibility metadata, not measured conductivity labels.

The [Laskowski–McHaffie–See supplementary table](https://authors.library.caltech.edu/records/64j4e-7fx68) adds **288 provisional, nonoverlapping room-temperature measurements** across 278 formulas after conservative extraction and official-test-reference exclusion. Each row records the PDF page, original reference number, conductivity, and any reported activation energy or ICSD identifier in `data/processed/laskowski_provisional_external_labels.csv`. The unified `data/processed/experimental_conductivity_label_inventory.csv` now has **906 rows** (449 OBELiX training + 169 Caltech + 288 Laskowski) and explicitly marks the 457 external rows as ineligible for current training; only the original **159** training labels have verified public structures. [Figure 14](figures/figure_14_experimental_data_expansion.png) and `data/processed/external_conductivity_source_audit.csv` show this distinction. The much larger [Shon–Min text-mined source](https://github.com/shonyeajin/Ionical_conductivity_database) lacks measurement temperatures and contains ambiguous extracted formulas, so its records were not mixed into room-temperature supervision.

**Data-quality gate (Sections 32–33).** Of the 906 inventory rows, 159 are verified structure–conductivity training pairs, 290 additional OBELiX rows support composition-only work, and 457 external rows need original-paper and phase review. The current tables record no measurement method or uncertainty; 16 of 45 repeated nominal formula–space-group training groups vary by more than 1 dex. The LiTraj proxy labels are complete but have a long 3D-barrier tail, and its author test split overlaps training chemical systems. The new quality audit, manual-review queue, and [Figure 15](figures/figure_15_conductivity_data_quality.png) define the next data work before model refinement.

## Research objective

Ionic conductivity is commonly measured experimentally or estimated using computationally expensive atomistic simulations. This project first builds a data-driven screening model that predicts conductivity and quantifies uncertainty, then uses verified evidence to prioritize a smaller set of materials for detailed physical simulation.

The machine-learning target is

$$
y=\log_{10}\!\left(\frac{\sigma_{\mathrm{ion}}}{1\ \mathrm{S\,cm^{-1}}}\right).
$$

The current room-temperature network includes an explicit Li-site-density transport layer,

$$
\widehat\sigma=\frac{q^2n_{\mathrm{Li}}}{k_{\mathrm B}T}\,D_{\mathrm{eff}},
\qquad D_{\mathrm{eff}}=10^{g_\theta(x)}\ \mathrm{cm^2\,s^{-1}},
$$

where $g_\theta$ is a neural head and the positive effective transport factor absorbs unknown mobile-ion fraction and transport correlations; it is not a measured self-diffusion coefficient. A separate activation energy cannot be identified from one temperature. When multi-temperature data become available, temperature dependence can be tested through the Arrhenius relation

$$
\widehat{\log_{10}\sigma(T)}=
\widehat{\log_{10}\sigma_0}
-\frac{\widehat E_a}{k_{\mathrm B}T\ln 10},
$$

where the neural network learns a positive activation energy $\widehat E_a$ and conductivity prefactor $\widehat{\sigma_0}$ from the material representation.

## Modeling approach

The implemented diagnostic model is a **small two-branch, occupancy-aware neural network with a physical transport layer**. The next research model uses transfer learning with disorder-aware crystal sites; an Arrhenius extension remains conditional on multi-temperature data:

- Composition and occupancy-aware structural features describe Li-site density, local geometry, and anion chemistry.
- A neural head predicts a positive effective transport factor; the physical layer combines it with Li density to produce room-temperature conductivity.
- The frozen-CHGNet ordered-structure pilot is an existing ablation; the next encoder must retain fractional site occupancies and be evaluated on the larger structured cohort.
- Activation-energy, prefactor, and Arrhenius physics heads are added only after multi-temperature labels make them identifiable.
- A multi-seed ensemble estimates model uncertainty.
- Direct-conductivity and composition-only multilayer perceptrons provide neural-network baselines.
- A second screening head learns the measured $10^{-4}$ S/cm threshold; its score is not calibrated as a probability.

All measurements for the same material remain in the same validation fold. A stronger generalization test holds out complete chemical systems so that reported performance reflects unfamiliar chemistry rather than duplicate-material leakage.

## Data foundation and project advancement

The first supervised source is [OBELiX](https://doi.org/10.1039/d5dd00441a), a peer-reviewed dataset of 599 experimental room-temperature lithium solid-electrolyte records. The official data include composition, space group, lattice parameters, source DOI, and 321 associated CIF structures and are distributed under CC BY 4.0 through the [NRC-Mila repository](https://github.com/NRC-Mila/OBELiX).

The project uses the literature benchmark as a foundation and adds the following developments:

- A provenance-preserving project schema with explicit unit, source, phase, structure-match, and nominal-temperature fields.
- Correct handling of 37 censored upper-bound values instead of treating them as exact regression labels.
- Connected validation groups that keep records sharing either material identity or source DOI together.
- Transfer learning and uncertainty calibration designed for the small structure-labeled subset.
- An identifiability gate that prevents fitting physical-looking Arrhenius parameters from room-temperature-only data.
- A path from ML-selected variables and materials to later DFT, MD, and phase-field simulations.

## Connection to multiscale simulation

The current four-week phase is entirely focused on data analysis and machine learning. After the model identifies influential variables and representative materials, the project can progress across physical scales:

1. **Density functional theory (DFT):** calculate defect energetics, stable structures, and selected lithium migration barriers.
2. **Molecular dynamics (MD):** estimate temperature-dependent lithium diffusion and conductivity.
3. **Phase-field modeling (PFM):** study how mesoscale structure, interfaces, and spatial heterogeneity affect transport.

The neural network is therefore a screening and hypothesis-generation tool, not a replacement for these simulations.

## Current contents

- `battery_materials_project.ipynb` — target definition, data contract, literature-backed audit, neural-network mathematics, and staged model design.
- `data/processed/ionic_conductivity_intake_template.csv` — schema for conductivity labels and source metadata.
- `data/raw/obelix_{all,train,test}.csv` — official OBELiX source tables and leakage-controlled split.
- `data/processed/obelix_conductivity_clean.csv` — 562 exact labels normalized to the project schema with linked validation groups.
- `data/processed/caltech_provisional_external_labels.csv` — 169 source-attributed experimental measurements with original-paper and dataset DOIs, pending phase and measurement review.
- `data/processed/caltech_obelix_overlap_summary.csv` — reproducible counts for source overlap and usable-data scale.
- `data/processed/laskowski_provisional_external_labels.csv` — 288 additional PDF-extracted conductivity values with page and original-reference provenance, pending manual verification.
- `data/processed/experimental_conductivity_label_inventory.csv` — unified 906-row source-attributed label inventory with current-training eligibility flags.
- `data/processed/experimental_conductivity_quality_{audit,summary}.csv` — row-level model eligibility, provenance and conflicting-value flags, plus aggregate counts.
- `data/processed/data_quality_review_queue.csv` — prioritized original-paper, label, and phase review cases.
- `data/processed/data_quality_model_error_slices.csv` — descriptive grouped-validation errors by quality flag.
- `data/processed/model_input_data_quality_summary.csv` — LiTraj proxy and MP inference-input coverage checks.
- `data/processed/external_conductivity_source_audit.csv` — published source sizes, inventory contributions, and inclusion decisions.
- `data/processed/litraj_bvel13k_index.csv` — author-split 12,807-structure Li migration-barrier proxy dataset for neural pretraining.
- `data/processed/litraj_bvel13k_coverage_audit.csv` — chemistry coverage and MP-identifier audit for the proxy dataset.
- `data/processed/obelix_mlp_cv_results.csv` — linked-group validation results for six neural-network configurations.
- `data/processed/obelix_mlp_test_predictions.csv` — untouched-test predictions, ensemble spread, errors, and train-calibrated intervals.
- `data/processed/obelix_structure_manifest.csv` — provenance, hashes, parsing results, occupancy flags, and metadata checks for all 599 OBELiX records.
- `data/processed/obelix_mp_structure_match_pairs.csv` — composition, symmetry, and geometric results for each shortlisted OBELiX–MP pair.
- `data/processed/obelix_mp_structure_match_summary.csv` — one matching status per OBELiX record, including records outside the screened MP subset.
- `data/processed/mp_obelix_formula_search_metadata.csv` — Materials Project phases returned by the wider formula query without screening filters.
- `data/processed/obelix_mp_wide_match_{pairs,summary}.csv` — geometric comparisons and one status per OBELiX record from that wider query.
- `data/processed/obelix_occupancy_sensitivity.csv` — changes in composition, Li content, and atom count under two diagnostic site-occupancy thresholds.
- `data/processed/obelix_chgnet_frozen_features.csv` — 64-dimensional pretrained CHGNet features for 41 ordered direct-source experimental CIFs, with source hashes and model version.
- `data/processed/obelix_chgnet_pilot_{cv,test}.csv` — connected-group model selection and predictions on the same official held-out records for both neural heads.
- `data/processed/obelix_occupancy_structural_features.csv` — versioned, occupancy-aware Li-neighborhood and density descriptors for 202 validated direct-source CIFs.
- `data/processed/obelix_structural_mlp_{cv,test,group_audit}.csv` — same-cohort neural comparison, official test predictions, and equal-weight-per-group error audit.
- `data/processed/obelix_retrospective_selection_audit.csv` — group-diverse top-five selection check at 10⁻⁴ and 10⁻³ S/cm on the official held-out data.
- `data/processed/obelix_physics_head_train_cv.csv` — training-only, connected-family out-of-fold predictions for direct and physical transport-layer neural heads.
- `data/processed/obelix_two_branch_train_cv.csv` — train-only out-of-fold conductivity predictions and screening scores for the compact direct and physical two-branch networks.
- `data/processed/obelix_two_branch_family_audit.csv` — family-level errors and fast-conductor retrieval audit for the two-branch prototype.
- `data/processed/obelix_refinement_{train_cv,summary}.csv` — training-only feature and loss ablations on linked-family and chemical-component folds.
- `data/processed/obelix_chemistry_components.csv` — connected groups joining records by publication/material family or exact chemical system.
- `data/processed/obelix_train_li_network_proxy.csv` — versioned, training-only periodic Li-site connectivity descriptors and source-CIF hashes.
- `data/processed/obelix_path_proxy_pilot_{cv,summary}.csv` — grouped neural ablation of the literature-guided path proxy.
- `data/processed/mp_neural_input_contract.csv` — each MP field's role as model input, optional ablation, feasibility gate, or identifier.
- `data/processed/mp_review_pool_exploratory_conductivity_predictions.csv` — numerical neural conductivity predictions and measured-analogue comparisons for MP review phases; exploratory and unvalidated.
- `data/processed/obelix_train_structural_eda.csv` — record-level, between-family, and within-family structural associations with conductivity.
- `data/processed/mp_obelix_descriptor_shift.csv` — matched experimental-CIF versus MP-relaxed descriptor differences for 35 phases.
- `data/processed/mp_senior_design_review_pool.csv` — 115 experimentally indexed, approximately in-domain MP phases in training-observed chemical systems, for literature review only.
- `data/processed/mp_review_pool_structural_inputs.csv` — full MP structure descriptors, source hashes, and input-completeness flags for all 115 review phases.
- `data/processed/mp_li_solid_electrolyte_candidates.csv` — 5,068 Materials Project entries passing the broad stability, band-gap, and composition prescreen.
- `data/processed/mp_li_solid_electrolyte_domain_audit.csv` — candidate applicability flags; it intentionally contains no conductivity predictions.
- `figures/figure_01_physics_informed_conductivity_model.png` — project architecture figure using the UIUC brand palette.
- `figures/figure_02_obelix_data_audit.png` — label, split, structure-coverage, and repeatability audit.
- `figures/figure_03_mlp_and_mp_screening_gate.png` — neural-baseline validation and Materials Project screening decision.
- `figures/figure_04_obelix_structure_manifest.png` — structure provenance, metadata validation, and partial-occupancy audit.
- `figures/figure_05_obelix_mp_structure_matching.png` — first-pass structure matching counts in the UIUC palette.
- `figures/figure_06_mp_wide_and_occupancy.png` — wider matching yield and occupancy sensitivity in the UIUC palette.
- `figures/figure_07_frozen_chgnet_pilot.png` — same-cohort neural comparison on the ordered experimental CIFs.
- `figures/figure_08_occupancy_structural_pilot.png` — record-weighted and group-balanced errors on the expanded structure cohort.
- `figures/figure_09_training_structural_eda.png` — training-only between-family and within-family associations plus input correlations.
- `figures/figure_15_conductivity_data_quality.png` — effective label scale, repeated-value spread, and metadata completeness in UIUC colors.
- `figures/figure_10_two_branch_train_validation.png` — UIUC-color comparison of family-balanced error and fast-conductor retrieval.
- `figures/figure_11_neural_refinement_ablation.png` — UIUC-color feature and screening-loss ablation on linked-family folds.
- `figures/figure_12_mp_review_structure_coverage.png` — UIUC-color MP input funnel and experimental-versus-MP structure-representation audit.
- `figures/figure_13_effective_label_scale.png` — UIUC-color supervised-data scale and external-source overlap audit.
- `figures/figure_14_experimental_data_expansion.png` — UIUC-color data expansion and PDF-extraction quality funnel.

The notebook retains the Materials Project setup for candidate crystal structures and structural descriptors. Because the existing local insertion-electrode cache does not contain ionic-conductivity labels, a separate conductivity table is required before training.

### Materials Project initialization

The notebook preserves the original cache-first Materials Project workflow. Normal execution reads `data/raw/mp_li_insertion_electrodes.json` without making a network request. To intentionally refresh the source, place `MP_API_KEY` in a local `.env` file and set `FORCE_MP_REFRESH = True`; the notebook negotiates fields supported by the current insertion-electrode endpoint and records the database version and returned fields in `data/raw/download_metadata.json`.

A second cache uses the official Materials Project summary endpoint to identify Li-containing, nonmetallic compounds with 3–6 elements, band gap at least 1 eV, and energy above hull no greater than 0.05 eV/atom. The current cache contains 5,068 unique materials from database version `2026.09.28`; set `REFRESH_MP_CANDIDATES = True` only when an intentional API refresh is required. The API key is never written to the notebook or project data files.

## Four-week ML plan

| Week | Primary outcome |
|---|---|
| 1 | Freeze the target and schema; collect, normalize, deduplicate, and audit conductivity records. |
| 2 | Resolve material identities and structures; create leakage-free splits; train the composition-only neural baseline. |
| 3 | Audit MP input variables, train a compact physical transport-layer neural head, and compare with direct neural baselines using connected-family folds. |
| 4 | Run feature and family ablations, validate selection reliability, and prepare three to five synthesis-aware design cards only if the evidence supports them. |

## Current status

The target, data contract, OBELiX audit, composition-only neural baseline, Materials Project consultation, and structure manifest now run end to end. The cleaned table contains 562 exact positive labels split into 449 training and 113 held-out test records. Of these, 285 have a public OBELiX CIF (224 direct-source and 61 close-analogue structures), while two additional records have external ICSD references but no public local file.

The selected 32-unit MLP reaches 1.66 dex MAE on the official held-out test set, compared with 1.55 dex for the training-median reference, and therefore fails the predefined screening gate. No Materials Project conductivity ranking is reported.

All 321 public CIFs parse successfully, 288 pass the conservative composition-and-symmetry metadata audit, and 249 contain partial or mixed occupancies requiring an explicit representation decision. Only 41 exact-label records are currently direct-source, metadata-validated, and already ordered. The first pass against the **screened** Materials Project pool found 31 ordered direct-source geometric matches. A wider formula query without the screening filters raises that count to 35, but remains limited to exact formula queries. Those 35 are already in the 41-record ordered cohort and add no new conductivity labels.

For 161 audited, disordered direct-source records, changing the occupancy threshold from 0.50 to 0.70 changes the resolved representation for 100 records. Median normalized composition drift is 0.19 and 0.33, respectively; the policies even remove every Li site for 52 and 83 records. These diagnostics rule out simple threshold rounding as universal CHGNet preprocessing.

The first frozen-CHGNet pilot uses 41 ordered direct-source CIFs (33 official training, 8 official test). The CHGNet neural head reaches about 2.96 dex test MAE, versus about 2.59 dex for a composition neural head and 3.54 dex for the training median. These are the **same eight test records** for all methods. Two source-reported conductivity extremes, 4.4 × 10⁻¹⁸ and 3.2 × 10⁻³ S/cm, account for over half the total absolute error. The paired group-bootstrap interval for the CHGNet-minus-composition MAE includes zero, so this pilot does not establish a reliable model ranking. Figure 7 now shows each material's error. The public OBELiX release includes positionally perturbed CIFs, but the authors' [supplementary benchmark](https://www.rsc.org/suppdata/d5/dd/d5dd00441a/d5dd00441a1.pdf) found no significant effect of this perturbation for its tested geometric models. The Materials Project conductivity ranking remains closed while we enlarge the labeled structure cohort and validate the representation.

The next neural comparison uses 202 direct-source, metadata-validated CIFs with exact labels (159 official training and 43 official test), including 161 with partial occupancy. Eleven interpretable structural descriptors retain site occupancies and exclude Li–Li separations below 1.5 Å that can represent mutually exclusive split sites. The structure-augmented neural head reaches **1.84 dex record-weighted test MAE**, versus **2.43 dex** for its same-cohort composition head and **1.82 dex** for the training median. The 43 test records form only 20 connected groups; when each group receives equal weight, the three MAEs are **2.73, 2.87, and 2.76 dex**, respectively. One ten-record family supplies much of the apparent record-level gain, and the paired group-bootstrap interval for the structure-minus-composition difference includes zero. This is a diagnostic advance in coverage, not a validated screening model.

For design selection, the notebook now asks whether a model can find materials above **10⁻⁴ S/cm** while selecting at most one material per connected family. Retrospectively, four of the structure model's top five distinct-family test picks meet that threshold, compared with three of five for the composition model. This is an exploratory result from only 20 test groups (random-choice tail probability about 0.058), and the structure model misses known fast conductors including Li₇P₃S₁₁ and Li₃InCl₆. At **10⁻³ S/cm**, only two of its top five picks qualify; just three independent direct-source training groups have a label above that stricter threshold. The next ML goal is therefore independent-family discovery and false-negative control, followed by applicability and stability checks on Materials Project candidates. The official test results have been inspected across several iterations, so future model choices must use training-only connected-family validation and a fresh external holdout for final confirmation. No MP conductivity ranking is released.

The first physical transport-layer ablation uses only the 159 structured training records and 68 connected groups. Its group-balanced cross-validation MAE is **1.86 dex**, essentially tied with **1.85 dex** for the same-size direct neural head. The physical factorization is therefore an explicit modeling constraint, not evidence of better accuracy. Training-only exploratory analysis shows that vacancy fraction and Li occupancy correlate with conductivity **between** material families but much less **within** families; correlated inputs are reduced through ablations rather than treated as universal rules. A paired check of 35 accepted MP–OBELiX structures quantifies descriptor shifts before MP inference.

The two-branch prototype uses separate composition and six-descriptor structure branches, a shared 16-unit layer, and joint conductivity and fast-conductor heads. It trains on the same 159 records using five connected-family folds, equal family loss weight, and two random seeds. The direct head obtains **1.93 dex** family-balanced MAE and identifies **7/10** fast families among the ten highest screening scores; the physical head obtains **2.06 dex** and **8/10**, respectively. The physical-minus-direct MAE difference is **+0.13 dex**, with a paired family-bootstrap descriptive 95% interval of **−0.07 to +0.41 dex**. Both heads find four fast families in their top five, and the training-fold median reference has 2.48 dex family-balanced MAE. This is a mixed result on a small cohort, so neither head is ready for MP ranking.

The refinement audit tests a **density-separated** physical head, so Li-site density enters the conductivity equation but not the mobility branch. It reaches **2.07 dex** family-balanced error and **7/10** fast-family retrieval on linked-family folds; the direct head remains better for regression at **1.93 dex**. In a harder holdout that removes exact chemical-system overlap, the density-separated head has **2.03 dex** error versus **2.27 dex** for the direct head, but the paired family-bootstrap interval includes zero. A compact chemistry representation and a lower screening-loss weight worsen the linked-family result. One chemistry fold has no fast records and another has 37, so these results cannot support a calibrated deployment claim.

The literature-guided path pilot adds a **periodic Li-site network** descriptor motivated by [three-dimensional migration-landscape work](https://doi.org/10.1063/5.0305688). It reduces the density-separated model's linked-family error from **2.07 to 1.86 dex**, while fast-family retrieval falls from **7/10 to 6/10**. On chemical-component holdout, error is essentially unchanged at **2.03 dex**, and retrieval rises from **4/10 to 6/10**. The paired error intervals include zero; moreover, network rank changes for **30 of 159** records when the edge cutoff moves from 3.5 to 4.5 Å. This is a provisional geometric proxy, not a migration barrier or a validated conductivity predictor. [Experimental descriptor evidence](https://doi.org/10.1080/14686996.2020.1824985) also identifies processing conditions as important, but measurement method and uncertainty are absent for all 159 training records. The next data task is phase-verified literature extraction of synthesis, grain-boundary, and measurement metadata, followed by a fresh external material-family holdout.

The MP input contract separates transport inputs (composition and verified structure-derived occupancy/geometry) from feasibility filters (band gap, metallicity, hull energy, and experimental structure index). The cached 5,068-entry broad prescreen narrows to 115 review phases after experimental-index, approximate-domain, and observed-chemical-system checks; 102 have hull energy at or below 25 meV/atom. We have now fetched **all 115 full MP structures**, and **110** contain every proposed neural input without imputation. Thirty-six share a reduced formula with the structured training cohort, but formula overlap does not verify an identical phase or add a new conductivity measurement. All 115 MP structures are ordered, compared with only 33 of 159 labeled training CIFs, so the model faces a material representation shift. These are **not recommended materials**: actual synthesis reports, exact phase identity, processing conditions, and electrode compatibility remain to be checked before design cards are issued.

The [MP materials API](https://docs.materialsproject.org/downloading-data/using-the-api/getting-started) provides structures and computed screening properties but no standard experimental room-temperature ionic-conductivity field. More MP structures therefore enlarge the candidate pool, not the supervised label count. [MPContribs](https://docs.materialsproject.org/services/mpcontribs) can host project-specific contributed measurements. The [OBELiX paper](https://doi.org/10.1039/d5dd00441a) reports substantial overlap among experimental sources; our new inventory removes exact formula overlap and official-test source references, while phase identity, original measurement conditions, and remaining cross-source relationships still require manual review.

The notebook uses `chgnet==0.4.2` and its pretrained `0.3.0` checkpoint for this pilot. Install the package in the notebook's environment with `python -m pip install chgnet==0.4.2`; the saved feature CSV allows cache-first reruns when package and CIF hashes match.

## Scope and limitations

The first model is limited to inorganic solid-state lithium-ion conductors. Its accuracy may be constrained by inconsistent literature measurements, limited chemical diversity, uncertain phase matching, and sparse structures. Ionic conductivity alone does not determine electrochemical stability, processability, cycle life, or full-cell performance.

## Methodological references

- Therrien et al., “OBELiX: a curated dataset of crystal structures and experimentally measured ionic conductivities for lithium solid-state electrolytes,” *Digital Discovery* (2026), [doi:10.1039/d5dd00441a](https://doi.org/10.1039/d5dd00441a).
- Deng et al., “CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling,” *Nature Machine Intelligence* (2023), [doi:10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3).
- Sendek et al., “Holistic computational structure screening of more than 12,000 candidates for solid lithium-ion conductor materials,” *Energy & Environmental Science* (2017), [doi:10.1039/C6EE02697D](https://doi.org/10.1039/C6EE02697D).
- Sendek et al., “Machine Learning-Assisted Discovery of Solid Li-Ion Conducting Materials,” *Chemistry of Materials* (2019), [doi:10.1021/acs.chemmater.8b03272](https://doi.org/10.1021/acs.chemmater.8b03272).
- Marcolongo and Marzari, “Ionic correlations and failure of Nernst–Einstein relation in solid-state electrolytes,” *Physical Review Materials* (2017), [doi:10.1103/PhysRevMaterials.1.025402](https://doi.org/10.1103/PhysRevMaterials.1.025402).
- Hashizume et al., “Prediction of solid-state lithium-ion conductivity using the three-dimensional potential landscape and distribution of lithium ions,” *APL Machine Learning* (2026), [doi:10.1063/5.0305688](https://doi.org/10.1063/5.0305688).
- Wu et al., “Essential structural and experimental descriptors for bulk and grain boundary conductivities of Li solid electrolytes,” *Science and Technology of Advanced Materials* (2020), [doi:10.1080/14686996.2020.1824985](https://doi.org/10.1080/14686996.2020.1824985).
