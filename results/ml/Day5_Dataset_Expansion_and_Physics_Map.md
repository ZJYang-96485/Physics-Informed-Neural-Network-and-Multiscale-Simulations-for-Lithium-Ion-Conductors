# Checkpoint 7 — Dataset Expansion and Physics-Informed Transport Map

## Executive conclusion

This checkpoint added a real cathode experiment, two computational migration sources at distinct fidelity levels, and a source-level Na-cathode review queue. The additions are useful, but they do **not** yet constitute one training table: the experimental files contain electrode time series or impedance spectra, while Devi/LiTraj, CHGNet, UMA, and BVSE/BVEL supply different computational targets.

The defensible increase in **verified independent cathode frameworks is currently zero**. The new computational sources contain 6,928 reported materials (Lian) and 383 ICSD structures (Saravanan), but neither source labels cathode role; the one newly acquired experimental composition is NCM831205. These sources are candidate coverage until phase, material role, and path identity are reviewed.

The source-level inventory is in [`transport_dataset_registry.csv`](../../data/processed/transport_dataset_registry.csv), and the Na-cathode queue is in [`na_cathode_transport_literature_review_queue.csv`](../../data/processed/na_cathode_transport_literature_review_queue.csv).

## 1. Experimental NMC cathode data

### Files acquired

The requested two structured Parquet files and associated metadata from [Zenodo record 15537285](https://doi.org/10.5281/zenodo.15537285) were preserved under `data/raw/experimental_cathode_transport/`. Record 15537285 does not contain frequency or impedance columns, so the small official companion EIS files and metadata from [record 15537281](https://doi.org/10.5281/zenodo.15537281) were also acquired. All files are distributed under CC BY 4.0.

| Local file or source | Rows | Columns | What is present |
|---|---:|---:|---|
| `18650-LG-3500-MJ1-cathode-GITT-EIS-Ch.parquet` | 1,542,076 | 6 | Time, current, voltage, temperature, command, row index |
| `18650-LG-3500-MJ1-cathode-GITT-EIS-Dc.parquet` | 1,411,305 | 6 | Same schema for discharge |
| `18650_LG_3500_MJ1_EIS_cathode_charge.parquet` | 2,592 | 5 | Frequency, real/imaginary impedance, phase, row index |
| `18650_LG_3500_MJ1_EIS_cathode_discharge.parquet` | 2,592 | 5 | Same schema for discharge |
| Zenodo record and RO-Crate JSON files | — | — | Provenance, license, protocol context, and file checksums |

The GITT+EIS time-series schema is:

- `indices`
- `timepoints [s]`
- `currents [A]`
- `voltages [V]`
- `T [°C]`
- `Command`

The EIS schema is:

- `indices`
- `frequencies [Hz]`
- `real_impedances [Ω]`
- `imaginary_impedances [Ω]`
- `phases [rad]`

### Protocol audit

The active material is LiNi0.83Co0.12Mn0.05O2 (NCM831205) in a three-electrode cell at nominal 25 °C.

| Property | Charge | Discharge |
|---|---:|---:|
| Direction commands | 36 charge pulses | 36 discharge pulses |
| Elapsed time | 552,404.7 s (6.39 d) | 423,690.7 s (4.90 d) |
| Mean measured temperature | 25.863 °C | 25.883 °C |
| Temperature range | 25.055–26.026 °C | 24.620–26.106 °C |
| Median pulse duration | 814.94 s | 814.85 s |
| Pulse-duration range | 814.93–2,037.31 s | 814.83–2,037.20 s |
| Post-pulse pause range | 3,630–25,441 s | 3,630–7,272 s |
| Electrochemical voltage magnitude | 3.608–4.522 V | 3.601–4.266 V |

The raw signs are device conventions: charge voltages are stored negative, while the charge/discharge current signs differ between related records. Raw files were not altered or silently sign-corrected.

The EIS files contain 36 spectra per direction and 72 frequencies per spectrum, from 0.01 Hz to 2 MHz. They do not include timestamps or an explicit SOC column; exact spectrum-to-SOC mapping needs a protocol/index crosswalk.

### What is and is not a label

- The time-series Parquets do **not** contain a diffusion-coefficient column, capacity column, or explicit SOC column.
- A GITT-derived coefficient must be recalculated from pulse and relaxation voltage changes using verified active-material mass, molar volume or particle geometry, electrode area, and the assumptions of the selected diffusion model. It is an apparent/chemical diffusivity under that model, not direct ionic conductivity.
- The EIS files contain raw spectra, not fitted charge-transfer resistance, Warburg coefficient, diffusivity, or intrinsic conductivity. An equivalent-circuit or physics-based impedance model and verified geometry are required.
- The stored temperature allows quality checks around 25 °C, but one nominal temperature cannot independently identify an Arrhenius barrier and prefactor.

### GITT-only record 15537295

The related [GITT-only dataset](https://doi.org/10.5281/zenodo.15537295) was inspected without retaining its two large Parquets. It contains separate measurement IDs, 36 pulses per direction, the same six-column time-series schema, and no `Set-Temp`/intermittent-EIS segments. It is therefore a **complementary clean GITT protocol**, not a duplicate of record 15537285. It also contains no precomputed diffusion coefficient or impedance spectrum.

The GITT-only record provides more detailed cell/protocol metadata, but those details must not automatically be assigned to the separate GITT+EIS measurements merely because both use the harvested NCM831205 cathode.

## 2. Computational migration datasets

### Lian et al.: fine-tuned CHGNet NEB

The [paper](https://doi.org/10.1088/1674-1056/ae88e2) reports 6,928 Li/Na ion-conductor materials and 20,320 inequivalent paths. The [Science Data Bank record](https://doi.org/10.57760/sciencedb.j00113.00321) is CC BY 4.0 and exposes one 2.16 GB ZIP (`Database_Code.zip`, MD5 `533254e43bb2af04480c095e72bfe5a9`). It was not downloaded because the checkpoint prohibits unnecessary multi-gigabyte transfers.

A byte-range ZIP audit found 26 members and approximately 8.19 GB uncompressed content. The main `database.csv` is approximately 8.13 GB uncompressed and has 58 columns. Its header and one complete record show:

- Materials Project ID, formula, crystal system, space group, and band gap.
- Mobile-ion type and ion-site counts/concentrations for pristine, supercell, and NEB structures.
- Embedded pymatgen structure JSON for pristine/defect structures and path images.
- Direction-specific CHGNet barriers, convergence flags, and trajectory structures.
- Optional aggregate/directional DFT-barrier and defect-energy fields.

The exact Li/Na distribution was not computed because that requires streaming the full 2.16 GB archive. The schema has no cathode/electrolyte role field, and the paper frames the data as solid-state ion-conductor screening. Accordingly, the reported 6,928 materials are **not** counted as cathodes. CHGNet-NEB barriers remain a separate MLIP fidelity level from DFT-NEB.

### Saravanan et al.: UMA-MLIP CI-NEB

The small energy CSV and metadata from the [Figshare record](https://doi.org/10.6084/m9.figshare.32221506) were downloaded to `data/raw/computational_transport/`. The [open-access paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13787076/) describes UMA-sm-OMat24 climbing-image NEB calculations for Li, Na, K, and Mg conductors.

The raw `UMA_NEB_energy.csv` contains 22,365 image energies:

- 383 unique ICSD structures.
- 3,195 unique paths.
- Exactly seven images per path: endpoints plus five intermediate images.
- Columns: `St_list` and absolute `UMA-energy` in eV.
- No missing rows in either column.

The optional 104 MB ZIP was inspected by byte range but not downloaded. It contains 22,365 CONTCAR-like image structures with filenames matching the energy table. Thus path trajectories and structures are available when descriptor generation begins.

The supplement provides path descriptors including barrier, KRA barrier, distance contraction, anion/cation decomposition, bond-valence terms, bottleneck radius, coordination, bond geometry, packing density, and local-distortion metrics. Its ion-specific S1 worksheets contain 1,216 Li, 634 Na, 814 K, and 553 Mg descriptor rows (3,217 total). This is 22 more rows than the 3,195 raw paths, so those counts cannot yet be assigned as the exact raw-path ion distribution; the discrepancy must be reconciled by ICSD/path matching.

For a path with initial, transition, and final energies, the kinetically resolved activation energy is

\[
E_{\mathrm{KRA}}=E_{\mathrm{TS}}-\frac{E_{\mathrm{IS}}+E_{\mathrm{FS}}}{2}.
\]

It removes the endpoint-energy asymmetry from the forward barrier and must not be mixed with ordinary forward DFT barriers. The paper's distance-contraction descriptor measures the largest reduction in a migrating ion–neighbor distance from equilibrium to the transition state; its anion/cation decomposition is a strong candidate for local path representation.

### Existing Devi and LiTraj sources

The existing [Devi et al. dataset](https://doi.org/10.1038/s41597-025-06196-x) still supplies 284 locally parsed Li/Na DFT-NEB paths: 175 Li and 109 Na. Only 71 paths have provisional cathode-title evidence; 159 remain unresolved and 54 have electrolyte evidence. No additional record can be promoted without paper review.

The [LiTraj source](https://doi.org/10.1038/s41524-025-01571-z) was already fully audited in Checkpoint 6:

| LiTraj subset | Records | Structures | Target/fidelity |
|---|---:|---:|---|
| nebDFT2k | 1,681 paths | 867 | DFT CI-NEB barrier and trajectory |
| nebBVSE122k | 122,421 paths | 11,290 | BVSE path barrier proxy |
| BVEL13k | 12,807 structures | 12,807 | 1D/2D/3D percolation barriers |

All three are Li-only and nested, not additive. The 224 nebDFT structures/350 paths in 137 composition groups absent from the current provisional Li subset remain the best **candidate** expansion pool, but none is a verified cathode phase yet. DFT path barriers, BVSE path barriers, and structure-level percolation thresholds remain separate targets.

## 3. Experimental Na-cathode literature queue

The starting [Zuo et al. Perspective](https://doi.org/10.1021/acsmaterialslett.1c00827) summarizes broad method-dependent ranges for Na0.44MnO2, layered NaxMnO2/NaxCoO2, and Na3V2(PO4)3. Those ranges are useful for source discovery but are not row-level labels.

Nine original-study leads are now queued with method and verification status. Highest-priority sources include:

- Na0.44MnO2 impedance and kinetics: [Ruffo et al.](https://doi.org/10.1016/j.electacta.2013.07.009) and [Ju et al.](https://doi.org/10.1016/j.jpowsour.2018.05.086).
- Layered NaxMnO2: [Zuo et al.](https://doi.org/10.1016/j.ensm.2019.11.024) for GITT and [Shibata et al.](https://doi.org/10.7567/APEX.7.067101) for composition- and temperature-dependent EIS.
- Layered NaxCoO2: [Han et al.](https://doi.org/10.1016/j.jpowsour.2014.11.150) and [Shibata et al.](https://doi.org/10.7567/APEX.6.097101), the latter covering 303–349 K and varying Na content.
- Na3V2(PO4)3: [Cong et al.](https://doi.org/10.1021/acs.inorgchem.0c01129) for GITT/CV with public supporting information and [Jiang et al.](https://doi.org/10.1002/cnma.201600111) as a composite-electrode lead.

No numeric points were copied from review tables. Each queue row remains ineligible for training until the original protocol, temperature, electrode/particle geometry, SOC or Na content, method equation/model, and numeric data source are verified. GITT, EIS, PITT, and CV estimates will be retained as distinct measurement types.

## 4. Physics-informed transport framework

### Atomic hop to jump diffusivity

For hop/path \(p\), transition-state theory gives

\[
\Gamma_p(T,x)=\nu_p(T,x)\exp\!\left[-\frac{E_{m,p}(x)}{k_B T}\right],
\]

where \(x\) denotes composition/SOC, \(\nu_p\) is the attempt frequency, and \(E_{m,p}\) is the path-specific migration barrier. A random-walk representation can be written

\[
D_J=f\,g\,a^2\nu\exp\!\left(-\frac{E_m}{k_B T}\right),
\]

where \(a\) is hop distance, \(g\) contains dimensionality/path multiplicity conventions, and \(f\) is a correlation factor when the chosen definition includes correlations. For a network, the scalar expression should be replaced by a rate matrix or a diffusion tensor built from every symmetry-distinct \(\Gamma_p\) and hop vector. A single low barrier is insufficient if it does not belong to a percolating network.

The variables have distinct physical roles:

- **Migration barrier:** exponential control of each elementary rate; depends on local coordination, bottleneck geometry, oxidation/SOC, defects, and lattice relaxation.
- **Attempt frequency:** vibrational prefactor; not a universal constant and may vary by path and state.
- **Hop distance/vector:** sets the displacement contributed by a successful hop and the anisotropic diffusion tensor.
- **Correlation:** successive hops are not necessarily independent; vacancy ordering and back-jumps can reduce tracer motion or alter collective charge motion.
- **Network connectivity:** identifies whether low-barrier paths form 1D, 2D, or 3D long-range transport and whether defects block the network.
- **Mobile-ion concentration:** counts charge-carrying defects/available carriers, not simply total Li/Na sites. It changes with SOC, vacancy fraction, disorder, and defect chemistry.

### Diffusivity definitions

| Quantity | Meaning | Appropriate evidence |
|---|---|---|
| Jump diffusivity \(D_J\) | Random-walk mobility implied by elementary hop rates and geometry | NEB barriers plus prefactors, hop vectors, and network model |
| Tracer diffusivity \(D^*\) | Long-time mean-square displacement of tagged ions | Isotope/tracer experiment or MD; includes single-particle correlations |
| Charge diffusivity \(D_\sigma\) | Collective motion that transports net charge | Charge-current correlations or conductivity/Haven analysis |
| Chemical diffusivity \(\widetilde D\) | Relaxation of a concentration/chemical-potential gradient | GITT, PITT, finite-length impedance, or composition-resolved continuum model |

A common convention is

\[
H_R=\frac{D^*}{D_\sigma},
\qquad
\widetilde D=D_\sigma\,\Theta,
\qquad
\Theta=\frac{\partial\ln a_{\mathrm{ion}}}{\partial\ln c},
\]

where \(H_R\) is the Haven ratio and \(\Theta\) is the thermodynamic factor. Exact conventions vary, so every extracted dataset must record the definition used. Near phase transitions, ordering, or two-phase regions, \(\Theta\) and the assumptions behind a simple GITT coefficient can change sharply.

### Charge diffusivity to intrinsic conductivity

The Nernst–Einstein form is

\[
\sigma_{\mathrm{ion}}(T,x)=
\frac{q^2 c_{\mathrm{mobile}}(x)D_\sigma(T,x)}{k_B T}.
\]

This equation requires charge diffusivity and the concentration of mobile charge carriers. Substituting a migration barrier alone, a total crystallographic ion density, or a GITT chemical diffusivity without correlation and thermodynamic corrections is not justified.

### Intrinsic crystal to effective electrode transport

Effective electrode behavior additionally requires:

- Active-material particle size/shape and crystallographic orientation.
- Intragrain and grain-boundary transport, defects, coatings, and secondary phases.
- Active-material volume fraction, connected fraction, and network tortuosity.
- Electrolyte-filled porosity and pore tortuosity.
- Electronic-conductive additive/current-collector connectivity.
- Particle–particle and particle–electrolyte contact resistances.
- Electrode thickness, loading, calendering, and SOC/temperature gradients.

For example, a porous-electrolyte contribution may be approximated as \(\kappa_{\mathrm{eff}}\approx\kappa\varepsilon_{p}/\tau_{p}\), while an active-solid contribution requires a separate connected-solid factor. These are phase-specific effective-medium relations, not a replacement for a microstructure-resolved electrode model.

### Proposed physics-informed data flow

1. **Path level:** structure + endpoints/images → \(E_m\), hop vector, bottleneck, coordination change, distance contraction, endpoint asymmetry, and uncertainty/fidelity.
2. **Network level:** symmetry/path graph + multiple rates → directional \(D_J\), percolation class, and rate-limiting-path statistics.
3. **State level:** SOC/composition + vacancy/defect population + thermodynamic factor + correlations → \(D^*\), \(D_\sigma\), and \(\widetilde D\) with definitions preserved.
4. **Intrinsic conductivity:** \(D_\sigma\), mobile-carrier concentration, charge, and temperature → \(\sigma_{\mathrm{ion}}\).
5. **Electrode scale:** intrinsic phase transport + particle/electrode microstructure + contacts/electrolyte → effective electrode response comparable to EIS/GITT.

This hierarchy supports a future multi-task model without pretending that every source measures the same target. Cross-fidelity learning should use dataset/fidelity indicators, paired calibration where available, and separate output heads.

## 5. Dataset organization and next descriptors

The registry contains 11 source entries and explicitly records role, target, method, fidelity, structures/paths, temperature/SOC, identifiers, license, overlap, local status, and recommended usage. No source tables were merged.

The next descriptor work should prioritize:

1. **Local path geometry:** hop vector/distance, bottleneck radius, coordination at endpoints/transition state, anion/cation distance contraction, local free volume, and endpoint energy asymmetry.
2. **Network descriptors:** path multiplicity, dimensionality/percolation, barrier distribution, rate-limiting barrier, site connectivity, and vacancy-order sensitivity.
3. **State descriptors:** Li/Na content, vacancy fraction, oxidation-state proxies, lattice parameters/volume versus SOC, phase-transition flags, and mobile-carrier concentration definition.
4. **Experimental context:** temperature, direction, pulse/rest duration, current density, voltage/SOC, particle size, electrode loading/thickness, electrolyte, and extraction method.
5. **Fidelity/provenance:** DFT functional/settings, MLIP name/version, convergence flag, KRA versus forward barrier, BVSE/BVEL status, DOI, and structure/path identifiers.

## 6. What remains missing

- **Li cathodes:** only one newly acquired experimental composition; no validated GITT diffusivity series yet; no exact EIS–SOC crosswalk; no multi-temperature NMC experiment.
- **Na cathodes:** no original numeric series have yet passed protocol review; the queue has promising temperature/composition-resolved studies but no machine-ready labels.
- **Computational cathode coverage:** Lian and Saravanan material roles are unresolved; LiTraj candidate matches are composition-level rather than phase/path verified; Na lacks a LiTraj-like DFT trajectory set in the local repository.
- **Physics bridge:** attempt frequencies, Haven/correlation factors, thermodynamic factors, mobile defect concentrations, and microstructure are mostly absent.
- **Overlap:** ICSD↔MP structure mapping and endpoint matching are needed to identify duplicate physical paths across UMA, LiTraj, Lian, Devi, and MP candidates.

## Checkpoint decision

The newly acquired NMC experiment provides genuinely new **operating-state time-series and impedance information**, while Lian and Saravanan provide genuinely new **Li/Na or multi-ion path/trajectory and local-descriptor opportunities**. They do not yet add verified cathode frameworks or direct conductivity labels.

For the next checkpoint, the most useful action is descriptor and identity engineering: verify cathode roles, build MP/ICSD structure crosswalks, extract path-local geometry, and calculate traceable GITT/EIS features for NCM831205 under explicitly documented assumptions. Model training should wait until those distinctions are preserved in prepared tables.
