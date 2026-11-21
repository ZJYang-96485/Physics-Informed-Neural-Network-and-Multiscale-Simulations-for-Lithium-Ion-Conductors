# Physics-Informed Neural Network and Multiscale Simulations for Lithium-Ion Conductors

This senior-design project develops a physics-informed neural network to predict lithium-ion conductivity from chemical composition, crystal structure, temperature, and documented measurement conditions. The trained model will guide the selection of representative materials and variables for later multiscale simulations.

![Physics-informed ionic-conductivity model](figures/figure_01_physics_informed_conductivity_model.png)

## Research objective

Ionic conductivity is commonly measured experimentally or estimated using computationally expensive atomistic simulations. This project first builds a data-driven screening model that predicts conductivity and reports uncertainty, then uses the resulting trends to prioritize a smaller set of materials for detailed physical simulation.

The machine-learning target is

$$
y=\log_{10}\!\left(\frac{\sigma_{\mathrm{ion}}}{1\ \mathrm{S\,cm^{-1}}}\right).
$$

Temperature dependence is introduced through the Arrhenius relation

$$
\widehat{\log_{10}\sigma(T)}=
\widehat{\log_{10}\sigma_0}
-\frac{\widehat E_a}{k_{\mathrm B}T\ln 10},
$$

where the neural network learns a positive activation energy $\widehat E_a$ and conductivity prefactor $\widehat{\sigma_0}$ from the material representation.

## Modeling approach

The planned primary model is a **Physics-Informed Arrhenius Crystal Graph Neural Network (PI-ACGNN)**:

- A pretrained crystal graph encoder, initially CHGNet, represents atoms and their periodic local environments.
- Neural heads estimate activation energy, conductivity prefactor, and data-dependent uncertainty.
- An Arrhenius physics layer generates temperature-conditioned conductivity predictions.
- A multi-seed ensemble estimates model uncertainty.
- A composition-only multilayer perceptron and an unconstrained crystal graph network provide neural-network baselines.

All measurements for the same material remain in the same validation fold. A stronger generalization test holds out complete chemical systems so that reported performance reflects unfamiliar chemistry rather than duplicate-material leakage.

## Connection to multiscale simulation

The current four-week phase is entirely focused on data analysis and machine learning. After the model identifies influential variables and representative materials, the project can progress across physical scales:

1. **Density functional theory (DFT):** calculate defect energetics, stable structures, and selected lithium migration barriers.
2. **Molecular dynamics (MD):** estimate temperature-dependent lithium diffusion and conductivity.
3. **Phase-field modeling (PFM):** study how mesoscale structure, interfaces, and spatial heterogeneity affect transport.

The neural network is therefore a screening and hypothesis-generation tool, not a replacement for these simulations.

## Current contents

- `battery_materials_project.ipynb` — Day 1 target definition, data contract, validation rules, neural-network mathematics, and architecture.
- `data/processed/ionic_conductivity_intake_template.csv` — schema for conductivity labels and source metadata.
- `figures/figure_01_physics_informed_conductivity_model.png` — project architecture figure using the UIUC brand palette.

The notebook retains the Materials Project setup for candidate crystal structures and structural descriptors. Because the existing local insertion-electrode cache does not contain ionic-conductivity labels, a separate conductivity table is required before training.

## Four-week ML plan

| Week | Primary outcome |
|---|---|
| 1 | Freeze the target and schema; collect, normalize, deduplicate, and audit conductivity records. |
| 2 | Resolve material identities and structures; create leakage-free splits; train the composition-only neural baseline. |
| 3 | Transfer-learn the crystal graph encoder and compare unconstrained and Arrhenius-constrained networks. |
| 4 | Calibrate uncertainty, run ablations, analyze influential variables, and select materials for later simulations. |

## Current status

Day 1 is complete: the prediction target, unit convention, row definition, required metadata, validation groups, neural architecture, and figure standard are defined. The next step is conductivity-data ingestion and quality auditing.

## Scope and limitations

The first model is limited to inorganic solid-state lithium-ion conductors. Its accuracy may be constrained by inconsistent literature measurements, limited chemical diversity, uncertain phase matching, and sparse structures. Ionic conductivity alone does not determine electrochemical stability, processability, cycle life, or full-cell performance.
