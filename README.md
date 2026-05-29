## Code for "Torus Graphs for Large Scale Neural Phase Analysis" [ICML 2026]

- Authors: [Jack Goffinet](https://jackgoffinet.github.io),  [Casey Hanks](https://scholar.google.com/citations?hl=en&user=1qPlymwAAAAJ), [David Carlson](https://carlson.pratt.duke.edu/)

### Install

Conda:

```bash
conda env create -f conda_requirements.yml
```

Alternatively using pip:
```bash
conda create -n torus python=3.11
conda activate torus
pip install -r requirements.txt
```
Then optionally install [IDTxl](https://github.com/pwollstadt/IDTxl) from source to run bivariate TE tests.

### Examples (``examples/`` subdirectory)
* ``tg_ssm_example.py``, Torus graph (TG) usage
* ``hmm_example.py`` Torus graph hidden Markov model (TG-HMM) usage
* ``artg_example.py`` autoregressive torus graph (AR-TG) usage
* ``mvte_example.py`` AR-TG multivariate transfer entropy (MVTE) estimation usage

### Experiments (``experiments/`` subdirectory)
* ``ssm_synthetic_heatmaps.py`` (Figure 2a,b)
* ``hmm_synthetic_heatmaps.py`` (Figure 2c,d)
* ``bvte_synthetic_comparison.py`` (Figure 3a,b)
* ``mvte_synthetic_heatmaps.py`` (Figure 3c,d)
* ``ssm_mouse_lfp_experiment.py`` (Figure 4)
* ``hmm_mouse_lfp_experiment.py`` (Figure 5)
* ``mvte_mouse_lfp_experiment.py`` (Figure 6)

### Source Code (``src/`` subdirectory)
* ``src.conditional_model_sgd`` stochastic gradient descent trainer for conditional TG models with von Mises outputs
* ``src.conditional_sgd`` learns conditional TG parameters with score-matching-style SGD
* ``src.fit_artg`` tools for fitting univariate and bivariate AR-TG models via MAP/BFGS optimizers
* ``src.fit_mv_artg`` stochastic score matching for multivariate AR-TG models using sequential batches
* ``src.hmm`` expectation-maximization routines for TG-based hidden Markov models
* ``src.hmm_eval`` alignment metrics, plots, and diagnostics for TG-HMM latent states
* ``src.imputation_model`` Gaussian conditioning model for imputing lagged cosine/sine features
* ``src.lfp_loader`` streaming loader that constructs Morlet-based LFP phase windows on the fly
* ``src.multivariate_transfer_entropy`` Monte Carlo estimator for AR-TG multivariate transfer entropy
* ``src.plots`` plotting helpers for circular statistics, complex covariances, and TG parameters
* ``src.reference_implementation`` Matlab TG reference algorithms ported to JAX
* ``src.sample`` MCMC samplers and utilities for drawing from torus graph distributions
* ``src.simulate_ar`` simulators and priors for autoregressive torus graphs
* ``src.simulate_hmm`` synthetic TG-HMM generator and sticky transition priors
* ``src.conditional_ssm`` score matching baseline for conditional TGs
* ``src.ssm`` stochastic score matching losses and optimizers for torus graphs
* ``src.stats`` circular sufficient statistics, Jacobians, and TG linear algebra utilities
* ``src.von_mises`` entropy, log-density, and helper functions for von Mises distributions

### Scripts (``scripts/`` subdirectory)
* ``scripts.align_sleep_spindles`` refines spindle detections with similarity-based ordering and QC plots
* ``scripts.find_sleep_spindles`` detects spindle events from LFPs and saves summary diagnostics
* ``scripts.make_schematic_figure`` reproduces the torus graph schematic using Morlet features and TG solves
* ``scripts.plot_lfp_ssm_figure`` assembles figures from score-matching TG fits on sleep LFPs
* ``scripts.plot_lfp_te_figure`` renders the multivariate transfer entropy LFP visualizations
* ``scripts.plot_spindle_hmm_figure`` creates PLV and TG-HMM visualizations for the sleep spindle figure

### Tests (``test/`` subdirectory)
* ``test_jacobian.py`` Tests the Jacobian math for stochastic score matching


### Paper BibTex
```bibtex
@inproceedings{Goffinet2026TorusGraphs,
  author       = {Goffinet, Jack and Hanks, Casey and Carlson, David E.},
  title        = {Torus Graphs for Large Scale Neural Phase Analysis},
  booktitle    = {International Conference on Machine Learning},
  year         = {2026},
  organization = {PMLR}
}
```

### Project Checklist
- [ ] Code on GitHub
- [ ] Camera-ready on arXiv
- [ ] Add links
