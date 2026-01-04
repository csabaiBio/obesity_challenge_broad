# Method Description

This submission implements a State-Space Model (SSM) based approach for predicting perturbation effects on cell differentiation trajectories. The model treats cellular differentiation as a dynamical system where perturbations deflect the natural trajectory of gene expression changes.

The approach consists of three main components. First, we compute control centroids from unperturbed cells and per-gene deltas representing the effect of each perturbation. Second, we project these deltas into a low-dimensional latent space using Principal Component Analysis (PCA). Third, we train a neural state-space model that learns to map program proportions (pre_adipo, adipo, lipo, other) to latent-space effects.

The state-space model implements a discrete-time dynamical system that evolves an internal hidden state over multiple time steps. Given a perturbation's program proportions as input, the model iteratively updates its hidden state. The final state is then decoded back to gene expression space using the learned PCA components. This allows the model to capture how perturbations dynamically affect the differentiation trajectory rather than assuming a static linear relationship.

For cell-level predictions, we sample synthetic cells from a diagonal Gaussian distribution centered at the predicted mean expression, with variance estimated from the training data. For program proportion predictions, we use a simple lookup strategy that retrieves proportions from the training data when available, otherwise falling back to global averages.

# Rationale

Traditional regression approaches assume that the effect of a perturbation is a static function of program features. However, biological differentiation is inherently a temporal process where cells transition through states over time. The state-space model framework naturally captures this by modeling the evolution of cell state as a dynamical system.

The key insight is that perturbations can deflect the differentiation trajectory at different points. For example, a knockout might halt the transition from pre_adipo to adipo, which requires modeling the temporal dynamics rather than just a static mapping. The SSM learns how perturbations enter the system and affect state evolution, similar to how control theory models external inputs affecting system dynamics.

By working in a latent space defined by PCA, we reduce the dimensionality of the problem while preserving the most important variation in perturbation effects. This makes the SSM training more stable and allows it to learn richer relationships between program proportions and expression changes.

The choice of a small neural network (64-dimensional hidden state, 16 time steps) balances model capacity with generalization. We found that larger models tended to overfit, while this configuration provides good performance on held-out perturbations.

# Data and Resources Used

The model is trained exclusively on the provided competition training data: obesity_challenge_1.h5ad. This dataset contains single-cell RNA sequencing measurements for cells under various gene perturbations, along with program proportion annotations (pre_adipo, adipo, lipo, other) for each cell.

We use the control cells (labeled "NC" for negative control) to establish a baseline expression profile. Perturbed cells are used to compute per-gene effect vectors (deltas) by comparing their centroids to the control centroid. Only perturbations with at least 20 cells are included in training to ensure reliable statistics.

The model uses standard Python scientific computing libraries: NumPy for numerical operations, Pandas for data manipulation, Scanpy and AnnData for single-cell data handling, PyTorch for neural network training, and scikit-learn for PCA decomposition. All dependencies are listed in requirements.txt.

No external datasets, pre-trained models, or additional resources beyond the competition-provided training data are used. The model architecture and hyperparameters were determined through local validation on the provided local ground truth split.



