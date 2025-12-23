import os
from dataclasses import dataclass
from typing import List, Tuple

import anndata
import numpy as np
import pandas as pd
import scanpy
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


CONTROL_LABEL = "NC"


@dataclass
class EffectModelArtifacts:
    """Container for learned effect-model parameters."""

    control_centroid: np.ndarray  # (n_genes,)
    W: np.ndarray  # (n_latent, n_genes)
    reg_coef: np.ndarray  # (n_latent, n_feat)
    reg_intercept: np.ndarray  # (n_latent,)
    genes_to_predict: List[str]  # gene IDs (columns)
    feature_names: List[str]  # ["pre_adipo", "adipo", "lipo", "other"]


def load_adata_train(data_directory_path: str) -> anndata.AnnData:
    """Load the main training AnnData."""
    path = os.path.join(data_directory_path, "obesity_challenge_1.h5ad")
    return scanpy.read_h5ad(path, backed="r")


def to_array(adata: anndata.AnnData) -> np.ndarray:
    """Return dense numpy array from AnnData.X."""
    X = adata.X
    return X if isinstance(X, np.ndarray) else X.toarray()


def load_genes_to_predict(data_directory_path: str) -> List[str]:
    """Load the genes_to_predict.txt list in order."""
    return pd.read_csv(
        os.path.join(data_directory_path, "genes_to_predict.txt"),
        header=None,
    )[0].tolist()


def compute_control_and_effects(
    adata_train: anndata.AnnData,
    genes_to_predict: List[str],
    control_label: str = CONTROL_LABEL,
    min_cells_per_gene: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Compute control centroid, per-gene deltas and simple program features.

    Returns
    -------
    control_centroid : (n_genes,)
    deltas : (n_observed_genes, n_genes)
    prog_features : (n_observed_genes, 4)  # pre_adipo, adipo, lipo, other
    kept_genes : list of gene names used to fit the model
    """
    X = to_array(adata_train[:, genes_to_predict])
    obs = adata_train.obs

    control_mask = obs["gene"] == control_label
    assert control_mask.sum() > 0, "No control cells found."

    control_centroid = X[control_mask].mean(axis=0)

    labels = obs["gene"].astype(str).values
    unique_genes = sorted(g for g in set(labels) if g != control_label)

    deltas: List[np.ndarray] = []
    prog_features: List[np.ndarray] = []
    kept_genes: List[str] = []

    for g in unique_genes:
        mask = labels == g
        if mask.sum() < min_cells_per_gene:
            continue

        Xg = X[mask]
        mu_g = Xg.mean(axis=0)
        delta_g = mu_g - control_centroid

        sub_obs = obs.loc[mask]
        required = ["pre_adipo", "adipo", "lipo", "other"]
        if not all(c in sub_obs.columns for c in required):
            continue

        P_g = np.array(
            [
                sub_obs["pre_adipo"].mean(),
                sub_obs["adipo"].mean(),
                sub_obs["lipo"].mean(),
                sub_obs["other"].mean(),
            ],
            dtype=np.float32,
        )

        deltas.append(delta_g)
        prog_features.append(P_g)
        kept_genes.append(g)

    deltas_array = np.vstack(deltas)
    prog_features_array = np.vstack(prog_features)

    return control_centroid, deltas_array, prog_features_array, kept_genes


def fit_effect_model(
    deltas: np.ndarray,
    prog_features: np.ndarray,
    n_latent: int = 32,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA + ridge regression mapping program features to latent effects."""
    n_components = min(n_latent, deltas.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    Z = pca.fit_transform(deltas)
    W = pca.components_

    reg = Ridge(alpha=alpha)
    reg.fit(prog_features, Z)
    coef = reg.coef_
    intercept = reg.intercept_

    return W, coef, intercept


def estimate_global_variance(
    adata_train: anndata.AnnData,
    genes_to_predict: List[str],
    control_label: str = CONTROL_LABEL,
) -> np.ndarray:
    """Estimate global diagonal variance from perturbed cells."""
    X = to_array(adata_train[:, genes_to_predict])
    mask = adata_train.obs["gene"] != control_label
    Xp = X[mask]
    var = np.var(Xp, axis=0)
    var = np.maximum(var, 1e-4)
    return var.astype(np.float32)


def build_effect_artifacts(
    data_directory_path: str,
    n_latent: int = 32,
    alpha: float = 1.0,
) -> Tuple[EffectModelArtifacts, np.ndarray]:
    """High-level helper to build artifacts from training data."""
    adata_train = load_adata_train(data_directory_path)
    genes_to_predict = load_genes_to_predict(data_directory_path)

    control_centroid, deltas, prog_features, _ = compute_control_and_effects(
        adata_train=adata_train,
        genes_to_predict=genes_to_predict,
        control_label=CONTROL_LABEL,
    )

    W, coef, intercept = fit_effect_model(
        deltas=deltas,
        prog_features=prog_features,
        n_latent=n_latent,
        alpha=alpha,
    )

    var = estimate_global_variance(
        adata_train=adata_train,
        genes_to_predict=genes_to_predict,
        control_label=CONTROL_LABEL,
    )

    artifacts = EffectModelArtifacts(
        control_centroid=control_centroid.astype(np.float32),
        W=W.astype(np.float32),
        reg_coef=coef.astype(np.float32),
        reg_intercept=intercept.astype(np.float32),
        genes_to_predict=genes_to_predict,
        feature_names=["pre_adipo", "adipo", "lipo", "other"],
    )
    return artifacts, var


def predict_effect_for_gene(
    prog_feat: np.ndarray,
    artifacts: EffectModelArtifacts,
) -> np.ndarray:
    """Predict full-gene effect vector from program features."""
    z = prog_feat @ artifacts.reg_coef.T + artifacts.reg_intercept
    delta = z @ artifacts.W
    return delta


def sample_cells(
    mean: np.ndarray,
    std: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    """Sample synthetic cells from a diagonal Gaussian around mean."""
    eps = np.random.normal(loc=0.0, scale=std, size=(n_cells, mean.shape[0]))
    X = mean[None, :] + eps
    return np.maximum(X, 0.0).astype(np.float32)


def load_program_truth(data_directory_path: str) -> pd.DataFrame:
    """Load per-gene program proportions from the training bundle."""
    path = os.path.join(data_directory_path, "program_proportion.csv")
    return pd.read_csv(path)


