import os
import json
from typing import List, Tuple

import anndata
import numpy as np
import pandas as pd

from model_utils import (
    CONTROL_LABEL,
    EffectModelArtifacts,
    build_effect_artifacts,
    load_adata_train,
    load_genes_to_predict,
    load_program_truth,
    predict_effect_for_gene,
    sample_cells,
)


ARTIFACTS_FILE = "effect_model_artifacts.npz"
META_FILE = "effect_model_meta.json"


def _save_artifacts(
    artifacts: EffectModelArtifacts,
    var: np.ndarray,
    model_directory_path: str,
) -> None:
    os.makedirs(model_directory_path, exist_ok=True)
    np.savez_compressed(
        os.path.join(model_directory_path, ARTIFACTS_FILE),
        control_centroid=artifacts.control_centroid,
        W=artifacts.W,
        coef=artifacts.reg_coef,
        intercept=artifacts.reg_intercept,
        var=var,
        genes_to_predict=np.array(artifacts.genes_to_predict),
        feature_names=np.array(artifacts.feature_names),
    )
    with open(os.path.join(model_directory_path, META_FILE), "w") as f:
        json.dump({"n_genes": len(artifacts.genes_to_predict)}, f)


def _load_artifacts(model_directory_path: str) -> Tuple[EffectModelArtifacts, np.ndarray]:
    path = os.path.join(model_directory_path, ARTIFACTS_FILE)
    data = np.load(path, allow_pickle=True)
    artifacts = EffectModelArtifacts(
        control_centroid=data["control_centroid"],
        W=data["W"],
        reg_coef=data["coef"],
        reg_intercept=data["intercept"],
        genes_to_predict=data["genes_to_predict"].tolist(),
        feature_names=data["feature_names"].tolist(),
    )
    var = data["var"].astype(np.float32)
    return artifacts, var


def train(
    data_directory_path: str,
    model_directory_path: str,
) -> None:
    """Train effect model and persist artifacts for later inference."""
    artifacts, var = build_effect_artifacts(
        data_directory_path=data_directory_path,
        n_latent=32,
        alpha=1.0,
    )
    _save_artifacts(artifacts, var, model_directory_path)


def infer(
    data_directory_path: str,
    prediction_directory_path: str,
    prediction_h5ad_file_path: str,
    program_proportion_csv_file_path: str,
    model_directory_path: str,
    predict_perturbations: List[str],
    genes_to_predict: List[str],
    cells_per_perturbation: int = 100,
) -> None:
    """Generate prediction.h5ad and predict_program_proportion.csv."""
    os.makedirs(prediction_directory_path, exist_ok=True)

    artifacts, var = _load_artifacts(model_directory_path)

    # Ensure runner-provided column order matches training artifacts.
    assert genes_to_predict == artifacts.genes_to_predict, "genes_to_predict mismatch."

    std = np.sqrt(var)
    control = artifacts.control_centroid

    # For var dataframe we reuse training var for the genes_to_predict slice.
    adata_train = load_adata_train(data_directory_path)
    var_df = adata_train[:, genes_to_predict].var.copy()

    prog_truth = load_program_truth(data_directory_path)
    cols = artifacts.feature_names
    prog_map = {
        row["gene"]: np.array([row[c] for c in cols], dtype=np.float32)
        for _, row in prog_truth.iterrows()
    }
    global_mean = np.array([prog_truth[c].mean() for c in cols], dtype=np.float32)

    n_genes = len(genes_to_predict)
    n_perts = len(predict_perturbations)
    n_cells = n_perts * cells_per_perturbation

    X_pred = np.zeros((n_cells, n_genes), dtype=np.float32)
    obs_gene: List[str] = []

    for i, g in enumerate(predict_perturbations):
        start = i * cells_per_perturbation
        end = (i + 1) * cells_per_perturbation

        prog_feat = prog_map.get(g, global_mean)
        delta = predict_effect_for_gene(prog_feat, artifacts)
        mu = control + delta

        Xg = sample_cells(mu, std, cells_per_perturbation)
        X_pred[start:end] = Xg
        obs_gene.extend([g] * cells_per_perturbation)

    prediction = anndata.AnnData(
        X=X_pred,
        obs={"gene": np.array(obs_gene)},
        var=var_df,
    )
    prediction.write_h5ad(prediction_h5ad_file_path)

    # Program proportion predictions: per-gene, normalized and clipped.
    rows = []
    for g in predict_perturbations:
        prog = prog_map.get(g, global_mean)
        prog = np.maximum(prog, 1e-6)
        prog = prog / prog.sum()
        rows.append(
            {
                "gene": g,
                "pre_adipo": prog[0],
                "adipo": prog[1],
                "lipo": prog[2],
                "other": prog[3],
            }
        )

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(program_proportion_csv_file_path, index=False)


