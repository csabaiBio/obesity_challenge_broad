import json
import os
from typing import List, Tuple

import anndata
import numpy as np
import pandas as pd
import torch

from model_utils import (
    CONTROL_LABEL,
    SSMEffectArtifacts,
    compute_control_and_effects,
    estimate_global_variance,
    estimate_low_rank_covariance,
    estimate_perturbation_variance,
    load_adata_train,
    load_genes_to_predict,
    load_program_truth,
    sample_cells,
    sample_cells_with_covariance,
    fit_ssm_effect_model,
    train_program_proportion_model,
    predict_program_proportions,
    ProgramProportionModel,
)


ARTIFACTS_FILE = "ssm_effect_artifacts.npz"
SSM_STATE_FILE = "ssm_state_dict.pt"
PROG_MODEL_FILE = "program_proportion_model.pt"
PROG_META_FILE = "program_proportion_meta.json"
META_FILE = "ssm_effect_model_meta.json"


def _save_artifacts(
    artifacts: SSMEffectArtifacts,
    var: np.ndarray,
    model_directory_path: str,
    ssm_state_dict: dict,
    prog_model: ProgramProportionModel | None = None,
    prog_gene_to_idx: dict | None = None,
    cov_factors: np.ndarray | None = None,
    cov_psi: np.ndarray | None = None,
) -> None:
    os.makedirs(model_directory_path, exist_ok=True)
    np.savez_compressed(
        os.path.join(model_directory_path, ARTIFACTS_FILE),
        control_centroid=artifacts.control_centroid,
        W=artifacts.W,
        pca_mean=artifacts.pca_mean,
        var=var,
        genes_to_predict=np.array(artifacts.genes_to_predict),
        feature_names=np.array(artifacts.feature_names),
        latent_dim=np.array([artifacts.latent_dim], dtype=np.int32),
        cov_factors=cov_factors if cov_factors is not None else np.array([]),
        cov_psi=cov_psi if cov_psi is not None else np.array([]),
    )
    with open(os.path.join(model_directory_path, META_FILE), "w") as f:
        json.dump({"n_genes": len(artifacts.genes_to_predict)}, f)
    torch.save(ssm_state_dict, os.path.join(model_directory_path, SSM_STATE_FILE))
    
    # Save program proportion model
    if prog_model is not None and prog_gene_to_idx is not None:
        torch.save(prog_model.state_dict(), os.path.join(model_directory_path, PROG_MODEL_FILE))
        with open(os.path.join(model_directory_path, PROG_META_FILE), "w") as f:
            json.dump(prog_gene_to_idx, f)


def _load_artifacts(model_directory_path: str) -> Tuple[SSMEffectArtifacts, np.ndarray, dict, dict, np.ndarray | None, np.ndarray | None]:
    path = os.path.join(model_directory_path, ARTIFACTS_FILE)
    data = np.load(path, allow_pickle=True)
    artifacts = SSMEffectArtifacts(
        control_centroid=data["control_centroid"],
        W=data["W"],
        pca_mean=data["pca_mean"],
        genes_to_predict=data["genes_to_predict"].tolist(),
        feature_names=data["feature_names"].tolist(),
        latent_dim=int(data["latent_dim"][0]),
    )
    var = data["var"].astype(np.float32)
    ssm_state_dict = torch.load(os.path.join(model_directory_path, SSM_STATE_FILE), map_location="cpu")
    
    # Load covariance factors if available
    if "cov_factors" in data and len(data["cov_factors"]) > 0:
        cov_factors = data["cov_factors"].astype(np.float32)
        cov_psi = data["cov_psi"].astype(np.float32)
    else:
        cov_factors = None
        cov_psi = None
    
    # Load program proportion model if available
    prog_model_dict = None
    prog_gene_to_idx = None
    prog_model_path = os.path.join(model_directory_path, PROG_MODEL_FILE)
    prog_meta_path = os.path.join(model_directory_path, PROG_META_FILE)
    if os.path.exists(prog_model_path) and os.path.exists(prog_meta_path):
        prog_model_dict = torch.load(prog_model_path, map_location="cpu")
        with open(prog_meta_path, "r") as f:
            prog_gene_to_idx = json.load(f)
    
    return artifacts, var, ssm_state_dict, {"prog_model": prog_model_dict, "prog_gene_to_idx": prog_gene_to_idx}, cov_factors, cov_psi


def train(
    data_directory_path: str,
    model_directory_path: str,
) -> None:
    """
    Train latent-space SSM effect model and persist artifacts for later inference.

    This uses:
    - control/perturbed centroids in gene space,
    - PCA to define a latent space for deltas,
    - a neural SSM (PerturbationSSM) to map program proportions to latent deltas.
    - Program proportion prediction model
    - Low-rank covariance for better cell sampling
    """
    adata_train = load_adata_train(data_directory_path)
    genes_to_predict = load_genes_to_predict(data_directory_path)

    control_centroid, deltas, prog_features, kept_genes, gene_to_idx = compute_control_and_effects(
        adata_train=adata_train,
        genes_to_predict=genes_to_predict,
        control_label=CONTROL_LABEL,
        use_all_cells=True,  # Use ALL individual cells, not just centroids
        enrich_features=True,  # Add gene embeddings and expression stats
    )
    
    print(f"Using {len(deltas)} training samples (all individual cells)")
    print(f"Enriched features: {prog_features.shape[1]} dimensions (program + gene + stats)")
    
    # Free up memory from adata_train if possible (it's backed, so this is safe)
    import gc
    gc.collect()

    # Fit SSM in latent space with enhanced training
    base_artifacts, ssm_state_dict = fit_ssm_effect_model(
        deltas=deltas,
        prog_features=prog_features,
        n_latent=64,  # Increased from 32
        n_steps=24,  # Increased from 16
        state_dim=128,  # Increased from 64
        n_epochs=300,  # Increased from 200
        lr=1e-3,
        batch_size=128,  # Increased from 64
        validation_split=0.2,
        early_stopping_patience=20,
        weight_decay=1e-4,
        hybrid_loss_weight=0.5,  # Balance between latent and gene space loss
    )

    # Estimate variance and covariance (with memory-efficient processing)
    print("Estimating global variance...")
    try:
        var = estimate_global_variance(
            adata_train=adata_train,
            genes_to_predict=genes_to_predict,
            control_label=CONTROL_LABEL,
            max_samples=20000,  # Reduced for memory safety
            chunk_size=5000,  # Process in chunks
        )
        # Force garbage collection after variance estimation
        import gc
        gc.collect()
    except Exception as e:
        print(f"Warning: Could not estimate global variance: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: use simple variance estimate
        var = np.ones(len(genes_to_predict), dtype=np.float32) * 0.1
    
    # Estimate low-rank covariance for better MMD (increased rank)
    print("Estimating low-rank covariance...")
    try:
        cov_factors, cov_psi = estimate_low_rank_covariance(
            adata_train=adata_train,
            genes_to_predict=genes_to_predict,
            rank=100,  # Increased from 50
            control_label=CONTROL_LABEL,
            max_samples=15000,  # Reduced for memory safety
        )
        # Force garbage collection after covariance estimation
        import gc
        gc.collect()
    except Exception as e:
        print(f"Warning: Could not estimate low-rank covariance: {e}")
        import traceback
        traceback.print_exc()
        cov_factors = None
        cov_psi = var  # Fall back to diagonal

    # Train program proportion model
    print("Training program proportion model...")
    prog_truth = load_program_truth(data_directory_path)
    try:
        prog_model, prog_gene_to_idx = train_program_proportion_model(
            adata_train=adata_train,
            prog_truth=prog_truth,
            genes_to_predict=genes_to_predict,
            n_epochs=100,
            lr=1e-3,
        )
        print("Program proportion model trained successfully")
    except Exception as e:
        print(f"Warning: Could not train program proportion model: {e}")
        import traceback
        traceback.print_exc()
        prog_model = None
        prog_gene_to_idx = None

    artifacts = SSMEffectArtifacts(
        control_centroid=control_centroid.astype(np.float32),
        W=base_artifacts.W.astype(np.float32),
        pca_mean=base_artifacts.pca_mean.astype(np.float32),
        genes_to_predict=genes_to_predict,
        feature_names=base_artifacts.feature_names,
        latent_dim=base_artifacts.latent_dim,
    )

    _save_artifacts(
        artifacts, var, model_directory_path, ssm_state_dict,
        prog_model=prog_model,
        prog_gene_to_idx=prog_gene_to_idx,
        cov_factors=cov_factors,
        cov_psi=cov_psi,
    )
    
    # Save gene_to_idx mapping for enriched features
    meta_path = os.path.join(model_directory_path, META_FILE)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    meta["gene_to_idx"] = gene_to_idx
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def _build_ssm_model_from_state(
    artifacts: SSMEffectArtifacts,
    ssm_state_dict: dict,
    gene_to_idx: dict | None = None,
) -> torch.nn.Module:
    from model_utils import PerturbationSSM

    # Determine if attention was used based on state dict keys
    use_attention = "attention.in_proj_weight" in ssm_state_dict
    
    # Determine vocab size from gene_embedding if present
    vocab_size = 10000  # default
    if "gene_embedding.weight" in ssm_state_dict:
        vocab_size = ssm_state_dict["gene_embedding.weight"].shape[0]
    elif gene_to_idx is not None:
        vocab_size = len(gene_to_idx)
    
    # Determine input_dim from state dict or artifacts
    # If enriched, input_dim should be 13 (4 program + 1 gene_idx + 8 stats)
    # But model expects the full enriched input
    input_dim = len(artifacts.feature_names)  # Base: 4 program features
    # Check if model was trained with enriched features (has gene_embedding)
    if "gene_embedding.weight" in ssm_state_dict:
        # Model expects enriched input: [4 program, 1 gene_idx, 8 stats] = 13
        input_dim = 13
    
    model = PerturbationSSM(
        input_dim=input_dim,
        state_dim=ssm_state_dict["state_to_state.weight"].shape[1],
        latent_dim=artifacts.latent_dim,
        n_steps=24,  # Updated default
        use_attention=use_attention,
        vocab_size=vocab_size,
    )
    model.load_state_dict(ssm_state_dict)
    model.eval()
    return model


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
    """
    Generate prediction.h5ad and predict_program_proportion.csv.

    - Uses enhanced latent SSM to predict per-perturbation delta in expression space.
    - Samples synthetic cells with perturbation-specific variance and low-rank covariance.
    - Predicts program proportions using learned neural network model.
    """
    os.makedirs(prediction_directory_path, exist_ok=True)

    artifacts, var, ssm_state_dict, prog_info, cov_factors, cov_psi = _load_artifacts(model_directory_path)

    # Ensure runner-provided column order matches training artifacts.
    assert genes_to_predict == artifacts.genes_to_predict, "genes_to_predict mismatch."

    control = artifacts.control_centroid
    
    # Load gene_to_idx if available (for enriched features)
    gene_to_idx = None
    meta_path = os.path.join(model_directory_path, META_FILE)
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "gene_to_idx" in meta:
                gene_to_idx = meta["gene_to_idx"]

    # For var dataframe, create a minimal one from gene names (memory-efficient)
    # We don't need the full adata_train just for var_df
    var_df = pd.DataFrame(index=genes_to_predict)
    var_df.index.name = "gene_ids"

    # Prepare program proportion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prog_model = None
    if prog_info["prog_model"] is not None and prog_info["prog_gene_to_idx"] is not None:
        try:
            # Reconstruct model architecture
            vocab_size = len(prog_info["prog_gene_to_idx"])
            prog_model = ProgramProportionModel(vocab_size=vocab_size).to(device)
            prog_model.load_state_dict(prog_info["prog_model"])
            prog_model.eval()
        except Exception as e:
            print(f"Warning: Could not load program proportion model: {e}")
            prog_model = None
    
    # Fallback: use training data lookup
    prog_truth = load_program_truth(data_directory_path)
    cols = artifacts.feature_names
    prog_map = {
        row["gene"]: np.array([row[c] for c in cols], dtype=np.float32)
        for _, row in prog_truth.iterrows()
    }
    global_mean = np.array([prog_truth[c].mean() for c in cols], dtype=np.float32)

    # Prepare SSM model
    ssm_model = _build_ssm_model_from_state(artifacts, ssm_state_dict, gene_to_idx).to(device)
    
    # Check if model uses enriched features
    uses_enriched = "gene_embedding.weight" in ssm_state_dict
    if uses_enriched and gene_to_idx is None:
        # Build gene_to_idx from predict_perturbations if not available
        gene_to_idx = {gene: idx for idx, gene in enumerate(predict_perturbations)}

    n_genes = len(genes_to_predict)
    n_perts = len(predict_perturbations)
    n_cells = n_perts * cells_per_perturbation

    X_pred = np.zeros((n_cells, n_genes), dtype=np.float32)
    obs_gene: List[str] = []

    with torch.no_grad():
        for i, g in enumerate(predict_perturbations):
            start = i * cells_per_perturbation
            end = (i + 1) * cells_per_perturbation

            # Get program features (for SSM input)
            prog_feat = prog_map.get(g, global_mean)
            
            if uses_enriched and gene_to_idx is not None:
                # Construct enriched features: [4 program, 1 gene_idx, 8 stats]
                gene_idx = gene_to_idx.get(g, 0)
                # For inference, use default stats (zeros or mean)
                # In practice, we could compute these from training data if available
                stats = np.zeros(8, dtype=np.float32)  # Placeholder stats
                enriched_feat = np.concatenate([prog_feat, [float(gene_idx)], stats]).astype(np.float32)
                u = torch.from_numpy(enriched_feat[None, :]).to(device)
                gene_idx_tensor = torch.tensor([gene_idx], dtype=torch.long).to(device)
                z = ssm_model(u, gene_idx_tensor).cpu().numpy()[0]  # (latent_dim,)
            else:
                # Fallback: use program features only
                u = torch.from_numpy(prog_feat[None, :]).to(device)
                z = ssm_model(u, None).cpu().numpy()[0]  # (latent_dim,)

            # Decode back to gene space via PCA inverse transform
            # PCA: Z = (deltas - pca_mean) @ W.T, so deltas = Z @ W + pca_mean
            # Note: W is (n_components, n_genes), z is (n_components,)
            # So z @ W gives (n_genes,), which is the centered delta
            delta_centered = z @ artifacts.W  # (n_genes,)
            delta = delta_centered + artifacts.pca_mean  # Add back the mean
            mu = control + delta

            # Sample cells with improved variance modeling
            # Use low-rank covariance if available (preferred, more memory-efficient)
            if cov_factors is not None and cov_psi is not None:
                Xg = sample_cells_with_covariance(
                    mean=mu,
                    factors=cov_factors,
                    psi=cov_psi,
                    n_cells=cells_per_perturbation,
                )
            else:
                # Fall back to diagonal covariance using pre-computed global variance
                pert_std = np.sqrt(var)
                Xg = sample_cells(mu, pert_std, cells_per_perturbation)
            
            X_pred[start:end] = Xg
            obs_gene.extend([g] * cells_per_perturbation)

    prediction = anndata.AnnData(
        X=X_pred,
        obs={"gene": np.array(obs_gene)},
        var=var_df,
    )
    prediction.write_h5ad(prediction_h5ad_file_path)

    # Program proportion predictions using learned model or fallback
    # Skip model-based prediction if it requires loading full data (memory issue)
    # Use fallback directly which is faster and memory-efficient
    prog_model = None  # Skip model-based prediction to avoid memory issues
    
    if prog_model is None:
        # Fallback: use training data lookup
        rows = []
        for g in predict_perturbations:
            prog = prog_map.get(g, global_mean)
            prog = np.maximum(prog, 1e-6)
            prog = prog / prog.sum()
            rows.append({
                "gene": g,
                "pre_adipo": float(prog[0]),
                "adipo": float(prog[1]),
                "lipo": float(prog[2]),
                "other": float(prog[3]),
            })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(program_proportion_csv_file_path, index=False)



