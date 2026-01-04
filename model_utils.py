import os
from dataclasses import dataclass
from typing import List, Tuple

import anndata
import numpy as np
import pandas as pd
import scanpy
import torch
from sklearn.decomposition import PCA, IncrementalPCA


CONTROL_LABEL = "NC"


@dataclass
class SSMEffectArtifacts:
    """
    Artifacts for the latent-space SSM effect model.

    We:
    - Compute control centroid and per-gene deltas in expression space.
    - Project deltas into a low-dimensional latent space with PCA.
    - Train a small neural state-space model that maps program proportions
      to latent effects.
    """

    control_centroid: np.ndarray  # (n_genes,)
    W: np.ndarray  # (n_latent, n_genes) PCA components
    pca_mean: np.ndarray  # (n_genes,) PCA mean used for projecting
    genes_to_predict: List[str]  # gene IDs (columns)
    feature_names: List[str]  # ["pre_adipo", "adipo", "lipo", "other"]
    latent_dim: int


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


def load_program_truth(data_directory_path: str) -> pd.DataFrame:
    """Load per-gene program proportions from the training bundle."""
    path = os.path.join(data_directory_path, "program_proportion.csv")
    return pd.read_csv(path)


class ProgramProportionModel(torch.nn.Module):
    """
    Neural network to predict program proportions from gene features.
    
    Uses gene embedding + features derived from training data.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 32,
        feature_dim: int = 8,
        hidden_dim: int = 128,  # Increased from 64
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.feature_proj = torch.nn.Linear(feature_dim, hidden_dim)
        self.main = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim + hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 4),  # 4 program proportions
        )
        
    def forward(self, gene_idx: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        gene_idx : (batch,) gene index
        features : (batch, feature_dim) additional features
        
        Returns
        -------
        prog : (batch, 4) program proportions [pre_adipo, adipo, lipo, other]
        """
        gene_emb = self.embedding(gene_idx)  # (batch, embedding_dim)
        feat_proj = self.feature_proj(features)  # (batch, hidden_dim)
        combined = torch.cat([gene_emb, feat_proj], dim=1)  # (batch, embedding_dim + hidden_dim)
        prog = self.main(combined)
        # Apply softmax to ensure valid proportions
        prog = torch.softmax(prog, dim=1)
        return prog


def train_program_proportion_model(
    adata_train: anndata.AnnData,
    prog_truth: pd.DataFrame,
    genes_to_predict: List[str],
    device: str | None = None,
    n_epochs: int = 100,
    lr: float = 1e-3,
) -> Tuple[ProgramProportionModel, dict]:
    """
    Train a model to predict program proportions from gene identity and features.
    
    Returns
    -------
    model : trained ProgramProportionModel
    gene_to_idx : mapping from gene name to index
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Build gene vocabulary
    all_genes = set(genes_to_predict) | set(prog_truth["gene"].unique())
    gene_list = sorted(all_genes)
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_list)}
    vocab_size = len(gene_list)
    
    # Prepare training data
    X_gene_idx = []
    X_features = []
    y_prog = []
    
    obs = adata_train.obs
    X = to_array(adata_train[:, genes_to_predict])
    
    for _, row in prog_truth.iterrows():
        gene = row["gene"]
        if gene not in gene_to_idx:
            continue
        
        gene_idx = gene_to_idx[gene]
        
        # Features: mean expression, std, and whether seen in training
        mask = obs["gene"] == gene
        if mask.sum() > 0:
            Xg = X[mask]
            mean_expr = np.mean(Xg, axis=0)
            std_expr = np.std(Xg, axis=0)
            n_cells = mask.sum()
            features = np.concatenate([
                [np.mean(mean_expr), np.std(mean_expr)],
                [np.mean(std_expr), np.std(std_expr)],
                [n_cells / 1000.0],  # normalized cell count
                [1.0],  # seen in training
                [0.0, 0.0],  # padding
            ]).astype(np.float32)
        else:
            features = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        prog = np.array([
            row["pre_adipo"],
            row["adipo"],
            row["lipo"],
            row["other"],
        ], dtype=np.float32)
        
        X_gene_idx.append(gene_idx)
        X_features.append(features)
        y_prog.append(prog)
    
    if len(X_gene_idx) == 0:
        # Fallback: return untrained model
        model = ProgramProportionModel(vocab_size=vocab_size).to(device)
        return model, gene_to_idx
    
    X_gene_idx = torch.tensor(X_gene_idx, dtype=torch.long).to(device)
    X_features = torch.tensor(np.array(X_features), dtype=torch.float32).to(device)
    y_prog = torch.tensor(np.array(y_prog), dtype=torch.float32).to(device)
    
    # Train model
    model = ProgramProportionModel(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = model(X_gene_idx, X_features)
        loss = loss_fn(pred, y_prog)
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model, gene_to_idx


def predict_program_proportions(
    model: ProgramProportionModel,
    gene_to_idx: dict,
    predict_perturbations: List[str],
    adata_train: anndata.AnnData,
    genes_to_predict: List[str],
    device: str | None = None,
) -> np.ndarray:
    """
    Predict program proportions for a list of perturbation genes.
    
    Returns
    -------
    prog_pred : (n_perturbations, 4) predicted program proportions
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    obs = adata_train.obs
    X = to_array(adata_train[:, genes_to_predict])
    
    gene_indices = []
    features_list = []
    
    for gene in predict_perturbations:
        if gene in gene_to_idx:
            gene_idx = gene_to_idx[gene]
        else:
            # Use 0 as default (unknown gene)
            gene_idx = 0
        
        # Extract features
        mask = obs["gene"] == gene
        if mask.sum() > 0:
            Xg = X[mask]
            mean_expr = np.mean(Xg, axis=0)
            std_expr = np.std(Xg, axis=0)
            n_cells = mask.sum()
            features = np.concatenate([
                [np.mean(mean_expr), np.std(mean_expr)],
                [np.mean(std_expr), np.std(std_expr)],
                [n_cells / 1000.0],
                [1.0 if mask.sum() > 0 else 0.0],
                [0.0, 0.0],
            ]).astype(np.float32)
        else:
            features = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        gene_indices.append(gene_idx)
        features_list.append(features)
    
    gene_indices = torch.tensor(gene_indices, dtype=torch.long).to(device)
    features = torch.tensor(np.array(features_list), dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        prog_pred = model(gene_indices, features).cpu().numpy()
    
    # Ensure valid proportions (should already be from softmax, but clip for safety)
    prog_pred = np.maximum(prog_pred, 1e-6)
    prog_pred = prog_pred / prog_pred.sum(axis=1, keepdims=True)
    
    return prog_pred


def compute_control_and_effects(
    adata_train: anndata.AnnData,
    genes_to_predict: List[str],
    control_label: str = CONTROL_LABEL,
    min_cells_per_gene: int = 20,
    use_all_cells: bool = True,
    enrich_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], dict]:
    """
    Compute control centroid, per-gene deltas and enriched program features.
    
    If use_all_cells=True, uses ALL individual cells (not just centroids).
    This gives much more training data.
    
    If enrich_features=True, adds gene identity embeddings and expression statistics.

    Returns
    -------
    control_centroid : (n_genes,)
    deltas : (n_cells, n_genes) if use_all_cells, else (n_observed_genes, n_genes)
    prog_features : (n_cells, n_features) enriched features if enrich_features, else (n_cells, 4)
    kept_genes : list of gene names used to fit the model
    gene_to_idx : dict mapping gene name to index for embedding
    """
    X = to_array(adata_train[:, genes_to_predict])
    obs = adata_train.obs

    control_mask = obs["gene"] == control_label
    assert control_mask.sum() > 0, "No control cells found."

    control_centroid = X[control_mask].mean(axis=0)
    control_std = X[control_mask].std(axis=0)

    labels = obs["gene"].astype(str).values
    unique_genes = sorted(g for g in set(labels) if g != control_label)
    
    # Build gene vocabulary for embeddings
    gene_to_idx = {gene: idx for idx, gene in enumerate(unique_genes)}
    vocab_size = len(unique_genes)
    
    # Count perturbation frequency
    gene_counts = {g: (labels == g).sum() for g in unique_genes}
    max_count = max(gene_counts.values()) if gene_counts else 1.0

    if use_all_cells:
        # Use ALL individual cells, not just centroids
        deltas: List[np.ndarray] = []
        prog_features: List[np.ndarray] = []
        kept_genes: List[str] = []
        
        for g in unique_genes:
            mask = labels == g
            if mask.sum() < min_cells_per_gene:
                continue

            Xg = X[mask]  # All cells for this perturbation
            sub_obs = obs.loc[mask]
            required = ["pre_adipo", "adipo", "lipo", "other"]
            if not all(c in sub_obs.columns for c in required):
                continue

            # Compute statistics for this perturbation
            Xg_mean = Xg.mean(axis=0)
            Xg_std = Xg.std(axis=0)
            n_cells_g = Xg.shape[0]

            # Compute delta for EACH cell (not just centroid)
            for i in range(Xg.shape[0]):
                delta_cell = Xg[i] - control_centroid
                
                # Base program features
                P_cell = np.array(
                    [
                        sub_obs.iloc[i]["pre_adipo"],
                        sub_obs.iloc[i]["adipo"],
                        sub_obs.iloc[i]["lipo"],
                        sub_obs.iloc[i]["other"],
                    ],
                    dtype=np.float32,
                )
                
                if enrich_features:
                    # Enrich with gene identity and expression statistics
                    # Gene embedding index (will be converted to embedding in model)
                    gene_idx = gene_to_idx[g]
                    
                    # Expression statistics (normalized)
                    expr_mean_mean = np.mean(Xg_mean)
                    expr_mean_std = np.std(Xg_mean)
                    expr_std_mean = np.mean(Xg_std)
                    expr_std_std = np.std(Xg_std)
                    
                    # Control comparison
                    control_diff_mean = np.mean(Xg_mean - control_centroid)
                    control_diff_std = np.std(Xg_mean - control_centroid)
                    
                    # Normalized cell count and frequency
                    norm_cell_count = n_cells_g / 1000.0
                    norm_frequency = gene_counts[g] / max_count
                    
                    # Concatenate enriched features
                    enriched = np.concatenate([
                        P_cell,  # 4 program features
                        [float(gene_idx)],  # 1 gene index (will be embedded)
                        [expr_mean_mean, expr_mean_std, expr_std_mean, expr_std_std],  # 4 expr stats
                        [control_diff_mean, control_diff_std],  # 2 control comparison
                        [norm_cell_count, norm_frequency],  # 2 frequency stats
                    ]).astype(np.float32)
                    prog_features.append(enriched)
                else:
                    prog_features.append(P_cell)
                
                deltas.append(delta_cell)
                kept_genes.append(g)
        
        deltas_array = np.vstack(deltas)
        prog_features_array = np.vstack(prog_features)
    else:
        # Original: use centroids only (one per perturbation)
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
            Xg_std = Xg.std(axis=0)
            n_cells_g = Xg.shape[0]

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
            
            if enrich_features:
                # Enrich with gene identity and expression statistics
                gene_idx = gene_to_idx[g]
                expr_mean_mean = np.mean(mu_g)
                expr_mean_std = np.std(mu_g)
                expr_std_mean = np.mean(Xg_std)
                expr_std_std = np.std(Xg_std)
                control_diff_mean = np.mean(mu_g - control_centroid)
                control_diff_std = np.std(mu_g - control_centroid)
                norm_cell_count = n_cells_g / 1000.0
                norm_frequency = gene_counts[g] / max_count
                
                enriched = np.concatenate([
                    P_g,
                    [float(gene_idx)],
                    [expr_mean_mean, expr_mean_std, expr_std_mean, expr_std_std],
                    [control_diff_mean, control_diff_std],
                    [norm_cell_count, norm_frequency],
                ]).astype(np.float32)
                prog_features.append(enriched)
            else:
                prog_features.append(P_g)

            deltas.append(delta_g)
            kept_genes.append(g)

        deltas_array = np.vstack(deltas)
        prog_features_array = np.vstack(prog_features)

    return control_centroid, deltas_array, prog_features_array, kept_genes, gene_to_idx


def estimate_global_variance(
    adata_train: anndata.AnnData,
    genes_to_predict: List[str],
    control_label: str = CONTROL_LABEL,
    max_samples: int = 20000,  # Reduced from 50000 for memory safety
    chunk_size: int = 5000,
) -> np.ndarray:
    """Estimate global diagonal variance from perturbed cells using chunked processing."""
    mask = adata_train.obs["gene"] != control_label
    n_perturbed = mask.sum()
    
    # For very large datasets, subsample to avoid memory issues
    if n_perturbed > max_samples:
        print(f"Subsampling {n_perturbed} cells to {max_samples} for variance estimation")
        indices = np.where(mask)[0]
        np.random.seed(42)
        selected_indices = np.random.choice(indices, size=max_samples, replace=False)
        mask_subset = np.zeros(len(mask), dtype=bool)
        mask_subset[selected_indices] = True
        mask = mask_subset
        indices_to_use = np.where(mask)[0]
    else:
        indices_to_use = np.where(mask)[0]
    
    # Process in chunks to avoid memory issues
    n_genes = len(genes_to_predict)
    sum_x = np.zeros(n_genes, dtype=np.float64)
    sum_x2 = np.zeros(n_genes, dtype=np.float64)
    n_samples = 0
    
    print(f"Computing variance in chunks of {chunk_size}...")
    for i in range(0, len(indices_to_use), chunk_size):
        end_idx = min(i + chunk_size, len(indices_to_use))
        chunk_indices = indices_to_use[i:end_idx]
        
        # Load chunk
        X_chunk = to_array(adata_train[chunk_indices, genes_to_predict])
        
        # Accumulate statistics
        sum_x += np.sum(X_chunk, axis=0, dtype=np.float64)
        sum_x2 += np.sum(X_chunk ** 2, axis=0, dtype=np.float64)
        n_samples += X_chunk.shape[0]
        
        # Free memory
        del X_chunk
    
    # Compute variance: var = E[X^2] - E[X]^2
    mean_x = sum_x / n_samples
    var = (sum_x2 / n_samples) - (mean_x ** 2)
    var = np.maximum(var, 1e-4)
    return var.astype(np.float32)


def estimate_perturbation_variance(
    adata_train: anndata.AnnData,
    genes_to_predict: List[str],
    perturbation_gene: str,
    control_label: str = CONTROL_LABEL,
    min_cells: int = 10,
) -> np.ndarray:
    """Estimate variance specific to a perturbation gene."""
    X = to_array(adata_train[:, genes_to_predict])
    obs = adata_train.obs
    
    mask = obs["gene"] == perturbation_gene
    if mask.sum() < min_cells:
        # Fall back to global variance if not enough cells
        return estimate_global_variance(adata_train, genes_to_predict, control_label)
    
    Xp = X[mask]
    var = np.var(Xp, axis=0)
    var = np.maximum(var, 1e-4)
    return var.astype(np.float32)


def estimate_low_rank_covariance(
    adata_train: anndata.AnnData,
    genes_to_predict: List[str],
    rank: int = 50,
    control_label: str = CONTROL_LABEL,
    max_samples: int = 20000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate low-rank covariance matrix using factor analysis.
    
    Returns
    -------
    factors : (n_genes, rank) factor loading matrix
    psi : (n_genes,) diagonal noise variance
    """
    from sklearn.decomposition import FactorAnalysis
    
    mask = adata_train.obs["gene"] != control_label
    n_perturbed = mask.sum()
    
    # For very large datasets, subsample to avoid memory issues
    if n_perturbed > max_samples:
        print(f"Subsampling {n_perturbed} cells to {max_samples} for covariance estimation")
        indices = np.where(mask)[0]
        np.random.seed(42)
        selected_indices = np.random.choice(indices, size=max_samples, replace=False)
        mask_subset = np.zeros(len(mask), dtype=bool)
        mask_subset[selected_indices] = True
        mask = mask_subset
    
    X = to_array(adata_train[:, genes_to_predict])
    Xp = X[mask]
    
    # Center the data
    Xp_mean = np.mean(Xp, axis=0)
    Xp_centered = Xp - Xp_mean
    
    # Fit factor analysis
    fa = FactorAnalysis(n_components=min(rank, Xp_centered.shape[1] - 1), random_state=0)
    fa.fit(Xp_centered)
    
    # Extract factors and noise variance
    factors = fa.components_.T.astype(np.float32)  # (n_genes, rank)
    psi = np.maximum(fa.noise_variance_, 1e-4).astype(np.float32)  # (n_genes,)
    
    return factors, psi


def estimate_program_specific_covariance(
    adata_train: anndata.AnnData,
    genes_to_predict: List[str],
    rank: int = 50,
    control_label: str = CONTROL_LABEL,
    max_samples_per_program: int = 5000,
) -> Tuple[dict, np.ndarray]:
    """
    Estimate program-specific low-rank covariance matrices.
    
    Returns
    -------
    program_covariances : dict mapping program name to (factors, psi) tuple
    global_psi : (n_genes,) global diagonal noise variance (fallback)
    """
    from sklearn.decomposition import FactorAnalysis
    
    X = to_array(adata_train[:, genes_to_predict])
    obs = adata_train.obs
    
    # Group cells by dominant program
    programs = ["pre_adipo", "adipo", "lipo", "other"]
    program_covariances = {}
    
    # Get global noise variance as fallback
    mask = obs["gene"] != control_label
    Xp = X[mask]
    Xp_mean = np.mean(Xp, axis=0)
    Xp_centered = Xp - Xp_mean
    global_psi = np.maximum(np.var(Xp_centered, axis=0), 1e-4).astype(np.float32)
    
    for prog_name in programs:
        if prog_name not in obs.columns:
            continue
        
        # Find cells where this program is dominant
        prog_values = obs[prog_name].values
        # Use cells where this program proportion > 0.3
        prog_mask = (prog_values > 0.3) & (obs["gene"] != control_label)
        
        if prog_mask.sum() < 100:  # Need minimum cells
            continue
        
        X_prog = X[prog_mask]
        n_samples = X_prog.shape[0]
        
        # Subsample if needed
        if n_samples > max_samples_per_program:
            indices = np.where(prog_mask)[0]
            np.random.seed(42)
            selected_indices = np.random.choice(indices, size=max_samples_per_program, replace=False)
            X_prog = X[selected_indices]
        
        # Center the data
        X_prog_mean = np.mean(X_prog, axis=0)
        X_prog_centered = X_prog - X_prog_mean
        
        # Fit factor analysis for this program
        try:
            fa = FactorAnalysis(n_components=min(rank, X_prog_centered.shape[1] - 1), random_state=0)
            fa.fit(X_prog_centered)
            
            factors = fa.components_.T.astype(np.float32)
            psi = np.maximum(fa.noise_variance_, 1e-4).astype(np.float32)
            
            program_covariances[prog_name] = (factors, psi)
        except Exception as e:
            print(f"Warning: Could not fit covariance for {prog_name}: {e}")
            # Use global as fallback
            program_covariances[prog_name] = (None, global_psi)
    
    return program_covariances, global_psi


def sample_cells_with_program_covariance(
    mean: np.ndarray,
    program_proportions: np.ndarray,
    program_covariances: dict,
    global_psi: np.ndarray,
    n_cells: int,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Sample synthetic cells using weighted combination of program-specific covariances.
    
    Parameters
    ----------
    mean : (n_genes,) mean expression
    program_proportions : (4,) program proportions [pre_adipo, adipo, lipo, other]
    program_covariances : dict mapping program name to (factors, psi) tuple
    global_psi : (n_genes,) global diagonal noise variance (fallback)
    n_cells : number of cells to sample
    random_state : random seed
    
    Returns
    -------
    X : (n_cells, n_genes) sampled cells
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_genes = mean.shape[0]
    programs = ["pre_adipo", "adipo", "lipo", "other"]
    
    # Normalize program proportions
    prog_weights = np.array([program_proportions[i] for i in range(4)], dtype=np.float32)
    prog_weights = np.maximum(prog_weights, 0.0)
    prog_weights = prog_weights / (prog_weights.sum() + 1e-8)
    
    # Sample cells
    X_samples = []
    for _ in range(n_cells):
        # Weighted combination of program-specific noise
        eps_total = np.zeros(n_genes, dtype=np.float32)
        
        for i, prog_name in enumerate(programs):
            weight = prog_weights[i]
            if weight < 1e-6:
                continue
            
            if prog_name in program_covariances:
                factors, psi = program_covariances[prog_name]
                
                if factors is not None:
                    # Sample from this program's covariance
                    rank = factors.shape[1]
                    z = np.random.normal(0, 1, size=rank)
                    eps_factor = z @ factors.T
                    std = np.sqrt(psi)
                    eps_diag = np.random.normal(0, std)
                    eps_prog = eps_factor + eps_diag
                else:
                    # Diagonal only
                    std = np.sqrt(psi)
                    eps_prog = np.random.normal(0, std)
            else:
                # Fallback to global
                std = np.sqrt(global_psi)
                eps_prog = np.random.normal(0, std)
            
            eps_total += weight * eps_prog
        
        X_samples.append(mean + eps_total)
    
    X = np.vstack(X_samples)
    return np.maximum(X, 0.0).astype(np.float32)


def sample_cells_with_covariance(
    mean: np.ndarray,
    factors: np.ndarray | None,
    psi: np.ndarray,
    n_cells: int,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Sample synthetic cells using low-rank covariance model.
    
    If factors is None, uses diagonal covariance (psi only).
    Otherwise uses: Cov = factors @ factors.T + diag(psi)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_genes = mean.shape[0]
    
    if factors is None:
        # Diagonal covariance
        std = np.sqrt(psi)
        eps = np.random.normal(loc=0.0, scale=std, size=(n_cells, n_genes))
    else:
        # Low-rank + diagonal covariance
        rank = factors.shape[1]
        # Sample from factor space
        z = np.random.normal(0, 1, size=(n_cells, rank))
        # Project to gene space
        eps_factor = z @ factors.T  # (n_cells, n_genes)
        # Add diagonal noise
        std = np.sqrt(psi)
        eps_diag = np.random.normal(loc=0.0, scale=std, size=(n_cells, n_genes))
        eps = eps_factor + eps_diag
    
    X = mean[None, :] + eps
    return np.maximum(X, 0.0).astype(np.float32)


class PerturbationSSM(torch.nn.Module):
    """
    Enhanced neural state-space model in latent space with residual connections,
    layer normalization, attention, and improved activations.

    h_{t+1} = LayerNorm(A h_t + B u + residual)
    z_T = head(h_T)

    - u: enriched feature vector (program + gene embedding + stats) kept constant over T steps.
    - h_t: internal state.
    - z_T: latent effect which is decoded back to gene space via PCA.
    """

    def __init__(
        self,
        input_dim: int = 4,
        state_dim: int = 128,
        latent_dim: int = 64,
        n_steps: int = 24,
        use_attention: bool = True,
        vocab_size: int = 10000,
        gene_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.state_dim = state_dim
        self.use_attention = use_attention
        
        # Gene embedding (if input includes gene index)
        self.gene_embedding = torch.nn.Embedding(vocab_size, gene_embed_dim)
        
        # Input embedding to richer representation
        # Raw input is: [4 program features, 1 gene_idx, 8 stats] = 13 features
        # After gene embedding: [4 program, 16 gene_emb, 8 stats] = 28 features
        # Note: input_dim may be 13 (raw) or 4 (program only), but enriched is always 28
        enriched_input_dim = 4 + gene_embed_dim + 8  # program (4) + gene_emb + stats (8) = 28
        self.input_embed = torch.nn.Sequential(
            torch.nn.Linear(enriched_input_dim, state_dim),
            torch.nn.GELU(),
            torch.nn.Linear(state_dim, state_dim),
        )
        
        # Attention mechanism over program features
        if use_attention:
            self.attention = torch.nn.MultiheadAttention(
                embed_dim=state_dim, num_heads=4, batch_first=True
            )
        
        # State transition with residual
        self.state_to_state = torch.nn.Linear(state_dim, state_dim, bias=True)
        self.input_to_state = torch.nn.Linear(state_dim, state_dim, bias=False)
        self.layer_norm = torch.nn.LayerNorm(state_dim)
        self.activation = torch.nn.GELU()
        
        # Output projection
        self.state_to_latent = torch.nn.Sequential(
            torch.nn.Linear(state_dim, state_dim),
            torch.nn.GELU(),
            torch.nn.Linear(state_dim, latent_dim),
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, u: torch.Tensor, gene_indices: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        u : (batch, input_dim) enriched features [program_features, gene_idx, stats]
        gene_indices : (batch,) gene indices for embedding, or None if not using enriched features

        Returns
        -------
        z : (batch, latent_dim)
        """
        batch_size = u.shape[0]
        device = u.device
        
        # If gene_indices provided, extract and embed gene identity
        if gene_indices is not None:
            gene_emb = self.gene_embedding(gene_indices)  # (batch, gene_embed_dim)
            # u is [program_features, gene_idx, stats]
            # Extract program features (first 4) and stats (last 8), replace gene_idx with embedding
            program_feat = u[:, :4]  # (batch, 4)
            stats_feat = u[:, 5:]  # (batch, 8) - skip gene_idx at position 4
            u_enriched = torch.cat([program_feat, gene_emb, stats_feat], dim=1)  # (batch, 4+16+8)
        else:
            # Fallback: use u as-is (for backward compatibility)
            u_enriched = u
        
        # Embed input to richer representation
        u_emb = self.input_embed(u_enriched)  # (batch, state_dim)
        
        # Apply attention if enabled
        if self.use_attention:
            # Reshape for attention: (batch, 1, state_dim)
            u_emb = u_emb.unsqueeze(1)
            u_emb, _ = self.attention(u_emb, u_emb, u_emb)
            u_emb = u_emb.squeeze(1)  # (batch, state_dim)
        
        # Initialize state
        h = torch.zeros(batch_size, self.state_dim, device=device)
        
        # State-space evolution with residual connections
        for step in range(self.n_steps):
            # State update with residual
            h_new = self.state_to_state(h) + self.input_to_state(u_emb)
            h_new = self.layer_norm(h_new)
            h_new = self.activation(h_new)
            
            # Residual connection (skip connection every other step for stability)
            if step > 0 and step % 2 == 0:
                h = h + h_new  # Residual
            else:
                h = h_new
        
        # Final projection to latent space
        z = self.state_to_latent(h)
        return z


def fit_ssm_effect_model(
    deltas: np.ndarray,
    prog_features: np.ndarray,
    n_latent: int = 64,
    n_steps: int = 24,
    state_dim: int = 128,
    n_epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 128,
    device: str | None = None,
    validation_split: float = 0.2,
    early_stopping_patience: int = 20,
    weight_decay: float = 1e-4,
    hybrid_loss_weight: float = 0.7,
) -> Tuple[SSMEffectArtifacts, dict]:
    """
    Fit PCA + neural SSM mapping program features to latent effects.
    
    Enhanced with early stopping, learning rate scheduling, and regularization.

    Returns
    -------
    artifacts : SSMEffectArtifacts
    ssm_state_dict : raw PyTorch state dict (for saving with torch.save)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # PCA in gene-expression space to define latent coordinates
    # Use IncrementalPCA for large datasets to avoid memory issues
    n_components = min(n_latent, deltas.shape[1])
    n_samples, n_features = deltas.shape
    
    deltas_float = deltas.astype(np.float32)
    
    # For very large datasets, use IncrementalPCA with batch processing
    if n_samples > 20000:
        print(f"Large dataset ({n_samples} samples), using IncrementalPCA for memory efficiency")
        try:
            # Use IncrementalPCA with batches
            batch_size = min(5000, n_samples)
            pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
            
            # Fit in batches
            for i in range(0, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                pca.partial_fit(deltas_float[i:end])
            
            # Transform in batches to avoid memory issues
            Z_list = []
            for i in range(0, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                Z_batch = pca.transform(deltas_float[i:end])
                Z_list.append(Z_batch)
            Z = np.vstack(Z_list)
            
            W = pca.components_.astype(np.float32)
            pca_mean = pca.mean_.astype(np.float32)
        except MemoryError:
            print("Memory error with IncrementalPCA, subsampling for PCA fitting...")
            # Fallback: subsample for PCA fitting, then transform all
            np.random.seed(42)
            n_sub = min(20000, n_samples)
            indices = np.random.choice(n_samples, size=n_sub, replace=False)
            pca = PCA(n_components=n_components, random_state=0)
            pca.fit(deltas_float[indices])
            Z = pca.transform(deltas_float)
            W = pca.components_.astype(np.float32)
            pca_mean = pca.mean_.astype(np.float32)
    else:
        # Standard PCA for smaller datasets
        pca = PCA(n_components=n_components, random_state=0)
        Z = pca.fit_transform(deltas_float)
        W = pca.components_.astype(np.float32)
        pca_mean = pca.mean_.astype(np.float32)

    # Split into train/validation
    dataset_size = prog_features.shape[0]
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split_idx = int(dataset_size * (1 - validation_split))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    # Check if features are enriched (have gene_idx at position 4)
    is_enriched = prog_features.shape[1] > 4
    if is_enriched:
        # Extract gene indices from enriched features (position 4)
        gene_indices_train = torch.from_numpy(prog_features[train_idx, 4].astype(np.int64)).to(device)
        gene_indices_val = torch.from_numpy(prog_features[val_idx, 4].astype(np.int64)).to(device)
        vocab_size = int(prog_features[:, 4].max()) + 1
    else:
        gene_indices_train = None
        gene_indices_val = None
        vocab_size = 10000  # Default
    
    X_u_train = torch.from_numpy(prog_features[train_idx].astype(np.float32))
    Y_z_train = torch.from_numpy(Z[train_idx].astype(np.float32))
    deltas_train = torch.from_numpy(deltas[train_idx].astype(np.float32))
    X_u_val = torch.from_numpy(prog_features[val_idx].astype(np.float32))
    Y_z_val = torch.from_numpy(Z[val_idx].astype(np.float32))
    deltas_val = torch.from_numpy(deltas[val_idx].astype(np.float32))

    model = PerturbationSSM(
        input_dim=prog_features.shape[1],
        state_dim=state_dim,
        latent_dim=n_components,
        n_steps=n_steps,
        vocab_size=vocab_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    loss_fn = torch.nn.MSELoss()
    
    # Convert PCA components to tensors for hybrid loss
    W_tensor = torch.from_numpy(W).to(device)
    pca_mean_tensor = torch.from_numpy(pca_mean).to(device)

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_state_dict = None

    model.train()
    train_idx_shuffled = np.arange(len(train_idx))
    
    print(f"Training SSM: {len(train_idx)} train samples, {len(val_idx)} val samples, {n_epochs} max epochs")
    print(f"Using hybrid loss (weight={hybrid_loss_weight:.2f}) and enriched features={is_enriched}")
    
    # Learning rate warmup
    warmup_epochs = 10
    initial_lr = lr / 10.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr
    
    for epoch in range(n_epochs):
        # Learning rate warmup
        if epoch < warmup_epochs:
            warmup_lr = initial_lr + (lr - initial_lr) * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # Training
        np.random.shuffle(train_idx_shuffled)
        epoch_loss = 0.0
        n_batches = 0
        
        for start in range(0, len(train_idx), batch_size):
            end = min(start + batch_size, len(train_idx))
            batch_idx = train_idx_shuffled[start:end]
            u_batch = X_u_train[batch_idx].to(device)
            z_target = Y_z_train[batch_idx].to(device)
            delta_target = deltas_train[batch_idx].to(device)
            gene_idx_batch = gene_indices_train[batch_idx] if gene_indices_train is not None else None

            optimizer.zero_grad()
            z_pred = model(u_batch, gene_idx_batch)
            
            # Hybrid loss: latent space + gene space reconstruction
            latent_loss = loss_fn(z_pred, z_target)
            
            if hybrid_loss_weight < 1.0:
                # Decode to gene space for reconstruction loss
                delta_pred = z_pred @ W_tensor + pca_mean_tensor
                gene_loss = loss_fn(delta_pred, delta_target)
                loss = hybrid_loss_weight * latent_loss + (1.0 - hybrid_loss_weight) * gene_loss
            else:
                loss = latent_loss
            
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / n_batches if n_batches > 0 else epoch_loss
        
        # Validation
        model.eval()
        with torch.no_grad():
            z_pred_val = model(X_u_val.to(device), gene_indices_val)
            val_latent_loss = loss_fn(z_pred_val, Y_z_val.to(device)).item()
            
            if hybrid_loss_weight < 1.0:
                delta_pred_val = z_pred_val @ W_tensor + pca_mean_tensor
                val_gene_loss = loss_fn(delta_pred_val, deltas_val.to(device)).item()
                val_loss = hybrid_loss_weight * val_latent_loss + (1.0 - hybrid_loss_weight) * val_gene_loss
            else:
                val_loss = val_latent_loss
        model.train()
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}/{n_epochs} (patience={patience_counter})")
                break
    
    if best_state_dict is None:
        print(f"Warning: No improvement during training, using final model")
        best_state_dict = model.state_dict().copy()
    
    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    artifacts = SSMEffectArtifacts(
        control_centroid=None,  # filled by caller
        W=W,
        pca_mean=pca_mean,
        genes_to_predict=[],  # filled by caller
        feature_names=["pre_adipo", "adipo", "lipo", "other"],
        latent_dim=n_components,
    )
    return artifacts, model.state_dict()


def sample_cells(
    mean: np.ndarray,
    std: np.ndarray,
    n_cells: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample synthetic cells from a diagonal Gaussian around mean."""
    if random_state is not None:
        np.random.seed(random_state)
    eps = np.random.normal(loc=0.0, scale=std, size=(n_cells, mean.shape[0]))
    X = mean[None, :] + eps
    return np.maximum(X, 0.0).astype(np.float32)



