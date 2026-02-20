from tqdm import tqdm
import torch
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.functional import pearson_corrcoef
import pandas as pd
if not hasattr(pd.Series, "nonzero"):
    pd.Series.nonzero = lambda self: self.to_numpy().nonzero()


def person_corr(x_pert, x_ctrl, x_recon_pert,top_n= 20):
    real_delta_vector = (x_pert - x_ctrl)
    pred_delta_vector = (x_recon_pert - x_ctrl)
    top_vals, top_indices = torch.topk(torch.abs(real_delta_vector), k=top_n)
    # Create mask for genes that moved significantly

    real_top = real_delta_vector[top_indices]
    pred_top = pred_delta_vector[top_indices]
    
    pearson_all = pearson_corrcoef(pred_delta_vector, real_delta_vector)            
    pearson_top5 = pearson_corrcoef(pred_top, real_top)
    return pearson_top5.item(), pearson_all.item(), 

def direction_error(x_pert,x_ctrl,x_recon_pert, top_n=20):
    real_delta = (x_pert - x_ctrl)
    pred_delta = (x_recon_pert - x_ctrl)
    top_vals, top_indices = torch.topk(torch.abs(real_delta), k=top_n)
    all_signs_real = torch.sign(real_delta)
    all_signs_pred = torch.sign(pred_delta)
    all_sign_product = all_signs_real * all_signs_pred
    n_opposite_all = (all_sign_product < 0).float().sum()
    percent_opposite_all = n_opposite_all / float(len(real_delta))

    real_signs = torch.sign(real_delta[top_indices])
    pred_signs = torch.sign(pred_delta[top_indices])
    sign_product = real_signs * pred_signs
    n_opposite = (sign_product < 0).float().sum()
    percent_opposite = n_opposite / float(top_n)
    return percent_opposite.item(), percent_opposite_all.item()

def nmse_metric(x_pert,x_ctrl,x_recon_pert,top_n=20):
    real_delta = (x_pert - x_ctrl)
    pred_delta = (x_recon_pert - x_ctrl)
    mse_all = torch.mean((real_delta - pred_delta) ** 2)
    nmse_all = mse_all / torch.mean(real_delta ** 2)
    top_vals, top_indices = torch.topk(torch.abs(real_delta), k=top_n)
    r_top = real_delta[top_indices]
    p_top = pred_delta[top_indices]
    mse = torch.mean((r_top - p_top) ** 2)
    nmse = mse / torch.mean(r_top ** 2)
    
    return nmse.item(), nmse_all.item()


import torch
import numpy as np
from scipy.stats import wasserstein_distance, pearsonr

class DistributionEvaluator:
    def __init__(self, device='cuda'):
        self.device = device

    def compute_mmd(self, x_real, x_pred, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        Calculates Maximum Mean Discrepancy (MMD) using RBF Kernel.
        Low MMD = Distributions are similar.
        High MMD = Distributions are different.
        """
        n_samples = int(x_real.size(0))
        total = torch.cat([x_real, x_pred], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            with torch.no_grad():
                x = x_real
                dists = torch.cdist(x, x) ** 2
                bandwidth = torch.median(dists[dists > 0])
            
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        
        kernels = sum(kernel_val)
        
        XX = kernels[:n_samples, :n_samples]
        YY = kernels[n_samples:, n_samples:]
        XY = kernels[:n_samples, n_samples:]
        YX = kernels[n_samples:, :n_samples]
        
        loss = torch.mean(XX + YY - XY - YX)
        return loss.item()
    

    def compute_sliced_wasserstein(self,x_real,x_pred,num_projections=100):
        """
        Sliced Wasserstein Distance between joint distributions.
        Projects high-D gene expression onto random 1D directions.
        """

        # Move to CPU
        real = x_real.detach().cpu().numpy()
        pred = x_pred.detach().cpu().numpy()

        n_genes = real.shape[1]
        sw_dist = 0.0

        for _ in range(num_projections):
            # Random direction on unit sphere
            theta = np.random.normal(0, 1, size=(n_genes,))
            theta /= np.linalg.norm(theta) + 1e-8

            # Project
            real_proj = real @ theta
            pred_proj = pred @ theta

            # 1D Wasserstein
            sw_dist += wasserstein_distance(real_proj, pred_proj)

        return sw_dist / num_projections

    def compute_variance_preservation(self, x_real, x_pred):
        """
        Checks if the model captures the biological noise correctly.
        Calculates Pearson Correlation between Real SD and Predicted SD per gene.
        """
        # Calculate Standard Deviation per gene (axis 0 = cells)
        real_std = torch.std(x_real, dim=0).detach().cpu().numpy()
        pred_std = torch.std(x_pred, dim=0).detach().cpu().numpy()
        
        # Avoid NaNs if std is 0
        valid_idx = (real_std > 1e-6) & (pred_std > 1e-6)
        
        if valid_idx.sum() > 2:
            corr, _ = pearsonr(real_std[valid_idx], pred_std[valid_idx])
        else:
            corr = 0.0
            
        return corr
    
    def compute_l1_distance(self, x_real, x_pred):
        """
        Calculates Mean Absolute Error (L1) between the means of the populations.
        Good for checking if the 'center of mass' is correct.
        """
        # We compare the average expression profile of the batch
        real_mean = x_real.mean(dim=0)
        pred_mean = x_pred.mean(dim=0)
        
        l1_dist = torch.abs(real_mean - pred_mean).mean()
        return l1_dist.item()

    def compute_energy_loss(self, x_real, x_pred):
        """
        Calculates Energy Distance (Energy Statistic).
        D_E(X, Y) = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
        This is a robust distance metric similar to MMD but using Euclidean norms.
        """
        n = x_real.size(0)
        m = x_pred.size(0)
        
        # Concatenate for efficient pairwise calculation
        total = torch.cat([x_real, x_pred], dim=0) # [2N, D]
        
        # Compute pairwise Euclidean distances using CDIST (More memory efficient than expand)
        # dists[i, j] = ||total[i] - total[j]||
        dists = torch.cdist(total, total, p=2) 
        
        # Extract sub-matrices
        # XX: Distances within Real
        dist_xx = dists[:n, :n].sum() / (n * n)
        
        # YY: Distances within Pred
        dist_yy = dists[n:, n:].sum() / (m * m)
        
        # XY: Distances between Real and Pred
        dist_xy = dists[:n, n:].sum() / (n * m)
        
        # Energy Distance Formula
        energy_loss = 2 * dist_xy - dist_xx - dist_yy
        return energy_loss.item()

    def evaluate_batch(self, x_real, x_pred):
        """
        Runs all metrics on a batch of cells.
        Expects inputs: [N_Cells, N_Genes]
        """
        return self.compute_mmd(x_real, x_pred),self.compute_sliced_wasserstein(x_real, x_pred), self.compute_variance_preservation(x_real, x_pred), self.compute_energy_loss(x_real, x_pred), self.compute_l1_distance(x_real, x_pred)
        

#from CycleTransformer.cycleTransformer import CycleTransformer
from CycleTransformer.cycleTransformer import CycleTransformer
from omegaconf import OmegaConf
import anndata as ad
import pandas as pd
from CycleTransformer.dataloader import get_loadersBence as get_loaders

cfg = OmegaConf.load("config.yaml")

def get_gene_to_idx():
    trainloader, valloader, gene_to_idx, idx_to_gene = get_loaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    return gene_to_idx, idx_to_gene

def process_x(x,num_paddings = 8, seq_length = 32):
    x = np.concatenate([x, np.zeros(num_paddings, dtype=np.float32)])
    return x.reshape(seq_length, -1)


def get_model_output(model, pert_idx,x_ctrl):

    x_ctrl = torch.from_numpy(x_ctrl).float().to(model.device)
    pert_idx = torch.from_numpy(pert_idx).long().to(model.device)
    z_ctrl = model.encoder(x_ctrl) 
    z_prompt_fwd = model.get_perturbation_prompt(x_ctrl, pert_idx)
    delta_fwd = model.transition_fwd(z_ctrl, z_prompt_fwd)
    z_fake_pert = z_ctrl + delta_fwd
    x_recon_pert = model.decoder(z_fake_pert)
    x_recon_pert = x_recon_pert.detach().cpu().numpy()

    return x_recon_pert.reshape(x_recon_pert.shape[0],-1)[:,:21592]

import pickle
import gc # Garbage Collector

GENE_LENGTH = 21592

def eval_precision(cpkt_path):
    test_adata = ad.read_h5ad('data/test_data.h5ad')
    train_adata = ad.read_h5ad('data/train_data.h5ad')
    ctrl_adata = ad.read_h5ad('data/ctrl_data.h5ad')
    model = CycleTransformer.load_from_checkpoint(cpkt_path)
    model = model.to('cpu')
    model.eval()
    with open('data/gene_to_idx.pkl', 'rb') as f:
        gene_to_idx = pickle.load(f)
    #gene_to_idx, idx_to_gene = get_gene_to_idx()
    

    available_genes = [gene for gene in list(gene_to_idx.keys()) if gene in test_adata.obs['gene'].unique()]

    res_dict = {"all_direction_errors": [],
            "top_k_direction_errors": [],
            "all_nmse": [],
            "top_k_nmse": [],
            "all_pearson": [],
            "top_k_pearson": []}
    pbar = tqdm(available_genes)
    with torch.no_grad():
        for gene in pbar:
            pert_idx = gene_to_idx[gene]
            X_pert = test_adata[test_adata.obs['gene']==gene].X.toarray()
            x_pert = np.array([process_x(x) for x in X_pert])
            x_pert = x_pert.reshape(x_pert.shape[0], -1)[:,:GENE_LENGTH]
            x_ctrl_idxs = np.random.choice(ctrl_adata.X.toarray().shape[0], size=x_pert.shape[0], replace=False)
            x_ctrl = ctrl_adata.X.toarray()[x_ctrl_idxs]
            x_ctrl = np.array([process_x(x) for x in x_ctrl])
            pert_idx = np.array([ pert_idx for _ in range(x_ctrl.shape[0]) ])
            x_pred = get_model_output(model, pert_idx, x_ctrl)
            
            x_ctrl = x_ctrl.reshape(x_ctrl.shape[0], -1)[:,:GENE_LENGTH]
            x_pert = torch.from_numpy(x_pert).float().mean(dim=0)
            x_ctrl = torch.from_numpy(x_ctrl).float().mean(dim=0)
            x_pred = torch.from_numpy(x_pred).float().mean(dim=0)

            percent_opposite_top_k, percent_opposite_all = direction_error(x_pert,x_ctrl,x_pred)
            nmse_top_k, nmse_all = nmse_metric(x_pert,x_ctrl,x_pred)
            pearson_top_k, pearson_all = person_corr(x_pert,x_ctrl,x_pred)


            res_dict["all_direction_errors"].append(percent_opposite_all)
            res_dict["top_k_direction_errors"].append(percent_opposite_top_k)
            res_dict["all_nmse"].append(nmse_all)
            res_dict["top_k_nmse"].append(nmse_top_k)
            res_dict["all_pearson"].append(pearson_all)
            res_dict["top_k_pearson"].append(pearson_top_k)

            

            pbar.postfix = {"top_pearson":np.mean(res_dict["top_k_pearson"]), "top_nmse":np.mean(res_dict["top_k_nmse"]), "top_direction_error":np.mean(res_dict["top_k_direction_errors"])}
    with open("results/precision_all_gene_pearson.pkl", "wb") as f:  
        pickle.dump(res_dict, f)
    print("Finished")


def debug_eval(cpkt_path):
    test_adata = ad.read_h5ad('data/GW_test.h5ad')
    train_adata = ad.read_h5ad('data/GW_train.h5ad')
    print(test_adata.obs['gene'].unique())
    print(test_adata.obs)
    ctrl_adata = train_adata[train_adata.obs.gene == "NC"]
    model = CycleTransformer.load_from_checkpoint(cpkt_path)
    model = model.to('cpu')
    model.eval()
    gene_to_idx, idx_to_gene = get_gene_to_idx()
    
    available_genes = [gene for gene in list(gene_to_idx.keys())]
    
    gene = available_genes[0]
    pert_idx = gene_to_idx[gene]
    X_pert = test_adata[test_adata.obs['gene']==gene].X.toarray()
    x_pert = np.array([process_x(x) for x in X_pert])
    x_pert = x_pert.reshape(x_pert.shape[0], -1)[:,:GENE_LENGTH]
    x_ctrl_idxs = np.random.choice(ctrl_adata.X.toarray().shape[0], size=x_pert.shape[0], replace=False)
    x_ctrl = ctrl_adata.X.toarray()[x_ctrl_idxs]
    x_ctrl = np.array([process_x(x) for x in x_ctrl])
    pert_idx = np.array([ pert_idx for _ in range(x_ctrl.shape[0]) ])
    x_pred = get_model_output(model, pert_idx, x_ctrl)
    print("Done")


def main(cpkt_path):
    # --- CONFIG ---
    MAX_CELLS_FOR_MMD = 500  # Cap N cells to prevent MMD matrix explosion
    
    test_adata = ad.read_h5ad('data/test_data.h5ad')
    # Optimization: We only need Control data from train, not the whole object
    ctrl_adata = ad.read_h5ad('data/ctrl_data.h5ad')
    
    # Pre-calculate shape to avoid .toarray() later
    N_CONTROLS = ctrl_adata.shape[0]
    GENE_LENGTH = ctrl_adata.shape[1]
    
    # Free up train_adata to save RAM
    gc.collect()

    with open('data/gene_to_idx.pkl', 'rb') as f:
        gene_to_idx = pickle.load(f)
    available_genes = [gene for gene in list(gene_to_idx.keys()) if gene in test_adata.obs['gene'].unique()]
    
    # 2. Load Model
    model = CycleTransformer.load_from_checkpoint(cpkt_path)
    model = model.to('cpu')
    model.eval()
    
    # 3. Prepare Test Data Metadata (Subset once, not in loop)
    test_adata = test_adata[test_adata.obs['gene'].isin(available_genes)]
    
        # Initialize Metrics
    res_dict = {"MMD":[],"Wasserstein":[],"Var_pearson":[],"Energy":[],"L1":[]}
    dist_eval = DistributionEvaluator(device="cpu")
    
    pbar = tqdm(available_genes)
    
    print(f"Starting evaluation on {len(available_genes)} genes...")
    
    with torch.no_grad():
        for gene in pbar:
            pert_idx_int = gene_to_idx[gene]
            
            # --- OPTIMIZATION 1: Efficient Data Access ---
            # Get sparse slice first
            gene_mask = test_adata.obs['gene'] == gene
            X_pert_sparse = test_adata.X[gene_mask]
            
            # --- OPTIMIZATION 2: Subsampling ---
            # MMD is heavy (N^2). If we have 2000 cells, it crashes CPU RAM.
            # Downsample to MAX_CELLS_FOR_MMD (e.g. 500) if necessary.
            n_cells = X_pert_sparse.shape[0]
            if n_cells > MAX_CELLS_FOR_MMD:
                # Randomly select indices
                indices = np.random.choice(n_cells, MAX_CELLS_FOR_MMD, replace=False)
                X_pert_raw = X_pert_sparse[indices].toarray()
            else:
                X_pert_raw = X_pert_sparse.toarray()
            
            # Now n_cells is manageable
            current_n_cells = X_pert_raw.shape[0]
            print(f"Evaluating gene {gene} with {current_n_cells} cells...\n")
            
            # Process Real Perturbation
            x_pert = np.array([process_x(x) for x in X_pert_raw])
            print(x_pert.shape)
            x_pert = x_pert.reshape(current_n_cells, -1)[:,:GENE_LENGTH]
            
            # --- OPTIMIZATION 3: Efficient Control Sampling ---
            # Sample indices only
            if N_CONTROLS >= current_n_cells:
                ctrl_idxs = np.random.choice(N_CONTROLS, size=current_n_cells, replace=False)
            else:
                ctrl_idxs = np.random.choice(N_CONTROLS, size=current_n_cells, replace=True)
                
            # Slice SPARSE -> then Densify (Huge RAM saver)
            x_ctrl_raw = ctrl_adata.X[ctrl_idxs].toarray()
            
            # Process Control
            x_ctrl = np.array([process_x(x) for x in x_ctrl_raw])
            
            # Prepare Model Inputs
            pert_idx_arr = np.array([pert_idx_int] * current_n_cells)
            
            # --- PREDICTIONS ---
            # 1. Your Model
            x_pred = get_model_output(model, pert_idx_arr, x_ctrl)
            
            # 2. GEARS (Repeated to match population size)
            #gears_vec = gears_vec.flatten()[:GENE_LENGTH]
            
            # Convert to Tensor
            t_pert = torch.from_numpy(x_pert).float()
            t_pred = torch.from_numpy(x_pred).float()
            
            # Evaluate
            mmd, wasserstein, var_pres, energy, l1 = dist_eval.evaluate_batch(t_pert, t_pred)

            res_dict["MMD"].append(mmd)
            res_dict["Wasserstein"].append(wasserstein)
            res_dict["Var_pearson"].append(var_pres)
            res_dict["Energy"].append(energy)
            res_dict["L1"].append(l1)

            # Update Pbar with running averages (ignoring NaNs)
            pbar.set_postfix({
                "MMD": f"{np.nanmean(res_dict['MMD']):.3f}",
                "Energy": f"{np.nanmean(res_dict['Energy']):.3f}",
                "L1": f"{np.nanmean(res_dict['L1']):.3f}",
                "Sliced_Wasserstein": f"{np.nanmean(res_dict['Wasserstein']):.3f}",
            })

    # Save Results
    with open("results/distribution_all_gene.pkl", "wb") as f:  
        pickle.dump(res_dict, f)
    print("Finished")

if __name__ == "__main__":
    cfg = OmegaConf.load("config.yaml")
    
    cpkt_path = f"mlruns/887674207364271553/68ddf9acae564a9ca6ce1e0f45c5e2d9/checkpoints/best-all_gene_pearson.ckpt"
    #debug_eval(cpkt_path)
    print("Evaluating Precision Metrics...")
    main(cpkt_path)
    eval_precision(cpkt_path)