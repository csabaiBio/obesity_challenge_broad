import anndata as ad
from CycleTransformer.cycleTransformer import CycleTransformer
import pickle
import torch
import numpy as np
import pandas as pd

NUM_GENES = 21592

def get_model_output(model, pert_idx,x_ctrl):
    x_ctrl = torch.from_numpy(x_ctrl).float().to(model.device)
    pert_idx = torch.from_numpy(pert_idx).long().to(model.device)
    z_ctrl = model.encoder(x_ctrl) 
    z_prompt_fwd = model.get_perturbation_prompt(x_ctrl, pert_idx)
    delta_fwd = model.transition_fwd(z_ctrl, z_prompt_fwd)
    z_fake_pert = z_ctrl + delta_fwd
    logits = model.latentClassifier(z_fake_pert)
    x_recon_pert = model.decoder(z_fake_pert)
    x_recon_pert = x_recon_pert.detach().cpu().numpy()
    return x_recon_pert.reshape(x_recon_pert.shape[0],-1)[:,:NUM_GENES], logits.detach().cpu()

def get_model_input(x, seq_length=32):
    x_row = x.toarray()
    batch_size = x_row.shape[0]
    x = np.concatenate([x_row, np.zeros((100, 8), dtype=np.float32)], axis=1)
    return x.reshape(batch_size,seq_length, -1)

def infer(
    data_directory_path: str,
    prediction_directory_path: str,
    prediction_h5ad_file_path: str,
    program_proportion_csv_file_path: str,
    model_directory_path: str,
    predict_perturbations: list[str],
    genes_to_predict: list[str],cells_per_perturbation: int = 100):

    adata = ad.read_h5ad(data_directory_path)
    ctrl_cells = adata[adata.obs.gene == "NC"]
    
    #savePath = prediction_h5ad_file_path if prediction_h5ad_file_path else prediction_directory_path + "/prediction.h5ad"
    #savePropPath = program_proportion_csv_file_path if program_proportion_csv_file_path else prediction_directory_path + "/predict_program_proportion.csv"
    
    model = CycleTransformer.load_from_checkpoint(model_directory_path + "/best-all_gene_pearson.ckpt", map_location="cpu", weights_only=False)
    model.eval()
    numCellstoPredict = cells_per_perturbation
    
    with open("gene_to_idx.pkl", "rb") as f:
        gene_to_idx = pickle.load(f)
    
    num_ctrl_genes = ctrl_cells.shape[0]
    idx_to_class = { 0:'lipo', 1:'other', 2:'pre_adipo', 3:'adipo'}
    X_data = []
    propotions = pd.DataFrame(columns=['gene','pre_adipo','adipo','other','lipo','lipo_adipo'])
    
    for pert in predict_perturbations:
        print(f"Gene:{pert}")
        ctrl_idxs = np.random.choice(num_ctrl_genes, 100,replace=False)
        ctrl_genes = ctrl_cells[ctrl_idxs]
        x_ctrl = get_model_input(ctrl_genes.X)
        pert_idx = np.array([gene_to_idx[pert]] * numCellstoPredict)
        x_pred, logits = get_model_output(model, pert_idx, x_ctrl)
        X_data.append(x_pred)
        categories = torch.argmax(logits, axis=1)
        counts = torch.bincount(categories,minlength  = 4)
        proportion = counts / counts.sum()
        lipo = proportion[0].item()
        other = proportion[1].item()
        pre_adipo = proportion[2].item()
        adipo = proportion[3].item()
        lipo_adipo = lipo/adipo
        tmp = pd.DataFrame({'gene':[pert],'adipo': [adipo], 'pre_adipo': [pre_adipo],'other':[other], 'lipo':[lipo], 'lipo_adipo':[lipo_adipo]})
        propotions = pd.concat([propotions, tmp], ignore_index=True)
        
    X_data = np.array(X_data).reshape(-1, NUM_GENES)
    adata_pred = ad.AnnData(X=X_data)
    adata_pred.obs['perturbation'] = np.repeat(predict_perturbations, numCellstoPredict)
    gene_mask = np.isin(adata.var_names, genes_to_predict)
    adata_pred = adata_pred[:, gene_mask]

    adata_pred.write_h5ad(prediction_h5ad_file_path)
    propotions.to_csv(program_proportion_csv_file_path, index=False)