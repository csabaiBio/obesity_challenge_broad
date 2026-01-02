import numpy as np
import anndata as ad
import torch

import scipy.sparse

class CellDataset(torch.utils.data.Dataset):
    def __init__(self, adataTarget, adataInput, gene_to_idx, seq_length: int = 32):
        # 1. OPTIMIZATION: Ensure X is in CSR format for fast row slicing
        # (Slicing a CSC matrix by row is 100x slower)
        if scipy.sparse.isspmatrix_csc(adataTarget.X):
            self.target_X = adataTarget.X.tocsr()
        else:
            self.target_X = adataTarget.X

        if scipy.sparse.isspmatrix_csc(adataInput.X):
            self.input_X = adataInput.X.tocsr()
        else:
            self.input_X = adataInput.X

        # 2. OPTIMIZATION: Pre-convert Pandas columns to Numpy arrays
        # (Avoids slow .iloc lookup inside the loop)
        self.target_genes = adataTarget.obs['gene'].values  # Array of strings
        self.target_states = adataTarget.obs.iloc[:, [-1,-2,-3,-4]].to_numpy(dtype=np.float32)
        
        self.input_states = adataInput.obs.iloc[:, [-1,-2,-3,-4]].to_numpy(dtype=np.float32)
        
        self.gene_to_idx = gene_to_idx
        self.seq_length = seq_length
        self.num_inputs = self.input_X.shape[0]
        self.num_genes = len(gene_to_idx)
    def __len__(self):
        return self.target_X.shape[0]

    def __getitem__(self, idx):
        # 1. Fast Sparse Slicing (CSR is O(1) for rows)
        x_row = self.target_X[idx].toarray().ravel()
        x = np.concatenate([x_row, np.zeros(8, dtype=np.float32)]) 
        
        # 2. Random input selection
        inputidx = np.random.randint(0, self.num_inputs)
        x_in_row = self.input_X[inputidx].toarray().ravel()
        x_input = np.concatenate([x_in_row, np.zeros(8, dtype=np.float32)])
        
        # 3. Fast Numpy Lookup (No Pandas overhead)
        gene_name = self.target_genes[idx]
        gene_idx = self.gene_to_idx[gene_name]
        
        state = self.target_states[idx]
        input_state = self.input_states[inputidx]

        x = x.astype(np.float32)
        x_input = x_input.astype(np.float32)
        
        return (x.reshape(self.seq_length, -1), 
                x_input.reshape(self.seq_length, -1), 
                gene_idx, 
                state, 
                input_state)
    
def get_data(path:str):
    dataroot = path + "data"
    h5data = "obesity_challenge_1"
    obdata = ad.read_h5ad(f"{dataroot}/{h5data}.h5ad")
    gene_to_idx = {g: i for i, g in enumerate(obdata.var.index.to_numpy())}
    idx_to_gene = {i: g for g, i in gene_to_idx.items()}
    return obdata, gene_to_idx, idx_to_gene

def get_loaders(path:str,batch_size:int = 64,num_workers:int = 6):
    obdata, gene_to_idx, idx_to_gene = get_data(path)
    targetData = obdata[obdata.obs.gene != "NC"]
    targetData = targetData[targetData.obs.gene.isin(targetData.var.index.to_numpy()), :]
    inputData = obdata[obdata.obs.gene == "NC"]
    pin_memory = True if num_workers > 0 else False
    persistent_workers = True if num_workers > 0 else False

    dataset = CellDataset(targetData, inputData, gene_to_idx)
    traiset,valset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
    trainloader = torch.utils.data.DataLoader(traiset, batch_size = batch_size,shuffle= True,num_workers=num_workers,persistent_workers=persistent_workers,pin_memory=pin_memory)
    valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size,shuffle= False,num_workers=num_workers,persistent_workers=persistent_workers,pin_memory=pin_memory    )
    return trainloader, valloader, gene_to_idx, idx_to_gene