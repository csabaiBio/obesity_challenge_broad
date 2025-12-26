import numpy as np
import anndata as ad
import torch

class CellDataset():
    def __init__(self,adataTarget,adataInput, gene_to_idx,seq_length:int = 32):
        self.adata = adataTarget
        self.adataInput = adataInput
        self.gene_to_idx = gene_to_idx
        self.seq_length = seq_length
    def __len__(self):
        return self.adata.X.shape[0]

    def __getitem__(self,idx):
        x = np.concatenate([self.adata.X[idx].toarray().ravel(), [0,0,0,0,0,0,0,0]],dtype = np.float32) # Extra padding to have nice sequences
        inputidx = np.random.randint(0,self.adataInput.X.shape[0])
        x_input = np.concatenate([self.adataInput.X[inputidx].toarray().ravel(), [0,0,0,0,0,0,0,0]],dtype = np.float32) # Extra padding to have nice sequences
        
        gene = self.adata.obs.gene.iloc[idx]
        gene = self.gene_to_idx[gene]
        state = self.adata.obs.iloc[idx,[-1,-2,-3,-4]].to_numpy(dtype=np.float32)
        input_state = self.adataInput.obs.iloc[inputidx,[-1,-2,-3,-4]].to_numpy(dtype=np.float32)
        return x.reshape(self.seq_length,-1), x_input.reshape(self.seq_length,-1), gene, state, input_state
    
def get_data(path:str):
    dataroot = path + "data"
    h5data = "obesity_challenge_1"
    obdata = ad.read_h5ad(f"{dataroot}/{h5data}.h5ad")
    obdata.var.index.unique().to_numpy()
    gene_to_idx = {g: i for i, g in enumerate(obdata.var.index.to_numpy())}
    idx_to_gene = {i: g for g, i in gene_to_idx.items()}
    return obdata, gene_to_idx, idx_to_gene

def get_loaders(path:str,batch_size:int = 64):
    obdata, gene_to_idx, idx_to_gene = get_data(path)
    targetData = obdata[obdata.obs.gene != "NC"]
    inputData = obdata[obdata.obs.gene == "NC"]
    dataset = CellDataset(targetData, inputData, gene_to_idx)
    traiset,valset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
    trainloader = torch.utils.data.DataLoader(traiset, batch_size = batch_size,shuffle= True,)
    valloader = torch.utils.data.DataLoader(valset, batch_size = batch_size,shuffle= False)
    return trainloader, valloader, gene_to_idx, idx_to_gene