from src.models.CycleTransformerv2 import CycleTransformer as CycleTransformerv2
from src.data.perturbation_data import get_loaders
import torch
import anndata as ad


bestPath = "model.ckpt"

def main():
    trainloader, valloader, testloader, _, _ = get_loaders("",batch_size=64,num_workers=6)
    model = CycleTransformerv2.load_from_checkpoint(checkpoint_path = bestPath)
    model.configure_cycle()
    model.eval()
    