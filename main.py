import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from CycleTransformer.dataloader import get_loaders
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from CycleTransformer.CycleTransformer_v2 import CycleTransformer
import torch
checkpointRoot = "cpkts/"
import numpy as np

def phase1(cfg,data_directory_path,model_directory_path):
    trainloader, valloader, *_ = get_loaders(path = data_directory_path, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    modelKwargs = cfg.model_kwargs
    model = CycleTransformer(phase=1,**modelKwargs)
    logger = MLFlowLogger(experiment_name="Obesity Challange",run_name= "changed_class_phase1",log_model=False)
    max_var = ModelCheckpoint(monitor="val/Health_Var_Preservation",mode="max",save_top_k=1,filename="best-var_pres_changed_class",dirpath=model_directory_path)
    trainer = pl.Trainer(max_epochs=cfg.max_epochs,accelerator="auto",devices="auto" ,callbacks=[max_var],logger=logger
                        ,strategy="ddp_find_unused_parameters_true"
                                )
    trainer.fit(model,train_dataloaders=trainloader, val_dataloaders=valloader)

def phase2(cfg,data_directory_path,model_directory_path):
    trainloader, valloader, *_ = get_loaders(path = data_directory_path, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    logger = MLFlowLogger(experiment_name="Obesity Challange",run_name= "changed_class_phase2",log_model=False)
    best_all_pearson = ModelCheckpoint(monitor="val/Pearson_All",mode="max",save_top_k=1,filename="best-all_gene_pearson_changed_class",dirpath=model_directory_path)
    cpkt_path = f"{model_directory_path}/best-var_pres_changed_class.ckpt"
    model = CycleTransformer.load_from_checkpoint(cpkt_path)
    model.init_phase2()
    trainer = pl.Trainer(max_epochs=cfg.max_epochs,strategy="ddp_find_unused_parameters_true",num_sanity_val_steps=0,callbacks=[best_all_pearson],logger=logger)
    trainer.fit(model,train_dataloaders=trainloader, val_dataloaders=valloader)


def train(data_directory_path,model_directory_path=checkpointRoot):
    cfg = OmegaConf.load("config.yaml")
    phase1(cfg,data_directory_path,model_directory_path)
    phase2(cfg,data_directory_path,model_directory_path)



if __name__ == "__main__":
    data_directory_path = "data/obesity_challenge_1.h5ad"
    model_directory_path = checkpointRoot
    train(data_directory_path,model_directory_path)
    #predict_perturbations = np.loadtxt("../data/predict_perturbations.txt", dtype=str)
    #genes_to_predict = np.loadtxt("../data/genes_to_predict.txt", dtype=str)


    #infer(data_directory_path="../data/obesity_challenge_1.h5ad",prediction_directory_path="preds",prediction_h5ad_file_path="preds/prediction.h5ad", program_proportion_csv_file_path="preds/predict_program_propotion.csv",
    #    model_directory_path=checkpointRoot, predict_perturbations=predict_perturbations, genes_to_predict=genes_to_predict)