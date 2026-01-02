import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"  
print(os.getcwd())

from src.data.perturbation_data import get_loaders
from src.models.CycleTransformer import CycleTransformer
import pytorch_lightning as pl
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

def main_phase1(cfg):
    trainloader, valloader, *_ = get_loaders("",batch_size=cfg.batch_size,num_workers=cfg.num_workers)  
    modelKwargs = cfg.model_kwargs
    modelKwargs["phase"] = 1
    model = CycleTransformer(**modelKwargs)
    cpkt_callback = ModelCheckpoint(monitor="val/total_loss",mode="min",save_top_k=1,filename="best-checkpoint")
    mlflow_logger = MLFlowLogger(experiment_name=cfg.experiment_name,run_name=cfg.run_name)
    trainer = pl.Trainer(max_epochs=cfg.max_epochs,logger=mlflow_logger,accelerator="auto",devices="auto", strategy="ddp_find_unused_parameters_true"
                        ,callbacks=[cpkt_callback])
    trainer.fit(model,train_dataloaders=trainloader,val_dataloaders=valloader)



def main_phase2(cfg):
    trainloader, valloader, *_ = get_loaders("",batch_size=cfg.batch_size,num_workers=cfg.num_workers)  
    modelKwargs = cfg.model_kwargs
    modelKwargs["phase"] = 2
    pretrained_path = "misc/best_runs/cycle/checkpoints/best-checkpoint.ckpt"
    model = CycleTransformer.load_from_checkpoint(checkpoint_path = pretrained_path)#(**modelKwargs)
    model.replaceTransition()
    model.add_factors(cfg.obsessionFactor, cfg.diversityFactor)
    cpkt_callback = ModelCheckpoint(monitor="val/total_loss",mode="min",save_top_k=1,filename="best-checkpoint")
    mlflow_logger = MLFlowLogger(experiment_name=cfg.experiment_name,run_name=cfg.run_name)
    trainer = pl.Trainer(max_epochs=cfg.max_epochs,logger=mlflow_logger,accelerator="auto",devices="auto" #, strategy="ddp_find_unused_parameters_true"
                        ,callbacks=[cpkt_callback])
    trainer.fit(model,train_dataloaders=trainloader,val_dataloaders=valloader)

if __name__ == "__main__":
    cfg = OmegaConf.load("configs/cycle_transformer.yaml")
    phase = 1
    if phase == 1:
        main_phase1(cfg)
    else:
        main_phase2(cfg)
