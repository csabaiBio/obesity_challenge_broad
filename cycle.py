import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

from src.data.perturbation_data import get_loaders
from src.models.CycleTransformer import CycleTransformer
from src.models.CycleTransformerv2 import CycleTransformer as CycleTransformerv2
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
    cpkt_name ="latent_reg"
    pretrained_path = f"misc/best_runs/{cpkt_name}/checkpoints/best-checkpoint.ckpt"
    model = CycleTransformerv2.load_from_checkpoint(checkpoint_path = pretrained_path)#(**modelKwargs)
    model.configure_cycle()
    cpkt_callback = ModelCheckpoint(monitor="val/g_total_loss",mode="min",save_top_k=1,filename="best-checkpoint")
    cfg.experiment_name = "LearningAllPerturb"
    cfg.run_name = "CycleTransformer_v2_allPerturbations"
    mlflow_logger = MLFlowLogger(experiment_name=cfg.experiment_name,run_name=cfg.run_name)
    trainer = pl.Trainer(max_epochs=cfg.max_epochs,logger=mlflow_logger,accelerator="auto",devices="auto" ,callbacks=[cpkt_callback],strategy="ddp_find_unused_parameters_true")
    trainer.fit(model,train_dataloaders=trainloader,val_dataloaders=valloader)


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/cycle_transformer.yaml")
    phase = 2
    if phase == 1:
        main_phase1(cfg)
    else:
        main_phase2(cfg)
