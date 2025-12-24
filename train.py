import os
print("Training started")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  
print("Using GPU:", os.environ["CUDA_VISIBLE_DEVICES"])
from src.data.vae_data import get_loaders
from src.models.transformerVAE import TransformerVAEEncoder, TransformerVAEDecoder,TransformerClassifier ,StateTrainer
from dataclasses import dataclass, asdict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
import wandb
import numpy as np
from omegaconf import OmegaConf


def main(encoder_config,decoder_config,classifier_config,trainer_config,training_config):
    #!TODO: Increase class factor over time.
    #!TODO: Alternating generator-discriminator training (epoch-epoch)
    # Keep kl at 0.8??
    modelcpkt = ModelCheckpoint(monitor="Val/total_loss",save_top_k=3,mode="min")
    
    trainloader, valloader, *_ = get_loaders("",batch_size=training_config.batch_size)   
    encoder = TransformerVAEEncoder(**encoder_config)
    decoder = TransformerVAEDecoder(**decoder_config)
    classifier = TransformerClassifier(**classifier_config)
    model = StateTrainer(encoder,decoder,categorizer=classifier,**trainer_config)
    
    mlfLogger = MLFlowLogger(experiment_name=traning_config.projectName,run_name = training_config.run_name + str(training_config.version)) 
    wandb_logger = WandbLogger(project=training_config.projectName,name=training_config.run_name+ str(training_config.version),offline=True) 

    trainer = pl.Trainer(max_epochs=training_config.max_epochs, accelerator="auto", devices="auto",logger = [mlfLogger,wandb_logger], callbacks=[modelcpkt])
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)

if __name__ == "__main__":
    encoder_config = OmegaConf.load("configs/encoder.yaml")
    decoder_config = OmegaConf.load("configs/decoder.yaml")
    classifier_config = OmegaConf.load("configs/classifier.yaml")
    trainer_config = OmegaConf.load("configs/trainer.yaml")
    traning_config = OmegaConf.load("configs/traning.yaml")
    traning_config.version = traning_config.version+1
    OmegaConf.save(traning_config, "configs/traning.yaml")
    main(encoder_config=encoder_config,decoder_config=decoder_config,classifier_config=classifier_config,trainer_config=trainer_config,training_config=traning_config)