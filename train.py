import os
print("Training started")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  
print("Using GPU:", os.environ["CUDA_VISIBLE_DEVICES"])
from src.data.vae_data import get_loaders
from src.models.transformerVAE import TransformerVAEEncoder, TransformerVAEDecoder,TransformerClassifier, Transfomer_latent_Classifier
from src.models.vae_trainers import StateTrainer, StateTrainer_latent
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from omegaconf import OmegaConf


def main(encoder_config,decoder_config,classifier_config,trainer_config,training_config):
    #!TODO: Increase class factor over time.
    #!TODO: Alternating generator-discriminator training (epoch-epoch)
    # Keep kl at 0.8??
    modelcpkt_adipo = ModelCheckpoint(monitor="Val/AUROC_adipo",save_top_k=3,mode="max")
    modelcpkt_lipo = ModelCheckpoint(monitor="Val/AUROC_lipo",save_top_k=3,mode="max")
    earlystop = pl.callbacks.EarlyStopping(monitor="Val/AUROC_adipo",patience=10,mode="max")
    trainloader, valloader, *_ = get_loaders("",batch_size=training_config.batch_size)   
    encoder = TransformerVAEEncoder(**encoder_config)
    decoder = TransformerVAEDecoder(**decoder_config)
    classifier = TransformerClassifier(**classifier_config) if training_config.classify_on =="after" else Transfomer_latent_Classifier(**classifier_config)
    model = StateTrainer(encoder,decoder,categorizer=classifier,**trainer_config) if traning_config.classify_on =="after" else StateTrainer_latent(encoder,decoder,categorizer=classifier,**trainer_config)
    
    mlfLogger = MLFlowLogger(experiment_name=traning_config.projectName,run_name = training_config.run_name) 
    #wandb_logger = WandbLogger(project=training_config.projectName,name=training_config.run_name+ str(training_config.version),offline=True) 

    trainer = pl.Trainer(max_epochs=training_config.max_epochs, accelerator="auto", devices="auto",logger = mlfLogger, callbacks=[modelcpkt_adipo,modelcpkt_lipo,earlystop])
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)

if __name__ == "__main__":
    traning_config = OmegaConf.load("configs/traning.yaml")
    classtype = traning_config.classify_on
    encoder_config = OmegaConf.load("configs/encoder.yaml")
    decoder_config = OmegaConf.load("configs/decoder.yaml")
    if classtype == "latent":
        classifier_config = OmegaConf.load("configs/classifier_latent.yaml")
        print("Using latent classifier config")
    else:
        classifier_config = OmegaConf.load("configs/classifier.yaml")
    trainer_config = OmegaConf.load("configs/trainer.yaml")
    run_name_base = traning_config.run_name

    for latent_dim in [16,64,96,128]:
        classifier_config.z_dim = latent_dim
        decoder_config.z_dim = latent_dim
        encoder_config.z_dim = latent_dim
        traning_config.run_name = f"{run_name_base}_dim={latent_dim}"
        traning_config.version = traning_config.version+1
        OmegaConf.save(traning_config, "configs/traning.yaml")
        main(encoder_config=encoder_config,decoder_config=decoder_config,classifier_config=classifier_config,trainer_config=trainer_config,training_config=traning_config)