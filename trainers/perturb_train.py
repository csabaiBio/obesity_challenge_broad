import os

from src.utils import CosineMSELoss
print("Training started")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  
print("Using GPU:", os.environ["CUDA_VISIBLE_DEVICES"])
from src.data.perturbation_data import get_loaders
from src.models.transformerVAE import TransformerVAEEncoder, TransformerVAEDecoder, Transfomer_latent_Classifier 
from src.models.vae_trainers import StateTrainer, StateTrainer_latent, PerturbationAdder
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from omegaconf import OmegaConf
import torch

def pretrainedmodels(encoder_config,decoder_config,trainer_config,pretrained=True):
    encoder = TransformerVAEEncoder(**encoder_config)
    decoder = TransformerVAEDecoder(**decoder_config)
    classifier_config = OmegaConf.load("configs/classifier_latent.yaml")
    classifier_config.z_dim = encoder_config.z_dim
    classifier = Transfomer_latent_Classifier(**classifier_config)
    if pretrained:
        cpkt_path = "misc/best_runs/latent_reg/checkpoints/epoch=19-step=5520.ckpt"
        state_dict = torch.load(cpkt_path,weights_only=False)
        model = StateTrainer_latent(encoder,decoder,categorizer=classifier,**trainer_config)
        model.load_state_dict(state_dict["state_dict"])
        return model.encoder, model.decoder, model.categorizer
    else:
        return encoder, decoder, classifier 

def main(encoder_config,decoder_config,classifier_config,trainer_config,training_config):
    modelcpkt_adipo = ModelCheckpoint(monitor="Val/AUROC_adipo",save_top_k=3,mode="max")
    modelcpkt_lipo = ModelCheckpoint(monitor="Val/AUROC_lipo",save_top_k=3,mode="max")

    trainloader, valloader, *_ = get_loaders("",batch_size=training_config.batch_size)
    reconLoss = CosineMSELoss(alpha=training_config.recon_alpha)
    encoder, decoder, classifier = pretrainedmodels(encoder_config,decoder_config,trainer_config,pretrained=training_config.pretrained)
    model = PerturbationAdder(encoder,decoder,categorizer=classifier,**trainer_config, reconLoss=reconLoss)
    mlfLogger = MLFlowLogger(experiment_name=traning_config.projectName,run_name = training_config.run_name + str(training_config.version//4)) 

    trainer = pl.Trainer(max_epochs=training_config.max_epochs, accelerator="auto", devices="auto",logger = mlfLogger, callbacks=[modelcpkt_adipo,modelcpkt_lipo])
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)

if __name__ == "__main__":
    traning_config = OmegaConf.load("configs/pert_training.yaml")
    encoder_config = OmegaConf.load("configs/encoder.yaml")
    decoder_config = OmegaConf.load("configs/decoder.yaml")
    #classifier_config = OmegaConf.load("configs/classifier_latent.yaml")
    classifier_config = OmegaConf.load("configs/classifier.yaml")
    trainer_config = OmegaConf.load("configs/pert_trainer.yaml")
    latent_dim = 128
    decoder_config.z_dim = latent_dim
    encoder_config.z_dim = latent_dim
    traning_config.version = traning_config.version+1
    #for classFactor in [0.05,0.1,0.25]:
    main(encoder_config=encoder_config,decoder_config=decoder_config,classifier_config=classifier_config,trainer_config=trainer_config,training_config=traning_config)  
    OmegaConf.save(traning_config, "configs/pert_training.yaml")