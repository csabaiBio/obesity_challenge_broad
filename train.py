import os
print("Training started")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  
print("Using GPU:", os.environ["CUDA_VISIBLE_DEVICES"])
from src.data.vae_data import get_loaders
from src.models.transformerVAE import TransformerVAEEncoder, TransformerVAEDecoder,TransformerClassifier ,StateTrainer
from dataclasses import dataclass, asdict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger


@dataclass
class EncoderConfig:
    input_dim: int = 675
    d_model:int  = 512
    n_layers:int = 6
    n_heads:int = 8
    z_dim:int = 32

@dataclass
class DecoderConfig:
    output_dim: int = 675
    d_model:int  = 512
    num_tokens:int = 32
    n_layers:int = 2
    n_heads:int = 8
    z_dim:int = 32

@dataclass
class ClassifierConfig:
    input_dim: int = 675
    d_model:int  = 256
    n_layers:int = 2
    n_heads:int = 8
    num_classes:int = 4

@dataclass
class TrainerConfig:
    free_bits:float = 0.02
    kl_warmup_steps = 10_000
    noise_warmup = 10_000
    kldFactor = 1.75
    reconFactor = 1.2
    classFactor = 0.5
    max_epochs:int = 20
    multiple_optimizers:bool = True
    batch_size:int = 128 
    beta_start:float = 0.8

encoder_config = EncoderConfig()
decoder_config = DecoderConfig()
classifier_config = ClassifierConfig()
trainer_config = TrainerConfig()

def main():
    modelcpkt = ModelCheckpoint(monitor="Val/total_loss",save_top_k=3,mode="min")
    
    trainloader, valloader, *_ = get_loaders("",batch_size=trainer_config.batch_size)   
    print("Data loaders created")
    encoder = TransformerVAEEncoder(**asdict(encoder_config))
    decoder = TransformerVAEDecoder(**asdict(decoder_config))
    classifier = TransformerClassifier(**asdict(classifier_config))
    print("Encoder and Decoder created")

    model = StateTrainer(encoder,decoder,categorizer=classifier,reconFacor=trainer_config.reconFactor,kldFactor=trainer_config.kldFactor,classFactor=trainer_config.classFactor,
                        kl_warmup_steps=trainer_config.kl_warmup_steps,noise_warmup=trainer_config.noise_warmup ,free_bits=trainer_config.free_bits,
                        multiple_optimizers=trainer_config.multiple_optimizers,beta_start=trainer_config.beta_start)
    
    mlfLogger = MLFlowLogger(experiment_name="transformerVAE_class",
                            tags={"model":"transformerVAE","note":"initial_run","type":"stability_search"}
                            ,run_name = "AUC_logging_test") 
    
    trainer = pl.Trainer(max_epochs=trainer_config.max_epochs, accelerator="auto", devices="auto",logger = mlfLogger, callbacks=[modelcpkt])
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)

if __name__ == "__main__":
    main()