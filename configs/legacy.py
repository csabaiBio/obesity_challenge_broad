from dataclasses import dataclass
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
    kldFactor = 0.5
    reconFactor = 1.
    classFactor = 0.5
    multiple_optimizers:bool = True
    beta_start:float = 0.8
    kld_mode = "constant"
    class_warmup_steps:int = 20_000 

@dataclass
class TraningConfig:
    batch_size:int = 128 
    max_epochs:int = 20
    run_name:str = "transformerVAE_experiment"
    projectName:str = "obesity_challange"
    version:int = 0
