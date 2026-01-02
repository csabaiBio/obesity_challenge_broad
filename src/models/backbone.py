from src.models.transformerVAE import AttentionConditionEncoder, FiLMLatentTransition, ConditionEncoder, SimpleLatentTransition
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np  
import torchmetrics
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
import os




class ModelBase(pl.LightningModule):
    def __init__(self,
                reconFactor=1.0, kldFactor=1.0, classFactor=1.0,
                kl_warmup_steps=10_000,class_warmup_steps=10_000,free_bits=0.5,noise_warmup=10_000,KLD_MAX:float = 100.,beta_start:float=0.0,
                multiple_optimizers=False, reductionType ="mean",kld_mode="linear",categoryWeights= [1., 1.0, 1.0, 1.0],pretrained= True,
                latent_dim=128
                
                ):
        super().__init__()

        #self.encoder = encoder
        #self.decoder = decoder
        #if pretrained:
        #    self.decoder.eval()
        #    self.encoder.eval()
        self.reconFactor = reconFactor
        self.kldFactor = kldFactor
        self.classFactor = classFactor
        #self.categorizer = categorizer
        self.KLD_MAX = KLD_MAX
        self.noise_warmup = noise_warmup
        self.kl_warmup_steps = kl_warmup_steps
        self.class_warmup_steps = class_warmup_steps
        self.multiple_optimizers = multiple_optimizers
        if multiple_optimizers:
            self.automatic_optimization = False
        self.reduction = reductionType
        self.free_bits = free_bits
        self.kld_mode = kld_mode
        self.beta_start = beta_start
        self.pretrained = pretrained
        
        self.aucMetric = torchmetrics.AUROC(num_classes=4, average=None,task="multiclass")
        self.confmat = MulticlassConfusionMatrix(num_classes=4)
        classes = ['pre_adipo', 'adipo', 'lipo', 'other']
        self.lossWeights = torch.tensor(categoryWeights)
        self.class_names = classes
        
        self.save_hyperparameters(ignore=['encoder','decoder','categorizer','cond_embedder'])
        self.classwise_auc = ClasswiseWrapper(self.aucMetric,labels=classes)
        self.EntropyLoss = nn.CrossEntropyLoss(weight=self.lossWeights)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = 1e-4)
        
    def compute_kl_loss(self, mu, logvar):
        # KL per latent dimension
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)

        # Sum over latent dims, mean over batch
        kl = kl_per_dim.sum(dim=1).mean()
        return kl
    
    def kl_weight(self):
        # Linear warm-up
        if self.kld_mode == "sigmoid":
            x = min(self.kldFactor,0.1 +  self.kldFactor * self.global_step / self.kl_warmup_steps)
            sigmoid = lambda x: 1/(1 + np.exp(-x))
            return sigmoid(10 * (x - 0.5))
        elif self.kld_mode == "linear":
            return min(self.kldFactor, self.beta_start + (self.kldFactor * self.global_step / self.kl_warmup_steps))
        else:
            return self.kldFactor
        
    def classFactor_weight(self):
        x = min(self.classFactor,0.8 +  self.classFactor * self.global_step / self.class_warmup_steps)
        return x

    @property
    def current_class_weight(self):
            # MODIFIED: Start at 20% strength, ramp to 100% over 2000 steps
            # We want immediate guidance, but avoiding a "shock" in the first batch.
        progress = 0.2 + 0.8 * min(1.0, self.global_step / 2000.0)
        
        # NOTE: I recommend reducing classFactor to 0.1 or 0.2 based on your logs
        return self.classFactor * progress
    
    def logging_step(self,lossDict,beta,class_logits,state_indices,mode="Train"):
        for key, value in lossDict.items():
            self.log(f"{mode}/{key}", value, prog_bar=True,sync_dist=True)
        
        self.log(f"{mode}/beta", beta, prog_bar=True,sync_dist=True)
        
        if self.categorizer is not None:
            auc_scores = self.classwise_auc(class_logits, state_indices)           
            for aval, class_name in zip(auc_scores.values(),self.class_names): # Use your stored class names
                    self.log(f"{mode}/AUROC_{class_name}", aval, prog_bar=False, sync_dist=True)

        if mode == "Val" and self.categorizer is not None:
            self.confmat.update(class_logits, state_indices)

    def shared_step(self, batch, mode="Train"):
        Xs, x_in,target_gene, state, input_state = batch
        with torch.no_grad():
            mu, logvar = self.encoder(x_in)
            logvar = torch.clamp(logvar, min=-6.0, max=2.0)
            z = self.reparameterize(mu, logvar)

        cond = self.cond_embedder(target_gene)
        z = z + self.LatentTransition(z, cond)
        
        Xs_hat = self.decoder(z)
        loss_recon = nn.functional.mse_loss(Xs_hat, Xs, reduction=self.reduction)
        loss_kld = self.compute_kl_loss(mu, logvar)
        #loss_kld = torch.clamp(loss_kld, max=self.KLD_MAX)
        beta = self.kl_weight()

        if self.categorizer is not None:
            class_logits = self.categorizer(z)
            loss_class = nn.functional.cross_entropy(class_logits, state)
        else:
            class_logits = None
            loss_class = torch.tensor(0.0, device=Xs.device)


        if state.shape == class_logits.shape:
            state_indices = torch.argmax(state, dim=1)
            # 2. If state is [Batch, 1] -> Squeeze to [Batch]
        elif state.ndim == 2 and state.shape[1] == 1:
            state_indices = state.squeeze(1)
        else:
            state_indices = state
        classWeight = self.current_class_weight
        #self.classFactor_weight() if beta > 0.95 else 0.8
        total_loss = (self.reconFactor * loss_recon)+ (classWeight * loss_class)

        # Logging
        self.logging_step(loss_recon, loss_kld, beta, loss_class, total_loss, class_logits, state_indices, mode=mode)

        return total_loss
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        eps_scale = min(1.0, self.global_step / self.noise_warmup)
        return mu + eps_scale*eps * std



    def getcfmx(self):
        cm_tensor = self.confmat.compute()
        cm_numpy = cm_tensor.cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_numpy, 
            annot=True, 
            fmt='g',            # 'g' prevents scientific notation (e.g., 1e3)
            cmap='Blues', 
            xticklabels=self.class_names, 
            yticklabels=self.class_names,
            ax=ax
            )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix - Epoch {self.current_epoch}')
        temp_filename = f"confusion_matrix{self.current_epoch}.png"
        return fig, temp_filename

    def on_validation_epoch_end(self):
        if self.categorizer is not None:
            fig, temp_filename = self.getcfmx()
            unique_filename = f"conf_matrix_epoch_{self.current_epoch:03d}_rank_{self.global_rank}.png"
            fig.savefig(unique_filename) 
            plt.close(fig)
            if self.loggers:
                for logger in self.loggers:
                    if isinstance(logger, MLFlowLogger):
                        logger.experiment.log_artifact(
                            run_id=logger.run_id, 
                            local_path=unique_filename, 
                            artifact_path="plots"
                        )
            elif isinstance(self.logger, MLFlowLogger):
                self.logger.experiment.log_artifact(
                    run_id=self.logger.run_id, 
                    local_path=unique_filename, 
                    artifact_path="plots"
                )

        self.confmat.reset()
        if os.path.exists(unique_filename):
            os.remove(unique_filename)
    
