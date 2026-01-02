from src.models.transformerVAE import AttentionConditionEncoder, FiLMLatentTransition, ConditionEncoder, SimpleLatentTransition
from src.models.backbone import ModelBase
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



class StateTrainer(ModelBase):
    def __init__(self,encoder,decoder,categorizer,
                reconFactor=1.0, kldFactor=1.0, classFactor=1.0,
                kl_warmup_steps=10_000,class_warmup_steps=10_000,free_bits=0.5,noise_warmup=10_000,KLD_MAX:float = 100.,beta_start:float=0.0,
                multiple_optimizers=False, reductionType ="mean",kld_mode="linear",categoryWeights= [1., 1.0, 1.0, 1.0]):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.reconFactor = reconFactor
        self.kldFactor = kldFactor
        self.classFactor = classFactor
        self.categorizer = categorizer
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
        
        self.aucMetric = torchmetrics.AUROC(num_classes=4, average=None,task="multiclass")
        self.confmat = MulticlassConfusionMatrix(num_classes=4)
        classes = ['pre_adipo', 'adipo', 'lipo', 'other']
        self.class_names = classes
        self.classwise_auc = ClasswiseWrapper(self.aucMetric,labels=classes)
        self.lossWeights = torch.tensor(categoryWeights)
        
        self.save_hyperparameters(ignore=['encoder','decoder','categorizer'])
        self.EntropyLoss = nn.CrossEntropyLoss(weight=self.lossWeights)

    def configure_optimizers(self):
        params = [{'params':self.categorizer.parameters(),'lr':1e-3},{'params':self.encoder.parameters(),'lr':1e-3},{'params':self.decoder.parameters(),'lr':4e-3}]
        if self.multiple_optimizers:
            return torch.optim.Adam(params)
        else:
            return torch.optim.Adam(self.parameters(),lr = 1e-3)
        

    def shared_step(self, batch, mode="Train"):
        Xs, _, state = batch
        mu, logvar = self.encoder(Xs)
        logvar = torch.clamp(logvar, min=-6.0, max=2.0)
        z = self.reparameterize(mu, logvar)
        #z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
        Xs_hat = self.decoder(z)

        loss_recon = nn.functional.mse_loss(Xs_hat, Xs, reduction=self.reduction)
        loss_kld = self.compute_kl_loss(mu, logvar)
        loss_kld = torch.clamp(loss_kld, max=self.KLD_MAX)
        beta = self.kl_weight()

        if self.categorizer is not None:
            class_logits = self.categorizer(Xs_hat)
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
        classWeight = self.classFactor 
        #self.classFactor_weight() if beta > 0.95 else 0.8
        total_loss = (self.reconFactor * loss_recon+ beta * loss_kld+ classWeight * loss_class)

        # Logging
        self.logging_step(loss_recon, loss_kld, beta, loss_class, total_loss, class_logits, state_indices, mode=mode)
        return total_loss


class StateTrainer_latent(pl.LightningModule):
    def __init__(self,encoder,decoder,categorizer,
                reconFactor=1.0, kldFactor=1.0, classFactor=1.0,
                kl_warmup_steps=10_000,class_warmup_steps=10_000,free_bits=0.5,noise_warmup=10_000,KLD_MAX:float = 100.,beta_start:float=0.0,
                multiple_optimizers=False, reductionType ="mean",kld_mode="linear",categoryWeights= [1., 1.0, 1.0, 1.0],reconLoss= nn.MSELoss()):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.reconFactor = reconFactor
        self.kldFactor = kldFactor
        self.classFactor = classFactor
        self.categorizer = categorizer
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
        
        self.aucMetric = torchmetrics.AUROC(num_classes=4, average=None,task="multiclass")
        self.confmat = MulticlassConfusionMatrix(num_classes=4)
        classes = ['pre_adipo', 'adipo', 'lipo', 'other']
        self.lossWeights = torch.tensor(categoryWeights)
        self.class_names = classes
        
        self.save_hyperparameters(ignore=['encoder','decoder','categorizer'])
        self.classwise_auc = ClasswiseWrapper(self.aucMetric,labels=classes)
        self.EntropyLoss = nn.CrossEntropyLoss(weight=self.lossWeights)
        self.reconLoss = reconLoss

    def configure_optimizers(self):
        encoderOptimizer = torch.optim.Adam(self.encoder.parameters(),lr = 1e-3)
        decoderOptimizer = torch.optim.Adam(self.decoder.parameters(),lr = 4e-3)
        if self.multiple_optimizers:
            return [encoderOptimizer,decoderOptimizer]
        else:
            return torch.optim.Adam(self.parameters(),lr = 1e-3)
        
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
        x = min(self.classFactor, self.classFactor * self.global_step / self.class_warmup_steps)
        return x

    def logging_step(self,loss_recon,loss_kld,beta,loss_class,total_loss,class_logits,state_indices,mode="Train"):
        self.log(f"{mode}/loss_recon", loss_recon, prog_bar=True,sync_dist=True)
        self.log(f"{mode}/loss_kld", loss_kld, prog_bar=True,sync_dist=True)
        self.log(f"{mode}/beta", beta, prog_bar=True,sync_dist=True)
        self.log(f"{mode}/loss_class", loss_class, prog_bar=True,sync_dist=True)
        self.log(f"{mode}/total_loss", total_loss, prog_bar=True,   sync_dist=True)
        
        if self.categorizer is not None:
            auc_scores = self.classwise_auc(class_logits, state_indices)           
            for aval, class_name in zip(auc_scores.values(),self.class_names): # Use your stored class names
                    self.log(f"{mode}/AUROC_{class_name}", aval, prog_bar=False, sync_dist=True)

        if mode == "Val" and self.categorizer is not None:
            self.confmat.update(class_logits, state_indices)


    def compute_retrieval_accuracy(self, x_true, x_hat):
        """
        Checks if x_hat[i] is closer to x_true[i] than to any other x_true[j].
        """
        # 1. Flatten the inputs from [Batch, Tokens, Genes] to [Batch, Total_Genes]
        # start_dim=1 keeps the Batch dim (0) and merges everything else
        x_true_flat = x_true.flatten(start_dim=1) 
        x_hat_flat  = x_hat.flatten(start_dim=1)

        # 2. Normalize flattened vectors to unit length
        #x_true_norm = nn.functional.normalize(x_true_flat, p=2, dim=1)
        #x_hat_norm  = nn.functional.normalize(x_hat_flat,  p=2, dim=1)
        
        # 3. Compute Pairwise Similarity Matrix (Batch x Batch)
        # Now we have [B, N] @ [N, B] -> [B, B]
        similarity_matrix = torch.mm(x_hat_flat, x_true_flat.t())
        
        # 4. Find the index of the "closest match"
        predicted_indices = torch.argmax(similarity_matrix, dim=1)
        
        # 5. Calculate Accuracy
        true_indices = torch.arange(x_true.size(0), device=x_true.device)
        correct_matches = (predicted_indices == true_indices).sum()
        
        accuracy = correct_matches.float() / x_true.size(0)
        return accuracy

    def shared_step(self, batch, mode="Train"):
        Xs, _, state = batch
        mu, logvar = self.encoder(Xs)
        logvar = torch.clamp(logvar, min=-6.0, max=2.0)
        z = self.reparameterize(mu, logvar)
        # Shape of Z : (Batch, z_dim)
        #z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
        Xs_hat = self.decoder(z)

        loss_recon = self.reconLoss(Xs_hat, Xs)
        loss_kld = self.compute_kl_loss(mu, logvar)
        kld_origin = loss_kld
        loss_kld = torch.clamp(loss_kld, max=self.KLD_MAX)
        beta = self.kl_weight()

        if self.categorizer is not None:
            ### Check z-shape and adapt if necessary
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
        classWeight = 1.#self.classFactor_weight() #self.classFactor if currentEpoch > 5 else 0.0 
        #self.classFactor_weight() if beta > 0.95 else 0.8
        total_loss = (self.reconFactor * loss_recon+ beta * loss_kld+ classWeight * loss_class)

        # Logging
        retrival_accuracy = self.compute_retrieval_accuracy(Xs, Xs_hat)
        self.log(f"{mode}/retrival_accuracy", retrival_accuracy, prog_bar=True,sync_dist=True)
        self.logging_step(loss_recon, kld_origin, beta, loss_class, total_loss, class_logits, state_indices, mode=mode)

        return total_loss

    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        eps_scale = min(1.0, self.global_step / self.noise_warmup)
        return mu + eps_scale*eps * std
    
    def training_step(self,batch,batch_idx):
        if self.multiple_optimizers:
            encoder_opt, decoder_opt = self.optimizers()
            loss = self.shared_step(batch,mode="Train")
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            self.manual_backward(loss)
            encoder_opt.step()
            decoder_opt.step()
            return loss
        else:
            loss = self.shared_step(batch,mode="Train")
            return loss
    
    def validation_step(self,batch,batch_idx):
        loss = self.shared_step(batch,mode="Val")
        return loss

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


class PerturbationAdder(ModelBase):
    def __init__(self,encoder,decoder,categorizer,
                reconFactor=1.0, kldFactor=1.0, classFactor=1.0,
                kl_warmup_steps=10_000,class_warmup_steps=10_000,free_bits=0.5,noise_warmup=10_000,KLD_MAX:float = 100.,beta_start:float=0.0,
                multiple_optimizers=False, reductionType ="mean",kld_mode="linear",categoryWeights= [1., 1.0, 1.0, 1.0],pretrained= True,
                latent_dim=128
                
                ):
        super().__init__()
        
        if pretrained:
            for param in decoder.parameters():
                param.requires_grad = False
            for param in encoder.parameters():
                param.requires_grad = False

        self.encoder = encoder
        self.decoder = decoder
        if pretrained:
            self.decoder.eval()
            self.encoder.eval()
        self.cond_embedder = AttentionConditionEncoder()
        self.LatentTransition = FiLMLatentTransition(latent_dim=latent_dim,cond_dim=latent_dim)

        self.reconFactor = reconFactor
        self.kldFactor = kldFactor
        self.classFactor = classFactor
        self.categorizer = categorizer
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


class CycleConsistentPerturbationAdder(PerturbationAdder):
    def __init__(self,encoder,decoder,categorizer,
                reconFactor=1.0, kldFactor=1.0, classFactor=1.0,
                kl_warmup_steps=10_000,class_warmup_steps=10_000,free_bits=0.5,noise_warmup=10_000,KLD_MAX:float = 100.,beta_start:float=0.0,
                multiple_optimizers=False, reductionType ="mean",kld_mode="linear",categoryWeights= [1., 1.0, 1.0, 1.0],pretrained= True,
                latent_dim=128, cycleFactor=1.0
                
                ):
        super().__init__(encoder,decoder,categorizer,
                reconFactor, kldFactor, classFactor,
                kl_warmup_steps,class_warmup_steps,free_bits,noise_warmup,KLD_MAX,beta_start,
                multiple_optimizers, reductionType,kld_mode,categoryWeights,pretrained,
                latent_dim
                )
        self.cycleFactor = cycleFactor

    def configure_optimizers(self):
        params = [{'params':self.LatentTransition.parameters(),'lr':1e-3},{'params':self.cond_embedder.parameters(),'lr':1e-3},{'params':self.categorizer.parameters(),'lr':1e-3}
                ,{'params':self.encoder.parameters(),'lr':1e-4},{'params':self.decoder.parameters(),'lr':1e-4}]
        return torch.optim.Adam(params,lr = 1e-3)
    
    def shared_step(self, batch, mode="Train"):
        Xs, x_in,target_gene, state, input_state = batch
        with torch.no_grad():
            mu, logvar = self.encoder(x_in)
            logvar = torch.clamp(logvar, min=-6.0, max=2.0)
            z = self.reparameterize(mu, logvar)

        cond = self.cond_embedder(target_gene)
        z_perturbed = z + self.LatentTransition(z, cond)
        
        Xs_hat = self.decoder(z_perturbed)
        loss_recon = nn.functional.mse_loss(Xs_hat, Xs, reduction=self.reduction)
        loss_kld = self.compute_kl_loss(mu, logvar)
        #loss_kld = torch.clamp(loss_kld, max=self.KLD_MAX)
        beta = self.kl_weight()

        # Cycle consistenc
        revese_cond = target_gene*0
        revese_cond[-1,-1] =1  # The last index was added as padding condition anyway, now it serves as no-perturbation
        # Target gene is in ohe encoding, so reverse by subtracting 1 and multiplying by -1 is it the best way?
        cond_reverse = self.cond_embedder(revese_cond)
        z_reconstructed = z_perturbed - self.LatentTransition(z_perturbed, cond_reverse)
        Xs_reconstructed = self.decoder(z_reconstructed)
        loss_cycle = nn.functional.mse_loss(Xs_reconstructed, x_in, reduction=self.reduction)

        if self.categorizer is not None:
            class_logits = self.categorizer(z_perturbed)
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
        total_loss = (self.reconFactor * loss_recon)+ (classWeight * loss_class) + (self.cycleFactor * loss_cycle)

        # Logging
        self.logging_step(loss_recon, loss_kld, loss_cycle, loss_class, total_loss , class_logits, state_indices, mode=mode)

    def logging_step(self,loss_recon,loss_kld,loss_cycle,loss_class,total_loss,class_logits,state_indices,mode="Train"):
        self.log(f"{mode}/loss_recon", loss_recon, prog_bar=True,sync_dist=True)
        self.log(f"{mode}/loss_kld", loss_kld, prog_bar=True,sync_dist=True)
        self.log(f"{mode}/loss_class", loss_class, prog_bar=True,sync_dist=True)
        self.log(f"{mode}/loss_cycle", loss_cycle, prog_bar=True,sync_dist=True)
        self.log(f"{mode}/total_loss", total_loss, prog_bar=True,   sync_dist=True)
        
        if self.categorizer is not None:
            auc_scores = self.classwise_auc(class_logits, state_indices)           
            for aval, class_name in zip(auc_scores.values(),self.class_names): # Use your stored class names
                    self.log(f"{mode}/AUROC_{class_name}", aval, prog_bar=False, sync_dist=True)

        if mode == "Val" and self.categorizer is not None:
            self.confmat.update(class_logits, state_indices)