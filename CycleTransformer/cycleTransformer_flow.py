import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np  
import torchmetrics
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torchmetrics
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.functional import pearson_corrcoef
import torch.nn.functional as F
import umap
import pandas as pd
from CycleTransformer.utils import CenterLoss, mmd_loss

from CycleTransformer.modules import TransformerCycleEncoder, TransformerCycleDecoder, DeltaTransition,  LatentClassifier, SemanticDiscriminator, VectorField


class CycleTransformer(pl.LightningModule):
    def __init__(self, input_dim=274, d_model=256, n_layers=4, n_heads=8, z_dim=256
                ,reconFactr = 1., lossLatentFactr = 0.5, lossCycleFactr = 10., lossRegFactr = 1e-4
                , lossClassFactr=2.5, centerFactr=0.5, useVarLoss:bool = False,semanticCycleFactr = 2.0,
                adversarialFactr = 5.0, mmdFactr = 100.0,
                classes = ['G1', 'G2M', 'S'],class_weights = [1.0/96969, 1.0/43942, 1.0/61392],
                phase:int = 2,num_classes:int = 3, n_real_genes:int=8749, seq_len:int=32,
                cond_dim = 256, 
                use_hvg_mask:bool = False # This is new
                ):
        super().__init__()
        
        self.phase = phase
        #Actual Training and hyperparameters        
        ##Loss Factors
        total_input_dim = input_dim*seq_len
        self.reconFactr = reconFactr
        self.lossLatentFactr = lossLatentFactr
        self.lossCycleFactr = lossCycleFactr
        self.lossRegFactr = lossRegFactr
        self.LossClassFactr = lossClassFactr
        self.centerFactr = centerFactr
        self.semanticCycleFactr = semanticCycleFactr
        self.adversarialFactr = adversarialFactr
        self.mmdFactr = mmdFactr

        #Model parameters
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.useVarLoss = useVarLoss

        #Model Components
        self.encoder = TransformerCycleEncoder(input_dim, d_model, n_layers, n_heads, z_dim)
        self.decoder = TransformerCycleDecoder(input_dim, d_model, n_layers, n_heads, z_dim)
        self.transition_fwd = DeltaTransition(z_dim)
        self.transition_bwd = DeltaTransition(z_dim)
        self.latentClassifier = LatentClassifier(z_dim, num_classes)
        self.discriminator = SemanticDiscriminator(z_dim)
        #self.vectorField = VectorField(self.z_dim,hidden_dim=1024)


        mask = torch.zeros(total_input_dim)
        mask[:n_real_genes] = 1.0
        mask = mask.view(1, seq_len, input_dim) 
        self.register_buffer('gene_mask', mask)
        # NEW, extra HVG mask for GEARS
        if use_hvg_mask:
            hvg_mask = np.load("data/replogle_gears/hvg_mask.npy")
            padding = total_input_dim - len(hvg_mask)
            hvg_mask = np.concatenate([hvg_mask, np.zeros(padding)])
            hvg_mask_tensor = torch.tensor(hvg_mask, dtype=torch.float32).view(1, seq_len, input_dim)
            self.gene_mask = self.gene_mask * hvg_mask_tensor
        #Traning phase settings
        if phase == 1:
            self.automatic_optimization = False
            self.shared_step = self.phase_one_shared_step
            for param in self.transition_fwd.parameters():
                param.requires_grad = False
            for param in self.transition_bwd.parameters():
                param.requires_grad = False
            for param in self.discriminator.parameters():
                param.requires_grad = False
            self.transition_fwd.eval()
            self.transition_bwd.eval()
            self.discriminator.eval()


        
        #Losses
        self.criterion = nn.MSELoss()
        weights = torch.tensor(class_weights)
        weights = weights / weights.sum()
        self.latent_criterion = nn.CrossEntropyLoss(weight=weights)
        self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=z_dim)
        #self.varianceLoss = VarianceLoss(percentage=0.5)
        # Checking metrics and classifiers
    
        self.aucMetric = torchmetrics.AUROC(num_classes=num_classes, average=None,task="multiclass")
        self.confmat = MulticlassConfusionMatrix(num_classes=num_classes)
        self.confmat2 = MulticlassConfusionMatrix(num_classes=num_classes)
        self.class_names = classes
        self.classwise_auc = ClasswiseWrapper(self.aucMetric,labels=classes)
        self.save_hyperparameters()

    def init_phase2(self,lossCycleFactr=None, semanticCycleFactr=None, adversarialFactr=None, mmdFactr=None,vectorField_hidden_dim=512):
        for param in self.encoder.parameters():
                param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.latentClassifier.parameters():
            param.requires_grad = False
        for param in self.center_loss.parameters():
            param.requires_grad = False
        for param in self.transition_fwd.parameters():
            param.requires_grad = False
        for param in self.transition_bwd.parameters():
            param.requires_grad = False
        for param in self.discriminator.parameters():
            param.requires_grad = True
        if lossCycleFactr is not None:
            self.lossCycleFactr = lossCycleFactr
        if semanticCycleFactr is not None:
            self.semanticCycleFactr = semanticCycleFactr
        if adversarialFactr is not None:
            self.adversarialFactr = adversarialFactr
        if mmdFactr is not None:
            self.mmdFactr = mmdFactr
        self.shared_step = self.shared_step_cycle_gan
        self.configure_optimizers = self.configure_optimizers_cycle_gan
        self.encoder.eval()
        self.decoder.eval()
        self.latentClassifier.eval()

        self.transition_fwd.train()
        self.transition_bwd.train()
        self.discriminator.train()
        self.automatic_optimization = False
        self.phase = 2
        self.mmd_loss = mmd_loss
        #self.pearsonLoss = PearsonLoss()
        #self.topk_criterion = TopKDELoss()
        self.vectorField = VectorField(self.z_dim,hidden_dim=vectorField_hidden_dim)

    def configure_optimizers_cycle_gan(self):
        gen_params = list(self.vectorField.parameters())
        opt_g = torch.optim.AdamW(gen_params, lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
        
        # Discriminator
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
        
        # Return list of optimizers
        return [opt_g, opt_d]

    
    def add_instance_noise(self, data, std=0.1):
        if self.current_epoch > 50: return data # Decay noise later
        noise = torch.randn_like(data) * std
        return data + noise
    

    def get_perturbation_prompt(self, x_ctrl, pert_indices,normalized=True):
        """
        Creates a 'Prompt Vector' that represents the direction of the perturbation.
        """
        # 1. Clone the input so we don't break the original
        B, num_patches, patch_dim = x_ctrl.shape
        total_genes = num_patches * patch_dim
        one_hot = torch.zeros(B, total_genes, device=self.device)
        one_hot.scatter_(1, pert_indices.unsqueeze(1), 1.0)
        one_hot_reshaped = one_hot.view(B, num_patches, patch_dim)
        silence_val = 0.0 if normalized else -5.0
        keep_mask = 1.0 - one_hot_reshaped
        x_silenced = (x_ctrl * keep_mask) + (one_hot_reshaped * silence_val)
        with torch.no_grad():
            z_ctrl = self.encoder(x_ctrl)
            z_silenced = self.encoder(x_silenced)
        z_prompt = z_silenced - z_ctrl
        
        return z_prompt
        
    
    def shared_step_cycle_gan(self, batch, optimizer_idx=0, mode='train'):
        x_pert, x_ctrl, pert_idx, pert_state, ctrl_state = batch
        
        with torch.no_grad():
            z_ctrl = self.encoder(x_ctrl)       # Real Control
            z_real_pert = self.encoder(x_pert)  # Real Perturbed

        # Generation step
        if optimizer_idx == 0:
            lossDict = {}
            # 1. Forward: Add perturbation
            z_prompt_fwd = self.get_perturbation_prompt(x_ctrl, pert_idx)
            z_prompt_bwd = z_prompt_fwd
            t = torch.rand(z_ctrl.shape[0], 1, device=self.device)
            z_t_target = (1 - t) * z_ctrl + t * z_real_pert
            target_velocity = z_real_pert - z_ctrl
            pred_velocity = self.vectorField(z_t_target, t, z_prompt_fwd)
            loss_fm = F.mse_loss(pred_velocity, target_velocity)
            lossDict['loss_fm'] = loss_fm
            steps = 8
        
            # Forward: Ctrl -> Fake Pert (t=0 -> t=1)
            z_fake_pert = self.ode_solve(z_ctrl, z_prompt_fwd, t_start=0, t_end=1, steps=steps)
            z_rec_ctrl = self.ode_solve(z_fake_pert, z_prompt_bwd, t_start=1, t_end=0, steps=steps)
            z_fake_ctrl = self.ode_solve(z_real_pert, z_prompt_bwd, t_start=1, t_end=0, steps=steps)
            loss_cycle_ctrl = F.mse_loss(z_rec_ctrl, z_ctrl)
            
            # Reconstruction: Fake Ctrl -> Rec Pert (t=0 -> t=1)
            z_rec_pert = self.ode_solve(z_fake_ctrl, z_prompt_bwd, t_start=0, t_end=1, steps=steps)
            loss_cycle_pert = F.mse_loss(z_rec_pert, z_real_pert)
            
            # Loss: Cycle Consistency
            lossDict['loss_cycle_ctrl'] = loss_cycle_ctrl
            lossDict['loss_cycle_pert'] = loss_cycle_pert

            x_fake_pert = self.decoder(z_fake_pert)

            # Check if we can fool the discriminator 
            logits_fake = self.discriminator(z_fake_pert, z_prompt_fwd)
            loss_adv = F.mse_loss(logits_fake, torch.ones_like(logits_fake))
            lossDict['loss_adv_g'] = loss_adv

            # Classification to further regularize latent space            
            logits_rec_ctrl = self.latentClassifier(z_rec_ctrl)
            loss_sem_ctrl = self.latent_criterion(logits_rec_ctrl, ctrl_state)
            lossDict['loss_sem_ctrl'] = loss_sem_ctrl
            logits_rec_pert = self.latentClassifier(z_rec_pert)
            loss_sem_pert = self.latent_criterion(logits_rec_pert, pert_state)
            lossDict['loss_sem_pert'] = loss_sem_pert
            loss_semantic_cycle = loss_sem_ctrl + loss_sem_pert

            loss_pearson = self.pearsonLoss(x_fake_pert, x_pert)
            lossDict['loss_pearson'] = loss_pearson
            

            loss_top20 = self.topk_criterion(x_fake_pert, x_pert, x_ctrl)
            lossDict['loss_top20'] = loss_top20
            
            g_loss = (self.lossCycleFactr * (loss_cycle_ctrl + loss_cycle_pert) + 
                      self.semanticCycleFactr * loss_semantic_cycle + 
                      self.adversarialFactr * loss_adv +  
                        20*loss_fm)

            lossDict["g_total_loss"] = g_loss
            self.logging_metrics(mode, lossDict)
            
            # Log AUC only during Generator step
            with torch.no_grad():
                logits_real = self.latentClassifier(z_real_pert)                
                self.log_collapse_metrics2(mode,z_ctrl,z_fake_pert,z_real_pert)
                x_recon_pert = self.decoder(z_fake_pert)
                x_recon_ctrl = self.decoder(z_fake_ctrl)
                
                if mode == "val":
                    self.logAUROC(mode, logits_rec_pert, logits_real, pert_state)
                    self.log_reconstruction(mode, x_pert, x_ctrl, x_recon_pert, x_recon_ctrl)
                    self.log_pearson_metrics(mode, x_pert, x_ctrl, pert_idx)
                    self.log_nmse_top_genes(mode, x_pert, x_ctrl, pert_idx, top_n=20)
                    self.log_direction_error(mode, x_pert, x_ctrl, pert_idx, top_n=20)

            return g_loss

        #Discriminator
        if optimizer_idx == 1:        
            # 1. Real Data
            z_prompt = self.get_perturbation_prompt(x_ctrl, pert_idx)
            logits_real = self.discriminator(self.add_instance_noise(z_real_pert), z_prompt)
            loss_real = F.mse_loss(logits_real, torch.ones_like(logits_real)*0.9)
            
            # 2. Fake Data
            with torch.no_grad():
                z_fake_pert = self.ode_solve(z_ctrl, z_prompt, t_start=0, t_end=1, steps=4)
            
            logits_fake = self.discriminator(self.add_instance_noise(z_fake_pert.detach()), z_prompt)
            loss_fake = F.mse_loss(logits_fake, torch.zeros_like(logits_fake))
            
            # Total Discriminator Loss
            d_loss = 0.5 * (loss_real + loss_fake)
            
            self.logging_metrics(mode, {'d_loss': d_loss, 'd_real': logits_real.mean(), 'd_fake': logits_fake.mean()})
            return d_loss

    def phase_one_shared_step(self,batch,mode):
        x_pert, x_ctrl, pert_idx, pert_state, ctrl_state = batch
        if pert_state.dim() > 1 and pert_state.size(1) > 1:
            pert_state = torch.argmax(pert_state, dim=1)
        else:
            pert_state = pert_state.long()
        loss_dict = {}
        z_ctrl = self.encoder(x_ctrl)
        z_real_pert = self.encoder(x_pert)
        
        x_recon_ctrl = self.decoder(z_ctrl)
        x_recon_pert = self.decoder(z_real_pert)

        logits_pert = self.latentClassifier(z_real_pert)
        loss_class = self.latent_criterion(logits_pert, pert_state)
        loss_dict['loss_class'] = loss_class

        logits_pert_detached = self.latentClassifier(z_real_pert.detach())
        loss_class_critic = self.latent_criterion(logits_pert_detached, pert_state)
        loss_dict['loss_class_critic'] = loss_class_critic

        #loss_recon = self.criterion(x_recon_ctrl, x_ctrl) + self.criterion(x_recon_pert, x_pert)
        batch_size = x_ctrl.size(0)
        n_valid_elements = self.gene_mask.sum() * batch_size

        diff_ctrl = (x_recon_ctrl - x_ctrl) * self.gene_mask
        diff_pert = (x_recon_pert - x_pert) * self.gene_mask
        loss_recon_ctrl = (diff_ctrl ** 2).sum() / n_valid_elements
        loss_recon_pert = (diff_pert ** 2).sum() / n_valid_elements
        loss_recon = loss_recon_ctrl + loss_recon_pert
        
        loss_dict['loss_recon'] = loss_recon
        loss_reg = torch.mean(z_ctrl**2) + torch.mean(z_real_pert**2)
        loss_dict['loss_reg'] = loss_reg

        z_recon_pert = self.encoder(x_recon_pert)
        loss_consistency = self.criterion(z_recon_pert, z_real_pert.detach())
        loss_dict['loss_consistency'] = loss_consistency
        
        loss_labels = torch.argmax(pert_state, dim=1) if pert_state.dim() > 1 and pert_state.size(1) > 1 else pert_state.long()
        center_loss = self.center_loss(z_real_pert, loss_labels)
        loss_dict['center_loss'] = center_loss

        total_loss = (self.reconFactr * loss_recon + 
                      self.lossRegFactr * loss_reg +
                      self.LossClassFactr * loss_class +
                      self.lossLatentFactr * loss_consistency+
                      self.centerFactr * center_loss)
        
        if self.useVarLoss:
            var_loss = self.varianceLoss(x_pert, x_recon_pert)
            loss_dict['var_loss_pert'] = var_loss
            var_loss_ctr = self.varianceLoss(x_ctrl, x_recon_ctrl)
            loss_dict['var_loss_ctrl'] = var_loss_ctr
            var_loss = var_loss + var_loss_ctr
            total_loss += var_loss*5.

        loss_dict['total_loss'] = total_loss
        self.logging_metrics(mode, loss_dict)
        
        if mode == "val":
            self.logAUROC(mode, logits_pert, logits_pert, pert_state)
            self.log_variance_health(mode, x_pert, x_recon_pert)
        return total_loss, loss_class_critic


    def log_reconstruction(self,mode, x_pert, x_ctrl, x_recon_pert, x_recon_ctrl):
        batch_size = x_ctrl.size(0)
        n_valid_elements = self.gene_mask.sum() * batch_size

        diff_ctrl = (x_recon_ctrl - x_ctrl) * self.gene_mask
        diff_pert = (x_recon_pert - x_pert) * self.gene_mask
        loss_recon_ctrl = (diff_ctrl ** 2).sum() / n_valid_elements
        loss_recon_pert = (diff_pert ** 2).sum() / n_valid_elements
        loss_recon = loss_recon_ctrl + loss_recon_pert
        self.log(f"{mode}/Reconstruction_Loss", loss_recon, sync_dist=True,prog_bar=False)
        self.log(f"{mode}/Reconstruction_Pert_Loss", loss_recon_pert, sync_dist=True,prog_bar=False)
        self.log(f"{mode}/Reconstruction_Ctrl_Loss", loss_recon_ctrl, sync_dist=True,prog_bar=False)

        self.log_variance_health(mode, x_pert, x_recon_pert,postfix="_pert")
        self.log_variance_health(mode, x_ctrl, x_recon_ctrl,postfix="_ctrl")


    def log_nmse_top_genes(self, mode, x_pert, x_ctrl, pert_idx, top_n=20):
        """
        Calculates Normalized Mean Squared Error (NMSE) on the Top N 
        Differentially Expressed (DE) genes.
        """
        # 1. Generate Prediction
        with torch.no_grad():
            z_ctrl = self.encoder(x_ctrl)
            z_prompt = self.get_perturbation_prompt(x_ctrl, pert_idx)
            delta_pred = self.transition_fwd(z_ctrl, z_prompt)
            z_fake_pert = z_ctrl + delta_pred
            x_recon_pert = self.decoder(z_fake_pert)

        # 2. Calculate Deltas & Flatten [Seq, Dim] -> [Total_Genes]
        real_delta = (x_pert - x_ctrl).mean(dim=0).flatten()
        pred_delta = (x_recon_pert - x_ctrl).mean(dim=0).flatten()

        # 3. Identify Top N Movers
        top_vals, top_indices = torch.topk(torch.abs(real_delta), k=top_n)

        # 4. Extract values for these specific genes
        r_top = real_delta[top_indices]
        p_top = pred_delta[top_indices]

        # 5. Calculate MSE (The Error)
        mse = torch.mean((r_top - p_top) ** 2)

        # 6. Calculate Normalizer (The Magnitude of the real perturbation)
        # This represents the error if the model just predicted 0 (Identity)
        normalizer = torch.mean(r_top ** 2)

        # 7. Calculate NMSE
        # Add epsilon to prevent div by zero
        nmse = mse / (normalizer + 1e-8)

        # 8. Log it
        self.log(f"{mode}/NMSE_Top{top_n}", nmse, prog_bar=False, sync_dist=True)
        
        return nmse


    def log_pearson_metrics_old(self, mode, x_pert, x_ctrl, pert_idx):
        """
        Calculates and logs Pearson Correlation for:
        1. All Genes (Global Trend)
        2. Top 5% 'Mover' Genes (The most important active genes)
        """
        with torch.no_grad():
            z_ctrl = self.encoder(x_ctrl)
            z_prompt = self.get_perturbation_prompt(x_ctrl, pert_idx)
            # Predict the shift
            delta_pred = self.transition_fwd(z_ctrl, z_prompt)
            z_fake_pert = z_ctrl + delta_pred
            x_recon_pert = self.decoder(z_fake_pert)

        real_delta_vector = (x_pert - x_ctrl).mean(dim=0).flatten()
        pred_delta_vector = (x_recon_pert - x_ctrl).mean(dim=0).flatten()

        
        threshold = torch.quantile(torch.abs(real_delta_vector), 0.95)
        # Create mask for genes that moved significantly
        mask = torch.abs(real_delta_vector) > threshold
        if mask.sum() > 2:
            real_top = real_delta_vector[mask]
            pred_top = pred_delta_vector[mask]
            
            pearson_all = pearson_corrcoef(pred_delta_vector, real_delta_vector)
            self.log(f"{mode}/Pearson_All", pearson_all, prog_bar=False, sync_dist=True)
            
            pearson_top5 = pearson_corrcoef(pred_top, real_top)
            self.log(f"{mode}/Pearson_Top5_Percent", pearson_top5, prog_bar=False, sync_dist=True)
        else:
            # If the perturbation was silent (nothing moved), we skip this metric for this batch
            pass

    def log_pearson_metrics(self, mode, x_pert, x_ctrl, pert_idx):
        """
        Calculates Pearson Correlation per UNIQUE perturbation in the batch.
        Prevents "muddying" the signal by averaging different gene knockouts together.
        """
        with torch.no_grad():
            z_ctrl = self.encoder(x_ctrl)
            z_prompt = self.get_perturbation_prompt(x_ctrl, pert_idx)
            delta_pred = self.transition_fwd(z_ctrl, z_prompt)
            z_fake_pert = z_ctrl + delta_pred
            x_recon_pert = self.decoder(z_fake_pert)

        # 1. Identify unique perturbations in this batch
        unique_perts = torch.unique(pert_idx)
        
        # Lists to store scores for this batch
        batch_pearson_all = []
        batch_pearson_top5 = []

        # 2. Iterate through each unique perturbation
        for p_id in unique_perts:
            # Create a boolean mask for this specific perturbation
            # This selects only the rows (cells) that belong to p_id
            mask = (pert_idx == p_id)
            
            # Safety check: ensure we have at least 1 cell (though unique guarantees it)
            if mask.sum() == 0: continue

            # 3. Calculate Pseudo-bulk Vectors (Mean of this specific group)
            # Note: We subtract the control mean of the SAME group subset
            # to ensure we are comparing the shift relative to the paired controls
            real_delta_vector = (x_pert[mask] - x_ctrl[mask]).mean(dim=0).flatten()
            pred_delta_vector = (x_recon_pert[mask] - x_ctrl[mask]).mean(dim=0).flatten()
            
            # Safety: Check for flat vectors (std=0) to avoid NaNs
            if torch.std(real_delta_vector) < 1e-6 or torch.std(pred_delta_vector) < 1e-6:
                continue

            # --- Metric 1: Global Pearson ---
            p_all = pearson_corrcoef(pred_delta_vector, real_delta_vector)
            batch_pearson_all.append(p_all)

            # --- Metric 2: Top 5% Pearson ---
            # Calculate threshold on the REAL biological signal
            threshold = torch.quantile(torch.abs(real_delta_vector), 0.95)
            top_mask = torch.abs(real_delta_vector) > threshold
            
            # Only calculate if there are enough moving genes (e.g., >2)
            if top_mask.sum() > 2:
                real_top = real_delta_vector[top_mask]
                pred_top = pred_delta_vector[top_mask]
                
                p_top5 = pearson_corrcoef(pred_top, real_top)
                batch_pearson_top5.append(p_top5)

        # 4. Aggregate and Log (Average of the scores in this batch)
        if len(batch_pearson_all) > 0:
            avg_pearson_all = torch.stack(batch_pearson_all).mean()
            self.log(f"{mode}/Pearson_All", avg_pearson_all, prog_bar=False, sync_dist=True)
        
        if len(batch_pearson_top5) > 0:
            avg_pearson_top5 = torch.stack(batch_pearson_top5).mean()
            self.log(f"{mode}/Pearson_Top5_Percent", avg_pearson_top5, prog_bar=False, sync_dist=True)


    def log_direction_error(self, mode, x_pert, x_ctrl, pert_idx, top_n=20):
        """
        Calculates Direction Error per UNIQUE perturbation in the batch.
        Prevents averaging different perturbations together.
        """
        # 1. Generate Prediction (Batch-wise for speed)
        with torch.no_grad():
            z_ctrl = self.encoder(x_ctrl)
            z_prompt = self.get_perturbation_prompt(x_ctrl, pert_idx)
            delta_pred = self.transition_fwd(z_ctrl, z_prompt)
            z_fake_pert = z_ctrl + delta_pred
            x_recon_pert = self.decoder(z_fake_pert)

        # 2. Identify unique perturbations
        unique_perts = torch.unique(pert_idx)
        batch_errors = []

        # 3. Iterate through each perturbation group
        for p_id in unique_perts:
            mask = (pert_idx == p_id)
            
            # Safety: Ensure we have cells for this perturbation
            if mask.sum() == 0: continue

            # 4. Calculate Pseudo-bulk Vectors for THIS perturbation only
            # We average the cells belonging to p_id
            real_delta = (x_pert[mask] - x_ctrl[mask]).mean(dim=0).flatten()
            pred_delta = (x_recon_pert[mask] - x_ctrl[mask]).mean(dim=0).flatten()

            # 5. Identify Top N Movers (Standard Logic)
            # Find genes with largest ABSOLUTE real change
            top_vals, top_indices = torch.topk(torch.abs(real_delta), k=top_n)

            # 6. Check Signs on those specific genes
            real_signs = torch.sign(real_delta[top_indices])
            pred_signs = torch.sign(pred_delta[top_indices])

            # 7. Calculate Error
            # Product is negative if signs are opposite (e.g. 1 * -1 = -1)
            sign_product = real_signs * pred_signs
            n_opposite = (sign_product < 0).float().sum()
            percent_opposite = n_opposite / float(top_n)
            
            batch_errors.append(percent_opposite)

        # 8. Aggregate and Log
        if len(batch_errors) > 0:
            # Average the error across all perturbations in this batch
            avg_error = torch.stack(batch_errors).mean()
            
            self.log(f"{mode}/Direction_Error_Top{top_n}", avg_error * 100.0, 
                    prog_bar=False, sync_dist=True)
            
            return avg_error.item()
        else:
            return 0.0



    def logAUROC(self, mode, latentLogits, logits_real, pert_state):
        if pert_state.shape == latentLogits.shape:
            state_indices = torch.argmax(pert_state, dim=1)
        elif pert_state.ndim == 2 and pert_state.shape[1] == 1:
            state_indices = pert_state.squeeze(1)
        else:
            state_indices = pert_state
        self.classwise_auc.update(latentLogits, state_indices)
        

        if mode == "val" and self.current_epoch % 5 == 0 and not self.trainer.sanity_checking:
            self.confmat.update(latentLogits, state_indices)
            self.confmat2.update(logits_real, state_indices)

    def logging_metrics(self, mode, lossDict):
        for key, value in lossDict.items():
            self.log(f'{mode}/{key}', value, prog_bar=True,sync_dist=True)
        
    def training_step(self, batch, batch_idx):
        if self.phase == 1:
            loss, classLoss = self.phase_one_shared_step(batch,mode="Train")
            main_opt, classifier_opt = self.optimizers()
            main_opt.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(main_opt,gradient_clip_val=0.5,gradient_clip_algorithm="norm")
            main_opt.step()
            classifier_opt.zero_grad()
            self.manual_backward(classLoss)
            classifier_opt.step()
            return loss
        
        else:
            opt_g, opt_d = self.optimizers()
            
            g_loss = self.shared_step_cycle_gan(batch, optimizer_idx=0, mode="Train")
            
            opt_g.zero_grad()
            self.manual_backward(g_loss)
            self.clip_gradients(opt_g, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opt_g.step()
            
            if batch_idx % 5 == 0:
                d_loss = self.shared_step_cycle_gan(batch, optimizer_idx=1, mode="Train")
                
                opt_d.zero_grad()
                self.manual_backward(d_loss)
                opt_d.step()
            return g_loss

    def validation_step(self, batch, batch_idx):
        if self.phase == 1:
            if batch_idx == 0:
                self.classwise_auc.reset()
            loss,class_loss = self.phase_one_shared_step(batch,mode="val")
            return loss + class_loss
        else:
            g_loss = self.shared_step_cycle_gan(batch, optimizer_idx=0, mode='val')
            d_loss = self.shared_step_cycle_gan(batch, optimizer_idx=1, mode='val')
            return g_loss + d_loss
        

    def configure_optimizers(self):
        classifier_params = list(self.latentClassifier.parameters()) 
        classifier_ids = list(map(id, classifier_params))

        main_params = [p for p in self.parameters() if id(p) not in classifier_ids]
        optimizer_main = torch.optim.AdamW(main_params, lr=1e-4, weight_decay=1e-3)
        optimizer_probe = torch.optim.AdamW(classifier_params, lr=1e-3, weight_decay=0.0)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_main, mode='min', factor=0.5, patience=5)
        #scheduler = LambdaLR(optimizer_main, lr_lambda=warmup_cosine_schedule)
        return [optimizer_main, optimizer_probe]


    def log_collapse_metrics2(self, mode, z_ctrl, z_fake_pert, z_real_pert):
        # Measure how far the model actually moved the points
        delta = z_fake_pert - z_ctrl
        delta_mag = torch.norm(delta, p=2, dim=1).mean()
        
        # Compare to how big the latent vectors are naturally
        z_mag = torch.norm(z_ctrl, p=2, dim=1).mean()
        movement_ratio = delta_mag / (z_mag + 1e-8)
        
        self.log(f"{mode}/Health_Delta_Mag", delta_mag, sync_dist=True)
        self.log(f"{mode}/Health_Movement_Ratio", movement_ratio, sync_dist=True)

        
        # Check the standard deviation of the OUTPUT batch.
        fake_std = z_fake_pert.std(dim=0).mean() 
        real_std = z_real_pert.std(dim=0).mean()
        
        # We want the Fake Diversity to match Real Diversity (Ratio ~ 1.0)
        diversity_ratio = fake_std / (real_std + 1e-8)
        
        self.log(f"{mode}/Health_Diversity_Real", real_std, sync_dist=True)
        self.log(f"{mode}/Health_Diversity_Fake", fake_std, sync_dist=True)
        self.log(f"{mode}/Health_Diversity_Ratio", diversity_ratio, sync_dist=True)

        delta_std = delta.std(dim=0).mean()
        self.log(f"{mode}/Health_Delta_Diversity", delta_std, sync_dist=True)


    def log_variance_health(self, mode, x_real, x_recon,postfix=""):
        """
        Checks if the model is capturing the dynamic range (variance) of the data
        or just predicting the mean (which has 0 variance).
        """
        # 1. Flatten Batch and Patch dimensions to get [Total_Cells, Genes]
        # We want variance across the entire batch/population
        # Assuming input is [Batch, Patches, Genes] -> [Batch*Patches, Genes]
        if x_real.dim() == 3:
            real_flat = x_real.reshape(-1, x_real.shape[-1])
            recon_flat = x_recon.reshape(-1, x_recon.shape[-1])
        else:
            real_flat = x_real
            recon_flat = x_recon

        # 2. Calculate Variance per Gene (dim=0)
        var_real = torch.var(real_flat, dim=0)
        var_recon = torch.var(recon_flat, dim=0)

        # 3. Identify the Top 50 Most Variable Genes in REAL data
        # (These are the ones that matter: Cell Cycle genes, CRISPR targets)
        top_vals, top_indices = torch.topk(var_real, k=50)

        # 4. Compare Variances ONLY on these Top 50 genes
        var_real_top = var_real[top_indices]
        var_recon_top = var_recon[top_indices]

        # 5. The Metric: Ratio of Reconstruction Variance to Real Variance
        # Avoid div by zero with 1e-8
        variance_preservation = var_recon_top.mean() / (var_real_top.mean() + 1e-8)

        # 6. Log it
        self.log(f"{mode}/Health_Var_Preservation{postfix}", variance_preservation, 
                prog_bar=False, sync_dist=True)
        
        # Optional: Log the Raw Variance numbers to see scale
        self.log(f"{mode}/Debug_Var_Real_Mean{postfix}", var_real_top.mean(), sync_dist=True)
        self.log(f"{mode}/Debug_Var_Recon_Mean{postfix}", var_recon_top.mean(), sync_dist=True)

        return variance_preservation


    def on_validation_epoch_end(self):
        if self.phase == 1:
            try:
                # 1. Compute
                auc_scores = self.classwise_auc.compute()
                
                # 2. Log (Use log_dict for efficiency)
                metrics_to_log = {f"Classification/val/AUROC_{name}": val for val, name in zip(auc_scores.values(), self.class_names) }
                self.log_dict(metrics_to_log, prog_bar=False, sync_dist=True)
                
            except Exception as e:
                print(f"WARNING: Metric computation failed on Rank {self.global_rank}: {e}")
                # This usually happens if a rank saw no data
                
            finally:
                self.classwise_auc.reset()


    def visualize_latent_space(self, title="Latent Space UMAP"):
        """
        Runs the full validation set through the Encoder and plots UMAP.
        """
        self.eval() # Set to evaluation mode
        all_z = []
        all_labels = []
        dataloader = self.trainer.val_dataloaders
        
        # Lightning might wrap it in a list, handle that:
        if isinstance(dataloader, list):
            dataloader = dataloader[0]
        # 1. Collect all latent vectors and labels
        with torch.no_grad():
            for batch in dataloader:
                # Unpack your batch (adjust based on your actual structure)
                x_pert, x_ctrl, pert_idx, pert_state, ctrl_state = batch
                
                # Move to device
                x_pert = x_pert.to(self.device)
                
                # Encode
                z = self.encoder(x_pert)
                
                # Handle Labels (Convert One-Hot to Index if needed)
                if pert_state.dim() > 1 and pert_state.size(1) > 1:
                    labels = torch.argmax(pert_state, dim=1)
                else:
                    labels = pert_state
                
                # Store in CPU list
                all_z.append(z.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate into one big array
        X = np.concatenate(all_z, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        # 2. Run UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(X)
        
        # 3. Plot using Seaborn
        plt.figure(figsize=(10, 8))
        
        # Create a DataFrame for easier plotting with names
        # Map indices 0,1,2,3 to names if you have them
        label_names = [self.class_names[i] for i in y]
        
        df = pd.DataFrame({
            'UMAP-1': embedding[:, 0],
            'UMAP-2': embedding[:, 1],
            'Class': label_names
        })
        
        sns.scatterplot(
            data=df, 
            x='UMAP-1', 
            y='UMAP-2', 
            hue='Class', 
            palette='viridis', 
            s=10, 
            alpha=0.7
        )
        
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 4. Save and Show
        save_path = f"imgs/umap_{title.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        #self.logger.experiment.log_artifact(
        #    run_id=self.logger.run_id, 
        #    local_path=save_path, 
        #    artifact_path="plots_latent")
    def ode_solve(self, z0, prompt, t_start=0.0, t_end=1.0, steps=4):
        """
        Differentiable Euler Solver. 
        Low steps (4) during training for speed. High steps (10+) during eval for precision.
        """
        dt = (t_end - t_start) / steps
        z_t = z0
        
        for i in range(steps):
            # Current time t
            t_val = t_start + i * dt
            t_tensor = torch.ones(z0.size(0), 1, device=self.device) * t_val
            
            # Predict velocity
            velocity = self.vectorField(z_t, t_tensor, prompt)
            
            # Update state (Euler step)
            z_t = z_t + velocity * dt
            
        return z_t