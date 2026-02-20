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
from CycleTransformer.utils import CenterLoss, mmd_loss, VarianceLoss

from CycleTransformer.modules import TransformerCycleEncoder, TransformerCycleDecoder, DeltaTransition,  LatentClassifier, SemanticDiscriminator


class CycleTransformer(pl.LightningModule):
    def __init__(self, input_dim=274, d_model=256, n_layers=4, n_heads=8, z_dim=256
                ,reconFactr = 1., lossLatentFactr = 0.5, lossCycleFactr = 10., lossRegFactr = 1e-4
                , lossClassFactr=2.5, centerFactr=0.5, useVarLoss:bool = False,semanticCycleFactr = 2.0,
                adversarialFactr = 5.0, mmdFactr = 100.0,
                classes = ['lipo', 'other', 'pre_adipo', 'adipo'],class_weights = [1.0/1200, 1.0/3000, 1.0/5100, 1.0/5500],
                phase:int = 2,num_classes:int = 4, n_real_genes:int=8749, seq_len:int=32,
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
        self.lineage_classifier = nn.Linear(z_dim, 3) # other, pre, adipo
        self.subtype_classifier = nn.Linear(z_dim, 1) # lipo
        self.discriminator = SemanticDiscriminator(z_dim)

        mask = torch.zeros(total_input_dim)
        mask[:n_real_genes] = 1.0
        mask = mask.view(1, seq_len, input_dim) 
        self.register_buffer('gene_mask', mask)
        if use_hvg_mask:
            hvg_mask = np.load("data/genes_to_predict.txt", dtype=str)
            padding = total_input_dim - len(hvg_mask)
            hvg_mask = np.concatenate([np.isin(self.gene_mask.squeeze().cpu().numpy(), hvg_mask), np.zeros(padding)])
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
        self.center_loss = CenterLoss(num_classes=3, feat_dim=z_dim)
        self.varianceLoss = VarianceLoss(percentage=0.5)
        # Checking metrics and classifiers
    
        self.aucMetric = torchmetrics.AUROC(num_classes=num_classes, average=None,task="multiclass")
        self.confmat = MulticlassConfusionMatrix(num_classes=num_classes)
        self.confmat2 = MulticlassConfusionMatrix(num_classes=num_classes)
        self.class_names = classes
        self.classwise_auc = ClasswiseWrapper(
            torchmetrics.AUROC(num_labels=4, average=None, task="multilabel"),
            labels=classes,
            prefix="Classification/val/AUROC_" # This automatically formats your output names!
        )
        self.save_hyperparameters()

    def init_phase2(self,lossCycleFactr=None, semanticCycleFactr=None, adversarialFactr=None, mmdFactr=None):
        for param in self.encoder.parameters():
                param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.latentClassifier.parameters():
            param.requires_grad = False
        for param in self.center_loss.parameters():
            param.requires_grad = False
        for param in self.transition_fwd.parameters():
            param.requires_grad = True
        for param in self.transition_bwd.parameters():
            param.requires_grad = True
        for param in self.discriminator.parameters():
            param.requires_grad = True
        
        for param in self.lineage_classifier.parameters():
            param.requires_grad = False
        for param in self.subtype_classifier.parameters():
            param.requires_grad = False
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
        self.lineage_classifier.eval()
        self.subtype_classifier.eval()

        self.transition_fwd.train()
        self.transition_bwd.train()
        self.discriminator.train()
        self.automatic_optimization = False
        self.phase = 2
        self.mmd_loss = mmd_loss


    def configure_optimizers_cycle_gan(self):
        gen_params = list(self.transition_fwd.parameters()) + list(self.transition_bwd.parameters())
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
            delta_fwd = self.transition_fwd(z_ctrl, z_prompt_fwd)
            z_fake_pert = z_ctrl + delta_fwd
            
            
            # Note: We use pert_indices as start state for backward

            delta_bwd_rec = self.transition_bwd(z_fake_pert, z_prompt_bwd)
            z_rec_ctrl = z_fake_pert + delta_bwd_rec
            
            # Loss: Cycle Consistency
            loss_cycle_ctrl = F.mse_loss(z_rec_ctrl, z_ctrl)
            lossDict['loss_cycle_ctrl'] = loss_cycle_ctrl

            # Generating fake control latent states
            delta_bwd_real = self.transition_bwd(z_real_pert, z_prompt_bwd)
            z_fake_ctrl = z_real_pert + delta_bwd_real
            
            # Check if we can reconstruct real perturbed data
            delta_fwd_rec = self.transition_fwd(z_fake_ctrl, z_prompt_fwd)
            z_rec_pert = z_fake_ctrl + delta_fwd_rec
            
            # Loss: Cycle Consistency
            loss_cycle_pert = F.mse_loss(z_rec_pert, z_real_pert)
            lossDict['loss_cycle_pert'] = loss_cycle_pert
            
            # Chek if we can fool the discriminator 
            logits_fake = self.discriminator(z_fake_pert, z_prompt_fwd)
            loss_adv = F.mse_loss(logits_fake, torch.ones_like(logits_fake))
            lossDict['loss_adv_g'] = loss_adv

            # Classification to further regularize latent space            
            logits_rec_ctrl = self.latentClassifier(z_rec_ctrl)
            loss_sem_ctrl = self.compute_hierarchical_semantic_loss(z_rec_ctrl, ctrl_state)
            lossDict['loss_sem_ctrl'] = loss_sem_ctrl
            
            loss_sem_pert = self.compute_hierarchical_semantic_loss(z_rec_pert, pert_state)
            lossDict['loss_sem_pert'] = loss_sem_pert
            
            loss_semantic_cycle = loss_sem_ctrl + loss_sem_pert

            
            # 2. MMD (Distribution Matching - Optional but recommended for stability)
            loss_mmd = 0.0
            pert_labels = torch.argmax(pert_state, dim=1)
            unique_labels = torch.unique(pert_labels)
            for label in unique_labels:
                mask = (pert_labels == label)
                if mask.sum() > 1:
                    loss_mmd += self.mmd_loss(z_fake_pert[mask], z_real_pert[mask])
            if mask.sum() > 1:
                    loss_mmd += self.mmd_loss(z_fake_pert[mask], z_real_pert[mask])
            lossDict['loss_mmd'] = loss_mmd

            logits_lineage = self.lineage_classifier(z_rec_pert)
            logits_subtype = self.subtype_classifier(z_rec_pert)
            loss_lineage = F.cross_entropy(logits_lineage, lineage_targets)
            adipo_mask = (lineage_targets == 2) # assuming index 2 is adipo
            if adipo_mask.sum() > 0:
                loss_subtype = F.binary_cross_entropy_with_logits(
                    logits_subtype[adipo_mask], 
                    lipo_targets[adipo_mask].float()
                )
            else:
                loss_subtype = 0.0
            
            g_loss = (self.lossCycleFactr * (loss_cycle_ctrl + loss_cycle_pert) + 
                      self.semanticCycleFactr * loss_semantic_cycle + 
                      self.adversarialFactr * loss_adv +   # GAN loss usually has lower weight in CycleGAN
                      self.mmdFactr * loss_mmd)
            lossDict["g_total_loss"] = g_loss
            self.logging_metrics(mode, lossDict)
            
            # Log AUC only during Generator step
            with torch.no_grad():
                logits_lin_rec = self.lineage_classifier(z_rec_pert)
                logits_sub_rec = self.subtype_classifier(z_rec_pert)
                combined_probs_rec = self.get_hierarchical_probs(logits_lin_rec, logits_sub_rec)
                
                self.log_collapse_metrics2(mode, z_ctrl, z_fake_pert, z_real_pert)
                x_recon_pert = self.decoder(z_fake_pert)
                x_recon_ctrl = self.decoder(z_fake_ctrl)
                
                if mode == "val":
                    # Pass the combined probabilities to our updated logAUROC
                    self.logAUROC(mode, combined_probs_rec, pert_state)
                    
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
                delta_fwd = self.transition_fwd(z_ctrl, z_prompt)
                z_fake_pert = z_ctrl + delta_fwd
            
            logits_fake = self.discriminator(self.add_instance_noise(z_fake_pert.detach()), z_prompt)
            loss_fake = F.mse_loss(logits_fake, torch.zeros_like(logits_fake))
            
            # Total Discriminator Loss
            d_loss = 0.5 * (loss_real + loss_fake)
            
            self.logging_metrics(mode, {'d_loss': d_loss, 'd_real': logits_real.mean(), 'd_fake': logits_fake.mean()})
            return d_loss

    def phase_one_shared_step(self, batch, mode):
        x_pert, x_ctrl, pert_idx, pert_state, ctrl_state = batch
        
        # 1. Parse Original Labels
        # Assuming original self.class_names = ['lipo', 'other', 'pre_adipo', 'adipo'] (indices 0, 1, 2, 3)
        if pert_state.dim() > 1 and pert_state.size(1) > 1:
            original_labels = torch.argmax(pert_state, dim=1)
        else:
            original_labels = pert_state.long()
            
        loss_dict = {}
        z_ctrl = self.encoder(x_ctrl)
        z_real_pert = self.encoder(x_pert)
        x_recon_ctrl = self.decoder(z_ctrl)
        x_recon_pert = self.decoder(z_real_pert)

        # --- HIERARCHICAL CLASSIFICATION START ---
        
        # A. Map to Lineage Targets (0: other, 1: pre_adipo, 2: adipo_lineage)
        lineage_targets = torch.zeros_like(original_labels)
        lineage_targets[original_labels == 1] = 0 # other
        lineage_targets[original_labels == 2] = 1 # pre_adipo
        lineage_targets[original_labels == 3] = 2 # adipo
        lineage_targets[original_labels == 0] = 2 # lipo is ALSO part of the adipo lineage

        # B. Map to Subtype Targets (1.0 for lipo, 0.0 for non-lipo)
        subtype_targets = torch.zeros_like(original_labels, dtype=torch.float32)
        subtype_targets[original_labels == 0] = 1.0 

        # C. Forward Pass
        logits_lineage = self.lineage_classifier(z_real_pert)
        logits_subtype = self.subtype_classifier(z_real_pert).squeeze(-1) # Output shape: [Batch]

        # D. Calculate Masked Loss
        loss_lineage = F.cross_entropy(logits_lineage, lineage_targets)
        
        # Only calculate subtype loss for cells that belong to the adipo lineage
        adipo_mask = (lineage_targets == 2)
        if adipo_mask.sum() > 0:
            loss_subtype = F.binary_cross_entropy_with_logits(
                logits_subtype[adipo_mask], 
                subtype_targets[adipo_mask]
            )
        else:
            loss_subtype = torch.tensor(0.0, device=self.device)

        loss_class = loss_lineage + loss_subtype
        loss_dict['loss_class_lineage'] = loss_lineage
        loss_dict['loss_class_subtype'] = loss_subtype
        combined_probs = self.get_hierarchical_probs(logits_lineage, logits_subtype)
        # E. Same logic for the Critic (Detached)
        logits_lineage_critic = self.lineage_classifier(z_real_pert.detach())
        logits_subtype_critic = self.subtype_classifier(z_real_pert.detach()).squeeze(-1)
        
        loss_lineage_critic = F.cross_entropy(logits_lineage_critic, lineage_targets)
        if adipo_mask.sum() > 0:
            loss_subtype_critic = F.binary_cross_entropy_with_logits(
                logits_subtype_critic[adipo_mask], 
                subtype_targets[adipo_mask]
            )
        else:
            # Requires_grad=True is needed here to prevent distributed graph crashes if batch has no adipocytes
            loss_subtype_critic = torch.tensor(0.0, device=self.device, requires_grad=True) 

        loss_class_critic = loss_lineage_critic + loss_subtype_critic
        loss_dict['loss_class_critic'] = loss_class_critic
        
        # --- HIERARCHICAL CLASSIFICATION END ---

        # Reconstruction & Regularization (unchanged)
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
        
        # --- CENTER LOSS ---
        # Crucial: We use the `original_labels` (0 to 3) for CenterLoss, NOT the lineage targets.
        # This ensures the formula structurally enforces distinct manifolds for lipo and adipo, 
        # giving the subtype classifier the geometric variance it needs to separate them.
        center_loss = self.center_loss(z_real_pert, original_labels)
        loss_dict['center_loss'] = center_loss

        total_loss = (self.reconFactr * loss_recon + 
                      self.lossRegFactr * loss_reg +
                      self.LossClassFactr * loss_class +
                      self.lossLatentFactr * loss_consistency +
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
            # You will need to update logAUROC to handle the two separate logit streams
            self.logAUROC(mode, combined_probs, original_labels) 
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

    def log_pearson_metrics(self, mode, x_pert, x_ctrl, pert_idx):
        """
        Calculates Pearson Correlation per UNIQUE perturbation in the batch.
        Respects self.gene_mask to ignore padded genes.
        """
        with torch.no_grad():
            z_ctrl = self.encoder(x_ctrl)
            z_prompt = self.get_perturbation_prompt(x_ctrl, pert_idx)
            # Use flow matching solver instead of direct addition if applicable
            # z_fake_pert = self.ode_solve(z_ctrl, z_prompt, t_start=0, t_end=1, steps=4)
            # Or keep your current transition if you haven't switched yet:
            delta_pred = self.transition_fwd(z_ctrl, z_prompt)
            z_fake_pert = z_ctrl + delta_pred
            
            x_recon_pert = self.decoder(z_fake_pert)

        # 1. Identify unique perturbations in this batch
        unique_perts = torch.unique(pert_idx)
        
        # Lists to store scores for this batch
        batch_pearson_all = []
        batch_pearson_top5 = []

        # Prepare the mask (flattened)
        # Assuming self.gene_mask is [1, Seq_Len, Dim] -> [1, Total_Genes]
        flat_mask = self.gene_mask.view(-1).bool()

        # 2. Iterate through each unique perturbation
        for p_id in unique_perts:
            mask = (pert_idx == p_id)
            
            if mask.sum() == 0: continue

            # 3. Calculate Pseudo-bulk Vectors (Mean of this specific group)
            # Calculate mean across cells first [Batch_Subset, Genes] -> [Genes]
            real_delta_mean = (x_pert[mask] - x_ctrl[mask]).mean(dim=0).flatten()
            pred_delta_mean = (x_recon_pert[mask] - x_ctrl[mask]).mean(dim=0).flatten()
            
            # --- KEY CHANGE: Apply Gene Mask ---
            # We index into the flattened vector using the boolean mask
            real_delta_vector = real_delta_mean[flat_mask]
            pred_delta_vector = pred_delta_mean[flat_mask]
            
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
            
            if top_mask.sum() > 2:
                real_top = real_delta_vector[top_mask]
                pred_top = pred_delta_vector[top_mask]
                
                # Verify std again for the top subset
                if torch.std(real_top) > 1e-6 and torch.std(pred_top) > 1e-6:
                    p_top5 = pearson_corrcoef(pred_top, real_top)
                    batch_pearson_top5.append(p_top5)

        # 4. Aggregate and Log
        if len(batch_pearson_all) > 0:
            avg_pearson_all = torch.stack(batch_pearson_all).mean()
            self.log(f"{mode}/Pearson_All", avg_pearson_all, prog_bar=False, sync_dist=True)
        
        if len(batch_pearson_top5) > 0:
            avg_pearson_top5 = torch.stack(batch_pearson_top5).mean()
            self.log(f"{mode}/Pearson_Top5_Percent", avg_pearson_top5, prog_bar=False, sync_dist=True)


    def log_pearson_metrics_v2(self, mode, x_pert, x_ctrl, pert_idx):
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



    def logAUROC(self, mode, combined_probs, pert_state):
        # 1. Parse original mutually-exclusive labels (0, 1, 2, or 3)
        if pert_state.dim() > 1 and pert_state.size(1) > 1:
            original_labels = torch.argmax(pert_state, dim=1)
        else:
            original_labels = pert_state.long()
            
        # 2. Build Multi-Hot Targets for AUROC
        B = original_labels.size(0)
        multi_hot_targets = torch.zeros((B, 4), dtype=torch.long, device=self.device)
        multi_hot_targets.scatter_(1, original_labels.unsqueeze(1), 1)
        
        # The Overlap Rule: If it's lipo (Index 0), it is ALSO adipo (Index 3)
        lipo_mask = (original_labels == 0)
        multi_hot_targets[lipo_mask, 3] = 1 
        
        # 3. Update Multi-Label AUROC (Accumulate only)
        self.classwise_auc.update(combined_probs, multi_hot_targets)
        
        if mode == "val":
            # REMOVED: self.log_dict(...) from here!
            
            # 4. Update Confusion Matrix 
            if self.current_epoch % 5 == 0 and not self.trainer.sanity_checking:
                preds_discrete = torch.argmax(combined_probs, dim=1)
                self.confmat.update(preds_discrete, original_labels)

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
            loss,class_loss = self.phase_one_shared_step(batch,mode="val")
            return loss + class_loss
        else:
            g_loss = self.shared_step_cycle_gan(batch, optimizer_idx=0, mode='val')
            d_loss = self.shared_step_cycle_gan(batch, optimizer_idx=1, mode='val')
            return g_loss + d_loss
        

    def configure_optimizers(self):
        classifier_params = list(self.latentClassifier.parameters()) + list(self.lineage_classifier.parameters()) + list(self.subtype_classifier.parameters())
        classifier_ids = list(map(id, classifier_params))

        main_params = [p for p in self.parameters() if id(p) not in classifier_ids]
        optimizer_main = torch.optim.AdamW(main_params, lr=1e-4, weight_decay=1e-3)
        optimizer_probe = torch.optim.AdamW(classifier_params, lr=1e-3, weight_decay=0.0)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_main, mode='min', factor=0.5, patience=5)
        #scheduler = LambdaLR(optimizer_main, lr_lambda=warmup_cosine_schedule)
        return [optimizer_main, optimizer_probe]

#        return [ {"optimizer": optimizer_main, "lr_scheduler": { "scheduler": scheduler,"monitor": "val/total_loss"}},{"optimizer": optimizer_probe}]

    
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
    
    def log_collapse_metrics(self, z, delta, z_cycle):
        # Calculate std across the batch
        z_std = z.std(dim=0) 
        # How many dimensions have significant variance?
        active_dims = (z_std > 0.01).sum().float()
        
        # Check for "Lazy" Transitions (Identity Collapse)
        delta_mag = delta.norm(dim=1).mean()
        z_mag = z.norm(dim=1).mean()
        delta_ratio = delta_mag / (z_mag + 1e-8)
        # Check Reversibility
        cycle_error = self.criterion(z, z_cycle)

        self.log(f"Health/Active_Dims", active_dims,sync_dist=True)      
        self.log(f"Health/Delta_Strength", delta_mag, sync_dist=True)     
        self.log(f"Health/Delta_to_Z_Ratio", delta_ratio, sync_dist=True)  
        self.log(f"Health/Z_Vector_Size", z_mag, sync_dist=True)          
        self.log(f"Health/Cycle_Integrity", cycle_error, sync_dist=True)  

    def getcfmx(self,cm_tensor):
            cm_numpy = cm_tensor.cpu().numpy()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm_numpy, annot=True, fmt='g',cmap='Blues', xticklabels=self.class_names, 
                        yticklabels=self.class_names,ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix - Epoch {self.current_epoch}')
            temp_filename = f"confusion_matrix{self.current_epoch}.png"
            return fig, temp_filename

    def log_cfm(self,fig,name_tag=""):
        unique_filename = f"imgs/conf_matrix_latent_epoch_{self.current_epoch:03d}_rank_{self.global_rank}_{name_tag}.png"
        fig.savefig(unique_filename) 
        plt.close(fig)


    def on_validation_epoch_end(self):
        # We wrap in a try-except just in case DDP sanity checks run with empty batches
        try:
            # 1. Compute the dictionary of class-wise AUROC scores
            auc_dict = self.classwise_auc.compute()
            
            # 2. Log the entire dictionary at once
            self.log_dict(auc_dict, sync_dist=True, prog_bar=False)
            
        except Exception as e:
            # Lightning sanity checks can sometimes cause empty metric computes
            print(f"Skipping AUROC compute on Rank {self.global_rank}: {e}")
            
        finally:
            # 3. Safely reset the metric for the next epoch
            self.classwise_auc.reset()
  
    def get_hierarchical_probs(self, logits_lineage, logits_subtype):
        """
        Reconstructs the 4-class probabilities from the hierarchical heads.
        Lineage indices: 0: other, 1: pre_adipo, 2: adipo_lineage
        """
        probs_lineage = F.softmax(logits_lineage, dim=1)
        probs_subtype = torch.sigmoid(logits_subtype).squeeze(-1) # [Batch]
        
        p_other = probs_lineage[:, 0]
        p_pre_adipo = probs_lineage[:, 1]
        
        # Total adipocyte probability
        p_adipo = probs_lineage[:, 2] 
        
        # Lipogenic probability is conditional on being an adipocyte
        p_lipo = probs_subtype * p_adipo 
        
        # Stack them exactly in the order of self.class_names: 
        # ['lipo', 'other', 'pre_adipo', 'adipo']
        combined_probs = torch.stack([p_lipo, p_other, p_pre_adipo, p_adipo], dim=1)
        
        return combined_probs


def compute_hierarchical_semantic_loss(self, z, state):
        """
        Calculates the hierarchical semantic loss for a given latent vector 
        using the frozen Phase 1 classifiers.
        """
        # 1. Parse original labels
        if state.dim() > 1 and state.size(1) > 1:
            labels = torch.argmax(state, dim=1)
        else:
            labels = state.long()

        # 2. Map targets
        lineage_targets = torch.zeros_like(labels)
        lineage_targets[labels == 1] = 0 # other
        lineage_targets[labels == 2] = 1 # pre_adipo
        lineage_targets[labels == 3] = 2 # adipo
        lineage_targets[labels == 0] = 2 # lipo (belongs to adipo lineage)

        subtype_targets = torch.zeros_like(labels, dtype=torch.float32)
        subtype_targets[labels == 0] = 1.0

        # 3. Forward pass
        logits_lineage = self.lineage_classifier(z)
        logits_subtype = self.subtype_classifier(z).squeeze(-1)

        # 4. Compute Loss
        loss_lineage = F.cross_entropy(logits_lineage, lineage_targets)
        
        adipo_mask = (lineage_targets == 2)
        if adipo_mask.sum() > 0:
            loss_subtype = F.binary_cross_entropy_with_logits(
                logits_subtype[adipo_mask], 
                subtype_targets[adipo_mask]
            )
        else:
            loss_subtype = torch.tensor(0.0, device=self.device)

        return loss_lineage + loss_subtype