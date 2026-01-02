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
import torchmetrics
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import MulticlassConfusionMatrix
import torch.nn.functional as F
import umap
import pandas as pd
from src.utils import CenterLoss

from src.models.cycle_modules import TransformerCycleEncoder, TransformerCycleDecoder, LatentTransition, LatentTransition2, LatentClassifier


class CycleTransformer(pl.LightningModule):
    def __init__(self, input_dim=675, d_model=256, n_layers=4, n_heads=8, z_dim=256, lr=1e-3, num_genes=21600
                ,reconFactr = 10., lossLatentFactr = 1.0, lossCycleFactr = .5, lossRegFactr = 1e-4
                ,guidance_weight=0.1, lossClassFactr=0.5, pushLossFactr=5.0, centerFactr=0.5, 
                obsessionFactor=1.0, diversityFactor=1.0, 
                phase:int = 2):
        super().__init__()
        
        self.phase = phase
        #Actual Training and hyperparameters        
        ##Loss Factors
        self.reconFactr = reconFactr
        self.lossLatentFactr = lossLatentFactr
        self.lossCycleFactr = lossCycleFactr
        self.lossRegFactr = lossRegFactr
        self.guidance_weight = guidance_weight
        self.LossClassFactr = lossClassFactr
        self.pushLossFactr = pushLossFactr
        self.centerFactr = centerFactr
        self.obsessionFactor = obsessionFactor
        self.diversityFactor = diversityFactor

        #Model parameters
        self.z_dim = z_dim
        self.num_genes = num_genes
        self.cond_dim = 256
        self.ncidx = 21592

        #Model Components
        self.encoder = TransformerCycleEncoder(input_dim, d_model, n_layers, n_heads, z_dim)
        self.decoder = TransformerCycleDecoder(input_dim, d_model, n_layers, n_heads, z_dim)
        self.transition = LatentTransition2(num_genes, z_dim,cond_dim=self.cond_dim)
        self.latentClassifier = LatentClassifier(z_dim)

        #Traning phase settings
        if phase == 1:
            self.automatic_optimization = False
            self.configure_optimizers = self.configure_optimizers_phase_one
            self.shared_step = self.phase_one_shared_step
            for param in self.transition.parameters():
                param.requires_grad = False
        if self.phase == 2:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.latentClassifier.parameters():
                param.requires_grad = False
            self.shared_step = self.shared_step_transitionOnly
            self.encoder.eval()
            self.decoder.eval()
        
        #Losses
        self.criterion = nn.MSELoss()
        self.lr = lr
        weights = torch.tensor([1.0/1200, 1.0/3000, 1.0/5100, 1.0/5500])
        weights = weights / weights.sum()
        self.latent_criterion = nn.CrossEntropyLoss(weight=weights)
        self.center_loss = CenterLoss(num_classes=4, feat_dim=z_dim)
        # Checking metrics and classifiers
    
        self.aucMetric = torchmetrics.AUROC(num_classes=4, average=None,task="multiclass")
        self.confmat = MulticlassConfusionMatrix(num_classes=4)
        self.confmat2 = MulticlassConfusionMatrix(num_classes=4)
        classes = ['pre_adipo', 'adipo', 'lipo', 'other']
        self.class_names = classes
        self.classwise_auc = ClasswiseWrapper(self.aucMetric,labels=classes)
        self.save_hyperparameters()

    def phase_one_shared_step(self,batch,mode):
        x_pert, x_ctrl, pert_idx, pert_state, ctrl_state = batch
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

        loss_recon = self.criterion(x_recon_ctrl, x_ctrl) + self.criterion(x_recon_pert, x_pert)
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
        loss_dict['total_loss'] = total_loss
        self.logging_metrics(mode, loss_dict)
        self.logAUROC(mode, logits_pert, logits_pert, pert_state)
        return total_loss, loss_class_critic

    
    def shared_step_transitionOnly(self, batch, mode):
        x_pert, x_ctrl, pert_idx, pert_state, ctrl_state = batch
        with torch.no_grad():
            z_ctrl = self.encoder(x_ctrl)
            z_real_pert = self.encoder(x_pert)
        
        # Calculate Importance Weights (The "Dud" Fix)
        # If the real drug effect is huge, this sample is VIP.
            real_change_mag = torch.norm(z_real_pert - z_ctrl, p=2, dim=1)
            loss_weights = (real_change_mag + 0.1) / (real_change_mag.mean() + 0.1)

        delta = self.transition(z_ctrl, pert_idx, torch.argmax(ctrl_state,dim=1))
        z_pred = z_ctrl + delta
        loss_mmd_total = 0.0
        
        if pert_state.dim() > 1 and pert_state.size(1) > 1:
            # Convert One-Hot [B, 4] -> Indices [B] (e.g., [0, 2, 1, 0])
            pert_labels = torch.argmax(pert_state, dim=1)
        else:
            # It's already indices
            pert_labels = pert_state

        # Get unique classes present in this batch
        unique_states = torch.unique(pert_labels)
        
        for state in unique_states:
            # Create a 1D boolean mask (Shape: [Batch])
            mask = (pert_labels == state)
            
            # Now we can safely index the rows
            z_pred_subset = z_pred[mask]
            z_real_subset = z_real_pert[mask]
            
            # Calculate MMD if we have at least 2 samples
            if z_pred_subset.size(0) > 1:
                loss_mmd_total += self.mmd_loss(z_pred_subset, z_real_subset)
        
        # Average the loss by the number of valid states
        if len(unique_states) > 0:
            loss_mmd_total = loss_mmd_total / len(unique_states)
        loss_latent = F.mse_loss(z_pred, z_real_pert)
        
        loss_push = self.calculate_push_loss(z_ctrl, z_pred, z_real_pert)
        
        ## Adding Classification Loss
        logits_pred = self.latentClassifier(z_pred)
        loss_guidance = self.latent_criterion(logits_pred, pert_state)
        loss_anti_id = self.calculate_anti_identity_loss(logits_pred,ctrl_state)
        
        loss_diversity = self.calculate_entropy_loss(logits_pred)
        pred_classes = torch.argmax(logits_pred, dim=1)
        fraction_class_1 = (pred_classes == 0).float().mean()
        loss_obsession = F.relu(fraction_class_1 - 0.25) * 100.0

        
        total_loss = (100.0 * loss_mmd_total + 
                  0.5 * loss_latent + 
                  2.0 * loss_push+
                  50.0 * loss_guidance + 
                  self.diversityFactor * 500 * loss_diversity + 
                  10.0 * loss_anti_id +
                    self.obsessionFactor * loss_obsession
                )
        
        with torch.no_grad():
            logits_real = self.latentClassifier(z_real_pert)
            self.logAUROC(mode, logits_pred, logits_real, pert_state)
        
        
        lossDict = {'total_loss': total_loss, 'loss_latent': loss_latent, 'loss_push': loss_push,"loss_mmd": loss_mmd_total, "loss_guidance": loss_guidance,
                    "loss_diversity": loss_diversity, "loss_anti_id": loss_anti_id, "loss_obsession": loss_obsession}
        self.logging_metrics(mode, lossDict)
        return total_loss

    def logAUROC(self, mode, latentLogits, logits_real, pert_state):
        if pert_state.shape == latentLogits.shape:
            state_indices = torch.argmax(pert_state, dim=1)
            # 2. If state is [Batch, 1] -> Squeeze to [Batch]
        elif pert_state.ndim == 2 and pert_state.shape[1] == 1:
            state_indices = pert_state.squeeze(1)
        else:
            state_indices = pert_state
        auc_scores = self.classwise_auc(latentLogits, state_indices)
        auc_scores_real = self.classwise_auc(logits_real, state_indices)
        for aval, class_name in zip(auc_scores_real.values(),self.class_names): # Use your stored class names
            self.log(f"Classification/{mode}/AUROC_Real_{class_name}", aval, prog_bar=False, sync_dist=True)        
        for aval, class_name in zip(auc_scores.values(),self.class_names): # Use your stored class names
            self.log(f"Classification/{mode}/AUROC_{class_name}", aval, prog_bar=False, sync_dist=True)

        if mode == "val":
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
            loss = self.shared_step(batch,mode="Train")
        return loss

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, mode='val')

    def configure_optimizers_phase_one(self):
        classifier_params = list(self.latentClassifier.parameters()) 
        #+ \
        #                    list(self.BertLikeClassifier.parameters())
        classifier_ids = list(map(id, classifier_params))

        main_params = [p for p in self.parameters() if id(p) not in classifier_ids]
        optimizer_main = torch.optim.AdamW(main_params, lr=1e-4, weight_decay=1e-3)
        optimizer_probe = torch.optim.AdamW(classifier_params, lr=1e-3, weight_decay=0.0)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_main, mode='min', factor=0.5, patience=5)

        # 4. Return as a LIST of dictionaries
        return [ {"optimizer": optimizer_main, "lr_scheduler": { "scheduler": scheduler,"monitor": "val/total_loss"}},{"optimizer": optimizer_probe}]
    
    def configure_optimizers(self):
        optimizer_transition = torch.optim.AdamW(
        self.transition.parameters(), 
        lr=1e-3, # Faster learning rate for the MLP
        weight_decay=1e-5
    )
        return optimizer_transition

    def log_collapse_metrics(self, z, delta, z_cycle):
        # 1. Check for Dead Neurons (Dimensional Collapse)
        # Calculate std across the batch
        z_std = z.std(dim=0) 
        # How many dimensions have significant variance?
        active_dims = (z_std > 0.01).sum().float()
        
        # 2. Check for "Lazy" Transitions (Identity Collapse)
        delta_mag = delta.norm(dim=1).mean()
        z_mag = z.norm(dim=1).mean()
        delta_ratio = delta_mag / (z_mag + 1e-8)
        # 3. Check Reversibility
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
        unique_filename = f"conf_matrix_latent_epoch_{self.current_epoch:03d}_rank_{self.global_rank}_{name_tag}.png"
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

        if os.path.exists(unique_filename):
            os.remove(unique_filename)

    def on_validation_epoch_end(self):
        cfx1 = self.confmat.compute()
        fig, temp_filename = self.getcfmx(cfx1)
        self.log_cfm(fig,name_tag="generated")
        cfx2 = self.confmat2.compute()
        fig2, temp_filename2 = self.getcfmx(cfx2)
        self.log_cfm(fig2,name_tag="real")
        self.confmat.reset()
        self.confmat2.reset()
        if not self.trainer.sanity_checking:
            if (self.current_epoch % 10 == 0) or (self.current_epoch == self.trainer.max_epochs - 1):
                    self.visualize_latent_space(title=f"Epoch {self.current_epoch}")


    def calculate_push_loss(self, z_ctrl, z_pred, z_real_pert):
        """
        Penalizes the model if it fails to move 'z_pred' far enough from 'z_ctrl'.
        The minimum required distance is based on the actual distance between 
        control and perturbed data (z_real_pert).
        """
        # 1. How far did we actually move?
        # (We use p=2 Euclidean distance)
        dist_moved = torch.norm(z_pred - z_ctrl, p=2, dim=1)
        
        # 2. How far SHOULD we have moved?
        # We calculate the distance between the Real Control and Real Perturbed clusters.
        # CRITICAL: Detach this! We want the transition to move further, 
        # NOT for the encoder to pull the real clusters closer to cheat.
        with torch.no_grad():
            dist_target = torch.norm(z_real_pert - z_ctrl, p=2, dim=1)
            threshold = 0.9 * dist_target
        push_loss = F.relu(threshold - dist_moved).mean()
        
        return push_loss
        
    
    
    #def visualizeLatentSpace(self,z_pred,z_real_pert,pert_state,mode):
    #    pass

    def gaussian_kernel(self,x, y, sigma=2.0):
        # Calculate pairwise distances
        # x: [B, Z], y: [B, Z]
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        
        x = x.unsqueeze(1) # [B, 1, Z]
        y = y.unsqueeze(0) # [1, B, Z]
        
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        
        # Squared Euclidean distance
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
        return torch.exp(-kernel_input / (2 * (sigma**2)))

    def mmd_loss(self,source_features, target_features, sigma=2.0):
        xx = self.gaussian_kernel(source_features, source_features, sigma)
        yy = self.gaussian_kernel(target_features, target_features, sigma)
        xy = self.gaussian_kernel(source_features, target_features, sigma)
        
        return xx.mean() + yy.mean() - 2 * xy.mean()

    def calculate_entropy_loss(self, logits_pred):
            """
            Forces the batch of predictions to be diverse.
            If the model predicts 'Adipocyte' for everyone, this loss explodes.
            """
            # 1. Calculate the average probability distribution across the batch
            # Softmax per sample -> Mean across batch
            avg_probs = torch.softmax(logits_pred, dim=1).mean(dim=0) 
            target_probs = torch.full_like(avg_probs, 1.0 / avg_probs.size(0))
        
            diversity_loss = F.kl_div(avg_probs.log(), target_probs, reduction='batchmean')
            return diversity_loss

    def calculate_anti_identity_loss(self,logits_pred, ctrl_state):
            """
            Penalizes the model if the predicted class is the same as the starting control class.
            Only applies this penalty if the intended target (pert_state) IS different from control.
            """
            # Convert logits to log-probabilities
            if ctrl_state.dim() > 1 and ctrl_state.size(1) > 1:
                # If One-Hot [Batch, 4] -> Convert to indices [Batch]
                target_indices = torch.argmax(ctrl_state, dim=1)
            else:
                # If Float/Int [Batch] -> Cast to Long [Batch]
                target_indices = ctrl_state.long()
            log_probs = F.log_softmax(logits_pred, dim=1)
            
            # We want to MINIMIZE the probability of the control class (Input)
            # So we want to MAXIMIZE -log_prob(ctrl_class)
            # Which is equivalent to Minimizing log_prob(ctrl_class) -- wait, no.
            # We want probability of control class to be 0.
            
            # Get the log_prob of the class we started at
            # ctrl_state: [Batch] indices of start state
            ctrl_log_probs = log_probs.gather(1, target_indices.unsqueeze(1)).squeeze(1)        
            # If the model is confident this is still the control class, ctrl_log_probs is close to 0 (loss 0).
            # We want to punish high probability.
            # Loss = Probability of Control Class (0 to 1)
            prob_of_staying = torch.exp(ctrl_log_probs)
            
            return prob_of_staying.mean()

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
        save_path = f"umap_{title.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close()
        self.logger.experiment.log_artifact(
            run_id=self.logger.run_id, 
            local_path=save_path, 
            artifact_path="plots_latent")