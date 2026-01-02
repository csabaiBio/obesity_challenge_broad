from dataclasses import dataclass
import torch
import torch.nn.functional as F

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


def shared_step_classification(self, batch, mode):
        x_pert, x_ctrl, pert_idx, pert_state, ctrl_state = batch
        with torch.no_grad():
            z_ctrl = self.encoder(x_ctrl)
            z_real_pert = self.encoder(x_pert)

        delta = self.transition(z_ctrl, pert_idx, torch.argmax(ctrl_state,dim=1))
        z_pred = z_ctrl + delta

        with torch.no_grad():
            real_change_mag = torch.norm(z_real_pert - z_ctrl, p=2, dim=1)
            loss_weights = (real_change_mag + 0.1) / (real_change_mag.mean() + 0.1)
        
        raw_mse = F.mse_loss(z_pred, z_real_pert, reduction='none').mean(dim=1)
        loss_latent = (raw_mse * loss_weights).mean()

        logits_pred = self.latentClassifier(z_pred)
        logits_real = self.latentClassifier(z_real_pert)
        loss_guidance = self.latent_criterion(logits_pred, pert_state)

        loss_push = self.calculate_push_loss(z_ctrl, z_pred, z_real_pert)
        loss_reg_delta = torch.mean(delta**2)

        total_loss = (10.0 * loss_latent + 
                  1.0 * loss_guidance + 
                  2.0 * loss_push +     # High enough to force movement
                  1e-5 * loss_reg_delta)
        
        lossDict = {
        'total_loss': total_loss,
        'loss_latent': loss_latent, 
        'loss_guidance': loss_guidance,
        'loss_push': loss_push}

        self.logging_metrics(mode, lossDict)
        self.logAUROC(mode, logits_pred, logits_real, pert_state)
        self.log_collapse_metrics(z_pred, delta, z_ctrl)  # Using z_ctrl

        return total_loss


class CycleTransformer(): # Legacy functions, but might be useful later
    def shared_step_original(self, batch, mode):
        x_pert, x_ctrl, pert_idx, pert_state, ctrl_state = batch
        
        z_ctrl = self.encoder(x_ctrl)
        with torch.no_grad():
            z_real_detached = self.encoder(x_pert)
        

        z_real_pert = self.encoder(x_pert)

        with torch.no_grad():
            real_delta_mag = torch.norm(z_real_pert - z_ctrl, p=2, dim=1) 
            loss_weights = (real_delta_mag + 0.1) / (real_delta_mag.mean() + 0.1)
        
        #Transition encoded data in latent space
        delta = self.transition(z_ctrl, pert_idx, torch.argmax(ctrl_state,dim=1))
        z_pred = z_ctrl + delta
        
        #Cycle consistency
        delta_reverse = self.transition(z_pred, torch.full_like(pert_idx, self.ncidx), torch.argmax(pert_state,dim=1))
        z_cycle = z_pred + delta_reverse
        
        #Decoding predicted latent space back to data space
        x_pred = self.decoder(z_pred)
        
        # Checking classifications
        logits_real = self.latentClassifier(z_real_detached)
        loss_critic = self.latent_criterion(logits_real, pert_state)


        logits_shaping = self.latentClassifier(z_real_pert) 
        loss_class_shaping = self.latent_criterion(logits_shaping, pert_state)

        pred_logits = self.latentClassifier(z_pred)
        loss_guidance = self.latent_criterion(pred_logits, pert_state)

        #Calculating Losses
        #loss_latent = self.criterion(z_pred, z_real_pert.detach())
        raw_mse = F.mse_loss(z_pred,z_real_pert.detach(),reduction='none').mean(dim=1)
        loss_latent = (raw_mse * loss_weights).mean()
        loss_recon_mse = self.criterion(x_pred, x_pert)
        loss_cycle = self.criterion(z_cycle, z_ctrl)
        loss_reg = torch.mean(z_ctrl**2) + torch.mean(delta**2)
        guidance_weight = self.guidance_weight_warmup()
        push_loss = self.calculate_push_loss(z_ctrl, z_pred, z_real_pert)
        total_loss = (self.reconFactr * loss_recon_mse + 
                      #self.lossLatentFactr * loss_latent + 
                      self.lossCycleFactr * loss_cycle + 
                      self.lossRegFactr * loss_reg + 
                      self.LossClassFactr * loss_class_shaping
                      #guidance_weight * loss_guidance
                      #self.pushLossFactr * push_loss
                      )
        

        # Logging
        lossDict = {'total_loss': total_loss,'loss_recon_mse': loss_recon_mse,'loss_latent': loss_latent,'loss_cycle': loss_cycle,'loss_reg': loss_reg,"RealDataLoss": loss_class_shaping,"loss_guidance": loss_guidance}
        self.logging_metrics(mode, lossDict)
        self.logAUROC(mode, pred_logits, logits_real,  pert_state)
        self.log_collapse_metrics(z_pred, delta, z_cycle)
        
        return total_loss,loss_critic
        
    def guidance_weight_warmup(self):
        current_weight = min(self.guidance_weight, self.guidance_weight * (self.current_epoch / 10))
        return current_weight