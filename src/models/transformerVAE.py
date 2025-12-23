import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np  
import torchmetrics
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns

class TransformerVAEEncoder(nn.Module):
    def __init__(self, input_dim=675, d_model=512, n_layers=6, n_heads=8, z_dim=32):
        super().__init__()

        self.token_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.mu = nn.Linear(d_model, z_dim)
        self.logvar = nn.Linear(d_model, z_dim)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)
    def forward(self, x):
        # x: (B, 32, 675)
        h = self.token_proj(x)
        h = self.encoder(h)
        # Optionally add layernorm here
        h_pooled = h.mean(dim=1)
        return self.mu(h_pooled), self.logvar(h_pooled)


class TransformerVAEDecoder(nn.Module):
    def __init__(self,output_dim=675,num_tokens=32,d_model=512,n_layers=6,n_heads=8,z_dim=32):
        super().__init__()

        self.num_tokens = num_tokens

        # Project latent to transformer dimension
        self.latent_proj = nn.Linear(z_dim, d_model)

        # Learned token queries (permutation invariant)
        self.token_queries = nn.Parameter(torch.randn(1, num_tokens, d_model))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, n_layers)

        # Project back to original feature space
        self.film_gamma = nn.Linear(z_dim, d_model)
        self.film_beta  = nn.Linear(z_dim, d_model)
        self.output_proj = nn.Linear(d_model, output_dim)
        self.outact = nn.ReLU()

    def forward(self, z):
        # z: (B, z_dim)
        B = z.size(0)

        latent = self.latent_proj(z)              # (B, d_model)
        latent = latent.unsqueeze(1)              # (B, 1, d_model)
        queries = self.token_queries.expand(B, -1, -1)  # (B, 32, d_model)
        # Inject latent into every token
        h = queries + latent
        gamma = self.film_gamma(z).unsqueeze(1)
        beta  = self.film_beta(z).unsqueeze(1)
        
        h = self.decoder(h)
        h = gamma * h + beta                      # FiLM modulation
        x_hat = self.output_proj(h)                # (B, 32, 675)
        #x_hat = self.outact(x_hat)
        return x_hat

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=675, d_model=512, n_layers=4, n_heads=8, num_classes=5):
        super().__init__()

        self.token_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, 32, 675)
        h = self.token_proj(x)
        h = self.encoder(h)
        # Optionally add layernorm here
        h_pooled = h.mean(dim=1)
        class_logits = self.classifier(h_pooled)
        return class_logits


class StateTrainer(pl.LightningModule):
    def __init__(self,encoder,decoder,categorizer,
                reconFacor=1.0, kldFactor=1.0, classFactor=1.0,
                kl_warmup_steps=10_000,free_bits=0.5,noise_warmup=10_000,KLD_MAX:float = 100.,beta_start:float=0.0,
                multiple_optimizers=False, reductionType ="mean",kld_mode="linear"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconFactor = reconFacor
        self.kldFactor = kldFactor
        self.classFactor = classFactor
        self.categorizer = categorizer
        self.KLD_MAX = KLD_MAX
        self.noise_warmup = noise_warmup
        self.multiple_optimizers = multiple_optimizers
        if multiple_optimizers:
            self.automatic_optimization = False
        self.reduction = reductionType
        self.kl_warmup_steps = kl_warmup_steps
        self.free_bits = free_bits
        self.kld_mode = kld_mode
        self.beta_start = beta_start
        self.aucMetric = torchmetrics.AUROC(num_classes=4, average=None,task="multiclass")
        self.confmat = MulticlassConfusionMatrix(num_classes=4)
        classes = ['pre_adipo', 'adipo', 'lipo', 'other']
        self.class_names = classes
        self.save_hyperparameters(ignore=['encoder','decoder','categorizer'])
        self.classwise_auc = ClasswiseWrapper(self.aucMetric,labels=classes)
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
        else:
            return min(1.0, self.beta_start + (self.kldFactor * self.global_step / self.kl_warmup_steps))

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
            loss_class = torch.tensor(0.0, device=Xs.device)


        if state.shape == class_logits.shape:
            state_indices = torch.argmax(state, dim=1)
            # 2. If state is [Batch, 1] -> Squeeze to [Batch]
        elif state.ndim == 2 and state.shape[1] == 1:
            state_indices = state.squeeze(1)
        else:
            state_indices = state
        total_loss = (self.reconFactor * loss_recon+ beta * loss_kld+ self.classFactor * loss_class)

        # Logging
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

    def on_validation_epoch_end(self):
        if self.categorizer is not None:
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
            temp_filename = "confusion_matrix.png"
            fig.savefig(temp_filename) 
            plt.close(fig)
            if hasattr(self.logger.experiment, 'log_artifact'):
                self.logger.experiment.log_artifact(
                run_id=self.logger.run_id, 
                local_path=temp_filename, 
                artifact_path="plots")