import torch
import torch.nn as nn

class TransformerCycleEncoder(nn.Module):
    def __init__(self, input_dim=675, d_model=256, n_layers=4, n_heads=8, z_dim=256):
        super().__init__()
        self.token_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=n_heads,dim_feedforward=4 * d_model,batch_first=True,norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.flatten_dim = d_model * 32  #sequence length is 32
        self.latent = nn.Linear(self.flatten_dim, z_dim)
        self.norm = nn.LayerNorm(z_dim)

    def forward(self, x):
        # x: (B, 32, 675)
        x = self.token_proj(x)  # (B, 32, d_model)
        x = self.encoder(x)     # (B, 32, d_model)

        x = x.flatten(start_dim=1)  # Flatten the sequence dimension
        z = self.latent(x)     # (B, z_dim)
        z= self.norm(z)
        return z

class TransformerCycleDecoder(nn.Module):
    def __init__(self, output_dim=675, d_model=256, n_layers=4, n_heads=8, z_dim=256, seq_len=32):
        super().__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.flattened_dim = seq_len * d_model
        self.latent_expansion = nn.Linear(z_dim, self.flattened_dim)
        
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,dim_feedforward=4 * d_model,batch_first=True,norm_first=True,dropout=0.1)
        self.decoder_transformer = nn.TransformerEncoder(decoder_layer, n_layers)
        
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, z):
        # z: (B, z_dim)
        x = self.latent_expansion(z) # (B, 32 * 256)
        x = x.view(-1, self.seq_len, self.d_model) # (B, 32, 256)
        x = self.decoder_transformer(x) # (B, 32, 256)
        out = self.output_proj(x)       # (B, 32, 675)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
    def forward(self, x):
        return x + self.block(x) # Skip connection

class LatentTransition2(nn.Module):
    def __init__(self, num_genes, z_dim=256, cond_dim=128, hidden_dim=512,num_states=4):
        super().__init__()
        
        # 1. Condition Embedding (Make it larger: 64 -> 128)
        self.cond_embedding = nn.Embedding(
            num_embeddings=num_genes + 1, 
            embedding_dim=cond_dim, 
            padding_idx=num_genes
        )
        self.state_embedding = nn.Embedding(num_states, cond_dim)
        # 2. The FiLM Generator
        # Predicts Scale (gamma) and Shift (beta) for the hidden layer
        self.film_gen = nn.Sequential(
            nn.Linear(cond_dim+cond_dim, hidden_dim * 2), # *2 because we need gamma AND beta
            nn.LeakyReLU(0.2)
        )
        
        # 3. The Core Network (Pre-activation style)
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )
        self.fc_out = nn.Linear(hidden_dim, z_dim)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
        #self.act = nn.LeakyReLU(0.2)

    def forward(self, z, gene_index, state_index):
        # 1. Get Condition
        # gene_index: [Batch]
        c = self.cond_embedding(gene_index) # [B, cond_dim]
        s = self.state_embedding(state_index)
        context = torch.cat([c, s], dim=1)
        
        # 2. Generate Modulators (Gamma, Beta)
        film_params = self.film_gen(context) # [B, hidden_dim * 2]
        gamma, beta = torch.chunk(film_params, 2, dim=1) # Split into two parts
        
        # 3. Apply to Z
        # Layer 1
        h = self.fc1(z) # [B, hidden_dim]
        h = h * (1 + gamma) + beta 
        #h = self.act(h)
        h = self.res_blocks(h)
        
        
        delta = self.fc_out(h)
        # delta = torch.tanh(delta) * 5.0 # Cap max change to +/- 5 units
        return delta
    


class LatentTransition(nn.Module):
    def __init__(self, num_genes, z_dim=256, cond_dim=128, hidden_dim=512,num_states=4):
        super().__init__()
        
        # 1. Condition Embedding (Make it larger: 64 -> 128)
        self.cond_embedding = nn.Embedding(
            num_embeddings=num_genes + 1, 
            embedding_dim=cond_dim, 
            padding_idx=num_genes
        )
        self.state_embedding = nn.Embedding(num_states, cond_dim)
        # 2. The FiLM Generator
        # Predicts Scale (gamma) and Shift (beta) for the hidden layer
        self.film_gen = nn.Sequential(
            nn.Linear(cond_dim+cond_dim, hidden_dim * 2), # *2 because we need gamma AND beta
            nn.LeakyReLU(0.2)
        )
        
        # 3. The Core Network (Pre-activation style)
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, z_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, z, gene_index, state_index):
        # 1. Get Condition
        # gene_index: [Batch]
        c = self.cond_embedding(gene_index) # [B, cond_dim]
        s = self.state_embedding(state_index)
        context = torch.cat([c, s], dim=1)
        
        # 2. Generate Modulators (Gamma, Beta)
        film_params = self.film_gen(context) # [B, hidden_dim * 2]
        gamma, beta = torch.chunk(film_params, 2, dim=1) # Split into two parts
        
        # 3. Apply to Z
        # Layer 1
        h = self.fc1(z) # [B, hidden_dim]
        h = h * (1 + gamma) + beta 
        h = self.act(h)
        h = self.act(self.fc2(h))
        delta = self.fc_out(h)
        # delta = torch.tanh(delta) * 5.0 # Cap max change to +/- 5 units
        return delta
    
import torch.nn.utils.spectral_norm as spectral_norm

class LatentClassifier(nn.Module):
    def __init__(self, z_dim=256, num_classes=4, hidden_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            # Layer 1
            spectral_norm(nn.Linear(z_dim, hidden_dim)),
            nn.BatchNorm1d(hidden_dim),  
            nn.LeakyReLU(0.2),           
            nn.Dropout(0.2),             

            # Layer 2 (Output)
            spectral_norm(nn.Linear(hidden_dim, num_classes))
        )

    def forward(self, z):
        return self.classifier(z)
    
class BertLikeClassifier(nn.Module):
    def __init__(self, input_dim=675, d_model=256, n_layers=4, n_heads=8, z_dim=256, num_classes=4):
        super().__init__()
        self.encoder = TransformerCycleEncoder(input_dim, d_model, n_layers, n_heads, z_dim)
        self.classifier = LatentClassifier(z_dim, num_classes)
        
    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits

class LatentDiscriminator(nn.Module):
    def __init__(self, z_dim=256, num_genes=21600):
        super().__init__()
        # Condition on the Gene being perturbed!
        self.gene_embed = nn.Embedding(num_genes, 32)
        
        self.net = nn.Sequential(
            nn.Linear(z_dim + 32, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1) # Output: Real (1) vs Fake (0)
        )
        
    def forward(self, z, gene_idx):
        # Concatenate Latent Vector + Gene Embedding
        gene_emb = self.gene_embed(gene_idx) # [B, 32]
        inp = torch.cat([z, gene_emb], dim=1)
        return self.net(inp)


class SemanticDiscriminator(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        # REMOVED: self.gene_embed = nn.Embedding(num_genes, 32)
        # We no longer need a lookup table. The "Prompt" is the embedding.
        
        self.net = nn.Sequential(
            # Input: Latent State (z_dim) + Semantic Prompt (z_dim)
            # 256 + 256 = 512 input features
            nn.Linear(z_dim * 2, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1) # Output: Real (1) vs Fake (0)
        )
        
    def forward(self, z, z_prompt):
        """
        z: The cell state to evaluate (Real or Fake)
        z_prompt: The 'Instruction' vector describing the perturbation
        """
        # Concatenate the State and the Instruction
        inp = torch.cat([z, z_prompt], dim=1)
        return self.net(inp)

class DeltaTransition(nn.Module):
    def __init__(self, z_dim=256, hidden_dim=512):
        super().__init__()
        
        # Input is now:
        # 1. z_ctrl (Current State)
        # 2. z_delta_theoretical (The "Instruction" from the Encoder)
        
        self.net = nn.Sequential(
            # We concatenate state + theoretical_change
            nn.Linear(z_dim * 2, hidden_dim), 
            nn.LeakyReLU(0.2),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, z_dim)
        )
        # Zero init for stability
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, z_ctrl, z_theoretical_delta):
        # z_theoretical_delta = Encoder(Input_Zeroed) - Encoder(Input_Ctrl)
        
        inp = torch.cat([z_ctrl, z_theoretical_delta], dim=1)
        
        # Predict the Biological Cascade (The "Real" Delta)
        delta_pred = self.net(inp)
        
        return delta_pred