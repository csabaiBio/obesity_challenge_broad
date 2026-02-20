import torch
import torch.nn.functional as F
import math
import torch.nn as nn


def calculate_entropy_loss(logits_pred):
        """
        Forces the batch of predictions to be diverse.
        """
        # Softmax per sample -> Mean across batch
        avg_probs = torch.softmax(logits_pred, dim=1).mean(dim=0) 
        target_probs = torch.full_like(avg_probs, 1.0 / avg_probs.size(0))
    
        diversity_loss = F.kl_div(avg_probs.log(), target_probs, reduction='batchmean')
        return diversity_loss


def calculate_anti_identity_loss(logits_pred, ctrl_state):
        """
        Penalizes the model if the predicted class is the same as the starting control class.
        """
        if ctrl_state.dim() > 1 and ctrl_state.size(1) > 1:
            target_indices = torch.argmax(ctrl_state, dim=1)
        else:
            target_indices = ctrl_state.long()
        log_probs = F.log_softmax(logits_pred, dim=1)
        
        
        ctrl_log_probs = log_probs.gather(1, target_indices.unsqueeze(1)).squeeze(1)        
        prob_of_staying = torch.exp(ctrl_log_probs)
        return prob_of_staying.mean()

def calculate_push_loss(z_ctrl, z_pred, z_real_pert):
    """
    Penalizes the model if it fails to move 'z_pred' far enough from 'z_ctrl'.
    The minimum required distance is based on the actual distance between 
    control and perturbed data (z_real_pert).
    """
    dist_moved = torch.norm(z_pred - z_ctrl, p=2, dim=1)

    with torch.no_grad():
        dist_target = torch.norm(z_real_pert - z_ctrl, p=2, dim=1)
        threshold = 0.9 * dist_target
    push_loss = F.relu(threshold - dist_moved).mean()
    
    return push_loss
    
def gaussian_kernel(x, y, sigma=2.0):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    
    x = x.unsqueeze(1) # [B, 1, Z]
    y = y.unsqueeze(0) # [1, B, Z]
    
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input / (2 * (sigma**2)))

def mmd_loss(source_features, target_features, sigma=2.0):
    xx = gaussian_kernel(source_features, source_features, sigma)
    yy = gaussian_kernel(target_features, target_features, sigma)
    xy = gaussian_kernel(source_features, target_features, sigma)
    
    return xx.mean() + yy.mean() - 2 * xy.mean()





def calculate_anti_identity_loss_v2( logits_pred, start_state_indices):
    """
    Strongly penalizes any probability mass assigned to the starting class.
    """
    probs = torch.softmax(logits_pred, dim=1)
    p_start = probs.gather(1, start_state_indices.unsqueeze(1)).squeeze(1)
    loss = -torch.log(1 - p_start + 1e-8).mean()

    return loss

def warmup_cosine_schedule(epoch):
    warmup_epochs = 10
    max_epochs = 100 # Set this to your estimated max epochs
    
    if epoch < warmup_epochs:
        # Linear Warmup: 0 -> 1
        return float(epoch + 1) / float(warmup_epochs)
    else:
        # Cosine Decay: 1 -> 0
        progress = float(epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
class CenterLoss(nn.Module):
    def __init__(self, num_classes=4, feat_dim=256, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        
        # The learnable centers for each class
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        x: feature matrix with shape (batch_size, feat_dim).
        labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        
        # Calculate distance of every point x to every center
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        
        # distmat = x^2 + c^2 - 2xc
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        # Get the distance to the CORRECT center for each sample
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
    

class VarianceLoss():
    def __init__(self,eps=1e-8,percentage=0.5):
        self.eps = eps
        self.percentage = percentage
    def __call__(self, x_real,x_recon):
        std_real = torch.sqrt(x_real.var(dim=0) + self.eps)
        std_recon = torch.sqrt(x_recon.var(dim=0) + self.eps)
        threshold = torch.quantile(std_real, self.percentage) 
        mask = std_real > threshold

        loss_var = F.mse_loss(std_recon[mask], std_real[mask])
        return loss_var