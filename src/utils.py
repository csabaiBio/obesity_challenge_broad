import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self, non_zero_weight=10.0):
        super().__init__()
        self.non_zero_weight = non_zero_weight

    def forward(self, input, target):
        # Calculate standard squared error
        diff = (input - target) ** 2
        
        # Create a weight mask: 
        # If target gene is expressed (>0), weight = 10. Else weight = 1.
        weights = torch.ones_like(target)
        weights[target > 0] = self.non_zero_weight
        
        # Apply weights
        weighted_loss = torch.mean(weights * diff)
        return weighted_loss

class CosineMSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha # Balance between Magnitude (MSE) and Direction (Cosine)

    def forward(self, input, target):
        # 1. MSE (Magnitude)
        mse = nn.functional.mse_loss(input, target)
        
        # 2. Cosine Distance (Direction/Pattern)
        # 1 - cos_sim gives 0 for identical directions, 1 for orthogonal, 2 for opposite
        cosine_sim = nn.functional.cosine_similarity(input, target, dim=-1)
        cosine_loss = torch.mean(1 - cosine_sim)
        
        # Combine
        return self.alpha * mse + (1 - self.alpha) * cosine_loss

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