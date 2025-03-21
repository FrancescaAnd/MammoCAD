import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# Adversarial Loss
def adversarial_loss(pred, target_is_real):
    ''' Computes the standard GAN loss (BCE Loss)'''
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn(pred, target)

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19.features)[:36])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.feature_extractor(x)


class ContentLoss(nn.Module):
    def __init__(self, feature_extractor):
        super(ContentLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.criterion = nn.MSELoss()

    def forward(self, sr, hr):
        # Convert grayscale → 3 channels
        sr = sr.repeat(1, 3, 1, 1)
        hr = hr.repeat(1, 3, 1, 1)

        # Resize to consistent shape (e.g., 224x224)
        sr = F.interpolate(sr, size=(224, 224), mode='bilinear', align_corners=False)
        hr = F.interpolate(hr, size=(224, 224), mode='bilinear', align_corners=False)

        sr_vgg_features = self.feature_extractor(sr)
        hr_vgg_features = self.feature_extractor(hr)

        return self.criterion(sr_vgg_features, hr_vgg_features)
