import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedL1Loss(nn.Module):
    """
    Computes L1 loss with a given mask.
    
    Input
    ----------
    input : tensor
    input_mask : tensor
        Mask filled with coefficients to compute L1 loss.
    target : tensor
    
    Output
    ----------
    output : tensor
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    
    def forward(self, input, input_mask, target):
        return F.l1_loss(input*input_mask, target*input_mask, reduction=self.reduction)
    

class TVLoss(nn.Module):
    """
    Computes total variation loss.
    
    Input
    ----------
    input : torch.FloatTensor of shape (B, C, H, W)
    
    Output
    ----------
    output : scalar
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input):
        height_variation = F.l1_loss(input[:, :, 1:, :], input[:, :, :-1, :], reduction=self.reduction)
        width_variation = F.l1_loss(input[:, :, :, 1:], input[:, :, :, :-1], reduction=self.reduction)
        
        return height_variation + width_variation


class PerceptualLoss(nn.Module):
    """
    Computes perceptual loss. 
    
    Input
    ----------
    input : torch.FloatTensor of shape (B, C, H, W)
        Predicted image.
    completed : torch.FloatTensor of shape (B, C, H, W)
        Predicted image with non-hole pixels set to ground truth.
    target : torch.FloatTensor of shape (B, C, H, W)
        Ground truth image.
    
    Output
    ----------
    pred_loss : scalar
        Perceptual loss for model output.
    comp_loss : scalar
        Perceptual loss for model output with non-hole pixels set to ground truth.
    """
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor

        
    def forward(self, input, completed, target):
        pred_loss = 0
        comp_loss = 0
        input_fm = input
        completed_fm = completed
        target_fm = target
        for i, layer in enumerate(self.feature_extractor.feature_layers):
            input_fm = layer(input_fm)
            completed_fm = layer(completed_fm)
            target_fm = layer(target_fm)
        
            pred_loss += F.l1_loss(input_fm, target_fm)
            comp_loss += F.l1_loss(completed_fm, target_fm)
            
        return pred_loss, comp_loss
    
    
class StyleLoss(nn.Module):
    """
    Computes style loss. 
    
    Input
    ----------
    input : torch.FloatTensor of shape (B, C, H, W)
        Predicted image.
    completed : torch.FloatTensor of shape (B, C, H, W)
        Predicted image with non-hole pixels set to ground truth.
    target : torch.FloatTensor of shape (B, C, H, W)
        Ground truth image.
    
    Output
    ----------
    pred_loss : scalar
        Style loss for model output.
    comp_loss : scalar
        Style loss for model output with non-hole pixels set to ground truth.
    """
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        
    @staticmethod 
    def batch_autocorrelation(t):
        normalization = 1  / (t.shape[1]*t.shape[2]*t.shape[3])
        
        features = t.view(t.shape[0], t.shape[1], -1)
        input_autocorrelation = features.bmm(features.transpose(1, 2)) * normalization
        
        return input_autocorrelation

        
    def forward(self, input, completed, target):
        pred_loss = 0
        comp_loss = 0
        input_fm = input
        completed_fm = completed
        target_fm = target
        for i, layer in enumerate(self.feature_extractor.feature_layers):
            input_fm = layer(input_fm)
            completed_fm = layer(completed_fm)
            target_fm = layer(target_fm)

            input_autocorrelation = self.batch_autocorrelation(input_fm)
            completed_autocorrelation = self.batch_autocorrelation(completed_fm)
            target_autocorrelation = self.batch_autocorrelation(target_fm)

            pred_loss += F.l1_loss(input_autocorrelation, target_autocorrelation)
            comp_loss += F.l1_loss(completed_autocorrelation, target_autocorrelation)
            
        return pred_loss, comp_loss
    
    
class PerceptualStyleLoss(nn.Module):
    """
    Computes perceptual and style loss simultaneously.
    Needed in order not to recalculate feature maps.
    
    Input
    ----------
    input : torch.FloatTensor of shape (B, C, H, W)
        Predicted image.
    completed : torch.FloatTensor of shape (B, C, H, W)
        Predicted image with non-hole pixels set to ground truth.
    target : torch.FloatTensor of shape (B, C, H, W)
        Ground truth image.
    
    Output
    ----------
    perceptual_pred_loss : scalar
        Perceptual loss for model output.
    perceptual_comp_loss : scalar
        Perceptual loss for model output with non-hole pixes set to ground truth.
    style_pred_loss : scalar
        Style loss for model output.
    style_comp_loss : scalar
        Style loss for model output with non-hole pixels set to ground truth.
    """
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        
    @staticmethod 
    def batch_autocorrelation(t):
        normalization = 1  / (t.shape[1]*t.shape[2]*t.shape[3])
        
        features = t.view(t.shape[0], t.shape[1], -1)
        input_autocorrelation = features.bmm(features.transpose(1, 2)) * normalization
        
        return input_autocorrelation

        
    def forward(self, input, completed, target):
        perceptual_pred_loss = 0
        perceptual_comp_loss = 0
        style_pred_loss = 0
        style_comp_loss = 0
        input_fm = input.clone()
        completed_fm = completed.clone()
        target_fm = target.clone()
        for i, layer in enumerate(self.feature_extractor.feature_layers):
            input_fm = layer(input_fm)
            completed_fm = layer(completed_fm)
            target_fm = layer(target_fm)

            input_autocorrelation = self.batch_autocorrelation(input_fm)
            completed_autocorrelation = self.batch_autocorrelation(completed_fm)
            target_autocorrelation = self.batch_autocorrelation(target_fm)
            
            perceptual_pred_loss += F.l1_loss(input_fm, target_fm)
            perceptual_comp_loss += F.l1_loss(completed_fm, target_fm)

            style_pred_loss += F.l1_loss(input_autocorrelation, target_autocorrelation)
            style_comp_loss += F.l1_loss(completed_autocorrelation, target_autocorrelation)
            
        return perceptual_pred_loss, perceptual_comp_loss, style_pred_loss, style_comp_loss
    
    
class InpaintingLoss(nn.Module):
    """
    Computes total loss as described in paper 
    https://arxiv.org/abs/1804.07723
    
    Input
    ----------
    input : torch.FloatTensor of shape (B, C, H, W)
        Predicted image.
    input_mask : torch.FloatTensor of shape (B, C, H, W)
        Mask with 0.0 on hole positions and 1.0 on valid positions.
    target : torch.FloatTensor of shape (B, C, H, W)
        Ground truth image.
    
    Output
    ----------
    output : scalar
    """
    def __init__(self, 
                 feature_extractor,
                 valid_l1_factor=1, 
                 hole_l1_factor=6,
                 pred_perceptual_factor=0.05, 
                 comp_perceptual_factor=0.05, 
                 pred_style_factor=120, 
                 comp_style_factor=120, 
                 tv_factor=0.1,
                ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.masked_l1 = MaskedL1Loss()
        self.tv_loss = TVLoss()
        self.ps_loss = PerceptualStyleLoss(self.feature_extractor)
        
        self.valid_l1_factor = valid_l1_factor 
        self.hole_l1_factor = hole_l1_factor
        self.pred_perceptual_factor = pred_perceptual_factor 
        self.comp_perceptual_factor = comp_perceptual_factor 
        self.pred_style_factor = pred_style_factor
        self.comp_style_factor = comp_style_factor 
        self.tv_factor = tv_factor

    def forward(self, input, input_mask, target, separate=False):
        completed = input.clone()
        bool_mask = input_mask.bool()
        completed[bool_mask] = target[bool_mask]
        
        valid_l1 = self.masked_l1(input, input_mask, target)
        hole_l1 = self.masked_l1(input, 1-input_mask, target)
        perceptual_pred, perceptual_comp, style_pred, style_comp = self.ps_loss(input, completed, target)
        tv = self.tv_loss(completed)
        
        valid_l1 *= self.valid_l1_factor
        hole_l1 *= self.hole_l1_factor 
        perceptual_pred *= self.pred_perceptual_factor 
        perceptual_comp *= self.comp_perceptual_factor 
        style_pred *= self.pred_style_factor
        style_comp *= self.comp_style_factor
        tv *= self.tv_factor

        if separate:
            return valid_l1, hole_l1, perceptual_pred, perceptual_comp, style_pred, style_comp, tv
        else:
            total_loss = valid_l1 + hole_l1 + perceptual_pred + perceptual_comp + style_pred + style_comp + tv
            return total_loss
        
        
class InpaintingAdversarialLoss(InpaintingLoss):
    def __init__(self, 
            feature_extractor,
            valid_l1_factor=1, 
            hole_l1_factor=6,
            pred_perceptual_factor=0.05, 
            comp_perceptual_factor=0.05, 
            pred_style_factor=120, 
            comp_style_factor=120, 
            tv_factor=0.1,
            adversarial_factor=1
        ):
        super().__init__(feature_extractor, valid_l1_factor, hole_l1_factor, pred_perceptual_factor, comp_perceptual_factor, pred_style_factor, comp_style_factor, tv_factor)

        self.bce = nn.BCELoss()
        self.adversarial_factor = adversarial_factor
        
    
    def forward(self, input, input_mask, target, fake_probas, y_ones=None, separate=False):
        if not y_ones:
            y_ones = torch.ones_like(fake_probas, device=fake_probas.device)
            
        adversarial_loss = self.bce(fake_probas, y_ones)
        adversarial_loss *= self.adversarial_factor
        
        valid_l1, hole_l1, perceptual_pred, perceptual_comp, style_pred, style_comp, tv = super().forward(input, input_mask, target, separate=True)
        
        if separate:
            return valid_l1, hole_l1, perceptual_pred, perceptual_comp, style_pred, style_comp, tv, adversarial_loss
        else:
            total_loss = valid_l1 + hole_l1 + perceptual_pred + perceptual_comp + style_pred + style_comp + tv + adversarial_loss
            return total_loss


class InpaintingWassersteinLoss(InpaintingLoss):
    def __init__(self, 
            feature_extractor,
            valid_l1_factor=1, 
            hole_l1_factor=6,
            pred_perceptual_factor=0.05, 
            comp_perceptual_factor=0.05, 
            pred_style_factor=120, 
            comp_style_factor=120, 
            tv_factor=0.1,
            adversarial_factor=1
        ):
        super().__init__(feature_extractor, valid_l1_factor, hole_l1_factor, pred_perceptual_factor, comp_perceptual_factor, pred_style_factor, comp_style_factor, tv_factor)

        self.adversarial_factor = adversarial_factor
        
    
    def forward(self, input, input_mask, target, fake_probas, y_ones=None, separate=False):
        if not y_ones:
            y_ones = torch.ones_like(fake_probas, device=fake_probas.device)
            
        adversarial_loss = -fake_probas.mean()
        adversarial_loss *= self.adversarial_factor
        
        valid_l1, hole_l1, perceptual_pred, perceptual_comp, style_pred, style_comp, tv = super().forward(input, input_mask, target, separate=True)
        
        if separate:
            return valid_l1, hole_l1, perceptual_pred, perceptual_comp, style_pred, style_comp, tv, adversarial_loss
        else:
            total_loss = valid_l1 + hole_l1 + perceptual_pred + perceptual_comp + style_pred + style_comp + tv + adversarial_loss
        
        
class DiscriminatorLoss(nn.Module):
    def __init__(self,
            real_factor = 0.5,
            fake_factor = 0.5,
            label_smooth = 0.1
        ):
        super().__init__()
        self.bce = nn.BCELoss()
        
        self.real_factor = real_factor
        self.fake_factor = fake_factor
        self.label_smooth = label_smooth
        
    
    def forward(self, fake_probas, true_probas, y_ones=None, y_zeros=None, separate=False):
        if not y_ones:
            y_ones = torch.ones_like(true_probas, device=fake_probas.device)*(1-self.label_smooth)
        if not y_zeros:
            y_zeros = torch.zeros_like(fake_probas, device=fake_probas.device)+self.label_smooth
            
        real_loss = self.bce(true_probas, y_ones) 
        fake_loss = self.bce(fake_probas, y_zeros)
        real_loss *= self.real_factor
        fake_loss *= self.fake_factor
        
        if separate:
            return real_loss, fake_loss
        else:
            total_loss = real_loss + fake_loss
            return total_loss


class WassersteinDiscriminatorLoss(nn.Module):
    def __init__(self,
            real_factor = 1,
            fake_factor = 1,
        ):
        super().__init__()
        self.real_factor = real_factor
        self.fake_factor = fake_factor
        
    
    def forward(self, fake_probas, true_probas, separate=False):
        real_loss = -true_probas.mean()
        fake_loss = fake_probas.mean()
        real_loss *= self.real_factor
        fake_loss *= self.fake_factor
        
        if separate:
            return real_loss, fake_loss
        else:
            total_loss = real_loss + fake_loss
            return total_loss
        
