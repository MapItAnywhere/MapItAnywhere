from typing import Optional, Dict
import torch.nn as nn
import torch
from .schema import LossConfiguration


def dice_loss(input: torch.Tensor,
              target: torch.Tensor,
              loss_mask: torch.Tensor,
              class_weights: Optional[torch.Tensor | bool],
              smooth=1e-5):
    '''
    :param input: (B, H, W, C) Logits for each class
    :param target: (B, H, W, C) Ground truth class labels in one_hot
    :param loss_mask: (B, H, W) Mask indicating valid regions of the image
    :param class_weights: (C) Weights for each class
    :param smooth: Smoothing factor to avoid division by zero, default 1.0
    '''
    
    if isinstance(class_weights, torch.Tensor):
        class_weights = class_weights.unsqueeze(0)
    elif class_weights is None or class_weights == False:
        class_weights = torch.ones(
            1, target.size(-1), dtype=target.dtype, device=target.device)
    elif class_weights == True:
        class_weights = target.sum(1)
        class_weights = torch.reciprocal(target.mean(1) + 1e-3)
        class_weights = class_weights.clamp(min=1e-5)
        # Only consider classes that are present
        class_weights *= (target.sum(1) != 0).float()
        class_weights.requires_grad = False

    intersect = (2 * input * target)
    intersect = (intersect) + smooth

    union = (input + target)
    union = (union) + smooth

    loss = 1 - (intersect / union)  # B, H, W, C
    loss *= class_weights.unsqueeze(0).unsqueeze(0)
    loss = loss.sum(-1) / class_weights.sum()
    loss *= loss_mask
    loss = loss.sum() / loss_mask.sum()  # 1

    return loss


class EnhancedLoss(nn.Module):
    def __init__(
        self,
        cfg: LossConfiguration,
    ):  # following params in the paper
        super(EnhancedLoss, self).__init__()
        self.num_classes = cfg.num_classes
        self.xent_weight = cfg.xent_weight
        self.focal = cfg.focal_loss
        self.focal_gamma = cfg.focal_loss_gamma
        self.dice_weight = cfg.dice_weight
        # self.class_mapping = 

        if self.xent_weight == 0. and self.dice_weight == 0.:
            raise ValueError(
                "At least one of xent_weight and dice_weight must be greater than 0.")
        
        if self.xent_weight > 0.:
            self.xent_loss = nn.BCEWithLogitsLoss(
                reduction="none"
            )

        if self.dice_weight > 0.:
            self.dice_loss = dice_loss

        if cfg.class_weights is not None and cfg.class_weights != True:
            self.register_buffer("class_weights", torch.tensor(
                cfg.class_weights), persistent=False)
        else:
            self.class_weights = cfg.class_weights

        self.class_weights: Optional[torch.Tensor | bool]

        self.requires_frustrum = cfg.requires_frustrum
        self.requires_flood_mask = cfg.requires_flood_mask
        self.label_smoothing = cfg.label_smoothing

    def forward(self, pred: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]):
        '''
        Args:
            pred: Dict containing the
                - output: (B, C, H, W) Probabilities for each class
                - valid_bev: (B, H, W) Mask indicating valid regions of the image
                - conf: (B, H, W) Confidence map
            data: Dict containing the
                - seg_masks: (B, H, W, C) Ground truth class labels, one-hot encoded
                - confidence_map: (B, H, W) Confidence map
        '''
        loss = {}

        probs = pred['output'].permute(0, 2, 3, 1)  # (B, H, W, C)
        logits = pred['logits'].permute(0, 2, 3, 1)  # (B, H, W, C)
        labels: torch.Tensor = data['seg_masks']  # (B, H, W, C)

        loss_mask = torch.ones(
            labels.shape[:3], device=labels.device, dtype=labels.dtype)

        if self.requires_frustrum:
            frustrum_mask = pred["valid_bev"][..., :-1] != 0
            loss_mask = loss_mask * frustrum_mask.float()

        if self.requires_flood_mask:
            flood_mask = data["flood_masks"] == 0
            loss_mask = loss_mask * flood_mask.float()

        if self.xent_weight > 0.:

            if self.label_smoothing > 0.:
                labels_ls = labels.float().clone()
                labels_ls = labels_ls * \
                    (1 - self.label_smoothing) + \
                    self.label_smoothing / self.num_classes

                xent_loss = self.xent_loss(logits, labels_ls)
            else:
                xent_loss = self.xent_loss(logits, labels)

            if self.focal:
                pt = torch.exp(-xent_loss)
                xent_loss = (1 - pt) ** self.focal_gamma * xent_loss

            xent_loss *= loss_mask.unsqueeze(-1)
            xent_loss = xent_loss.sum() / (loss_mask.sum() + 1e-5)
            loss['cross_entropy'] = xent_loss
            loss['total'] = xent_loss * self.xent_weight

        if self.dice_weight > 0.:
            dloss = self.dice_loss(
                probs, labels, loss_mask, self.class_weights)
            loss['dice'] = dloss

            if 'total' in loss:
                loss['total'] += dloss * self.dice_weight
            else:
                loss['total'] = dloss * self.dice_weight

        return loss
