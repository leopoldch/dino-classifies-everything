import torch
import torch.nn as nn


class DINOv3GeMClassifier(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes, num_register_tokens=0, p_init=3.0, eps=1e-6):
        super().__init__()
        self.backbone = backbone
        self.num_register_tokens = num_register_tokens
        self.gem_p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def signed_gem(self, patch_tokens):
        p = torch.clamp(self.gem_p, min=1.0)
        signed_power = torch.sign(patch_tokens) * torch.clamp(patch_tokens.abs(), min=self.eps).pow(p)
        pooled = signed_power.mean(dim=1)
        return torch.sign(pooled) * torch.clamp(pooled.abs(), min=self.eps).pow(1.0 / p)

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        sequence_output = outputs.last_hidden_state
        cls_token = sequence_output[:, 0]

        patch_start = 1 + self.num_register_tokens
        patch_tokens = sequence_output[:, patch_start:]
        pooled_patches = self.signed_gem(patch_tokens) if patch_tokens.size(1) > 0 else cls_token

        linear_input = torch.cat([cls_token, pooled_patches], dim=1)
        return self.classifier(linear_input)
