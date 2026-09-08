import torch
import torch.nn as nn


class DINOv2Classifier(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes, num_register_tokens=0):
        super().__init__()
        self.backbone = backbone
        self.num_register_tokens = num_register_tokens
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        sequence_output = outputs.last_hidden_state
        cls_token = sequence_output[:, 0]

        patch_start = 1 + self.num_register_tokens
        patch_tokens = sequence_output[:, patch_start:]
        pooled_patches = patch_tokens.mean(dim=1) if patch_tokens.size(1) > 0 else cls_token

        linear_input = torch.cat([cls_token, pooled_patches], dim=1)
        return self.classifier(linear_input)
