import torch.nn as nn

class DINOv3Classifier(nn.Module):
        def __init__(self, backbone, hidden_size, num_classes):
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            outputs = self.backbone(pixel_values=x)
            cls_token = outputs.last_hidden_state[:, 0]
            return self.classifier(cls_token)
