from torch import nn
from torchvision import models


def get_model_classification(
        model_name: str = 'resnet18',
        num_classes: int = 1000,
        pretrained: bool = True) -> nn.Module:
    model_fn = models.__dict__[model_name]
    model = model_fn(num_classes=1000, pretrained=pretrained)

    dim_feats = model.fc.in_features
    model.fc = nn.Linear(dim_feats, num_classes)

    return model
