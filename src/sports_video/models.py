from __future__ import annotations

import torch
from torch import nn
from safetensors.torch import load_file

from .data import LABELS


class ImageBackbone(nn.Module):
    def __init__(self, name: str, *, pretrained: bool = True, weights: str | None = None):
        super().__init__()
        self.name = name
        if name == "efficientnet_b0":
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained and weights is None else None)
            if weights is not None:
                model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            self.encoder = model.features
            self.pool = model.avgpool
            self.width = 1280
        elif name == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18
            model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained and weights is None else None)
            if weights is not None:
                model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            self.encoder = nn.Sequential(*list(model.children())[:-1])
            self.width = 512
        elif name in {"dinov2_small", "dinov3_vits16"}:
            from transformers import AutoModel
            repo = "facebook/dinov2-small" if name == "dinov2_small" else "facebook/dinov3-vits16-pretrain-lvd1689m"
            self.encoder = AutoModel.from_pretrained(weights or repo)
            self.width = self.encoder.config.hidden_size
        elif name in {"tf_efficientnet_b4", "vit_small_patch32_224"}:
            import timm
            self.encoder = timm.create_model(name, pretrained=pretrained and weights is None, num_classes=0)
            if weights is not None:
                missing, unexpected = self.encoder.load_state_dict(load_file(weights), strict=False)
                if missing:
                    raise ValueError(f"Incomplete pretrained weights for {name}: {missing[:5]}")
                if any(not key.startswith(("classifier.", "head.")) for key in unexpected):
                    raise ValueError(f"Unexpected checkpoint keys for {name}: {unexpected[:5]}")
            self.width = self.encoder.num_features
        else:
            raise ValueError(f"Unknown image backbone: {name}")

    def forward(self, x):
        if self.name == "efficientnet_b0":
            return self.pool(self.encoder(x)).flatten(1)
        if self.name == "resnet18":
            return self.encoder(x).flatten(1)
        if self.name == "dinov3_vits16":
            return self.encoder(pixel_values=x).pooler_output
        if self.name == "dinov2_small":
            return self.encoder(pixel_values=x).pooler_output
        return self.encoder(x)


class VideoClassifier(nn.Module):
    def __init__(self, name: str, *, pretrained: bool = True, freeze: bool = False,
                 weights: str | None = None):
        super().__init__()
        self.name = name
        if name == "x3d_xs":
            from pytorchvideo.models.hub import x3d_xs
            model = x3d_xs(pretrained=pretrained and weights is None)
            if weights is not None:
                state = torch.load(weights, map_location="cpu", weights_only=False)
                model.load_state_dict(state.get("model_state", state))
            self.encoder = nn.Sequential(*list(model.blocks)[:-1])
            self.width = 192  # Checkpoint pre-head representation channel count.
            self.temporal = True
        elif name == "r3d_18":
            from torchvision.models.video import R3D_18_Weights, r3d_18
            model = r3d_18(weights=R3D_18_Weights.DEFAULT if pretrained and weights is None else None)
            if weights is not None:
                model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            self.encoder = nn.Sequential(*list(model.children())[:-1])
            self.width = 512
            self.temporal = True
        else:
            self.encoder = ImageBackbone(name, pretrained=pretrained, weights=weights)
            self.width = self.encoder.width
            self.temporal = False
        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        self.heads = nn.ModuleDict({k: nn.Linear(self.width, len(v)) for k, v in LABELS.items()})

    def forward(self, x):
        if self.temporal:
            embedding = self.encoder(x).mean(dim=(2, 3, 4))
        else:
            batch, frames = x.shape[:2]
            pixels = x.flatten(0, 1)
            if not any(p.requires_grad for p in self.encoder.parameters()):
                with torch.no_grad():
                    features = self.encoder(pixels)
            else:
                features = self.encoder(pixels)
            embedding = features.reshape(batch, frames, -1).mean(1)
        return {name: head(embedding) for name, head in self.heads.items()}
