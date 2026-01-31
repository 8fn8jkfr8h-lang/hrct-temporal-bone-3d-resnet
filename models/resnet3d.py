"""
3D ResNet-18 backbone for medical imaging.

Compatible with MedicalNet pretrained weights and includes optional CBAM attention.
Reference: Chen et al., "Med3D: Transfer Learning for 3D Medical Image Analysis"
"""

import torch
import torch.nn as nn
from typing import Optional, List, Type
import os

from .cbam import CBAM3D


def conv3x3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1x1 convolution."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock3D(nn.Module):
    """
    Basic residual block for 3D ResNet-18/34.
    """
    expansion = 1
    
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        use_cbam: bool = False
    ):
        super().__init__()
        
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        
        # Optional CBAM attention
        self.cbam = CBAM3D(planes) if use_cbam else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply CBAM if enabled
        if self.cbam is not None:
            out = self.cbam(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet3D(nn.Module):
    """
    3D ResNet backbone for volumetric medical images.
    
    Args:
        block: Residual block type (BasicBlock3D for ResNet-18)
        layers: Number of blocks in each layer
        in_channels: Number of input channels (1 for grayscale CT)
        num_classes: Number of output classes (0 for feature extraction only)
        use_cbam: Whether to use CBAM attention (applied to layers 2, 3, 4)
    """
    
    def __init__(
        self,
        block: Type[BasicBlock3D] = BasicBlock3D,
        layers: List[int] = [2, 2, 2, 2],  # ResNet-18
        in_channels: int = 1,
        num_classes: int = 0,
        use_cbam: bool = True
    ):
        super().__init__()
        
        self.inplanes = 64
        self.use_cbam = use_cbam
        
        # Initial convolution
        self.conv1 = nn.Conv3d(
            in_channels, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, use_cbam=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_cbam=use_cbam)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_cbam=use_cbam)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_cbam=use_cbam)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Optional classifier
        self.fc = nn.Linear(512 * block.expansion, num_classes) if num_classes > 0 else None
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(
        self,
        block: Type[BasicBlock3D],
        planes: int,
        blocks: int,
        stride: int = 1,
        use_cbam: bool = False
    ) -> nn.Sequential:
        """Create a residual layer."""
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion)
            )
        
        layers = []
        # First block may have stride and downsample
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=use_cbam))
        self.inplanes = planes * block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=use_cbam))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, D, H, W)
            
        Returns:
            Feature vector (B, 512) or class logits (B, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if self.fc is not None:
            x = self.fc(x)
        
        return x
    
    def load_medicalnet_weights(self, weights_path: str, strict: bool = False, auto_download: bool = True):
        """
        Load MedicalNet pretrained weights.
        
        Args:
            weights_path: Path to resnet_18_23dataset.pth
            strict: Whether to strictly enforce matching keys
            auto_download: Whether to automatically download weights from Google Drive if not found
        """
        # Auto-download if weights don't exist
        if not os.path.exists(weights_path) and auto_download:
            print(f"Weights not found at {weights_path}, attempting to download...")
            try:
                from ..utils.download_weights import download_medicalnet_weights
                weights_path = download_medicalnet_weights()
            except Exception as e:
                print(f"Warning: Could not auto-download weights: {e}")
                raise FileNotFoundError(f"Weights file not found at {weights_path} and auto-download failed")
        
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        
        # Handle different state_dict formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        
        # Filter out incompatible keys
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in new_state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        
        loaded_keys = len(pretrained_dict)
        total_keys = len(model_dict)
        
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=strict)
        
        print(f"Loaded {loaded_keys}/{total_keys} parameters from MedicalNet weights")
        
        return loaded_keys, total_keys


class TemporalBoneClassifier(nn.Module):
    """
    Complete classifier for temporal bone pathology detection.
    
    Uses ResNet3D backbone with multi-task heads for:
    - Cholesteatoma detection
    - Ossicular chain discontinuity
    - Facial nerve dehiscence (optional, for production only)
    """
    
    def __init__(
        self,
        num_tasks: int = 2,
        use_cbam: bool = True,
        pretrained_path: Optional[str] = None
    ):
        """
        Args:
            num_tasks: Number of classification tasks (2 for validation, 3 for production)
            use_cbam: Whether to use CBAM attention in backbone
            pretrained_path: Path to MedicalNet pretrained weights
        """
        super().__init__()
        
        self.num_tasks = num_tasks
        
        # Backbone
        self.backbone = ResNet3D(
            block=BasicBlock3D,
            layers=[2, 2, 2, 2],
            in_channels=1,
            num_classes=0,
            use_cbam=use_cbam
        )
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            self.backbone.load_medicalnet_weights(pretrained_path)
        
        # Feature dimension from backbone
        feature_dim = 512
        
        # Task-specific heads
        self.head_cholesteatoma = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )
        
        self.head_ossicular = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )
        
        if num_tasks >= 3:
            self.head_facial_nerve = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, 1)
            )
        else:
            self.head_facial_nerve = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input volume (B, 1, D, H, W)
            
        Returns:
            Logits tensor (B, num_tasks)
        """
        # Extract features
        features = self.backbone(x)
        
        # Task-specific predictions
        pred_chole = self.head_cholesteatoma(features)
        pred_ossic = self.head_ossicular(features)
        
        if self.head_facial_nerve is not None:
            pred_facial = self.head_facial_nerve(features)
            return torch.cat([pred_chole, pred_ossic, pred_facial], dim=1)
        else:
            return torch.cat([pred_chole, pred_ossic], dim=1)


def resnet18_3d(pretrained_path: Optional[str] = None, auto_download: bool = True, **kwargs) -> ResNet3D:
    """
    Construct a 3D ResNet-18 model.
    
    Args:
        pretrained_path: Path to pretrained weights. If None and auto_download=True,
                        will download from Google Drive.
        auto_download: Whether to auto-download weights from Google Drive if not found
        **kwargs: Additional arguments for ResNet3D
    """
    model = ResNet3D(BasicBlock3D, [2, 2, 2, 2], **kwargs)
    
    if pretrained_path is None:
        pretrained_path = "pretrained/resnet_18_23dataset.pth"
    
    if auto_download or os.path.exists(pretrained_path):
        try:
            model.load_medicalnet_weights(pretrained_path, auto_download=auto_download)
        except FileNotFoundError:
            print(f"Proceeding without pretrained weights")
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing ResNet3D backbone...")
    backbone = ResNet3D(use_cbam=True)
    x = torch.randn(2, 1, 128, 128, 128)
    features = backbone(x)
    print(f"Input: {x.shape} -> Features: {features.shape}")
    print(f"Backbone parameters: {sum(p.numel() for p in backbone.parameters()):,}")
    
    print("\nTesting TemporalBoneClassifier (2 tasks)...")
    model = TemporalBoneClassifier(num_tasks=2, use_cbam=True)
    outputs = model(x)
    print(f"Input: {x.shape} -> Outputs: {outputs.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
