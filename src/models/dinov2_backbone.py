"""
DINOv2 backbone implementation for TransCrowd model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Backbone(nn.Module):
    """
    DINOv2 backbone for feature extraction.

    Args:
        model_size (str): Model size - 'small', 'base', 'large', 'giant'
        pretrained (bool): Whether to use pretrained weights
        img_size (int): Input image size
        patch_size (int): Patch size
        freeze_backbone (bool): Whether to freeze backbone weights
    """

    def __init__(self, model_size='base', pretrained=True, img_size=384,
                 patch_size=16, freeze_backbone=False):
        super(DINOv2Backbone, self).__init__()

        self.model_size = model_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.freeze_backbone = freeze_backbone

        # Model size configurations
        self.size_configs = {
            'small': {'embed_dim': 384, 'model_name': 'dinov2_vits14'},
            'base': {'embed_dim': 768, 'model_name': 'dinov2_vitb14'},
            'large': {'embed_dim': 1024, 'model_name': 'dinov2_vitl14'},
            'giant': {'embed_dim': 1536, 'model_name': 'dinov2_vitg14'}
        }

        if model_size not in self.size_configs:
            raise ValueError(f"Model size {model_size} not supported. Choose from {list(self.size_configs.keys())}")

        self.config = self.size_configs[model_size]
        self.embed_dim = self.config['embed_dim']

        # Load pretrained DINOv2 model
        if pretrained:
            try:
                self.backbone = torch.hub.load('facebookresearch/dinov2', self.config['model_name'])
                print(f"✓ Loaded pretrained DINOv2 {model_size} model")
            except Exception as e:
                print(f"Failed to load pretrained model: {e}")
                print("Falling back to random initialization...")
                self.backbone = self._create_dinov2_model()
        else:
            self.backbone = self._create_dinov2_model()

        # Freeze backbone if requested
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ Backbone weights frozen")

        # Calculate number of patches
        self.num_patches = (img_size // 14) ** 2  # DINOv2 uses 14x14 patches

        # Feature dimensions
        self.num_features = self.embed_dim

    def _create_dinov2_model(self):
        """Create DINOv2 model architecture (fallback if pretrained loading fails)"""
        print("Creating DINOv2 model from scratch...")

        try:
            from timm.models.vision_transformer import VisionTransformer

            depth_configs = {
                'small': 12,
                'base': 12,
                'large': 24,
                'giant': 40
            }

            num_heads_configs = {
                'small': 6,
                'base': 12,
                'large': 16,
                'giant': 24
            }

            model = VisionTransformer(
                img_size=self.img_size,
                patch_size=14,  # DINOv2 uses 14x14 patches
                embed_dim=self.embed_dim,
                depth=depth_configs[self.model_size],
                num_heads=num_heads_configs[self.model_size],
                num_classes=0,  # No classification head
                global_pool='',  # No global pooling
            )

            return model

        except ImportError:
            print("timm not available, creating basic transformer...")
            # Fallback to basic implementation
            return self._create_basic_transformer()

    def _create_basic_transformer(self):
        """Create basic transformer as fallback"""
        class BasicTransformer(nn.Module):
            def __init__(self, embed_dim, img_size):
                super().__init__()
                self.embed_dim = embed_dim
                self.img_size = img_size
                patch_size = 14
                num_patches = (img_size // patch_size) ** 2

                # Patch embedding
                self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

                # Position embedding
                self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)

                # CLS token
                self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

                # Transformer layers (simplified)
                self.norm = nn.LayerNorm(embed_dim)

            def forward(self, x):
                B = x.shape[0]

                # Patch embedding
                x = self.patch_embed(x)  # (B, embed_dim, H/14, W/14)
                x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

                # Add CLS token
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)

                # Add position embedding
                x = x + self.pos_embed

                # Apply norm (simplified transformer)
                x = self.norm(x)

                return x[:, 0]  # Return CLS token

        return BasicTransformer(self.embed_dim, self.img_size)

    def forward_features(self, x):
        """
        Extract features from DINOv2 backbone.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Features of shape (B, embed_dim)
        """
        # Get features from DINOv2
        if hasattr(self.backbone, 'forward_features'):
            # Use forward_features if available (timm models)
            features = self.backbone.forward_features(x)
        else:
            # Use regular forward for torch.hub models
            if self.freeze_backbone:
                with torch.no_grad():
                    features = self.backbone(x)
            else:
                features = self.backbone(x)

        # Handle different output formats
        if isinstance(features, dict):
            # Some models return dict with 'x' key
            features = features['x'] if 'x' in features else features['last_hidden_state']
        elif isinstance(features, tuple):
            # Some models return tuple, take the first element
            features = features[0]

        # If features include CLS token, remove it or use it
        if len(features.shape) == 3:  # (B, N, D) format
            # Take CLS token (first token) for global representation
            features = features[:, 0, :]  # (B, D)
        elif len(features.shape) == 4:  # (B, C, H, W) format
            # Global average pooling
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)

        return features

    def forward(self, x):
        """
        Forward pass through DINOv2 backbone.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Features of shape (B, embed_dim)
        """
        return self.forward_features(x)


class DINOv2WithRegression(nn.Module):
    """
    DINOv2 backbone with regression head for crowd counting.
    """

    def __init__(self, model_size='base', pretrained=True, img_size=384,
                 dropout_rate=0.2, freeze_backbone=False):
        super(DINOv2WithRegression, self).__init__()

        # DINOv2 backbone
        self.backbone = DINOv2Backbone(
            model_size=model_size,
            pretrained=pretrained,
            img_size=img_size,
            freeze_backbone=freeze_backbone
        )

        # Improved regression head with better initialization
        self.regression_head = nn.Sequential(
            nn.LayerNorm(self.backbone.num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

        # Initialize regression head
        self._init_regression_head()

    def _init_regression_head(self):
        """Initialize regression head weights with better initialization."""
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                # Use smaller initialization for better convergence
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Predicted crowd count of shape (B,)
        """
        # Extract features
        features = self.backbone(x)

        # Regression head
        count = self.regression_head(features)

        return count.squeeze(-1)  # Return (B,) tensor


def create_dinov2_model(config):
    """
    Create DINOv2 model based on configuration.

    Args:
        config (dict): Model configuration

    Returns:
        nn.Module: DINOv2 model
    """
    model = DINOv2WithRegression(
        model_size=config.get('dinov2_size', 'base'),
        pretrained=config.get('pretrained', True),
        img_size=config.get('img_size', 384),
        dropout_rate=config.get('dropout_rate', 0.2),
        freeze_backbone=config.get('freeze_backbone', False)
    )

    return model