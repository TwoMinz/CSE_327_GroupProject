import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
import yaml
import os
from typing import Dict, List, Tuple, Optional, Union


class CrowdViT(nn.Module):
    """
    Vision Transformer based model for crowd counting and wait time prediction.
    Uses a pre-trained ViT and adds regression heads for people counting and wait time prediction.
    """

    def __init__(
        self,
        vit_model: str = "vit_base_patch16_384",
        pretrained: bool = True,
        dropout_rate: float = 0.1,
        use_auxiliary_head: bool = True,
    ):
        """
        Initialize the CrowdViT model.

        Args:
            vit_model (str): Name of the Vision Transformer model from timm
            pretrained (bool): Whether to use pretrained weights
            dropout_rate (float): Dropout rate for the regression heads
            use_auxiliary_head (bool): Whether to use an auxiliary head for people counting
        """
        super(CrowdViT, self).__init__()

        # Load the ViT backbone
        self.backbone = timm.create_model(
            vit_model,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        # Get the feature dimension from the backbone
        feature_dim = self.backbone.embed_dim

        # Common feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Wait time prediction head
        self.wait_time_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

        # Optional people counting head (as an auxiliary task)
        self.use_auxiliary_head = use_auxiliary_head
        if use_auxiliary_head:
            self.people_count_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 1),
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images [B, C, H, W]

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing wait_time and optionally people_count
        """
        # Extract features from the backbone
        features = self.backbone(x)

        # Process features through common layers
        shared_features = self.feature_extractor(features)

        # Wait time prediction
        wait_time = self.wait_time_head(shared_features)

        # Prepare output dictionary
        output = {"wait_time": wait_time.squeeze(-1)}

        # People counting (auxiliary task)
        if self.use_auxiliary_head:
            people_count = self.people_count_head(shared_features)
            output["people_count"] = people_count.squeeze(-1)

        return output


class CrowdDensityLoss(nn.Module):
    """
    Custom loss function for crowd counting and wait time prediction.
    Combines MSE loss for wait time prediction and people counting.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        """
        Initialize the loss function.

        Args:
            alpha (float): Weight for wait time loss
            beta (float): Weight for people count loss
        """
        super(CrowdDensityLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the loss.

        Args:
            predictions (Dict[str, torch.Tensor]): Model predictions containing 'wait_time' and optionally 'people_count'
            targets (Dict[str, torch.Tensor]): Ground truth containing 'wait_time' and 'num_people'

        Returns:
            tuple: (total_loss, loss_dict) where loss_dict contains individual loss components
        """
        # Wait time loss
        wait_time_loss = self.mse(predictions["wait_time"], targets["wait_time"])

        # Initialize loss dictionary
        loss_dict = {"wait_time_loss": wait_time_loss}

        # Total loss initialization
        total_loss = self.alpha * wait_time_loss

        # People count loss (if available)
        if "people_count" in predictions and "num_people" in targets:
            people_count_loss = self.mse(
                predictions["people_count"], targets["num_people"]
            )
            loss_dict["people_count_loss"] = people_count_loss
            total_loss += self.beta * people_count_loss

        loss_dict["total_loss"] = total_loss

        return total_loss, loss_dict


def load_model_config(config_path: str) -> Dict:
    """
    Load model configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file

    Returns:
        Dict: Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config: Dict) -> Tuple[nn.Module, nn.Module]:
    """
    Create model and loss function from configuration.

    Args:
        config (Dict): Model configuration dictionary

    Returns:
        tuple: (model, criterion)
    """
    model = CrowdViT(
        vit_model=config["model"]["vit_model"],
        pretrained=config["model"]["pretrained"],
        dropout_rate=config["model"]["dropout_rate"],
        use_auxiliary_head=config["model"]["use_auxiliary_head"],
    )

    criterion = CrowdDensityLoss(
        alpha=config["loss"]["alpha"], beta=config["loss"]["beta"]
    )

    return model, criterion


def save_model_config(config: Dict, save_path: str) -> None:
    """
    Save model configuration to a YAML file.

    Args:
        config (Dict): Model configuration dictionary
        save_path (str): Path to save the configuration YAML file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


# Default model configuration
default_model_config = {
    "model": {
        "vit_model": "vit_base_patch16_384",
        "pretrained": True,
        "dropout_rate": 0.1,
        "use_auxiliary_head": True,
    },
    "loss": {
        "alpha": 0.7,  # Weight for wait time loss
        "beta": 0.3,  # Weight for people count loss
    },
    "training": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "batch_size": 16,
        "num_epochs": 50,
        "warmup_epochs": 5,
        "early_stopping_patience": 10,
    },
    "data": {
        "dataset": "ShanghaiTech",
        "part": "A",
        "image_size": [384, 384],
        "data_dir": "./data/ShanghaiTech",
    },
}


if __name__ == "__main__":
    # Example usage

    # Save default configuration
    save_model_config(default_model_config, "./config/model_config.yaml")

    # Create model from default configuration
    model, criterion = create_model_from_config(default_model_config)

    # Print model summary
    print(model)

    # Test forward pass with random input
    batch_size = 2
    channels = 3
    height, width = default_model_config["data"]["image_size"]
    x = torch.randn(batch_size, channels, height, width)

    # Forward pass
    outputs = model(x)

    # Print output shapes
    for key, value in outputs.items():
        print(f"{key} shape: {value.shape}")
